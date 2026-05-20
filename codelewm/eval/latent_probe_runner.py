"""Manifest-backed latent representation probe evaluation."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.data import OptionalDependencyError
from codelewm.observability import (
    ArtifactManifestError,
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security import require_trusted_checkpoint
from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE

from .latent_probe import (
    LATENT_PROBE_REPORT_SCHEMA_VERSION,
    LATENT_PROBE_TARGETS,
    LATENT_PROBE_VIEWS,
    LatentProbeConfig,
    LatentProbeError,
    LatentProbeReport,
    LatentProbeRow,
    build_latent_probe_report,
    write_latent_probe_report,
)
from .retrieval_runner import (
    _action_batch,
    _display_path,
    _embed_rows,
    _infer_training_artifact_manifest_path,
    _load_split_rows,
    _load_torch_checkpoint,
    _PackPaths,
    _read_verified_artifact_manifest,
    _require_torch_runtime,
    _resolve_device,
    _resolve_pack_paths,
)


LATENT_PROBE_EVAL_RUN_SCHEMA_VERSION = "codelewm.eval.latent_probe_run.v1"


@dataclass(frozen=True)
class LatentProbeEvalResult:
    """CLI-facing summary for a manifest-backed latent probe run."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    parent_artifacts: tuple[str, ...]
    row_count: int
    split_counts: Mapping[str, int]
    claim_boundary: Mapping[str, Any]
    schema_version: str = LATENT_PROBE_EVAL_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "parent_artifacts": list(self.parent_artifacts),
            "row_count": self.row_count,
            "split_counts": dict(self.split_counts),
            "claim_boundary": dict(self.claim_boundary),
        }


def run_latent_probe_evaluation(
    *,
    checkpoint: Path | str,
    data: Path | str,
    out: Path | str,
    device: str = "cpu",
    max_examples_per_split: int = 1000,
    bootstrap_samples: int = 200,
    seed: int = 0,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "latent-probe"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> LatentProbeEvalResult:
    """Run latent representation probes and write a manifest-backed report."""

    _positive_int(max_examples_per_split, "max_examples_per_split")
    _non_negative_int(bootstrap_samples, "bootstrap_samples")
    checkpoint_path = Path(checkpoint).resolve()
    out_dir = Path(out).resolve()
    _reject_existing_latent_probe_outputs(out_dir, overwrite=overwrite)

    pack_paths = _resolve_pack_paths(data)
    dataset_artifact = _read_verified_artifact_manifest(pack_paths.artifact_manifest_path, root=pack_paths.root)
    if dataset_artifact.artifact_kind != "dataset":
        raise ArtifactManifestError("latent-probe --data manifest must be a dataset artifact")
    training_artifact_path = _infer_training_artifact_manifest_path(checkpoint_path)
    training_artifact = read_artifact_manifest(training_artifact_path)
    if training_artifact.artifact_kind != "training_run":
        raise ArtifactManifestError("checkpoint parent manifest must be a training_run artifact")
    checkpoint_manifest = require_trusted_checkpoint(checkpoint_path)
    validate_artifact_checksums(training_artifact, root=training_artifact_path.parent)
    runtime = _require_torch_runtime()
    selected_device = _resolve_device(device, runtime)
    model, checkpoint_payload = _load_torch_checkpoint(
        checkpoint_path,
        device=selected_device,
        runtime=runtime,
    )
    action_view = str(checkpoint_manifest.metadata.action_view)
    if action_view != str(model.config.action_view):
        raise LatentProbeError(
            "checkpoint manifest action_view does not match checkpoint payload: "
            f"{action_view!r} != {model.config.action_view!r}"
        )

    rows = _load_probe_eval_rows(
        pack_paths,
        action_view=model.config.action_view,
        max_examples_per_split=max_examples_per_split,
        seed=seed,
    )
    report = _evaluate_probe_rows(
        rows,
        model=model,
        runtime=runtime,
        device=selected_device,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        metadata={
            "checkpoint": {
                "path": _display_path(checkpoint_path),
                "sha256": checkpoint_manifest.checkpoint_sha256,
                "step": _optional_int(checkpoint_payload.get("step")),
                "model_class": "TorchCodeTransitionModel",
                "backend": "torch",
            },
            "dataset": {
                "path": _display_path(pack_paths.root),
                "artifact_id": dataset_artifact.artifact_id,
                "split_counts": dict(dataset_artifact.metadata.get("split_counts", {})),
            },
            "training_artifact_id": training_artifact.artifact_id,
            "action_view": model.config.action_view,
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "schema_version": LATENT_PROBE_EVAL_RUN_SCHEMA_VERSION,
        "checkpoint": _display_path(checkpoint_path),
        "data": _display_path(pack_paths.root),
        "out": _display_path(out_dir),
        "device": str(selected_device),
        "max_examples_per_split": max_examples_per_split,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "action_view": model.config.action_view,
        "targets": list(LATENT_PROBE_TARGETS),
        "views": list(LATENT_PROBE_VIEWS),
    }
    config_path = out_dir / "config.json"
    report_path = out_dir / "reports" / "latent_probe_report.json"
    _write_json(config_payload, config_path)
    write_latent_probe_report(report, report_path)

    parent_artifacts = (training_artifact.artifact_id, dataset_artifact.artifact_id)
    artifact_manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=out_dir,
        files=(config_path, report_path),
        command=command,
        config=config_payload,
        parent_artifacts=parent_artifacts,
        source_git_sha=source_git_sha,
        created_at=created_at,
        metadata={
            "schema_version": LATENT_PROBE_EVAL_RUN_SCHEMA_VERSION,
            "report_schema_version": report.schema_version,
            "report_path": "reports/latent_probe_report.json",
            "checkpoint_sha256": checkpoint_manifest.checkpoint_sha256,
            "checkpoint_action_view": model.config.action_view,
            "checkpoint_step": _optional_int(checkpoint_payload.get("step")),
            "dataset_artifact_id": dataset_artifact.artifact_id,
            "training_artifact_id": training_artifact.artifact_id,
            "row_count": report.row_count,
            "split_counts": dict(report.split_counts),
            "claim_boundary": dict(report.claim_boundary),
        },
    )
    manifest_path = out_dir / "manifest.json"
    write_artifact_manifest(artifact_manifest, manifest_path)

    return LatentProbeEvalResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path="reports/latent_probe_report.json",
        parent_artifacts=parent_artifacts,
        row_count=report.row_count,
        split_counts=report.split_counts,
        claim_boundary=report.claim_boundary,
    )


def _evaluate_probe_rows(
    rows: tuple[Any, ...],
    *,
    model: Any,
    runtime: Any,
    device: Any,
    seed: int,
    bootstrap_samples: int,
    metadata: Mapping[str, Any],
) -> LatentProbeReport:
    if not rows:
        raise LatentProbeError("latent probe requires at least one packed row")
    z_before, z_pred_after, z_after = _embed_rows(rows, model=model, runtime=runtime, device=device)
    z_pred_after_shuffled = _embed_shuffled_predictions(
        rows,
        z_before=z_before,
        model=model,
        runtime=runtime,
        device=device,
        seed=seed,
    )
    z_pred_np = _to_numpy(z_pred_after)
    random_latent = _random_latent_like(z_pred_np, seed=seed)
    config = LatentProbeConfig(bootstrap_samples=bootstrap_samples, seed=seed)
    return build_latent_probe_report(
        tuple(_probe_row(row) for row in rows),
        embeddings={
            "z_before": _to_numpy(z_before),
            "z_after": _to_numpy(z_after),
            "z_pred_after": z_pred_np,
        },
        baselines={
            "random_latent": random_latent,
            "no_action": _to_numpy(z_before),
            "shuffled_action": _to_numpy(z_pred_after_shuffled),
        },
        config=config,
        metadata=metadata,
    )


def _load_probe_eval_rows(
    pack_paths: _PackPaths,
    *,
    action_view: str,
    max_examples_per_split: int,
    seed: int,
) -> tuple[Any, ...]:
    rows: list[Any] = []
    for split in ("train", "val", "test"):
        split_rows = list(
            _load_split_rows(
                pack_paths,
                split=split,
                action_view=action_view,
                vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
            )
        )
        if len(split_rows) > max_examples_per_split:
            rng = random.Random(seed + _stable_seed_offset(split))
            rng.shuffle(split_rows)
            split_rows = sorted(split_rows[:max_examples_per_split], key=lambda row: row.transition_id)
        rows.extend(split_rows)
    if not rows:
        raise LatentProbeError("packed dataset has no rows for latent probe evaluation")
    return tuple(rows)


def _embed_shuffled_predictions(
    rows: tuple[Any, ...],
    *,
    z_before: Any,
    model: Any,
    runtime: Any,
    device: Any,
    seed: int,
) -> Any:
    action_rows = [row.action for row in rows]
    if len(action_rows) > 1:
        rng = random.Random(seed + 991)
        rng.shuffle(action_rows)
    action = _action_batch(
        tuple(action_rows),
        runtime=runtime,
        device=device,
        action_view=model.config.action_view,
    )
    was_training = bool(model.training)
    model.eval()
    with runtime.no_grad():
        action_emb = model.encode_action(action)
        z_pred_after = model.predict_after(z_before, action_emb)
    if was_training:
        model.train()
    return z_pred_after.float()


def _probe_row(row: Any) -> LatentProbeRow:
    edit_bucket = str(row.metadata.get("edit_size_bucket") or _edit_size_bucket(row.edit_size))
    action_cluster = str(row.metadata.get("action_cluster") or "")
    state_after_kind = str(row.metadata.get("state_after_kind") or "")
    state_after_symbol = str(row.metadata.get("state_after_symbol") or "")
    state_after_fallback = str(row.metadata.get("state_after_fallback_reason") or "")
    labels = {
        "edit_class": str(row.metadata.get("diff_shape") or edit_bucket),
        "ast_node_kind": state_after_kind or None,
        "symbol_kind": _symbol_kind(state_after_kind, state_after_symbol, state_after_fallback),
        "edit_size_bucket": edit_bucket,
        "action_cluster": action_cluster or None,
        "source_family": row.source or None,
    }
    metadata_features = {
        "source": row.source,
        "path_suffix": Path(row.path).suffix or "<none>",
        "edit_size_bucket": edit_bucket,
        "action_cluster": action_cluster or "<none>",
        "ast_node_kind": state_after_kind or "<missing>",
        "symbol_kind": labels["symbol_kind"] or "<missing>",
    }
    return LatentProbeRow(
        transition_id=row.transition_id,
        split=row.split,
        labels=labels,
        metadata_features=metadata_features,
        lexical_tokens=_active_tokens(row.state_after) + _active_tokens(row.action),
    )


def _symbol_kind(kind: str, symbol: str, fallback_reason: str) -> str | None:
    if symbol and kind:
        return f"{kind}:named"
    if kind in {"region", "small_file"}:
        return f"{kind}:fallback:{fallback_reason or 'unknown'}"
    if kind:
        return f"{kind}:unnamed"
    return None


def _active_tokens(group: Mapping[str, Any]) -> tuple[int, ...]:
    ids = [] if group.get("input_ids") is None else list(group["input_ids"])
    masks = [] if group.get("attention_mask") is None else list(group["attention_mask"])
    if not masks:
        masks = [token != 0 for token in ids]
    return tuple(int(token) for token, keep in zip(ids, masks) if keep and int(token) != 0)


def _to_numpy(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise LatentProbeError("latent probe embedding matrix contains NaN or inf")
    return np.asarray(array, dtype=np.float64)


def _random_latent_like(matrix: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(matrix.shape).astype(np.float64)


def _reject_existing_latent_probe_outputs(out_dir: Path, *, overwrite: bool) -> None:
    for path in (
        out_dir / "config.json",
        out_dir / "reports" / "latent_probe_report.json",
        out_dir / "manifest.json",
    ):
        if path.exists() and not overwrite:
            raise LatentProbeError(f"output already exists; pass --overwrite to replace: {path}")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _edit_size_bucket(edit_size: int) -> str:
    start = (int(edit_size) // 10) * 10
    return f"{start}-{start + 9}"


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LatentProbeError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise LatentProbeError(f"{name} must be a positive integer")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LatentProbeError(f"{name} must be a non-negative integer")
    parsed = int(value)
    if parsed < 0:
        raise LatentProbeError(f"{name} must be a non-negative integer")
    return parsed


def _stable_seed_offset(value: str) -> int:
    digest = 0
    for byte in value.encode("utf-8"):
        digest = ((digest * 131) + byte) % (2**31 - 1)
    return digest
