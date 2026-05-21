"""Manifest-backed latent matrix diagnostic evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codelewm.observability import (
    ArtifactManifestError,
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security import require_trusted_checkpoint
from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE

from .latent_matrix import (
    LATENT_MATRIX_REPORT_SCHEMA_VERSION,
    LATENT_MATRIX_VIEWS,
    LatentMatrixConfig,
    LatentMatrixError,
    LatentMatrixReport,
    build_latent_matrix_report,
    read_optional_latent_probe_report,
    write_latent_matrix_report,
)
from .latent_probe_runner import (
    _probe_row,
    _to_numpy,
)
from .retrieval_runner import (
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


LATENT_MATRIX_EVAL_RUN_SCHEMA_VERSION = "codelewm.eval.latent_matrix_run.v1"


@dataclass(frozen=True)
class LatentMatrixEvalResult:
    """CLI-facing summary for a manifest-backed latent matrix run."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    parent_artifacts: tuple[str, ...]
    row_count: int
    split_counts: Mapping[str, int]
    view_shapes: Mapping[str, Any]
    claim_boundary: Mapping[str, Any]
    schema_version: str = LATENT_MATRIX_EVAL_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "parent_artifacts": list(self.parent_artifacts),
            "row_count": self.row_count,
            "split_counts": dict(self.split_counts),
            "view_shapes": dict(self.view_shapes),
            "claim_boundary": dict(self.claim_boundary),
        }


def run_latent_matrix_evaluation(
    *,
    checkpoint: Path | str,
    data: Path | str,
    out: Path | str,
    device: str = "cpu",
    max_examples_per_split: int = 1000,
    matrix_dimension_limit: int = 32,
    top_dimensions: int = 16,
    max_pairwise_rows: int = 512,
    latent_probe_report: Path | str | None = None,
    seed: int = 0,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "latent-matrix"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> LatentMatrixEvalResult:
    """Run latent matrix diagnostics and write a manifest-backed report."""

    _positive_int(max_examples_per_split, "max_examples_per_split")
    checkpoint_path = Path(checkpoint).resolve()
    out_dir = Path(out).resolve()
    _reject_existing_latent_matrix_outputs(out_dir, overwrite=overwrite)

    pack_paths = _resolve_pack_paths(data)
    dataset_artifact = _read_verified_artifact_manifest(pack_paths.artifact_manifest_path, root=pack_paths.root)
    if dataset_artifact.artifact_kind != "dataset":
        raise ArtifactManifestError("latent-matrix --data manifest must be a dataset artifact")
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
        raise LatentMatrixError(
            "checkpoint manifest action_view does not match checkpoint payload: "
            f"{action_view!r} != {model.config.action_view!r}"
        )

    rows = _load_latent_matrix_rows(
        pack_paths,
        action_view=model.config.action_view,
        max_examples_per_split=max_examples_per_split,
        seed=seed,
    )
    linked_probe_report = read_optional_latent_probe_report(latent_probe_report)
    report = _evaluate_latent_matrix_rows(
        rows,
        model=model,
        runtime=runtime,
        device=selected_device,
        seed=seed,
        matrix_dimension_limit=matrix_dimension_limit,
        top_dimensions=top_dimensions,
        max_pairwise_rows=max_pairwise_rows,
        latent_probe_report=linked_probe_report,
        latent_probe_report_path=latent_probe_report,
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
        "schema_version": LATENT_MATRIX_EVAL_RUN_SCHEMA_VERSION,
        "checkpoint": _display_path(checkpoint_path),
        "data": _display_path(pack_paths.root),
        "out": _display_path(out_dir),
        "device": str(selected_device),
        "max_examples_per_split": max_examples_per_split,
        "matrix_dimension_limit": matrix_dimension_limit,
        "top_dimensions": top_dimensions,
        "max_pairwise_rows": max_pairwise_rows,
        "latent_probe_report": None
        if latent_probe_report is None
        else _display_path(Path(latent_probe_report).resolve()),
        "seed": seed,
        "action_view": model.config.action_view,
        "views": list(LATENT_MATRIX_VIEWS),
    }
    config_path = out_dir / "config.json"
    report_path = out_dir / "reports" / "latent_matrix_report.json"
    _write_json(config_payload, config_path)
    write_latent_matrix_report(report, report_path)

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
            "schema_version": LATENT_MATRIX_EVAL_RUN_SCHEMA_VERSION,
            "report_schema_version": report.schema_version,
            "report_path": "reports/latent_matrix_report.json",
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

    return LatentMatrixEvalResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path="reports/latent_matrix_report.json",
        parent_artifacts=parent_artifacts,
        row_count=report.row_count,
        split_counts=report.split_counts,
        view_shapes={view: payload["shape"] for view, payload in report.views.items()},
        claim_boundary=report.claim_boundary,
    )


def _evaluate_latent_matrix_rows(
    rows: tuple[Any, ...],
    *,
    model: Any,
    runtime: Any,
    device: Any,
    seed: int,
    matrix_dimension_limit: int,
    top_dimensions: int,
    max_pairwise_rows: int,
    latent_probe_report: Any,
    latent_probe_report_path: Path | str | None,
    metadata: Mapping[str, Any],
) -> LatentMatrixReport:
    if not rows:
        raise LatentMatrixError("latent matrix evaluation requires at least one packed row")
    z_before, z_pred_after, z_after = _embed_rows(rows, model=model, runtime=runtime, device=device)
    config = LatentMatrixConfig(
        matrix_dimension_limit=matrix_dimension_limit,
        top_dimensions=top_dimensions,
        max_pairwise_rows=max_pairwise_rows,
        seed=seed,
    )
    return build_latent_matrix_report(
        tuple(_probe_row(row) for row in rows),
        embeddings={
            "z_before": _to_numpy(z_before),
            "z_after": _to_numpy(z_after),
            "z_pred_after": _to_numpy(z_pred_after),
        },
        config=config,
        latent_probe_report=latent_probe_report,
        latent_probe_report_path=latent_probe_report_path,
        metadata=metadata,
    )


def _load_latent_matrix_rows(
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
            split_rows = _stable_sample(split_rows, limit=max_examples_per_split, seed=seed, split=split)
        rows.extend(split_rows)
    if not rows:
        raise LatentMatrixError("packed dataset has no rows for latent matrix evaluation")
    return tuple(rows)


def _stable_sample(rows: list[Any], *, limit: int, seed: int, split: str) -> list[Any]:
    import random

    rng = random.Random(seed + _stable_seed_offset(split))
    rng.shuffle(rows)
    return sorted(rows[:limit], key=lambda row: row.transition_id)


def _reject_existing_latent_matrix_outputs(out_dir: Path, *, overwrite: bool) -> None:
    for path in (
        out_dir / "config.json",
        out_dir / "reports" / "latent_matrix_report.json",
        out_dir / "manifest.json",
    ):
        if path.exists() and not overwrite:
            raise LatentMatrixError(f"output already exists; pass --overwrite to replace: {path}")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LatentMatrixError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise LatentMatrixError(f"{name} must be a positive integer")
    return parsed


def _stable_seed_offset(value: str) -> int:
    digest = 0
    for byte in value.encode("utf-8"):
        digest = ((digest * 131) + byte) % (2**31 - 1)
    return digest
