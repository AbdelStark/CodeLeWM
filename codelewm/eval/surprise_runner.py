"""Manifest-backed patch-surprise evaluation over packed CodeLeWM artifacts."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.observability import (
    ArtifactManifestError,
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security import require_trusted_checkpoint
from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE

from .retrieval_runner import (
    _EvalRow,
    _action_discriminative_claim_ready,
    _action_discriminative_hard_negative_pools,
    _action_cluster,
    _display_path,
    _infer_training_artifact_manifest_path,
    _load_heldout_rows,
    _load_torch_checkpoint,
    _optional_int,
    _read_action_discriminative_report,
    _read_verified_artifact_manifest,
    _require_torch_runtime,
    _resolve_device,
    _resolve_pack_paths,
    _state_batch,
    _write_json,
)
from .surprise import (
    SURPRISE_DECOY_CATEGORIES,
    SurpriseEvalError,
    SurpriseExampleResult,
    SurpriseMetrics,
    build_surprise_report,
    write_surprise_report,
)


SURPRISE_EVAL_RUN_SCHEMA_VERSION = "codelewm.eval.surprise_run.v1"


@dataclass(frozen=True)
class SurpriseEvalResult:
    """CLI-facing summary for a manifest-backed surprise evaluation run."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    parent_artifacts: tuple[str, ...]
    metrics: SurpriseMetrics
    metadata: Mapping[str, Any]
    schema_version: str = SURPRISE_EVAL_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "parent_artifacts": list(self.parent_artifacts),
            "metrics": self.metrics.to_dict(),
            "metadata": dict(self.metadata),
        }


def run_surprise_evaluation(
    *,
    checkpoint: Path | str,
    data: Path | str,
    out: Path | str,
    device: str = "cpu",
    max_examples: int = 1000,
    random_decoys: int = 1,
    same_file_decoys: int = 1,
    mutation_decoys: int = 1,
    action_cluster_decoys: int = 1,
    seed: int = 0,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "surprise"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> SurpriseEvalResult:
    """Run model-backed patch-surprise evaluation and write an artifact."""

    _positive_int(max_examples, "max_examples")
    _non_negative_int(random_decoys, "random_decoys")
    _non_negative_int(same_file_decoys, "same_file_decoys")
    _non_negative_int(mutation_decoys, "mutation_decoys")
    _non_negative_int(action_cluster_decoys, "action_cluster_decoys")
    if random_decoys + same_file_decoys + mutation_decoys + action_cluster_decoys <= 0:
        raise SurpriseEvalError("at least one decoy category must request a positive count")

    checkpoint_path = Path(checkpoint).resolve()
    out_dir = Path(out).resolve()
    _reject_existing_surprise_outputs(out_dir, overwrite=overwrite)

    pack_paths = _resolve_pack_paths(data)
    dataset_artifact = _read_verified_artifact_manifest(
        pack_paths.artifact_manifest_path,
        root=pack_paths.root,
    )
    if dataset_artifact.artifact_kind != "dataset":
        raise ArtifactManifestError("surprise --data manifest must be a dataset artifact")
    action_discriminative_report = _read_action_discriminative_report(pack_paths, dataset_artifact)
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
        raise SurpriseEvalError(
            "checkpoint manifest action_view does not match checkpoint payload: "
            f"{action_view!r} != {model.config.action_view!r}"
        )

    rows = _select_eval_rows(
        _load_heldout_rows(
            pack_paths,
            action_view=model.config.action_view,
            vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
        ),
        max_examples=max_examples,
        seed=seed,
    )
    results, category_slices = _evaluate_rows(
        rows,
        model=model,
        runtime=runtime,
        device=selected_device,
        seed=seed,
        random_decoys=random_decoys,
        same_file_decoys=same_file_decoys,
        mutation_decoys=mutation_decoys,
        action_cluster_decoys=action_cluster_decoys,
    )
    report = build_surprise_report(
        results,
        decoy_seed=seed,
        score_direction="lower_is_better",
        metadata={
            "checkpoint": {
                "path": _display_path(checkpoint_path),
                "sha256": checkpoint_manifest.checkpoint_sha256,
                "step": _optional_int(checkpoint_payload.get("step"), "checkpoint.step"),
                "model_class": "TorchCodeTransitionModel",
                "backend": "torch",
            },
            "dataset": {
                "path": _display_path(pack_paths.root),
                "artifact_id": dataset_artifact.artifact_id,
                "split_counts": dict(dataset_artifact.metadata.get("split_counts", {})),
            },
            "action_discriminative_shard_report": dict(action_discriminative_report),
            "training_artifact_id": training_artifact.artifact_id,
            "action_view": model.config.action_view,
            "score": {
                "direction": "lower_is_better",
                "value": "squared_l2",
            },
            "decoy_policy": {
                "random": random_decoys,
                "same_file": same_file_decoys,
                "mutation": mutation_decoys,
                "action_cluster": action_cluster_decoys,
                "max_examples": max_examples,
            },
            "category_slices": category_slices,
            "category_caveats": {
                category: slice_payload["caveat"]
                for category, slice_payload in category_slices.items()
                if slice_payload.get("caveat")
            },
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "schema_version": SURPRISE_EVAL_RUN_SCHEMA_VERSION,
        "checkpoint": _display_path(checkpoint_path),
        "data": _display_path(pack_paths.root),
        "out": _display_path(out_dir),
        "device": str(selected_device),
        "max_examples": max_examples,
        "random_decoys": random_decoys,
        "same_file_decoys": same_file_decoys,
        "mutation_decoys": mutation_decoys,
        "action_cluster_decoys": action_cluster_decoys,
        "seed": seed,
        "action_view": model.config.action_view,
        "score_direction": "lower_is_better",
    }
    config_path = out_dir / "config.json"
    report_path = out_dir / "reports" / "surprise_report.json"
    _write_json(config_payload, config_path)
    write_surprise_report(report, report_path)

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
            "schema_version": SURPRISE_EVAL_RUN_SCHEMA_VERSION,
            "report_schema_version": report.schema_version,
            "report_path": "reports/surprise_report.json",
            "checkpoint_sha256": checkpoint_manifest.checkpoint_sha256,
            "checkpoint_action_view": model.config.action_view,
            "checkpoint_step": _optional_int(checkpoint_payload.get("step"), "checkpoint.step"),
            "dataset_artifact_id": dataset_artifact.artifact_id,
            "training_artifact_id": training_artifact.artifact_id,
            "example_count": report.metrics.example_count,
            "decoy_counts": report.metrics.decoy_counts,
            "metrics": report.metrics.to_dict(),
            "category_caveats": report.metadata.get("category_caveats", {}),
            "action_discriminative_claim_ready": _action_discriminative_claim_ready(
                action_discriminative_report
            ),
            "action_discriminative_hard_negative_pools": _action_discriminative_hard_negative_pools(
                action_discriminative_report
            ),
        },
    )
    manifest_path = out_dir / "manifest.json"
    write_artifact_manifest(artifact_manifest, manifest_path)

    return SurpriseEvalResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path="reports/surprise_report.json",
        parent_artifacts=parent_artifacts,
        metrics=report.metrics,
        metadata={
            "action_view": model.config.action_view,
            "example_count": report.metrics.example_count,
            "decoy_counts": report.metrics.decoy_counts,
            "category_caveats": report.metadata.get("category_caveats", {}),
            "action_discriminative_claim_ready": _action_discriminative_claim_ready(
                action_discriminative_report
            ),
        },
    )


def _evaluate_rows(
    rows: tuple[_EvalRow, ...],
    *,
    model: Any,
    runtime: Any,
    device: Any,
    seed: int,
    random_decoys: int,
    same_file_decoys: int,
    mutation_decoys: int,
    action_cluster_decoys: int,
) -> tuple[tuple[SurpriseExampleResult, ...], dict[str, dict[str, Any]]]:
    if not rows:
        raise SurpriseEvalError("surprise evaluation requires at least one held-out row")
    _, z_pred_after, z_after = _embed_rows_for_surprise(
        rows,
        model=model,
        runtime=runtime,
        device=device,
    )
    requested = {
        "random": random_decoys,
        "same_file": same_file_decoys,
        "mutation": mutation_decoys,
        "action_cluster": action_cluster_decoys,
    }
    decoy_counts = {category: 0 for category in SURPRISE_DECOY_CATEGORIES}
    examples_with_decoys = {category: 0 for category in SURPRISE_DECOY_CATEGORIES}
    results: list[SurpriseExampleResult] = []

    for query_index, row in enumerate(rows):
        true_score = _energy(z_pred_after[query_index], z_after[query_index], runtime=runtime)
        decoy_scores: dict[str, list[float]] = {category: [] for category in SURPRISE_DECOY_CATEGORIES}
        for decoy_index in _sample_row_indices(
            rows,
            query_index=query_index,
            category="random",
            count=random_decoys,
            seed=seed,
        ):
            decoy_scores["random"].append(
                _energy(z_pred_after[query_index], z_after[decoy_index], runtime=runtime)
            )
        for decoy_index in _sample_row_indices(
            rows,
            query_index=query_index,
            category="same_file",
            count=same_file_decoys,
            seed=seed,
        ):
            decoy_scores["same_file"].append(
                _energy(z_pred_after[query_index], z_after[decoy_index], runtime=runtime)
            )
        for decoy_index in _sample_row_indices(
            rows,
            query_index=query_index,
            category="action_cluster",
            count=action_cluster_decoys,
            seed=seed,
        ):
            decoy_scores["action_cluster"].append(
                _energy(z_pred_after[query_index], z_after[decoy_index], runtime=runtime)
            )
        for mutation_state in _build_mutation_states(row, count=mutation_decoys, seed=seed):
            mutation_z = _embed_state(mutation_state, model=model, runtime=runtime, device=device)
            decoy_scores["mutation"].append(_energy(z_pred_after[query_index], mutation_z[0], runtime=runtime))

        for category, scores in decoy_scores.items():
            decoy_counts[category] += len(scores)
            if scores:
                examples_with_decoys[category] += 1
        flat_scores = [true_score, *(score for scores in decoy_scores.values() for score in scores)]
        results.append(
            SurpriseExampleResult(
                transition_id=row.transition_id,
                true_score=true_score,
                decoy_scores_by_category={
                    category: tuple(scores)
                    for category, scores in decoy_scores.items()
                },
                true_rank=_rank_lower_is_better(flat_scores, true_index=0),
                candidate_count=len(flat_scores),
            )
        )

    if sum(decoy_counts.values()) == 0:
        raise SurpriseEvalError("surprise evaluation requires at least one generated decoy")
    metrics = build_surprise_report(results, decoy_seed=seed).metrics
    category_slices = {
        category: {
            "requested_per_example": requested[category],
            "decoy_count": decoy_counts[category],
            "example_count_with_decoy": examples_with_decoys[category],
            "example_count": len(rows),
            "pairwise_auc": metrics.pairwise_auc_by_category.get(category),
            "caveat": _category_caveat(
                category,
                requested_count=requested[category],
                decoy_count=decoy_counts[category],
            ),
        }
        for category in SURPRISE_DECOY_CATEGORIES
    }
    return tuple(results), category_slices


def _embed_rows_for_surprise(
    rows: tuple[_EvalRow, ...],
    *,
    model: Any,
    runtime: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    if not rows:
        raise SurpriseEvalError("surprise evaluation requires at least one held-out row")
    state_before = _state_batch(tuple(row.state_before for row in rows), runtime=runtime, device=device)
    state_after = _state_batch(tuple(row.state_after for row in rows), runtime=runtime, device=device)
    action = _action_batch_for_model(rows, model=model, runtime=runtime, device=device)
    was_training = bool(model.training)
    model.eval()
    with runtime.no_grad():
        z_before = model.encode_state(state_before)
        action_emb = model.encode_action(action)
        z_after = model.encode_state(state_after)
        z_pred_after = model.predict_after(z_before, action_emb)
    if was_training:
        model.train()
    return z_before.float(), z_pred_after.float(), z_after.float()


def _action_batch_for_model(rows: Sequence[_EvalRow], *, model: Any, runtime: Any, device: Any) -> Any:
    from .retrieval_runner import _action_batch

    return _action_batch(
        tuple(row.action for row in rows),
        runtime=runtime,
        device=device,
        action_view=model.config.action_view,
    )


def _embed_state(state: Mapping[str, np.ndarray], *, model: Any, runtime: Any, device: Any) -> Any:
    was_training = bool(model.training)
    model.eval()
    with runtime.no_grad():
        encoded = model.encode_state(_state_batch((state,), runtime=runtime, device=device)).float()
    if was_training:
        model.train()
    return encoded


def _sample_row_indices(
    rows: tuple[_EvalRow, ...],
    *,
    query_index: int,
    category: str,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    if count <= 0:
        return ()
    query = rows[query_index]
    if category == "random":
        pool = [
            index
            for index, row in enumerate(rows)
            if index != query_index and row.transition_id != query.transition_id
        ]
    elif category == "same_file":
        pool = [
            index
            for index, row in enumerate(rows)
            if index != query_index and row.repo == query.repo and row.path == query.path
        ]
    elif category == "action_cluster":
        query_cluster = str(query.metadata.get("action_cluster") or _action_cluster(query.metadata))
        pool = [
            index
            for index, row in enumerate(rows)
            if index != query_index and str(row.metadata.get("action_cluster")) == query_cluster
        ]
    else:  # pragma: no cover - guarded by caller constants.
        raise SurpriseEvalError(f"unsupported decoy category: {category}")
    if not pool:
        return ()
    rng = _category_rng(seed, query.transition_id, category)
    return tuple(
        rng.sample(
            sorted(pool, key=lambda index: rows[index].transition_id),
            k=min(count, len(pool)),
        )
    )


def _build_mutation_states(row: _EvalRow, *, count: int, seed: int) -> tuple[Mapping[str, np.ndarray], ...]:
    if count <= 0:
        return ()
    rng = _category_rng(seed, row.transition_id, "mutation")
    states: list[Mapping[str, np.ndarray]] = []
    for _ in range(count):
        states.append(_mutate_state_after(row.state_after, rng=rng))
    return tuple(states)


def _mutate_state_after(state: Mapping[str, np.ndarray], *, rng: random.Random) -> Mapping[str, np.ndarray]:
    mutated = {key: np.asarray(value).copy() for key, value in state.items()}
    input_ids = np.asarray(mutated["input_ids"], dtype=np.int64).copy()
    attention_mask = np.asarray(mutated["attention_mask"], dtype=bool).copy()
    active_positions = np.flatnonzero(attention_mask & (input_ids > 0))
    if active_positions.size == 0:
        if input_ids.size == 0:
            raise SurpriseEvalError("cannot build mutation decoy from empty state input_ids")
        position = 0
        attention_mask[position] = True
        input_ids[position] = 1
    else:
        position = int(rng.choice(active_positions.tolist()))
        old_value = int(input_ids[position])
        delta = rng.randint(1, DEFAULT_TRAINING_VOCAB_SIZE - 2)
        input_ids[position] = ((old_value + delta - 1) % (DEFAULT_TRAINING_VOCAB_SIZE - 1)) + 1
    mutated["input_ids"] = input_ids
    mutated["attention_mask"] = attention_mask
    return mutated


def _energy(query_vector: Any, candidate_vector: Any, *, runtime: Any) -> float:
    value = (query_vector - candidate_vector).float().pow(2).sum().detach().cpu().item()
    return _finite_float(value, "surprise score")


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise SurpriseEvalError(f"{name} must be finite")
    return result


def _rank_lower_is_better(scores: Sequence[float], *, true_index: int) -> int:
    true_score = scores[true_index]
    better = sum(1 for index, score in enumerate(scores) if index != true_index and score < true_score)
    ties = sum(1 for index, score in enumerate(scores) if index != true_index and score == true_score)
    return 1 + better + ties // 2


def _category_rng(seed: int, transition_id: str, category: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{transition_id}|{category}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _category_caveat(category: str, *, requested_count: int, decoy_count: int) -> str | None:
    if requested_count <= 0:
        return "category disabled by decoy policy"
    if decoy_count > 0:
        return None
    if category == "random":
        return "no other held-out after-states were available"
    if category == "same_file":
        return "no same-file held-out after-states were available"
    if category == "action_cluster":
        return "no same-action-cluster held-out after-states were available"
    if category == "mutation":
        return "no mutation decoys could be generated from held-out after-states"
    return "decoy category unavailable"


def _select_eval_rows(rows: tuple[_EvalRow, ...], *, max_examples: int, seed: int) -> tuple[_EvalRow, ...]:
    if len(rows) <= max_examples:
        return rows
    rng = random.Random(seed)
    selected = rng.sample(list(rows), k=max_examples)
    return tuple(sorted(selected, key=lambda row: row.transition_id))


def _reject_existing_surprise_outputs(out_dir: Path, *, overwrite: bool) -> None:
    for path in (
        out_dir / "config.json",
        out_dir / "reports" / "surprise_report.json",
        out_dir / "manifest.json",
    ):
        if path.exists() and not overwrite:
            raise SurpriseEvalError(f"output already exists; pass --overwrite to replace: {path}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise SurpriseEvalError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SurpriseEvalError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise SurpriseEvalError(f"{name} must be a positive integer")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise SurpriseEvalError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SurpriseEvalError(f"{name} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise SurpriseEvalError(f"{name} must be a non-negative integer")
    return result


__all__ = [
    "SURPRISE_EVAL_RUN_SCHEMA_VERSION",
    "SurpriseEvalResult",
    "run_surprise_evaluation",
]
