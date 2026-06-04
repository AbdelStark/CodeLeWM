"""Manifest-backed retrieval evaluation over packed CodeLeWM artifacts."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from codelewm.data import (
    DATASET_SCHEMA_VERSION,
    OptionalDependencyError,
    token_sequence_hash,
    token_sequence_simhash,
    validate_action_discriminative_shard_report_payload,
)
from codelewm.model import (
    ActionBatch,
    CodeStateBatch,
    TorchCodeTransitionModelConfig,
    build_torch_transition_model,
    resolve_ema_target_encoder_config,
    resolve_output_value_head_config,
    resolve_state_encoder_arch,
)
from codelewm.observability import (
    ArtifactManifest,
    ArtifactManifestError,
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security import require_trusted_checkpoint
from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE, TORCH_CHECKPOINT_SCHEMA_VERSION

from .action_policy import build_action_view_report_policy
from .retrieval import (
    ActionContrastPoolConfig,
    CandidatePoolEntry,
    HardNegativeSamplerConfig,
    RetrievalEvalError,
    RetrievalMetrics,
    RetrievalReport,
    build_action_contrast_pool_report,
    build_baseline_metrics,
    build_easy_candidate_pool,
    build_hard_candidate_pool,
    build_hard_negative_sampler_report,
    build_retrieval_report,
    lexical_baseline_ranks,
    no_action_baseline_ranks,
    random_baseline_ranks,
    rank_targets,
    shuffled_action_baseline_ranks,
    write_action_contrast_pool_report,
    validate_required_headline_baselines,
    write_retrieval_report,
)


RETRIEVAL_EVAL_RUN_SCHEMA_VERSION = "codelewm.eval.retrieval_run.v1"


@dataclass(frozen=True)
class RetrievalEvalResult:
    """CLI-facing summary for a manifest-backed retrieval evaluation run."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    hard_negative_sampler_report_path: str
    action_contrast_pool_report_path: str
    parent_artifacts: tuple[str, ...]
    metrics: RetrievalMetrics
    baselines: Mapping[str, RetrievalMetrics]
    metadata: Mapping[str, Any]
    schema_version: str = RETRIEVAL_EVAL_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "hard_negative_sampler_report_path": self.hard_negative_sampler_report_path,
            "action_contrast_pool_report_path": self.action_contrast_pool_report_path,
            "parent_artifacts": list(self.parent_artifacts),
            "metrics": self.metrics.to_dict(),
            "baselines": {
                name: metrics.to_dict()
                for name, metrics in sorted(self.baselines.items())
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _PackPaths:
    root: Path
    artifact_manifest_path: Path
    hdf5_dir: Path
    parquet_dir: Path


@dataclass(frozen=True)
class _EvalRow:
    transition_id: str
    split: str
    source: str
    repo: str
    path: str
    edit_size: int
    state_before: Mapping[str, np.ndarray]
    state_after: Mapping[str, np.ndarray]
    action: Mapping[str, np.ndarray]
    action_text: str
    candidate_text: str
    metadata: Mapping[str, Any]

    def candidate_entry(self) -> CandidatePoolEntry:
        return CandidatePoolEntry(
            transition_id=self.transition_id,
            split=self.split,
            source=self.source,
            repo=self.repo,
            path=self.path,
            edit_size=self.edit_size,
            metadata=self.metadata,
        )


def run_retrieval_evaluation(
    *,
    checkpoint: Path | str,
    data: Path | str,
    out: Path | str,
    device: str = "cpu",
    max_candidates: int = 1000,
    hard_negatives: int = 1000,
    seed: int = 0,
    report_scope: Literal["headline", "ablation", "diagnostic"] = "headline",
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "retrieval"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> RetrievalEvalResult:
    """Run retrieval evaluation and write report plus artifact manifest."""

    _positive_int(max_candidates, "max_candidates")
    _positive_int(hard_negatives, "hard_negatives")
    if report_scope not in {"headline", "ablation", "diagnostic"}:
        raise RetrievalEvalError("report_scope must be headline, ablation, or diagnostic")

    checkpoint_path = Path(checkpoint).resolve()
    out_dir = Path(out).resolve()
    _reject_existing_retrieval_outputs(out_dir, overwrite=overwrite)

    pack_paths = _resolve_pack_paths(data)
    dataset_artifact = _read_verified_artifact_manifest(pack_paths.artifact_manifest_path, root=pack_paths.root)
    if dataset_artifact.artifact_kind != "dataset":
        raise ArtifactManifestError("retrieval --data manifest must be a dataset artifact")
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
        raise RetrievalEvalError(
            "checkpoint manifest action_view does not match checkpoint payload: "
            f"{action_view!r} != {model.config.action_view!r}"
        )
    policy = build_action_view_report_policy(
        model.config.action_view,
        report_scope=report_scope,
    )

    rows = _load_heldout_rows(
        pack_paths,
        action_view=model.config.action_view,
        vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
    )
    report, hard_report, action_contrast_report = _evaluate_rows(
        rows,
        model=model,
        runtime=runtime,
        device=selected_device,
        max_candidates=max_candidates,
        hard_negatives=hard_negatives,
        seed=seed,
        action_view=model.config.action_view,
        policy=policy.to_dict(),
        checkpoint_sha256=checkpoint_manifest.checkpoint_sha256,
        checkpoint_step=_optional_int(checkpoint_payload.get("step"), "checkpoint.step"),
        dataset_artifact=dataset_artifact,
        training_artifact=training_artifact,
        data_path=_display_path(pack_paths.root),
        checkpoint_path=_display_path(checkpoint_path),
        action_discriminative_report=action_discriminative_report,
    )
    if report_scope == "headline":
        validate_required_headline_baselines(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "schema_version": RETRIEVAL_EVAL_RUN_SCHEMA_VERSION,
        "checkpoint": _display_path(checkpoint_path),
        "data": _display_path(pack_paths.root),
        "out": _display_path(out_dir),
        "device": str(selected_device),
        "max_candidates": max_candidates,
        "hard_negatives": hard_negatives,
        "seed": seed,
        "report_scope": report_scope,
        "action_view": model.config.action_view,
    }
    config_path = out_dir / "config.json"
    report_path = out_dir / "reports" / "retrieval_report.json"
    hard_report_path = out_dir / "reports" / "hard_negative_sampler_report.json"
    action_contrast_report_path = out_dir / "reports" / "action_contrast_pool_report.json"
    _write_json(config_payload, config_path)
    write_retrieval_report(report, report_path)
    _write_json(hard_report.to_dict(), hard_report_path)
    write_action_contrast_pool_report(action_contrast_report, action_contrast_report_path)

    parent_artifacts = (training_artifact.artifact_id, dataset_artifact.artifact_id)
    artifact_manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=out_dir,
        files=(config_path, report_path, hard_report_path, action_contrast_report_path),
        command=command,
        config=config_payload,
        parent_artifacts=parent_artifacts,
        source_git_sha=source_git_sha,
        created_at=created_at,
        metadata={
            "schema_version": RETRIEVAL_EVAL_RUN_SCHEMA_VERSION,
            "report_schema_version": report.schema_version,
            "report_path": "reports/retrieval_report.json",
            "hard_negative_sampler_report_path": "reports/hard_negative_sampler_report.json",
            "action_contrast_pool_report_path": "reports/action_contrast_pool_report.json",
            "checkpoint_sha256": checkpoint_manifest.checkpoint_sha256,
            "checkpoint_action_view": model.config.action_view,
            "checkpoint_step": _optional_int(checkpoint_payload.get("step"), "checkpoint.step"),
            "dataset_artifact_id": dataset_artifact.artifact_id,
            "training_artifact_id": training_artifact.artifact_id,
            "query_count": report.metrics.query_count,
            "candidate_count_min": report.metrics.candidate_count_min,
            "candidate_count_max": report.metrics.candidate_count_max,
            "metrics": report.metrics.to_dict(),
            "baseline_deltas": {
                name: delta.to_dict()
                for name, delta in sorted(report.baseline_deltas.items())
            },
            "action_use_claim_gate": None
            if report.action_use_claim_gate is None
            else report.action_use_claim_gate.to_dict(),
            "action_discriminative_claim_ready": _action_discriminative_claim_ready(
                report.metadata.get("action_discriminative_shard_report")
            ),
            "action_discriminative_hard_negative_pools": _action_discriminative_hard_negative_pools(
                report.metadata.get("action_discriminative_shard_report")
            ),
            "required_baselines": sorted(report.baselines),
            "action_contrast_pool_report": action_contrast_report.summary_dict(),
        },
    )
    manifest_path = out_dir / "manifest.json"
    write_artifact_manifest(artifact_manifest, manifest_path)

    return RetrievalEvalResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path="reports/retrieval_report.json",
        hard_negative_sampler_report_path="reports/hard_negative_sampler_report.json",
        action_contrast_pool_report_path="reports/action_contrast_pool_report.json",
        parent_artifacts=parent_artifacts,
        metrics=report.metrics,
        baselines=report.baselines,
        metadata={
            "action_view": model.config.action_view,
            "report_scope": report_scope,
            "query_count": report.metrics.query_count,
            "candidate_count_min": report.metrics.candidate_count_min,
            "candidate_count_max": report.metrics.candidate_count_max,
            "action_use_claim_gate": None
            if report.action_use_claim_gate is None
            else report.action_use_claim_gate.to_dict(),
            "action_discriminative_claim_ready": _action_discriminative_claim_ready(
                report.metadata.get("action_discriminative_shard_report")
            ),
        },
    )


def _evaluate_rows(
    rows: tuple[_EvalRow, ...],
    *,
    model: Any,
    runtime: Any,
    device: Any,
    max_candidates: int,
    hard_negatives: int,
    seed: int,
    action_view: str,
    policy: Mapping[str, Any],
    checkpoint_sha256: str,
    checkpoint_step: int | None,
    dataset_artifact: ArtifactManifest,
    training_artifact: ArtifactManifest,
    data_path: str,
    checkpoint_path: str,
    action_discriminative_report: Mapping[str, Any],
) -> tuple[RetrievalReport, Any, Any]:
    entries = tuple(row.candidate_entry() for row in rows)
    easy_pool = build_easy_candidate_pool(
        entries,
        max_size=max_candidates,
        seed=seed,
        name="easy-1k" if max_candidates == 1000 else f"easy-{max_candidates}",
    )
    row_by_id = {row.transition_id: row for row in rows}
    row_index_by_id = {row.transition_id: index for index, row in enumerate(rows)}
    query_ids = easy_pool.candidate_ids
    query_indices = tuple(row_index_by_id[transition_id] for transition_id in query_ids)
    candidate_indices = tuple(row_index_by_id[transition_id] for transition_id in easy_pool.candidate_ids)
    candidate_ids_by_query = tuple(easy_pool.candidate_ids for _ in query_ids)

    z_before, z_pred_after, z_after = _embed_rows(rows, model=model, runtime=runtime, device=device)
    model_scores_all = _negative_squared_l2(z_pred_after, z_after, runtime=runtime)
    no_action_scores_all = _negative_squared_l2(z_before, z_after, runtime=runtime)
    score_rows = _select_score_rows(model_scores_all, query_indices, candidate_indices)
    no_action_score_rows = _select_score_rows(no_action_scores_all, query_indices, candidate_indices)
    candidate_counts = tuple(len(candidate_ids) for candidate_ids in candidate_ids_by_query)
    ranks = rank_targets(score_rows, candidate_ids_by_query, query_ids)
    query_texts = tuple(_query_text(row_by_id[query_id]) for query_id in query_ids)
    candidate_texts = tuple(row_by_id[candidate_id].candidate_text for candidate_id in easy_pool.candidate_ids)
    candidate_texts_by_query = tuple(candidate_texts for _ in query_ids)
    baseline_counts = {name: candidate_counts for name in ("random", "lexical", "no_action", "shuffled_action")}
    baselines = build_baseline_metrics(
        {
            "random": random_baseline_ranks(candidate_ids_by_query, query_ids, seed=seed),
            "lexical": lexical_baseline_ranks(
                query_texts,
                candidate_texts_by_query,
                candidate_ids_by_query,
                query_ids,
            ),
            "no_action": no_action_baseline_ranks(no_action_score_rows, candidate_ids_by_query, query_ids),
            "shuffled_action": shuffled_action_baseline_ranks(
                score_rows,
                candidate_ids_by_query,
                query_ids,
                seed=seed + 17,
            ),
        },
        candidate_counts=baseline_counts,
    )

    hard_config = HardNegativeSamplerConfig(max_negatives=hard_negatives, seed=seed, pool_name="hard-1k")
    hard_score_rows: list[tuple[float, ...]] = []
    hard_candidate_ids_by_query: list[tuple[str, ...]] = []
    hard_samples = []
    all_entries = tuple(row.candidate_entry() for row in rows)
    for query_id in query_ids:
        query_entry = row_by_id[query_id].candidate_entry()
        hard_pool, sample = build_hard_candidate_pool(
            query_entry,
            all_entries,
            target_id=query_id,
            config=hard_config,
        )
        hard_samples.append(sample)
        hard_ids = hard_pool.candidate_ids
        hard_indices = tuple(row_index_by_id[transition_id] for transition_id in hard_ids)
        hard_candidate_ids_by_query.append(hard_ids)
        hard_score_rows.append(
            tuple(
                float(model_scores_all[row_index_by_id[query_id], candidate_index])
                for candidate_index in hard_indices
            )
        )
    hard_ranks = rank_targets(tuple(hard_score_rows), tuple(hard_candidate_ids_by_query), query_ids)
    hard_counts = tuple(len(ids) for ids in hard_candidate_ids_by_query)
    hard_report = build_hard_negative_sampler_report(hard_samples, config=hard_config)
    action_contrast_config = ActionContrastPoolConfig(
        max_queries=len(query_ids),
        max_candidates_per_pool=min(max(hard_negatives, 1), 16),
        seed=seed,
        near_before_hamming_threshold=hard_config.near_before_hamming_threshold,
    )
    action_contrast_report = build_action_contrast_pool_report(
        all_entries,
        query_ids=query_ids,
        config=action_contrast_config,
    )

    slices = _build_slices(
        rows_by_id=row_by_id,
        query_ids=query_ids,
        ranks=ranks,
        candidate_counts=candidate_counts,
        hard_ranks=hard_ranks,
        hard_counts=hard_counts,
        action_view=action_view,
    )
    action_contrast_slices, action_contrast_metrics = _build_action_contrast_metrics(
        action_contrast_report,
        model_scores_all=model_scores_all,
        no_action_scores_all=no_action_scores_all,
        row_by_id=row_by_id,
        row_index_by_id=row_index_by_id,
        seed=seed,
    )
    slices.update(action_contrast_slices)
    report = build_retrieval_report(
        ranks,
        candidate_pool=easy_pool,
        candidate_counts=candidate_counts,
        baselines=baselines,
        slices=slices,
        metadata={
            "checkpoint": {
                "path": checkpoint_path,
                "sha256": checkpoint_sha256,
                "step": checkpoint_step,
                "model_class": "TorchCodeTransitionModel",
                "backend": "torch",
            },
            "dataset": {
                "path": data_path,
                "artifact_id": dataset_artifact.artifact_id,
                "split_counts": dict(dataset_artifact.metadata.get("split_counts", {})),
            },
            "action_discriminative_shard_report": dict(action_discriminative_report),
            "training_artifact_id": training_artifact.artifact_id,
            "action_view_policy": dict(policy),
            "score": {
                "direction": "larger_is_better",
                "value": "negative_squared_l2",
            },
            "hard_negative_sampler_report": hard_report.to_dict(),
            "action_contrast_pool_report": action_contrast_report.summary_dict(),
            "action_contrast_metrics": action_contrast_metrics,
        },
    )
    return report, hard_report, action_contrast_report


def _build_slices(
    *,
    rows_by_id: Mapping[str, _EvalRow],
    query_ids: Sequence[str],
    ranks: Sequence[int],
    candidate_counts: Sequence[int],
    hard_ranks: Sequence[int],
    hard_counts: Sequence[int],
    action_view: str,
) -> dict[str, RetrievalMetrics]:
    slices: dict[str, RetrievalMetrics] = {
        f"action_view:{action_view}": _metrics_for(ranks, candidate_counts),
        "easy-1k": _metrics_for(ranks, candidate_counts),
        "hard-1k": _metrics_for(hard_ranks, hard_counts),
    }
    by_source: dict[str, list[int]] = {}
    by_source_counts: dict[str, list[int]] = {}
    by_edit_bucket: dict[str, list[int]] = {}
    by_edit_bucket_counts: dict[str, list[int]] = {}
    for rank, count, query_id in zip(ranks, candidate_counts, query_ids):
        row = rows_by_id[query_id]
        by_source.setdefault(row.source, []).append(rank)
        by_source_counts.setdefault(row.source, []).append(count)
        bucket = f"{(row.edit_size // 10) * 10}-{(row.edit_size // 10) * 10 + 9}"
        by_edit_bucket.setdefault(bucket, []).append(rank)
        by_edit_bucket_counts.setdefault(bucket, []).append(count)
    for source, source_ranks in by_source.items():
        slices[f"source:{source}"] = _metrics_for(source_ranks, by_source_counts[source])
    for bucket, bucket_ranks in by_edit_bucket.items():
        slices[f"edit_size:{bucket}"] = _metrics_for(bucket_ranks, by_edit_bucket_counts[bucket])
    return slices


def _build_action_contrast_metrics(
    action_contrast_report: Any,
    *,
    model_scores_all: np.ndarray,
    no_action_scores_all: np.ndarray,
    row_by_id: Mapping[str, _EvalRow],
    row_index_by_id: Mapping[str, int],
    seed: int,
) -> tuple[dict[str, RetrievalMetrics], dict[str, Any]]:
    slices: dict[str, RetrievalMetrics] = {}
    metadata: dict[str, Any] = {}
    for pool_name in action_contrast_report.config.pool_names:
        score_rows: list[tuple[float, ...]] = []
        no_action_score_rows: list[tuple[float, ...]] = []
        candidate_ids_by_query: list[tuple[str, ...]] = []
        candidate_texts_by_query: list[tuple[str, ...]] = []
        query_texts: list[str] = []
        target_ids: list[str] = []

        for sample in action_contrast_report.samples:
            negative_ids = tuple(sample.pools.get(pool_name, ()))
            if not negative_ids:
                continue
            candidate_ids = (sample.target_id, *negative_ids)
            query_index = row_index_by_id[sample.query_id]
            candidate_indices = tuple(row_index_by_id[candidate_id] for candidate_id in candidate_ids)
            score_rows.append(
                tuple(
                    float(model_scores_all[query_index, candidate_index])
                    for candidate_index in candidate_indices
                )
            )
            no_action_score_rows.append(
                tuple(
                    float(no_action_scores_all[query_index, candidate_index])
                    for candidate_index in candidate_indices
                )
            )
            candidate_ids_by_query.append(candidate_ids)
            candidate_texts_by_query.append(
                tuple(row_by_id[candidate_id].candidate_text for candidate_id in candidate_ids)
            )
            query_texts.append(_query_text(row_by_id[sample.query_id]))
            target_ids.append(sample.target_id)

        if not score_rows:
            continue

        ranks = rank_targets(tuple(score_rows), tuple(candidate_ids_by_query), tuple(target_ids))
        candidate_counts = tuple(len(candidate_ids) for candidate_ids in candidate_ids_by_query)
        metrics = _metrics_for(ranks, candidate_counts)
        baseline_counts = {
            name: candidate_counts
            for name in ("random", "lexical", "no_action", "shuffled_action")
        }
        baseline_metrics = build_baseline_metrics(
            {
                "random": random_baseline_ranks(
                    tuple(candidate_ids_by_query),
                    tuple(target_ids),
                    seed=seed + _stable_seed_offset(f"{pool_name}:random"),
                ),
                "lexical": lexical_baseline_ranks(
                    tuple(query_texts),
                    tuple(candidate_texts_by_query),
                    tuple(candidate_ids_by_query),
                    tuple(target_ids),
                ),
                "no_action": no_action_baseline_ranks(
                    tuple(no_action_score_rows),
                    tuple(candidate_ids_by_query),
                    tuple(target_ids),
                ),
                "shuffled_action": shuffled_action_baseline_ranks(
                    tuple(score_rows),
                    tuple(candidate_ids_by_query),
                    tuple(target_ids),
                    seed=seed + _stable_seed_offset(f"{pool_name}:shuffled"),
                ),
            },
            candidate_counts=baseline_counts,
        )
        slices[f"action_contrast:{pool_name}"] = metrics
        metadata[pool_name] = {
            "metrics": metrics.to_dict(),
            "baselines": {
                name: baseline.to_dict()
                for name, baseline in sorted(baseline_metrics.items())
            },
        }
    return slices, metadata


def _metrics_for(ranks: Sequence[int], candidate_counts: Sequence[int]) -> RetrievalMetrics:
    from .retrieval import compute_retrieval_metrics

    return compute_retrieval_metrics(ranks, candidate_counts=candidate_counts)


def _embed_rows(rows: tuple[_EvalRow, ...], *, model: Any, runtime: Any, device: Any) -> tuple[Any, Any, Any]:
    if not rows:
        raise RetrievalEvalError("retrieval evaluation requires at least one held-out row")
    state_before = _state_batch(tuple(row.state_before for row in rows), runtime=runtime, device=device)
    state_after = _state_batch(tuple(row.state_after for row in rows), runtime=runtime, device=device)
    action = _action_batch(
        tuple(row.action for row in rows),
        runtime=runtime,
        device=device,
        action_view=model.config.action_view,
    )
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


def _state_batch(rows: Sequence[Mapping[str, np.ndarray]], *, runtime: Any, device: Any) -> CodeStateBatch:
    return CodeStateBatch(
        input_ids=_tensor(rows, "input_ids", runtime=runtime, device=device).long(),
        attention_mask=_tensor(rows, "attention_mask", runtime=runtime, device=device).bool(),
        segment_ids=_tensor(rows, "segment_ids", runtime=runtime, device=device).long(),
        changed_hunk_mask=_tensor(rows, "changed_hunk_mask", runtime=runtime, device=device).bool(),
    )


def _action_batch(
    rows: Sequence[Mapping[str, np.ndarray]],
    *,
    runtime: Any,
    device: Any,
    action_view: str,
) -> ActionBatch:
    return ActionBatch(
        input_ids=_tensor(rows, "input_ids", runtime=runtime, device=device).long(),
        attention_mask=_tensor(rows, "attention_mask", runtime=runtime, device=device).bool(),
        action_view=action_view,  # type: ignore[arg-type]
    )


def _tensor(rows: Sequence[Mapping[str, np.ndarray]], key: str, *, runtime: Any, device: Any) -> Any:
    values = np.stack([np.asarray(row[key]) for row in rows], axis=0)
    return runtime.as_tensor(values, device=device)


def _negative_squared_l2(query_vectors: Any, candidate_vectors: Any, *, runtime: Any) -> np.ndarray:
    distances = runtime.cdist(query_vectors, candidate_vectors, p=2).pow(2)
    scores = -distances.detach().cpu().numpy()
    if not np.isfinite(scores).all():
        raise RetrievalEvalError("retrieval score matrix contains NaN or inf")
    return scores


def _select_score_rows(
    scores: np.ndarray,
    query_indices: Sequence[int],
    candidate_indices: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(scores[query_index, candidate_index]) for candidate_index in candidate_indices)
        for query_index in query_indices
    )


def _load_heldout_rows(
    pack_paths: _PackPaths,
    *,
    action_view: str,
    vocab_size: int,
) -> tuple[_EvalRow, ...]:
    rows: list[_EvalRow] = []
    for split in ("val", "test"):
        rows.extend(
            _load_split_rows(
                pack_paths,
                split=split,
                action_view=action_view,
                vocab_size=vocab_size,
            )
        )
    if not rows:
        raise RetrievalEvalError("packed dataset has no held-out val/test rows for retrieval evaluation")
    return tuple(rows)


def _load_split_rows(
    pack_paths: _PackPaths,
    *,
    split: str,
    action_view: str,
    vocab_size: int,
) -> tuple[_EvalRow, ...]:
    h5py = _require_h5py()
    metadata_rows = _read_parquet_rows(pack_paths.parquet_dir / split)
    hdf5_path = pack_paths.hdf5_dir / f"{split}.hdf5"
    if not hdf5_path.is_file():
        raise RetrievalEvalError(f"packed HDF5 split does not exist: {hdf5_path}")
    action_group = "action_text" if action_view == "text" else "action_abs"
    with h5py.File(hdf5_path, "r") as handle:
        schema_version = _hdf5_attr_text(handle.attrs.get("schema_version"))
        if schema_version != DATASET_SCHEMA_VERSION:
            raise RetrievalEvalError(
                f"packed HDF5 schema_version must be {DATASET_SCHEMA_VERSION!r}; got {schema_version!r}"
            )
        row_count = int(handle.attrs.get("row_count", -1))
        if row_count != len(metadata_rows):
            raise RetrievalEvalError(
                f"HDF5 row_count for {split!r} does not match parquet rows: "
                f"{row_count} != {len(metadata_rows)}"
            )
        state_before = _read_state_group(handle, "state_before", vocab_size=vocab_size)
        state_after = _read_state_group(handle, "state_after", vocab_size=vocab_size)
        action = _read_action_group(handle, action_group, vocab_size=vocab_size)

    rows: list[_EvalRow] = []
    for index, metadata in enumerate(metadata_rows):
        observed_split = str(metadata.get("split", split))
        if observed_split != split:
            raise RetrievalEvalError(
                f"parquet split row mismatch for {metadata.get('transition_id')!r}: "
                f"{observed_split!r} != {split!r}"
            )
        transition_id = str(metadata["transition_id"])
        state_before_hash = _metadata_token_hash(metadata, "state_before")
        state_after_hash = _metadata_token_hash(metadata, "state_after")
        rows.append(
            _EvalRow(
                transition_id=transition_id,
                split=split,
                source=str(metadata.get("source", "unknown")),
                repo=str(metadata.get("repo", "")),
                path=str(metadata.get("path", "")),
                edit_size=int(metadata.get("edit_size", 0) or 0),
                state_before={key: value[index] for key, value in state_before.items()},
                state_after={key: value[index] for key, value in state_after.items()},
                action={key: value[index] for key, value in action.items()},
                action_text=_tokens_to_text(
                    metadata.get("action_text_input_ids", ()),
                    metadata.get("action_text_attention_mask", ()),
                ),
                candidate_text=_tokens_to_text(
                    metadata.get("state_after_input_ids", ()),
                    metadata.get("state_after_attention_mask", ()),
                ),
                metadata={
                    "commit": str(metadata.get("commit", "")),
                    "token_count_before": int(metadata.get("token_count_before", 0) or 0),
                    "token_count_after": int(metadata.get("token_count_after", 0) or 0),
                    "action_cluster": _action_cluster(metadata),
                    "action_abs_cluster": _action_cluster(metadata),
                    "state_before_hash": state_before_hash,
                    "state_after_hash": state_after_hash,
                    "state_before_simhash": _metadata_token_simhash(metadata, "state_before"),
                    "edit_size_bucket": f"{(int(metadata.get('edit_size', 0) or 0) // 10) * 10}-"
                    f"{(int(metadata.get('edit_size', 0) or 0) // 10) * 10 + 9}",
                    "diff_shape": _metadata_diff_shape(metadata),
                    "state_before_kind": str(metadata.get("state_before_kind") or ""),
                    "state_after_kind": str(metadata.get("state_after_kind") or ""),
                    "state_before_symbol": str(metadata.get("state_before_symbol") or ""),
                    "state_after_symbol": str(metadata.get("state_after_symbol") or ""),
                    "state_before_fallback_reason": str(
                        metadata.get("state_before_fallback_reason") or ""
                    ),
                    "state_after_fallback_reason": str(
                        metadata.get("state_after_fallback_reason") or ""
                    ),
                },
            )
        )
    return tuple(rows)


def _read_action_discriminative_report(
    pack_paths: _PackPaths,
    dataset_artifact: ArtifactManifest,
) -> dict[str, Any]:
    path_value = dataset_artifact.metadata.get("action_discriminative_shard_report")
    if not isinstance(path_value, str) or not path_value:
        return {
            "available": False,
            "reason": "packed dataset artifact does not expose action_discriminative_shard_report",
        }
    report_path = pack_paths.root / path_value
    if not report_path.is_file():
        return {
            "available": False,
            "reason": f"action-discriminative report missing: {path_value}",
        }
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RetrievalEvalError("action-discriminative shard report must be a JSON object")
    return validate_action_discriminative_shard_report_payload(payload)


def _action_discriminative_claim_ready(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    claim = value.get("claim_readiness")
    if not isinstance(claim, Mapping):
        return False
    return bool(claim.get("positive_action_use_claim_ready"))


def _action_discriminative_hard_negative_pools(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    pools = value.get("hard_negative_pools")
    if not isinstance(pools, Mapping):
        return {}
    return dict(pools)


def _read_parquet_rows(directory: Path) -> tuple[Mapping[str, Any], ...]:
    pq = _require_pyarrow_parquet()
    if not directory.is_dir():
        raise RetrievalEvalError(f"packed parquet split directory does not exist: {directory}")
    rows: list[Mapping[str, Any]] = []
    for path in sorted(directory.glob("*.parquet")):
        table = pq.read_table(path)
        rows.extend(table.to_pylist())
    return tuple(rows)


def _metadata_token_hash(metadata: Mapping[str, Any], prefix: str) -> str:
    return token_sequence_hash(
        _metadata_active_tokens(
            metadata.get(f"{prefix}_input_ids", ()),
            metadata.get(f"{prefix}_attention_mask", ()),
        )
    )


def _metadata_token_simhash(metadata: Mapping[str, Any], prefix: str) -> str:
    return token_sequence_simhash(
        _metadata_active_tokens(
            metadata.get(f"{prefix}_input_ids", ()),
            metadata.get(f"{prefix}_attention_mask", ()),
        )
    )


def _metadata_active_tokens(input_ids: Any, attention_mask: Any) -> tuple[int, ...]:
    ids = [] if input_ids is None else list(input_ids)
    masks = [] if attention_mask is None else list(attention_mask)
    if not masks:
        masks = [token != 0 for token in ids]
    return tuple(int(token) for token, keep in zip(ids, masks) if keep and int(token) != 0)


def _load_torch_checkpoint(checkpoint_path: Path, *, device: Any, runtime: Any) -> tuple[Any, Mapping[str, Any]]:
    try:
        payload = runtime.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility.
        payload = runtime.load(checkpoint_path, map_location=device)
    if not isinstance(payload, Mapping):
        raise RetrievalEvalError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != TORCH_CHECKPOINT_SCHEMA_VERSION:
        raise RetrievalEvalError(
            f"checkpoint schema_version is unsupported: {payload.get('schema_version')!r}"
        )
    compatibility = payload.get("compatibility_config")
    if not isinstance(compatibility, Mapping):
        raise RetrievalEvalError("checkpoint compatibility_config must be a mapping")
    wm = compatibility.get("wm")
    if not isinstance(wm, Mapping):
        raise RetrievalEvalError("checkpoint compatibility_config.wm must be a mapping")
    action_view = str(wm.get("action_view", "text"))
    if action_view not in {"text", "abstract"}:
        raise RetrievalEvalError("patch action is diagnostic only and cannot be a headline retrieval model")
    encoder_type, encoder_layers, encoder_heads = resolve_state_encoder_arch(
        wm, payload.get("model_state_dict")
    )
    enable_ema_target_encoder, ema_target_decay = resolve_ema_target_encoder_config(
        wm, payload.get("model_state_dict")
    )
    enable_output_value_head = resolve_output_value_head_config(
        wm, payload.get("model_state_dict")
    )
    config = TorchCodeTransitionModelConfig(
        action_view=action_view,  # type: ignore[arg-type]
        latent_dim=int(wm.get("embed_dim", 256)),
        state_sequence_length=int(wm.get("state_sequence_length", 1024)),
        action_sequence_length=int(wm.get("action_sequence_length", 256 if action_view == "text" else 192)),
        vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
        dropout=0.0,
        action_fusion=str(wm.get("action_fusion", "conditional_transformer")),
        enable_inverse_action_head=bool(
            wm.get("enable_inverse_action_head")
            or (
                isinstance(compatibility.get("loss"), Mapping)
                and compatibility["loss"].get("enable_inverse_action_reconstruction")
            )
        ),
        enable_output_value_head=enable_output_value_head,
        enable_ema_target_encoder=enable_ema_target_encoder,
        ema_target_decay=ema_target_decay,
        state_encoder_type=encoder_type,
        state_encoder_layers=encoder_layers,
        state_encoder_heads=encoder_heads,
    )
    model = build_torch_transition_model(config)
    try:
        model.load_state_dict(payload["model_state_dict"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RetrievalEvalError(f"checkpoint model state could not be loaded: {exc}") from exc
    model.to(device)
    model.eval()
    return model, payload


def _resolve_pack_paths(value: Path | str) -> _PackPaths:
    raw = Path(value).resolve()
    if raw.is_file():
        if raw.name != "manifest.json":
            raise RetrievalEvalError("--data file input must be a packed artifact manifest.json")
        root = raw.parent
    elif raw.is_dir() and (raw / "manifest.json").is_file():
        root = raw
    elif raw.is_dir() and raw.name == "hdf5" and (raw.parent / "manifest.json").is_file():
        root = raw.parent
    else:
        raise RetrievalEvalError("--data must be a packed dataset directory or manifest.json")
    return _PackPaths(
        root=root,
        artifact_manifest_path=root / "manifest.json",
        hdf5_dir=root / "hdf5",
        parquet_dir=root / "parquet",
    )


def _read_verified_artifact_manifest(path: Path, *, root: Path) -> ArtifactManifest:
    manifest = read_artifact_manifest(path)
    validate_artifact_checksums(manifest, root=root)
    return manifest


def _infer_training_artifact_manifest_path(checkpoint: Path) -> Path:
    try:
        run_dir = checkpoint.parent.parent
    except IndexError as exc:  # pragma: no cover - Path.parent is total, kept defensive.
        raise ArtifactManifestError("checkpoint path does not have an inferable run directory") from exc
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactManifestError(
            "retrieval evaluation requires a training-run artifact manifest at "
            f"{manifest_path}"
        )
    return manifest_path


def _reject_existing_retrieval_outputs(out_dir: Path, *, overwrite: bool) -> None:
    for path in (
        out_dir / "config.json",
        out_dir / "reports" / "retrieval_report.json",
        out_dir / "reports" / "hard_negative_sampler_report.json",
        out_dir / "reports" / "action_contrast_pool_report.json",
        out_dir / "manifest.json",
    ):
        if path.exists() and not overwrite:
            raise RetrievalEvalError(f"output already exists; pass --overwrite to replace: {path}")


def _read_state_group(handle: Any, group_name: str, *, vocab_size: int) -> dict[str, np.ndarray]:
    return {
        "input_ids": _read_token_matrix(handle, f"{group_name}/input_ids", vocab_size=vocab_size),
        "attention_mask": _read_bool_matrix(handle, f"{group_name}/attention_mask"),
        "segment_ids": _read_int_matrix(handle, f"{group_name}/segment_ids"),
        "changed_hunk_mask": _read_bool_matrix(handle, f"{group_name}/changed_hunk_mask"),
    }


def _read_action_group(handle: Any, group_name: str, *, vocab_size: int) -> dict[str, np.ndarray]:
    return {
        "input_ids": _read_token_matrix(handle, f"{group_name}/input_ids", vocab_size=vocab_size),
        "attention_mask": _read_bool_matrix(handle, f"{group_name}/attention_mask"),
    }


def _read_token_matrix(handle: Any, key: str, *, vocab_size: int) -> np.ndarray:
    values = _read_int_matrix(handle, key)
    return np.where(values > 0, ((values - 1) % (vocab_size - 1)) + 1, 0).astype(np.int64)


def _read_int_matrix(handle: Any, key: str) -> np.ndarray:
    if key not in handle:
        raise RetrievalEvalError(f"packed HDF5 is missing dataset {key!r}")
    values = np.asarray(handle[key], dtype=np.int64)
    if values.ndim != 2:
        raise RetrievalEvalError(f"packed HDF5 dataset {key!r} must be rank 2")
    if values.size and values.min() < 0:
        raise RetrievalEvalError(f"packed HDF5 dataset {key!r} contains negative ids")
    return values


def _read_bool_matrix(handle: Any, key: str) -> np.ndarray:
    if key not in handle:
        raise RetrievalEvalError(f"packed HDF5 is missing dataset {key!r}")
    values = np.asarray(handle[key], dtype=bool)
    if values.ndim != 2:
        raise RetrievalEvalError(f"packed HDF5 dataset {key!r} must be rank 2")
    return values


def _resolve_device(device: str, runtime: Any) -> Any:
    if device not in {"cpu", "cuda", "mps", "auto"}:
        raise RetrievalEvalError("device must be cpu, cuda, mps, or auto")
    if device == "auto":
        if runtime.cuda.is_available():
            return runtime.device("cuda")
        if hasattr(runtime.backends, "mps") and runtime.backends.mps.is_available():
            return runtime.device("mps")
        return runtime.device("cpu")
    if device == "cuda" and not runtime.cuda.is_available():
        raise RetrievalEvalError("CUDA device requested but torch.cuda is unavailable")
    if device == "mps" and not (
        hasattr(runtime.backends, "mps") and runtime.backends.mps.is_available()
    ):
        raise RetrievalEvalError("MPS device requested but torch.backends.mps is unavailable")
    return runtime.device(device)


def _require_torch_runtime() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise OptionalDependencyError(
            "retrieval evaluation requires torch; install with `uv sync --group train --group data --group dev`"
        )
    if importlib.util.find_spec("einops") is None:
        raise OptionalDependencyError(
            "retrieval evaluation requires einops; install with `uv sync --group train --group data --group dev`"
        )
    import torch

    return torch


def _require_h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "retrieval evaluation requires h5py; install with `uv sync --group data --group dev`"
        ) from exc
    return h5py


def _require_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "retrieval evaluation requires pyarrow; install with `uv sync --group data --group dev`"
        ) from exc
    return pq


def _tokens_to_text(input_ids: Any, attention_mask: Any) -> str:
    ids = [] if input_ids is None else list(input_ids)
    masks = [] if attention_mask is None else list(attention_mask)
    if not masks:
        masks = [token != 0 for token in ids]
    return " ".join(str(token) for token, keep in zip(ids, masks) if keep and int(token) != 0)


def _query_text(row: _EvalRow) -> str:
    before_tokens = _tokens_to_text(row.state_before["input_ids"], row.state_before["attention_mask"])
    return f"{before_tokens} {row.action_text}".strip()


def _action_cluster(metadata: Mapping[str, Any]) -> str:
    ids = metadata.get("action_abs_input_ids") or metadata.get("action_text_input_ids") or ()
    masks = metadata.get("action_abs_attention_mask") or metadata.get("action_text_attention_mask") or ()
    text = _tokens_to_text(ids, masks)
    if not text:
        return "empty-action"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:12]


def _metadata_diff_shape(metadata: Mapping[str, Any]) -> str:
    for value in metadata.get("dedup_keys", ()) or ():
        text = str(value)
        if text.startswith("diff_shape:"):
            return text.split(":", 1)[1]
    edit_size = int(metadata.get("edit_size", 0) or 0)
    return f"edit_size:{(edit_size // 10) * 10}-{(edit_size // 10) * 10 + 9}"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _stable_seed_offset(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _hdf5_attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return "" if value is None else str(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RetrievalEvalError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RetrievalEvalError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise RetrievalEvalError(f"{name} must be a positive integer")
    return result


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RetrievalEvalError(f"{name} must be an integer")
    result = int(value)
    if result != value or result < 0:
        raise RetrievalEvalError(f"{name} must be a non-negative integer")
    return result


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
