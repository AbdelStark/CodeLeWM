"""Manifest-backed v0.6 execution-pack evaluation runners.

These runners are the JSONL execution-pack counterparts to the older
HDF5-backed eval commands. They consume the v0.6 pack shape directly:
``code_tokens`` are the before state, ``input_tokens`` are the action,
and ``output_tokens`` are the target state.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.data.execution_pack import EXECUTION_PACK_MANIFEST_SCHEMA_VERSION
from codelewm.model import (
    ActionBatch,
    CodeStateBatch,
    TorchCodeTransitionModelConfig,
    build_torch_transition_model,
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
from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE

from .crash_prediction import (
    CRASH_PREDICTION_REPORT_SCHEMA_VERSION,
    CrashPredictionError,
    CrashPredictionReport,
    CrashSample,
    evaluate_crash_prediction,
)
from .execution_probe_targets import EXECUTION_PROBE_TARGETS, extract_labels
from .execution_surprise_decoys import (
    DecoyGenerationReport,
    DecoyPair,
    generate_same_code_different_input_pairs,
    generate_same_problem_different_submission_pairs,
)
from .semantic_decoy_pack import LoadedSemanticDecoyPack, load_semantic_decoy_pack
from .latent_probe import (
    LATENT_PROBE_VIEWS,
    LatentProbeConfig,
    LatentProbeError,
    LatentProbeRow,
    build_latent_probe_report,
    write_latent_probe_report,
)
from .retrieval import (
    CandidatePool,
    CandidatePoolEntry,
    RetrievalEvalError,
    RetrievalMetrics,
    build_baseline_metrics,
    build_retrieval_report,
    lexical_baseline_ranks,
    no_action_baseline_ranks,
    random_baseline_ranks,
    rank_targets,
    shuffled_action_baseline_ranks,
    write_retrieval_report,
)
from .retrieval_runner import (
    _display_path,
    _infer_training_artifact_manifest_path,
    _optional_int,
    _require_torch_runtime,
    _resolve_device,
    _write_json,
)
from .surprise import (
    ALLOWED_SURPRISE_DECOY_CATEGORIES,
    SurpriseEvalError,
    SurpriseExampleResult,
    build_surprise_report,
    write_surprise_report,
)


EXECUTION_RETRIEVAL_EVAL_RUN_SCHEMA_VERSION = "codelewm.eval.execution_retrieval_run.v1"
EXECUTION_SURPRISE_EVAL_RUN_SCHEMA_VERSION = "codelewm.eval.execution_surprise_run.v1"
EXECUTION_PROBE_EVAL_RUN_SCHEMA_VERSION = "codelewm.eval.execution_probe_run.v1"
CRASH_PREDICTION_EVAL_RUN_SCHEMA_VERSION = "codelewm.eval.crash_prediction_run.v1"
EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION = "codelewm.execution_train_checkpoint.v1"


class ExecutionEvalError(ValueError):
    """Raised when a v0.6 execution-pack eval cannot run."""


@dataclass(frozen=True)
class ExecutionEvalResult:
    """CLI-facing summary for one execution-pack eval artifact."""

    schema_version: str
    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    parent_artifacts: tuple[str, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "parent_artifacts": list(self.parent_artifacts),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _ExecutionPackPaths:
    root: Path
    pack_jsonl_path: Path
    execution_manifest_path: Path
    artifact_manifest_path: Path


@dataclass(frozen=True)
class _ExecutionRow:
    record_id: str
    split: str
    source_dataset: str
    source_problem_id: str
    source_submission_id: str
    input_id: str
    state_before: Mapping[str, np.ndarray]
    action: Mapping[str, np.ndarray]
    state_after: Mapping[str, np.ndarray]
    code_text: str
    input_text: str
    output_text: str
    record: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def candidate_entry(self) -> CandidatePoolEntry:
        return CandidatePoolEntry(
            transition_id=self.record_id,
            split=self.split,
            source=self.source_dataset,
            repo=self.source_problem_id,
            path=self.source_submission_id,
            edit_size=abs(
                _active_count(self.state_after) - _active_count(self.state_before)
            ),
            metadata=dict(self.metadata),
        )


def run_execution_retrieval_evaluation(
    *,
    checkpoint: Path | str,
    pack: Path | str,
    out: Path | str,
    baselines: Sequence[str] = ("random", "no_action", "shuffled_action"),
    device: str = "cpu",
    max_candidates: int = 1000,
    seed: int = 0,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "execution-retrieval"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> ExecutionEvalResult:
    """Run retrieval over v0.6 execution-pack JSONL records."""

    _positive_int(max_candidates, "max_candidates")
    selected_baselines = _parse_baselines(baselines)
    context = _prepare_context(checkpoint=checkpoint, pack=pack, device=device)
    out_dir = Path(out).resolve()
    _reject_existing(out_dir, ("config.json", "reports/retrieval_report.json", "manifest.json"), overwrite=overwrite)
    rows = _select_rows(context.rows, splits=("val", "test"), max_rows=max_candidates, seed=seed)
    if len(rows) < 1:
        raise RetrievalEvalError("execution retrieval requires at least one val/test row")

    z_before, z_pred_after, z_after = _embed_rows(
        rows,
        model=context.model,
        runtime=context.runtime,
        device=context.device,
    )
    scores = _negative_squared_l2(z_pred_after, z_after, runtime=context.runtime)
    no_action_scores = _negative_squared_l2(z_before, z_after, runtime=context.runtime)
    row_index = {row.record_id: index for index, row in enumerate(rows)}
    query_ids = tuple(row.record_id for row in rows)
    candidate_ids = query_ids
    candidate_indices = tuple(row_index[candidate_id] for candidate_id in candidate_ids)
    score_rows = _select_score_rows(
        scores,
        query_indices=tuple(row_index[query_id] for query_id in query_ids),
        candidate_indices=candidate_indices,
    )
    candidate_ids_by_query = tuple(candidate_ids for _ in query_ids)
    candidate_counts = tuple(len(candidate_ids) for _ in query_ids)
    ranks = rank_targets(score_rows, candidate_ids_by_query, query_ids)
    baseline_inputs: dict[str, Any] = {}
    if "random" in selected_baselines:
        baseline_inputs["random"] = random_baseline_ranks(
            candidate_ids_by_query, query_ids, seed=seed
        )
    if "lexical" in selected_baselines:
        baseline_inputs["lexical"] = lexical_baseline_ranks(
            tuple(f"{row.code_text} {row.input_text}" for row in rows),
            tuple(tuple(candidate.output_text for candidate in rows) for _ in rows),
            candidate_ids_by_query,
            query_ids,
        )
    if "no_action" in selected_baselines:
        no_action_score_rows = _select_score_rows(
            no_action_scores,
            query_indices=tuple(row_index[query_id] for query_id in query_ids),
            candidate_indices=candidate_indices,
        )
        baseline_inputs["no_action"] = no_action_baseline_ranks(
            no_action_score_rows, candidate_ids_by_query, query_ids
        )
    if "shuffled_action" in selected_baselines:
        baseline_inputs["shuffled_action"] = shuffled_action_baseline_ranks(
            score_rows, candidate_ids_by_query, query_ids, seed=seed + 17
        )
    baseline_metrics = build_baseline_metrics(
        baseline_inputs,
        candidate_counts={name: candidate_counts for name in baseline_inputs},
    )
    pool = CandidatePool(
        name=f"execution-jsonl-{len(candidate_ids)}",
        entries=tuple(row.candidate_entry() for row in rows),
        seed=seed,
        max_size=max_candidates,
        excluded_splits=("train",),
        metadata={
            "substrate": "execution_trace_v1",
            "pack_artifact_id": context.pack_artifact.artifact_id,
        },
    )
    report = build_retrieval_report(
        ranks,
        candidate_pool=pool,
        candidate_counts=candidate_counts,
        baselines=baseline_metrics,
        slices={
            "execution_jsonl": ranks,
            **_source_slices(rows, ranks, candidate_counts),
        },
        metadata=_base_report_metadata(context, checkpoint_path=context.checkpoint_path)
        | {
            "score": {
                "direction": "larger_is_better",
                "value": "negative_squared_l2",
            },
            "baseline_policy": {"requested": list(selected_baselines)},
        },
    )

    config_payload = {
        "schema_version": EXECUTION_RETRIEVAL_EVAL_RUN_SCHEMA_VERSION,
        "checkpoint": _display_path(context.checkpoint_path),
        "pack": _display_path(context.pack_paths.root),
        "out": _display_path(out_dir),
        "device": str(context.device),
        "max_candidates": max_candidates,
        "baselines": list(selected_baselines),
        "seed": seed,
    }
    config_path = out_dir / "config.json"
    report_path = out_dir / "reports" / "retrieval_report.json"
    _write_json(config_payload, config_path)
    write_retrieval_report(report, report_path)
    return _write_eval_artifact(
        out_dir=out_dir,
        files=(config_path, report_path),
        command=command,
        config=config_payload,
        context=context,
        result_schema_version=EXECUTION_RETRIEVAL_EVAL_RUN_SCHEMA_VERSION,
        report_path="reports/retrieval_report.json",
        report_schema_version=report.schema_version,
        metadata={
            "query_count": report.metrics.query_count,
            "metrics": report.metrics.to_dict(),
            "baselines": {
                name: metrics.to_dict()
                for name, metrics in sorted(report.baselines.items())
            },
        },
        source_git_sha=source_git_sha,
        created_at=created_at,
    )


def run_execution_surprise_evaluation(
    *,
    checkpoint: Path | str,
    pack: Path | str,
    out: Path | str,
    decoys: Sequence[str] = (
        "mutation",
        "same_problem_different_submission",
        "same_code_different_input",
    ),
    device: str = "cpu",
    max_examples: int = 1000,
    seed: int = 0,
    semantic_decoy_manifest: Path | str | None = None,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "execution-surprise"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> ExecutionEvalResult:
    """Run surprise AUC over execution-specific decoy categories."""

    _positive_int(max_examples, "max_examples")
    selected_decoys = _parse_decoys(decoys)
    context = _prepare_context(checkpoint=checkpoint, pack=pack, device=device)
    out_dir = Path(out).resolve()
    _reject_existing(out_dir, ("config.json", "reports/surprise_report.json", "manifest.json"), overwrite=overwrite)
    rows = _select_rows(context.rows, splits=("val", "test"), max_rows=max_examples, seed=seed)
    if not rows:
        raise SurpriseEvalError("execution surprise requires at least one val/test row")

    z_before, z_pred_after, z_after = _embed_rows(
        rows,
        model=context.model,
        runtime=context.runtime,
        device=context.device,
    )
    row_index = {row.record_id: index for index, row in enumerate(rows)}
    semantic_decoy_pack = (
        load_semantic_decoy_pack(semantic_decoy_manifest)
        if semantic_decoy_manifest is not None
        else None
    )
    decoy_pairs, decoy_reports = _build_execution_decoy_pairs(
        [row.record for row in rows],
        selected_decoys=selected_decoys,
        seed=seed,
        semantic_decoy_pack=semantic_decoy_pack,
    )
    pairs_by_query: dict[str, list[DecoyPair]] = {}
    for pair in decoy_pairs:
        pairs_by_query.setdefault(pair.query_record_id, []).append(pair)

    results: list[SurpriseExampleResult] = []
    mutation_count = 0
    for query_index, row in enumerate(rows):
        true_score = _energy(
            z_pred_after[query_index],
            z_after[query_index],
            runtime=context.runtime,
        )
        decoy_scores: dict[str, list[float]] = {
            category: [] for category in ALLOWED_SURPRISE_DECOY_CATEGORIES
        }
        if "mutation" in selected_decoys:
            mutation_state = _mutated_state_after(row.state_after, seed=seed)
            mutation_z = _embed_state(
                mutation_state,
                model=context.model,
                runtime=context.runtime,
                device=context.device,
            )
            decoy_scores["mutation"].append(
                _energy(z_pred_after[query_index], mutation_z[0], runtime=context.runtime)
            )
            mutation_count += 1
        for pair in pairs_by_query.get(row.record_id, ()):
            decoy_index = row_index.get(pair.decoy_record_id)
            if decoy_index is None:
                continue
            decoy_scores[pair.category].append(
                _energy(
                    z_pred_after[query_index],
                    z_after[decoy_index],
                    runtime=context.runtime,
                )
            )
        flat_scores = [
            true_score,
            *(score for scores in decoy_scores.values() for score in scores),
        ]
        if len(flat_scores) == 1:
            continue
        results.append(
            SurpriseExampleResult(
                transition_id=row.record_id,
                true_score=true_score,
                decoy_scores_by_category={
                    category: tuple(scores)
                    for category, scores in decoy_scores.items()
                    if scores
                },
                true_rank=_rank_lower_is_better(flat_scores, true_index=0),
                candidate_count=len(flat_scores),
            )
        )
    if not results:
        raise SurpriseEvalError("execution surprise generated zero decoys")

    decoy_report_payload = {
        "schema_version": "codelewm.eval.execution_surprise_decoy_summary.v1",
        "mutation_pair_count": mutation_count,
        "reports": [report.as_dict() for report in decoy_reports],
    }
    if semantic_decoy_pack is not None:
        decoy_report_payload["semantic_decoy_pack"] = semantic_decoy_pack.metadata()
    report = build_surprise_report(
        results,
        decoy_seed=seed,
        score_direction="lower_is_better",
        metadata=_base_report_metadata(context, checkpoint_path=context.checkpoint_path)
        | {
            "decoy_policy": {"requested": list(selected_decoys)},
            "execution_decoy_generation": decoy_report_payload,
        },
    )

    config_payload = {
        "schema_version": EXECUTION_SURPRISE_EVAL_RUN_SCHEMA_VERSION,
        "checkpoint": _display_path(context.checkpoint_path),
        "pack": _display_path(context.pack_paths.root),
        "out": _display_path(out_dir),
        "device": str(context.device),
        "max_examples": max_examples,
        "decoys": list(selected_decoys),
        "seed": seed,
        "score_direction": "lower_is_better",
        "semantic_decoy_manifest": None
        if semantic_decoy_manifest is None
        else str(semantic_decoy_manifest),
    }
    config_path = out_dir / "config.json"
    report_path = out_dir / "reports" / "surprise_report.json"
    decoy_report_path = out_dir / "reports" / "execution_decoy_report.json"
    _write_json(config_payload, config_path)
    write_surprise_report(report, report_path)
    _write_json(decoy_report_payload, decoy_report_path)
    return _write_eval_artifact(
        out_dir=out_dir,
        files=(config_path, report_path, decoy_report_path),
        command=command,
        config=config_payload,
        context=context,
        result_schema_version=EXECUTION_SURPRISE_EVAL_RUN_SCHEMA_VERSION,
        report_path="reports/surprise_report.json",
        report_schema_version=report.schema_version,
        metadata={
            "example_count": report.metrics.example_count,
            "metrics": report.metrics.to_dict(),
            "execution_decoy_report_path": "reports/execution_decoy_report.json",
            "semantic_decoy_pack_artifact_id": None
            if semantic_decoy_pack is None
            else semantic_decoy_pack.artifact_manifest.artifact_id,
        },
        extra_parent_artifacts=()
        if semantic_decoy_pack is None
        else (semantic_decoy_pack.artifact_manifest.artifact_id,),
        source_git_sha=source_git_sha,
        created_at=created_at,
    )


def run_execution_probe_evaluation(
    *,
    checkpoint: Path | str,
    pack: Path | str,
    out: Path | str,
    targets: Sequence[str] = (
        "output_type",
        "will_raise",
        "output_magnitude_bucket",
        "output_length_bucket",
    ),
    device: str = "cpu",
    max_examples_per_split: int = 1000,
    bootstrap_samples: int = 200,
    seed: int = 0,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "execution-probe"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> ExecutionEvalResult:
    """Run frozen-latent probes using execution-specific labels."""

    _positive_int(max_examples_per_split, "max_examples_per_split")
    _non_negative_int(bootstrap_samples, "bootstrap_samples")
    selected_targets = _parse_targets(targets)
    context = _prepare_context(checkpoint=checkpoint, pack=pack, device=device)
    out_dir = Path(out).resolve()
    _reject_existing(out_dir, ("config.json", "reports/latent_probe_report.json", "manifest.json"), overwrite=overwrite)
    rows = _select_rows_per_split(
        context.rows,
        max_examples_per_split=max_examples_per_split,
        seed=seed,
    )
    if not rows:
        raise LatentProbeError("execution probe requires at least one packed row")
    z_before, z_pred_after, z_after = _embed_rows(
        rows,
        model=context.model,
        runtime=context.runtime,
        device=context.device,
    )
    shuffled = _embed_shuffled_predictions(
        rows,
        z_before=z_before,
        model=context.model,
        runtime=context.runtime,
        device=context.device,
        seed=seed,
    )
    probe_rows = _execution_probe_rows(rows, selected_targets=selected_targets)
    z_pred_np = _to_numpy(z_pred_after)
    report = build_latent_probe_report(
        probe_rows,
        embeddings={
            "z_before": _to_numpy(z_before),
            "z_after": _to_numpy(z_after),
            "z_pred_after": z_pred_np,
        },
        baselines={
            "random_latent": _random_latent_like(z_pred_np, seed=seed),
            "no_action": _to_numpy(z_before),
            "shuffled_action": _to_numpy(shuffled),
        },
        config=LatentProbeConfig(
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            targets=tuple(selected_targets),
            views=LATENT_PROBE_VIEWS,
        ),
        metadata=_base_report_metadata(context, checkpoint_path=context.checkpoint_path)
        | {"probe_target_schema_version": "codelewm.eval.execution_probe_target.v1"},
    )

    config_payload = {
        "schema_version": EXECUTION_PROBE_EVAL_RUN_SCHEMA_VERSION,
        "checkpoint": _display_path(context.checkpoint_path),
        "pack": _display_path(context.pack_paths.root),
        "out": _display_path(out_dir),
        "device": str(context.device),
        "max_examples_per_split": max_examples_per_split,
        "bootstrap_samples": bootstrap_samples,
        "targets": list(selected_targets),
        "seed": seed,
    }
    config_path = out_dir / "config.json"
    report_path = out_dir / "reports" / "latent_probe_report.json"
    _write_json(config_payload, config_path)
    write_latent_probe_report(report, report_path)
    return _write_eval_artifact(
        out_dir=out_dir,
        files=(config_path, report_path),
        command=command,
        config=config_payload,
        context=context,
        result_schema_version=EXECUTION_PROBE_EVAL_RUN_SCHEMA_VERSION,
        report_path="reports/latent_probe_report.json",
        report_schema_version=report.schema_version,
        metadata={
            "row_count": report.row_count,
            "split_counts": dict(report.split_counts),
            "claim_boundary": dict(report.claim_boundary),
            "targets": list(selected_targets),
        },
        source_git_sha=source_git_sha,
        created_at=created_at,
    )


def run_crash_prediction_evaluation(
    *,
    checkpoint: Path | str,
    pack: Path | str,
    out: Path | str,
    device: str = "cpu",
    max_examples: int = 1000,
    seed: int = 0,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "crash-prediction"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> ExecutionEvalResult:
    """Run the scoped v0.6 crash-prediction fallback gate."""

    _positive_int(max_examples, "max_examples")
    context = _prepare_context(checkpoint=checkpoint, pack=pack, device=device)
    out_dir = Path(out).resolve()
    _reject_existing(out_dir, ("config.json", "reports/crash_prediction_report.json", "manifest.json"), overwrite=overwrite)
    rows = _select_rows(context.rows, splits=("val", "test"), max_rows=max_examples, seed=seed)
    if not rows:
        raise CrashPredictionError("crash prediction requires at least one val/test row")
    z_before, z_pred_after, _ = _embed_rows(
        rows,
        model=context.model,
        runtime=context.runtime,
        device=context.device,
    )
    samples = _crash_samples(
        rows,
        z_before=_to_numpy(z_before),
        z_pred_after=_to_numpy(z_pred_after),
        seed=seed,
    )
    positives = sum(1 for sample in samples if sample.will_raise)
    negatives = len(samples) - positives
    if positives and negatives:
        report = evaluate_crash_prediction(samples)
    else:
        report = _not_evaluable_crash_report(
            sample_count=len(samples),
            positives=positives,
            negatives=negatives,
        )

    config_payload = {
        "schema_version": CRASH_PREDICTION_EVAL_RUN_SCHEMA_VERSION,
        "checkpoint": _display_path(context.checkpoint_path),
        "pack": _display_path(context.pack_paths.root),
        "out": _display_path(out_dir),
        "device": str(context.device),
        "max_examples": max_examples,
        "seed": seed,
    }
    config_path = out_dir / "config.json"
    report_path = out_dir / "reports" / "crash_prediction_report.json"
    _write_json(config_payload, config_path)
    _write_json(report.as_dict(), report_path)
    return _write_eval_artifact(
        out_dir=out_dir,
        files=(config_path, report_path),
        command=command,
        config=config_payload,
        context=context,
        result_schema_version=CRASH_PREDICTION_EVAL_RUN_SCHEMA_VERSION,
        report_path="reports/crash_prediction_report.json",
        report_schema_version=report.schema_version,
        metadata={
            "sample_count": report.sample_count,
            "claim_allowed": report.claim_allowed,
            "claim_reason": report.claim_reason,
            "positives": positives,
            "negatives": negatives,
        },
        source_git_sha=source_git_sha,
        created_at=created_at,
    )


@dataclass(frozen=True)
class _ExecutionContext:
    checkpoint_path: Path
    pack_paths: _ExecutionPackPaths
    pack_artifact: ArtifactManifest
    training_artifact: ArtifactManifest
    checkpoint_payload: Mapping[str, Any]
    checkpoint_sha256: str
    model: Any
    runtime: Any
    device: Any
    rows: tuple[_ExecutionRow, ...]


def _prepare_context(*, checkpoint: Path | str, pack: Path | str, device: str) -> _ExecutionContext:
    checkpoint_path = Path(checkpoint).resolve()
    pack_paths = _resolve_execution_pack_paths(pack)
    pack_artifact = _read_pack_artifact(pack_paths)
    training_artifact_path = _infer_training_artifact_manifest_path(checkpoint_path)
    training_artifact = read_artifact_manifest(training_artifact_path)
    if training_artifact.artifact_kind != "training_run":
        raise ArtifactManifestError("checkpoint parent manifest must be a training_run artifact")
    validate_artifact_checksums(training_artifact, root=training_artifact_path.parent)
    checkpoint_manifest = require_trusted_checkpoint(checkpoint_path)
    runtime = _require_torch_runtime()
    selected_device = _resolve_device(device, runtime)
    model, checkpoint_payload = _load_execution_torch_checkpoint(
        checkpoint_path,
        device=selected_device,
        runtime=runtime,
    )
    action_view = str(checkpoint_manifest.metadata.action_view)
    if action_view != str(model.config.action_view):
        raise ExecutionEvalError(
            "checkpoint manifest action_view does not match checkpoint payload: "
            f"{action_view!r} != {model.config.action_view!r}"
        )
    rows = _load_execution_rows(
        pack_paths,
        state_sequence_length=int(model.config.state_sequence_length),
        action_sequence_length=int(model.config.action_sequence_length),
        output_sequence_length=int(model.config.state_sequence_length),
        vocab_size=int(model.config.vocab_size or DEFAULT_TRAINING_VOCAB_SIZE),
    )
    return _ExecutionContext(
        checkpoint_path=checkpoint_path,
        pack_paths=pack_paths,
        pack_artifact=pack_artifact,
        training_artifact=training_artifact,
        checkpoint_payload=checkpoint_payload,
        checkpoint_sha256=checkpoint_manifest.checkpoint_sha256,
        model=model,
        runtime=runtime,
        device=selected_device,
        rows=rows,
    )


def _resolve_execution_pack_paths(value: Path | str) -> _ExecutionPackPaths:
    raw = Path(value).resolve()
    if raw.is_file() and raw.name == "pack.jsonl":
        root = raw.parent
    elif raw.is_file() and raw.name == "manifest.json":
        root = raw.parent
    elif raw.is_file() and raw.name == "artifact_manifest.json":
        root = raw.parent
    elif raw.is_dir():
        root = raw
    else:
        raise ExecutionEvalError("--pack must be an execution pack directory, pack.jsonl, manifest.json, or artifact_manifest.json")
    paths = _ExecutionPackPaths(
        root=root,
        pack_jsonl_path=root / "pack.jsonl",
        execution_manifest_path=root / "manifest.json",
        artifact_manifest_path=root / "artifact_manifest.json",
    )
    if not paths.pack_jsonl_path.is_file():
        raise ExecutionEvalError(f"execution pack JSONL not found: {paths.pack_jsonl_path}")
    if not paths.execution_manifest_path.is_file():
        raise ExecutionEvalError(f"execution pack manifest not found: {paths.execution_manifest_path}")
    if not paths.artifact_manifest_path.is_file():
        raise ExecutionEvalError(f"execution pack artifact manifest not found: {paths.artifact_manifest_path}")
    return paths


def _read_pack_artifact(paths: _ExecutionPackPaths) -> ArtifactManifest:
    payload = json.loads(paths.execution_manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXECUTION_PACK_MANIFEST_SCHEMA_VERSION:
        raise ArtifactManifestError(
            "execution pack manifest schema_version is unsupported: "
            f"{payload.get('schema_version')!r}"
        )
    artifact = read_artifact_manifest(paths.artifact_manifest_path)
    if artifact.artifact_kind != "dataset":
        raise ArtifactManifestError("execution pack artifact manifest must have artifact_kind='dataset'")
    validate_artifact_checksums(artifact, root=paths.root)
    pack_id = payload.get("pack_id")
    if isinstance(pack_id, str) and artifact.artifact_id != pack_id:
        raise ArtifactManifestError(
            f"execution pack artifact_id mismatch: {artifact.artifact_id!r} != {pack_id!r}"
        )
    return artifact


def _load_execution_rows(
    paths: _ExecutionPackPaths,
    *,
    state_sequence_length: int,
    action_sequence_length: int,
    output_sequence_length: int,
    vocab_size: int,
) -> tuple[_ExecutionRow, ...]:
    rows: list[_ExecutionRow] = []
    with paths.pack_jsonl_path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ExecutionEvalError(
                    f"{paths.pack_jsonl_path}:{line_no}: invalid JSON"
                ) from exc
            record_id = str(record.get("record_id") or "")
            if not record_id:
                raise ExecutionEvalError(f"{paths.pack_jsonl_path}:{line_no}: missing record_id")
            code_tokens = _read_integer_tokens(record, "code_tokens", paths, line_no)
            input_tokens = _read_integer_tokens(record, "input_tokens", paths, line_no)
            output_tokens = _read_integer_tokens(record, "output_tokens", paths, line_no)
            state_before = _state_from_tokens(
                code_tokens,
                length=state_sequence_length,
                vocab_size=vocab_size,
            )
            action = _action_from_tokens(
                input_tokens,
                length=action_sequence_length,
                vocab_size=vocab_size,
            )
            state_after = _state_from_tokens(
                output_tokens,
                length=output_sequence_length,
                vocab_size=vocab_size,
            )
            metadata = {
                "source_problem_id": str(record.get("source_problem_id") or ""),
                "source_submission_id": str(record.get("source_submission_id") or ""),
                "input_id": str(record.get("input_id") or ""),
                "output_type": str(record.get("output_type") or ""),
                "output_kind": str(record.get("output_kind") or ""),
                "execution_status": str(record.get("execution_status") or ""),
                "held_out_for_eval": bool(record.get("held_out_for_eval")),
                "license": str(record.get("license") or ""),
                "output_repr_checksum": str(record.get("output_repr_checksum") or ""),
                "input_repr_checksum": str(record.get("input_repr_checksum") or ""),
            }
            rows.append(
                _ExecutionRow(
                    record_id=record_id,
                    split=str(record.get("split") or "train"),
                    source_dataset=str(record.get("source_dataset") or "unknown"),
                    source_problem_id=str(record.get("source_problem_id") or ""),
                    source_submission_id=str(record.get("source_submission_id") or ""),
                    input_id=str(record.get("input_id") or ""),
                    state_before=state_before,
                    action=action,
                    state_after=state_after,
                    code_text=_tokens_to_text(code_tokens),
                    input_text=_tokens_to_text(input_tokens),
                    output_text=_tokens_to_text(output_tokens),
                    record=record,
                    metadata=metadata,
                )
            )
    if not rows:
        raise ExecutionEvalError(f"execution pack has no records: {paths.pack_jsonl_path}")
    return tuple(rows)


def _load_execution_torch_checkpoint(
    checkpoint_path: Path, *, device: Any, runtime: Any
) -> tuple[Any, Mapping[str, Any]]:
    try:
        payload = runtime.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility.
        payload = runtime.load(checkpoint_path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ExecutionEvalError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION:
        raise ExecutionEvalError(
            "checkpoint schema_version is unsupported for execution eval: "
            f"{payload.get('schema_version')!r}"
        )
    compatibility = payload.get("compatibility_config")
    if not isinstance(compatibility, Mapping):
        raise ExecutionEvalError("checkpoint compatibility_config must be a mapping")
    wm = compatibility.get("wm")
    if not isinstance(wm, Mapping):
        raise ExecutionEvalError("checkpoint compatibility_config.wm must be a mapping")
    action_view = str(wm.get("action_view", "text"))
    if action_view not in {"text", "abstract"}:
        raise ExecutionEvalError(
            "patch action is diagnostic only and cannot be an execution eval model"
        )
    objective = compatibility.get("objective")
    try:
        inverse_weight = (
            float(objective.get("inverse_action_reconstruction_weight", 0.0))
            if isinstance(objective, Mapping)
            else 0.0
        )
    except (TypeError, ValueError) as exc:
        raise ExecutionEvalError(
            "checkpoint objective.inverse_action_reconstruction_weight must be numeric"
        ) from exc
    config = TorchCodeTransitionModelConfig(
        action_view=action_view,  # type: ignore[arg-type]
        latent_dim=int(wm.get("embed_dim", 256)),
        state_sequence_length=int(wm.get("state_sequence_length", 1024)),
        action_sequence_length=int(
            wm.get("action_sequence_length", 256 if action_view == "text" else 192)
        ),
        vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
        dropout=0.0,
        action_fusion=str(wm.get("action_fusion", "conditional_transformer")),
        enable_inverse_action_head=inverse_weight > 0.0,
    )
    model = build_torch_transition_model(config)
    try:
        model.load_state_dict(payload["model_state_dict"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ExecutionEvalError(
            f"checkpoint model state could not be loaded: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model, payload


def _read_integer_tokens(
    record: Mapping[str, Any],
    key: str,
    paths: _ExecutionPackPaths,
    line_no: int,
) -> tuple[int, ...]:
    raw = record.get(key) or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ExecutionEvalError(
            f"{paths.pack_jsonl_path}:{line_no}: {key} must be a sequence of integers"
        )
    try:
        return tuple(int(token) for token in raw)
    except (TypeError, ValueError) as exc:
        raise ExecutionEvalError(
            f"{paths.pack_jsonl_path}:{line_no}: {key} must contain only integers"
        ) from exc


def _state_from_tokens(tokens: Sequence[int], *, length: int, vocab_size: int) -> dict[str, np.ndarray]:
    input_ids, attention_mask = _token_array(tokens, length=length, vocab_size=vocab_size)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": np.zeros_like(input_ids, dtype=np.int64),
        "changed_hunk_mask": np.zeros_like(attention_mask, dtype=bool),
    }


def _action_from_tokens(tokens: Sequence[int], *, length: int, vocab_size: int) -> dict[str, np.ndarray]:
    input_ids, attention_mask = _token_array(tokens, length=length, vocab_size=vocab_size)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _token_array(tokens: Sequence[int], *, length: int, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((length,), dtype=np.int64)
    mask = np.zeros((length,), dtype=bool)
    kept = tuple(tokens[:length])
    for index, token in enumerate(kept):
        values[index] = _fold_token_id(int(token), vocab_size=vocab_size)
        mask[index] = values[index] != 0
    return values, mask


def _fold_token_id(token: int, *, vocab_size: int) -> int:
    if token <= 0:
        return 0
    return ((token - 1) % (vocab_size - 1)) + 1


def _embed_rows(rows: Sequence[_ExecutionRow], *, model: Any, runtime: Any, device: Any) -> tuple[Any, Any, Any]:
    if not rows:
        raise ExecutionEvalError("cannot embed zero rows")
    state_before = _state_batch(tuple(row.state_before for row in rows), runtime=runtime, device=device)
    state_after = _state_batch(tuple(row.state_after for row in rows), runtime=runtime, device=device)
    action = _action_batch(tuple(row.action for row in rows), runtime=runtime, device=device, action_view=model.config.action_view)
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


def _embed_shuffled_predictions(
    rows: Sequence[_ExecutionRow],
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
    action = _action_batch(tuple(action_rows), runtime=runtime, device=device, action_view=model.config.action_view)
    was_training = bool(model.training)
    model.eval()
    with runtime.no_grad():
        action_emb = model.encode_action(action)
        z_pred_after = model.predict_after(z_before, action_emb)
    if was_training:
        model.train()
    return z_pred_after.float()


def _embed_state(state: Mapping[str, np.ndarray], *, model: Any, runtime: Any, device: Any) -> Any:
    was_training = bool(model.training)
    model.eval()
    with runtime.no_grad():
        encoded = model.encode_state(_state_batch((state,), runtime=runtime, device=device)).float()
    if was_training:
        model.train()
    return encoded


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
    return runtime.as_tensor(np.stack([np.asarray(row[key]) for row in rows], axis=0), device=device)


def _negative_squared_l2(query_vectors: Any, candidate_vectors: Any, *, runtime: Any) -> np.ndarray:
    scores = -runtime.cdist(query_vectors, candidate_vectors, p=2).pow(2).detach().cpu().numpy()
    if not np.isfinite(scores).all():
        raise ExecutionEvalError("score matrix contains NaN or inf")
    return np.asarray(scores, dtype=np.float64)


def _select_score_rows(
    scores: np.ndarray,
    *,
    query_indices: Sequence[int],
    candidate_indices: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(scores[query_index, candidate_index]) for candidate_index in candidate_indices)
        for query_index in query_indices
    )


def _build_execution_decoy_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_decoys: Sequence[str],
    seed: int,
    semantic_decoy_pack: LoadedSemanticDecoyPack | None = None,
) -> tuple[list[DecoyPair], list[DecoyGenerationReport]]:
    pairs: list[DecoyPair] = []
    reports: list[DecoyGenerationReport] = []
    loaded_categories: set[str] = set()
    if semantic_decoy_pack is not None:
        loaded_pairs = [
            pair for pair in semantic_decoy_pack.pairs if pair.category in selected_decoys
        ]
        pairs.extend(loaded_pairs)
        loaded_categories = {pair.category for pair in loaded_pairs}
        reports.extend(semantic_decoy_pack.generation_reports(loaded_categories))
    if "same_problem_different_submission" in selected_decoys:
        if "same_problem_different_submission" not in loaded_categories:
            generated, report = generate_same_problem_different_submission_pairs(records, seed=seed)
            pairs.extend(generated)
            reports.append(report)
    if "same_code_different_input" in selected_decoys:
        if "same_code_different_input" not in loaded_categories:
            generated, report = generate_same_code_different_input_pairs(records, seed=seed)
            pairs.extend(generated)
            reports.append(report)
    return pairs, reports


def _mutated_state_after(state: Mapping[str, np.ndarray], *, seed: int) -> Mapping[str, np.ndarray]:
    mutated = {key: np.asarray(value).copy() for key, value in state.items()}
    input_ids = mutated["input_ids"].copy()
    attention_mask = mutated["attention_mask"].copy()
    active = np.flatnonzero(attention_mask & (input_ids > 0))
    rng = random.Random(seed + int(input_ids.sum()) % 9973)
    if active.size:
        pos = int(rng.choice(active.tolist()))
    else:
        pos = 0
        attention_mask[pos] = True
    old = int(input_ids[pos])
    input_ids[pos] = ((old + rng.randint(1, DEFAULT_TRAINING_VOCAB_SIZE - 2) - 1) % (DEFAULT_TRAINING_VOCAB_SIZE - 1)) + 1
    mutated["input_ids"] = input_ids
    mutated["attention_mask"] = attention_mask
    return mutated


def _energy(query_vector: Any, candidate_vector: Any, *, runtime: Any) -> float:
    value = (query_vector - candidate_vector).float().pow(2).sum().detach().cpu().item()
    result = float(value)
    if not np.isfinite(result):
        raise SurpriseEvalError("execution surprise energy must be finite")
    return result


def _rank_lower_is_better(scores: Sequence[float], *, true_index: int) -> int:
    true_score = scores[true_index]
    better = sum(1 for index, score in enumerate(scores) if index != true_index and score < true_score)
    ties = sum(1 for index, score in enumerate(scores) if index != true_index and score == true_score)
    return 1 + better + ties // 2


def _execution_probe_rows(
    rows: Sequence[_ExecutionRow], *, selected_targets: Sequence[str]
) -> tuple[LatentProbeRow, ...]:
    label_by_target = {
        target: extract_labels([row.record for row in rows], target=target)
        for target in selected_targets
    }
    probe_rows: list[LatentProbeRow] = []
    for index, row in enumerate(rows):
        labels = {
            target: label_by_target[target].labels[index]
            for target in selected_targets
        }
        probe_rows.append(
            LatentProbeRow(
                transition_id=row.record_id,
                split=row.split,
                labels=labels,
                metadata_features={
                    "source_dataset": row.source_dataset,
                    "output_type": str(row.record.get("output_type") or ""),
                    "output_kind": str(row.record.get("output_kind") or ""),
                    "execution_status": str(row.record.get("execution_status") or ""),
                    "function_name": str(row.record.get("function_name") or ""),
                },
                lexical_tokens=_active_tokens(row.state_before)
                + _active_tokens(row.action)
                + _active_tokens(row.state_after),
            )
        )
    return tuple(probe_rows)


def _crash_samples(
    rows: Sequence[_ExecutionRow],
    *,
    z_before: np.ndarray,
    z_pred_after: np.ndarray,
    seed: int,
) -> tuple[CrashSample, ...]:
    code_norm = _minmax_scores(np.linalg.norm(z_before, axis=1))
    pred_norm = _minmax_scores(np.linalg.norm(z_pred_after, axis=1))
    combo_norm = _minmax_scores(np.linalg.norm(z_before + z_pred_after, axis=1))
    lexical = _minmax_scores(np.asarray([_active_count(row.state_before) for row in rows], dtype=np.float64))
    static = np.asarray([
        1.0 if str(row.record.get("output_kind") or "") == "exception" else 0.0
        for row in rows
    ], dtype=np.float64)
    random_scores = np.asarray([
        _stable_unit_float(f"{seed}:{row.record_id}") for row in rows
    ], dtype=np.float64)
    return tuple(
        CrashSample(
            record_id=row.record_id,
            will_raise=str(row.record.get("output_kind") or "") == "exception",
            exception_class=(
                str(row.record.get("output_type") or "exception")
                if str(row.record.get("output_kind") or "") == "exception"
                else None
            ),
            source_dataset=row.source_dataset,
            scores={
                "linear_code": float(code_norm[index]),
                "linear_code_input": float(combo_norm[index]),
                "linear_predicted_output": float(pred_norm[index]),
                "lexical": float(lexical[index]),
                "static": float(static[index]),
                "random": float(random_scores[index]),
            },
        )
        for index, row in enumerate(rows)
    )


def _not_evaluable_crash_report(*, sample_count: int, positives: int, negatives: int) -> CrashPredictionReport:
    reason = (
        "not_evaluable: need both positive and negative val/test samples; "
        f"got positives={positives}, negatives={negatives}"
    )
    return CrashPredictionReport(
        schema_version=CRASH_PREDICTION_REPORT_SCHEMA_VERSION,
        sample_count=sample_count,
        methods=(),
        best_latent_method=None,
        best_latent_auc=0.0,
        best_non_latent_method=None,
        best_non_latent_auc=0.0,
        latent_lift_auc=0.0,
        per_exception_class_auc={},
        per_source_dataset_auc={},
        claim_allowed=False,
        claim_reason=reason,
        min_latent_lift_for_claim=0.05,
    )


def _write_eval_artifact(
    *,
    out_dir: Path,
    files: Sequence[Path],
    command: Sequence[str],
    config: Mapping[str, Any],
    context: _ExecutionContext,
    result_schema_version: str,
    report_path: str,
    report_schema_version: str,
    metadata: Mapping[str, Any],
    source_git_sha: str | None,
    created_at: str | None,
    extra_parent_artifacts: Sequence[str] = (),
) -> ExecutionEvalResult:
    parent_artifacts = (
        context.training_artifact.artifact_id,
        context.pack_artifact.artifact_id,
        *tuple(extra_parent_artifacts),
    )
    artifact_metadata = {
        "schema_version": result_schema_version,
        "report_schema_version": report_schema_version,
        "report_path": report_path,
        "checkpoint_sha256": context.checkpoint_sha256,
        "checkpoint_step": _optional_int(context.checkpoint_payload.get("step"), "checkpoint.step"),
        "training_artifact_id": context.training_artifact.artifact_id,
        "dataset_artifact_id": context.pack_artifact.artifact_id,
        **dict(metadata),
    }
    artifact_manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=out_dir,
        files=tuple(files),
        command=command,
        config=config,
        parent_artifacts=parent_artifacts,
        source_git_sha=source_git_sha,
        created_at=created_at,
        metadata=artifact_metadata,
    )
    manifest_path = out_dir / "manifest.json"
    write_artifact_manifest(artifact_manifest, manifest_path)
    return ExecutionEvalResult(
        schema_version=result_schema_version,
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path=report_path,
        parent_artifacts=parent_artifacts,
        metadata=artifact_metadata,
    )


def _base_report_metadata(context: _ExecutionContext, *, checkpoint_path: Path) -> dict[str, Any]:
    return {
        "checkpoint": {
            "path": _display_path(checkpoint_path),
            "sha256": context.checkpoint_sha256,
            "step": _optional_int(context.checkpoint_payload.get("step"), "checkpoint.step"),
            "model_class": "TorchCodeTransitionModel",
            "backend": "torch",
        },
        "dataset": {
            "path": _display_path(context.pack_paths.root),
            "artifact_id": context.pack_artifact.artifact_id,
            "schema_version": EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
            "record_count": len(context.rows),
        },
        "training_artifact_id": context.training_artifact.artifact_id,
        "action_view": str(context.model.config.action_view),
        "substrate": "execution_trace_v1",
    }


def _select_rows(
    rows: Sequence[_ExecutionRow],
    *,
    splits: Sequence[str],
    max_rows: int,
    seed: int,
) -> tuple[_ExecutionRow, ...]:
    selected = [row for row in rows if row.split in set(splits)]
    if not selected:
        selected = list(rows)
    if len(selected) > max_rows:
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = sorted(selected[:max_rows], key=lambda row: row.record_id)
    return tuple(selected)


def _select_rows_per_split(
    rows: Sequence[_ExecutionRow], *, max_examples_per_split: int, seed: int
) -> tuple[_ExecutionRow, ...]:
    selected: list[_ExecutionRow] = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.split == split]
        if len(split_rows) > max_examples_per_split:
            rng = random.Random(seed + _stable_seed_offset(split))
            rng.shuffle(split_rows)
            split_rows = sorted(split_rows[:max_examples_per_split], key=lambda row: row.record_id)
        selected.extend(split_rows)
    if not selected:
        selected = list(rows[:max_examples_per_split])
    return tuple(selected)


def _source_slices(
    rows: Sequence[_ExecutionRow],
    ranks: Sequence[int],
    candidate_counts: Sequence[int],
) -> dict[str, RetrievalMetrics]:
    from .retrieval import compute_retrieval_metrics

    by_source: dict[str, list[int]] = {}
    counts_by_source: dict[str, list[int]] = {}
    for row, rank, count in zip(rows, ranks, candidate_counts):
        by_source.setdefault(row.source_dataset, []).append(rank)
        counts_by_source.setdefault(row.source_dataset, []).append(count)
    return {
        f"source:{source}": compute_retrieval_metrics(source_ranks, candidate_counts=counts_by_source[source])
        for source, source_ranks in by_source.items()
    }


def _parse_baselines(values: Sequence[str]) -> tuple[str, ...]:
    allowed = {"random", "lexical", "no_action", "shuffled_action"}
    return _parse_csv_values(values, allowed=allowed, name="baselines")


def _parse_decoys(values: Sequence[str]) -> tuple[str, ...]:
    allowed = {"mutation", "same_problem_different_submission", "same_code_different_input"}
    return _parse_csv_values(values, allowed=allowed, name="decoys")


def _parse_targets(values: Sequence[str]) -> tuple[str, ...]:
    return _parse_csv_values(values, allowed=set(EXECUTION_PROBE_TARGETS), name="targets")


def _parse_csv_values(values: Sequence[str], *, allowed: set[str], name: str) -> tuple[str, ...]:
    parsed: list[str] = []
    for raw in values:
        parsed.extend(part.strip() for part in str(raw).split(",") if part.strip())
    if not parsed:
        raise ExecutionEvalError(f"{name} must not be empty")
    unsupported = sorted(set(parsed) - allowed)
    if unsupported:
        raise ExecutionEvalError(f"unsupported {name}: {', '.join(unsupported)}")
    return tuple(dict.fromkeys(parsed))


def _reject_existing(out_dir: Path, relative_paths: Sequence[str], *, overwrite: bool) -> None:
    for relative in relative_paths:
        path = out_dir / relative
        if path.exists() and not overwrite:
            raise ExecutionEvalError(f"output already exists; pass --overwrite to replace: {path}")


def _tokens_to_text(tokens: Sequence[int]) -> str:
    return " ".join(str(int(token)) for token in tokens if int(token) != 0)


def _active_tokens(group: Mapping[str, Any]) -> tuple[int, ...]:
    ids = list(group["input_ids"])
    masks = list(group["attention_mask"])
    return tuple(int(token) for token, keep in zip(ids, masks) if keep and int(token) != 0)


def _active_count(group: Mapping[str, Any]) -> int:
    return len(_active_tokens(group))


def _to_numpy(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise ExecutionEvalError("latent matrix contains NaN or inf")
    return np.asarray(array, dtype=np.float64)


def _random_latent_like(matrix: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(matrix.shape).astype(np.float64)


def _minmax_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi == lo:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return (values.astype(np.float64) - lo) / (hi - lo)


def _stable_unit_float(value: str) -> float:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _stable_seed_offset(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ExecutionEvalError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ExecutionEvalError(f"{name} must be a positive integer")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ExecutionEvalError(f"{name} must be a non-negative integer")
    parsed = int(value)
    if parsed < 0:
        raise ExecutionEvalError(f"{name} must be a non-negative integer")
    return parsed


__all__ = [
    "CRASH_PREDICTION_EVAL_RUN_SCHEMA_VERSION",
    "EXECUTION_PROBE_EVAL_RUN_SCHEMA_VERSION",
    "EXECUTION_RETRIEVAL_EVAL_RUN_SCHEMA_VERSION",
    "EXECUTION_SURPRISE_EVAL_RUN_SCHEMA_VERSION",
    "ExecutionEvalError",
    "ExecutionEvalResult",
    "run_crash_prediction_evaluation",
    "run_execution_probe_evaluation",
    "run_execution_retrieval_evaluation",
    "run_execution_surprise_evaluation",
]
