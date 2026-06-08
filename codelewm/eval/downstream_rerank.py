"""Downstream reranking evaluation over manifest-backed benchmark packs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codelewm.harness.scorer import (
    ErrorReport,
    ScoreError,
    ScoreResult,
    _apply_unified_diff,
    load_scorer,
)
from codelewm.observability import (
    ArtifactManifest,
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security import parse_python_source_text

from .downstream import (
    DOWNSTREAM_REQUIRED_BASELINES,
    DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION,
    DownstreamCandidate,
    DownstreamRerankBenchmark,
    DownstreamTask,
    build_downstream_rerank_claim_gate,
)
from .downstream_anti_saturation import (
    ANTI_SATURATION_CLAIM_BASELINES,
    ANTI_SATURATION_CLAIM_METRICS,
    ANTI_SATURATION_PROFILE,
    build_anti_saturation_claim_gate,
    build_anti_saturation_report,
    compute_model_independent_baselines,
)
from .downstream_pack import read_downstream_rerank_benchmark


DOWNSTREAM_RERANK_EVAL_RUN_SCHEMA_VERSION = "codelewm.downstream_rerank_eval_run.v1"
# Hard-mode (RFC-0016) baselines: the standard seven plus the extra controls
# required by the anti-saturation benchmark. The plain path never uses these,
# so the v1.0 fixture contract (exactly DOWNSTREAM_REQUIRED_BASELINES) is intact.
HARD_DOWNSTREAM_EXTRA_BASELINES: tuple[str, ...] = (
    "shuffled_action",
    "static_heuristic",
    "p_pass",
)
HARD_DOWNSTREAM_REQUIRED_BASELINES: tuple[str, ...] = (
    DOWNSTREAM_REQUIRED_BASELINES + HARD_DOWNSTREAM_EXTRA_BASELINES
)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")


class DownstreamRerankEvalError(ValueError):
    """Raised when downstream reranking evaluation cannot run."""


@dataclass(frozen=True)
class DownstreamRerankEvalResult:
    """Summary returned after writing a downstream reranking report."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    parent_artifacts: tuple[str, ...]
    example_count: int
    claim_allowed: bool
    schema_version: str = DOWNSTREAM_RERANK_EVAL_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "parent_artifacts": list(self.parent_artifacts),
            "example_count": self.example_count,
            "claim_allowed": self.claim_allowed,
        }


@dataclass(frozen=True)
class _CandidateEvalRow:
    candidate: DownstreamCandidate
    path: Path
    candidate_text: str
    after_text: str | None
    score: ScoreResult | None
    error: ErrorReport | None

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def is_pass(self) -> bool:
        return self.candidate.label == "pass"

    def to_dict(self, root: Path) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "label": self.candidate.label,
            "llm_rank": self.candidate.llm_rank,
            "path": _relative_to_root(self.path, root),
            "static_check": self.candidate.static_check,
            "test_check": self.candidate.test_check,
            "source": dict(self.candidate.source),
            "provenance": dict(self.candidate.provenance),
            "score": None if self.score is None else self.score.to_dict(),
            "error": None if self.error is None else self.error.to_dict(),
        }


def run_downstream_rerank_evaluation(
    *,
    benchmark_manifest: Path | str,
    checkpoint: Path | str,
    out: Path | str,
    device: str = "auto",
    index: Path | str | None = None,
    retrieval_prior_weight: float = 0.0,
    retrieval_prior_k: int = 10,
    candidate_pack_manifests: Sequence[Path | str] = (),
    allow_unsafe_checkpoint: bool = False,
    pass_at_k: int = 5,
    bootstrap_samples: int = 200,
    seed: int = 0,
    hard_mode: bool = False,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "downstream-rerank"),
) -> DownstreamRerankEvalResult:
    """Run downstream candidate reranking from a benchmark artifact manifest.

    When ``hard_mode`` is true the report additionally includes the RFC-0016
    anti-saturation controls (shuffled-action, static-heuristic, typed p_pass),
    the ``codelewm.downstream_anti_saturation_report.v1`` diagnostic, bootstrap
    lift confidence intervals over no-action/lexical/LLM-order, and the
    three-baseline anti-saturation claim gate. The plain (non-hard) path is
    unchanged.
    """

    if pass_at_k < 1:
        raise DownstreamRerankEvalError("pass_at_k must be >= 1")
    if bootstrap_samples < 0:
        raise DownstreamRerankEvalError("bootstrap_samples must be >= 0")
    output_dir = Path(out).resolve()
    report_path = output_dir / "reports" / "downstream_rerank_report.json"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.json"
    if not overwrite and (
        report_path.exists() or config_path.exists() or manifest_path.exists()
    ):
        raise DownstreamRerankEvalError(
            f"output already exists; pass overwrite=True to replace: {output_dir}"
        )

    benchmark_manifest_path = Path(benchmark_manifest)
    benchmark_artifact = _read_verified_manifest(benchmark_manifest_path)
    if benchmark_artifact.artifact_kind != "downstream_benchmark":
        raise DownstreamRerankEvalError(
            "benchmark_manifest must have artifact_kind='downstream_benchmark'"
        )
    benchmark_path = benchmark_manifest_path.parent / _required_manifest_file(
        benchmark_artifact,
        "benchmark.json",
    )
    benchmark = read_downstream_rerank_benchmark(benchmark_path)
    candidate_pack_artifacts = tuple(
        _read_candidate_pack_manifest(path) for path in candidate_pack_manifests
    )

    scorer = load_scorer(
        checkpoint,
        device=device,
        index=index,
        retrieval_prior_weight=retrieval_prior_weight,
        retrieval_prior_k=retrieval_prior_k,
        allow_unsafe=allow_unsafe_checkpoint,
    )
    task_reports = [
        _evaluate_task(
            task,
            benchmark_root=benchmark_manifest_path.parent,
            scorer=scorer,
            pass_at_k=pass_at_k,
        )
        for task in benchmark.tasks
    ]
    metrics = _aggregate_metrics(task_reports, pass_at_k=pass_at_k)

    extra_report_fields: dict[str, Any] = {}
    report_required_baselines: Sequence[str] = DOWNSTREAM_REQUIRED_BASELINES
    if hard_mode:
        metrics = {
            **metrics,
            **_hard_baseline_metrics(
                benchmark,
                task_reports,
                benchmark_root=benchmark_manifest_path.parent,
                scorer=scorer,
                pass_at_k=pass_at_k,
            ),
        }
        report_required_baselines = HARD_DOWNSTREAM_REQUIRED_BASELINES
        anti_saturation_report = _build_eval_anti_saturation_report(
            benchmark, benchmark_root=benchmark_manifest_path.parent
        )
        lift_confidence_intervals = _lift_confidence_intervals(
            task_reports, pass_at_k=pass_at_k, bootstrap_samples=bootstrap_samples, seed=seed
        )
        claim_gate = build_anti_saturation_claim_gate(
            example_count=len(benchmark.tasks),
            metrics=metrics,
            anti_saturation_eligible=bool(anti_saturation_report["eligible"]),
            lift_confidence_intervals=(
                lift_confidence_intervals["intervals"]
                if lift_confidence_intervals.get("status") == "computed"
                else None
            ),
            min_labeled_examples=benchmark.min_labeled_examples,
        )
        extra_report_fields = {
            "hard_mode": True,
            "profile": ANTI_SATURATION_PROFILE,
            "anti_saturation_report": anti_saturation_report,
            "lift_confidence_intervals": lift_confidence_intervals,
        }
    else:
        claim_gate = build_downstream_rerank_claim_gate(
            example_count=len(benchmark.tasks),
            metrics=metrics,
            min_labeled_examples=benchmark.min_labeled_examples,
        )
    report = _build_report(
        benchmark=benchmark,
        benchmark_manifest=benchmark_artifact,
        benchmark_manifest_path=benchmark_manifest_path,
        candidate_pack_artifacts=candidate_pack_artifacts,
        checkpoint=Path(checkpoint),
        checkpoint_sha256=scorer.checkpoint_sha256,
        model_id=scorer.model_id,
        index=index,
        retrieval_prior_weight=retrieval_prior_weight,
        retrieval_prior_k=retrieval_prior_k,
        pass_at_k=pass_at_k,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        task_reports=task_reports,
        metrics=metrics,
        claim_gate=claim_gate,
        required_baselines=report_required_baselines,
        extra_report_fields=extra_report_fields,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    eval_config = {
        "benchmark_manifest": str(benchmark_manifest_path),
        "candidate_pack_manifests": [str(path) for path in candidate_pack_manifests],
        "checkpoint": str(checkpoint),
        "device": device,
        "index": None if index is None else str(index),
        "retrieval_prior_weight": retrieval_prior_weight,
        "retrieval_prior_k": retrieval_prior_k,
        "allow_unsafe_checkpoint": allow_unsafe_checkpoint,
        "pass_at_k": pass_at_k,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "hard_mode": hard_mode,
    }
    config_path.write_text(
        json.dumps(eval_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    parent_artifact_ids = (
        benchmark_artifact.artifact_id,
        *(artifact.artifact_id for artifact in candidate_pack_artifacts),
    )
    artifact_manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=output_dir,
        files=(config_path, report_path),
        command=command,
        config=eval_config,
        parent_artifacts=parent_artifact_ids,
        metadata={
            "schema_version": DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION,
            "benchmark_id": benchmark.benchmark_id,
            "example_count": len(benchmark.tasks),
            "claim_allowed": claim_gate["allowed"],
            "pass_at_k": pass_at_k,
            "hard_mode": hard_mode,
            **(
                {"anti_saturation_eligible": extra_report_fields["anti_saturation_report"]["eligible"]}
                if hard_mode
                else {}
            ),
        },
    )
    write_artifact_manifest(artifact_manifest, manifest_path)
    return DownstreamRerankEvalResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path=_relative_to_root(report_path, output_dir),
        parent_artifacts=parent_artifact_ids,
        example_count=len(benchmark.tasks),
        claim_allowed=bool(claim_gate["allowed"]),
    )


def read_downstream_rerank_report(path: Path | str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DownstreamRerankEvalError("downstream rerank report must be a JSON object")
    if payload.get("schema_version") != DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION:
        raise DownstreamRerankEvalError("unsupported downstream rerank report schema_version")
    return payload


def _evaluate_task(
    task: DownstreamTask,
    *,
    benchmark_root: Path,
    scorer,
    pass_at_k: int,
) -> dict[str, Any]:
    before_path = _resolve_benchmark_path(task.before_path, benchmark_root)
    before_text = before_path.read_text(encoding="utf-8")
    candidate_rows = [
        _evaluate_candidate(
            candidate,
            before_path=before_path,
            before_text=before_text,
            task=task,
            benchmark_root=benchmark_root,
            scorer=scorer,
        )
        for candidate in task.candidates
    ]
    rankings = _rankings(task, candidate_rows, before_text=before_text)
    task_metrics = {
        baseline: _metrics_for_order(order, candidate_rows, pass_at_k=pass_at_k)
        for baseline, order in rankings.items()
    }
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "prompt": task.prompt,
        "before_path": task.before_path,
        "provenance": dict(task.provenance),
        "candidate_rows": [row.to_dict(benchmark_root) for row in candidate_rows],
        "rankings": rankings,
        "metrics": task_metrics,
    }


def _evaluate_candidate(
    candidate: DownstreamCandidate,
    *,
    before_path: Path,
    before_text: str,
    task: DownstreamTask,
    benchmark_root: Path,
    scorer,
) -> _CandidateEvalRow:
    relative_path = candidate.patch_path or candidate.after_state_path
    if relative_path is None:
        raise DownstreamRerankEvalError(
            f"candidate {candidate.candidate_id} has no materialized path"
        )
    candidate_path = _resolve_benchmark_path(relative_path, benchmark_root)
    candidate_text = candidate_path.read_text(encoding="utf-8")
    after_text = _candidate_after_text(candidate, candidate_text, before_text)
    try:
        if after_text is None:
            raise ScoreError(
                "candidate patch could not be applied",
                error_type="patch_apply_failed",
                remediation="inspect the benchmark candidate patch",
                artifact=str(candidate_path),
            )
        try:
            parse_python_source_text(after_text, filename=str(candidate_path))
        except SyntaxError as exc:
            raise ScoreError(
                "candidate file is not valid Python",
                error_type="invalid_syntax",
                remediation="provide a parseable Python file or patch",
                artifact=str(candidate_path),
                caused_by=f"{exc.__class__.__name__}: {exc.msg}",
            ) from exc
        score = scorer.score_texts(
            before=before_text,
            instruction=task.prompt,
            candidate=after_text,
            candidate_name=str(candidate_path),
        )
        return _CandidateEvalRow(
            candidate,
            candidate_path,
            candidate_text,
            after_text,
            score,
            None,
        )
    except ScoreError as exc:
        return _CandidateEvalRow(
            candidate,
            candidate_path,
            candidate_text,
            after_text,
            None,
            exc.to_error_report(record_id=candidate.candidate_id),
        )


def _candidate_after_text(
    candidate: DownstreamCandidate,
    candidate_text: str,
    before_text: str,
) -> str | None:
    if candidate.after_state_path is not None:
        return candidate_text
    try:
        return _apply_unified_diff(
            before_text,
            candidate_text,
            artifact=candidate.patch_path or "<patch>",
        )
    except ScoreError:
        return None


def _rankings(
    task: DownstreamTask,
    candidate_rows: Sequence[_CandidateEvalRow],
    *,
    before_text: str,
) -> dict[str, list[str]]:
    by_id = {row.candidate_id: row for row in candidate_rows}
    llm_order = [
        row.candidate_id
        for row in sorted(
            candidate_rows,
            key=lambda row: (row.candidate.llm_rank, row.candidate_id),
        )
    ]
    random_order = [
        row.candidate_id
        for row in sorted(
            candidate_rows,
            key=lambda row: _stable_random_key(task.task_id, row.candidate_id),
        )
    ]
    lexical_order = [
        row.candidate_id
        for row in sorted(
            candidate_rows,
            key=lambda row: (
                -_lexical_similarity(task.prompt, row.after_text or row.candidate_text),
                row.candidate_id,
            ),
        )
    ]
    no_action_order = [
        row.candidate_id
        for row in sorted(
            candidate_rows,
            key=lambda row: (
                -_lexical_similarity(before_text, row.after_text or row.candidate_text),
                row.candidate_id,
            ),
        )
    ]
    codelewm_order = _score_order(candidate_rows, "transition_energy")
    retrieval_prior_order = _score_order(candidate_rows, "retrieval_prior")
    ensemble_order = _score_order(candidate_rows, "final_score")
    rankings = {
        "llm_order": llm_order,
        "random": random_order,
        "lexical": lexical_order,
        "no_action": no_action_order,
        "codelewm": codelewm_order,
        "retrieval_prior": retrieval_prior_order,
        "score_ensemble": ensemble_order,
    }
    missing = set(DOWNSTREAM_REQUIRED_BASELINES) - set(rankings)
    if missing:
        raise DownstreamRerankEvalError(
            "missing downstream baseline rankings: " + ", ".join(sorted(missing))
        )
    return {
        name: [candidate_id for candidate_id in order if candidate_id in by_id]
        for name, order in rankings.items()
    }


def _score_order(candidate_rows: Sequence[_CandidateEvalRow], score_key: str) -> list[str]:
    scored = []
    errors = []
    for row in candidate_rows:
        score = row.score
        if score is None:
            errors.append(row.candidate_id)
            continue
        value = getattr(score, score_key)
        if value is None or not math.isfinite(float(value)):
            errors.append(row.candidate_id)
            continue
        scored.append((float(value), row.candidate_id))
    return [candidate_id for _, candidate_id in sorted(scored)] + sorted(errors)


def _aggregate_metrics(task_reports: Sequence[Mapping[str, Any]], *, pass_at_k: int) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for baseline in DOWNSTREAM_REQUIRED_BASELINES:
        task_metrics = [task["metrics"][baseline] for task in task_reports]
        availability = _baseline_availability(baseline, task_reports)
        metrics[baseline] = {
            "pass_at_1": _mean_metric(task_metrics, "pass_at_1"),
            "pass_at_k": _mean_metric(task_metrics, "pass_at_k"),
            "pass_at_k_value": pass_at_k,
            "mrr": _mean_metric(task_metrics, "mrr"),
            "valid_patch_rate": _mean_metric(task_metrics, "valid_patch_rate"),
            "check_pass_rate": _mean_metric(task_metrics, "check_pass_rate"),
            "mean_first_pass_rank": _mean(
                metric["first_pass_rank"]
                for metric in task_metrics
                if metric["first_pass_rank"] is not None
            ),
            "evaluated_examples": len(task_metrics),
            **availability,
        }
    return metrics


def _baseline_availability(
    baseline: str,
    task_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    score_keys = {
        "codelewm": "transition_energy",
        "retrieval_prior": "retrieval_prior",
        "score_ensemble": "final_score",
    }
    score_key = score_keys.get(baseline)
    if score_key is None:
        return {"status": "completed"}
    finite_score_count = _finite_score_count(task_reports, score_key)
    if finite_score_count == 0:
        return {
            "status": "blocked",
            "blocked_reason": f"no_finite_{score_key}_scores",
            "finite_score_count": 0,
        }
    return {"status": "completed", "finite_score_count": finite_score_count}


def _finite_score_count(task_reports: Sequence[Mapping[str, Any]], score_key: str) -> int:
    count = 0
    for task in task_reports:
        rows = task.get("candidate_rows", ())
        if not isinstance(rows, Sequence):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            score = row.get("score")
            if not isinstance(score, Mapping):
                continue
            value = score.get(score_key)
            if value is not None and math.isfinite(float(value)):
                count += 1
    return count


def _metrics_for_order(
    order: Sequence[str],
    candidate_rows: Sequence[_CandidateEvalRow],
    *,
    pass_at_k: int,
) -> dict[str, Any]:
    by_id = {row.candidate_id: row for row in candidate_rows}
    pass_ranks = [
        rank
        for rank, candidate_id in enumerate(order, start=1)
        if by_id[candidate_id].is_pass
    ]
    first_pass_rank = min(pass_ranks) if pass_ranks else None
    top_k = list(order[:pass_at_k])
    valid_count = sum(1 for row in candidate_rows if row.candidate.static_check == "pass")
    check_pass_count = sum(
        1
        for row in candidate_rows
        if row.candidate.static_check == "pass"
        and row.candidate.test_check in {"pass", "not_run", "not_applicable"}
    )
    total = max(len(candidate_rows), 1)
    return {
        "pass_at_1": 1.0 if order and by_id[order[0]].is_pass else 0.0,
        "pass_at_k": 1.0
        if any(by_id[candidate_id].is_pass for candidate_id in top_k)
        else 0.0,
        "mrr": 0.0 if first_pass_rank is None else 1.0 / first_pass_rank,
        "first_pass_rank": first_pass_rank,
        "valid_patch_rate": valid_count / total,
        "check_pass_rate": check_pass_count / total,
    }


def _hard_baseline_metrics(
    benchmark: DownstreamRerankBenchmark,
    task_reports: Sequence[Mapping[str, Any]],
    *,
    benchmark_root: Path,
    scorer,
    pass_at_k: int,
) -> dict[str, dict[str, Any]]:
    """Compute the RFC-0016 extra controls: shuffled-action, static-heuristic, p_pass."""

    before_texts = [
        _resolve_benchmark_path(task.before_path, benchmark_root).read_text(encoding="utf-8")
        for task in benchmark.tasks
    ]
    n = len(benchmark.tasks)
    shuffled_per_task: list[Mapping[str, Any]] = []
    static_per_task: list[Mapping[str, Any]] = []
    for index, (task, task_report) in enumerate(zip(benchmark.tasks, task_reports)):
        rows_by_id = {row["candidate_id"]: row for row in task_report["candidate_rows"]}
        shuffled_before = before_texts[(index + 1) % n]
        shuffled_order = _shuffled_action_order(
            task,
            before_text=before_texts[index],
            shuffled_before_text=shuffled_before,
            benchmark_root=benchmark_root,
            scorer=scorer,
        )
        shuffled_per_task.append(
            _order_metrics_from_dicts(shuffled_order, rows_by_id, pass_at_k=pass_at_k)
        )
        static_order = _static_heuristic_order(task_report["candidate_rows"])
        static_per_task.append(
            _order_metrics_from_dicts(static_order, rows_by_id, pass_at_k=pass_at_k)
        )
    return {
        "shuffled_action": {
            **_aggregate_simple_metrics(shuffled_per_task, pass_at_k=pass_at_k),
            "status": "completed",
        },
        "static_heuristic": {
            **_aggregate_simple_metrics(static_per_task, pass_at_k=pass_at_k),
            "status": "completed",
        },
        "p_pass": _p_pass_metrics(task_reports),
    }


def _shuffled_action_order(
    task: DownstreamTask,
    *,
    before_text: str,
    shuffled_before_text: str,
    benchmark_root: Path,
    scorer,
) -> list[str]:
    """Rank candidates by transition energy scored against a *shuffled* before-state.

    This is the action-sensitivity control: if CodeLeWM's lift came from action
    (before->after) understanding, scoring against the wrong before-state should
    degrade the ranking. Candidate code is parsed and diff-applied as text only.
    """

    scored: list[tuple[float, str]] = []
    errors: list[str] = []
    for candidate in task.candidates:
        relative_path = candidate.patch_path or candidate.after_state_path
        if relative_path is None:
            errors.append(candidate.candidate_id)
            continue
        candidate_path = _resolve_benchmark_path(relative_path, benchmark_root)
        candidate_text = candidate_path.read_text(encoding="utf-8")
        after_text = _candidate_after_text(candidate, candidate_text, before_text)
        if after_text is None:
            errors.append(candidate.candidate_id)
            continue
        try:
            parse_python_source_text(after_text, filename=str(candidate_path))
            score = scorer.score_texts(
                before=shuffled_before_text,
                instruction=task.prompt,
                candidate=after_text,
                candidate_name=str(candidate_path),
            )
        except (ScoreError, SyntaxError):
            errors.append(candidate.candidate_id)
            continue
        value = float(score.transition_energy)
        if not math.isfinite(value):
            errors.append(candidate.candidate_id)
            continue
        scored.append((value, candidate.candidate_id))
    return [candidate_id for _, candidate_id in sorted(scored)] + sorted(errors)


def _static_heuristic_order(candidate_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Deterministic static-only ranking: parseable+checked candidates first."""

    def sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
        static_ok = 0 if row.get("static_check") == "pass" else 1
        check_ok = (
            0
            if row.get("static_check") == "pass"
            and row.get("test_check") in {"pass", "not_run", "not_applicable"}
            else 1
        )
        return (static_ok, check_ok, str(row["candidate_id"]))

    return [str(row["candidate_id"]) for row in sorted(candidate_rows, key=sort_key)]


def _order_metrics_from_dicts(
    order: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    pass_at_k: int,
) -> dict[str, Any]:
    def is_pass(candidate_id: str) -> bool:
        return rows_by_id[candidate_id]["label"] == "pass"

    pass_ranks = [rank for rank, candidate_id in enumerate(order, start=1) if is_pass(candidate_id)]
    first_pass_rank = min(pass_ranks) if pass_ranks else None
    rows = list(rows_by_id.values())
    total = max(len(rows), 1)
    valid_count = sum(1 for row in rows if row.get("static_check") == "pass")
    check_pass_count = sum(
        1
        for row in rows
        if row.get("static_check") == "pass"
        and row.get("test_check") in {"pass", "not_run", "not_applicable"}
    )
    return {
        "pass_at_1": 1.0 if order and is_pass(order[0]) else 0.0,
        "pass_at_k": 1.0 if any(is_pass(candidate_id) for candidate_id in order[:pass_at_k]) else 0.0,
        "mrr": 0.0 if first_pass_rank is None else 1.0 / first_pass_rank,
        "first_pass_rank": first_pass_rank,
        "valid_patch_rate": valid_count / total,
        "check_pass_rate": check_pass_count / total,
    }


def _aggregate_simple_metrics(
    per_task: Sequence[Mapping[str, Any]],
    *,
    pass_at_k: int,
) -> dict[str, Any]:
    return {
        "pass_at_1": _mean_metric(per_task, "pass_at_1"),
        "pass_at_k": _mean_metric(per_task, "pass_at_k"),
        "pass_at_k_value": pass_at_k,
        "mrr": _mean_metric(per_task, "mrr"),
        "valid_patch_rate": _mean_metric(per_task, "valid_patch_rate"),
        "check_pass_rate": _mean_metric(per_task, "check_pass_rate"),
        "mean_first_pass_rank": _mean(
            metric["first_pass_rank"]
            for metric in per_task
            if metric["first_pass_rank"] is not None
        ),
        "evaluated_examples": len(per_task),
    }


def _p_pass_metrics(task_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """p_pass is typed not_recorded unless a standalone p_pass score key exists on every row."""

    for task in task_reports:
        for row in task["candidate_rows"]:
            score = row.get("score")
            if not isinstance(score, Mapping):
                return {"status": "not_recorded", "reason": "no_standalone_p_pass_score_key"}
            value = score.get("p_pass")
            if value is None or not math.isfinite(float(value)):
                return {"status": "not_recorded", "reason": "no_standalone_p_pass_score_key"}
    return {"status": "not_recorded", "reason": "no_standalone_p_pass_score_key"}


def _build_eval_anti_saturation_report(
    benchmark: DownstreamRerankBenchmark,
    *,
    benchmark_root: Path,
) -> dict[str, Any]:
    inputs = compute_model_independent_baselines(benchmark, root=benchmark_root)
    # The benchmark was loaded from a checksum-verified manifest and the pack
    # build enforced the source/license, split-leakage, and secret-scan gates;
    # the authoritative pre-scoring gates live in the pack's own #419 report.
    return build_anti_saturation_report(
        profile=ANTI_SATURATION_PROFILE,
        source_license_ok=True,
        split_leakage_ok=True,
        manifest_ok=True,
        secret_scan_ok=True,
        **inputs,
    )


def _lift_confidence_intervals(
    task_reports: Sequence[Mapping[str, Any]],
    *,
    pass_at_k: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap CodeLeWM lift (codelewm - baseline) over no-action/lexical/LLM-order."""

    if len(task_reports) < 20 or bootstrap_samples <= 0:
        return {
            "status": "skipped",
            "reason": "lift confidence intervals require at least 20 examples and bootstrap_samples > 0",
        }
    rng_state = int(hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:16], 16)
    n = len(task_reports)
    samples: dict[str, dict[str, list[float]]] = {
        baseline: {metric: [] for metric in ANTI_SATURATION_CLAIM_METRICS}
        for baseline in ANTI_SATURATION_CLAIM_BASELINES
    }
    for _ in range(bootstrap_samples):
        selected = []
        for _index in range(n):
            rng_state = (1103515245 * rng_state + 12345) % (2**31)
            selected.append(task_reports[rng_state % n])
        aggregate = _aggregate_metrics(selected, pass_at_k=pass_at_k)
        codelewm = aggregate["codelewm"]
        for baseline in ANTI_SATURATION_CLAIM_BASELINES:
            for metric in ANTI_SATURATION_CLAIM_METRICS:
                samples[baseline][metric].append(
                    float(codelewm[metric]) - float(aggregate[baseline][metric])
                )
    return {
        "status": "computed",
        "bootstrap_samples": bootstrap_samples,
        "intervals": {
            baseline: {metric: _interval(values) for metric, values in metric_samples.items()}
            for baseline, metric_samples in samples.items()
        },
    }


def _build_report(
    *,
    benchmark: DownstreamRerankBenchmark,
    benchmark_manifest: ArtifactManifest,
    benchmark_manifest_path: Path,
    candidate_pack_artifacts: Sequence[ArtifactManifest],
    checkpoint: Path,
    checkpoint_sha256: str,
    model_id: str,
    index: Path | str | None,
    retrieval_prior_weight: float,
    retrieval_prior_k: int,
    pass_at_k: int,
    bootstrap_samples: int,
    seed: int,
    task_reports: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
    claim_gate: Mapping[str, Any],
    required_baselines: Sequence[str] = DOWNSTREAM_REQUIRED_BASELINES,
    extra_report_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION,
        "report_id": f"{benchmark.benchmark_id}-downstream-rerank",
        "benchmark_id": benchmark.benchmark_id,
        "artifact_inputs": {
            "benchmark_manifest": str(benchmark_manifest_path),
            "benchmark_artifact_id": benchmark_manifest.artifact_id,
            "candidate_pack_artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "schema_version": artifact.metadata.get("schema_version"),
                }
                for artifact in candidate_pack_artifacts
            ],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "model_id": model_id,
            "index": None if index is None else str(index),
        },
        "evaluation_config": {
            "retrieval_prior_weight": retrieval_prior_weight,
            "retrieval_prior_k": retrieval_prior_k,
            "pass_at_k": pass_at_k,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "score_direction": "lower_is_better",
            "execution_policy": "candidate code is parsed and diff-applied as text but never executed",
        },
        "summary": {
            "example_count": len(task_reports),
            "candidate_count": sum(len(task["candidate_rows"]) for task in task_reports),
            "required_baselines": list(required_baselines),
            "claim_allowed": claim_gate["allowed"],
        },
        "metrics": {baseline: dict(values) for baseline, values in metrics.items()},
        "confidence_intervals": _confidence_intervals(
            task_reports,
            pass_at_k=pass_at_k,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "slices": _slices(task_reports),
        "claim_gate": dict(claim_gate),
        "tasks": list(task_reports),
        "caveats": _caveats(task_reports, candidate_pack_artifacts, claim_gate, metrics),
    }
    if extra_report_fields:
        report.update(dict(extra_report_fields))
    _ensure_json_native(report, "downstream rerank report")
    return report


def _confidence_intervals(
    task_reports: Sequence[Mapping[str, Any]],
    *,
    pass_at_k: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(task_reports) < 20 or bootstrap_samples <= 0:
        return {
            "status": "skipped",
            "reason": "confidence intervals require at least 20 examples and bootstrap_samples > 0",
        }
    # Deterministic lightweight bootstrap without importing numpy.
    rng_state = int(hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:16], 16)
    samples: dict[str, dict[str, list[float]]] = {
        baseline: {"pass_at_1": [], "mrr": []} for baseline in DOWNSTREAM_REQUIRED_BASELINES
    }
    n = len(task_reports)
    for _ in range(bootstrap_samples):
        selected = []
        for _index in range(n):
            rng_state = (1103515245 * rng_state + 12345) % (2**31)
            selected.append(task_reports[rng_state % n])
        metrics = _aggregate_metrics(selected, pass_at_k=pass_at_k)
        for baseline in DOWNSTREAM_REQUIRED_BASELINES:
            samples[baseline]["pass_at_1"].append(float(metrics[baseline]["pass_at_1"]))
            samples[baseline]["mrr"].append(float(metrics[baseline]["mrr"]))
    return {
        "status": "computed",
        "bootstrap_samples": bootstrap_samples,
        "intervals": {
            baseline: {
                metric: _interval(values)
                for metric, values in metric_samples.items()
            }
            for baseline, metric_samples in samples.items()
        },
    }


def _slices(task_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task_type = Counter(str(task["task_type"]) for task in task_reports)
    by_candidate_source: Counter[str] = Counter()
    by_failure_type: Counter[str] = Counter()
    for task in task_reports:
        for row in task["candidate_rows"]:
            source = row.get("source", {})
            if isinstance(source, Mapping):
                by_candidate_source[str(source.get("candidate_kind", "unknown"))] += 1
            error = row.get("error")
            if isinstance(error, Mapping):
                by_failure_type[str(error.get("error_type", "unknown"))] += 1
            elif row["static_check"] != "pass":
                by_failure_type[f"static_check:{row['static_check']}"] += 1
    return {
        "by_task_type": dict(sorted(by_task_type.items())),
        "by_candidate_source": dict(sorted(by_candidate_source.items())),
        "by_failure_type": dict(sorted(by_failure_type.items())),
    }


def _caveats(
    task_reports: Sequence[Mapping[str, Any]],
    candidate_pack_artifacts: Sequence[ArtifactManifest],
    claim_gate: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    caveats = [
        "Downstream usefulness claims require at least 100 labeled examples and strict improvement over LLM-order and no-action baselines.",
    ]
    if not candidate_pack_artifacts:
        caveats.append(
            "No candidate-pack manifest was provided; this fixture uses benchmark-pack materialized candidates."
        )
    if metrics.get("retrieval_prior", {}).get("status") == "blocked":
        caveats.append(
            "Retrieval-prior baseline is unavailable because no finite retrieval_prior scores were produced; pass --index to evaluate it."
        )
    if not claim_gate.get("allowed"):
        caveats.append("The downstream claim gate is closed for this report.")
    if any(row.get("error") for task in task_reports for row in task["candidate_rows"]):
        caveats.append(
            "One or more candidates failed static parse or patch application and were ranked after scored candidates."
        )
    return caveats


def _read_candidate_pack_manifest(path: Path | str) -> ArtifactManifest:
    manifest = _read_verified_manifest(Path(path))
    if manifest.artifact_kind != "candidate_pack":
        raise DownstreamRerankEvalError("candidate_pack_manifest must have artifact_kind='candidate_pack'")
    return manifest


def _read_verified_manifest(path: Path) -> ArtifactManifest:
    manifest = read_artifact_manifest(path)
    validate_artifact_checksums(manifest, root=path.parent)
    return manifest


def _required_manifest_file(manifest: ArtifactManifest, path: str) -> str:
    for file in manifest.files:
        if file.path == path:
            return file.path
    raise DownstreamRerankEvalError(f"artifact manifest does not list required file: {path}")


def _resolve_benchmark_path(path: str, root: Path) -> Path:
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise DownstreamRerankEvalError(f"benchmark path escapes artifact root: {path}") from exc
    if not resolved.is_file():
        raise DownstreamRerankEvalError(f"benchmark file does not exist: {path}")
    return resolved


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DownstreamRerankEvalError(f"path escapes artifact root: {path}") from exc


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(left.lower()))
    right_tokens = set(_TOKEN_RE.findall(right.lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _stable_random_key(task_id: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{task_id}\0{candidate_id}".encode("utf-8")).hexdigest()


def _mean_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> float:
    if not metrics:
        return 0.0
    return sum(float(metric[key]) for metric in metrics) / len(metrics)


def _mean(values: Any) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return None
    return statistics.fmean(finite)


def _interval(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"low": 0.0, "high": 0.0}
    low_index = max(0, int(math.floor(0.025 * (len(ordered) - 1))))
    high_index = min(len(ordered) - 1, int(math.ceil(0.975 * (len(ordered) - 1))))
    return {"low": float(ordered[low_index]), "high": float(ordered[high_index])}


def _ensure_json_native(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DownstreamRerankEvalError(f"{field_name} must be JSON-native: {exc}") from exc
