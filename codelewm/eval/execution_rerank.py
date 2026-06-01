"""HumanEval / MBPP-Plus pass@1 reranking evaluation.

This is the headline downstream gate for the v0.6 substrate pivot
(RFC-0014, #268). The protocol is:

1. For each problem, the operator already sampled N completions from a
   reference LLM and ran the hidden tests in a sandbox so every
   completion carries a ground-truth pass / fail label.
2. The evaluator computes a CodeLeWM rerank score for each completion
   (the caller supplies a ``scorer`` callable so the evaluator stays
   model-agnostic and testable).
3. The evaluator computes pass@1 under each baseline ordering and
   reports the lift over LLM original order plus a bootstrap 95% CI.

The module is callable directly from Python and from the CLI
``codelewm eval rerank-humaneval`` / ``codelewm eval rerank-mbpp-plus``.
Live LLM sampling and live hidden-test execution are operator-driven;
this module consumes the labeled-completion artifact those steps
produce.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codelewm.harness.scorer import ScoreError, load_scorer
from codelewm.observability import (
    ArtifactManifest,
    build_artifact_manifest,
    read_artifact_manifest,
    sha256_file,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security.secret_scan import scan_paths


EXECUTION_RERANK_REPORT_SCHEMA_VERSION = "codelewm.eval.execution_rerank_report.v1"
EXECUTION_RERANK_EVAL_RUN_SCHEMA_VERSION = "codelewm.eval.execution_rerank_eval_run.v1"
COMPLETION_LABEL_SCHEMA_VERSION = "codelewm.eval.completion_label.v1"
COMPLETION_SCORE_SCHEMA_VERSION = "codelewm.eval.completion_score.v1"
EXECUTION_RERANK_BASELINES: tuple[str, ...] = (
    "random",
    "lexical",
    "llm_order",
    "no_action",
    "shuffled_action",
    "codelewm",
)
RerankBaseline = Literal[
    "random", "lexical", "llm_order", "no_action", "shuffled_action", "codelewm"
]


class ExecutionRerankError(ValueError):
    """Raised when the rerank inputs do not satisfy the protocol."""


@dataclass(frozen=True)
class CompletionLabel:
    """One LLM completion with its hidden-test outcome."""

    problem_id: str
    completion_id: str
    code: str
    llm_order_rank: int  # 1-indexed rank under LLM sampling order
    passed: bool
    scoring_inputs: tuple["CompletionScoringInput", ...] = ()


@dataclass(frozen=True)
class CompletionScoringInput:
    """One execution input repr used only for model scoring, not correctness."""

    input_id: str
    input_repr: str
    input_kind: str = "function_call"
    function_name: str | None = None

    @classmethod
    def from_mapping(cls, payload: Any) -> "CompletionScoringInput":
        if not isinstance(payload, dict):
            raise ValueError("scoring input must be a JSON object")
        input_id = payload.get("input_id")
        input_repr = payload.get("input_repr")
        if not isinstance(input_id, str) or not input_id:
            raise ValueError("scoring input input_id must be a non-empty string")
        if not isinstance(input_repr, str) or not input_repr:
            raise ValueError("scoring input input_repr must be a non-empty string")
        input_kind = payload.get("input_kind", "function_call")
        function_name = payload.get("function_name")
        return cls(
            input_id=input_id,
            input_repr=input_repr,
            input_kind=str(input_kind),
            function_name=None if function_name is None else str(function_name),
        )


@dataclass(frozen=True)
class ScoredCompletion:
    """A completion paired with the scores assigned by each baseline."""

    label: CompletionLabel
    scores: dict[str, float]  # baseline -> score (higher = better)


@dataclass(frozen=True)
class BaselineSummary:
    """Pass@1 metrics for one baseline ordering."""

    baseline: str
    pass_at_1: float
    pass_count: int
    problem_count: int
    pass_at_k: float | None = None
    pass_at_k_value: int | None = None
    mrr: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "baseline": self.baseline,
            "pass_at_1": self.pass_at_1,
            "pass_count": self.pass_count,
            "problem_count": self.problem_count,
        }
        if self.pass_at_k is not None:
            payload["pass_at_k"] = self.pass_at_k
        if self.pass_at_k_value is not None:
            payload["pass_at_k_value"] = self.pass_at_k_value
        if self.mrr is not None:
            payload["mrr"] = self.mrr
        return payload


@dataclass(frozen=True)
class ExecutionRerankReport:
    """Aggregate report. JSON-serializable."""

    schema_version: str
    benchmark: str
    problem_count: int
    completions_per_problem: int
    baselines: tuple[BaselineSummary, ...]
    codelewm_lift_over_llm_order: float
    bootstrap_lift_ci: tuple[float, float]
    bootstrap_seed: int
    claim_allowed: bool
    claim_reason: str
    codelewm_lift_over_no_action: float | None = None
    bootstrap_lift_over_no_action_ci: tuple[float, float] | None = None
    pass_at_k: int = 5
    confidence_level: float = 0.95
    scoring_summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "problem_count": self.problem_count,
            "completions_per_problem": self.completions_per_problem,
            "baselines": [b.as_dict() for b in self.baselines],
            "codelewm_lift_over_llm_order": self.codelewm_lift_over_llm_order,
            "bootstrap_lift_ci": list(self.bootstrap_lift_ci),
            "bootstrap_seed": self.bootstrap_seed,
            "claim_allowed": self.claim_allowed,
            "claim_reason": self.claim_reason,
            "pass_at_k": self.pass_at_k,
            "confidence_level": self.confidence_level,
            "scoring_summary": dict(self.scoring_summary),
        }
        if self.codelewm_lift_over_no_action is not None:
            payload["codelewm_lift_over_no_action"] = self.codelewm_lift_over_no_action
        if self.bootstrap_lift_over_no_action_ci is not None:
            payload["bootstrap_lift_over_no_action_ci"] = list(
                self.bootstrap_lift_over_no_action_ci
            )
        return payload


@dataclass(frozen=True)
class ExecutionRerankEvalResult:
    """Summary returned after writing a completion-label rerank report."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    score_rows_path: str
    parent_artifacts: tuple[str, ...]
    benchmark: str
    problem_count: int
    completion_count: int
    claim_allowed: bool
    schema_version: str = EXECUTION_RERANK_EVAL_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "score_rows_path": self.score_rows_path,
            "parent_artifacts": list(self.parent_artifacts),
            "benchmark": self.benchmark,
            "problem_count": self.problem_count,
            "completion_count": self.completion_count,
            "claim_allowed": self.claim_allowed,
        }


def rerank_completions(
    *,
    completions: Sequence[ScoredCompletion],
    benchmark: str,
    bootstrap_seed: int = 17,
    bootstrap_samples: int = 2000,
    min_lift_for_claim: float = 3.0,
    confidence_level: float = 0.95,
    pass_at_k: int = 5,
    require_no_action_for_claim: bool = False,
    scoring_summary: dict[str, Any] | None = None,
) -> ExecutionRerankReport:
    """Compute pass@1 under each baseline and return the report.

    The scoring oracle is intentionally outside this function: the
    caller passes ``ScoredCompletion`` objects with whatever baseline
    scores it has computed (lexical similarity, no-action, CodeLeWM
    energy, etc.). The evaluator's only job is to rank, compute
    pass@1, and bootstrap the lift CI.
    """

    if not completions:
        raise ExecutionRerankError("completions sequence is empty")
    if pass_at_k < 1:
        raise ExecutionRerankError("pass_at_k must be >= 1")
    if bootstrap_samples < 1:
        raise ExecutionRerankError("bootstrap_samples must be >= 1")

    by_problem: dict[str, list[ScoredCompletion]] = {}
    for c in completions:
        by_problem.setdefault(c.label.problem_id, []).append(c)

    per_problem_counts = {len(v) for v in by_problem.values()}
    if len(per_problem_counts) != 1:
        raise ExecutionRerankError(
            f"every problem must have the same number of completions; got {per_problem_counts}"
        )
    n_per_problem = per_problem_counts.pop()

    seen_baselines: set[str] = set()
    for batch in by_problem.values():
        for sc in batch:
            seen_baselines.update(sc.scores.keys())
    if "codelewm" not in seen_baselines:
        raise ExecutionRerankError(
            "ScoredCompletion.scores must include a 'codelewm' baseline"
        )
    if "llm_order" not in seen_baselines and not all(
        sc.label.llm_order_rank > 0 for batch in by_problem.values() for sc in batch
    ):
        raise ExecutionRerankError(
            "CompletionLabel.llm_order_rank must be set (1-indexed) for every label"
        )

    summaries: dict[str, BaselineSummary] = {}
    pass_at_1_by_problem: dict[str, dict[str, bool]] = {
        pid: {} for pid in by_problem
    }
    for baseline in seen_baselines | {"llm_order"}:
        pass_count = 0
        pass_at_k_count = 0
        reciprocal_rank_total = 0.0
        for pid, batch in by_problem.items():
            ranked = _rank_batch(batch, baseline)
            top = ranked[0]
            passed = top.label.passed
            pass_at_1_by_problem[pid][baseline] = passed
            if passed:
                pass_count += 1
            top_k = ranked[: min(pass_at_k, len(ranked))]
            if any(item.label.passed for item in top_k):
                pass_at_k_count += 1
            first_pass_rank = next(
                (
                    rank
                    for rank, item in enumerate(ranked, start=1)
                    if item.label.passed
                ),
                None,
            )
            if first_pass_rank is not None:
                reciprocal_rank_total += 1.0 / first_pass_rank
        n_problems = len(by_problem)
        summaries[baseline] = BaselineSummary(
            baseline=baseline,
            pass_at_1=pass_count / n_problems if n_problems else 0.0,
            pass_count=pass_count,
            problem_count=n_problems,
            pass_at_k=pass_at_k_count / n_problems if n_problems else 0.0,
            pass_at_k_value=pass_at_k,
            mrr=reciprocal_rank_total / n_problems if n_problems else 0.0,
        )

    codelewm_summary = summaries["codelewm"]
    llm_summary = summaries["llm_order"]
    lift = (codelewm_summary.pass_at_1 - llm_summary.pass_at_1) * 100.0
    no_action_lift = None
    no_action_ci = None
    if "no_action" in summaries:
        no_action_summary = summaries["no_action"]
        no_action_lift = (
            codelewm_summary.pass_at_1 - no_action_summary.pass_at_1
        ) * 100.0

    ci_lo, ci_hi = _bootstrap_lift_ci(
        pass_at_1_by_problem,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        a="codelewm",
        b="llm_order",
    )
    if "no_action" in summaries:
        no_action_ci = _bootstrap_lift_ci(
            pass_at_1_by_problem,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            a="codelewm",
            b="no_action",
        )

    failure_reasons: list[str] = []
    if lift < min_lift_for_claim or ci_lo <= 0.0:
        failure_reasons.append(
            f"llm_order_lift={lift:.2f}pts ci=({ci_lo:.2f},{ci_hi:.2f}); "
            f"requires lift>={min_lift_for_claim} and ci_lo>0"
        )
    if require_no_action_for_claim:
        if no_action_lift is None or no_action_ci is None:
            failure_reasons.append("missing_no_action_baseline")
        elif no_action_lift < min_lift_for_claim or no_action_ci[0] <= 0.0:
            failure_reasons.append(
                f"no_action_lift={no_action_lift:.2f}pts "
                f"ci=({no_action_ci[0]:.2f},{no_action_ci[1]:.2f}); "
                f"requires lift>={min_lift_for_claim} and ci_lo>0"
            )
    claim_allowed = not failure_reasons
    claim_reason = (
        "lift_above_threshold_and_ci_excludes_zero"
        if claim_allowed
        else "; ".join(failure_reasons)
    )

    baseline_order = (
        "random",
        "lexical",
        "llm_order",
        "no_action",
        "shuffled_action",
        "codelewm",
    )
    ordered = tuple(
        summaries[b] for b in baseline_order if b in summaries
    ) + tuple(
        summaries[b]
        for b in sorted(seen_baselines | {"llm_order"})
        if b not in baseline_order
    )

    return ExecutionRerankReport(
        schema_version=EXECUTION_RERANK_REPORT_SCHEMA_VERSION,
        benchmark=benchmark,
        problem_count=len(by_problem),
        completions_per_problem=n_per_problem,
        baselines=ordered,
        codelewm_lift_over_llm_order=lift,
        bootstrap_lift_ci=(ci_lo, ci_hi),
        bootstrap_seed=bootstrap_seed,
        claim_allowed=claim_allowed,
        claim_reason=claim_reason,
        codelewm_lift_over_no_action=no_action_lift,
        bootstrap_lift_over_no_action_ci=no_action_ci,
        pass_at_k=pass_at_k,
        confidence_level=confidence_level,
        scoring_summary=scoring_summary or {},
    )


def _select_top(
    batch: list[ScoredCompletion], baseline: str
) -> ScoredCompletion:
    """Return the top-1 completion under ``baseline``.

    For ``llm_order`` the rank is read off the label (1-indexed).
    For ``random`` we pick the lexicographic-first completion_id;
    randomness is hashed across the seed in the bootstrap path
    instead so this function stays deterministic.
    """

    return _rank_batch(batch, baseline)[0]


def _rank_batch(batch: list[ScoredCompletion], baseline: str) -> list[ScoredCompletion]:
    if baseline == "llm_order":
        return sorted(batch, key=lambda c: (c.label.llm_order_rank, c.label.completion_id))
    return sorted(
        batch,
        key=lambda c: (
            -c.scores.get(baseline, float("-inf")),
            c.label.completion_id,
        ),
    )


def _bootstrap_lift_ci(
    pass_per_problem: dict[str, dict[str, bool]],
    *,
    seed: int,
    samples: int,
    confidence_level: float,
    a: str,
    b: str,
) -> tuple[float, float]:
    if not pass_per_problem:
        return 0.0, 0.0
    problem_ids = sorted(pass_per_problem)
    rng = random.Random(seed)
    n = len(problem_ids)
    lifts: list[float] = []
    for _ in range(samples):
        bootstrap = rng.choices(problem_ids, k=n)
        a_count = sum(1 for pid in bootstrap if pass_per_problem[pid].get(a, False))
        b_count = sum(1 for pid in bootstrap if pass_per_problem[pid].get(b, False))
        lifts.append((a_count - b_count) / n * 100.0)
    lifts.sort()
    alpha = (1.0 - confidence_level) / 2.0
    lo_idx = max(0, int(samples * alpha))
    hi_idx = min(samples - 1, int(samples * (1.0 - alpha)))
    return lifts[lo_idx], lifts[hi_idx]


def load_completion_labels(
    path: Path, *, benchmark_id: str, strict: bool = False
) -> tuple[CompletionLabel, ...]:
    """Load completion labels from a JSONL artifact.

    Each line is::

        {
          "benchmark_id": "humaneval" | "mbpp_plus" | ...,
          "problem_id": "HumanEval/0",
          "completion_id": "HumanEval/0::0",
          "code": "def f(): ...",
          "llm_order_rank": 1,
          "passed": true
        }

    Newer operator-generated artifacts may instead use
    :data:`COMPLETION_LABEL_SCHEMA_VERSION` and store the candidate text
    under ``completion_text`` with a string ``label`` of ``"pass"`` or
    ``"fail"``. The compatibility fields above remain accepted so older
    checked-in fixtures and reports continue to load.

    The function filters to ``benchmark_id`` and returns a tuple of
    :class:`CompletionLabel`. Records that fail validation are dropped by
    default for older smoke-fixture compatibility. Pass ``strict=True`` for
    manifest-backed evaluator runs, where malformed rows should be explicit
    failures.
    """

    out: list[CompletionLabel] = []
    normalized_benchmark = _normalize_benchmark_id(benchmark_id)
    if not path.is_file():
        raise ExecutionRerankError(f"completion labels file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ExecutionRerankError(
                    f"{path}:{line_no}: invalid JSON"
                ) from exc
            row_benchmark = row.get("benchmark_id")
            if (
                not isinstance(row_benchmark, str)
                or _normalize_benchmark_id(row_benchmark) != normalized_benchmark
            ):
                continue
            try:
                code = row.get("code")
                if not isinstance(code, str):
                    code = row["completion_text"]
                passed = row.get("passed")
                if not isinstance(passed, bool):
                    passed = _passed_from_label(row["label"])
                out.append(
                    CompletionLabel(
                        problem_id=str(row["problem_id"]),
                        completion_id=str(row["completion_id"]),
                        code=str(code),
                        llm_order_rank=int(row["llm_order_rank"]),
                        passed=bool(passed),
                        scoring_inputs=tuple(
                            CompletionScoringInput.from_mapping(item)
                            for item in row.get("scoring_inputs", ())
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                if strict:
                    raise ExecutionRerankError(
                        f"{path}:{line_no}: invalid completion label row"
                    ) from exc
                continue
    if not out:
        raise ExecutionRerankError(
            f"no valid completion labels for benchmark_id={benchmark_id!r} in {path}"
        )
    return tuple(out)


def _passed_from_label(label: object) -> bool:
    if label == "pass":
        return True
    if label == "fail":
        return False
    raise ValueError(f"unsupported completion label: {label!r}")


def _normalize_benchmark_id(value: str) -> str:
    return value.replace("-", "_").lower()


class ExecutionRerankEvalError(ValueError):
    """Raised when completion-label reranking cannot run."""


def run_execution_rerank_evaluation(
    *,
    completion_manifest: Path | str,
    checkpoint: Path | str,
    out: Path | str,
    benchmark: str | None = None,
    labels_path: Path | str | None = None,
    checkpoint_manifest: Path | str | None = None,
    device: str = "auto",
    index: Path | str | None = None,
    retrieval_prior_weight: float = 0.0,
    retrieval_prior_k: int = 10,
    allow_unsafe_checkpoint: bool = False,
    require_learned_scorer: bool = False,
    pass_at_k: int = 5,
    bootstrap_samples: int = 2000,
    seed: int = 17,
    min_lift_for_claim: float = 3.0,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "rerank-completions"),
) -> ExecutionRerankEvalResult:
    """Score completion-label artifacts and write a downstream rerank report."""

    output_dir = Path(out).resolve()
    report_path = output_dir / "reports" / "execution_rerank_report.json"
    score_rows_path = output_dir / "reports" / "completion_scores.jsonl"
    config_path = output_dir / "config.json"
    secret_scan_path = output_dir / "reports" / "secret_scan_report.json"
    manifest_path = output_dir / "manifest.json"
    _reject_existing_output(
        (report_path, score_rows_path, config_path, secret_scan_path, manifest_path),
        overwrite=overwrite,
        output_dir=output_dir,
    )

    completion_manifest_path = Path(completion_manifest)
    completion_artifact = read_artifact_manifest(completion_manifest_path)
    validate_artifact_checksums(completion_artifact, root=completion_manifest_path.parent)
    benchmark_id = _normalize_benchmark_id(
        benchmark
        or str(completion_artifact.metadata.get("benchmark_id") or "")
    )
    if benchmark_id not in {"humaneval", "mbpp_plus"}:
        raise ExecutionRerankEvalError(
            "benchmark must be humaneval, mbpp-plus, or present in the completion manifest"
        )
    label_file = _resolve_completion_label_path(
        completion_artifact,
        manifest_root=completion_manifest_path.parent,
        labels_path=labels_path,
    )
    labels = load_completion_labels(label_file, benchmark_id=benchmark_id, strict=True)
    missing_inputs = [label.completion_id for label in labels if not label.scoring_inputs]
    if missing_inputs:
        raise ExecutionRerankEvalError(
            "completion labels are missing scoring_inputs; regenerate labels with "
            "the current sampler or provide a compatible artifact"
        )
    prompts = _load_sampling_prompts(completion_manifest_path.parent)
    scorer = load_scorer(
        checkpoint,
        device=device,
        checkpoint_manifest=checkpoint_manifest,
        index=index,
        retrieval_prior_weight=retrieval_prior_weight,
        retrieval_prior_k=retrieval_prior_k,
        allow_unsafe=allow_unsafe_checkpoint,
        require_learned_backend=require_learned_scorer,
    )
    scored, score_rows = _score_completion_labels(
        labels,
        prompts=prompts,
        scorer=scorer,
    )
    report = rerank_completions(
        completions=scored,
        benchmark=benchmark_id,
        bootstrap_seed=seed,
        bootstrap_samples=bootstrap_samples,
        min_lift_for_claim=min_lift_for_claim,
        pass_at_k=pass_at_k,
        require_no_action_for_claim=True,
        scoring_summary={
            "completion_label_artifact_id": completion_artifact.artifact_id,
            "completion_label_path": str(label_file),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": scorer.checkpoint_sha256,
            "model_id": scorer.model_id,
            "score_direction": "higher_is_better_after_negating_energy",
            "problem_count": len({label.problem_id for label in labels}),
            "completion_count": len(labels),
            "score_error_count": sum(1 for row in score_rows if row.get("errors")),
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(score_rows_path, score_rows)
    config_payload = {
        "completion_manifest": str(completion_manifest_path),
        "labels_path": str(label_file),
        "benchmark": benchmark_id,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": None if checkpoint_manifest is None else str(checkpoint_manifest),
        "device": device,
        "index": None if index is None else str(index),
        "retrieval_prior_weight": retrieval_prior_weight,
        "retrieval_prior_k": retrieval_prior_k,
        "allow_unsafe_checkpoint": allow_unsafe_checkpoint,
        "require_learned_scorer": require_learned_scorer,
        "pass_at_k": pass_at_k,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "min_lift_for_claim": min_lift_for_claim,
    }
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scan = scan_paths(
        (report_path, score_rows_path, config_path),
        include_suffixes=(),
        recursive=False,
    )
    secret_scan_path.write_text(
        json.dumps(scan.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    parent_artifacts = (completion_artifact.artifact_id,)
    artifact_manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=output_dir,
        files=(report_path, score_rows_path, config_path, secret_scan_path),
        command=command,
        config=config_payload,
        parent_artifacts=parent_artifacts,
        metadata={
            "schema_version": EXECUTION_RERANK_EVAL_RUN_SCHEMA_VERSION,
            "report_schema_version": EXECUTION_RERANK_REPORT_SCHEMA_VERSION,
            "score_schema_version": COMPLETION_SCORE_SCHEMA_VERSION,
            "benchmark": benchmark_id,
            "problem_count": report.problem_count,
            "completion_count": len(labels),
            "claim_allowed": report.claim_allowed,
        },
    )
    write_artifact_manifest(artifact_manifest, manifest_path)
    return ExecutionRerankEvalResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path=_relative_to_root(manifest_path, output_dir),
        report_path=_relative_to_root(report_path, output_dir),
        score_rows_path=_relative_to_root(score_rows_path, output_dir),
        parent_artifacts=parent_artifacts,
        benchmark=benchmark_id,
        problem_count=report.problem_count,
        completion_count=len(labels),
        claim_allowed=report.claim_allowed,
    )


def _completion_labels_manifest_file(manifest: ArtifactManifest) -> str:
    metadata_path = manifest.metadata.get("labels_path")
    if isinstance(metadata_path, str) and metadata_path:
        return metadata_path
    candidates = [
        file.path
        for file in manifest.files
        if file.path.endswith("_completion_labels.jsonl")
    ]
    if len(candidates) != 1:
        raise ExecutionRerankEvalError(
            "completion manifest must list exactly one *_completion_labels.jsonl file"
        )
    return candidates[0]


def _resolve_completion_label_path(
    manifest: ArtifactManifest,
    *,
    manifest_root: Path,
    labels_path: Path | str | None,
) -> Path:
    manifest_label_path = _completion_labels_manifest_file(manifest)
    manifest_entry = next(
        (file for file in manifest.files if file.path == manifest_label_path),
        None,
    )
    if manifest_entry is None:
        raise ExecutionRerankEvalError(
            f"completion manifest does not list label file: {manifest_label_path}"
        )
    if labels_path is None:
        return manifest_root / manifest_label_path
    override = Path(labels_path)
    if not override.is_file():
        raise ExecutionRerankEvalError(f"labels file does not exist: {override}")
    observed_bytes = override.stat().st_size
    observed_sha256 = sha256_file(override)
    if (
        observed_bytes != manifest_entry.bytes
        or observed_sha256 != manifest_entry.sha256
    ):
        raise ExecutionRerankEvalError(
            "--labels must point to the manifest-listed completion labels file "
            "or a byte-identical copy"
        )
    return override


def _load_sampling_prompts(root: Path) -> dict[str, str]:
    path = root / "prompts" / "sampling_prompts.jsonl"
    prompts: dict[str, str] = {}
    if not path.is_file():
        return prompts
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            problem_id = row.get("problem_id")
            prompt_text = row.get("prompt_text")
            if isinstance(problem_id, str) and isinstance(prompt_text, str):
                prompts[problem_id] = prompt_text
    return prompts


def _score_completion_labels(
    labels: Sequence[CompletionLabel],
    *,
    prompts: dict[str, str],
    scorer: Any,
) -> tuple[tuple[ScoredCompletion, ...], list[dict[str, Any]]]:
    by_problem: dict[str, list[CompletionLabel]] = {}
    for label in labels:
        by_problem.setdefault(label.problem_id, []).append(label)
    shuffled_inputs = _shuffled_inputs_by_problem(by_problem)
    scored: list[ScoredCompletion] = []
    rows: list[dict[str, Any]] = []
    for label in labels:
        scores: dict[str, float] = {
            "llm_order": -float(label.llm_order_rank),
            "random": _deterministic_unit_score(label.completion_id),
            "lexical": _lexical_similarity(prompts.get(label.problem_id, label.problem_id), label.code),
        }
        errors: dict[str, str] = {}
        for baseline, inputs in (
            ("codelewm", label.scoring_inputs),
            ("no_action", (CompletionScoringInput("no_action", "<NO_ACTION>"),)),
            ("shuffled_action", shuffled_inputs[label.problem_id]),
        ):
            try:
                scores[baseline] = -_mean_score_for_inputs(
                    scorer,
                    label=label,
                    inputs=inputs,
                    baseline=baseline,
                )
            except ScoreError as exc:
                scores[baseline] = -1.0e12
                errors[baseline] = str(exc)
        scored.append(ScoredCompletion(label=label, scores=scores))
        rows.append(
            {
                "schema_version": COMPLETION_SCORE_SCHEMA_VERSION,
                "problem_id": label.problem_id,
                "completion_id": label.completion_id,
                "passed": label.passed,
                "llm_order_rank": label.llm_order_rank,
                "scores": scores,
                "errors": errors,
            }
        )
    return tuple(scored), rows


def _mean_score_for_inputs(
    scorer: Any,
    *,
    label: CompletionLabel,
    inputs: Sequence[CompletionScoringInput],
    baseline: str,
) -> float:
    if not inputs:
        raise ScoreError(f"{baseline} requires at least one scoring input")
    scores = [
        scorer.score_texts(
            before="",
            instruction=input_case.input_repr,
            candidate=label.code,
            candidate_name=f"{label.completion_id}::{baseline}::{input_case.input_id}",
        ).final_score
        for input_case in inputs
    ]
    value = sum(scores) / len(scores)
    if not math.isfinite(value):
        raise ScoreError(f"{baseline} score must be finite")
    return value


def _shuffled_inputs_by_problem(
    by_problem: dict[str, list[CompletionLabel]],
) -> dict[str, tuple[CompletionScoringInput, ...]]:
    problem_ids = sorted(by_problem)
    first_inputs = {
        problem_id: by_problem[problem_id][0].scoring_inputs
        for problem_id in problem_ids
    }
    if len(problem_ids) == 1:
        return {problem_ids[0]: first_inputs[problem_ids[0]]}
    shuffled: dict[str, tuple[CompletionScoringInput, ...]] = {}
    for index, problem_id in enumerate(problem_ids):
        other_problem = problem_ids[(index + 1) % len(problem_ids)]
        shuffled[problem_id] = first_inputs[other_problem]
    return shuffled


def _deterministic_unit_score(value: str) -> float:
    digest = value.encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(digest).digest()[:8], "big")
    return bucket / float(2**64 - 1)


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = set(left.lower().split())
    right_tokens = set(right.lower().split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _reject_existing_output(
    paths: Sequence[Path],
    *,
    overwrite: bool,
    output_dir: Path,
) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        rel = ", ".join(_relative_to_root(path, output_dir) for path in existing)
        raise ExecutionRerankEvalError(
            f"output already exists; pass --overwrite to replace: {rel}"
        )


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
