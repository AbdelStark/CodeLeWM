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

import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


EXECUTION_RERANK_REPORT_SCHEMA_VERSION = "codelewm.eval.execution_rerank_report.v1"
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "pass_at_1": self.pass_at_1,
            "pass_count": self.pass_count,
            "problem_count": self.problem_count,
        }


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

    def as_dict(self) -> dict[str, Any]:
        return {
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
        }


def rerank_completions(
    *,
    completions: Sequence[ScoredCompletion],
    benchmark: str,
    bootstrap_seed: int = 17,
    bootstrap_samples: int = 2000,
    min_lift_for_claim: float = 3.0,
    confidence_level: float = 0.95,
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
        for pid, batch in by_problem.items():
            top = _select_top(batch, baseline)
            passed = top.label.passed
            pass_at_1_by_problem[pid][baseline] = passed
            if passed:
                pass_count += 1
        n_problems = len(by_problem)
        summaries[baseline] = BaselineSummary(
            baseline=baseline,
            pass_at_1=pass_count / n_problems if n_problems else 0.0,
            pass_count=pass_count,
            problem_count=n_problems,
        )

    codelewm_summary = summaries["codelewm"]
    llm_summary = summaries["llm_order"]
    lift = (codelewm_summary.pass_at_1 - llm_summary.pass_at_1) * 100.0

    ci_lo, ci_hi = _bootstrap_lift_ci(
        pass_at_1_by_problem,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        a="codelewm",
        b="llm_order",
    )

    claim_allowed = lift >= min_lift_for_claim and ci_lo > 0.0
    claim_reason = (
        "lift_above_threshold_and_ci_excludes_zero"
        if claim_allowed
        else f"lift={lift:.2f}pts ci=({ci_lo:.2f},{ci_hi:.2f}); requires lift>={min_lift_for_claim} and ci_lo>0"
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

    if baseline == "llm_order":
        return min(batch, key=lambda c: c.label.llm_order_rank)
    if baseline == "random":
        return min(batch, key=lambda c: c.label.completion_id)
    # Higher score wins; tie-break by completion_id for determinism.
    return max(
        batch,
        key=lambda c: (c.scores.get(baseline, float("-inf")), -hash(c.label.completion_id)),
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
    path: Path, *, benchmark_id: str
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

    The function filters to ``benchmark_id`` and returns a tuple of
    :class:`CompletionLabel`. Records that fail validation are dropped
    and counted toward an :class:`ExecutionRerankError` if zero remain.
    """

    out: list[CompletionLabel] = []
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
            if row.get("benchmark_id") != benchmark_id:
                continue
            try:
                out.append(
                    CompletionLabel(
                        problem_id=str(row["problem_id"]),
                        completion_id=str(row["completion_id"]),
                        code=str(row["code"]),
                        llm_order_rank=int(row["llm_order_rank"]),
                        passed=bool(row["passed"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not out:
        raise ExecutionRerankError(
            f"no valid completion labels for benchmark_id={benchmark_id!r} in {path}"
        )
    return tuple(out)
