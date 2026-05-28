"""Visual view model for the execution-substrate rerank demo.

The existing :mod:`codelewm.harness.visual_view_model` renders the
commit-edit demo (patch hunks, edit metadata, energy bars). The
execution substrate needs a different set of panels:

- predicted output latent stats per completion
- hidden-test execution matrix per completion
- ranking comparison across baselines
- pass@1 lift summary with bootstrap CI
- claim-gate banner
- scorer trace per completion

This module is the JSON-shape generator that the existing HTML and
terminal renderers (and a future TUI panel) consume. Renderers can
ship in follow-on PRs without touching this contract.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION = (
    "codelewm.harness.execution_rerank_view_model.v1"
)
SCORE_DIRECTION_HIGHER_IS_BETTER = "higher_is_better"


class ExecutionRerankViewModelError(ValueError):
    """Raised when the inputs to the view model are malformed."""


@dataclass(frozen=True)
class CompletionPanelEntry:
    """One completion's panel data."""

    completion_id: str
    code_preview: str
    llm_order_rank: int
    passed: bool
    scores: dict[str, float]
    codelewm_rank: int
    rank_by_baseline: dict[str, int]
    test_results: tuple[dict[str, Any], ...]
    predicted_output_latent: dict[str, Any]


@dataclass(frozen=True)
class ExecutionRerankViewModel:
    """The full execution-rerank panel set, ready for renderers."""

    schema_version: str
    scenario_id: str
    benchmark_id: str
    problem_count: int
    completions_per_problem: int
    baselines: tuple[dict[str, Any], ...]
    pass_at_1_lift: float
    bootstrap_lift_ci: tuple[float, float]
    claim_allowed: bool
    claim_reason: str
    completion_panels: tuple[CompletionPanelEntry, ...]
    headline_panel: dict[str, Any]
    score_direction: str
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "benchmark_id": self.benchmark_id,
            "problem_count": self.problem_count,
            "completions_per_problem": self.completions_per_problem,
            "baselines": list(self.baselines),
            "pass_at_1_lift": self.pass_at_1_lift,
            "bootstrap_lift_ci": list(self.bootstrap_lift_ci),
            "claim_allowed": self.claim_allowed,
            "claim_reason": self.claim_reason,
            "completion_panels": [
                {
                    "completion_id": c.completion_id,
                    "code_preview": c.code_preview,
                    "llm_order_rank": c.llm_order_rank,
                    "passed": c.passed,
                    "scores": dict(c.scores),
                    "codelewm_rank": c.codelewm_rank,
                    "rank_by_baseline": dict(c.rank_by_baseline),
                    "test_results": list(c.test_results),
                    "predicted_output_latent": dict(c.predicted_output_latent),
                }
                for c in self.completion_panels
            ],
            "headline_panel": dict(self.headline_panel),
            "score_direction": self.score_direction,
            "notes": list(self.notes),
        }


def build_execution_rerank_view_model(
    *,
    rerank_report: Mapping[str, Any],
    scenario_id: str,
    completion_records: Sequence[Mapping[str, Any]],
    code_preview_chars: int = 240,
) -> ExecutionRerankViewModel:
    """Build the view model from a rerank report + per-completion records.

    ``completion_records`` are the per-completion details: ``code``,
    ``scores`` per baseline, hidden ``test_results``, and a
    ``predicted_output_latent`` block. The contract is intentionally
    loose so the operator's offline-labeling pipeline can supply
    whatever extra metadata it wants without breaking the view model.
    """

    _require_keys(
        rerank_report,
        (
            "schema_version",
            "benchmark",
            "problem_count",
            "completions_per_problem",
            "baselines",
            "codelewm_lift_over_llm_order",
            "bootstrap_lift_ci",
            "claim_allowed",
            "claim_reason",
        ),
        "rerank_report",
    )
    if not isinstance(completion_records, (list, tuple)):
        raise ExecutionRerankViewModelError(
            f"completion_records must be a sequence, got {type(completion_records).__name__}"
        )

    baselines = tuple(
        {
            "baseline": b.get("baseline"),
            "pass_at_1": float(b.get("pass_at_1", 0.0)),
            "pass_count": int(b.get("pass_count", 0)),
            "problem_count": int(b.get("problem_count", 0)),
        }
        for b in rerank_report.get("baselines") or ()
    )

    panels: list[CompletionPanelEntry] = []
    sorted_by_codelewm = sorted(
        enumerate(completion_records),
        key=lambda pair: -float((pair[1].get("scores") or {}).get("codelewm", 0.0)),
    )
    rank_by_index: dict[int, int] = {
        idx: rank + 1 for rank, (idx, _) in enumerate(sorted_by_codelewm)
    }
    sorted_by_lexical = sorted(
        enumerate(completion_records),
        key=lambda pair: -float((pair[1].get("scores") or {}).get("lexical", 0.0)),
    )
    lex_rank: dict[int, int] = {
        idx: rank + 1 for rank, (idx, _) in enumerate(sorted_by_lexical)
    }

    for idx, record in enumerate(completion_records):
        scores = {k: float(v) for k, v in (record.get("scores") or {}).items()}
        code = str(record.get("code", ""))
        preview = code[:code_preview_chars]
        if len(code) > code_preview_chars:
            preview += "..."
        panels.append(
            CompletionPanelEntry(
                completion_id=str(record.get("completion_id", f"c{idx}")),
                code_preview=preview,
                llm_order_rank=int(record.get("llm_order_rank", idx + 1)),
                passed=bool(record.get("passed", False)),
                scores=scores,
                codelewm_rank=rank_by_index[idx],
                rank_by_baseline={
                    "llm_order": int(record.get("llm_order_rank", idx + 1)),
                    "lexical": lex_rank[idx],
                    "codelewm": rank_by_index[idx],
                },
                test_results=tuple(record.get("test_results") or ()),
                predicted_output_latent=dict(
                    record.get("predicted_output_latent") or {}
                ),
            )
        )

    # Sort panels by codelewm rank so the renderer sees them in
    # CodeLeWM's chosen order; the rank_by_baseline dict preserves
    # comparison.
    panels.sort(key=lambda c: c.codelewm_rank)

    lift = float(rerank_report.get("codelewm_lift_over_llm_order", 0.0))
    ci_raw = rerank_report.get("bootstrap_lift_ci") or [0.0, 0.0]
    ci = (float(ci_raw[0]), float(ci_raw[1]))
    claim_allowed = bool(rerank_report.get("claim_allowed"))
    claim_reason = str(rerank_report.get("claim_reason", ""))

    codelewm_summary = next(
        (b for b in baselines if b["baseline"] == "codelewm"), None
    )
    llm_summary = next(
        (b for b in baselines if b["baseline"] == "llm_order"), None
    )

    headline_panel = {
        "codelewm_pass_at_1": codelewm_summary["pass_at_1"] if codelewm_summary else 0.0,
        "llm_order_pass_at_1": llm_summary["pass_at_1"] if llm_summary else 0.0,
        "pass_at_1_lift": lift,
        "bootstrap_lift_ci_lo": ci[0],
        "bootstrap_lift_ci_hi": ci[1],
        "claim_allowed": claim_allowed,
        "claim_reason": claim_reason,
    }

    notes: list[str] = []
    if not claim_allowed:
        notes.append(
            "Claim gate not satisfied — the report's claim_allowed=false. "
            "Public language must scope to workflow evidence only."
        )
    if not panels:
        notes.append(
            "No completion records were supplied; the panel is empty."
        )

    return ExecutionRerankViewModel(
        schema_version=EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION,
        scenario_id=scenario_id,
        benchmark_id=str(rerank_report.get("benchmark", "")),
        problem_count=int(rerank_report.get("problem_count", 0)),
        completions_per_problem=int(
            rerank_report.get("completions_per_problem", 0)
        ),
        baselines=baselines,
        pass_at_1_lift=lift,
        bootstrap_lift_ci=ci,
        claim_allowed=claim_allowed,
        claim_reason=claim_reason,
        completion_panels=tuple(panels),
        headline_panel=headline_panel,
        score_direction=SCORE_DIRECTION_HIGHER_IS_BETTER,
        notes=tuple(notes),
    )


def _require_keys(
    payload: Mapping[str, Any], keys: Sequence[str], where: str
) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ExecutionRerankViewModelError(
            f"{where} is missing required key(s): {missing}"
        )


def to_json(model: ExecutionRerankViewModel) -> str:
    """Serialize the view model to JSON with sorted keys for snapshot tests."""

    return json.dumps(model.as_dict(), indent=2, sort_keys=True)
