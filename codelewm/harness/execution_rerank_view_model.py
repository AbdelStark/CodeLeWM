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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codelewm.observability.logging import redact_value


EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION = (
    "codelewm.harness.execution_rerank_view_model.v1"
)
SCORE_DIRECTION_HIGHER_IS_BETTER = "higher_is_better"

# Diagnostic slots are always present so that a missing slot stays explicit in
# both the TUI and the web report rather than silently disappearing.
EXECUTION_RERANK_DIAGNOSTIC_SLOTS = ("retrieval_evidence", "checkpoint", "sandbox")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


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
    no_action_panel: dict[str, Any]
    diagnostics: dict[str, Any]
    artifact_lineage: dict[str, Any]
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
            "no_action_panel": dict(self.no_action_panel),
            "diagnostics": dict(self.diagnostics),
            "artifact_lineage": dict(self.artifact_lineage),
            "score_direction": self.score_direction,
            "notes": list(self.notes),
        }


def build_execution_rerank_view_model(
    *,
    rerank_report: Mapping[str, Any],
    scenario_id: str,
    completion_records: Sequence[Mapping[str, Any]],
    code_preview_chars: int = 240,
    diagnostics: Mapping[str, Any] | None = None,
    artifact_lineage: Mapping[str, Any] | None = None,
) -> ExecutionRerankViewModel:
    """Build the view model from a rerank report + per-completion records.

    ``completion_records`` are the per-completion details: ``code``,
    ``scores`` per baseline, hidden ``test_results``, and a
    ``predicted_output_latent`` block. The contract is intentionally
    loose so the operator's offline-labeling pipeline can supply
    whatever extra metadata it wants without breaking the view model.

    ``diagnostics`` and ``artifact_lineage`` are optional presentation
    blocks supplied by the demo runner. They feed the same TUI and web
    report, keeping the two surfaces in lockstep on one schema-versioned
    contract. Missing diagnostic slots stay explicit (``not_recorded``)
    rather than vanishing.
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
    no_action_summary = next(
        (b for b in baselines if b["baseline"] == "no_action"), None
    )

    codelewm_pass_at_1 = codelewm_summary["pass_at_1"] if codelewm_summary else 0.0
    no_action_pass_at_1 = (
        no_action_summary["pass_at_1"] if no_action_summary else None
    )
    headline_panel = {
        "codelewm_pass_at_1": codelewm_pass_at_1,
        "llm_order_pass_at_1": llm_summary["pass_at_1"] if llm_summary else 0.0,
        "no_action_pass_at_1": no_action_pass_at_1,
        "pass_at_1_lift": lift,
        "bootstrap_lift_ci_lo": ci[0],
        "bootstrap_lift_ci_hi": ci[1],
        "claim_allowed": claim_allowed,
        "claim_reason": claim_reason,
    }

    no_action_panel = _no_action_panel(
        rerank_report=rerank_report,
        codelewm_pass_at_1=codelewm_pass_at_1,
        no_action_pass_at_1=no_action_pass_at_1,
    )
    diagnostics_view = _diagnostics_view(diagnostics)
    artifact_lineage_view = _artifact_lineage_view(artifact_lineage)

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
        no_action_panel=no_action_panel,
        diagnostics=diagnostics_view,
        artifact_lineage=artifact_lineage_view,
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


def _no_action_panel(
    *,
    rerank_report: Mapping[str, Any],
    codelewm_pass_at_1: float,
    no_action_pass_at_1: float | None,
) -> dict[str, Any]:
    """Explicit CodeLeWM-vs-no-action comparison (higher pass@1 is better)."""

    lift_over_no_action = rerank_report.get("codelewm_lift_over_no_action")
    ci_raw = rerank_report.get("bootstrap_lift_over_no_action_ci")
    ci: list[float] | None = None
    if isinstance(ci_raw, (list, tuple)) and len(ci_raw) == 2:
        ci = [float(ci_raw[0]), float(ci_raw[1])]
    lift = None if lift_over_no_action is None else float(lift_over_no_action)
    if no_action_pass_at_1 is None:
        return {
            "status": "not_recorded",
            "no_action_pass_at_1": None,
            "codelewm_pass_at_1": float(codelewm_pass_at_1),
            "codelewm_minus_no_action": None,
            "codelewm_lift_over_no_action": lift,
            "bootstrap_lift_over_no_action_ci": ci,
            "interpretation": "not_recorded",
        }
    delta = float(codelewm_pass_at_1) - float(no_action_pass_at_1)
    return {
        "status": "available",
        "no_action_pass_at_1": float(no_action_pass_at_1),
        "codelewm_pass_at_1": float(codelewm_pass_at_1),
        "codelewm_minus_no_action": delta,
        "codelewm_lift_over_no_action": lift,
        "bootstrap_lift_over_no_action_ci": ci,
        "interpretation": _interpret_higher_is_better(delta),
    }


def _interpret_higher_is_better(delta: float) -> str:
    if delta > 0:
        return "better_than_no_action"
    if delta < 0:
        return "worse_than_no_action"
    return "tied_with_no_action"


def _diagnostics_view(diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize diagnostic slots so a missing slot stays explicit."""

    provided = diagnostics if isinstance(diagnostics, Mapping) else {}
    view: dict[str, Any] = {}
    for name in EXECUTION_RERANK_DIAGNOSTIC_SLOTS:
        view[name] = _diagnostic_slot(provided.get(name))
    # Preserve any extra operator-supplied slots verbatim, still explicit.
    for name, slot in provided.items():
        key = str(name)
        if key in view:
            continue
        view[key] = _diagnostic_slot(slot)
    return view


def _diagnostic_slot(slot: Any) -> dict[str, Any]:
    if isinstance(slot, Mapping) and slot:
        normalized = dict(slot)
        normalized.setdefault("status", "available")
        return normalized
    return {"status": "not_recorded"}


def ordered_diagnostic_slot_names(diagnostics: Mapping[str, Any]) -> list[str]:
    """Canonical diagnostic-slot order independent of dict key order.

    Both renderers (web report and TUI) use this so the slots appear in the
    same order regardless of whether the view model came from memory or from a
    ``sort_keys=True`` on-disk artifact, keeping the two surfaces in lockstep.
    """

    known = [name for name in EXECUTION_RERANK_DIAGNOSTIC_SLOTS if name in diagnostics]
    extras = sorted(
        str(name)
        for name in diagnostics
        if name not in EXECUTION_RERANK_DIAGNOSTIC_SLOTS
    )
    return known + extras


def _artifact_lineage_view(
    artifact_lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Deterministic provenance pointers shared by both surfaces.

    The artifact manifest id is intentionally absent: the manifest hashes
    this view model, so it cannot reference its own descendant. Lineage
    therefore records the upstream candidate-pack manifest ids, the demo
    command, and the run-relative artifact paths.
    """

    provided = artifact_lineage if isinstance(artifact_lineage, Mapping) else {}
    parents = provided.get("parent_artifact_ids")
    command = provided.get("command")
    return {
        "parent_artifact_ids": [str(item) for item in parents or ()],
        "command": [str(item) for item in command or ()],
        "manifest_path": _optional_str(provided.get("manifest_path")),
        "report_path": _optional_str(provided.get("report_path")),
        "view_model_path": _optional_str(provided.get("view_model_path")),
        "html_path": _optional_str(provided.get("html_path")),
        "asciicast_path": _optional_str(provided.get("asciicast_path")),
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def validate_execution_rerank_view_model_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an execution-rerank view-model payload; return its JSON dict."""

    if not isinstance(payload, Mapping):
        raise ExecutionRerankViewModelError(
            "execution rerank view model must be a JSON object"
        )
    if payload.get("schema_version") != EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION:
        raise ExecutionRerankViewModelError(
            "unsupported execution rerank view model schema_version; "
            f"expected {EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION!r}"
        )
    required = {
        "schema_version",
        "scenario_id",
        "benchmark_id",
        "problem_count",
        "completions_per_problem",
        "baselines",
        "pass_at_1_lift",
        "bootstrap_lift_ci",
        "claim_allowed",
        "claim_reason",
        "completion_panels",
        "headline_panel",
        "no_action_panel",
        "diagnostics",
        "artifact_lineage",
        "score_direction",
        "notes",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ExecutionRerankViewModelError(
            "execution rerank view model missing required key(s): "
            f"{', '.join(missing)}"
        )
    normalized = dict(redact_value(payload))
    _reject_ansi(normalized)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ExecutionRerankViewModelError(
            "execution rerank view model must be JSON-native"
        ) from exc
    return normalized


def write_execution_rerank_view_model(
    payload: Mapping[str, Any], path: Path | str
) -> None:
    """Write a validated execution-rerank view-model JSON artifact."""

    normalized = validate_execution_rerank_view_model_payload(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_execution_rerank_view_model(path: Path | str) -> Mapping[str, Any]:
    """Read and validate an execution-rerank view-model JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ExecutionRerankViewModelError(
            "execution rerank view model must be a JSON object"
        )
    return validate_execution_rerank_view_model_payload(payload)


def _reject_ansi(value: Any) -> None:
    if isinstance(value, str):
        if _ANSI_RE.search(value):
            raise ExecutionRerankViewModelError(
                "execution rerank view model must not contain ANSI escape codes"
            )
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_ansi(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_ansi(item)


def to_json(model: ExecutionRerankViewModel) -> str:
    """Serialize the view model to JSON with sorted keys for snapshot tests."""

    return json.dumps(model.as_dict(), indent=2, sort_keys=True)
