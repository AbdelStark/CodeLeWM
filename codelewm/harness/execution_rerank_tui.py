"""Optional Textual TUI for the v0.6 execution-rerank showcase.

This is the interactive counterpart to the execution-rerank web report.
Both surfaces read one schema-versioned view model
(:mod:`codelewm.harness.execution_rerank_view_model`) so they stay in
lockstep. Textual and Rich are imported lazily so the base package and
its non-interactive CLI work without the optional ``tui`` dependency
group, mirroring :mod:`codelewm.harness.demo_tui`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from codelewm.observability.logging import redact_value

from .execution_rerank_view_model import (
    ExecutionRerankViewModelError,
    ordered_diagnostic_slot_names,
    read_execution_rerank_view_model,
    validate_execution_rerank_view_model_payload,
)


EXECUTION_RERANK_TUI_SNAPSHOT_SCHEMA_VERSION = (
    "codelewm.harness.execution_rerank_tui_snapshot.v1"
)
TUI_INSTALL_HINT = "uv sync --group dev --group tui"


class ExecutionRerankTuiError(ValueError):
    """Raised when the optional execution-rerank TUI cannot be loaded."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "config_error",
        remediation: str = "inspect the execution-rerank view model and retry",
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.remediation = remediation


def resolve_execution_rerank_tui_view_model_path(
    *,
    view_model: str | Path | None = None,
    demo_dir: str | Path | None = None,
) -> Path:
    """Resolve the execution-rerank view-model path for a demo run."""

    if view_model is not None and demo_dir is not None:
        raise ExecutionRerankTuiError("pass either view_model or demo_dir, not both")
    if view_model is None and demo_dir is None:
        raise ExecutionRerankTuiError(
            "pass an execution-rerank view-model path or demo artifact directory"
        )
    if view_model is not None:
        return Path(view_model)
    return Path(demo_dir or "") / "reports" / "execution_rerank_view_model.json"


def load_execution_rerank_tui_view_model(path: str | Path) -> Mapping[str, Any]:
    """Load the shared execution-rerank view model used by the TUI."""

    source = Path(path)
    if not source.is_file():
        raise ExecutionRerankTuiError(
            f"execution-rerank view model does not exist: {source}",
            remediation=(
                "run the execution-rerank tour first or pass --demo-dir/--view-model"
            ),
        )
    try:
        return read_execution_rerank_view_model(source)
    except (OSError, json.JSONDecodeError, ExecutionRerankViewModelError) as exc:
        raise ExecutionRerankTuiError(
            f"execution-rerank view model is invalid: {source}",
            remediation=(
                "rerun the tour and keep reports/execution_rerank_view_model.json intact"
            ),
        ) from exc


def build_execution_rerank_tui_snapshot(view_model: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic, JSON-native snapshot for TUI tests and views."""

    payload = validate_execution_rerank_view_model_payload(view_model)
    headline = _mapping(payload.get("headline_panel"))
    no_action = _mapping(payload.get("no_action_panel"))
    diagnostics = _mapping(payload.get("diagnostics"))
    lineage = _mapping(payload.get("artifact_lineage"))
    completions = [
        _completion_row(panel)
        for panel in sorted(
            (item for item in payload.get("completion_panels", ()) if isinstance(item, Mapping)),
            key=lambda item: (
                _rank_sort_key(item.get("codelewm_rank")),
                str(item.get("completion_id", "")),
            ),
        )
    ]
    snapshot = {
        "schema_version": EXECUTION_RERANK_TUI_SNAPSHOT_SCHEMA_VERSION,
        "headline": {
            "scenario_id": _safe(payload.get("scenario_id")),
            "benchmark_id": _safe(payload.get("benchmark_id")),
            "score_direction": _safe(payload.get("score_direction")),
            "problem_count": _safe(payload.get("problem_count")),
            "completions_per_problem": _safe(payload.get("completions_per_problem")),
            "codelewm_pass_at_1": _score_text(headline.get("codelewm_pass_at_1")),
            "llm_order_pass_at_1": _score_text(headline.get("llm_order_pass_at_1")),
            "no_action_pass_at_1": _score_text(headline.get("no_action_pass_at_1")),
            "pass_at_1_lift": _delta_text(payload.get("pass_at_1_lift")),
            "bootstrap_lift_ci": _ci_text(payload.get("bootstrap_lift_ci")),
        },
        "no_action": {
            "status": _safe(no_action.get("status")),
            "codelewm_pass_at_1": _score_text(no_action.get("codelewm_pass_at_1")),
            "no_action_pass_at_1": _score_text(no_action.get("no_action_pass_at_1")),
            "codelewm_minus_no_action": _delta_text(
                no_action.get("codelewm_minus_no_action")
            ),
            "codelewm_lift_over_no_action": _delta_text(
                no_action.get("codelewm_lift_over_no_action")
            ),
            "interpretation": _safe(no_action.get("interpretation")),
        },
        "completions": completions,
        "baselines": _baseline_rows(payload.get("baselines")),
        "diagnostics": _diagnostic_rows(diagnostics),
        "artifact_lineage": {
            "parent_artifact_ids": [
                _safe(item) for item in lineage.get("parent_artifact_ids", ())
            ],
            "command": " ".join(_safe(item) for item in lineage.get("command", ())),
            "manifest_path": _safe(lineage.get("manifest_path")),
            "report_path": _safe(lineage.get("report_path")),
            "view_model_path": _safe(lineage.get("view_model_path")),
            "html_path": _safe(lineage.get("html_path")),
            "asciicast_path": _safe(lineage.get("asciicast_path")),
        },
        "notes": [_safe(item) for item in payload.get("notes", ())],
        "claim_gate": dict(
            redact_value(
                {
                    "allowed": bool(payload.get("claim_allowed")),
                    "reason": payload.get("claim_reason"),
                }
            )
        ),
    }
    json.dumps(snapshot, sort_keys=True, allow_nan=False)
    return snapshot


def create_execution_rerank_tui_app(
    view_model: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> Any:
    """Create the Textual app for an already-loaded view model."""

    textual = _import_textual()
    snapshot = build_execution_rerank_tui_snapshot(view_model)
    app_cls = _make_app_class(textual)
    return app_cls(
        snapshot=snapshot,
        source_path=None if source_path is None else str(source_path),
    )


def run_execution_rerank_tui(path: str | Path) -> int:
    """Run the interactive Textual TUI for an existing view model."""

    view_model = load_execution_rerank_tui_view_model(path)
    app = create_execution_rerank_tui_app(view_model, source_path=path)
    app.run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="open a Textual TUI for a CodeLeWM execution-rerank tour"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--view-model", type=Path, help="reports/execution_rerank_view_model.json path"
    )
    source.add_argument("--demo-dir", type=Path, help="execution-rerank demo directory")
    parser.add_argument(
        "--snapshot-json",
        action="store_true",
        help="emit the deterministic TUI snapshot instead of opening Textual",
    )
    args = parser.parse_args(argv)

    path = resolve_execution_rerank_tui_view_model_path(
        view_model=args.view_model, demo_dir=args.demo_dir
    )
    view_model = load_execution_rerank_tui_view_model(path)
    if args.snapshot_json:
        print(
            json.dumps(
                build_execution_rerank_tui_snapshot(view_model),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return run_execution_rerank_tui(path)


def _import_textual() -> dict[str, Any]:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.widgets import DataTable, Footer, Header, Static
        from rich.text import Text
    except ModuleNotFoundError as exc:  # pragma: no cover - exact package varies by env.
        if exc.name and (
            exc.name in {"rich", "textual"}
            or exc.name.startswith("rich.")
            or exc.name.startswith("textual.")
        ):
            raise ExecutionRerankTuiError(
                "Textual is not installed",
                error_type="optional_dependency_missing",
                remediation=f"install the optional TUI dependencies with `{TUI_INSTALL_HINT}`",
            ) from exc
        raise
    return {
        "App": App,
        "Binding": Binding,
        "ComposeResult": ComposeResult,
        "DataTable": DataTable,
        "Footer": Footer,
        "Header": Header,
        "Static": Static,
        "Text": Text,
    }


def _make_app_class(textual: Mapping[str, Any]) -> type:
    app_base = textual["App"]
    binding = textual["Binding"]
    data_table = textual["DataTable"]
    footer = textual["Footer"]
    header = textual["Header"]
    static = textual["Static"]
    text = textual["Text"]

    class CodeLeWMExecutionRerankTui(app_base):  # type: ignore[misc, valid-type]
        CSS = """
        Screen {
            layout: vertical;
        }
        #headline {
            height: auto;
            border: solid $accent;
            padding: 0 1;
            margin: 0 1 1 1;
        }
        #completion-table {
            height: 14;
            margin: 0 1 1 1;
        }
        #details {
            height: 1fr;
            border: solid $primary;
            padding: 0 1;
            margin: 0 1;
            overflow-y: auto;
        }
        """
        BINDINGS = [
            binding("q", "quit", "Quit"),
        ]

        def __init__(self, *, snapshot: Mapping[str, Any], source_path: str | None) -> None:
            super().__init__()
            self.snapshot = snapshot
            self.source_path = source_path

        def compose(self) -> Any:
            yield header()
            yield static(_headline_text(self.snapshot), id="headline")
            yield data_table(id="completion-table")
            yield static(_details_text(self.snapshot), id="details")
            yield footer()

        def on_mount(self) -> None:
            headline = _mapping(self.snapshot.get("headline"))
            self.title = "CodeLeWM Execution-Rerank Showcase"
            self.sub_title = _safe(headline.get("scenario_id"))
            table = self.query_one("#completion-table", data_table)
            table.cursor_type = "row"
            table.zebra_stripes = True
            table.add_columns(
                "codelewm rank", "completion", "tests", "score", "llm rank", "lexical rank"
            )
            for row in self.snapshot.get("completions", ()):
                if not isinstance(row, Mapping):
                    continue
                passed = bool(row.get("passed"))
                tests_cell = text(
                    "pass" if passed else "fail",
                    style="green" if passed else "red",
                )
                table.add_row(
                    _safe(row.get("codelewm_rank")),
                    _safe(row.get("completion_id")),
                    tests_cell,
                    _safe(row.get("codelewm_score")),
                    _safe(row.get("llm_order_rank")),
                    _safe(row.get("lexical_rank")),
                )

    return CodeLeWMExecutionRerankTui


def _headline_text(snapshot: Mapping[str, Any]) -> str:
    headline = _mapping(snapshot.get("headline"))
    no_action = _mapping(snapshot.get("no_action"))
    claim_gate = _mapping(snapshot.get("claim_gate"))
    return "\n".join(
        [
            f"scenario: {_safe(headline.get('scenario_id'))} | benchmark: {_safe(headline.get('benchmark_id'))}",
            f"problems: {_safe(headline.get('problem_count'))} x {_safe(headline.get('completions_per_problem'))} completions"
            f" | score direction: {_safe(headline.get('score_direction'))}",
            f"pass@1  codelewm={_safe(headline.get('codelewm_pass_at_1'))}"
            f" llm_order={_safe(headline.get('llm_order_pass_at_1'))}"
            f" no_action={_safe(headline.get('no_action_pass_at_1'))}",
            f"lift vs llm_order: {_safe(headline.get('pass_at_1_lift'))} pts (ci {_safe(headline.get('bootstrap_lift_ci'))})"
            f" | vs no-action: {_safe(no_action.get('interpretation'))}",
            f"claim allowed: {_safe(claim_gate.get('allowed'))}",
        ]
    )


def _details_text(snapshot: Mapping[str, Any]) -> str:
    lines: list[str] = []
    no_action = _mapping(snapshot.get("no_action"))
    lines.append("No-action comparison")
    lines.append(
        f"- status={_safe(no_action.get('status'))}"
        f" codelewm={_safe(no_action.get('codelewm_pass_at_1'))}"
        f" no_action={_safe(no_action.get('no_action_pass_at_1'))}"
        f" delta={_safe(no_action.get('codelewm_minus_no_action'))}"
        f" lift={_safe(no_action.get('codelewm_lift_over_no_action'))} pts"
        f" ({_safe(no_action.get('interpretation'))})"
    )
    lines.append("")
    lines.append("Baselines")
    for row in snapshot.get("baselines", ()):
        if isinstance(row, Mapping):
            lines.append(
                f"- {_safe(row.get('baseline'))}: pass@1={_safe(row.get('pass_at_1'))}"
                f" ({_safe(row.get('pass_count'))}/{_safe(row.get('problem_count'))})"
            )
    lines.append("")
    lines.append("Diagnostics")
    for row in snapshot.get("diagnostics", ()):
        if isinstance(row, Mapping):
            lines.append(
                f"- {_safe(row.get('name'))}: {_safe(row.get('status'))}"
                f" {_safe(row.get('detail'))}"
            )
    lines.append("")
    lines.append("Artifact lineage")
    lineage = _mapping(snapshot.get("artifact_lineage"))
    lines.append(f"- command: {_safe(lineage.get('command'))}")
    parents = lineage.get("parent_artifact_ids", ())
    if isinstance(parents, Sequence) and not isinstance(parents, (str, bytes)) and parents:
        for parent in parents:
            lines.append(f"- parent: {_safe(parent)}")
    else:
        lines.append("- parent: n/a")
    for key in ("manifest_path", "report_path", "view_model_path", "html_path", "asciicast_path"):
        lines.append(f"- {key}: {_safe(lineage.get(key))}")
    notes = [_safe(item) for item in snapshot.get("notes", ())]
    if notes:
        lines.append("")
        lines.append("Notes")
        lines.extend(f"- {item}" for item in notes)
    lines.append("")
    claim_gate = _mapping(snapshot.get("claim_gate"))
    lines.append(
        "Claim gate: "
        f"allowed={_safe(claim_gate.get('allowed'))} reason={_safe(claim_gate.get('reason'))}"
    )
    return "\n".join(lines)


def _completion_row(panel: Mapping[str, Any]) -> dict[str, Any]:
    scores = _mapping(panel.get("scores"))
    ranks = _mapping(panel.get("rank_by_baseline"))
    return {
        "codelewm_rank": _safe(panel.get("codelewm_rank")),
        "completion_id": _safe(panel.get("completion_id")),
        "passed": bool(panel.get("passed")),
        "codelewm_score": _score_text(scores.get("codelewm")),
        "llm_order_rank": _safe(ranks.get("llm_order")),
        "lexical_rank": _safe(ranks.get("lexical")),
    }


def _baseline_rows(baselines: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(baselines, Sequence) or isinstance(baselines, (str, bytes)):
        return rows
    for row in baselines:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "baseline": _safe(row.get("baseline")),
                "pass_at_1": _score_text(row.get("pass_at_1")),
                "pass_count": _safe(row.get("pass_count")),
                "problem_count": _safe(row.get("problem_count")),
            }
        )
    return rows


def _diagnostic_rows(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ordered_diagnostic_slot_names(diagnostics):
        slot_map = _mapping(diagnostics.get(name))
        detail = (
            slot_map.get("reason")
            or slot_map.get("model_id")
            or slot_map.get("scope")
            or slot_map.get("reference")
            or ""
        )
        rows.append(
            {
                "name": _safe(name),
                "status": _safe(slot_map.get("status")),
                "detail": _safe(detail),
            }
        )
    return rows


def _rank_sort_key(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 1_000_000


def _ci_text(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return f"[{_score_text(value[0])}, {_score_text(value[1])}]"
    return _safe(value)


def _score_text(value: object) -> str:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return f"{float(value):.4f}"
    return _safe(value)


def _delta_text(value: object) -> str:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        sign = "+" if float(value) >= 0 else ""
        return f"{sign}{float(value):.4f}"
    return _safe(value)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests.
    raise SystemExit(main())
