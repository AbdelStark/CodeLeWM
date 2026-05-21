"""Optional Textual TUI for LLM plus world-model demo reports."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from codelewm.observability.logging import redact_value

from .visual_view_model import (
    DemoVisualViewModelError,
    read_demo_visual_view_model,
    validate_demo_visual_view_model_payload,
)


DEMO_TUI_SNAPSHOT_SCHEMA_VERSION = "codelewm.harness.demo_tui_snapshot.v1"
TUI_INSTALL_HINT = "uv sync --group dev --group tui"


class TextualDemoTuiError(ValueError):
    """Raised when the optional demo TUI cannot be loaded or rendered."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "config_error",
        remediation: str = "inspect the demo TUI inputs and retry",
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.remediation = remediation


def resolve_demo_tui_view_model_path(
    *,
    view_model: str | Path | None = None,
    demo_dir: str | Path | None = None,
) -> Path:
    """Resolve the view-model path for an existing demo artifact."""

    if view_model is not None and demo_dir is not None:
        raise TextualDemoTuiError("pass either view_model or demo_dir, not both")
    if view_model is None and demo_dir is None:
        raise TextualDemoTuiError("pass a visual view-model path or demo artifact directory")
    if view_model is not None:
        return Path(view_model)
    return Path(demo_dir or "") / "reports" / "visual_view_model.json"


def load_demo_tui_view_model(path: str | Path) -> Mapping[str, Any]:
    """Load the shared visual view model used by the TUI."""

    source = Path(path)
    if not source.is_file():
        raise TextualDemoTuiError(
            f"demo visual view model does not exist: {source}",
            remediation="run `codelewm llm-demo` first or pass --demo-dir/--view-model",
        )
    try:
        return read_demo_visual_view_model(source)
    except (OSError, json.JSONDecodeError, DemoVisualViewModelError) as exc:
        raise TextualDemoTuiError(
            f"demo visual view model is invalid: {source}",
            remediation="rerun the demo and keep reports/visual_view_model.json intact",
        ) from exc


def build_demo_tui_snapshot(view_model: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic, JSON-native state snapshot for TUI tests and views."""

    payload = validate_demo_visual_view_model_payload(view_model)
    summary = _mapping(payload.get("summary"))
    generator = _mapping(payload.get("generator"))
    diagnostics = _mapping(payload.get("diagnostics"))
    artifact_gates = _mapping(payload.get("artifact_gates"))
    artifacts = _mapping(payload.get("artifacts"))
    candidates = [
        _candidate_row(candidate)
        for candidate in sorted(
            (item for item in payload.get("candidates", ()) if isinstance(item, Mapping)),
            key=lambda item: (
                _rank_sort_key(item.get("rank")),
                str(item.get("candidate_id", "")),
            ),
        )
    ]
    snapshot = {
        "schema_version": DEMO_TUI_SNAPSHOT_SCHEMA_VERSION,
        "summary": {
            "task_id": _safe(summary.get("task_id")),
            "mode": _safe(summary.get("mode")),
            "success": bool(summary.get("success")),
            "context_path": _safe(summary.get("context_path")),
            "checkpoint": _safe(summary.get("checkpoint_short_sha")),
            "scorer": _safe(summary.get("scorer")),
            "score_direction": _safe(summary.get("score_direction")),
            "best_candidate": _safe(summary.get("best_candidate")),
            "best_score": _score_text(summary.get("best_score")),
            "no_action_score": _score_text(summary.get("no_action_score")),
            "best_candidate_minus_no_action": _delta_text(
                summary.get("best_candidate_minus_no_action")
            ),
            "best_no_action_delta_interpretation": _safe(
                summary.get("best_no_action_delta_interpretation")
            ),
            "candidate_count": _safe(summary.get("candidate_count")),
            "valid_candidate_count": _safe(summary.get("valid_candidate_count")),
        },
        "generator": {
            "provider": _safe(generator.get("provider")),
            "model": _safe(generator.get("model")),
            "sdk": f"{_safe(generator.get('sdk'))} {_safe(generator.get('sdk_version'))}".strip(),
            "routing": _safe(generator.get("routing")),
            "byok_enabled": bool(generator.get("byok_enabled")),
        },
        "candidates": candidates,
        "diagnostics": _slot_rows(diagnostics),
        "artifact_gates": _gate_rows(artifact_gates),
        "warnings": [_safe(item) for item in payload.get("warnings", ())],
        "claim_gate": dict(redact_value(payload.get("claim_gate", {}))),
        "artifacts": {
            key: _safe(value)
            for key, value in artifacts.items()
            if value not in (None, "", "n/a")
        },
    }
    json.dumps(snapshot, sort_keys=True, allow_nan=False)
    return snapshot


def create_demo_tui_app(
    view_model: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> Any:
    """Create the Textual app for an already-loaded visual view model."""

    textual = _import_textual()
    snapshot = build_demo_tui_snapshot(view_model)
    app_cls = _make_app_class(textual)
    return app_cls(snapshot=snapshot, source_path=None if source_path is None else str(source_path))


def run_demo_tui(path: str | Path) -> int:
    """Run the interactive Textual TUI for an existing visual view model."""

    view_model = load_demo_tui_view_model(path)
    app = create_demo_tui_app(view_model, source_path=path)
    app.run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="open a Textual TUI for a CodeLeWM demo run")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--view-model", type=Path, help="reports/visual_view_model.json path")
    source.add_argument("--demo-dir", type=Path, help="demo artifact directory")
    parser.add_argument(
        "--snapshot-json",
        action="store_true",
        help="emit the deterministic TUI snapshot instead of opening Textual",
    )
    args = parser.parse_args(argv)

    path = resolve_demo_tui_view_model_path(view_model=args.view_model, demo_dir=args.demo_dir)
    view_model = load_demo_tui_view_model(path)
    if args.snapshot_json:
        print(json.dumps(build_demo_tui_snapshot(view_model), indent=2, sort_keys=True))
        return 0
    return run_demo_tui(path)


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
            raise TextualDemoTuiError(
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

    class CodeLeWMDemoTui(app_base):  # type: ignore[misc, valid-type]
        CSS = """
        Screen {
            layout: vertical;
        }
        #summary {
            height: auto;
            border: solid $accent;
            padding: 0 1;
            margin: 0 1 1 1;
        }
        #candidate-table {
            height: 12;
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
            yield static(_summary_text(self.snapshot), id="summary")
            yield data_table(id="candidate-table")
            yield static(_details_text(self.snapshot), id="details")
            yield footer()

        def on_mount(self) -> None:
            summary = _mapping(self.snapshot.get("summary"))
            self.title = "CodeLeWM Demo Inspector"
            self.sub_title = _safe(summary.get("task_id"))
            table = self.query_one("#candidate-table", data_table)
            table.cursor_type = "row"
            table.zebra_stripes = True
            table.add_columns("rank", "candidate", "score", "vs no-op", "status", "diff", "patch")
            for row in self.snapshot.get("candidates", ()):
                if not isinstance(row, Mapping):
                    continue
                interpretation = _safe(row.get("no_action_delta_interpretation"))
                style = "green" if interpretation == "better_than_no_action" else "yellow"
                table.add_row(
                    _safe(row.get("rank")),
                    _safe(row.get("candidate_id")),
                    _safe(row.get("score")),
                    text(_safe(row.get("candidate_minus_no_action")), style=style),
                    _safe(row.get("status")),
                    _safe(row.get("diff_summary")),
                    _safe(row.get("patch_short_sha")),
                )

    return CodeLeWMDemoTui


def _summary_text(snapshot: Mapping[str, Any]) -> str:
    summary = _mapping(snapshot.get("summary"))
    generator = _mapping(snapshot.get("generator"))
    return "\n".join(
        [
            f"task: {_safe(summary.get('task_id'))} | mode: {_safe(summary.get('mode'))} | success: {_safe(summary.get('success'))}",
            f"context: {_safe(summary.get('context_path'))} | checkpoint: {_safe(summary.get('checkpoint'))}",
            f"model: {_safe(generator.get('model'))} | scorer: {_safe(summary.get('scorer'))}",
            f"best: {_safe(summary.get('best_candidate'))} score={_safe(summary.get('best_score'))} | no-action={_safe(summary.get('no_action_score'))} | delta={_safe(summary.get('best_candidate_minus_no_action'))}",
            f"score direction: {_safe(summary.get('score_direction'))} | claim allowed: {_safe(_mapping(snapshot.get('claim_gate')).get('allowed'))}",
        ]
    )


def _details_text(snapshot: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Diagnostics")
    for row in snapshot.get("diagnostics", ()):
        if isinstance(row, Mapping):
            lines.append(f"- {_safe(row.get('name'))}: {_safe(row.get('status'))} {_safe(row.get('path'))}")
    lines.append("")
    lines.append("Artifact gates")
    for row in snapshot.get("artifact_gates", ()):
        if isinstance(row, Mapping):
            lines.append(
                f"- {_safe(row.get('name'))}: ok={_safe(row.get('ok'))} "
                f"findings={_safe(row.get('findings'))} files={_safe(row.get('files_checked'))}"
            )
    lines.append("")
    lines.append("Candidate diff previews")
    for row in snapshot.get("candidates", ()):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"- {_safe(row.get('candidate_id'))}: {_safe(row.get('diff_summary'))}; "
            f"files={_safe(row.get('changed_files'))}"
        )
        for preview_line in row.get("preview_lines", ()):
            lines.append(f"  {preview_line}")
    lines.append("")
    claim_gate = _mapping(snapshot.get("claim_gate"))
    lines.append(
        "Claim gate: "
        f"allowed={_safe(claim_gate.get('allowed'))} reason={_safe(claim_gate.get('reason'))}"
    )
    warnings = [_safe(item) for item in snapshot.get("warnings", ())]
    if warnings:
        lines.append("")
        lines.append("Warnings")
        lines.extend(f"- {item}" for item in warnings)
    lines.append("")
    lines.append("Artifacts")
    for key, value in _mapping(snapshot.get("artifacts")).items():
        lines.append(f"- {key}: {_safe(value)}")
    return "\n".join(lines)


def _candidate_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    patch_summary = _mapping(candidate.get("patch_summary"))
    changed_files = [str(item) for item in patch_summary.get("changed_files", ()) if item]
    additions = _int(patch_summary.get("additions"))
    deletions = _int(patch_summary.get("deletions"))
    hunk_count = _int(patch_summary.get("hunk_count"))
    return {
        "rank": _safe(candidate.get("rank")),
        "candidate_id": _safe(candidate.get("candidate_id")),
        "score": _safe(candidate.get("score_display")),
        "candidate_minus_no_action": _safe(candidate.get("no_action_delta_display")),
        "no_action_delta_interpretation": _safe(candidate.get("no_action_delta_interpretation")),
        "status": _safe(candidate.get("status")),
        "is_valid": bool(candidate.get("is_valid")),
        "patch_short_sha": _safe(candidate.get("patch_short_sha")),
        "changed_files": ", ".join(changed_files) if changed_files else "n/a",
        "diff_summary": f"+{additions}/-{deletions} hunks={hunk_count}",
        "preview_lines": [
            _safe(line) for line in patch_summary.get("preview_lines", ())[:8]
        ],
    }


def _slot_rows(slots: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("checkpoint_inspection", "latent_matrix", "run_timeline", "tensorboard"):
        slot = _mapping(slots.get(name))
        rows.append(
            {
                "name": name,
                "status": _safe(slot.get("status")),
                "path": _safe(slot.get("path")),
            }
        )
    return rows


def _gate_rows(gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("manifest_verify", "artifact_secret_scan", "html_secret_scan"):
        gate = _mapping(gates.get(name))
        rows.append(
            {
                "name": name,
                "status": _safe(gate.get("status")),
                "ok": _safe(gate.get("ok")),
                "findings": _safe(gate.get("findings")),
                "files_checked": _safe(gate.get("files_checked")),
            }
        )
    return rows


def _rank_sort_key(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 1_000_000


def _score_text(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return f"{float(value):.6f}"
    return _safe(value)


def _delta_text(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        sign = "+" if float(value) >= 0 else ""
        return f"{sign}{float(value):.6f}"
    return _safe(value)


def _int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


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
