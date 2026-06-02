"""Tests for the optional execution-rerank Textual TUI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codelewm.harness import (
    build_execution_rerank_view_model,
    read_execution_rerank_view_model,
    write_execution_rerank_view_model,
)
from codelewm.harness.execution_rerank_tui import (
    EXECUTION_RERANK_TUI_SNAPSHOT_SCHEMA_VERSION,
    ExecutionRerankTuiError,
    build_execution_rerank_tui_snapshot,
    create_execution_rerank_tui_app,
    resolve_execution_rerank_tui_view_model_path,
)


ROOT = Path(__file__).resolve().parents[2]


def _rerank_report() -> dict[str, object]:
    return {
        "schema_version": "codelewm.eval.execution_rerank_report.v1",
        "benchmark": "mbpp_demo",
        "problem_count": 1,
        "completions_per_problem": 2,
        "baselines": [
            {"baseline": "llm_order", "pass_at_1": 0.0, "pass_count": 0, "problem_count": 1},
            {"baseline": "codelewm", "pass_at_1": 1.0, "pass_count": 1, "problem_count": 1},
            {"baseline": "lexical", "pass_at_1": 0.0, "pass_count": 0, "problem_count": 1},
            {"baseline": "no_action", "pass_at_1": 0.0, "pass_count": 0, "problem_count": 1},
        ],
        "codelewm_lift_over_llm_order": 100.0,
        "bootstrap_lift_ci": [80.0, 100.0],
        "codelewm_lift_over_no_action": 100.0,
        "bootstrap_lift_over_no_action_ci": [60.0, 100.0],
        "claim_allowed": False,
        "claim_reason": "demo_tour_is_not_scaled_downstream_benchmark_evidence",
    }


def _completions() -> list[dict[str, object]]:
    return [
        {
            "completion_id": "p1::c1",
            "code": "def f():\n    return 1\n",
            "llm_order_rank": 1,
            "passed": False,
            "scores": {"codelewm": 0.2, "lexical": 0.5},
            "test_results": [{"input_id": "case-1", "passed": False}],
            "predicted_output_latent": {},
        },
        {
            "completion_id": "p1::c2",
            "code": "def f():\n    return 2\n",
            "llm_order_rank": 2,
            "passed": True,
            "scores": {"codelewm": 0.9, "lexical": 0.7},
            "test_results": [{"input_id": "case-1", "passed": True}],
            "predicted_output_latent": {},
        },
    ]


def _view_model() -> dict[str, object]:
    return build_execution_rerank_view_model(
        rerank_report=_rerank_report(),
        scenario_id="execution-rerank-mbpp",
        completion_records=_completions(),
        diagnostics={"checkpoint": {"model_id": "codelewm.scorer.v1", "device": "cpu"}},
        artifact_lineage={
            "parent_artifact_ids": ["candidate_pack-abc"],
            "command": ["scripts/llm-world-model-demo", "--tour", "1"],
            "manifest_path": "manifest.json",
            "view_model_path": "reports/execution_rerank_view_model.json",
            "html_path": "demo.html",
        },
    ).as_dict()


class ExecutionRerankTuiTest(unittest.TestCase):
    def test_snapshot_summarizes_required_panels(self) -> None:
        snapshot = build_execution_rerank_tui_snapshot(_view_model())
        self.assertEqual(
            snapshot["schema_version"], EXECUTION_RERANK_TUI_SNAPSHOT_SCHEMA_VERSION
        )
        headline = snapshot["headline"]
        self.assertEqual(headline["scenario_id"], "execution-rerank-mbpp")
        self.assertEqual(headline["score_direction"], "higher_is_better")
        self.assertEqual(headline["codelewm_pass_at_1"], "1.0000")
        # CodeLeWM rank winner (c2, score 0.9) is the first completion row.
        self.assertEqual(snapshot["completions"][0]["completion_id"], "p1::c2")
        self.assertEqual(snapshot["completions"][0]["codelewm_rank"], "1")
        self.assertTrue(snapshot["completions"][0]["passed"])
        # No-action comparison is explicit and positive in this fixture.
        self.assertEqual(snapshot["no_action"]["status"], "available")
        self.assertEqual(
            snapshot["no_action"]["interpretation"], "better_than_no_action"
        )
        # Diagnostics keep missing slots explicit.
        diag_by_name = {row["name"]: row["status"] for row in snapshot["diagnostics"]}
        self.assertEqual(diag_by_name["checkpoint"], "available")
        self.assertEqual(diag_by_name["retrieval_evidence"], "not_recorded")
        self.assertEqual(diag_by_name["sandbox"], "not_recorded")
        # Lineage carries parents + command.
        self.assertEqual(
            snapshot["artifact_lineage"]["parent_artifact_ids"], ["candidate_pack-abc"]
        )
        self.assertIn("--tour", snapshot["artifact_lineage"]["command"])
        # Claim gate is closed and JSON-native.
        self.assertFalse(snapshot["claim_gate"]["allowed"])
        json.dumps(snapshot, sort_keys=True, allow_nan=False)

    def test_diagnostics_order_is_stable_across_memory_and_disk(self) -> None:
        # The web report builds its snapshot from the in-memory view model while
        # the TUI always reads the sort_keys=True on-disk artifact. Both must
        # render diagnostics in the same canonical order to stay in lockstep.
        in_memory = build_execution_rerank_tui_snapshot(_view_model())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "execution_rerank_view_model.json"
            write_execution_rerank_view_model(_view_model(), path)
            round_tripped = build_execution_rerank_tui_snapshot(
                read_execution_rerank_view_model(path)
            )
        in_memory_order = [row["name"] for row in in_memory["diagnostics"]]
        round_trip_order = [row["name"] for row in round_tripped["diagnostics"]]
        self.assertEqual(in_memory_order, round_trip_order)
        self.assertEqual(
            in_memory_order[:3], ["retrieval_evidence", "checkpoint", "sandbox"]
        )

    def test_resolves_demo_dir_to_view_model_path(self) -> None:
        resolved = resolve_execution_rerank_tui_view_model_path(demo_dir=".artifacts/tour")
        self.assertEqual(
            resolved,
            Path(".artifacts/tour") / "reports" / "execution_rerank_view_model.json",
        )

    def test_create_app_reports_missing_textual_as_optional_dependency(self) -> None:
        # Drop any already-imported textual modules so the simulated absence
        # is observed even when an earlier test cached `textual.app`.
        saved = {
            key: value
            for key, value in sys.modules.items()
            if key == "textual" or key.startswith("textual.")
        }
        for key in saved:
            del sys.modules[key]
        try:
            with patch.dict(sys.modules, {"textual": None}):
                with self.assertRaisesRegex(
                    ExecutionRerankTuiError, "Textual is not installed"
                ):
                    create_execution_rerank_tui_app(_view_model())
                try:
                    create_execution_rerank_tui_app(_view_model())
                except ExecutionRerankTuiError as exc:
                    self.assertEqual(exc.error_type, "optional_dependency_missing")
                    self.assertIn("uv sync --group dev --group tui", exc.remediation)
        finally:
            for key in list(sys.modules):
                if key == "textual" or key.startswith("textual."):
                    del sys.modules[key]
            sys.modules.update(saved)

    def test_cli_snapshot_json_loads_view_model_without_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "execution_rerank_view_model.json"
            write_execution_rerank_view_model(_view_model(), path)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.execution_rerank_tui",
                    "--view-model",
                    str(path),
                    "--snapshot-json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(
            payload["schema_version"], EXECUTION_RERANK_TUI_SNAPSHOT_SCHEMA_VERSION
        )
        self.assertEqual(payload["headline"]["scenario_id"], "execution-rerank-mbpp")

    @unittest.skipUnless(
        importlib.util.find_spec("textual"), "textual optional dependency not installed"
    )
    def test_textual_app_mounts_fixture_snapshot(self) -> None:
        async def run_case() -> None:
            app = create_execution_rerank_tui_app(
                _view_model(),
                source_path="reports/execution_rerank_view_model.json",
            )
            async with app.run_test() as pilot:
                await pilot.pause()
                table = app.query_one("#completion-table")
                self.assertEqual(table.row_count, 2)

        import asyncio

        asyncio.run(run_case())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
