from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codelewm.harness.demo_tui import (
    DEMO_TUI_SNAPSHOT_SCHEMA_VERSION,
    TextualDemoTuiError,
    build_demo_tui_snapshot,
    create_demo_tui_app,
    resolve_demo_tui_view_model_path,
)
from codelewm.harness.visual_view_model import (
    build_demo_visual_view_model,
    write_demo_visual_view_model,
)


ROOT = Path(__file__).resolve().parents[2]


class DemoTuiTest(unittest.TestCase):
    def test_snapshot_summarizes_required_panels_without_raw_patch_dump(self) -> None:
        snapshot = build_demo_tui_snapshot(_view_model())

        self.assertEqual(snapshot["schema_version"], DEMO_TUI_SNAPSHOT_SCHEMA_VERSION)
        self.assertEqual(snapshot["summary"]["task_id"], "demo-fixture")
        self.assertEqual(snapshot["summary"]["score_direction"], "lower_is_better")
        self.assertEqual(snapshot["summary"]["best_no_action_delta_interpretation"], "better_than_no_action")
        self.assertEqual(snapshot["generator"]["provider"], "openrouter")
        self.assertEqual(snapshot["candidates"][0]["candidate_id"], "candidate_001")
        self.assertEqual(snapshot["candidates"][0]["diff_summary"], "+2/-1 hunks=1")
        self.assertEqual(snapshot["diagnostics"][2]["name"], "run_timeline")
        self.assertEqual(snapshot["diagnostics"][2]["sha256"], "n/a")
        self.assertEqual(snapshot["artifact_gates"][0]["name"], "manifest_verify")
        self.assertFalse(snapshot["claim_gate"]["allowed"])
        self.assertNotIn("do not print this whole raw patch", json.dumps(snapshot))
        json.dumps(snapshot, sort_keys=True, allow_nan=False)

    def test_resolves_demo_dir_to_visual_view_model_path(self) -> None:
        self.assertEqual(
            resolve_demo_tui_view_model_path(demo_dir=".artifacts/demo"),
            Path(".artifacts/demo") / "reports" / "visual_view_model.json",
        )

    def test_create_app_reports_missing_textual_as_optional_dependency(self) -> None:
        with patch.dict(sys.modules, {"textual": None}):
            with self.assertRaisesRegex(TextualDemoTuiError, "Textual is not installed") as ctx:
                create_demo_tui_app(_view_model(), source_path="reports/visual_view_model.json")

        self.assertEqual(ctx.exception.error_type, "optional_dependency_missing")
        self.assertIn("uv sync --group dev --group tui", ctx.exception.remediation)

    def test_cli_snapshot_json_loads_existing_view_model_without_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "visual_view_model.json"
            write_demo_visual_view_model(_view_model(), path)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "llm-demo-tui",
                    "--view-model",
                    str(path),
                    "--snapshot-json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], DEMO_TUI_SNAPSHOT_SCHEMA_VERSION)
        self.assertEqual(payload["summary"]["task_id"], "demo-fixture")

    @unittest.skipUnless(importlib.util.find_spec("textual"), "Textual optional dependency not installed")
    def test_textual_app_mounts_fixture_snapshot(self) -> None:
        async def run_case() -> None:
            app = create_demo_tui_app(_view_model(), source_path="reports/visual_view_model.json")
            async with app.run_test() as pilot:
                await pilot.pause()
                table = app.query_one("#candidate-table")
                self.assertEqual(table.row_count, 2)

        import asyncio

        asyncio.run(run_case())


def _view_model() -> dict[str, object]:
    return build_demo_visual_view_model(
        demo_report={
            "success": True,
            "task": {
                "task_id": "demo-fixture",
                "context_path": "app.py",
                "before_path": "/Users/example/project/app.py",
            },
            "artifacts": {
                "checkpoint_sha256": "a" * 64,
                "candidate_pack_manifest_path": "candidate_pack/manifest.json",
                "run_timeline_path": "reports/run_timeline.json",
            },
            "candidate_summary": {"candidate_count": 2, "valid_candidate_count": 2},
            "orders": {
                "llm": ["candidate_001", "candidate_002"],
                "codelewm": ["candidate_001", "candidate_002"],
                "lexical": ["candidate_002", "candidate_001"],
                "random": ["candidate_002", "candidate_001"],
                "no_action": ["no_action"],
            },
            "scores": {
                "model_id": "codelewm.torch_transition_scorer.v1",
                "score_direction": "lower_is_better",
                "no_action": {"final_score": 1.0},
                "codelewm_rerank": [
                    {"candidate": "/tmp/candidate_001.patch", "final_score": 0.5},
                    {"candidate": "/tmp/candidate_002.patch", "final_score": 1.5},
                ],
            },
            "claim_gate": {
                "allowed": False,
                "reason": "demo_report_is_not_downstream_benchmark_evidence",
            },
            "warnings": [],
        },
        candidate_pack={
            "generation_config": {"dry_run": True},
            "generator": {
                "provider": "openrouter",
                "model": "anthropic/claude-4.5-sonnet",
                "sdk": "openrouter",
                "sdk_version": "0.9.1",
            },
            "provider_routing": {
                "requested_provider_options": {"only": ["anthropic"], "allow_fallbacks": False},
                "byok": {"enabled": True},
                "response_metadata": {"model": "anthropic/claude-4.5-sonnet-20250929"},
            },
            "candidates": [
                {
                    "candidate_id": "candidate_001",
                    "parser_status": "parseable_python_after_state",
                    "dry_run_patch_status": "applied",
                    "normalized_patch_sha256": "1" * 64,
                    "patch_text": "\n".join(
                        [
                            "--- a/app.py",
                            "+++ b/app.py",
                            "@@ -1,2 +1,3 @@",
                            "-return value",
                            "+value = value.strip()",
                            "+return value or 'untitled'",
                            "# do not print this whole raw patch",
                        ]
                    ),
                },
                {
                    "candidate_id": "candidate_002",
                    "parser_status": "parseable_python_after_state",
                    "dry_run_patch_status": "applied",
                    "normalized_patch_sha256": "2" * 64,
                    "patch_text": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-return value\n+return value.strip()\n",
                },
            ],
        },
        out_dir=".artifacts/demo",
        manifest_verify={"ok": True, "files_checked": 4},
        secret_scan={"ok": True, "findings": []},
        html_secret_scan={"ok": True, "findings": []},
    )


if __name__ == "__main__":
    unittest.main()
