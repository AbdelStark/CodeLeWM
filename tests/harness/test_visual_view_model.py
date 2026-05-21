from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.harness import (
    DEMO_VISUAL_VIEW_MODEL_SCHEMA_VERSION,
    DemoVisualViewModelError,
    build_demo_visual_view_model,
    read_demo_visual_view_model,
    validate_demo_visual_view_model_payload,
    write_demo_visual_view_model,
)


class DemoVisualViewModelTest(unittest.TestCase):
    def test_builds_shared_view_model_without_terminal_layout_or_raw_patch_dump(self) -> None:
        payload = build_demo_visual_view_model(
            demo_report=_demo_report(),
            candidate_pack=_candidate_pack(),
            out_dir=".artifacts/demo",
            manifest_verify={"ok": True, "files_checked": 4},
            secret_scan={"ok": True, "findings": []},
            html_secret_scan={"ok": True, "findings": []},
        )

        self.assertEqual(payload["schema_version"], DEMO_VISUAL_VIEW_MODEL_SCHEMA_VERSION)
        self.assertEqual(payload["summary"]["score_direction"], "lower_is_better")
        self.assertEqual(payload["summary"]["best_candidate"], "candidate_001")
        self.assertEqual(
            payload["summary"]["best_no_action_delta_interpretation"],
            "better_than_no_action",
        )
        self.assertEqual(payload["artifact_gates"]["manifest_verify"]["files_checked"], 4)
        self.assertEqual(payload["diagnostics"]["run_timeline"]["status"], "available")
        self.assertEqual(payload["diagnostics"]["latent_matrix"]["status"], "not_configured")
        first = payload["candidates"][0]
        self.assertEqual(first["patch_summary"]["changed_files"], ["app.py"])
        self.assertEqual(first["patch_summary"]["hunk_count"], 1)
        self.assertEqual(first["patch_summary"]["additions"], 2)
        self.assertEqual(first["patch_summary"]["deletions"], 1)
        self.assertNotIn("print this whole raw patch", json.dumps(payload))
        json.dumps(payload, sort_keys=True, allow_nan=False)

    def test_rejects_ansi_escape_codes(self) -> None:
        payload = build_demo_visual_view_model(
            demo_report=_demo_report(),
            candidate_pack=_candidate_pack(),
            out_dir=".artifacts/demo",
        )
        payload["summary"]["task_id"] = "\x1b[31mbad"

        with self.assertRaisesRegex(DemoVisualViewModelError, "ANSI"):
            validate_demo_visual_view_model_payload(payload)

    def test_round_trips_view_model_json(self) -> None:
        payload = build_demo_visual_view_model(
            demo_report=_demo_report(),
            candidate_pack=_candidate_pack(),
            out_dir=".artifacts/demo",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "visual_view_model.json"
            write_demo_visual_view_model(payload, path)
            loaded = read_demo_visual_view_model(path)

        self.assertEqual(loaded, payload)

    def test_base_import_does_not_import_textual(self) -> None:
        sys.modules.pop("textual", None)
        import codelewm.harness  # noqa: PLC0415

        self.assertIn("build_demo_visual_view_model", codelewm.harness.__all__)
        self.assertNotIn("textual", sys.modules)


def _demo_report() -> dict[str, object]:
    return {
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
    }


def _candidate_pack() -> dict[str, object]:
    return {
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
    }


if __name__ == "__main__":
    unittest.main()
