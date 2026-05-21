from __future__ import annotations

import unittest

from codelewm.harness.demo_terminal import render_demo_terminal_report


class DemoTerminalReportTest(unittest.TestCase):
    def test_renderer_summarizes_pipeline_without_candidate_patch_text(self) -> None:
        output = render_demo_terminal_report(
            demo_run={
                "success": True,
                "artifact_manifest_id": "demo_report-abc123",
                "artifact_manifest_path": "manifest.json",
                "report_path": "reports/llm_world_model_demo_report.json",
                "html_path": "demo.html",
            },
            manifest_verify={"ok": True, "files_checked": 2},
            secret_scan={"ok": True, "findings": []},
            html_secret_scan={"ok": True, "findings": []},
            demo_report={
                "success": True,
                "task": {"before_path": "input/app.py"},
                "artifacts": {
                    "checkpoint_sha256": "a" * 64,
                    "candidate_pack_manifest_id": "candidate_pack-123",
                },
                "candidate_summary": {
                    "candidate_count": 2,
                    "valid_candidate_count": 2,
                },
                "orders": {"codelewm": ["candidate_002", "candidate_001"]},
                "scores": {
                    "model_id": "codelewm.torch_transition_scorer.v1",
                    "no_action": {"final_score": 1.0},
                    "codelewm_rerank": [
                        {
                            "candidate": "/tmp/candidate_002.patch",
                            "final_score": 0.5,
                        },
                        {
                            "candidate": "/tmp/candidate_001.patch",
                            "final_score": 0.7,
                        },
                    ],
                },
                "warnings": [
                    "learned torch transition model runtime loaded from checkpoint",
                    "checkpoint_step=4",
                    "action_view=text",
                ],
                "claim_gate": {
                    "allowed": False,
                    "reason": "demo_report_is_not_downstream_benchmark_evidence",
                },
            },
            candidate_pack={
                "generation_config": {"dry_run": False},
                "generator": {
                    "provider": "openrouter",
                    "model": "anthropic/claude-4.5-sonnet",
                    "sdk": "openrouter",
                    "sdk_version": "0.9.1",
                },
                "provider_routing": {
                    "requested_provider_options": {
                        "only": ["anthropic"],
                        "allow_fallbacks": False,
                    },
                    "byok": {"enabled": True},
                    "response_metadata": {
                        "model": "anthropic/claude-4.5-sonnet-20250929"
                    },
                },
                "candidates": [
                    {
                        "candidate_id": "candidate_001",
                        "parser_status": "parseable_python_after_state",
                        "dry_run_patch_status": "applied",
                        "normalized_patch_sha256": "1" * 64,
                        "patch_text": "do not print this patch",
                    },
                    {
                        "candidate_id": "candidate_002",
                        "parser_status": "parseable_python_after_state",
                        "dry_run_patch_status": "applied",
                        "normalized_patch_sha256": "2" * 64,
                    },
                ],
            },
            out_dir=".artifacts/llm-world-model-demo/run",
            color=False,
        )

        self.assertIn("Candidate generation", output)
        self.assertIn("World-model inference", output)
        self.assertIn("score direction: lower transition energy is better", output)
        self.assertIn("-0.500000 (better than no-op)", output)
        self.assertIn("codelewm.torch_transition_scorer.v1", output)
        self.assertIn("candidate_002", output)
        self.assertIn("Diagnostics", output)
        self.assertIn("view model:", output)
        self.assertIn("demo_report_is_not_downstream_benchmark_evidence", output)
        self.assertIn("uv run scripts/llm-world-model-demo --json", output)
        self.assertNotIn("do not print this patch", output)


if __name__ == "__main__":
    unittest.main()
