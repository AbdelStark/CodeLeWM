from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.harness import (
    LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION,
    LLM_WORLD_MODEL_DEMO_RUN_SCHEMA_VERSION,
    read_llm_world_model_demo_report,
    run_llm_world_model_demo,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


ROOT = Path(__file__).resolve().parents[2]


class LLMWorldModelDemoTest(unittest.TestCase):
    def test_runner_writes_manifested_fixture_demo_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "app.py"
            checkpoint = root / "checkpoint.bin"
            before.write_text("value = 1\n", encoding="utf-8")
            checkpoint.write_bytes(b"fixture checkpoint")

            result = run_llm_world_model_demo(
                before=before,
                instruction="add a comment candidate",
                checkpoint=checkpoint,
                out=root / "demo",
                task_id="demo-fixture",
                context_path="app.py",
                env={
                    "CODELEWM_LLM_PROVIDER": "openrouter",
                    "CODELEWM_LLM_DRY_RUN": "1",
                    "CODELEWM_LLM_MAX_CANDIDATES": "2",
                },
                allow_unsafe_checkpoint=True,
                command=("codelewm", "llm-demo"),
            )
            demo_manifest = read_artifact_manifest(root / "demo" / "manifest.json")
            checked = validate_artifact_checksums(demo_manifest, root=root / "demo")
            candidate_manifest = read_artifact_manifest(
                root / "demo" / result.candidate_pack_manifest_path
            )
            validate_artifact_checksums(candidate_manifest, root=root / "demo" / "candidate_pack")
            report = read_llm_world_model_demo_report(root / "demo" / result.report_path)
            timeline = json.loads((root / "demo" / "reports" / "run_timeline.json").read_text(encoding="utf-8"))
            html = (root / "demo" / result.html_path).read_text(encoding="utf-8")

        self.assertEqual(result.schema_version, LLM_WORLD_MODEL_DEMO_RUN_SCHEMA_VERSION)
        self.assertTrue(result.success)
        self.assertEqual(report["schema_version"], LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION)
        self.assertEqual(demo_manifest.artifact_kind, "demo_report")
        self.assertEqual(candidate_manifest.artifact_kind, "candidate_pack")
        self.assertEqual(demo_manifest.parent_artifacts, (candidate_manifest.artifact_id,))
        self.assertEqual({path.name for path in checked}, {"llm_world_model_demo_report.json", "demo.html", "run_timeline.json"})
        self.assertEqual(result.html_path, "demo.html")
        self.assertEqual(report["artifacts"]["run_timeline_path"], "reports/run_timeline.json")
        self.assertEqual(timeline["schema_version"], "codelewm.run_timeline.v1")
        self.assertEqual(timeline["status"], "completed")
        self.assertIn("candidate generation", [step["name"] for step in timeline["steps"]])
        self.assertIn("world model scoring", [step["name"] for step in timeline["steps"]])
        self.assertIn("Visual demo report", html)
        self.assertIn("fixture dry-run", html)
        self.assertIn("candidate_001", html)
        self.assertNotIn("—", html)
        self.assertEqual(report["candidate_summary"]["candidate_count"], 2)
        self.assertEqual(report["candidate_summary"]["valid_candidate_count"], 2)
        self.assertEqual(report["orders"]["llm"], ["candidate_001", "candidate_002"])
        self.assertEqual(len(report["orders"]["codelewm"]), 2)
        self.assertEqual(report["scores"]["score_direction"], "lower_is_better")
        self.assertIn("random", report["baselines"])
        self.assertIn("lexical", report["baselines"])
        self.assertIn("no_action", report["baselines"])
        self.assertFalse(report["claim_gate"]["allowed"])
        self.assertEqual(report["static_checks"]["status"], "not_configured")

    def test_cli_runs_fixture_demo_and_emits_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "app.py"
            checkpoint = root / "checkpoint.bin"
            out = root / "demo"
            before.write_text("value = 1\n", encoding="utf-8")
            checkpoint.write_bytes(b"fixture checkpoint")
            env = os.environ.copy()
            env.update(
                {
                    "CODELEWM_LLM_PROVIDER": "openrouter",
                    "CODELEWM_LLM_DRY_RUN": "1",
                    "CODELEWM_LLM_MAX_CANDIDATES": "2",
                }
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "llm-demo",
                    "--before",
                    str(before),
                    "--instruction",
                    "add a comment candidate",
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(out),
                    "--task-id",
                    "demo-cli",
                    "--context-path",
                    "app.py",
                    "--allow-unsafe-checkpoint",
                    "--json",
                ],
                cwd=ROOT,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            payload = json.loads(completed.stdout)
            report = read_llm_world_model_demo_report(out / payload["report_path"])
            html = (out / payload["html_path"]).read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertTrue(payload["success"])
        self.assertEqual(payload["schema_version"], LLM_WORLD_MODEL_DEMO_RUN_SCHEMA_VERSION)
        self.assertEqual(payload["html_path"], "demo.html")
        self.assertIn("Visual demo report", html)
        self.assertEqual(report["candidate_summary"]["candidate_count"], 2)
        self.assertFalse(report["claim_gate"]["allowed"])


if __name__ == "__main__":
    unittest.main()
