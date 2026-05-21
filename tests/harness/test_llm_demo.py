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
    read_demo_visual_view_model,
    read_llm_world_model_demo_report,
    run_llm_world_model_demo,
)
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


class LLMWorldModelDemoTest(unittest.TestCase):
    def test_runner_writes_manifested_fixture_demo_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "app.py"
            checkpoint = root / "checkpoint.bin"
            before.write_text("value = 1\n", encoding="utf-8")
            checkpoint.write_bytes(b"fixture checkpoint")
            checkpoint_diag = _write_diagnostic_artifact(
                root / "checkpoint-diag",
                report_relative_path="reports/model_checkpoint_inspection.json",
                schema_version="codelewm.model_checkpoint_inspection.v1",
                artifact_id="checkpoint-diag-fixture",
            )
            latent_diag = _write_diagnostic_artifact(
                root / "latent-diag",
                report_relative_path="reports/latent_matrix_report.json",
                schema_version="codelewm.eval.latent_matrix_report.v1",
                artifact_id="latent-diag-fixture",
            )
            tensorboard_diag = _write_diagnostic_artifact(
                root / "tensorboard-diag",
                report_relative_path="reports/tensorboard_export.json",
                schema_version="codelewm.training.tensorboard_export.v1",
                artifact_id="tensorboard-diag-fixture",
            )

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
                checkpoint_inspection_manifest=checkpoint_diag,
                latent_matrix_manifest=latent_diag,
                tensorboard_manifest=tensorboard_diag,
                command=("codelewm", "llm-demo"),
            )
            demo_manifest = read_artifact_manifest(root / "demo" / "manifest.json")
            checked = validate_artifact_checksums(demo_manifest, root=root / "demo")
            candidate_manifest = read_artifact_manifest(
                root / "demo" / result.candidate_pack_manifest_path
            )
            validate_artifact_checksums(candidate_manifest, root=root / "demo" / "candidate_pack")
            report = read_llm_world_model_demo_report(root / "demo" / result.report_path)
            view_model = read_demo_visual_view_model(root / "demo" / result.visual_view_model_path)
            timeline = json.loads((root / "demo" / "reports" / "run_timeline.json").read_text(encoding="utf-8"))
            html = (root / "demo" / result.html_path).read_text(encoding="utf-8")

        self.assertEqual(result.schema_version, LLM_WORLD_MODEL_DEMO_RUN_SCHEMA_VERSION)
        self.assertTrue(result.success)
        self.assertEqual(report["schema_version"], LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION)
        self.assertEqual(demo_manifest.artifact_kind, "demo_report")
        self.assertEqual(candidate_manifest.artifact_kind, "candidate_pack")
        self.assertEqual(
            demo_manifest.parent_artifacts,
            (
                candidate_manifest.artifact_id,
                "checkpoint-diag-fixture",
                "latent-diag-fixture",
                "tensorboard-diag-fixture",
            ),
        )
        self.assertEqual(result.parent_artifacts, demo_manifest.parent_artifacts)
        self.assertEqual(
            {path.name for path in checked},
            {
                "llm_world_model_demo_report.json",
                "demo.html",
                "visual_view_model.json",
                "run_timeline.json",
            },
        )
        self.assertEqual(result.html_path, "demo.html")
        self.assertEqual(result.visual_view_model_path, "reports/visual_view_model.json")
        self.assertEqual(report["artifacts"]["visual_view_model_path"], "reports/visual_view_model.json")
        self.assertEqual(report["artifacts"]["run_timeline_path"], "reports/run_timeline.json")
        self.assertEqual(report["diagnostics"]["checkpoint_inspection"]["status"], "available")
        self.assertEqual(report["diagnostics"]["checkpoint_inspection"]["artifact_id"], "checkpoint-diag-fixture")
        self.assertEqual(report["diagnostics"]["checkpoint_inspection"]["manifest_file_path"], "reports/model_checkpoint_inspection.json")
        self.assertEqual(report["diagnostics"]["latent_matrix"]["status"], "available")
        self.assertEqual(report["diagnostics"]["tensorboard"]["status"], "available")
        self.assertRegex(report["diagnostics"]["tensorboard"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(view_model["schema_version"], "codelewm.harness.visual_view_model.v1")
        self.assertEqual(view_model["summary"]["score_direction"], "lower_is_better")
        self.assertEqual(view_model["diagnostics"]["checkpoint_inspection"]["artifact_id"], "checkpoint-diag-fixture")
        self.assertEqual(view_model["diagnostics"]["latent_matrix"]["status"], "available")
        self.assertEqual(view_model["diagnostics"]["tensorboard"]["status"], "available")
        self.assertEqual(view_model["diagnostics"]["run_timeline"]["status"], "available")
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
        self.assertEqual(payload["visual_view_model_path"], "reports/visual_view_model.json")
        self.assertIn("Visual demo report", html)
        self.assertIn("Model and latent", html)
        self.assertEqual(report["candidate_summary"]["candidate_count"], 2)
        self.assertFalse(report["claim_gate"]["allowed"])
        self.assertEqual(report["diagnostics"]["checkpoint_inspection"]["status"], "not_configured")
        self.assertEqual(report["diagnostics"]["latent_matrix"]["status"], "not_configured")
        self.assertEqual(report["diagnostics"]["tensorboard"]["status"], "not_configured")


def _write_diagnostic_artifact(
    root: Path,
    *,
    report_relative_path: str,
    schema_version: str,
    artifact_id: str,
) -> Path:
    report_path = root / report_relative_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "summary": {"fixture": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=root,
        files=(report_path,),
        command=("codelewm", "diagnostic-fixture"),
        config={"schema_version": schema_version},
        artifact_id=artifact_id,
        metadata={"report_path": report_relative_path},
    )
    manifest_path = root / "manifest.json"
    write_artifact_manifest(manifest, manifest_path)
    return manifest_path


if __name__ == "__main__":
    unittest.main()
