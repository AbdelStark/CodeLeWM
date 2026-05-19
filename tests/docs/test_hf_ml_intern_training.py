from __future__ import annotations

import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = ROOT / ".env.example"
RUNBOOK = ROOT / "docs" / "operations" / "HF_ML_INTERN_TRAINING.md"
GOAL_PROMPT = ROOT / "docs" / "roadmap" / "HF_ML_INTERN_GOAL_PROMPT.md"
NEXT_GOAL = ROOT / "docs" / "roadmap" / "NEXT_GOAL_PROMPT.md"
SCRIPTS = (
    ROOT / "scripts" / "hf-launch-codelewm-job",
    ROOT / "scripts" / "hf-run-codelewm-pipeline",
    ROOT / "scripts" / "hf-publish-codelewm-artifacts",
)


class HFMLInternTrainingDocsTest(unittest.TestCase):
    def test_env_example_defines_project_scoped_hf_settings(self) -> None:
        self.assertTrue(ENV_EXAMPLE.is_file(), f"missing: {ENV_EXAMPLE}")
        text = ENV_EXAMPLE.read_text(encoding="utf-8")

        for marker in (
            "HF_TOKEN=hf_xxx",
            "CODELEWM_HF_DATASET_REPO_ID=",
            "CODELEWM_HF_MODEL_REPO_ID=",
            "CODELEWM_HF_RESULTS_REPO_ID=",
            "CODELEWM_HF_JOBS_DRY_RUN=1",
            "CODELEWM_HF_PUBLISH_DRY_RUN=1",
            "CODELEWM_HF_PIPELINE_MODE=smoke",
            "CODELEWM_DATASET_BUILD_CONFIG=",
            "CODELEWM_TRAIN_CONFIG=",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_scripts_exist_and_are_executable(self) -> None:
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                self.assertTrue(script.is_file(), f"missing: {script}")
                self.assertTrue(os.access(script, os.X_OK), f"not executable: {script}")

    def test_runbook_documents_hf_jobs_ml_intern_and_publish_gates(self) -> None:
        self.assertTrue(RUNBOOK.is_file(), f"missing: {RUNBOOK}")
        text = RUNBOOK.read_text(encoding="utf-8")

        for marker in (
            "ml-intern --max-iterations -1",
            "hf jobs run",
            "--secrets HF_TOKEN",
            "scripts/hf-launch-codelewm-job",
            "scripts/hf-run-codelewm-pipeline",
            "scripts/hf-publish-codelewm-artifacts",
            "CODELEWM_HF_JOBS_DRY_RUN=0",
            "CODELEWM_HF_PUBLISH_DRY_RUN=0",
            "CODELEWM_HF_PIPELINE_MODE=scaled",
            "config/train/scaled/codelewm_scaled_gpu_a10g.yaml",
            "docs/training/SCALED_TRAINING_RUNBOOK.md",
            "action-view ablation",
            "codelewm.hf_publish_plan.v1",
            "#118",
            "#119",
            "#120",
            "#121",
            "#122",
            "#138",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_goal_prompt_is_headless_and_secret_safe(self) -> None:
        self.assertTrue(GOAL_PROMPT.is_file(), f"missing: {GOAL_PROMPT}")
        text = GOAL_PROMPT.read_text(encoding="utf-8")

        for marker in (
            "Do not print, commit, paste, or summarize token values.",
            "CODELEWM_HF_JOBS_DRY_RUN=0",
            "CODELEWM_HF_PUBLISH_DRY_RUN=0",
            "hf jobs logs <job-id>",
            "ml-intern --max-iterations -1",
            "downloaded checkpoint",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_next_goal_points_to_hf_goal_prompt(self) -> None:
        text = NEXT_GOAL.read_text(encoding="utf-8")

        self.assertIn("docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md", text)


if __name__ == "__main__":
    unittest.main()
