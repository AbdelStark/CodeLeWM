from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = ROOT / ".env.example"
RUNBOOK = ROOT / "docs" / "operations" / "HF_ML_INTERN_TRAINING.md"
GOAL_PROMPT = ROOT / "docs" / "roadmap" / "HF_ML_INTERN_GOAL_PROMPT.md"
NEXT_GOAL = ROOT / "docs" / "roadmap" / "NEXT_GOAL_PROMPT.md"
V0_2_PLAN = ROOT / "docs" / "roadmap" / "V0_2_ACTION_USE_RESEARCH_PLAN.md"
SCRIPTS = (
    ROOT / "scripts" / "hf-launch-codelewm-job",
    ROOT / "scripts" / "hf-run-codelewm-pipeline",
    ROOT / "scripts" / "hf-publish-codelewm-artifacts",
    ROOT / "scripts" / "hf-verify-codelewm-run",
    ROOT / "scripts" / "hf-job-event-status",
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
            "CODELEWM_HF_PRIVATE=0",
            "CODELEWM_HF_JOBS_DRY_RUN=1",
            "CODELEWM_HF_PUBLISH_DRY_RUN=1",
            "CODELEWM_HF_PIPELINE_MODE=smoke",
            "CODELEWM_DATASET_BUILD_CONFIG=",
            "CODELEWM_TRAIN_CONFIG=",
            "CODELEWM_HF_SCORER_QUALITY_CONFIG=",
            "CODELEWM_HF_RETRIEVAL_PRIOR_WEIGHT=",
            "CODELEWM_HF_INDEX_BATCH_SIZE=",
            "CODELEWM_HF_SOURCE_DATASET_REPO_ID=bigcode/commitpackft",
            "CODELEWM_HF_SOURCE_DATASET_PATH=data/python/data.jsonl",
            "CODELEWM_HF_SOURCE_LOCAL_DIR=.artifacts/hf-sources/commitpackft",
            "OPENROUTER_API_KEY=openrouter_xxx",
            "OPENROUTER_MANAGEMENT_KEY=openrouter_management_key_here",
            "ANTHROPIC_API_KEY=anthropic_provider_key_here",
            "CODELEWM_OPENROUTER_BYOK=0",
            "CODELEWM_OPENROUTER_BYOK_MANAGEMENT_KEY_ENV=OPENROUTER_MANAGEMENT_KEY",
            "CODELEWM_LLM_DEMO_ROOT=.artifacts/llm-world-model-demo",
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
            "scripts/hf-verify-codelewm-run",
            "scripts/hf-job-event-status",
            "CODELEWM_HF_JOBS_DRY_RUN=0",
            "CODELEWM_HF_PUBLISH_DRY_RUN=0",
            "CODELEWM_HF_PIPELINE_MODE=scaled",
            "CODELEWM_HF_INDEX_BATCH_SIZE=64",
            "config/data/codelewm_public_shard_commitpackft_python.json",
            "config/train/scaled/codelewm_scaled_gpu_a10g.yaml",
            "config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml",
            "config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml",
            "hf download bigcode/commitpackft",
            "data/python/data.jsonl",
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
            "CODELEWM_HF_PRIVATE=0",
            "config/data/codelewm_public_shard_commitpackft_python.json",
            "bigcode/commitpackft:data/python/data.jsonl",
            "config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml",
            "config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml",
            "CODELEWM_HF_INDEX_BATCH_SIZE=64",
            "hf jobs logs <job-id>",
            "ml-intern --max-iterations -1",
            "downloaded checkpoint",
            "#167",
            "V0_2_ACTION_USE_RESEARCH_PLAN.md",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_next_goal_points_to_hf_goal_prompt(self) -> None:
        text = NEXT_GOAL.read_text(encoding="utf-8")
        normalized = " ".join(text.split())

        self.assertIn("docs/spec/11-llm-world-model-harness.md", text)
        self.assertIn("docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md", text)
        self.assertIn("docs/rfcs/RFC-0015-v0-7-execution-substrate-improvements.md", text)
        self.assertIn("docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md", text)
        self.assertIn("docs/roadmap/MEANINGFUL_HARNESS_DEMO.md", text)
        self.assertIn("docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md", text)
        self.assertIn("active prompt for the final v1.0 paper/demo release tracker #401", text)
        self.assertIn("#402 - v1.0 hygiene", text)
        self.assertIn("#408 - v1.0 release", text)
        self.assertIn("Do not relaunch completed #159, #172, v0.8, or v0.9 HF Jobs.", text)
        self.assertIn("HumanEval WS-D downstream positive slice", text)
        self.assertIn("MBPP-Plus shows zero lift over no-action", text)
        self.assertIn("Do not claim CodeLeWM generally improves coding.", normalized)
        self.assertIn("Issue #408 is the final release package", normalized)

    def test_v0_2_plan_records_research_gates_and_public_hf_policy(self) -> None:
        self.assertTrue(V0_2_PLAN.is_file(), f"missing: {V0_2_PLAN}")
        text = V0_2_PLAN.read_text(encoding="utf-8")

        for marker in (
            "Current Evidence Boundary",
            "Action-Use Gate",
            "Representation Gate",
            "Downstream Reranking Gate",
            "The existing HF repositories are public",
            "action-conditioned failures do not invalidate the entire",
            "#167",
            "#168",
            "#169",
            "#170",
            "#171",
            "#172",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_launcher_dry_run_preserves_bash_login_command_boundary(self) -> None:
        completed = subprocess.run(
            ["uv", "run", "scripts/hf-launch-codelewm-job"],
            cwd=ROOT,
            check=False,
            env={
                **os.environ,
                "CODELEWM_HF_JOBS_DRY_RUN": "1",
                "CODELEWM_HF_PIPELINE_MODE": "scaled",
                "CODELEWM_HF_JOBS_TIMEOUT": "24h",
                "CODELEWM_HF_PUBLISH_DRY_RUN": "0",
                "CODELEWM_HF_PRIVATE": "0",
                "CODELEWM_HF_REF": "main",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(" -- python:3.13-bookworm bash -lc ", completed.stdout)
        self.assertIn("CODELEWM_HF_INDEX_BATCH_SIZE=64", completed.stdout)
        self.assertIn("CODELEWM_HF_PRIVATE=0", completed.stdout)
        self.assertNotIn(" --label c ", completed.stdout)

    def test_download_verifier_dry_run_is_secret_safe_and_complete(self) -> None:
        completed = subprocess.run(
            [
                "uv",
                "run",
                "scripts/hf-verify-codelewm-run",
                "--run-id",
                "codelewm-test-run",
                "--results-repo-id",
                "owner/results",
                "--model-repo-id",
                "owner/model",
                "--dataset-repo-id",
                "owner/dataset",
                "--dry-run",
                "--json",
            ],
            cwd=ROOT,
            check=False,
            env={**os.environ, "HF_TOKEN": "hf_should_not_appear"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("hf_should_not_appear", completed.stdout)
        self.assertNotIn("HF_TOKEN", completed.stdout)

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "codelewm.hf_download_verification_plan.v1")
        self.assertTrue(payload["dry_run"])
        command_names = {command["name"] for command in payload["commands"]}
        for marker in (
            "download_results",
            "download_model",
            "download_dataset_pack",
            "verify_dataset_pack",
            "verify_model",
            "eval_retrieval",
            "eval_latent_probe",
            "verify_latent_probe",
            "eval_ablation",
            "eval_surprise",
            "verify_index",
            "eval_scorer_quality",
            "score_smoke",
            "rerank_smoke",
            "secret_scan",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, command_names)


if __name__ == "__main__":
    unittest.main()
