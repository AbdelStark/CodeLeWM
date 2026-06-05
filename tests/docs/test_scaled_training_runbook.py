from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = ROOT / "docs" / "training" / "SCALED_TRAINING_RUNBOOK.md"


class ScaledTrainingRunbookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = RUNBOOK.read_text(encoding="utf-8")

    def test_runbook_exists_and_lists_scaled_configs(self) -> None:
        self.assertTrue(RUNBOOK.is_file(), f"missing: {RUNBOOK}")
        for marker in (
            "config/train/scaled/codelewm_scaled_cpu.yaml",
            "config/train/scaled/codelewm_scaled_mps.yaml",
            "config/train/scaled/codelewm_scaled_gpu_a10g.yaml",
            "config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml",
            "config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml",
            "config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml",
            "config/data/codelewm_public_shard_commitpackft_python.json",
            "seed",
            "240119",
            "a10g-small",
            "action_use_margin_weight=0.25",
            "action_swap_contrastive_weight=0.20",
            "inverse_action_reconstruction_weight=0.10",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_runbook_documents_validation_resume_and_hf_cli_path(self) -> None:
        for marker in (
            "uv run scripts/validate-training-configs",
            "codelewm.train_config_validation.v1",
            "--resume-from",
            "codelewm manifest verify",
            "CODELEWM_HF_JOBS_TIMEOUT=24h",
            "hf jobs inspect <job-id>",
            "uv run scripts/hf-job-event-status <job-id>",
            "hf download",
            "bigcode/commitpackft",
            "data/python/data.jsonl",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_runbook_keeps_patch_action_diagnostic_only(self) -> None:
        self.assertIn("Patch-action remains diagnostic-only", self.text)
        self.assertIn("action_view=text", self.text)


if __name__ == "__main__":
    unittest.main()
