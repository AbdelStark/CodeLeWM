from __future__ import annotations

import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "first-results"
CONFIG_DIR = ROOT / "config" / "first_results"
REPORT = ROOT / "docs" / "benchmark" / "FIRST_RESULTS.md"


class FirstResultsRunnerContractTest(unittest.TestCase):
    def test_runner_and_config_bundle_exist(self) -> None:
        self.assertTrue(RUNNER.is_file(), f"missing: {RUNNER}")
        self.assertTrue(os.access(RUNNER, os.X_OK), "scripts/first-results must be executable")
        self.assertTrue((CONFIG_DIR / "dataset_build.json").is_file())
        self.assertTrue((CONFIG_DIR / "train_tiny.json").is_file())
        self.assertTrue((CONFIG_DIR / "scorer_quality.json").is_file())
        self.assertTrue((CONFIG_DIR / "scorer_quality_candidates").is_dir())

    def test_config_bundle_pins_fixture_and_tiny_torch_run(self) -> None:
        dataset_config = json.loads((CONFIG_DIR / "dataset_build.json").read_text(encoding="utf-8"))
        train_config = json.loads((CONFIG_DIR / "train_tiny.json").read_text(encoding="utf-8"))

        self.assertEqual(dataset_config["schema_version"], "codelewm.dataset_build_config.v1")
        self.assertEqual(dataset_config["seed"], 7)
        self.assertIn("tests/fixtures/dataset_build/records.jsonl", dataset_config["sources"][0]["path"])
        self.assertEqual(train_config["schema_version"], "codelewm.train_config.v1")
        self.assertEqual(train_config["seed"], 1337)
        self.assertEqual(train_config["wm"]["action_view"], "text")
        self.assertEqual(train_config["trainer"]["accelerator"], "cpu")
        self.assertGreaterEqual(train_config["trainer"]["max_steps"], 1)


class FirstResultsReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(REPORT.is_file(), f"missing: {REPORT}")
        self.text = REPORT.read_text(encoding="utf-8")

    def test_report_has_required_sections(self) -> None:
        for heading in (
            "# CodeLeWM First Results",
            "## Verdict",
            "## Reproduce",
            "## Artifact Cards",
            "## Exact Commands",
            "## Reproducibility Chain",
            "## Manifest Verification",
            "## Retrieval Evaluation",
            "### Action-Use Claim Gate",
            "## Action-View Ablation",
            "## Patch-Surprise Evaluation",
            "## Scorer And Reranker Quality",
            "## Security Evidence",
            "## Claim Checklist",
            "## Caveats",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, self.text)

    def test_report_records_reproducible_first_results_command(self) -> None:
        for marker in (
            "uv run scripts/first-results --overwrite",
            "uv run codelewm dataset build",
            "uv run codelewm dataset pack",
            "uv run codelewm train",
            "uv run codelewm eval retrieval",
            "uv run codelewm eval ablation",
            "uv run codelewm eval surprise",
            "uv run codelewm index",
            "uv run codelewm eval scorer-quality",
            "uv run codelewm manifest verify",
            "uv run codelewm secret-scan",
            ".artifacts/first-results/manifest_inventory.json",
            "docs/cards/codelewm-first-results-dataset-2026-05-19.md",
            "docs/cards/codelewm-first-results-model-2026-05-19.md",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_report_keeps_smoke_evidence_separate_from_research_claims(self) -> None:
        self.assertIn("Evidence tier: smoke fixture, not scaled research evidence", self.text)
        self.assertIn("Text-action does not beat all required baselines on this fixture", self.text)
        self.assertIn("Positive action-conditioning claim allowed: `false`", self.text)
        self.assertIn("Action-use claim gate allows a positive action-conditioning claim", self.text)
        self.assertIn("Action-discriminative shard report:", self.text)
        self.assertIn("Dataset shard has action-discriminative hard-negative coverage", self.text)
        self.assertIn("- [ ] This report supports a scaled research claim", self.text)
        self.assertIn("- [x] Every selected artifact manifest verifies", self.text)
        self.assertIn("- [x] Secret scan passes", self.text)

    def test_report_names_required_artifacts_and_schemas(self) -> None:
        for marker in (
            "dataset_build",
            "dataset_pack",
            "training_run",
            "retrieval_eval",
            "action_ablation",
            "surprise_eval",
            "transition_index",
            "scorer_quality",
            "codelewm.first_results.v1",
            "codelewm.eval.retrieval_report.v1",
            "codelewm.eval.action_ablation_report.v1",
            "codelewm.eval.surprise_report.v1",
            "codelewm.harness.scorer_quality_report.v1",
            "codelewm.data.action_discriminative_shard_report.v1",
            "codelewm.public_license_gate.v1",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)


if __name__ == "__main__":
    unittest.main()
