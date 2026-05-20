from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_CARD = ROOT / "docs" / "cards" / "codelewm-first-results-dataset-2026-05-19.md"
MODEL_CARD = ROOT / "docs" / "cards" / "codelewm-first-results-model-2026-05-19.md"
FIRST_RESULTS = ROOT / "docs" / "benchmark" / "FIRST_RESULTS.md"
SCALED_DATASET_CARD = ROOT / "docs" / "cards" / "codelewm-scaled-dataset-2026-05-20.md"
SCALED_MODEL_CARD = ROOT / "docs" / "cards" / "codelewm-scaled-model-2026-05-20.md"
SCALED_RESULTS = ROOT / "docs" / "benchmark" / "SCALED_HF_RESULTS_2026-05-20.md"


class FirstResultsArtifactCardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = DATASET_CARD.read_text(encoding="utf-8")
        self.model = MODEL_CARD.read_text(encoding="utf-8")
        self.report = FIRST_RESULTS.read_text(encoding="utf-8")
        self.scaled_dataset = SCALED_DATASET_CARD.read_text(encoding="utf-8")
        self.scaled_model = SCALED_MODEL_CARD.read_text(encoding="utf-8")
        self.scaled_report = SCALED_RESULTS.read_text(encoding="utf-8")

    def test_cards_exist_and_are_not_templates(self) -> None:
        for path, text in ((DATASET_CARD, self.dataset), (MODEL_CARD, self.model)):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"missing: {path}")
                self.assertNotIn("<", text)
                self.assertNotIn(">", text)
                self.assertNotIn("TODO", text)
                self.assertNotIn("Copy this file", text)

        for path, text in (
            (SCALED_DATASET_CARD, self.scaled_dataset),
            (SCALED_MODEL_CARD, self.scaled_model),
            (SCALED_RESULTS, self.scaled_report),
        ):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"missing: {path}")
                self.assertNotIn("TODO", text)
                self.assertNotIn("Copy this file", text)

    def test_first_results_report_links_filled_cards(self) -> None:
        self.assertIn("docs/cards/codelewm-first-results-dataset-2026-05-19.md", self.report)
        self.assertIn("docs/cards/codelewm-first-results-model-2026-05-19.md", self.report)

    def test_dataset_card_matches_source_and_license_evidence(self) -> None:
        self.assertIn("| `local_repo` | 4 | 3 | 1 |", self.dataset)
        self.assertIn("| Included rows | 3 |", self.dataset)
        self.assertIn("| Excluded rows | 1 |", self.dataset)
        self.assertIn("| Blocked rows | 0 |", self.dataset)
        self.assertIn("release_allowed=true", self.dataset)

        license_line = re.search(
            r"License gate: release_allowed `true`, included rows `(?P<included>\d+)`, "
            r"excluded rows `(?P<excluded>\d+)`, blocked rows `(?P<blocked>\d+)`",
            self.report,
        )
        self.assertIsNotNone(license_line)
        assert license_line is not None
        self.assertIn(f"| Included rows | {license_line.group('included')} |", self.dataset)
        self.assertIn(f"| Excluded rows | {license_line.group('excluded')} |", self.dataset)
        self.assertIn(f"| Blocked rows | {license_line.group('blocked')} |", self.dataset)

    def test_model_card_metrics_match_first_results_report(self) -> None:
        self.assertIn("| Recall@1 | 1.0 | `metrics.recall_at_1` |", self.model)
        self.assertIn("| MRR | 1.0 | `metrics.mrr` |", self.model)
        self.assertIn("| Pairwise AUC overall | 0.0 | `metrics.pairwise_auc_overall` |", self.model)
        self.assertIn("| Error candidates | 2 | `summary.error_count` |", self.model)
        self.assertIn("`invalid_syntax=1`, `patch_apply_failed=1`", self.model)

        self.assertIn("Recall@1 `1`", self.report)
        self.assertIn("MRR `1`", self.report)
        self.assertIn("Overall pairwise AUC: `0`", self.report)
        self.assertIn("Failure counts: `{\"invalid_syntax\": 1, \"patch_apply_failed\": 1}`", self.report)

    def test_model_card_separates_evidence_tiers(self) -> None:
        for marker in (
            "| Smoke evidence | present |",
            "| First-results evidence | present |",
            "| Scaled evidence | absent |",
            "not a public model release",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.model)

    def test_scaled_cards_match_verified_hf_report(self) -> None:
        for marker in (
            "codelewm-scaled-20260520-9699b53",
            "dataset-ef8ad3f4f48dea9e",
            "training_run-d9074199c0d58911",
            "09bf8d3880ec272a858dd9b19f2b29622a66a5ebbef6dbd1f8e4ebeb8b6392b8",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.scaled_report)

        self.assertIn("| Text action | 0.371 | 0.586 | 0.672 | 0.472984 |", self.scaled_model)
        self.assertIn("| No action | 0.459 | 0.641 | 0.712 | 0.546116 |", self.scaled_model)
        self.assertIn("## Action-Use Claim Gate", self.scaled_report)
        self.assertIn("claim_allowed=false", self.scaled_report)
        self.assertIn("no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action", self.scaled_report)
        self.assertIn("Text action beats the no-action baseline.", self.scaled_report)
        self.assertIn("- [ ] Text action beats the no-action baseline.", self.scaled_report)
        self.assertIn("| `commitpackft` | 56,025 | 23,015 | 33,010 |", self.scaled_dataset)


if __name__ == "__main__":
    unittest.main()
