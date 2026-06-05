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
ACTION_USE_DATASET_CARD = ROOT / "docs" / "cards" / "codelewm-action-use-dataset-2026-05-20.md"
ACTION_USE_MODEL_CARD = ROOT / "docs" / "cards" / "codelewm-action-use-model-2026-05-20.md"
ACTION_USE_RESULTS = ROOT / "docs" / "benchmark" / "ACTION_USE_HF_RESULTS_2026-05-20.md"
ACTION_USE_RETRIEVAL_DATASET_CARD = (
    ROOT / "docs" / "cards" / "codelewm-action-use-retrieval-dataset-2026-05-20.md"
)
ACTION_USE_RETRIEVAL_MODEL_CARD = (
    ROOT / "docs" / "cards" / "codelewm-action-use-retrieval-model-2026-05-20.md"
)
ACTION_USE_RETRIEVAL_RESULTS = ROOT / "docs" / "benchmark" / "ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md"
V0_2_ACTION_SWAP_DATASET_CARD = (
    ROOT / "docs" / "cards" / "codelewm-v0-2-action-swap-dataset-2026-05-20.md"
)
V0_2_ACTION_SWAP_MODEL_CARD = ROOT / "docs" / "cards" / "codelewm-v0-2-action-swap-model-2026-05-20.md"
V0_2_ACTION_SWAP_RESULTS = ROOT / "docs" / "benchmark" / "V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md"
V0_8_DATASET_CARD = ROOT / "docs" / "cards" / "codelewm-v0-8-execution-dataset-2026-06-05.md"
V0_8_MODEL_CARD_SEED_42 = ROOT / "docs" / "cards" / "codelewm-v0-8-execution-model-seed-42-2026-06-05.md"
V0_8_MODEL_CARD_SEED_1729 = ROOT / "docs" / "cards" / "codelewm-v0-8-execution-model-seed-1729-2026-06-05.md"
V0_8_RESULTS = ROOT / "docs" / "benchmark" / "EXECUTION_V0_8_RESULTS_2026-06-05.md"
V0_8_ARTIFACT_INDEX = ROOT / "docs" / "benchmark" / "PUBLIC_ARTIFACT_INDEX_2026-06-05.md"


class FirstResultsArtifactCardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = DATASET_CARD.read_text(encoding="utf-8")
        self.model = MODEL_CARD.read_text(encoding="utf-8")
        self.report = FIRST_RESULTS.read_text(encoding="utf-8")
        self.scaled_dataset = SCALED_DATASET_CARD.read_text(encoding="utf-8")
        self.scaled_model = SCALED_MODEL_CARD.read_text(encoding="utf-8")
        self.scaled_report = SCALED_RESULTS.read_text(encoding="utf-8")
        self.action_use_dataset = ACTION_USE_DATASET_CARD.read_text(encoding="utf-8")
        self.action_use_model = ACTION_USE_MODEL_CARD.read_text(encoding="utf-8")
        self.action_use_report = ACTION_USE_RESULTS.read_text(encoding="utf-8")
        self.action_use_retrieval_dataset = ACTION_USE_RETRIEVAL_DATASET_CARD.read_text(encoding="utf-8")
        self.action_use_retrieval_model = ACTION_USE_RETRIEVAL_MODEL_CARD.read_text(encoding="utf-8")
        self.action_use_retrieval_report = ACTION_USE_RETRIEVAL_RESULTS.read_text(encoding="utf-8")
        self.v0_2_action_swap_dataset = V0_2_ACTION_SWAP_DATASET_CARD.read_text(encoding="utf-8")
        self.v0_2_action_swap_model = V0_2_ACTION_SWAP_MODEL_CARD.read_text(encoding="utf-8")
        self.v0_2_action_swap_report = V0_2_ACTION_SWAP_RESULTS.read_text(encoding="utf-8")
        self.v0_8_dataset = V0_8_DATASET_CARD.read_text(encoding="utf-8")
        self.v0_8_model_seed_42 = V0_8_MODEL_CARD_SEED_42.read_text(encoding="utf-8")
        self.v0_8_model_seed_1729 = V0_8_MODEL_CARD_SEED_1729.read_text(encoding="utf-8")
        self.v0_8_report = V0_8_RESULTS.read_text(encoding="utf-8")
        self.v0_8_artifact_index = V0_8_ARTIFACT_INDEX.read_text(encoding="utf-8")

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
            (ACTION_USE_DATASET_CARD, self.action_use_dataset),
            (ACTION_USE_MODEL_CARD, self.action_use_model),
            (ACTION_USE_RESULTS, self.action_use_report),
            (ACTION_USE_RETRIEVAL_DATASET_CARD, self.action_use_retrieval_dataset),
            (ACTION_USE_RETRIEVAL_MODEL_CARD, self.action_use_retrieval_model),
            (ACTION_USE_RETRIEVAL_RESULTS, self.action_use_retrieval_report),
            (V0_2_ACTION_SWAP_DATASET_CARD, self.v0_2_action_swap_dataset),
            (V0_2_ACTION_SWAP_MODEL_CARD, self.v0_2_action_swap_model),
            (V0_2_ACTION_SWAP_RESULTS, self.v0_2_action_swap_report),
            (V0_8_DATASET_CARD, self.v0_8_dataset),
            (V0_8_MODEL_CARD_SEED_42, self.v0_8_model_seed_42),
            (V0_8_MODEL_CARD_SEED_1729, self.v0_8_model_seed_1729),
            (V0_8_RESULTS, self.v0_8_report),
            (V0_8_ARTIFACT_INDEX, self.v0_8_artifact_index),
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

    def test_action_use_cards_match_verified_hf_report(self) -> None:
        for marker in (
            "codelewm-action-use-20260520-6650183",
            "dataset-67895f8dc3e217c4",
            "training_run-ce98fe8768af2143",
            "1e361498c722893c9754abcc9c2efa4499a615590572b77c7f0de939e789ac66",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.action_use_report)

        self.assertIn("| Text action | 0.363 | 0.589 | 0.673 | 0.467875 |", self.action_use_model)
        self.assertIn("| No action | 0.469 | 0.640 | 0.700 | 0.549624 |", self.action_use_model)
        self.assertIn("## Action-Use Claim Gate", self.action_use_report)
        self.assertIn("claim_allowed=false", self.action_use_report)
        self.assertIn(
            "no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action",
            self.action_use_report,
        )
        self.assertIn("- [ ] Text action beats the no-action baseline.", self.action_use_report)
        self.assertIn("Claim-readiness gate | true", self.action_use_dataset)

    def test_action_use_retrieval_cards_match_verified_hf_report(self) -> None:
        for marker in (
            "codelewm-action-use-retrieval-20260520-7895d18",
            "dataset-5695087296ce4a97",
            "training_run-924cd056375f11ea",
            "0cb4daf1500495579f5c59cc9fd8aa39f5f70e88f55c0c121320d023b43ddeda",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.action_use_retrieval_report)

        self.assertIn("| Text action | 0.597 | 0.770 | 0.813 | 0.674500 |", self.action_use_retrieval_model)
        self.assertIn("| No action | 0.650 | 0.774 | 0.816 | 0.708037 |", self.action_use_retrieval_model)
        self.assertIn("## Action-Use Claim Gate", self.action_use_retrieval_report)
        self.assertIn("claim_allowed=false", self.action_use_retrieval_report)
        self.assertIn(
            "no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action",
            self.action_use_retrieval_report,
        )
        self.assertIn("- [ ] Text action beats the no-action baseline.", self.action_use_retrieval_report)
        self.assertIn("Claim-readiness gate | true", self.action_use_retrieval_dataset)

    def test_v0_2_action_swap_cards_match_verified_hf_report(self) -> None:
        for marker in (
            "codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b",
            "dataset-daecac9f9965c563",
            "training_run-0a41863d1da33737",
            "f2c5ba50ee0ec5e32ff5c3ceed848020e989ebdb1c98a917f17589ee523c6d7e",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.v0_2_action_swap_report)

        self.assertIn("| Text action | 0.263 | 0.478 | 0.596 | 0.370048 |", self.v0_2_action_swap_model)
        self.assertIn("| No action | 0.441 | 0.638 | 0.712 | 0.533105 |", self.v0_2_action_swap_model)
        self.assertIn("## Action-Contrast Retrieval Gate", self.v0_2_action_swap_report)
        self.assertIn("claim_allowed=false", self.v0_2_action_swap_report)
        self.assertIn(
            "no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action",
            self.v0_2_action_swap_report,
        )
        self.assertIn("- [ ] Text action beats no-action on headline retrieval.", self.v0_2_action_swap_report)
        self.assertIn("Selected train rows | 0", self.v0_2_action_swap_dataset)
        self.assertIn("semantic_structure_status=unsupported", self.v0_2_action_swap_model)

    def test_v0_8_cards_match_results_claim_boundary(self) -> None:
        for marker in (
            "codelewm-v0-8-short-execution-20260605-1b737e4-seed-42",
            "codelewm-v0-8-short-execution-20260605-1b737e4-seed-1729",
            "training_run-e2a757caf75cbcf2",
            "training_run-951983cbf59f6fa6",
            "HumanEval WS-D rerank | PASS",
            "MBPP-Plus WS-D rerank | NOT MET",
            "Overall v0.8 downstream claim | CLOSED",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.v0_8_report)

        self.assertIn("Records | 1,882", self.v0_8_dataset)
        self.assertIn("val / test | 1,646 / 51 / 185", self.v0_8_dataset)
        self.assertIn("zero magnitude-labeled rows", self.v0_8_dataset)
        self.assertIn("HumanEval WS-D positive", self.v0_8_model_seed_42)
        self.assertIn("HumanEval WS-D positive", self.v0_8_model_seed_1729)
        self.assertIn("overall downstream claim closed", self.v0_8_model_seed_42)
        self.assertIn("overall downstream claim closed", self.v0_8_model_seed_1729)
        self.assertIn("Completion-level ROC-AUC diagnostic | HumanEval `0.9622`; MBPP-Plus `0.5772`", self.v0_8_model_seed_42)
        self.assertIn("Completion-level ROC-AUC diagnostic | HumanEval `0.9699`; MBPP-Plus `0.6941`", self.v0_8_model_seed_1729)
        self.assertIn("benchmark-specific diagnostic result", self.v0_8_artifact_index)


if __name__ == "__main__":
    unittest.main()
