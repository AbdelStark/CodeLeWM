from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_CHECKLIST = ROOT / "docs" / "release" / "RELEASE_CHECKLIST.md"
RELEASE_FREEZE = ROOT / "docs" / "release" / "RELEASE_FREEZE_2026-05-20.md"
DATASET_CARD_TEMPLATE = ROOT / "docs" / "cards" / "DATASET_CARD_TEMPLATE.md"
MODEL_CARD_TEMPLATE = ROOT / "docs" / "cards" / "MODEL_CARD_TEMPLATE.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class ReleaseChecklistExistenceTest(unittest.TestCase):
    def test_release_checklist_file_exists_and_is_substantive(self) -> None:
        self.assertTrue(RELEASE_CHECKLIST.is_file(), f"missing: {RELEASE_CHECKLIST}")
        self.assertGreater(len(_read(RELEASE_CHECKLIST)), 2500)

    def test_release_checklist_has_required_sections(self) -> None:
        text = _read(RELEASE_CHECKLIST)
        for heading in (
            "# CodeLeWM Release Checklist",
            "## Pre-Flight",
            "## Tests",
            "## Manifests",
            "## Benchmark Evidence",
            "## Security Evidence",
            "## License Evidence",
            "## Cards And Documentation",
            "## Reproducibility",
            "## Communications",
            "## Sign-off",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, text)


class ReleaseChecklistContentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _read(RELEASE_CHECKLIST)

    def test_release_checklist_uses_only_unchecked_boxes(self) -> None:
        unchecked = self.text.count("- [ ]")
        self.assertGreaterEqual(unchecked, 20)
        self.assertEqual(self.text.count("- [x]"), 0)

    def test_release_checklist_pins_artifact_schema_versions(self) -> None:
        for schema in (
            "codelewm.dataset.v1",
            "codelewm.training_run.v1",
            "codelewm.checkpoint.v1",
            "codelewm.eval.retrieval_report.v1",
            "codelewm.eval.action_ablation_report.v1",
            "codelewm.eval.surprise_report.v1",
            "codelewm.harness.scorer_quality_report.v1",
            "codelewm.public_license_gate.v1",
        ):
            with self.subTest(schema=schema):
                self.assertIn(schema, self.text)

    def test_release_checklist_requires_security_evidence(self) -> None:
        for marker in (
            "codelewm manifest verify",
            "codelewm secret-scan",
            "pip-audit",
            "codelewm.release_provenance.v1",
            "checkpoint_error",
            "allow-unsafe-checkpoint",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_release_checklist_links_dataset_card_and_model_card(self) -> None:
        self.assertIn("DATASET_CARD_TEMPLATE.md", self.text)
        self.assertIn("MODEL_CARD_TEMPLATE.md", self.text)

    def test_release_checklist_requires_three_sign_offs(self) -> None:
        sign_off_section = self.text.split("## Sign-off", 1)[-1]
        for role in ("Release shepherd", "Codeowner", "Security reviewer"):
            with self.subTest(role=role):
                self.assertIn(role, sign_off_section)


class ReleaseFreezeReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _read(RELEASE_FREEZE)

    def test_release_freeze_report_exists(self) -> None:
        self.assertTrue(RELEASE_FREEZE.is_file())

    def test_release_freeze_records_diagnostic_boundary(self) -> None:
        for marker in (
            "private diagnostic artifact freeze",
            "blocked for public positive action-conditioning claims",
            "#159",
            "claim_allowed=false",
            "no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_release_freeze_records_package_and_provenance_evidence(self) -> None:
        for marker in (
            "codelewm-0.0.0-py3-none-any.whl",
            "codelewm-0.0.0.tar.gz",
            "codelewm.release_provenance.v1",
            "tracked_git_dirty=false",
            "pip-audit",
            "0 known vulnerabilities",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_release_freeze_records_hf_artifact_and_security_gates(self) -> None:
        for marker in (
            "codelewm-action-use-20260520-6650183",
            "6a0d7a763aba298b21d147a9",
            "abdelstark/codelewm-public-shard",
            "abdelstark/codelewm-transition-model",
            "hf auth whoami",
            "release_allowed=true",
            "checkpoint_error",
            "secret-scan",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)


class DatasetCardTemplateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _read(DATASET_CARD_TEMPLATE)

    def test_dataset_card_file_exists(self) -> None:
        self.assertTrue(DATASET_CARD_TEMPLATE.is_file())

    def test_dataset_card_lists_required_sections(self) -> None:
        for heading in (
            "## Summary",
            "## Source Mix",
            "## Schema Versions",
            "## Row Counts",
            "## License Policy",
            "## Curation Procedure",
            "## Known Limitations",
            "## Reproduction",
            "## Sign-off",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, self.text)

    def test_dataset_card_pins_required_schema_versions(self) -> None:
        for schema in (
            "codelewm.dataset.v1",
            "codelewm.artifact_manifest.v1",
            "codelewm.source_acquisition.v1",
            "codelewm.public_license_gate.v1",
            "codelewm.transition.v1",
        ):
            with self.subTest(schema=schema):
                self.assertIn(schema, self.text)

    def test_dataset_card_lists_permissive_license_set(self) -> None:
        for license_id in (
            "apache-2.0",
            "bsd-2-clause",
            "bsd-3-clause",
            "cc0-1.0",
            "isc",
            "mit",
            "unlicense",
        ):
            with self.subTest(license=license_id):
                self.assertIn(license_id, self.text)


class ModelCardTemplateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _read(MODEL_CARD_TEMPLATE)

    def test_model_card_file_exists(self) -> None:
        self.assertTrue(MODEL_CARD_TEMPLATE.is_file())

    def test_model_card_lists_required_sections(self) -> None:
        for heading in (
            "## Summary",
            "## Schema Versions",
            "## Architecture",
            "## Training",
            "## Intended Use",
            "## Out-of-Scope Use",
            "## Evaluation Evidence",
            "## Limitations And Risks",
            "## Reproduction",
            "## Sign-off",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, self.text)

    def test_model_card_requires_retrieval_and_surprise_evidence_sections(self) -> None:
        self.assertIn("### Retrieval (headline)", self.text)
        self.assertIn("### Patch Surprise", self.text)
        self.assertIn("### Action-View Diagnostic", self.text)
        self.assertIn("### Scorer / Reranker Quality", self.text)

    def test_model_card_pins_required_schema_versions(self) -> None:
        for schema in (
            "codelewm.checkpoint.v1",
            "codelewm.training_run.v1",
            "codelewm.artifact_manifest.v1",
            "codelewm.eval.retrieval_report.v1",
            "codelewm.eval.action_ablation_report.v1",
            "codelewm.eval.surprise_report.v1",
            "codelewm.harness.scorer_quality_report.v1",
        ):
            with self.subTest(schema=schema):
                self.assertIn(schema, self.text)

    def test_model_card_states_out_of_scope_uses(self) -> None:
        for forbidden in (
            "Generating new code",
            "Running candidate code through tests",
            "Modifying the user's working tree",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertIn(forbidden, self.text)


if __name__ == "__main__":
    unittest.main()
