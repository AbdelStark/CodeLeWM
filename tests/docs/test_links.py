from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs" / "benchmark"
TEMPLATE = BENCHMARK_DIR / "REPORT_TEMPLATE.md"
TESTING_STRATEGY = ROOT / "docs" / "spec" / "07-testing-strategy.md"

_RFC_RE = re.compile(r"`?(?P<rfc>RFC-\d{4})`?")
_REL_PATH_RE = re.compile(r"`(docs/[^`]+\.md)`")


class BenchmarkTemplateExistenceTest(unittest.TestCase):
    def test_benchmark_template_is_present_and_non_trivial(self) -> None:
        self.assertTrue(TEMPLATE.is_file(), f"missing: {TEMPLATE}")
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertGreater(len(text), 1500, "template must be more than a stub")

    def test_template_has_required_sections(self) -> None:
        text = TEMPLATE.read_text(encoding="utf-8")
        for heading in (
            "# CodeLeWM Benchmark Report Template",
            "## Reproducibility Chain",
            "## Retrieval Evaluation",
            "## Patch-Surprise Evaluation",
            "## License And Source Policy",
            "## Claim Checklist",
            "## Caveats",
            "## Sign-off",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, text)


class BenchmarkTemplateContentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = TEMPLATE.read_text(encoding="utf-8")

    def test_template_lists_required_baselines_for_headline_retrieval(self) -> None:
        for baseline in (
            "Random",
            "Lexical",
            "No-action",
            "Shuffled-action",
            "Patch-action",
        ):
            with self.subTest(baseline=baseline):
                self.assertIn(baseline, self.text)

    def test_template_lists_required_surprise_decoy_categories(self) -> None:
        for category in ("random", "same_file", "mutation", "action_cluster"):
            with self.subTest(category=category):
                self.assertIn(category, self.text)

    def test_template_pins_required_schema_versions(self) -> None:
        for schema in (
            "codelewm.dataset.v1",
            "codelewm.training_run.v1",
            "codelewm.checkpoint.v1",
            "codelewm.eval.retrieval_report.v1",
            "codelewm.eval.surprise_report.v1",
            "codelewm.public_license_gate.v1",
        ):
            with self.subTest(schema=schema):
                self.assertIn(schema, self.text)

    def test_template_references_manifest_verify_and_secret_scan_commands(self) -> None:
        self.assertIn("codelewm manifest verify", self.text)
        self.assertIn("codelewm secret-scan", self.text)

    def test_template_claim_checklist_uses_unchecked_boxes(self) -> None:
        unchecked = self.text.count("- [ ]")
        self.assertGreaterEqual(
            unchecked,
            6,
            "claim checklist must require ticking each claim by the reviewer",
        )
        self.assertEqual(
            self.text.count("- [x]"),
            0,
            "checked claims must not appear in the template",
        )


class BenchmarkTemplateLinkConsistencyTest(unittest.TestCase):
    def test_every_referenced_spec_doc_exists(self) -> None:
        text = TEMPLATE.read_text(encoding="utf-8")
        for relative_path in _REL_PATH_RE.findall(text):
            if "<" in relative_path or ">" in relative_path:
                # Placeholder paths such as `docs/benchmark/<release>-<date>.md`
                # are not real files; the template hints at the convention.
                continue
            with self.subTest(relative_path=relative_path):
                full_path = ROOT / relative_path
                self.assertTrue(
                    full_path.is_file(),
                    f"docs/benchmark/REPORT_TEMPLATE.md references missing file: {relative_path}",
                )

    def test_testing_strategy_does_not_contradict_template(self) -> None:
        strategy_text = TESTING_STRATEGY.read_text(encoding="utf-8")

        self.assertIn("Recall@1", strategy_text)
        self.assertIn("MRR", strategy_text)
        self.assertIn("patch-surprise", strategy_text.lower())


if __name__ == "__main__":
    unittest.main()
