from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
USAGE = ROOT / "docs" / "usage" / "USAGE.md"
PUBLIC_API_SPEC = ROOT / "docs" / "spec" / "02-public-api.md"
README = ROOT / "README.md"

_BACKTICK_PATH_RE = re.compile(r"`(docs/[^`]+\.md)`")
_BARE_PATH_RE = re.compile(r"\bdocs/[A-Za-z0-9_./-]+\.md\b")


class UsageGuideExistenceTest(unittest.TestCase):
    def test_usage_doc_exists_and_is_substantive(self) -> None:
        self.assertTrue(USAGE.is_file(), f"missing: {USAGE}")
        self.assertGreater(len(USAGE.read_text(encoding="utf-8")), 2500)

    def test_readme_links_to_usage_guide(self) -> None:
        text = README.read_text(encoding="utf-8")
        self.assertIn("docs/usage/USAGE.md", text)


class UsageGuideContentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = USAGE.read_text(encoding="utf-8")

    def test_usage_doc_covers_install_path(self) -> None:
        self.assertRegex(self.text, r"##\s+Install")
        self.assertIn("uv sync --group dev", self.text)
        self.assertIn("uv sync --group dev --group data", self.text)

    def test_usage_doc_documents_score_and_rerank_commands(self) -> None:
        for command in ("codelewm score", "codelewm rerank"):
            with self.subTest(command=command):
                self.assertIn(command, self.text)

    def test_usage_doc_documents_landed_commands(self) -> None:
        for command in (
            "codelewm dataset build",
            "codelewm dataset pack",
            "codelewm train",
            "codelewm eval retrieval",
            "codelewm eval latent-probe",
            "codelewm eval ablation",
            "codelewm eval surprise",
            "codelewm eval scorer-quality",
            "codelewm index",
        ):
            with self.subTest(command=command):
                self.assertIn(command, self.text)

    def test_usage_doc_lists_python_api_example_for_load_scorer(self) -> None:
        self.assertIn("load_scorer", self.text)
        self.assertIn("from codelewm.harness import load_scorer", self.text)

    def test_usage_doc_lists_eval_apis(self) -> None:
        for symbol in (
            "build_retrieval_report",
            "build_surprise_report",
            "score_surprise_example",
            "validate_required_headline_baselines",
        ):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, self.text)

    def test_usage_doc_pins_required_schema_versions(self) -> None:
        for schema in (
            "codelewm.score.v1",
            "codelewm.rerank.v1",
            "codelewm.error.v1",
            "codelewm.artifact_manifest.v1",
            "codelewm.training_run.v1",
            "codelewm.index_build.v1",
            "codelewm.eval.retrieval_report.v1",
            "codelewm.eval.action_contrast_pool_report.v1",
            "codelewm.eval.latent_probe_report.v1",
            "codelewm.eval.latent_probe_run.v1",
            "codelewm.eval.action_ablation_report.v1",
            "codelewm.eval.action_ablation_run.v1",
            "codelewm.eval.surprise_report.v1",
            "codelewm.harness.scorer_quality_config.v1",
            "codelewm.harness.scorer_quality_report.v1",
            "codelewm.harness.scorer_quality_run.v1",
            "codelewm.transition_index.v1",
            "codelewm.secret_scan.v1",
            "codelewm.release_provenance.v1",
            "benchmark_readiness",
            "baseline_controls",
            "transition_energy_only",
            "retrieval_prior_only",
        ):
            with self.subTest(schema=schema):
                self.assertIn(schema, self.text)

    def test_usage_doc_keeps_evidence_boundary_explicit(self) -> None:
        for marker in (
            "## Evidence Boundary",
            "docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md",
            "docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md",
            "docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md",
            "docs/benchmark/DOWNSTREAM_RERANKING_BENCHMARK.md",
            "text-action still loses to no-action",
            "positive action-conditioned quality claim",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_usage_doc_marks_legacy_root_scripts_as_compatibility_only(self) -> None:
        for marker in (
            "## Legacy Compatibility Scripts",
            "Root `train.py`, root `eval.py`",
            "not the CodeLeWM first-results or scaled-artifact path",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_usage_doc_links_security_and_changelog_anchors(self) -> None:
        self.assertIn("docs/spec/06-security.md", self.text)
        self.assertIn("CHANGELOG.md", self.text)

    def test_usage_doc_states_trust_boundary_summary(self) -> None:
        self.assertIn("non-execution", self.text.lower())
        self.assertIn("require_trusted_checkpoint", self.text)
        self.assertIn("scan_paths", self.text)


class UsageGuideLinkConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = USAGE.read_text(encoding="utf-8")

    def test_every_referenced_doc_file_exists(self) -> None:
        candidates = set(_BACKTICK_PATH_RE.findall(self.text))
        candidates.update(_BARE_PATH_RE.findall(self.text))
        self.assertGreater(len(candidates), 0)
        for relative_path in candidates:
            if "<" in relative_path or ">" in relative_path:
                continue
            with self.subTest(relative_path=relative_path):
                self.assertTrue(
                    (ROOT / relative_path).is_file(),
                    f"docs/usage/USAGE.md references missing file: {relative_path}",
                )

    def test_public_api_spec_lists_the_same_command_surface(self) -> None:
        spec_text = PUBLIC_API_SPEC.read_text(encoding="utf-8")

        for command in ("codelewm score", "codelewm rerank", "codelewm eval ablation"):
            with self.subTest(command=command):
                self.assertIn(command, spec_text)

    def test_public_api_spec_marks_legacy_root_scripts_as_compatibility_only(self) -> None:
        spec_text = PUBLIC_API_SPEC.read_text(encoding="utf-8")

        self.assertIn("Root `train.py`, root `eval.py`", spec_text)
        self.assertIn("not the public CodeLeWM artifact path", spec_text)


if __name__ == "__main__":
    unittest.main()
