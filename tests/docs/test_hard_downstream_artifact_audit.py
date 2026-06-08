from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "docs" / "benchmark" / "DOWNSTREAM_RERANKING_BENCHMARK.md"
ROADMAP = ROOT / "docs" / "roadmap" / "HARD_DOWNSTREAM_RERANKING_BENCHMARK.md"
PUBLISH_MODULE = ROOT / "codelewm" / "eval" / "hard_downstream_publish.py"

# Wrap-safe fragment of the RFC-0016 diagnostic fallback wording used while the
# claim gate is closed.
DIAGNOSTIC_FRAGMENT = "downstream coding-usefulness claim"


def _collapsed(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


class HardDownstreamArtifactAuditTest(unittest.TestCase):
    def test_benchmark_doc_documents_publication_machinery(self) -> None:
        text = BENCHMARK.read_text(encoding="utf-8")
        for marker in (
            "Implementation status",
            "codelewm eval downstream-pack",
            "downstream-rerank --hard-mode",
            "codelewm eval hard-downstream-publish",
            "codelewm.hard_downstream_claim_audit.v1",
            "codelewm.llm_candidate_ingest_report.v1",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)
        self.assertIn("claim gate remains closed", _collapsed(BENCHMARK))

    def test_benchmark_doc_uses_diagnostic_wording_not_a_positive_claim(self) -> None:
        collapsed = _collapsed(BENCHMARK)
        # The gate is closed, so the doc carries the diagnostic fallback wording
        # and must NOT assert that CodeLeWM improves generated code.
        self.assertIn(DIAGNOSTIC_FRAGMENT, collapsed)
        raw = BENCHMARK.read_text(encoding="utf-8")
        self.assertNotIn("CodeLeWM improves generated code", raw)
        self.assertNotIn("CodeLeWM improves coding", raw)

    def test_doc_wording_matches_code_constant(self) -> None:
        # Code and docs agree on the exact fallback wording (wrap-insensitive).
        self.assertIn(DIAGNOSTIC_FRAGMENT, _collapsed(PUBLISH_MODULE))
        self.assertIn(DIAGNOSTIC_FRAGMENT, _collapsed(BENCHMARK))

    def test_roadmap_marks_children_merged_and_claim_closed(self) -> None:
        text = ROADMAP.read_text(encoding="utf-8")
        for issue in ("#418", "#419", "#420", "#421", "#422", "#423"):
            with self.subTest(issue=issue):
                self.assertIn(issue, text)
        self.assertIn("merged", text)
        self.assertIn("claim gate remains closed", _collapsed(ROADMAP))

    def test_publish_module_declares_required_schemas(self) -> None:
        source = PUBLISH_MODULE.read_text(encoding="utf-8")
        for schema in (
            "codelewm.hard_downstream_publication.v1",
            "codelewm.hard_downstream_artifact_index.v1",
            "codelewm.hard_downstream_claim_audit.v1",
        ):
            with self.subTest(schema=schema):
                self.assertIn(schema, source)


if __name__ == "__main__":
    unittest.main()
