"""Tests that the substrate-pivot documentation set is present and linked.

The substrate pivot's documentation stack (#259) is load-bearing — every
later artifact (dataset cards, model cards, benchmark report, paper)
cites these files. The test ensures every file exists and that the
cross-links between them remain intact.
"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


REQUIRED_DOCS = (
    ROOT / "docs" / "rfcs" / "RFC-0014-execution-trace-world-model-substrate.md",
    ROOT / "docs" / "roadmap" / "EXECUTION_TRACE_WORLD_MODEL.md",
    ROOT / "docs" / "operations" / "V0_6_EXECUTION_RUN_RUNBOOK.md",
    ROOT / "docs" / "operations" / "sandbox_policy.md",
    ROOT / "docs" / "benchmark" / "EXECUTION_V0_6_RESULTS_TEMPLATE.md",
    ROOT / "docs" / "papers" / "two_substrate_outline.md",
    ROOT / "docs" / "cards" / "dataset_card.execution_pack.v1.md",
    ROOT / "codelewm" / "security" / "claim_boundaries" / "execution_substrate.v1.md",
)


class SubstratePivotDocsPresenceTest(unittest.TestCase):
    def test_every_required_doc_is_present(self) -> None:
        for path in REQUIRED_DOCS:
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertTrue(path.is_file(), msg=f"missing {path}")


class BenchmarkTemplateLinksTest(unittest.TestCase):
    def test_template_references_required_schemas(self) -> None:
        text = (
            ROOT / "docs" / "benchmark" / "EXECUTION_V0_6_RESULTS_TEMPLATE.md"
        ).read_text(encoding="utf-8")
        for schema in (
            "codelewm.execution_pack_manifest.v1",
            "codelewm.execution_train_config.v1",
            "codelewm.execution_launch_plan.v1",
            "codelewm.eval.execution_rerank_report.v1",
            "codelewm.eval.crash_prediction_report.v1",
        ):
            with self.subTest(schema=schema):
                self.assertIn(schema, text, msg=f"template missing {schema}")
        for gate_field in (
            "Headline Retrieval (Claim Gate)",
            "Collapse And SIGReg Diagnostics",
            "Surprise Evaluation",
            "Latent Probe Matrix",
            "Downstream Reranking",
            "Crash Prediction",
            "Claim-Gate Summary",
            "Allowed Public Language (If All Gates Pass)",
            "Allowed Public Language (If Any Gate Fails)",
        ):
            with self.subTest(section=gate_field):
                self.assertIn(gate_field, text)


class PaperOutlineLinksTest(unittest.TestCase):
    def test_outline_references_v0_2_and_v0_6_benchmark_reports(self) -> None:
        text = (
            ROOT / "docs" / "papers" / "two_substrate_outline.md"
        ).read_text(encoding="utf-8")
        self.assertIn("V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md", text)
        self.assertIn("EXECUTION_V0_6_RESULTS_TEMPLATE.md", text)
        # Both gate-fail and gate-pass framings must be present so the
        # paper ships from the same artifact set regardless of outcome.
        self.assertIn("If the v0.6 headline rerank gate passes", text)
        self.assertIn("If the v0.6 headline rerank gate fails", text)


class ExplainerLinksTest(unittest.TestCase):
    def test_project_explainer_links_substrate_pivot_stack(self) -> None:
        text = (
            ROOT / "docs" / "PROJECT_EXPLAINER.md"
        ).read_text(encoding="utf-8")
        for ref in (
            "RFC-0014-execution-trace-world-model-substrate.md",
            "EXECUTION_TRACE_WORLD_MODEL.md",
            "V0_6_EXECUTION_RUN_RUNBOOK.md",
            "EXECUTION_V0_6_RESULTS_TEMPLATE.md",
            "two_substrate_outline.md",
        ):
            with self.subTest(ref=ref):
                self.assertIn(
                    ref, text, msg=f"PROJECT_EXPLAINER does not link {ref!r}"
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
