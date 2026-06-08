from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RFC = ROOT / "docs" / "rfcs" / "RFC-0016-hard-downstream-reranking-benchmark.md"
ROADMAP = ROOT / "docs" / "roadmap" / "HARD_DOWNSTREAM_RERANKING_BENCHMARK.md"
SPEC = ROOT / "docs" / "spec" / "11-llm-world-model-harness.md"
BENCHMARK = ROOT / "docs" / "benchmark" / "DOWNSTREAM_RERANKING_BENCHMARK.md"
IMPLEMENTATION = ROOT / "docs" / "roadmap" / "IMPLEMENTATION.md"
FULL_COMPLETION = ROOT / "docs" / "roadmap" / "FULL_COMPLETION.md"
NEXT_GOAL = ROOT / "docs" / "roadmap" / "NEXT_GOAL_PROMPT.md"
ROOT_SPEC = ROOT / "SPEC.md"


class HardDownstreamBenchmarkDocsTest(unittest.TestCase):
    def test_rfc_locks_scientific_question_and_claim_gate(self) -> None:
        text = RFC.read_text(encoding="utf-8")

        for marker in (
            "RFC-0016: Hard Anti-Saturation Downstream Reranking Benchmark",
            "Can a learned code world-model score add value",
            "no-action pass@1 is below `0.85`",
            "lexical pass@1 is below `0.85`",
            "LLM-order pass@1 is below `0.90`",
            "codelewm.downstream_anti_saturation_report.v1",
            "no-action baits",
            "partial fixes",
            "wrong-symbol or wrong-branch fixes",
            "over-broad fixes",
            "deterministic semantic mutants",
            "LLM-generated unified diffs",
            "CodeLeWM beats no-action, lexical, and LLM-order",
            "confidence intervals excluding zero",
            "Candidate code is untrusted",
            "Reviewer-Facing Evaluation Matrix",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_roadmap_maps_tracker_and_child_issues(self) -> None:
        text = ROADMAP.read_text(encoding="utf-8")

        for marker in (
            "Tracker: #417",
            "#418",
            "#419",
            "#420",
            "#421",
            "#422",
            "#423",
            "Benchmark schema/config and anti-saturation report",
            "Public-safe hard-negative candidate-pack builder",
            "LLM candidate-pack ingestion",
            "Baseline and CodeLeWM scoring/evaluation gate",
            "Artifact publication, claim audit, and paper addendum",
            "no-action and lexical below `0.85`",
            "LLM-order below `0.90`",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_spec_and_benchmark_docs_define_profile(self) -> None:
        for path in (SPEC, BENCHMARK):
            text = " ".join(path.read_text(encoding="utf-8").split())
            with self.subTest(path=path.name):
                self.assertIn("anti_saturation_semantic_v1", text)
                self.assertIn("codelewm.downstream_anti_saturation_report.v1", text)
                self.assertIn("no-action pass@1 is below `0.85`", text)
                self.assertIn("lexical pass@1 is below `0.85`", text)
                self.assertIn("LLM-order pass@1 is below `0.90`", text)
                self.assertIn("CodeLeWM beats no-action, lexical, and LLM-order", text)

    def test_project_trackers_point_to_v1_5_follow_up(self) -> None:
        for path in (IMPLEMENTATION, FULL_COMPLETION, NEXT_GOAL):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("#417", text)
                self.assertIn("#418", text)
                self.assertIn("#423", text)
                self.assertIn("anti-saturation downstream", text)

    def test_root_spec_lists_rfc_0016(self) -> None:
        text = ROOT_SPEC.read_text(encoding="utf-8")

        self.assertIn("RFC-0016-hard-downstream-reranking-benchmark.md", text)


if __name__ == "__main__":
    unittest.main()
