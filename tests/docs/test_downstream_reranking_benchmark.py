from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "docs" / "benchmark" / "DOWNSTREAM_RERANKING_BENCHMARK.md"
REPORT_TEMPLATE = ROOT / "docs" / "benchmark" / "REPORT_TEMPLATE.md"


class DownstreamRerankingBenchmarkDocsTest(unittest.TestCase):
    def test_benchmark_doc_records_current_blocker_and_component_metrics(self) -> None:
        text = BENCHMARK.read_text(encoding="utf-8")

        for marker in (
            "#190",
            "#184",
            "codelewm.downstream_rerank_benchmark.v1",
            "codelewm.downstream_rerank_report.v1",
            "codelewm.downstream_rerank_claim_gate.v1",
            "codelewm.harness.scorer_quality_report.v1",
            "component_metrics.final_score",
            "component_metrics.transition_energy_only",
            "component_metrics.retrieval_prior_only",
            "baseline_controls",
            "required baselines: LLM order, random, lexical, no-action, CodeLeWM",
            "retrieval prior, and score ensemble",
            "benchmark_readiness.scaled_evaluation_ready=false",
            "benchmark_readiness.downstream_claim_allowed=false",
            "at least `100` labeled reranking examples",
            "CodeLeWM is strictly above LLM order on pass@1 and MRR",
            "Downstream reranking usefulness is **not supported yet**",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_benchmark_doc_requires_downloaded_hf_score_and_rerank_commands(self) -> None:
        text = BENCHMARK.read_text(encoding="utf-8")

        for marker in (
            "hf download \"$CODELEWM_HF_RESULTS_REPO_ID\"",
            "hf download \"$CODELEWM_HF_MODEL_REPO_ID\"",
            "uv run codelewm score",
            "uv run codelewm rerank",
            "uv run codelewm eval scorer-quality",
            "--index .artifacts/hf-download/<run-id>/results/runs/<run-id>/index",
            "--parent-manifest .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/manifest.json",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_report_template_separates_transition_energy_and_retrieval_prior(self) -> None:
        text = REPORT_TEMPLATE.read_text(encoding="utf-8")

        for marker in (
            "benchmark_readiness.scaled_evaluation_ready",
            "benchmark_readiness.downstream_claim_allowed",
            "component_metrics",
            "baseline_controls",
            "transition_energy_only",
            "retrieval_prior_only",
            "final_score",
            "checkpoint_159",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
