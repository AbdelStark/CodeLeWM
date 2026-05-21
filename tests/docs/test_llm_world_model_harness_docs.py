from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "spec" / "11-llm-world-model-harness.md"
RFC = ROOT / "docs" / "rfcs" / "RFC-0013-llm-world-model-harness-and-publication.md"
ROADMAP = ROOT / "docs" / "roadmap" / "POST_V0_2_SHOWCASE_ROADMAP.md"
RESULTS = ROOT / "docs" / "benchmark" / "PRELIMINARY_RESULTS_2026-05-21.md"
ENV_EXAMPLE = ROOT / ".env.example"


class LLMWorldModelHarnessDocsTest(unittest.TestCase):
    def test_spec_locks_openrouter_and_claim_boundary(self) -> None:
        text = SPEC.read_text(encoding="utf-8")

        for marker in (
            "OPENROUTER_API_KEY",
            "CODELEWM_LLM_MODEL",
            "openrouter==0.9.1",
            "codelewm.openrouter_candidate_request.v1",
            "codelewm.llm_candidate_pack.v1",
            "codelewm.harness.demo_report.v1",
            "codelewm.downstream_rerank_report.v1",
            "must not read a raw `ANTHROPIC_API_KEY`",
            "OPENROUTER_DEBUG",
            "provider_routing",
            "Demo success requires",
            "allowed=false",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_rfc_maps_streams_to_github_issues(self) -> None:
        text = RFC.read_text(encoding="utf-8")

        for marker in (
            "#183 tracks the stream",
            "#184 tracks the stream",
            "#185 tracks the stream",
            "#186 locks the OpenRouter harness contract",
            "#192 runs the downstream comparison",
            "OpenRouter SDK is beta",
            "openrouter==0.9.1",
            "codelewm.openrouter_candidate_request.v1",
            "Demo success proves only",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_roadmap_contains_next_goal_prompt_and_order(self) -> None:
        text = ROADMAP.read_text(encoding="utf-8")

        for marker in (
            "# Post-v0.2 Showcase Roadmap",
            "Stream A: LLM + World-Model Harness Demo",
            "Stream B: Downstream Candidate-Reranking Benchmark",
            "Stream C: Preliminary Results Publication",
            "/goal Continue CodeLeWM",
            "#186",
            "#193",
            "#194",
            "#192",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_preliminary_results_are_negative_and_publishable(self) -> None:
        text = RESULTS.read_text(encoding="utf-8")

        for marker in (
            "The first action-conditioned hypotheses failed",
            "Validated",
            "Invalidated Or Unsupported",
            "Public Artifacts",
            "What We Can Publish Now",
            "We cannot publish it as",
            "LLM + world-model harness",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_env_example_declares_llm_harness_settings(self) -> None:
        text = ENV_EXAMPLE.read_text(encoding="utf-8")

        for marker in (
            "CODELEWM_LLM_PROVIDER=openrouter",
            "OPENROUTER_API_KEY=openrouter_xxx",
            "CODELEWM_LLM_MODEL=anthropic/claude-4.5-sonnet",
            "CODELEWM_LLM_DRY_RUN=1",
            "CODELEWM_LLM_PROVIDER_OPTIONS_JSON=",
            "CODELEWM_LLM_RETRY_LIMIT=2",
            "OPENROUTER_APP_TITLE=CodeLeWM",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
