from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "spec" / "11-llm-world-model-harness.md"
RFC = ROOT / "docs" / "rfcs" / "RFC-0013-llm-world-model-harness-and-publication.md"
ROADMAP = ROOT / "docs" / "roadmap" / "POST_V0_2_SHOWCASE_ROADMAP.md"
RESULTS = ROOT / "docs" / "benchmark" / "PRELIMINARY_RESULTS_2026-05-21.md"
ARTIFACT_INDEX = ROOT / "docs" / "benchmark" / "PUBLIC_ARTIFACT_INDEX_2026-05-21.md"
ANNOUNCEMENT = ROOT / "docs" / "announcements" / "PRELIMINARY_RESULTS_2026-05-21.md"
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
            "codelewm.downstream_rerank_benchmark_config.v1",
            "codelewm.downstream_benchmark_readiness.v1",
            "codelewm.downstream_rerank_eval_run.v1",
            "codelewm.downstream_rerank_report.v1",
            "codelewm eval downstream-pack",
            "codelewm eval downstream-rerank",
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
            "The first scaled action-conditioned hypotheses failed",
            "Run Ledger",
            "codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b",
            "6a0dea258229e585f969c808",
            "docs/cards/codelewm-v0-2-action-swap-model-2026-05-20.md",
            "| #159 margin+retrieval | 0.597 | 0.650 | -0.053",
            "semantic_structure_status=unsupported",
            "scaled downstream benchmark requires at least 100 labeled examples; got 1",
            "Validated",
            "Invalidated Or Unsupported",
            "Public Artifacts",
            "What We Can Publish Now",
            "Blog / README Summary",
            "We cannot publish it as",
            "LLM + world-model harness",
            "The downstream benchmark must pass #192",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_artifact_index_lists_public_hf_paths_and_cards(self) -> None:
        text = ARTIFACT_INDEX.read_text(encoding="utf-8")

        for marker in (
            "CodeLeWM Public Artifact Index 2026-05-21",
            "abdelstark/codelewm-public-shard/runs/codelewm-scaled-20260520-9699b53/pack",
            "abdelstark/codelewm-transition-model/checkpoints/codelewm-action-use-20260520-6650183",
            "abdelstark/codelewm-runs/runs/codelewm-action-use-retrieval-20260520-7895d18",
            "abdelstark/codelewm-public-shard/runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/pack",
            "docs/cards/codelewm-v0-2-action-swap-model-2026-05-20.md",
            "not a model-quality claim",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_announcement_copy_is_claim_safe(self) -> None:
        text = ANNOUNCEMENT.read_text(encoding="utf-8")

        for marker in (
            "Short Announcement",
            "X / Twitter Draft",
            "Longer Post",
            "The scientific result is negative",
            "action-conditioned variants still lose to",
            "the no-action baseline",
            "Do Not Say",
            "CodeLeWM improves coding",
            "Those claims remain blocked",
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
