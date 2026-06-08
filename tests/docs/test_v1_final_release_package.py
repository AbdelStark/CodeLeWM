from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FINAL_INDEX = ROOT / "docs" / "benchmark" / "PUBLIC_ARTIFACT_INDEX_2026-06-08.md"
RELEASE_CARD = ROOT / "docs" / "cards" / "codelewm-v1-0-final-release-2026-06-08.md"
DEMO_CARD = ROOT / "docs" / "cards" / "codelewm-v1-0-paper-demo-2026-06-08.md"
CHECKLIST = ROOT / "docs" / "release" / "V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md"
ANNOUNCEMENT = ROOT / "docs" / "announcements" / "FINAL_V1_0_RELEASE_2026-06-08.md"
README = ROOT / "README.md"
NEXT_GOAL = ROOT / "docs" / "roadmap" / "NEXT_GOAL_PROMPT.md"
FULL_COMPLETION = ROOT / "docs" / "roadmap" / "FULL_COMPLETION.md"
IMPLEMENTATION = ROOT / "docs" / "roadmap" / "IMPLEMENTATION.md"
POST_V0_2 = ROOT / "docs" / "roadmap" / "POST_V0_2_SHOWCASE_ROADMAP.md"
V0_9_DATASET_CARD = ROOT / "docs" / "cards" / "codelewm-v0-9-execution-dataset-2026-06-07.md"
V0_9_SEED_42_CARD = ROOT / "docs" / "cards" / "codelewm-v0-9-execution-model-seed-42-2026-06-07.md"
V0_9_SEED_1729_CARD = ROOT / "docs" / "cards" / "codelewm-v0-9-execution-model-seed-1729-2026-06-07.md"


class V1FinalReleasePackageTest(unittest.TestCase):
    def test_final_release_files_exist(self) -> None:
        for path in (FINAL_INDEX, RELEASE_CARD, DEMO_CARD, CHECKLIST, ANNOUNCEMENT):
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertTrue(path.is_file(), f"missing {path}")
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("TODO", text)
                self.assertNotIn("Copy this file", text)

    def test_final_index_links_all_public_surfaces(self) -> None:
        text = FINAL_INDEX.read_text(encoding="utf-8")
        for marker in (
            "docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md",
            "docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md",
            "docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md",
            "docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md",
            "docs/benchmark/v1_0/paper_demo/reports/paper_demo_report.json",
            "docs/cards/codelewm-v0-9-execution-dataset-2026-06-07.md",
            "docs/cards/codelewm-v0-9-execution-model-seed-42-2026-06-07.md",
            "docs/cards/codelewm-v0-9-execution-model-seed-1729-2026-06-07.md",
            "docs/cards/codelewm-v1-0-paper-demo-2026-06-08.md",
            "docs/cards/codelewm-v1-0-final-release-2026-06-08.md",
            "docs/papers/codelewm_final_paper.tex",
            "docs/papers/codelewm_final_claim_audit.md",
            "docs/papers/codelewm_final_paper.pdf",
            "docs/papers/codelewm_final_arxiv_source.tar.gz",
            "docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md",
            "docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md",
            "demo_report-e6fc06c328eed245",
            "claim_allowed=false",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_cards_and_announcement_preserve_claim_boundary(self) -> None:
        release_card = RELEASE_CARD.read_text(encoding="utf-8")
        demo_card = DEMO_CARD.read_text(encoding="utf-8")
        announcement = ANNOUNCEMENT.read_text(encoding="utf-8")

        for marker in (
            "broad model-quality claim closed",
            "narrow HumanEval WS-D reranking slice",
            "CodeLeWM generally improves coding",
            "MBPP-Plus CodeLeWM, no-action, and lexical pass@1 are all `1.0000`",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, release_card)

        for marker in (
            "demo_report-e6fc06c328eed245",
            "Aggregate claim gate: `claim_allowed=false`",
            "HumanEval WS-D",
            "MBPP-Plus WS-D",
            "It does not support a broad claim that CodeLeWM generally improves coding",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, demo_card)

        for marker in (
            "Do not say CodeLeWM generally improves coding.",
            "narrow HumanEval WS-D reranking slice",
            "aggregate downstream model-quality claim remains closed",
            "MBPP-Plus is saturated against no-action and lexical controls",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, announcement)

    def test_reproducibility_checklist_has_clean_checkout_gates(self) -> None:
        text = CHECKLIST.read_text(encoding="utf-8")
        for marker in (
            "uv sync --group dev --group data --group train",
            "uv run scripts/paper-demo",
            "--manifest docs/benchmark/v1_0/paper_demo/manifest.json",
            "--parent-manifest docs/benchmark/v0_9/seed-42/rerank/humaneval/manifest.json",
            "--parent-manifest docs/benchmark/v0_9/seed-1729/rerank/mbpp_plus/manifest.json",
            "uv run codelewm secret-scan docs/benchmark/v1_0/paper_demo",
            "scripts/build-codelewm-final-paper",
            "uv run pytest tests/docs -q",
            "uv run python -m compileall -q",
            "git diff --check",
            "Close #401 only after",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_readme_and_roadmaps_are_final_not_active_queue(self) -> None:
        readme = README.read_text(encoding="utf-8")
        next_goal = NEXT_GOAL.read_text(encoding="utf-8")
        full_completion = " ".join(FULL_COMPLETION.read_text(encoding="utf-8").split())
        implementation = IMPLEMENTATION.read_text(encoding="utf-8")
        post_v0_2 = POST_V0_2.read_text(encoding="utf-8")

        for marker in (
            "docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md",
            "docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md",
            "docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md",
            "#408 published the final artifact index",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, readme)

        self.assertIn("completion record for the final v1.0 paper/demo release tracker #401", next_goal)
        self.assertIn("#408 - complete", next_goal)
        self.assertIn("No active final release child issue remains.", next_goal)
        self.assertNotIn("#408 - next", next_goal)
        self.assertIn("Issues #402 through #408 are now complete", full_completion)
        self.assertIn("#401 v1.0 paper/demo release: complete", full_completion)
        self.assertIn(
            "| #408 | v1.0 release: publish final artifact index, cards, README, and announcement package | docs/release/results | p1 | m | follow-up | Closed |",
            implementation,
        )
        self.assertIn(
            "| #401 | [TRACKER] v1.0 paper/demo release: downstream learned-world-model evidence and final CodeLeWM conclusions | evaluation/harness/docs/results | p1 | l | follow-up | Closed |",
            implementation,
        )
        self.assertIn("final v1.0 paper/demo tracker #401 is complete", post_v0_2)
        self.assertNotIn("next: repair the cross-benchmark data/eval gaps", post_v0_2)

    def test_v0_9_cards_point_to_final_release_index(self) -> None:
        for path in (V0_9_DATASET_CARD, V0_9_SEED_42_CARD, V0_9_SEED_1729_CARD):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md", text)
                self.assertIn("docs/cards/codelewm-v1-0-final-release-2026-06-08.md", text)


if __name__ == "__main__":
    unittest.main()
