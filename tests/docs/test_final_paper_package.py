from __future__ import annotations

import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs" / "papers" / "codelewm_final_paper.tex"
CLAIM_AUDIT = ROOT / "docs" / "papers" / "codelewm_final_claim_audit.md"
ARXIV = ROOT / "docs" / "papers" / "ARXIV_SUBMISSION.md"
BUILD_SCRIPT = ROOT / "scripts" / "build-codelewm-final-paper"
README = ROOT / "README.md"
FULL_COMPLETION = ROOT / "docs" / "roadmap" / "FULL_COMPLETION.md"
IMPLEMENTATION = ROOT / "docs" / "roadmap" / "IMPLEMENTATION.md"


class FinalPaperPackageTest(unittest.TestCase):
    def test_final_paper_files_are_present(self) -> None:
        for path in (PAPER, CLAIM_AUDIT, ARXIV, BUILD_SCRIPT):
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertTrue(path.is_file(), f"missing {path}")
        self.assertTrue(os.access(BUILD_SCRIPT, os.X_OK), "build script must be executable")

    def test_paper_matches_final_claim_boundary(self) -> None:
        text = PAPER.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for marker in (
            "Negative Action-Use Results and a Narrow Downstream Reranking Slice",
            "not as evidence that the model generally improves coding",
            "Text-action Recall@1 was 0.263",
            "no-action reached Recall@1 0.441",
            "HumanEval WS-D",
            "0.9787",
            "+10.64",
            "+8.51",
            "MBPP-Plus is the counterweight",
            "1.0000 & 1.0000 & 0.1765 & 1.0000 & +0.00",
            "demo\\_report-e6fc06c328eed245",
            "docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md",
            "scripts/build-codelewm-final-paper",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, normalized)

    def test_claim_audit_blocks_unsupported_claims(self) -> None:
        text = CLAIM_AUDIT.read_text(encoding="utf-8")
        for marker in (
            "docs/papers/codelewm_final_paper.tex",
            "V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md",
            "v1.0 HumanEval WS-D lift over no-action",
            "v1.0 MBPP-Plus lift over no-action",
            "broad coding improvement",
            "semantic latent-axis claims",
            "standalone downstream `p_pass` scoring claims",
            "The paper conclusion must keep the aggregate public model-quality claim closed.",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_arxiv_metadata_points_to_final_package(self) -> None:
        text = ARXIV.read_text(encoding="utf-8")
        for marker in (
            "CodeLeWM Final Paper arXiv Submission Package",
            "docs/papers/codelewm_final_paper.tex",
            "docs/papers/codelewm_final_claim_audit.md",
            "scripts/build-codelewm-final-paper",
            "docs/papers/codelewm_final_paper.pdf",
            "docs/papers/codelewm_final_arxiv_source.tar.gz",
            "docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md",
            "Historical v0.6 two-substrate paper package",
            "superseded for final v1.0 release wording",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_readme_and_roadmaps_advance_to_408(self) -> None:
        readme = README.read_text(encoding="utf-8")
        full_completion = " ".join(FULL_COMPLETION.read_text(encoding="utf-8").split())
        implementation = IMPLEMENTATION.read_text(encoding="utf-8")

        self.assertIn("docs/papers/codelewm_final_paper.tex", readme)
        self.assertIn("#407 rewrote the paper", readme)
        self.assertIn("#408 published the final artifact index", readme)
        self.assertIn("Issues #402 through #408 are now complete", full_completion)
        self.assertIn("docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md", full_completion)
        self.assertIn(
            "| #407 | v1.0 paper: rewrite CodeLeWM paper around final downstream evidence | docs/results | p1 | l | follow-up | Closed |",
            implementation,
        )
        self.assertIn(
            "| #408 | v1.0 release: publish final artifact index, cards, README, and announcement package | docs/release/results | p1 | m | follow-up | Closed |",
            implementation,
        )


if __name__ == "__main__":
    unittest.main()
