from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_INDEX = ROOT / "docs" / "benchmark" / "PUBLIC_ARTIFACT_INDEX_2026-05-31.md"
SEED_42_CARD = ROOT / "docs" / "cards" / "codelewm-v0-6-execution-model-seed-42-2026-05-31.md"
SEED_1729_CARD = ROOT / "docs" / "cards" / "codelewm-v0-6-execution-model-seed-1729-2026-05-31.md"
PUBLICATION_COORDINATION = ROOT / "docs" / "release" / "V0_6_PUBLICATION_COORDINATION.md"
RESULTS_TEMPLATE = ROOT / "docs" / "benchmark" / "EXECUTION_V0_6_RESULTS_TEMPLATE.md"


class V06PublicArtifactIndexTest(unittest.TestCase):
    def test_v0_6_checkpoint_refs_use_public_run_artifact_paths(self) -> None:
        paths = (
            PUBLIC_INDEX,
            SEED_42_CARD,
            SEED_1729_CARD,
            PUBLICATION_COORDINATION,
            RESULTS_TEMPLATE,
        )
        for path in paths:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("codelewm-transition-model@v0.6.0-seed", text)
                self.assertNotIn("v0.6.0-seed-42", text)
                self.assertNotIn("v0.6.0-seed-1729", text)

        index = PUBLIC_INDEX.read_text(encoding="utf-8")
        self.assertIn(
            "runs/codelewm-v0-6-execution-20260530-af1a114-seed-{42,1729}/checkpoints/last.pt",
            index,
        )
        self.assertIn(
            "the resolving public v0.6 files are the run-artifact",
            index,
        )

    def test_model_cards_link_resolving_checkpoint_surfaces(self) -> None:
        expectations = {
            SEED_42_CARD: (
                "abdelstark/codelewm-runs/runs/"
                "codelewm-v0-6-execution-20260530-af1a114-seed-42/checkpoints/last.pt"
            ),
            SEED_1729_CARD: (
                "abdelstark/codelewm-runs/runs/"
                "codelewm-v0-6-execution-20260530-af1a114-seed-1729/checkpoints/last.pt"
            ),
        }
        for path, expected in expectations.items():
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn(expected, text)
                self.assertIn("canonical public checkpoint surface", text)


if __name__ == "__main__":
    unittest.main()
