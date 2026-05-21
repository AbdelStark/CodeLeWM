from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "benchmark" / "VISUAL_OBSERVABILITY_ARTIFACTS_2026-05-21.md"


class VisualObservabilityArtifactsReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = REPORT.read_text(encoding="utf-8")

    def test_report_exists_and_links_public_artifacts(self) -> None:
        self.assertTrue(REPORT.is_file(), f"missing: {REPORT}")
        self.assertIn("codelewm-visual-observability-20260521-6a8ac81", self.text)
        self.assertIn("huggingface.co/datasets/abdelstark/codelewm-runs", self.text)

    def test_report_lists_required_visual_surfaces(self) -> None:
        for marker in (
            "tensorboard_export.json",
            "model_checkpoint_inspection.json",
            "latent_matrix_report.json",
            "visual_view_model.json",
            "run_timeline.json",
            "demo_fixture_terminal.txt",
            "demo_live_tui_snapshot.json",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_report_keeps_claim_gate_closed(self) -> None:
        for marker in (
            "claim gate: closed",
            "claim_gate.allowed=false",
            "CodeLeWM improves generated code",
            "useful semantic latent axes",
            "scaled downstream and representation gates",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)


if __name__ == "__main__":
    unittest.main()
