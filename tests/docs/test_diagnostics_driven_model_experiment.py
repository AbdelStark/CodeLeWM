from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs" / "roadmap" / "DIAGNOSTICS_DRIVEN_MODEL_EXPERIMENT.md"


class DiagnosticsDrivenModelExperimentDocTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = PLAN.read_text(encoding="utf-8")

    def test_plan_exists_and_is_substantive(self) -> None:
        self.assertTrue(PLAN.is_file(), f"missing: {PLAN}")
        self.assertGreater(len(self.text), 4000)

    def test_plan_defines_one_falsifiable_intervention(self) -> None:
        for marker in (
            "Name: candidate-contrast action training.",
            "Hypothesis:",
            "pairwise energy margin",
            "Expected Failure Modes",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_plan_requires_required_gates_and_baselines(self) -> None:
        for marker in (
            "at least 100 labeled public-safe examples",
            "no-action",
            "LLM original order",
            "shuffled-action",
            "latent_probe_report",
            "latent_matrix_report",
            "manifest verify",
            "secret-scan",
            "hf jobs inspect",
            "hf download",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_plan_keeps_failed_experiment_claims_blocked(self) -> None:
        for blocked_claim in (
            "CodeLeWM improves generated code",
            "useful semantic latent axes",
            "action conditioning is better than no-action",
            "the harness scorer is reliable for candidate selection",
        ):
            with self.subTest(blocked_claim=blocked_claim):
                self.assertIn(blocked_claim, self.text)


if __name__ == "__main__":
    unittest.main()
