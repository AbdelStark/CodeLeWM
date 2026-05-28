"""Tests for the execution-substrate rerank LLM demo scenario."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codelewm.harness import (
    EXECUTION_RERANK_SCENARIO_ID,
    DemoScenarioError,
    get_demo_scenario,
    list_demo_scenarios,
    materialize_demo_scenario,
)


class ExecutionRerankScenarioTest(unittest.TestCase):
    def test_scenario_is_registered(self) -> None:
        scenario_ids = {s.scenario_id for s in list_demo_scenarios()}
        self.assertIn(EXECUTION_RERANK_SCENARIO_ID, scenario_ids)

    def test_scenario_metadata_is_substrate_pivot_shaped(self) -> None:
        scenario = get_demo_scenario(EXECUTION_RERANK_SCENARIO_ID)
        self.assertEqual(scenario.scenario_id, EXECUTION_RERANK_SCENARIO_ID)
        self.assertEqual(scenario.title, "Execution-substrate rerank: complete the MBPP-style function")
        constraints = dict(scenario.expected_static_constraints)
        self.assertEqual(constraints["touched_symbols"], ["compute_square"])
        self.assertEqual(constraints["example_input_repr"], "[3]")
        self.assertEqual(constraints["expected_output_repr"], "9")
        self.assertEqual(constraints["benchmark_id"], "mbpp_demo")

    def test_scenario_materialises_into_a_root_dir(self) -> None:
        scenario = get_demo_scenario(EXECUTION_RERANK_SCENARIO_ID)
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = materialize_demo_scenario(scenario, Path(tmpdir))
            self.assertEqual(metadata["scenario_id"], EXECUTION_RERANK_SCENARIO_ID)
            scenario_dir = (
                Path(tmpdir) / EXECUTION_RERANK_SCENARIO_ID
            )
            app = scenario_dir / "app.py"
            self.assertTrue(app.is_file())
            content = app.read_text(encoding="utf-8")
            self.assertIn("def compute_square", content)

    def test_unknown_scenario_raises(self) -> None:
        with self.assertRaises(DemoScenarioError):
            get_demo_scenario("nonexistent-scenario")

    def test_publication_notes_reference_sandbox_subsystem(self) -> None:
        scenario = get_demo_scenario(EXECUTION_RERANK_SCENARIO_ID)
        full = " ".join(scenario.publication_notes).lower()
        self.assertIn("sandbox", full)
        self.assertIn("operator", full)
        # The full HumanEval / MBPP-Plus rerank is operator-driven.
        self.assertIn("operator-driven", full)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
