from __future__ import annotations

import unittest

from codelewm.eval.execution_runner import _build_execution_surprise_claim_gates
from codelewm.eval.surprise import SurpriseMetrics


class ExecutionSurpriseClaimGateTest(unittest.TestCase):
    def test_score_and_count_gates_are_reported_separately(self) -> None:
        gates = _build_execution_surprise_claim_gates(
            metrics=SurpriseMetrics(
                pairwise_auc_overall=1.0,
                pairwise_auc_by_category={
                    "mutation": 1.0,
                    "same_code_different_input": 1.0,
                    "same_problem_different_submission": 1.0,
                },
                mean_true_rank=1.0,
                median_true_rank=1.0,
                recall_at_1=1.0,
                decoy_counts={
                    "mutation": 236,
                    "same_code_different_input": 352,
                    "same_problem_different_submission": 6,
                },
                example_count=236,
            ),
            selected_decoys=(
                "mutation",
                "same_problem_different_submission",
                "same_code_different_input",
            ),
            semantic_decoy_pack_metadata={
                "artifact_id": "downstream_benchmark-fixture",
                "pair_count": 358,
                "summary": {
                    "claim_gate": {"claim_allowed": True},
                    "distinct_problem_count": 68,
                    "pair_count_by_category": {
                        "same_code_different_input": 352,
                        "same_problem_different_submission": 6,
                    },
                },
            },
        )

        self.assertFalse(gates["claim_allowed"])
        self.assertTrue(all(gate["passed"] for gate in gates["score_gates"]))
        count_gates = {
            gate["category"]: gate for gate in gates["pair_count_gates"]
        }
        self.assertFalse(count_gates["same_problem_different_submission"]["passed"])
        self.assertTrue(count_gates["same_code_different_input"]["passed"])
        self.assertTrue(gates["semantic_decoy_pack_count_gate"]["passed"])
        self.assertEqual(
            gates["semantic_decoy_category_counts"]["scored_decoy_counts"][
                "same_problem_different_submission"
            ],
            6,
        )
        self.assertIn(
            "pair_count_gate_failed:same_problem_different_submission:6<30",
            gates["failure_reasons"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
