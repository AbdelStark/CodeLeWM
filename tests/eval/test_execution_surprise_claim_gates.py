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
            decoy_coverage_summary={
                "blockers": [
                    {
                        "type": "semantic_decoy_pair_count_blocker",
                        "category": "same_problem_different_submission",
                        "reason": (
                            "semantic_decoy_pair_count_blocker:"
                            "same_problem_different_submission:6<30"
                        ),
                    }
                ],
                "generated_pair_count_by_category": {
                    "same_code_different_input": 352,
                    "same_problem_different_submission": 6,
                },
                "candidate_pair_count_by_category": {
                    "same_code_different_input": 352,
                    "same_problem_different_submission": 6,
                },
                "scorable_pair_count_by_category": {
                    "same_code_different_input": 352,
                    "same_problem_different_submission": 6,
                },
                "missing_query_record_count_by_category": {},
                "missing_decoy_record_count_by_category": {},
            },
        )

        self.assertFalse(gates["claim_allowed"])
        score_gates = {gate["category"]: gate for gate in gates["score_gates"]}
        self.assertTrue(score_gates["mutation"]["passed"])
        self.assertTrue(score_gates["same_code_different_input"]["passed"])
        self.assertFalse(score_gates["same_problem_different_submission"]["passed"])
        self.assertEqual(
            score_gates["same_problem_different_submission"]["status"],
            "blocked_by_pair_count",
        )
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
            "semantic_decoy_pair_count_blocker:same_problem_different_submission:6<30",
            gates["failure_reasons"],
        )
        self.assertEqual(
            gates["coverage_blockers"][0]["type"],
            "semantic_decoy_pair_count_blocker",
        )
        self.assertEqual(
            gates["semantic_decoy_category_counts"][
                "scorable_pair_count_by_category"
            ]["same_problem_different_submission"],
            6,
        )

    def test_missing_aligned_pairs_block_score_gate_before_claim(self) -> None:
        gates = _build_execution_surprise_claim_gates(
            metrics=SurpriseMetrics(
                pairwise_auc_overall=0.0,
                pairwise_auc_by_category={},
                mean_true_rank=0.0,
                median_true_rank=0.0,
                recall_at_1=0.0,
                decoy_counts={"same_code_different_input": 0},
                example_count=10,
            ),
            selected_decoys=("same_code_different_input",),
            semantic_decoy_pack_metadata={
                "artifact_id": "downstream_benchmark-old-pack",
                "summary": {
                    "claim_gate": {"claim_allowed": True},
                    "pair_count": 120,
                    "distinct_problem_count": 40,
                    "pair_count_by_category": {
                        "same_code_different_input": 120,
                    },
                },
            },
            decoy_coverage_summary={
                "blockers": [
                    {
                        "type": "semantic_decoy_pair_count_blocker",
                        "category": "same_code_different_input",
                        "reason": (
                            "semantic_decoy_pair_count_blocker:"
                            "same_code_different_input:0<100"
                        ),
                    }
                ],
                "pack_pair_count_by_category": {
                    "same_code_different_input": 120,
                },
                "candidate_pair_count_by_category": {
                    "same_code_different_input": 120,
                },
                "scorable_pair_count_by_category": {},
                "missing_query_record_count_by_category": {
                    "same_code_different_input": 120,
                },
                "missing_decoy_record_count_by_category": {
                    "same_code_different_input": 120,
                },
            },
        )

        score_gate = gates["score_gates"][0]
        count_gate = gates["pair_count_gates"][0]
        self.assertFalse(gates["claim_allowed"])
        self.assertEqual(score_gate["status"], "blocked_by_pair_count")
        self.assertEqual(score_gate["observed_win_rate"], None)
        self.assertEqual(count_gate["pack_pair_count"], 120)
        self.assertEqual(count_gate["candidate_pair_count"], 120)
        self.assertEqual(count_gate["scorable_pair_count"], 0)
        self.assertEqual(count_gate["missing_query_record_count"], 120)
        self.assertIn(
            "semantic_decoy_pair_count_blocker:same_code_different_input:0<100",
            gates["failure_reasons"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
