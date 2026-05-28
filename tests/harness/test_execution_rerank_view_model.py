"""Tests for the execution-substrate rerank visual view model."""

from __future__ import annotations

import json
import unittest

from codelewm.harness import (
    EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION,
    ExecutionRerankViewModelError,
    build_execution_rerank_view_model,
)


def _rerank_report() -> dict[str, object]:
    return {
        "schema_version": "codelewm.eval.execution_rerank_report.v1",
        "benchmark": "mbpp_demo",
        "problem_count": 1,
        "completions_per_problem": 3,
        "baselines": [
            {
                "baseline": "llm_order",
                "pass_at_1": 0.0,
                "pass_count": 0,
                "problem_count": 1,
            },
            {
                "baseline": "codelewm",
                "pass_at_1": 1.0,
                "pass_count": 1,
                "problem_count": 1,
            },
            {
                "baseline": "lexical",
                "pass_at_1": 0.0,
                "pass_count": 0,
                "problem_count": 1,
            },
        ],
        "codelewm_lift_over_llm_order": 100.0,
        "bootstrap_lift_ci": [80.0, 100.0],
        "claim_allowed": True,
        "claim_reason": "lift_above_threshold_and_ci_excludes_zero",
    }


def _completions() -> list[dict[str, object]]:
    return [
        {
            "completion_id": "c1",
            "code": "def compute_square(n):\n    return n + 1\n",
            "llm_order_rank": 1,
            "passed": False,
            "scores": {"codelewm": 0.1, "lexical": 0.5},
            "test_results": [{"name": "case-1", "passed": False}],
            "predicted_output_latent": {"norm": 1.2, "top_dims": [3, 7, 12]},
        },
        {
            "completion_id": "c2",
            "code": "def compute_square(n):\n    return n * n\n",
            "llm_order_rank": 2,
            "passed": True,
            "scores": {"codelewm": 0.9, "lexical": 0.7},
            "test_results": [{"name": "case-1", "passed": True}],
            "predicted_output_latent": {"norm": 2.5, "top_dims": [1, 4, 9]},
        },
        {
            "completion_id": "c3",
            "code": "def compute_square(n):\n    return 0\n",
            "llm_order_rank": 3,
            "passed": False,
            "scores": {"codelewm": 0.2, "lexical": 0.3},
            "test_results": [{"name": "case-1", "passed": False}],
            "predicted_output_latent": {"norm": 0.7, "top_dims": [22]},
        },
    ]


class ExecutionRerankViewModelTest(unittest.TestCase):
    def test_builds_view_model_with_required_schema(self) -> None:
        model = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        )
        self.assertEqual(
            model.schema_version, EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION
        )
        self.assertEqual(model.scenario_id, "execution-rerank-mbpp")
        self.assertEqual(model.benchmark_id, "mbpp_demo")
        self.assertEqual(model.pass_at_1_lift, 100.0)
        self.assertEqual(model.bootstrap_lift_ci, (80.0, 100.0))
        self.assertTrue(model.claim_allowed)
        self.assertEqual(model.problem_count, 1)
        self.assertEqual(model.completions_per_problem, 3)

    def test_codelewm_ranking_orders_panels(self) -> None:
        model = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        )
        # The codelewm winner (c2, score 0.9) ranks first.
        self.assertEqual(model.completion_panels[0].completion_id, "c2")
        self.assertEqual(model.completion_panels[0].codelewm_rank, 1)
        # The losing completions take rank 2 and 3 by codelewm score.
        ranks = [p.codelewm_rank for p in model.completion_panels]
        self.assertEqual(ranks, [1, 2, 3])

    def test_rank_by_baseline_disagreement_is_recorded(self) -> None:
        model = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        )
        c2 = next(p for p in model.completion_panels if p.completion_id == "c2")
        # c2 is rank 1 under codelewm but rank 2 under llm_order.
        self.assertEqual(c2.rank_by_baseline["codelewm"], 1)
        self.assertEqual(c2.rank_by_baseline["llm_order"], 2)

    def test_headline_panel_carries_lift(self) -> None:
        model = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        )
        self.assertEqual(model.headline_panel["pass_at_1_lift"], 100.0)
        self.assertTrue(model.headline_panel["claim_allowed"])

    def test_claim_blocked_emits_note(self) -> None:
        report = _rerank_report()
        report["claim_allowed"] = False
        report["claim_reason"] = "lift=0.0"
        model = build_execution_rerank_view_model(
            rerank_report=report,
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        )
        self.assertFalse(model.claim_allowed)
        full_notes = " ".join(model.notes).lower()
        self.assertIn("claim gate not satisfied", full_notes)

    def test_view_model_is_json_serializable(self) -> None:
        model = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        )
        encoded = json.dumps(model.as_dict(), sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(
            decoded["schema_version"],
            EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION,
        )
        self.assertEqual(decoded["completion_panels"][0]["completion_id"], "c2")

    def test_missing_required_key_raises(self) -> None:
        broken = _rerank_report()
        del broken["benchmark"]
        with self.assertRaises(ExecutionRerankViewModelError):
            build_execution_rerank_view_model(
                rerank_report=broken,
                scenario_id="execution-rerank-mbpp",
                completion_records=_completions(),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
