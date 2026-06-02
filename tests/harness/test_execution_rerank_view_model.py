"""Tests for the execution-substrate rerank visual view model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.harness import (
    EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION,
    ExecutionRerankViewModelError,
    build_execution_rerank_view_model,
    read_execution_rerank_view_model,
    validate_execution_rerank_view_model_payload,
    write_execution_rerank_view_model,
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

    def test_no_action_panel_marks_missing_baseline_as_not_recorded(self) -> None:
        model = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        )
        panel = model.no_action_panel
        self.assertEqual(panel["status"], "not_recorded")
        self.assertIsNone(panel["no_action_pass_at_1"])
        self.assertEqual(panel["interpretation"], "not_recorded")

    def test_no_action_panel_computes_delta_when_baseline_present(self) -> None:
        report = _rerank_report()
        report["baselines"].append(
            {
                "baseline": "no_action",
                "pass_at_1": 0.0,
                "pass_count": 0,
                "problem_count": 1,
            }
        )
        report["codelewm_lift_over_no_action"] = 100.0
        report["bootstrap_lift_over_no_action_ci"] = [60.0, 100.0]
        model = build_execution_rerank_view_model(
            rerank_report=report,
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        )
        panel = model.no_action_panel
        self.assertEqual(panel["status"], "available")
        self.assertEqual(panel["no_action_pass_at_1"], 0.0)
        self.assertEqual(panel["codelewm_pass_at_1"], 1.0)
        self.assertEqual(panel["codelewm_minus_no_action"], 1.0)
        self.assertEqual(panel["interpretation"], "better_than_no_action")
        self.assertEqual(panel["codelewm_lift_over_no_action"], 100.0)
        self.assertEqual(panel["bootstrap_lift_over_no_action_ci"], [60.0, 100.0])
        self.assertEqual(model.headline_panel["no_action_pass_at_1"], 0.0)

    def test_diagnostic_slots_stay_explicit(self) -> None:
        model = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
            diagnostics={
                "checkpoint": {"model_id": "codelewm.scorer.v1", "device": "cpu"},
            },
        )
        diagnostics = model.diagnostics
        # Provided slot gets an explicit available status.
        self.assertEqual(diagnostics["checkpoint"]["status"], "available")
        self.assertEqual(diagnostics["checkpoint"]["model_id"], "codelewm.scorer.v1")
        # Unprovided slots are still present and explicit.
        self.assertEqual(diagnostics["retrieval_evidence"]["status"], "not_recorded")
        self.assertEqual(diagnostics["sandbox"]["status"], "not_recorded")

    def test_artifact_lineage_carries_parents_command_and_paths(self) -> None:
        model = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
            artifact_lineage={
                "parent_artifact_ids": ["candidate_pack-abc123"],
                "command": ["scripts/llm-world-model-demo", "--tour", "2"],
                "manifest_path": "manifest.json",
                "view_model_path": "reports/execution_rerank_view_model.json",
                "html_path": "demo.html",
            },
        )
        lineage = model.artifact_lineage
        self.assertEqual(lineage["parent_artifact_ids"], ["candidate_pack-abc123"])
        self.assertEqual(
            lineage["command"], ["scripts/llm-world-model-demo", "--tour", "2"]
        )
        self.assertEqual(lineage["manifest_path"], "manifest.json")
        self.assertEqual(lineage["html_path"], "demo.html")
        self.assertIsNone(lineage["asciicast_path"])

    def test_validate_round_trips_through_disk(self) -> None:
        model = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        )
        payload = model.as_dict()
        validated = validate_execution_rerank_view_model_payload(payload)
        self.assertEqual(
            validated["schema_version"], EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "view_model.json"
            write_execution_rerank_view_model(payload, path)
            reloaded = read_execution_rerank_view_model(path)
        self.assertEqual(reloaded["completion_panels"][0]["completion_id"], "c2")
        self.assertIn("no_action_panel", reloaded)
        self.assertIn("diagnostics", reloaded)
        self.assertIn("artifact_lineage", reloaded)

    def test_validate_rejects_wrong_schema_version(self) -> None:
        payload = build_execution_rerank_view_model(
            rerank_report=_rerank_report(),
            scenario_id="execution-rerank-mbpp",
            completion_records=_completions(),
        ).as_dict()
        payload["schema_version"] = "codelewm.harness.execution_rerank_view_model.v0"
        with self.assertRaises(ExecutionRerankViewModelError):
            validate_execution_rerank_view_model_payload(payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
