from __future__ import annotations

import unittest

from codelewm.eval import (
    ANTI_SATURATION_PROFILE,
    DOWNSTREAM_ANTI_SATURATION_CLAIM_GATE_SCHEMA_VERSION,
    DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION,
    DownstreamAntiSaturationError,
    anti_saturation_report_json_schema,
    build_anti_saturation_claim_gate,
    build_anti_saturation_report,
    validate_anti_saturation_claim_gate,
    validate_anti_saturation_report,
    validate_hard_negative_class,
)


def _eligible_kwargs(**overrides):
    base = dict(
        profile=ANTI_SATURATION_PROFILE,
        problem_count=120,
        pool_sizes=[6, 8, 10],
        baseline_pass_at_1={
            "no_action": 0.40,
            "lexical": 0.42,
            "llm_order": 0.55,
            "random": 0.20,
        },
        expected_random_pass_at_1=0.22,
        dual_hard_negative_fraction=0.80,
        hard_negative_class_coverage={"no_action_bait": 40, "wrong_symbol": 35},
        parser_apply_failure_rate=0.05,
    )
    base.update(overrides)
    return base


class AntiSaturationReportTest(unittest.TestCase):
    def test_eligible_slice(self) -> None:
        report = build_anti_saturation_report(**_eligible_kwargs())
        self.assertEqual(
            report["schema_version"], DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION
        )
        self.assertTrue(report["eligible"])
        self.assertEqual(report["blocked_reasons"], [])
        self.assertTrue(report["no_action_below_ceiling"])
        self.assertTrue(report["lexical_below_ceiling"])
        self.assertTrue(report["llm_order_below_ceiling"])
        self.assertTrue(report["candidate_pool_size"]["ok"])
        validate_anti_saturation_report(report)

    def test_saturated_slice_no_action(self) -> None:
        report = build_anti_saturation_report(
            **_eligible_kwargs(
                baseline_pass_at_1={
                    "no_action": 0.91,
                    "lexical": 0.42,
                    "llm_order": 0.55,
                    "random": 0.20,
                }
            )
        )
        self.assertFalse(report["eligible"])
        self.assertFalse(report["no_action_below_ceiling"])
        self.assertIn("no_action_saturated:0.91>=0.85", report["blocked_reasons"])

    def test_saturated_slice_llm_order(self) -> None:
        report = build_anti_saturation_report(
            **_eligible_kwargs(
                baseline_pass_at_1={
                    "no_action": 0.40,
                    "lexical": 0.42,
                    "llm_order": 0.95,
                    "random": 0.20,
                }
            )
        )
        self.assertFalse(report["eligible"])
        self.assertIn("llm_order_saturated:0.95>=0.9", report["blocked_reasons"])

    def test_too_small_slice(self) -> None:
        report = build_anti_saturation_report(**_eligible_kwargs(problem_count=40))
        self.assertFalse(report["eligible"])
        self.assertIn("problem_count_below_minimum:40<100", report["blocked_reasons"])

    def test_pool_size_out_of_range(self) -> None:
        report = build_anti_saturation_report(**_eligible_kwargs(pool_sizes=[3, 6]))
        self.assertFalse(report["eligible"])
        self.assertFalse(report["candidate_pool_size"]["ok"])
        self.assertTrue(
            any(r.startswith("candidate_pool_size_out_of_range") for r in report["blocked_reasons"])
        )

    def test_missing_baseline_is_typed_blocker(self) -> None:
        report = build_anti_saturation_report(
            **_eligible_kwargs(
                baseline_pass_at_1={"no_action": 0.40, "lexical": 0.42, "random": 0.20}
            )
        )
        self.assertFalse(report["eligible"])
        self.assertIn("missing_baseline:llm_order", report["blocked_reasons"])
        self.assertIsNone(report["llm_order_below_ceiling"])
        self.assertIsNone(report["baseline_pass_at_1"]["llm_order"])

    def test_dual_coverage_below_threshold(self) -> None:
        report = build_anti_saturation_report(
            **_eligible_kwargs(dual_hard_negative_fraction=0.50)
        )
        self.assertFalse(report["eligible"])
        self.assertIn(
            "dual_hard_negative_coverage_below:0.5<0.7", report["blocked_reasons"]
        )

    def test_open_gates_block(self) -> None:
        report = build_anti_saturation_report(
            **_eligible_kwargs(source_license_ok=False, split_leakage_ok=False)
        )
        self.assertFalse(report["eligible"])
        self.assertIn("source_license_gate_open", report["blocked_reasons"])
        self.assertIn("split_leakage_gate_open", report["blocked_reasons"])

    def test_report_json_schema_id(self) -> None:
        schema = anti_saturation_report_json_schema()
        self.assertEqual(schema["$id"], DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION)


class AntiSaturationClaimGateTest(unittest.TestCase):
    def _winning_metrics(self):
        return {
            "codelewm": {"pass_at_1": 0.70, "mrr": 0.80},
            "no_action": {"pass_at_1": 0.40, "mrr": 0.55},
            "lexical": {"pass_at_1": 0.42, "mrr": 0.57},
            "llm_order": {"pass_at_1": 0.55, "mrr": 0.66},
        }

    def test_claim_gate_allows_when_codelewm_beats_all_three(self) -> None:
        gate = build_anti_saturation_claim_gate(
            example_count=150,
            metrics=self._winning_metrics(),
            anti_saturation_eligible=True,
        )
        self.assertEqual(
            gate["schema_version"], DOWNSTREAM_ANTI_SATURATION_CLAIM_GATE_SCHEMA_VERSION
        )
        self.assertTrue(gate["allowed"])
        self.assertEqual(gate["checked_baselines"], ["no_action", "lexical", "llm_order"])
        validate_anti_saturation_claim_gate(gate)

    def test_claim_gate_blocks_when_slice_not_eligible(self) -> None:
        gate = build_anti_saturation_claim_gate(
            example_count=150,
            metrics=self._winning_metrics(),
            anti_saturation_eligible=False,
        )
        self.assertFalse(gate["allowed"])
        self.assertIn("anti_saturation_slice_not_eligible", gate["failure_reasons"])

    def test_claim_gate_blocks_when_not_strictly_above(self) -> None:
        metrics = self._winning_metrics()
        metrics["llm_order"]["pass_at_1"] = 0.70  # tie with codelewm
        gate = build_anti_saturation_claim_gate(
            example_count=150, metrics=metrics, anti_saturation_eligible=True
        )
        self.assertFalse(gate["allowed"])
        self.assertTrue(
            any(r.startswith("not_strictly_above:llm_order:pass_at_1") for r in gate["failure_reasons"])
        )

    def test_claim_gate_blocks_on_small_example_count(self) -> None:
        gate = build_anti_saturation_claim_gate(
            example_count=40,
            metrics=self._winning_metrics(),
            anti_saturation_eligible=True,
        )
        self.assertFalse(gate["allowed"])
        self.assertIn("example_count_below_minimum:40<100", gate["failure_reasons"])

    def test_claim_gate_requires_lift_ci_above_zero(self) -> None:
        intervals = {
            "no_action": {"pass_at_1": {"low": 0.10, "high": 0.30}, "mrr": {"low": 0.10, "high": 0.30}},
            "lexical": {"pass_at_1": {"low": 0.08, "high": 0.28}, "mrr": {"low": 0.08, "high": 0.28}},
            "llm_order": {"pass_at_1": {"low": -0.01, "high": 0.20}, "mrr": {"low": 0.05, "high": 0.25}},
        }
        gate = build_anti_saturation_claim_gate(
            example_count=150,
            metrics=self._winning_metrics(),
            anti_saturation_eligible=True,
            lift_confidence_intervals=intervals,
        )
        self.assertFalse(gate["allowed"])
        self.assertTrue(
            any(r.startswith("lift_ci_includes_zero:llm_order:pass_at_1") for r in gate["failure_reasons"])
        )


class HardNegativeClassTest(unittest.TestCase):
    def test_valid_class_passes(self) -> None:
        self.assertEqual(validate_hard_negative_class("wrong_symbol"), "wrong_symbol")

    def test_invalid_class_raises(self) -> None:
        with self.assertRaises(DownstreamAntiSaturationError):
            validate_hard_negative_class("totally_made_up")


if __name__ == "__main__":
    unittest.main()
