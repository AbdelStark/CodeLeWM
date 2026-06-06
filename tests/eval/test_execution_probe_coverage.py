"""Tests for execution probe label coverage preflight and gate tables."""

from __future__ import annotations

import unittest

from codelewm.eval import (
    EXECUTION_PROBE_LABEL_BLOCKER_SCHEMA_VERSION,
    EXECUTION_PROBE_LABEL_COVERAGE_SCHEMA_VERSION,
    EXECUTION_PROBE_REPRESENTATION_GATE_TABLE_SCHEMA_VERSION,
    build_execution_probe_label_coverage,
    build_execution_probe_representation_gate_table,
)


class ExecutionProbeLabelCoverageTest(unittest.TestCase):
    def test_full_coverage_is_ready(self) -> None:
        report = build_execution_probe_label_coverage(
            (
                _record("train", passed=True, magnitude="small"),
                _record("train", passed=False, magnitude="large"),
                _record("val", passed=True, magnitude="small"),
                _record("test", passed=False, magnitude="large"),
            ),
            targets=("passed", "output_magnitude_bucket"),
        )

        self.assertEqual(
            report["schema_version"], EXECUTION_PROBE_LABEL_COVERAGE_SCHEMA_VERSION
        )
        self.assertTrue(report["coverage_ready"])
        self.assertEqual(report["blockers"], [])
        self.assertEqual(
            report["target_coverage"]["passed"]["split_applicable_counts"],
            {"test": 1, "train": 2, "val": 1},
        )

    def test_missing_val_labels_are_typed_blockers(self) -> None:
        report = build_execution_probe_label_coverage(
            (
                _record("train", passed=True, magnitude="small"),
                _record("train", passed=False, magnitude="large"),
                _record("val", passed=None, magnitude="small"),
                _record("test", passed=True, magnitude="small"),
            ),
            targets=("passed",),
        )

        blocker = report["blockers"][0]
        self.assertFalse(report["coverage_ready"])
        self.assertEqual(blocker["schema_version"], EXECUTION_PROBE_LABEL_BLOCKER_SCHEMA_VERSION)
        self.assertEqual(blocker["type"], "probe_label_eval_split_blocker")
        self.assertEqual(blocker["target"], "passed")
        self.assertEqual(blocker["split"], "val")

    def test_missing_test_labels_are_typed_blockers(self) -> None:
        report = build_execution_probe_label_coverage(
            (
                _record("train", passed=True, magnitude="small"),
                _record("train", passed=False, magnitude="large"),
                _record("val", passed=True, magnitude="small"),
                _record("test", passed=None, magnitude="large"),
            ),
            targets=("passed",),
        )

        self.assertFalse(report["coverage_ready"])
        self.assertEqual(report["blockers"][0]["split"], "test")
        self.assertIn("probe_label_eval_split_blocker:passed:test", report["blockers"][0]["reason"])

    def test_single_class_train_labels_are_typed_blockers(self) -> None:
        report = build_execution_probe_label_coverage(
            (
                _record("train", passed=True, magnitude="small"),
                _record("train", passed=True, magnitude="small"),
                _record("val", passed=False, magnitude="large"),
                _record("test", passed=True, magnitude="small"),
            ),
            targets=("passed",),
        )

        self.assertFalse(report["coverage_ready"])
        self.assertEqual(report["blockers"][0]["type"], "probe_label_train_class_blocker")
        self.assertEqual(report["blockers"][0]["observed"], 1)


class ExecutionProbeRepresentationGateTableTest(unittest.TestCase):
    def test_multi_seed_aggregation_separates_probe_control_and_claim_status(self) -> None:
        table = build_execution_probe_representation_gate_table(
            (
                _probe_report(seed_status="unsupported", passed_available=True, magnitude_available=True),
                _probe_report(seed_status="not_evaluable", passed_available=True, magnitude_available=False),
            ),
            seeds=(42, 1729),
        )

        self.assertEqual(
            table["schema_version"],
            EXECUTION_PROBE_REPRESENTATION_GATE_TABLE_SCHEMA_VERSION,
        )
        self.assertFalse(table["claim_allowed"])
        self.assertEqual(table["seed_count"], 2)
        rows = {(row["seed"], row["target"]): row for row in table["rows"]}
        self.assertEqual(rows[(42, "passed")]["best_probe_test_accuracy"], 0.75)
        self.assertEqual(rows[(42, "passed")]["best_control_test_accuracy"], 0.5)
        self.assertEqual(
            rows[(1729, "output_magnitude_bucket")]["status"],
            "not_evaluable",
        )
        self.assertEqual(table["blockers"][0]["target"], "output_magnitude_bucket")


def _record(
    split: str,
    *,
    passed: bool | None,
    magnitude: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "output_type": "int",
        "output_magnitude_bucket": magnitude,
    }
    if passed is not None:
        row["passed"] = passed
    return row


def _probe_report(
    *,
    seed_status: str,
    passed_available: bool,
    magnitude_available: bool,
) -> dict[str, object]:
    return {
        "target_reports": {
            "passed": _target_report(passed_available, probe=0.75, control=0.5),
            "output_magnitude_bucket": _target_report(
                magnitude_available,
                probe=0.6,
                control=0.55,
            ),
        },
        "claim_boundary": {
            "semantic_structure_status": seed_status,
            "positive_representation_claim_allowed": False,
        },
    }


def _target_report(available: bool, *, probe: float, control: float) -> dict[str, object]:
    if not available:
        return {
            "available": False,
            "unavailable_reason": "val and test splits must each contain target labels",
            "views": {},
            "baselines": {},
        }
    return {
        "available": True,
        "views": {
            "z_pred_after": {
                "splits": {
                    "test": {"accuracy": probe},
                },
            }
        },
        "baselines": {
            "majority": {
                "splits": {
                    "test": {"accuracy": control},
                }
            }
        },
    }


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
