from __future__ import annotations

import json
import unittest

from codelewm.data.hard_negative_labeler import (
    HardNegativeLabelerError,
    LabelTestCase,
    build_sandbox_label_construction_report,
    label_candidate,
    label_candidates,
)
from codelewm.data.sandbox import run_one


REFERENCE = (
    "def accumulate(values):\n"
    "    total = 0\n"
    "    for value in values:\n"
    "        total = total + value\n"
    "    return total\n"
)
WRONG = (
    "def accumulate(values):\n"
    "    total = 0\n"
    "    for value in values:\n"
    "        total = total - value\n"
    "    return total\n"
)
# A single-element outer list splats to one positional arg (the values list).
INPUT_REPR = json.dumps([[1, 2, 3, 4]])


def _expected_output() -> str:
    result = run_one(REFERENCE, input_repr=INPUT_REPR, function_name="accumulate")
    assert result.ok, result.exit_code
    assert result.output_repr is not None
    return result.output_repr


class HardNegativeLabelerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = (
            LabelTestCase(
                input_id="case_0",
                input_repr=INPUT_REPR,
                expected_output=_expected_output(),
                function_name="accumulate",
            ),
        )

    def test_passing_candidate_is_labeled_pass(self) -> None:
        label = label_candidate(
            candidate_id="ref", after_text=REFERENCE, test_cases=self.cases
        )
        self.assertEqual(label.label, "pass")
        self.assertEqual(len(label.case_results), 1)
        self.assertTrue(label.case_results[0]["passed"])
        self.assertEqual(label.case_results[0]["exit_code"], "ok")

    def test_wrong_candidate_is_labeled_fail(self) -> None:
        label = label_candidate(
            candidate_id="wrong", after_text=WRONG, test_cases=self.cases
        )
        self.assertEqual(label.label, "fail")
        self.assertFalse(label.case_results[0]["passed"])

    def test_label_candidates_batch_and_report(self) -> None:
        labels = label_candidates(
            {"ref": REFERENCE, "wrong": WRONG}, test_cases=self.cases
        )
        self.assertEqual(labels["ref"].label, "pass")
        self.assertEqual(labels["wrong"].label, "fail")
        report = build_sandbox_label_construction_report(labels)
        self.assertTrue(report["sandbox_used"])
        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["label_counts"]["pass"], 1)
        self.assertEqual(report["label_counts"]["fail"], 1)
        self.assertEqual(report["sandbox_policy_version"], "codelewm.sandbox_policy.v1")
        self.assertEqual(report["label_source_counts"], {"sandbox": 2})

    def test_requires_at_least_one_test_case(self) -> None:
        with self.assertRaises(HardNegativeLabelerError):
            label_candidate(candidate_id="ref", after_text=REFERENCE, test_cases=())


if __name__ == "__main__":
    unittest.main()
