"""Tests for the execution-substrate surprise decoy generators."""

from __future__ import annotations

import unittest

from codelewm.eval import (
    EXECUTION_SURPRISE_DECOY_CATEGORIES,
    generate_same_code_different_input_pairs,
    generate_same_problem_different_submission_pairs,
)


def _rec(
    *,
    problem: str,
    submission: str,
    input_id: str,
    output_repr: str,
) -> dict[str, str]:
    return {
        "record_id": f"{problem}::{submission}::{input_id}",
        "source_problem_id": problem,
        "source_submission_id": submission,
        "input_id": input_id,
        "output_repr": output_repr,
    }


class SameProblemDecoyTest(unittest.TestCase):
    def test_emits_pair_when_different_submissions_produce_different_outputs(
        self,
    ) -> None:
        records = [
            _rec(problem="p1", submission="s1", input_id="i1", output_repr="9"),
            _rec(problem="p1", submission="s2", input_id="i1", output_repr="8"),
            _rec(problem="p1", submission="s3", input_id="i2", output_repr="10"),
        ]
        pairs, report = generate_same_problem_different_submission_pairs(records)
        self.assertEqual(report.category, "same_problem_different_submission")
        self.assertEqual(report.pair_count, len(pairs))
        # s1 and s2 share problem+input, differ in output -> 2 pairs (one
        # in each direction).
        self.assertEqual(len(pairs), 2)
        problem_ids = {p.query_record_id.split("::")[0] for p in pairs}
        self.assertEqual(problem_ids, {"p1"})

    def test_identical_outputs_filtered_with_skip_reason(self) -> None:
        records = [
            _rec(problem="p1", submission="s1", input_id="i1", output_repr="9"),
            _rec(problem="p1", submission="s2", input_id="i1", output_repr="9"),
        ]
        pairs, report = generate_same_problem_different_submission_pairs(records)
        self.assertEqual(report.pair_count, 0)
        self.assertGreaterEqual(report.skipped_reasons.get("outputs_identical", 0), 2)

    def test_solo_problem_skipped(self) -> None:
        records = [
            _rec(problem="p1", submission="s1", input_id="i1", output_repr="9"),
        ]
        pairs, report = generate_same_problem_different_submission_pairs(records)
        self.assertEqual(pairs, [])
        self.assertGreaterEqual(
            report.skipped_reasons.get("no_other_submission", 0), 1
        )

    def test_deterministic_under_same_seed(self) -> None:
        records = [
            _rec(problem="p1", submission="s1", input_id="i1", output_repr="1"),
            _rec(problem="p1", submission="s2", input_id="i1", output_repr="2"),
            _rec(problem="p1", submission="s3", input_id="i1", output_repr="3"),
        ]
        a, _ = generate_same_problem_different_submission_pairs(records, seed=7)
        b, _ = generate_same_problem_different_submission_pairs(records, seed=7)
        self.assertEqual([p.decoy_record_id for p in a], [p.decoy_record_id for p in b])


class SameCodeDifferentInputDecoyTest(unittest.TestCase):
    def test_emits_pair_when_same_submission_has_different_inputs(self) -> None:
        records = [
            _rec(problem="p1", submission="s1", input_id="i1", output_repr="9"),
            _rec(problem="p1", submission="s1", input_id="i2", output_repr="8"),
        ]
        pairs, report = generate_same_code_different_input_pairs(records)
        self.assertEqual(report.category, "same_code_different_input")
        self.assertEqual(len(pairs), 2)
        for pair in pairs:
            self.assertTrue(
                pair.query_record_id.split("::")[1]
                == pair.decoy_record_id.split("::")[1]
            )

    def test_solo_input_skipped(self) -> None:
        records = [
            _rec(problem="p1", submission="s1", input_id="i1", output_repr="9"),
        ]
        pairs, _ = generate_same_code_different_input_pairs(records)
        self.assertEqual(pairs, [])

    def test_categories_are_disjoint(self) -> None:
        # The constant must list both new categories and only those.
        self.assertEqual(
            set(EXECUTION_SURPRISE_DECOY_CATEGORIES),
            {"same_problem_different_submission", "same_code_different_input"},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
