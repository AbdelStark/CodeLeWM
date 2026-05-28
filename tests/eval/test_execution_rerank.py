"""Tests for the execution-substrate rerank evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    EXECUTION_RERANK_REPORT_SCHEMA_VERSION,
    CompletionLabel,
    ExecutionRerankError,
    ScoredCompletion,
    load_completion_labels,
    rerank_completions,
)


def _make(
    problem: str,
    completion: str,
    rank: int,
    passed: bool,
    scores: dict[str, float],
) -> ScoredCompletion:
    return ScoredCompletion(
        label=CompletionLabel(
            problem_id=problem,
            completion_id=completion,
            code="def f(): pass\n",
            llm_order_rank=rank,
            passed=passed,
        ),
        scores=scores,
    )


class RerankProtocolTest(unittest.TestCase):
    def test_codelewm_pass_at_1_when_score_picks_passing_completion(self) -> None:
        completions = [
            _make("p1", "c1", 1, False, {"codelewm": 0.1, "lexical": 0.5}),
            _make("p1", "c2", 2, True, {"codelewm": 0.9, "lexical": 0.3}),
        ]
        report = rerank_completions(
            completions=completions, benchmark="fixture", min_lift_for_claim=0.0
        )
        self.assertEqual(
            report.schema_version, EXECUTION_RERANK_REPORT_SCHEMA_VERSION
        )
        codelewm = next(b for b in report.baselines if b.baseline == "codelewm")
        llm = next(b for b in report.baselines if b.baseline == "llm_order")
        self.assertEqual(codelewm.pass_at_1, 1.0)
        self.assertEqual(llm.pass_at_1, 0.0)
        self.assertEqual(report.codelewm_lift_over_llm_order, 100.0)

    def test_uniform_completion_counts_required(self) -> None:
        completions = [
            _make("p1", "c1", 1, True, {"codelewm": 0.9}),
            _make("p2", "c1", 1, False, {"codelewm": 0.1}),
            _make("p2", "c2", 2, True, {"codelewm": 0.9}),
        ]
        with self.assertRaises(ExecutionRerankError):
            rerank_completions(completions=completions, benchmark="fixture")

    def test_codelewm_score_required(self) -> None:
        completions = [
            _make("p1", "c1", 1, True, {"lexical": 0.9}),
            _make("p1", "c2", 2, False, {"lexical": 0.1}),
        ]
        with self.assertRaises(ExecutionRerankError):
            rerank_completions(completions=completions, benchmark="fixture")

    def test_empty_completions_rejected(self) -> None:
        with self.assertRaises(ExecutionRerankError):
            rerank_completions(completions=[], benchmark="fixture")

    def test_lift_ci_excludes_zero_when_codelewm_dominates(self) -> None:
        # 5 problems where CodeLeWM picks the passing one and LLM order
        # picks the failing one every time.
        completions = []
        for i in range(5):
            completions.append(
                _make(
                    f"p{i}",
                    f"p{i}-c1",
                    1,
                    False,
                    {"codelewm": 0.1, "lexical": 0.5},
                )
            )
            completions.append(
                _make(
                    f"p{i}",
                    f"p{i}-c2",
                    2,
                    True,
                    {"codelewm": 0.9, "lexical": 0.3},
                )
            )
        report = rerank_completions(
            completions=completions,
            benchmark="fixture",
            bootstrap_samples=500,
            min_lift_for_claim=3.0,
        )
        self.assertGreater(report.codelewm_lift_over_llm_order, 50.0)
        self.assertGreater(report.bootstrap_lift_ci[0], 0.0)
        self.assertTrue(report.claim_allowed)

    def test_claim_blocked_when_lift_is_small(self) -> None:
        completions = []
        for i in range(5):
            # Both orderings pass on each problem -> zero lift.
            completions.append(
                _make(
                    f"p{i}", f"p{i}-c1", 1, True, {"codelewm": 0.9}
                )
            )
            completions.append(
                _make(
                    f"p{i}", f"p{i}-c2", 2, True, {"codelewm": 0.1}
                )
            )
        report = rerank_completions(
            completions=completions,
            benchmark="fixture",
            bootstrap_samples=200,
            min_lift_for_claim=3.0,
        )
        self.assertEqual(report.codelewm_lift_over_llm_order, 0.0)
        self.assertFalse(report.claim_allowed)


class CompletionLabelsLoaderTest(unittest.TestCase):
    def test_load_filters_by_benchmark_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "benchmark_id": "humaneval",
                            "problem_id": "HumanEval/0",
                            "completion_id": "HumanEval/0::0",
                            "code": "def f():\n    return 1\n",
                            "llm_order_rank": 1,
                            "passed": True,
                        },
                        {
                            "benchmark_id": "mbpp_plus",
                            "problem_id": "Mbpp/1",
                            "completion_id": "Mbpp/1::0",
                            "code": "def g():\n    return 2\n",
                            "llm_order_rank": 1,
                            "passed": False,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            humaneval = load_completion_labels(path, benchmark_id="humaneval")
            mbpp = load_completion_labels(path, benchmark_id="mbpp_plus")
            self.assertEqual(len(humaneval), 1)
            self.assertEqual(humaneval[0].problem_id, "HumanEval/0")
            self.assertEqual(len(mbpp), 1)
            self.assertEqual(mbpp[0].problem_id, "Mbpp/1")

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ExecutionRerankError):
            load_completion_labels(
                Path("/nonexistent.jsonl"), benchmark_id="humaneval"
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
