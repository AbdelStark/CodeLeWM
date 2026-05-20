from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    ACTION_USE_CLAIM_GATE_SCHEMA_VERSION,
    CANDIDATE_POOL_SCHEMA_VERSION,
    RETRIEVAL_REPORT_SCHEMA_VERSION,
    CandidatePoolEntry,
    RetrievalMetrics,
    RetrievalEvalError,
    build_action_use_claim_gate,
    build_easy_candidate_pool,
    build_fixture_candidate_pool,
    build_retrieval_report,
    compute_retrieval_metrics,
    rank_targets,
    read_retrieval_report,
    validate_retrieval_report_payload,
    write_retrieval_report,
)


class RetrievalMetricsTest(unittest.TestCase):
    def test_known_rank_metrics_match_recall_mrr_and_median(self) -> None:
        metrics = compute_retrieval_metrics((1, 2, 6, 11), candidate_counts=(12, 12, 12, 12))

        self.assertEqual(metrics.query_count, 4)
        self.assertEqual(metrics.candidate_count_min, 12)
        self.assertEqual(metrics.candidate_count_max, 12)
        self.assertAlmostEqual(metrics.recall_at_1, 0.25)
        self.assertAlmostEqual(metrics.recall_at_5, 0.50)
        self.assertAlmostEqual(metrics.recall_at_10, 0.75)
        self.assertAlmostEqual(metrics.mrr, (1.0 + 0.5 + 1.0 / 6.0 + 1.0 / 11.0) / 4.0)
        self.assertAlmostEqual(metrics.median_rank, 4.0)

    def test_rank_targets_uses_scores_and_deterministic_tie_order(self) -> None:
        ranks = rank_targets(
            score_rows=((0.9, 0.1, 0.2), (0.6, 0.6, 0.1), (0.1, 0.2, 0.3)),
            candidate_ids_by_query=(("target", "a", "b"), ("a", "target", "b"), ("a", "b", "target")),
            target_ids=("target", "target", "target"),
        )

        self.assertEqual(ranks, (1, 2, 1))

    def test_rank_targets_supports_energy_where_lower_is_better(self) -> None:
        ranks = rank_targets(
            score_rows=((0.2, 0.1, 0.4),),
            candidate_ids_by_query=(("a", "target", "b"),),
            target_ids=("target",),
            larger_is_better=False,
        )

        self.assertEqual(ranks, (1,))

    def test_rank_targets_rejects_missing_target_and_nonfinite_scores(self) -> None:
        with self.assertRaisesRegex(RetrievalEvalError, "exactly once"):
            rank_targets(((1.0,),), (("a",),), ("target",))
        with self.assertRaisesRegex(RetrievalEvalError, "NaN or inf"):
            rank_targets(((float("nan"),),), (("target",),), ("target",))


class CandidatePoolTest(unittest.TestCase):
    def test_easy_pool_is_deterministic_and_excludes_train_rows(self) -> None:
        rows = [
            _row("train-0", split="train"),
            _row("test-0", split="test"),
            _row("test-1", split="test"),
            _row("val-0", split="val"),
        ]

        pool_a = build_easy_candidate_pool(rows, max_size=2, seed=7)
        pool_b = build_easy_candidate_pool(reversed(rows), max_size=2, seed=7)

        self.assertEqual(pool_a.schema_version, CANDIDATE_POOL_SCHEMA_VERSION)
        self.assertEqual(pool_a.name, "easy-1k")
        self.assertEqual(pool_a.candidate_ids, pool_b.candidate_ids)
        self.assertEqual(len(pool_a.entries), 2)
        self.assertNotIn("train-0", pool_a.candidate_ids)
        self.assertTrue(all(entry.split in {"val", "test"} for entry in pool_a.entries))

    def test_fixture_pool_preserves_explicit_order(self) -> None:
        rows = [_row("test-0", split="test"), _row("val-0", split="val")]

        pool = build_fixture_candidate_pool(rows, candidate_ids=("val-0", "test-0"))

        self.assertEqual(pool.name, "fixture")
        self.assertEqual(pool.candidate_ids, ("val-0", "test-0"))

    def test_fixture_pool_rejects_training_leakage(self) -> None:
        rows = [_row("train-0", split="train"), _row("test-0", split="test")]

        with self.assertRaisesRegex(RetrievalEvalError, "training rows"):
            build_fixture_candidate_pool(rows)

    def test_fixture_pool_rejects_duplicate_transition_ids(self) -> None:
        rows = [_row("test-0", split="test"), _row("test-0", split="val")]

        with self.assertRaisesRegex(RetrievalEvalError, "duplicate candidate row id"):
            build_fixture_candidate_pool(rows)


class RetrievalReportTest(unittest.TestCase):
    def test_report_round_trips_json_with_baselines_slices_and_candidate_pool(self) -> None:
        pool = build_fixture_candidate_pool(
            [_row("test-0", split="test"), _row("test-1", split="test")],
            candidate_ids=("test-0", "test-1"),
        )
        report = build_retrieval_report(
            (1, 2),
            candidate_counts=(2, 2),
            candidate_pool=pool,
            baselines={"random": (2, 1)},
            slices={"source:synthetic": (1,)},
            metadata={"action_view": "text"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrieval-report.json"
            write_retrieval_report(report, path)
            payload = json.loads(path.read_text())
            loaded = read_retrieval_report(path)

        self.assertEqual(payload["schema_version"], RETRIEVAL_REPORT_SCHEMA_VERSION)
        self.assertEqual(payload["candidate_pool"]["entry_count"], 2)
        self.assertIn("random", payload["baselines"])
        self.assertIn("baseline_deltas", payload)
        self.assertIn("action_use_claim_gate", payload)
        self.assertFalse(payload["action_use_claim_gate"]["claim_allowed"])
        self.assertIn("source:synthetic", payload["slices"])
        self.assertEqual(loaded.to_dict(), report.to_dict())

    def test_report_payload_validation_rejects_unknown_schema(self) -> None:
        report = build_retrieval_report((1,))
        payload = report.to_dict()
        payload["schema_version"] = "codelewm.eval.retrieval_report.v0"

        with self.assertRaisesRegex(RetrievalEvalError, "unsupported retrieval report schema"):
            validate_retrieval_report_payload(payload)

    def test_report_payload_validation_rejects_mismatched_top_level_metric(self) -> None:
        report = build_retrieval_report((1, 2))
        payload = report.to_dict()
        payload["recall_at_1"] = 0.0

        with self.assertRaisesRegex(RetrievalEvalError, "does not match metrics"):
            validate_retrieval_report_payload(payload)

    def test_report_payload_validation_rejects_mismatched_candidate_pool_count(self) -> None:
        pool = build_fixture_candidate_pool([_row("test-0", split="test")])
        report = build_retrieval_report((1,), candidate_pool=pool)
        payload = report.to_dict()
        payload["candidate_pool"]["entry_count"] = 2

        with self.assertRaisesRegex(RetrievalEvalError, "entry_count"):
            validate_retrieval_report_payload(payload)


class ActionUseClaimGateTest(unittest.TestCase):
    def test_gate_passes_when_text_action_beats_required_baselines(self) -> None:
        metrics = compute_retrieval_metrics((1, 2, 2), candidate_counts=(3, 3, 3))
        baselines = {
            "random": compute_retrieval_metrics((2, 3, 3), candidate_counts=(3, 3, 3)),
            "lexical": compute_retrieval_metrics((2, 3, 3), candidate_counts=(3, 3, 3)),
            "no_action": compute_retrieval_metrics((2, 3, 3), candidate_counts=(3, 3, 3)),
            "shuffled_action": compute_retrieval_metrics((2, 3, 3), candidate_counts=(3, 3, 3)),
        }

        gate = build_action_use_claim_gate(metrics, baselines)

        self.assertEqual(gate.schema_version, ACTION_USE_CLAIM_GATE_SCHEMA_VERSION)
        self.assertTrue(gate.claim_allowed)
        self.assertEqual(gate.failure_reasons, ())
        self.assertGreater(gate.baseline_deltas["no_action"].recall_at_1_delta, 0.0)
        self.assertTrue(gate.baseline_deltas["no_action"].text_action_beats_baseline)

    def test_gate_fails_scaled_run_numbers_on_no_action_dominance(self) -> None:
        metrics = RetrievalMetrics(
            query_count=1000,
            candidate_count_min=1000,
            candidate_count_max=1000,
            recall_at_1=0.371,
            recall_at_5=0.586,
            recall_at_10=0.672,
            mrr=0.472984,
            median_rank=3.0,
        )
        baselines = {
            "random": RetrievalMetrics(
                query_count=1000,
                candidate_count_min=1000,
                candidate_count_max=1000,
                recall_at_1=0.001,
                recall_at_5=0.004,
                recall_at_10=0.008,
                mrr=0.007118,
                median_rank=502.0,
            ),
            "lexical": RetrievalMetrics(
                query_count=1000,
                candidate_count_min=1000,
                candidate_count_max=1000,
                recall_at_1=0.045,
                recall_at_5=0.130,
                recall_at_10=0.190,
                mrr=0.093745,
                median_rank=152.0,
            ),
            "no_action": RetrievalMetrics(
                query_count=1000,
                candidate_count_min=1000,
                candidate_count_max=1000,
                recall_at_1=0.459,
                recall_at_5=0.641,
                recall_at_10=0.712,
                mrr=0.546116,
                median_rank=2.0,
            ),
            "shuffled_action": RetrievalMetrics(
                query_count=1000,
                candidate_count_min=1000,
                candidate_count_max=1000,
                recall_at_1=0.001,
                recall_at_5=0.006,
                recall_at_10=0.011,
                mrr=0.007518,
                median_rank=510.0,
            ),
        }

        gate = build_action_use_claim_gate(metrics, baselines)

        self.assertFalse(gate.claim_allowed)
        self.assertIn(
            "no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action",
            gate.failure_reasons,
        )
        no_action_delta = gate.baseline_deltas["no_action"]
        self.assertAlmostEqual(no_action_delta.recall_at_1_delta, -0.088)
        self.assertAlmostEqual(no_action_delta.mrr_delta, -0.073132)
        self.assertFalse(no_action_delta.text_action_beats_baseline)


def _row(transition_id: str, *, split: str) -> CandidatePoolEntry:
    return CandidatePoolEntry(
        transition_id=transition_id,
        split=split,
        source="synthetic",
        repo="example/repo",
        path="pkg/mod.py",
        edit_size=2,
    )


if __name__ == "__main__":
    unittest.main()
