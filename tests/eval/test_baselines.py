from __future__ import annotations

import unittest

from codelewm.eval import (
    REQUIRED_HEADLINE_BASELINES,
    CandidatePool,
    CandidatePoolEntry,
    RetrievalEvalError,
    build_baseline_metrics,
    build_retrieval_report,
    lexical_baseline_ranks,
    no_action_baseline_ranks,
    random_baseline_ranks,
    shuffled_action_baseline_ranks,
    validate_required_headline_baselines,
)


class RetrievalBaselineTest(unittest.TestCase):
    def test_random_baseline_is_seeded_and_returns_valid_ranks(self) -> None:
        candidate_ids = (("a", "target", "b"), ("x", "y", "target"))
        target_ids = ("target", "target")

        ranks_a = random_baseline_ranks(candidate_ids, target_ids, seed=17)
        ranks_b = random_baseline_ranks(candidate_ids, target_ids, seed=17)

        self.assertEqual(ranks_a, ranks_b)
        self.assertEqual(len(ranks_a), 2)
        self.assertTrue(all(1 <= rank <= 3 for rank in ranks_a))

    def test_lexical_baseline_prefers_matching_candidate_text(self) -> None:
        ranks = lexical_baseline_ranks(
            query_texts=("add timeout retry", "parse json payload"),
            candidate_texts_by_query=(
                ("retry timeout handling", "render button color"),
                ("database transaction", "json payload parser"),
            ),
            candidate_ids_by_query=(("target-0", "wrong-0"), ("wrong-1", "target-1")),
            target_ids=("target-0", "target-1"),
        )

        self.assertEqual(ranks, (1, 1))

    def test_no_action_and_shuffled_action_paths_use_score_rows(self) -> None:
        candidate_ids = (("target-0", "wrong-0"), ("wrong-1", "target-1"))
        target_ids = ("target-0", "target-1")

        no_action = no_action_baseline_ranks(
            ((0.1, 0.9), (0.8, 0.2)),
            candidate_ids,
            target_ids,
        )
        shuffled = shuffled_action_baseline_ranks(
            ((0.9, 0.1), (0.2, 0.8)),
            candidate_ids,
            target_ids,
            seed=3,
        )

        self.assertEqual(no_action, (2, 2))
        self.assertEqual(shuffled, (2, 2))

    def test_shuffled_action_degrades_controlled_same_before_fixture(self) -> None:
        candidate_ids = (
            ("target-add-timeout", "wrong-rename-symbol"),
            ("wrong-add-timeout", "target-rename-symbol"),
        )
        target_ids = ("target-add-timeout", "target-rename-symbol")
        action_conditioned_scores = (
            (0.95, 0.10),
            (0.10, 0.95),
        )

        text_action = no_action_baseline_ranks(
            action_conditioned_scores,
            candidate_ids,
            target_ids,
        )
        shuffled_action = shuffled_action_baseline_ranks(
            action_conditioned_scores,
            candidate_ids,
            target_ids,
            seed=0,
        )

        self.assertEqual(text_action, (1, 1))
        self.assertEqual(shuffled_action, (2, 2))

    def test_required_headline_baselines_validate_report_contract(self) -> None:
        baselines = build_baseline_metrics(
            {
                "random": (2, 1),
                "lexical": (1, 1),
                "no_action": (2, 2),
                "shuffled_action": (2, 2),
            }
        )
        report = build_retrieval_report((1, 2), baselines=baselines)

        validated = validate_required_headline_baselines(report)

        self.assertEqual(validated, report)
        self.assertEqual(tuple(baselines), REQUIRED_HEADLINE_BASELINES)

    def test_required_headline_baselines_reject_missing_baseline(self) -> None:
        report = build_retrieval_report((1,), baselines={"random": (1,)})

        with self.assertRaisesRegex(RetrievalEvalError, "missing required baselines"):
            validate_required_headline_baselines(report)

    def test_candidate_pool_round_trips_seed_zero(self) -> None:
        pool = CandidatePool(
            name="fixture",
            seed=0,
            max_size=10,
            entries=(
                CandidatePoolEntry(
                    transition_id="held-out-1",
                    split="val",
                    source="fixture",
                    repo="example/repo",
                    path="pkg/example.py",
                ),
            ),
        )

        round_tripped = CandidatePool.from_dict(pool.to_dict())

        self.assertEqual(round_tripped.seed, 0)
        self.assertEqual(round_tripped.to_dict(), pool.to_dict())


if __name__ == "__main__":
    unittest.main()
