from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    SURPRISE_DECOY_CATEGORIES,
    SURPRISE_REPORT_SCHEMA_VERSION,
    SurpriseEvalError,
    SurpriseExampleInput,
    build_decoys,
    build_surprise_report,
    compute_surprise_metrics,
    read_surprise_report,
    score_surprise_example,
    surprise_report_json_schema,
    validate_surprise_report_payload,
    write_surprise_report,
)


def _corpus() -> tuple[SurpriseExampleInput, ...]:
    return (
        SurpriseExampleInput(
            transition_id="t-a",
            repo="example/repo",
            path="pkg/a.py",
            action_cluster="rename",
            true_after="def a():\n    return 1\n",
            same_file_after_states=(
                "def a():\n    return 2\n",
                "def a():\n    return 3\n",
            ),
        ),
        SurpriseExampleInput(
            transition_id="t-b",
            repo="example/repo",
            path="pkg/b.py",
            action_cluster="rename",
            true_after="def b():\n    return 10\n",
            same_file_after_states=(
                "def b():\n    return 11\n",
            ),
            action_cluster_after_states=(
                "def b():\n    return 20\n",
                "def b():\n    return 30\n",
            ),
        ),
        SurpriseExampleInput(
            transition_id="t-c",
            repo="example/repo",
            path="pkg/c.py",
            action_cluster="refactor",
            true_after="def c():\n    return 'c'\n",
        ),
    )


def _true_score_only(example: SurpriseExampleInput, candidate: str) -> float:
    # The "true" after-state always wins under lower_is_better semantics.
    return 0.0 if candidate == example.true_after else 1.0


def _deterministic_score(example: SurpriseExampleInput, candidate: str) -> float:
    # The "true" gets the lowest score; every other candidate is unique
    # and slightly worse.
    base = 0.0 if candidate == example.true_after else 0.5
    bonus = (hash(candidate) % 100) / 1000.0
    return base + bonus


class DecoyConstructionTest(unittest.TestCase):
    def test_build_decoys_returns_each_requested_category_with_unique_ids(self) -> None:
        corpus = _corpus()
        example = corpus[1]  # has both same_file and action_cluster

        decoys = build_decoys(
            example,
            corpus=corpus,
            seed=7,
            random_count=2,
            same_file_count=1,
            mutation_count=1,
            action_cluster_count=2,
        )

        categories = {decoy.category for decoy in decoys}
        self.assertSetEqual(categories, set(SURPRISE_DECOY_CATEGORIES))
        decoy_ids = [decoy.decoy_id for decoy in decoys]
        self.assertEqual(len(decoy_ids), len(set(decoy_ids)), "decoy_ids must be unique")
        for decoy in decoys:
            self.assertEqual(decoy.transition_id, example.transition_id)
            self.assertNotEqual(decoy.after, example.true_after)

    def test_build_decoys_is_deterministic_for_same_seed(self) -> None:
        corpus = _corpus()
        example = corpus[1]

        first = build_decoys(example, corpus=corpus, seed=42)
        second = build_decoys(example, corpus=corpus, seed=42)

        self.assertEqual(
            [(d.category, d.after) for d in first],
            [(d.category, d.after) for d in second],
        )

    def test_build_decoys_changes_with_different_seed(self) -> None:
        corpus = _corpus()
        example = corpus[1]

        first = build_decoys(example, corpus=corpus, seed=0, random_count=2, same_file_count=0,
                             mutation_count=2, action_cluster_count=2)
        second = build_decoys(example, corpus=corpus, seed=99, random_count=2, same_file_count=0,
                              mutation_count=2, action_cluster_count=2)

        self.assertNotEqual(
            [(d.category, d.after) for d in first],
            [(d.category, d.after) for d in second],
        )

    def test_same_file_decoys_drop_when_no_same_file_pool(self) -> None:
        corpus = _corpus()
        example = corpus[2]  # no same_file_after_states

        decoys = build_decoys(
            example,
            corpus=corpus,
            seed=1,
            random_count=1,
            same_file_count=2,
            mutation_count=0,
            action_cluster_count=0,
        )

        self.assertEqual([decoy.category for decoy in decoys], ["random"])

    def test_action_cluster_decoys_fall_back_to_corpus_when_intra_empty(self) -> None:
        corpus = _corpus()
        example = corpus[0]  # action_cluster="rename", no action_cluster_after_states

        decoys = build_decoys(
            example,
            corpus=corpus,
            seed=1,
            random_count=0,
            same_file_count=0,
            mutation_count=0,
            action_cluster_count=1,
        )

        self.assertEqual(len(decoys), 1)
        self.assertEqual(decoys[0].category, "action_cluster")

    def test_mutation_decoys_perturb_true_after(self) -> None:
        corpus = _corpus()
        example = corpus[0]

        decoys = build_decoys(
            example,
            corpus=corpus,
            seed=11,
            random_count=0,
            same_file_count=0,
            mutation_count=3,
            action_cluster_count=0,
        )

        mutated_texts = {decoy.after for decoy in decoys}
        self.assertNotIn(example.true_after, mutated_texts)
        for decoy in decoys:
            self.assertEqual(decoy.category, "mutation")
            self.assertGreater(len(decoy.after), 0)


class SurpriseScoringTest(unittest.TestCase):
    def test_score_surprise_example_assigns_rank_one_when_true_is_best(self) -> None:
        corpus = _corpus()
        example = corpus[1]
        decoys = build_decoys(example, corpus=corpus, seed=1)

        result = score_surprise_example(
            example,
            decoys=decoys,
            score_fn=_true_score_only,
        )

        self.assertEqual(result.true_rank, 1)
        self.assertEqual(result.true_score, 0.0)
        self.assertEqual(result.candidate_count, 1 + len(decoys))
        for category in SURPRISE_DECOY_CATEGORIES:
            scores = result.decoy_scores_by_category[category]
            self.assertTrue(all(math.isclose(score, 1.0) for score in scores))

    def test_score_surprise_example_rejects_foreign_decoy(self) -> None:
        corpus = _corpus()
        example = corpus[0]
        foreign_example = corpus[1]
        foreign_decoys = build_decoys(foreign_example, corpus=corpus, seed=1)

        with self.assertRaises(SurpriseEvalError):
            score_surprise_example(
                example,
                decoys=foreign_decoys,
                score_fn=_true_score_only,
            )

    def test_score_surprise_example_rejects_unknown_score_direction(self) -> None:
        corpus = _corpus()
        example = corpus[0]
        decoys = build_decoys(example, corpus=corpus, seed=1)

        with self.assertRaises(SurpriseEvalError):
            score_surprise_example(
                example,
                decoys=decoys,
                score_fn=_true_score_only,
                score_direction="unsupported",
            )


class SurpriseMetricsTest(unittest.TestCase):
    def test_metrics_reports_perfect_auc_when_true_always_wins(self) -> None:
        corpus = _corpus()
        results = [
            score_surprise_example(
                example,
                decoys=build_decoys(example, corpus=corpus, seed=7),
                score_fn=_true_score_only,
            )
            for example in corpus
        ]

        metrics = compute_surprise_metrics(results)

        self.assertAlmostEqual(metrics.pairwise_auc_overall, 1.0)
        for category, auc in metrics.pairwise_auc_by_category.items():
            if metrics.decoy_counts[category] > 0:
                self.assertAlmostEqual(auc, 1.0, msg=f"category={category}")
        self.assertAlmostEqual(metrics.recall_at_1, 1.0)
        self.assertEqual(metrics.example_count, len(corpus))
        self.assertEqual(metrics.mean_true_rank, 1.0)
        self.assertEqual(metrics.median_true_rank, 1.0)

    def test_metrics_reports_zero_auc_when_true_always_loses(self) -> None:
        corpus = _corpus()
        results = [
            score_surprise_example(
                example,
                decoys=build_decoys(example, corpus=corpus, seed=7),
                score_fn=lambda ex, candidate, _ex=example: 0.0 if candidate != _ex.true_after else 1.0,
            )
            for example in corpus
        ]

        metrics = compute_surprise_metrics(results)

        self.assertAlmostEqual(metrics.pairwise_auc_overall, 0.0)
        self.assertEqual(metrics.recall_at_1, 0.0)

    def test_metrics_with_higher_is_better_flips_winner_semantics(self) -> None:
        corpus = _corpus()

        def _higher_is_better_scorer(example: SurpriseExampleInput, candidate: str) -> float:
            return 1.0 if candidate == example.true_after else 0.0

        results = [
            score_surprise_example(
                example,
                decoys=build_decoys(example, corpus=corpus, seed=7),
                score_fn=_higher_is_better_scorer,
                score_direction="higher_is_better",
            )
            for example in corpus
        ]
        metrics = compute_surprise_metrics(results, score_direction="higher_is_better")

        self.assertAlmostEqual(metrics.pairwise_auc_overall, 1.0)
        self.assertEqual(metrics.recall_at_1, 1.0)

    def test_metrics_handles_ties_as_half_credit(self) -> None:
        corpus = _corpus()[:1]
        example = corpus[0]
        decoys = build_decoys(example, corpus=corpus, seed=7, random_count=0, same_file_count=2,
                              mutation_count=0, action_cluster_count=0)

        result = score_surprise_example(
            example,
            decoys=decoys,
            score_fn=lambda ex, candidate: 0.0,  # all ties
        )
        metrics = compute_surprise_metrics([result])

        self.assertAlmostEqual(metrics.pairwise_auc_overall, 0.5)


class SurpriseReportTest(unittest.TestCase):
    def test_build_surprise_report_round_trips_through_validation(self) -> None:
        corpus = _corpus()
        results = [
            score_surprise_example(
                example,
                decoys=build_decoys(example, corpus=corpus, seed=3),
                score_fn=_deterministic_score,
            )
            for example in corpus
        ]

        report = build_surprise_report(results, decoy_seed=3, metadata={"run": "fixture"})
        payload = report.to_dict()
        loaded = validate_surprise_report_payload(payload)
        rendered = json.dumps(payload, sort_keys=True, allow_nan=False)

        self.assertEqual(report.schema_version, SURPRISE_REPORT_SCHEMA_VERSION)
        self.assertEqual(loaded.to_dict(), payload)
        self.assertEqual(loaded.metrics.example_count, len(corpus))
        self.assertIn(SURPRISE_REPORT_SCHEMA_VERSION, rendered)

    def test_surprise_report_persists_and_reloads_from_disk(self) -> None:
        corpus = _corpus()
        results = [
            score_surprise_example(
                example,
                decoys=build_decoys(example, corpus=corpus, seed=5),
                score_fn=_deterministic_score,
            )
            for example in corpus
        ]
        report = build_surprise_report(results, decoy_seed=5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surprise.json"
            write_surprise_report(report, path)

            loaded = read_surprise_report(path)

        self.assertEqual(loaded.to_dict(), report.to_dict())

    def test_surprise_report_schema_pins_required_fields(self) -> None:
        schema = surprise_report_json_schema()

        self.assertEqual(schema["properties"]["schema_version"]["const"], SURPRISE_REPORT_SCHEMA_VERSION)
        for required in ("metrics", "examples", "decoy_seed", "score_direction", "metadata"):
            with self.subTest(required=required):
                self.assertIn(required, schema["required"])


if __name__ == "__main__":
    unittest.main()
