from __future__ import annotations

import unittest

from codelewm.eval import (
    CandidatePoolEntry,
    HardNegativeSamplerConfig,
    HardNegativeSamplerReport,
    RetrievalEvalError,
    build_hard_candidate_pool,
    build_hard_negative_sampler_report,
    sample_hard_negatives,
)


class HardNegativeSamplerTest(unittest.TestCase):
    def test_sampler_prefers_hard_features_and_excludes_target_and_train_rows(self) -> None:
        query = _row(
            "target",
            split="test",
            source="synthetic",
            edit_size=7,
            metadata={"action_cluster": "return-change"},
        )
        candidates = [
            query,
            _row(
                "train-near",
                split="train",
                source="synthetic",
                edit_size=8,
                metadata={"action_cluster": "return-change", "similarity": 1.0},
            ),
            _row(
                "same-all",
                split="test",
                source="synthetic",
                edit_size=8,
                metadata={"action_cluster": "return-change", "similarity": 0.4},
            ),
            _row(
                "same-source-action",
                split="test",
                source="synthetic",
                edit_size=24,
                metadata={"action_cluster": "return-change", "similarity": 0.9},
            ),
            _row("same-bucket", split="val", source="local_repo", edit_size=5),
            _row("fallback", split="test", source="local_repo", edit_size=31),
        ]

        sample = sample_hard_negatives(
            query,
            candidates,
            config=HardNegativeSamplerConfig(max_negatives=3, seed=3, edit_size_bucket_width=10),
        )

        self.assertEqual(sample.negative_ids, ("same-all", "same-source-action", "same-bucket"))
        self.assertEqual(sample.rejected, {"true_target": 1, "train_leakage": 1})
        self.assertEqual(sample.composition["same_source"], 2)
        self.assertEqual(sample.composition["same_edit_size_bucket"], 2)
        self.assertEqual(sample.composition["same_action_cluster"], 2)
        self.assertEqual(sample.composition["similarity_ranked"], 2)
        self.assertEqual(sample.composition["fallback"], 0)

    def test_hard_candidate_pool_contains_target_first_and_sampler_metadata(self) -> None:
        query = _row(
            "target",
            split="test",
            source="synthetic",
            edit_size=4,
            metadata={"weak_action_cluster": "guard"},
        )
        candidates = [
            query,
            _row(
                "same-action",
                split="test",
                source="synthetic",
                edit_size=6,
                metadata={"weak_action_cluster": "guard"},
            ),
            _row("fallback", split="val", source="local_repo", edit_size=99),
        ]
        config = HardNegativeSamplerConfig(max_negatives=2, seed=5, edit_size_bucket_width=10)

        pool, sample = build_hard_candidate_pool(query, candidates, config=config)

        self.assertEqual(pool.name, "hard-1k")
        self.assertEqual(pool.candidate_ids, ("target", "same-action", "fallback"))
        self.assertNotIn("target", sample.negative_ids)
        self.assertEqual(pool.metadata["query_id"], "target")
        self.assertEqual(pool.metadata["sampler"]["returned_negatives"], 2)

    def test_sampler_rejects_train_query_duplicate_ids_and_nonfinite_similarity(self) -> None:
        train_query = _row("target", split="train")
        with self.assertRaisesRegex(RetrievalEvalError, "must be held out"):
            sample_hard_negatives(train_query, [train_query])

        query = _row("target", split="test")
        duplicate = [_row("dup", split="test"), _row("dup", split="val")]
        with self.assertRaisesRegex(RetrievalEvalError, "duplicate candidate row id"):
            sample_hard_negatives(query, duplicate)

        nonfinite = [_row("bad", split="test", metadata={"similarity": "nan"})]
        with self.assertRaisesRegex(RetrievalEvalError, "non-finite similarity"):
            sample_hard_negatives(query, nonfinite)

    def test_sampler_report_aggregates_composition_and_round_trips(self) -> None:
        query = _row(
            "target",
            split="test",
            source="synthetic",
            edit_size=2,
            metadata={"action_cluster": "rename"},
        )
        candidates = [
            query,
            _row(
                "near",
                split="test",
                source="synthetic",
                edit_size=3,
                metadata={"action_cluster": "rename", "similarity": 0.7},
            ),
            _row("fallback", split="val", source="local_repo", edit_size=99),
        ]
        config = HardNegativeSamplerConfig(max_negatives=2, seed=11, edit_size_bucket_width=10)
        sample = sample_hard_negatives(query, candidates, config=config)

        report = build_hard_negative_sampler_report([sample], config=config)
        payload = report.to_dict()
        loaded = HardNegativeSamplerReport.from_dict(payload)

        self.assertEqual(payload["sample_count"], 1)
        self.assertEqual(payload["returned_negatives"], 2)
        self.assertEqual(payload["composition"]["same_action_cluster"], 1)
        self.assertEqual(loaded.to_dict(), payload)

    def test_sampler_reports_action_discriminative_hard_negative_features(self) -> None:
        query = _row(
            "target",
            split="test",
            metadata={
                "state_before_hash": "before-a",
                "state_after_hash": "after-target",
                "state_before_simhash": "0000000000000000",
                "action_cluster": "guard",
            },
        )
        candidates = [
            query,
            _row(
                "same-before",
                split="test",
                metadata={
                    "state_before_hash": "before-a",
                    "state_after_hash": "after-other",
                    "state_before_simhash": "0000000000000000",
                    "action_cluster": "guard",
                },
            ),
            _row(
                "near-before",
                split="test",
                path="pkg/other.py",
                metadata={
                    "state_before_hash": "before-b",
                    "state_after_hash": "after-third",
                    "state_before_simhash": "0000000000000001",
                    "action_cluster": "guard",
                },
            ),
        ]

        sample = sample_hard_negatives(
            query,
            candidates,
            config=HardNegativeSamplerConfig(
                max_negatives=2,
                seed=13,
                near_before_hamming_threshold=1,
            ),
        )

        self.assertEqual(sample.negative_ids, ("same-before", "near-before"))
        self.assertEqual(sample.composition["same_before_different_after"], 1)
        self.assertEqual(sample.composition["near_before_different_after"], 1)
        self.assertEqual(sample.composition["same_file"], 1)
        self.assertEqual(sample.composition["action_discriminative"], 2)


def _row(
    transition_id: str,
    *,
    split: str,
    source: str = "synthetic",
    edit_size: int = 1,
    path: str = "pkg/mod.py",
    metadata: dict[str, object] | None = None,
) -> CandidatePoolEntry:
    return CandidatePoolEntry(
        transition_id=transition_id,
        split=split,
        source=source,
        repo="example/repo",
        path=path,
        edit_size=edit_size,
        metadata={} if metadata is None else metadata,
    )


if __name__ == "__main__":
    unittest.main()
