from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    ACTION_CONTRAST_POOL_REPORT_SCHEMA_VERSION,
    ActionContrastPoolConfig,
    CandidatePoolEntry,
    RetrievalEvalError,
    build_action_contrast_pool_report,
    read_action_contrast_pool_report,
    write_action_contrast_pool_report,
)


class ActionContrastPoolReportTest(unittest.TestCase):
    def test_report_builds_deterministic_pools_and_records_leakage_proofs(self) -> None:
        rows = _fixture_rows()
        config = ActionContrastPoolConfig(max_queries=2, max_candidates_per_pool=3, seed=17)

        report = build_action_contrast_pool_report(
            rows,
            query_ids=("same-before-a", "same-before-b"),
            config=config,
        )
        repeat = build_action_contrast_pool_report(
            reversed(rows),
            query_ids=("same-before-a", "same-before-b"),
            config=config,
        )

        self.assertEqual(report.to_dict(), repeat.to_dict())
        self.assertEqual(report.schema_version, ACTION_CONTRAST_POOL_REPORT_SCHEMA_VERSION)
        self.assertEqual(report.query_count, 2)
        self.assertEqual(report.leakage["input_train_rows"], 1)
        self.assertEqual(report.leakage["selected_train_rows"], 0)
        self.assertFalse(report.leakage["leakage_detected"])
        self.assertIn("exact_same_before", report.available_pools)
        self.assertIn("near_before", report.available_pools)
        self.assertIn("same_file", report.available_pools)
        self.assertIn("action_cluster", report.available_pools)
        self.assertIn("edit_shape", report.available_pools)
        self.assertIn("mutation", report.available_pools)
        self.assertIn("random", report.available_pools)

        first = report.samples[0]
        self.assertEqual(first.query_id, "same-before-a")
        self.assertEqual(first.pools["exact_same_before"], ("same-before-b",))
        self.assertNotIn("train-leak", first.pools["exact_same_before"])
        self.assertTrue(first.no_action_challenge)
        self.assertTrue(report.no_action_challenge["no_action_prior_insufficient"])
        self.assertEqual(report.no_action_challenge["same_before_multi_action_query_count"], 2)
        self.assertEqual(report.no_action_challenge["synthetic_controlled_same_before_query_count"], 2)
        self.assertEqual(
            report.split_membership_proofs["exact_same_before"]["split_counts"],
            {"test": 1, "val": 1},
        )

    def test_report_round_trips_json(self) -> None:
        report = build_action_contrast_pool_report(
            _fixture_rows(),
            query_ids=("same-before-a", "same-before-b"),
            config=ActionContrastPoolConfig(max_queries=2, max_candidates_per_pool=2, seed=3),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "action_contrast_pool_report.json"
            write_action_contrast_pool_report(report, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = read_action_contrast_pool_report(path)

        self.assertEqual(payload["schema_version"], ACTION_CONTRAST_POOL_REPORT_SCHEMA_VERSION)
        self.assertEqual(loaded.to_dict(), report.to_dict())

    def test_report_rejects_train_query(self) -> None:
        with self.assertRaisesRegex(RetrievalEvalError, "must be held out"):
            build_action_contrast_pool_report(
                _fixture_rows(),
                query_ids=("train-leak",),
                config=ActionContrastPoolConfig(max_queries=1),
            )

    def test_report_records_unavailable_pools_explicitly(self) -> None:
        rows = [
            _row("target", split="test", before="before-a", after="after-a"),
        ]

        report = build_action_contrast_pool_report(rows, query_ids=("target",))

        self.assertEqual(set(report.unavailable_pools), set(report.config.pool_names))
        self.assertEqual(report.pool_counts["exact_same_before"], 0)
        self.assertFalse(report.no_action_challenge["no_action_prior_insufficient"])


def _fixture_rows() -> list[CandidatePoolEntry]:
    return [
        _row(
            "same-before-a",
            split="val",
            source="synthetic",
            before="before-shared",
            after="after-a",
            simhash="0000000000000000",
            action_cluster="rename",
            edit_shape="shape-a",
        ),
        _row(
            "same-before-b",
            split="test",
            source="synthetic",
            before="before-shared",
            after="after-b",
            simhash="0000000000000000",
            action_cluster="return-none",
            edit_shape="shape-a",
        ),
        _row(
            "near-before",
            split="test",
            before="before-near",
            after="after-c",
            simhash="0000000000000001",
            action_cluster="rename",
            edit_shape="shape-b",
        ),
        _row(
            "same-file",
            split="test",
            before="before-file",
            after="after-d",
            path="pkg/shared.py",
            action_cluster="guard",
            edit_shape="shape-c",
        ),
        _row(
            "same-action",
            split="val",
            before="before-action",
            after="after-e",
            path="pkg/action.py",
            action_cluster="rename",
            edit_shape="shape-d",
        ),
        _row(
            "same-shape",
            split="test",
            before="before-shape",
            after="after-f",
            path="pkg/shape.py",
            action_cluster="shape-other",
            edit_shape="shape-a",
        ),
        _row(
            "mutation",
            split="test",
            before="before-mut",
            after="after-g",
            path="pkg/mut.py",
            action_cluster="mutation",
            edit_shape="shape-m",
            candidate_kind="mutation",
        ),
        _row(
            "train-leak",
            split="train",
            before="before-shared",
            after="after-train",
            simhash="0000000000000000",
            action_cluster="train",
            edit_shape="shape-a",
        ),
    ]


def _row(
    transition_id: str,
    *,
    split: str,
    before: str,
    after: str,
    source: str = "local_repo",
    repo: str = "example/repo",
    path: str = "pkg/shared.py",
    edit_size: int = 4,
    simhash: str = "000000000000000f",
    action_cluster: str = "edit",
    edit_shape: str = "shape-default",
    candidate_kind: str | None = None,
) -> CandidatePoolEntry:
    metadata: dict[str, object] = {
        "state_before_hash": before,
        "state_after_hash": after,
        "state_before_simhash": simhash,
        "action_cluster": action_cluster,
        "diff_shape": edit_shape,
    }
    if candidate_kind is not None:
        metadata["candidate_kind"] = candidate_kind
    return CandidatePoolEntry(
        transition_id=transition_id,
        split=split,
        source=source,
        repo=repo,
        path=path,
        edit_size=edit_size,
        metadata=metadata,
    )


if __name__ == "__main__":
    unittest.main()
