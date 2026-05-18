from __future__ import annotations

import unittest

from codelewm.data import (
    DedupPolicy,
    RawEditRecord,
    SplitPolicy,
    assign_split,
    compute_dedup_keys,
    hamming_distance_hex,
    split_and_deduplicate,
    split_key,
)


def _record(**overrides: object) -> RawEditRecord:
    values: dict[str, object] = {
        "source": "commitpackft",
        "repo": "Example/Repo",
        "commit": "abc123",
        "path_before": "pkg/mod.py",
        "path_after": "pkg/mod.py",
        "before": "def f(value):\n    return value + 1\n",
        "after": "def f(value):\n    return value + 2\n",
        "message": "change return value",
        "license": "mit",
        "metadata": {},
    }
    values.update(overrides)
    return RawEditRecord(**values)  # type: ignore[arg-type]


class SplitDedupTest(unittest.TestCase):
    def test_split_key_normalizes_repository_names(self) -> None:
        self.assertEqual(split_key(_record(repo="  Example/Repo  ")), "example/repo")

    def test_same_repo_gets_same_deterministic_split(self) -> None:
        first = _record(commit="a", repo="Example/Repo")
        second = _record(commit="b", repo="example/repo")

        self.assertEqual(assign_split(first), assign_split(second))

    def test_split_overrides_make_fixture_splits_explicit(self) -> None:
        policy = SplitPolicy(split_overrides={"example/repo": "val"})

        split, key = assign_split(_record(), policy=policy)

        self.assertEqual(split, "val")
        self.assertEqual(key, "example/repo")

    def test_synthetic_records_inherit_source_split_metadata(self) -> None:
        record = _record(source="synthetic", metadata={"source_split": "test"})

        split, _ = assign_split(record)

        self.assertEqual(split, "test")

    def test_dedup_keys_are_stable_and_json_native(self) -> None:
        record = _record()
        first = compute_dedup_keys(record)
        second = compute_dedup_keys(record)

        self.assertEqual(first, second)
        self.assertEqual(len(first.exact_transition), 64)
        self.assertEqual(len(first.exact_before_after), 64)
        self.assertEqual(len(first.near_state), 16)
        self.assertEqual(len(first.diff_shape), 64)
        self.assertEqual(hamming_distance_hex(first.near_state, second.near_state), 0)
        self.assertEqual(first.to_dict()["near_state"], first.near_state)

    def test_exact_duplicate_is_dropped_with_machine_readable_reason(self) -> None:
        first = _record(commit="a")
        duplicate = _record(commit="b")

        result = split_and_deduplicate([first, duplicate])

        self.assertEqual(len(result.kept), 1)
        self.assertEqual(result.report.drop_reasons, {"exact_duplicate": 1})
        self.assertEqual(result.dropped[0].reason, "exact_duplicate")
        self.assertEqual(result.dropped[0].details["match_type"], "exact_transition")

    def test_validation_exact_leakage_against_train_is_rejected(self) -> None:
        train = _record(repo="train/repo", commit="train")
        validation = _record(repo="val/repo", commit="val")
        policy = SplitPolicy(split_overrides={"train/repo": "train", "val/repo": "val"})

        result = split_and_deduplicate([validation, train], split_policy=policy)

        self.assertEqual([assignment.split for assignment in result.kept], ["train"])
        self.assertEqual(result.dropped[0].reason, "train_leakage")
        self.assertEqual(result.dropped[0].details["match_type"], "exact_before_after")
        self.assertEqual(result.report.per_split, {"train": 1, "val": 0, "test": 0})

    def test_test_near_duplicate_against_train_is_rejected(self) -> None:
        train = _record(repo="train/repo", commit="train")
        test = _record(
            repo="test/repo",
            commit="test",
            after="def f(value):\n    return value + 3\n",
        )
        policy = SplitPolicy(split_overrides={"train/repo": "train", "test/repo": "test"})

        result = split_and_deduplicate([train, test], split_policy=policy)

        self.assertEqual(len(result.kept), 1)
        self.assertEqual(result.dropped[0].reason, "train_leakage")
        self.assertEqual(result.dropped[0].details["match_type"], "near_state")
        self.assertEqual(result.report.drop_reasons, {"train_leakage": 1})

    def test_custom_near_duplicate_threshold_can_disable_near_leakage_rejection(self) -> None:
        train = _record(repo="train/repo", commit="train")
        validation = _record(
            repo="val/repo",
            commit="val",
            before="def unrelated():\n    return 'x'\n",
            after="def unrelated():\n    return 'y'\n",
        )
        split_policy = SplitPolicy(split_overrides={"train/repo": "train", "val/repo": "val"})

        result = split_and_deduplicate(
            [train, validation],
            split_policy=split_policy,
            dedup_policy=DedupPolicy(near_duplicate_hamming_threshold=0),
        )

        self.assertEqual(len(result.kept), 2)
        self.assertEqual(result.report.drop_reasons, {})
        self.assertEqual(result.report.per_split, {"train": 1, "val": 1, "test": 0})


if __name__ == "__main__":
    unittest.main()
