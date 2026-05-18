from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from codelewm.data import (
    RawEditRecord,
    SourceRecordError,
    SourceSpec,
    SourceUnavailableError,
    load_source,
)


def _commitpackft_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "commit": "abc123",
        "old_file": "./pkg/mod.py",
        "new_file": "pkg/mod.py",
        "old_contents": "def f():\n    return 1\n",
        "new_contents": "def f():\n    return 2\n",
        "subject": "Change return value",
        "message": "Change return value\n",
        "lang": "Python",
        "license": "MIT",
        "repos": " example/repo ",
    }
    row.update(overrides)
    return row


class CommitPackFTLoaderTest(unittest.TestCase):
    def test_loads_local_jsonl_shard_as_raw_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "data.jsonl"
            shard.write_text(json.dumps(_commitpackft_row()))

            records = list(load_source(SourceSpec(source="commitpackft", path=shard, name="python-smoke")))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, RawEditRecord)
        self.assertEqual(record.source, "commitpackft")
        self.assertEqual(record.repo, "example/repo")
        self.assertEqual(record.commit, "abc123")
        self.assertEqual(record.path_before, "pkg/mod.py")
        self.assertEqual(record.path_after, "pkg/mod.py")
        self.assertEqual(record.before, "def f():\n    return 1\n")
        self.assertEqual(record.after, "def f():\n    return 2\n")
        self.assertEqual(record.message, "Change return value")
        self.assertEqual(record.license, "mit")
        self.assertEqual(record.metadata["language"], "Python")
        self.assertEqual(record.metadata["subject"], "Change return value")
        self.assertEqual(record.metadata["source_name"], "python-smoke")

    def test_loads_sorted_directory_shards_without_json_array_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "python"
            nested.mkdir()
            (nested / "b.jsonl").write_text(
                json.dumps(_commitpackft_row(commit="b", repos=["org/repo", "other/repo"]))
            )
            (nested / "a.jsonl").write_text(json.dumps(_commitpackft_row(commit="a")))

            records = list(load_source(SourceSpec(source="commitpackft", path=root)))

        self.assertEqual([record.commit for record in records], ["a", "b"])
        self.assertEqual(records[1].repo, "org/repo,other/repo")

    def test_loads_gzipped_jsonl_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "data.jsonl.gz"
            with gzip.open(shard, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(_commitpackft_row(commit="gz")) + "\n")

            records = list(load_source(SourceSpec(source="commitpackft", path=shard)))

        self.assertEqual(records[0].commit, "gz")

    def test_missing_required_field_reports_source_record_error(self) -> None:
        row = _commitpackft_row()
        row.pop("old_contents")
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "data.jsonl"
            shard.write_text(json.dumps(row))

            with self.assertRaisesRegex(SourceRecordError, "old_contents"):
                list(load_source(SourceSpec(source="commitpackft", path=shard)))

    def test_non_python_language_is_explicit_error_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "data.jsonl"
            shard.write_text(json.dumps(_commitpackft_row(lang="Java")))

            with self.assertRaisesRegex(SourceRecordError, "expected 'Python'"):
                list(load_source(SourceSpec(source="commitpackft", path=shard)))

    def test_expected_language_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "data.jsonl"
            shard.write_text(json.dumps(_commitpackft_row(lang="python")))

            records = list(
                load_source(
                    SourceSpec(source="commitpackft", path=shard, options={"language": "python"})
                )
            )

        self.assertEqual(records[0].metadata["language"], "python")

    def test_unavailable_source_reports_source_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.jsonl"

            with self.assertRaisesRegex(SourceUnavailableError, "does not exist"):
                list(load_source(SourceSpec(source="commitpackft", path=missing)))

            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(SourceUnavailableError, "contains no"):
                list(load_source(SourceSpec(source="commitpackft", path=empty)))


if __name__ == "__main__":
    unittest.main()
