from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.data import (
    FixtureSourceAdapter,
    RawEditRecord,
    SourceRecordError,
    SourceSpec,
    SourceUnavailableError,
    get_source_adapter,
    load_source,
)


def _record(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "local_repo",
        "repo": "example/repo",
        "commit": "abc123",
        "path_before": "pkg/mod.py",
        "path_after": "pkg/mod.py",
        "before": "def f():\n    return 1\n",
        "after": "def f():\n    return 2\n",
        "message": "update return value",
        "license": "MIT",
        "timestamp": "2026-05-18T00:00:00Z",
        "metadata": {"fixture": True},
    }
    row.update(overrides)
    return row


class SourceAdapterTest(unittest.TestCase):
    def test_fixture_adapter_loads_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "records.jsonl"
            rows = [_record(commit="a"), _record(commit="b", metadata={"index": 2})]
            source_path.write_text("\n".join(json.dumps(row) for row in rows))

            records = list(load_source(SourceSpec(source="fixture", path=source_path)))

        self.assertEqual(len(records), 2)
        self.assertTrue(all(isinstance(record, RawEditRecord) for record in records))
        self.assertEqual(records[0].source, "local_repo")
        self.assertEqual(records[1].metadata["index"], 2)

    def test_fixture_adapter_supports_json_list_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "records.json"
            source_path.write_text(json.dumps([_record(source="synthetic")]))

            records = list(load_source(SourceSpec(source="fixture", path=source_path)))

        self.assertEqual(records[0].source, "synthetic")

    def test_fixture_adapter_can_default_record_source(self) -> None:
        row = _record()
        row.pop("source")
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "records.jsonl"
            source_path.write_text(json.dumps(row))

            records = list(
                load_source(
                    SourceSpec(
                        source="fixture",
                        path=source_path,
                        options={"record_source": "commitpackft"},
                    )
                )
            )

        self.assertEqual(records[0].source, "commitpackft")

    def test_missing_fixture_path_raises_source_unavailable(self) -> None:
        adapter = FixtureSourceAdapter()

        with self.assertRaisesRegex(SourceUnavailableError, "requires a path"):
            list(adapter.iter_records(SourceSpec(source="fixture")))

        with self.assertRaisesRegex(SourceUnavailableError, "does not exist"):
            list(adapter.iter_records(SourceSpec(source="fixture", path=Path("missing.jsonl"))))

    def test_unknown_adapter_raises_source_unavailable(self) -> None:
        with self.assertRaisesRegex(SourceUnavailableError, "No source adapter"):
            get_source_adapter("commitpackft")

    def test_invalid_record_reports_schema_error(self) -> None:
        row = _record()
        row.pop("before")

        with self.assertRaisesRegex(SourceRecordError, "before"):
            RawEditRecord.from_mapping(row)

    def test_metadata_must_be_mapping(self) -> None:
        row = _record(metadata=["not", "a", "mapping"])

        with self.assertRaisesRegex(SourceRecordError, "metadata"):
            RawEditRecord.from_mapping(row)


if __name__ == "__main__":
    unittest.main()
