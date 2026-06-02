"""Tests for content dedup in load_execution_source (RFC-0015 WS-B3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.data.execution_sources import load_execution_source


_TASK_A = {
    "task_id": 1,
    "text": "Square a number",
    "code": "def square(n):\n    return n * n\n",
    "test_list": ["assert square(2) == 4", "assert square(0) == 0"],
    "test_setup_code": "",
    "challenge_test_list": [],
}
_TASK_B = {
    "task_id": 2,
    "text": "Sum of a list",
    "code": "def total(xs):\n    return sum(xs)\n",
    "test_list": ["assert total([1, 2, 3]) == 6", "assert total([]) == 0"],
    "test_setup_code": "",
    "challenge_test_list": [],
}


def _write_source(path: Path, tasks: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8"
    )


class LoadExecutionSourceDedupTest(unittest.TestCase):
    def test_dedup_skips_byte_identical_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Task A appears twice (byte-identical) plus a distinct task B.
            src = root / "mbpp_dups.jsonl"
            _write_source(src, [_TASK_A, _TASK_B, dict(_TASK_A)])

            raw = load_execution_source(
                source="mbpp", source_path=src, output_path=root / "raw.jsonl"
            )
            deduped = load_execution_source(
                source="mbpp",
                source_path=src,
                output_path=root / "deduped.jsonl",
                deduplicate=True,
            )

            self.assertFalse(raw["deduplicate"])
            self.assertEqual(raw["duplicate_skipped_count"], 0)
            self.assertTrue(deduped["deduplicate"])
            # The duplicated task's submissions are dropped exactly once each.
            self.assertGreaterEqual(deduped["duplicate_skipped_count"], 1)
            self.assertEqual(
                deduped["submission_count"],
                raw["submission_count"] - deduped["duplicate_skipped_count"],
            )
            # No raw_hash appears twice in the deduped output.
            hashes = [
                json.loads(line)["raw_hash"]
                for line in (root / "deduped.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(hashes), len(set(hashes)))

    def test_default_is_no_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "mbpp.jsonl"
            _write_source(src, [_TASK_A, dict(_TASK_A)])
            summary = load_execution_source(
                source="mbpp", source_path=src, output_path=root / "raw.jsonl"
            )
            # Default keeps duplicates (no behaviour change for existing builds).
            self.assertEqual(summary["duplicate_skipped_count"], 0)
            self.assertFalse(summary["deduplicate"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
