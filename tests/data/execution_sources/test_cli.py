"""CLI integration test for ``codelewm dataset ingest``."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from codelewm.harness.cli import build_parser


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class DatasetIngestCLITest(unittest.TestCase):
    def test_mbpp_ingest_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mbpp.jsonl"
            parser = build_parser()
            namespace = parser.parse_args(
                [
                    "dataset",
                    "ingest",
                    "--source",
                    "mbpp",
                    "--input",
                    str(FIXTURES / "mbpp_tiny.jsonl"),
                    "--output",
                    str(out),
                    "--json",
                ]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = namespace.func(namespace)
            self.assertEqual(exit_code, 0)
            summary = json.loads(buf.getvalue())
            self.assertEqual(summary["source"], "mbpp")
            self.assertGreaterEqual(int(summary["submission_count"]), 1)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), int(summary["submission_count"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
