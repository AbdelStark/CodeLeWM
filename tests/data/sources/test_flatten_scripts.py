"""Tests for the dataset-flattener scripts.

The scripts download from Hugging Face by default; this test exercises
the path that does not touch the network — feeding the post-flatten
JSONL into the existing adapter and asserting the round-trip works.
A separate live-download CI lane is documented in the pack-build
report and is not run on every PR.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FLATTEN_CODENET = REPO_ROOT / "scripts" / "dataset" / "flatten-codenet"


class FlattenCodeNetCLITest(unittest.TestCase):
    def test_codenet_flattener_writes_jsonl_against_local_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Build a minimal CodeNet-shape tree.
            problem_dir = root / "data" / "p00001" / "Python"
            problem_dir.mkdir(parents=True)
            (problem_dir / "s001.py").write_text(
                "n = int(input())\nprint(n * n)\n", encoding="utf-8"
            )
            metadata_dir = root / "metadata"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "p00001.csv").write_text(
                "submission_id,status\ns001,Accepted\n",
                encoding="utf-8",
            )
            io_dir = root / "derived" / "input_output" / "data" / "p00001"
            io_dir.mkdir(parents=True)
            (io_dir / "input.txt").write_text("3\n", encoding="utf-8")
            (io_dir / "output.txt").write_text("9\n", encoding="utf-8")

            out = root / "codenet.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(FLATTEN_CODENET),
                    "--root",
                    str(root),
                    "--output",
                    str(out),
                ],
                env={
                    "PYTHONPATH": str(REPO_ROOT),
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode, 0,
                msg=f"stderr={completed.stderr!r}",
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["rows_written"], 1)

            # Round-trip through the existing CodeNet adapter.
            from codelewm.data.execution_sources import (
                get_execution_source_adapter,
            )

            adapter = get_execution_source_adapter("codenet")
            records = list(adapter.iter_submissions(source_path=out))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].source_problem_id, "p00001")
            self.assertEqual(records[0].inputs[0].input_kind, "stdin")
            self.assertEqual(records[0].judge_verdict, "accepted")

    def test_codenet_flattener_filters_non_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            problem_dir = root / "data" / "p00002" / "Python"
            problem_dir.mkdir(parents=True)
            (problem_dir / "s001.py").write_text("print('x')", encoding="utf-8")
            metadata_dir = root / "metadata"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "p00002.csv").write_text(
                "submission_id,status\ns001,Wrong Answer\n",
                encoding="utf-8",
            )
            io_dir = root / "derived" / "input_output" / "data" / "p00002"
            io_dir.mkdir(parents=True)
            (io_dir / "input.txt").write_text("3\n", encoding="utf-8")
            (io_dir / "output.txt").write_text("9\n", encoding="utf-8")

            out = root / "codenet.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(FLATTEN_CODENET),
                    "--root",
                    str(root),
                    "--output",
                    str(out),
                    "--keep-verdicts",
                    "accepted",
                ],
                env={
                    "PYTHONPATH": str(REPO_ROOT),
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            # Exit code 1 because no rows survive the filter, but the
            # script wrote a summary to stdout that the test can inspect.
            self.assertEqual(completed.returncode, 1)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["rows_written"], 0)
            self.assertGreaterEqual(
                payload["rows_skipped_by_reason"].get("verdict_filtered", 0), 1
            )


class MBPPTupleAssertionResilienceTest(unittest.TestCase):
    """The MBPP adapter must skip rather than abort on JSON-incompatible literals."""

    def test_tuple_keyed_dict_assertion_does_not_abort_stream(self) -> None:
        from codelewm.data.execution_sources import (
            get_execution_source_adapter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = Path(tmpdir) / "mbpp.jsonl"
            rows = [
                # Row 0: tuple-keyed dict in assertion - must be skipped.
                {
                    "task_id": 1000,
                    "text": "tuple key",
                    "code": "def f(d):\n    return d\n",
                    "test_list": ["assert f({(1, 2): 3}) == {(1, 2): 3}"],
                    "test_setup_code": "",
                    "challenge_test_list": [],
                },
                # Row 1: simple square - must round-trip.
                {
                    "task_id": 1001,
                    "text": "square",
                    "code": "def square(n):\n    return n * n\n",
                    "test_list": ["assert square(2) == 4"],
                    "test_setup_code": "",
                    "challenge_test_list": [],
                },
            ]
            fixture.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )
            adapter = get_execution_source_adapter("mbpp")
            records = list(adapter.iter_submissions(source_path=fixture))
            ids = [r.source_problem_id for r in records]
            # Row 0 may either be skipped entirely (no parseable assertion)
            # or kept with the tuple-keyed assertion skipped. Either way,
            # row 1 must survive.
            self.assertIn("mbpp/1001", ids)

    def test_tuple_arg_serializes_via_default(self) -> None:
        from codelewm.data.execution_sources import (
            get_execution_source_adapter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = Path(tmpdir) / "mbpp.jsonl"
            rows = [
                # A tuple as a positional arg is JSON-encodable via
                # default= coercion to list.
                {
                    "task_id": 1010,
                    "text": "tuple-arg",
                    "code": "def first(pair):\n    return pair[0]\n",
                    "test_list": ["assert first((1, 2)) == 1"],
                    "test_setup_code": "",
                    "challenge_test_list": [],
                },
            ]
            fixture.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )
            adapter = get_execution_source_adapter("mbpp")
            records = list(adapter.iter_submissions(source_path=fixture))
            self.assertEqual(len(records), 1)
            # The tuple (1, 2) coerces to [1, 2] in the JSON input_repr.
            self.assertEqual(
                json.loads(records[0].inputs[0].input_repr), [[1, 2]]
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
