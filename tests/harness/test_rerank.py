from __future__ import annotations

import difflib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.harness import (
    ERROR_REPORT_SCHEMA_VERSION,
    RERANK_RESULT_SCHEMA_VERSION,
    SCORE_RESULT_SCHEMA_VERSION,
    ErrorReport,
    ScoreResult,
    load_scorer,
    validate_rerank_result_payload,
)


ROOT = Path(__file__).resolve().parents[2]


class RerankHarnessTest(unittest.TestCase):
    def test_rerank_scores_after_files_and_patches_with_errors_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            checkpoint = root / "checkpoint.bin"
            candidates = root / "candidates"
            candidates.mkdir()
            marker = root / "marker.txt"
            before_text = "def value():\n    return 1\n"
            patch_after = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
                "def value():\n"
                "    return 2\n"
            )
            after_file = candidates / "after_b.py"
            patch_file = candidates / "after_a.patch"
            invalid_file = candidates / "after_c.py"
            before.write_text(before_text)
            checkpoint.write_bytes(b"fixture checkpoint")
            after_file.write_text("def value():\n    return 3\n")
            patch_file.write_text(_unified_diff(before_text, patch_after))
            invalid_file.write_text("def broken(:\n    return 4\n")

            result = load_scorer(checkpoint).rerank_files(
                before=before,
                instruction="change return value",
                candidates=candidates,
            )

            score_results = [item for item in result.results if isinstance(item, ScoreResult)]
            error_reports = [item for item in result.results if isinstance(item, ErrorReport)]
            self.assertEqual(result.schema_version, RERANK_RESULT_SCHEMA_VERSION)
            self.assertEqual(len(score_results), 2)
            self.assertEqual(len(error_reports), 1)
            self.assertEqual(result.results[-1].schema_version, ERROR_REPORT_SCHEMA_VERSION)
            self.assertEqual(error_reports[0].error_type, "invalid_syntax")
            self.assertEqual(error_reports[0].artifact, str(invalid_file))
            self.assertEqual(
                [item.final_score for item in score_results],
                sorted(item.final_score for item in score_results),
            )
            self.assertEqual(before.read_text(), before_text)
            self.assertFalse(marker.exists())

    def test_rerank_reports_patch_apply_failure_after_valid_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            checkpoint = root / "checkpoint.bin"
            candidates = root / "candidates"
            candidates.mkdir()
            valid = candidates / "after.py"
            bad_patch = candidates / "bad.patch"
            before.write_text("value = 1\n")
            checkpoint.write_bytes(b"fixture checkpoint")
            valid.write_text("value = 2\n")
            bad_patch.write_text(
                "--- before.py\n"
                "+++ after.py\n"
                "@@ -1 +1 @@\n"
                "-missing = 1\n"
                "+value = 3\n"
            )

            result = load_scorer(checkpoint).rerank_files(
                before=before,
                instruction="change value",
                candidates=candidates,
            )

            self.assertIsInstance(result.results[0], ScoreResult)
            self.assertIsInstance(result.results[-1], ErrorReport)
            self.assertEqual(result.results[-1].error_type, "patch_apply_failed")
            self.assertEqual(result.results[-1].artifact, str(bad_patch))

    def test_rerank_cli_emits_schema_versioned_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            instruction = root / "instruction.txt"
            checkpoint = root / "checkpoint.bin"
            candidates = root / "candidates"
            candidates.mkdir()
            before.write_text("value = 1\n")
            instruction.write_text("increment value")
            checkpoint.write_bytes(b"fixture checkpoint")
            (candidates / "good.py").write_text("value = 2\n")
            (candidates / "bad.py").write_text("def broken(:\n    return 1\n")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "rerank",
                    "--before",
                    str(before),
                    "--instruction",
                    str(instruction),
                    "--candidates",
                    str(candidates),
                    "--checkpoint",
                    str(checkpoint),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        result = validate_rerank_result_payload(payload)
        self.assertEqual(result.schema_version, RERANK_RESULT_SCHEMA_VERSION)
        self.assertEqual(result.results[0].schema_version, SCORE_RESULT_SCHEMA_VERSION)
        self.assertEqual(result.results[-1].schema_version, ERROR_REPORT_SCHEMA_VERSION)


def _unified_diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before.py",
            tofile="after.py",
        )
    )


if __name__ == "__main__":
    unittest.main()
