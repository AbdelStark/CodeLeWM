from __future__ import annotations

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
    error_report_json_schema,
    rerank_result_json_schema,
    score_result_json_schema,
    validate_error_report_payload,
)


ROOT = Path(__file__).resolve().parents[2]


class HarnessOutputSchemaTest(unittest.TestCase):
    def test_score_error_and_rerank_schemas_expose_required_fields(self) -> None:
        score_schema = score_result_json_schema()
        error_schema = error_report_json_schema()
        rerank_schema = rerank_result_json_schema()

        self.assertEqual(score_schema["properties"]["schema_version"]["const"], SCORE_RESULT_SCHEMA_VERSION)
        self.assertIn("checkpoint_sha256", score_schema["required"])
        self.assertEqual(error_schema["properties"]["schema_version"]["const"], ERROR_REPORT_SCHEMA_VERSION)
        self.assertIn("invalid_syntax", error_schema["properties"]["error_type"]["enum"])
        self.assertEqual(rerank_schema["properties"]["schema_version"]["const"], RERANK_RESULT_SCHEMA_VERSION)
        self.assertIn("results", rerank_schema["required"])

    def test_error_report_round_trips_json_native_payload(self) -> None:
        report = ErrorReport(
            error_type="patch_apply_failed",
            message="candidate patch did not apply",
            remediation="provide a clean candidate patch",
            record_id="candidate-1",
            artifact="candidate.patch",
            caused_by="dry-run patch failed",
        )

        loaded = validate_error_report_payload(report.to_dict())

        self.assertEqual(loaded.to_dict(), report.to_dict())

    def test_score_cli_returns_json_error_for_invalid_candidate_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            candidate = root / "after.py"
            checkpoint = root / "checkpoint.bin"
            before.write_text("value = 1\n")
            candidate.write_text("def broken(:\n    return 1\n")
            checkpoint.write_bytes(b"fixture checkpoint")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "score",
                    "--before",
                    str(before),
                    "--instruction",
                    "increment value",
                    "--candidate",
                    str(candidate),
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

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        report = validate_error_report_payload(payload)
        self.assertEqual(report.schema_version, ERROR_REPORT_SCHEMA_VERSION)
        self.assertEqual(report.error_type, "invalid_syntax")
        self.assertEqual(report.artifact, str(candidate))
        self.assertNotIn("def broken", report.message)
        self.assertNotIn("return 1", report.caused_by or "")


if __name__ == "__main__":
    unittest.main()
