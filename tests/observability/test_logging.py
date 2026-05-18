from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.observability import (
    LOG_EVENT_SCHEMA_VERSION,
    LogEvent,
    log_event_json_schema,
    redact_text,
    redact_value,
    validate_log_event_payload,
    write_log_event_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]


class StructuredLoggingTest(unittest.TestCase):
    def test_log_event_round_trips_json_native_redacted_payload(self) -> None:
        home_secret_path = Path.home() / "private" / "repo.py"
        long_source = "\n".join(f"line_{index} = {index}" for index in range(25))
        event = LogEvent(
            event="dataset.build.start",
            level="info",
            run_id="run-1",
            artifact_id="dataset-1",
            step="load",
            message=f"reading {home_secret_path}",
            fields={
                "api_token": "super-secret-token",
                "path": str(home_secret_path),
                "source_snippet": long_source,
                "nested": {"password": "not-for-logs"},
            },
        )

        payload = event.to_dict()
        loaded = validate_log_event_payload(payload)

        self.assertEqual(payload["schema_version"], LOG_EVENT_SCHEMA_VERSION)
        self.assertEqual(loaded.to_dict(), payload)
        self.assertEqual(payload["fields"]["api_token"], "[REDACTED_SECRET]")
        self.assertEqual(payload["fields"]["nested"]["password"], "[REDACTED_SECRET]")
        self.assertNotIn(str(Path.home()), json.dumps(payload, sort_keys=True))
        self.assertIn("[REDACTED_LONG_TEXT", payload["fields"]["source_snippet"])

    def test_jsonl_writer_appends_valid_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            first = LogEvent(event="score.start", level="info", run_id="run-1", step="score", message="start")
            second = LogEvent(
                event="score.complete",
                level="info",
                run_id="run-1",
                step="score",
                message="complete",
                fields={"final_score": 0.25},
            )

            write_log_event_jsonl(first, path)
            write_log_event_jsonl(second, path)

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 2)
        self.assertEqual(validate_log_event_payload(rows[0]).event, "score.start")
        self.assertEqual(validate_log_event_payload(rows[1]).fields["final_score"], 0.25)

    def test_score_cli_writes_jsonl_logs_without_raw_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            candidate = root / "after.py"
            checkpoint = root / "checkpoint.bin"
            log_path = root / "logs" / "events.jsonl"
            before.write_text("value = 1\n")
            candidate.write_text("value = 2\n")
            checkpoint.write_bytes(b"fixture checkpoint")
            raw_instruction = "increment value with password=do-not-log"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "score",
                    "--before",
                    str(before),
                    "--instruction",
                    raw_instruction,
                    "--candidate",
                    str(candidate),
                    "--checkpoint",
                    str(checkpoint),
                    "--json",
                    "--log-jsonl",
                    str(log_path),
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            log_text = log_path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in log_text.splitlines()]

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual([row["event"] for row in rows], ["harness.score.start", "harness.score.complete"])
        self.assertNotIn("do-not-log", log_text)
        self.assertIn("instruction_sha256", rows[0]["fields"])

    def test_redaction_helpers_cover_secret_values_and_schema(self) -> None:
        schema = log_event_json_schema()

        self.assertEqual(schema["properties"]["schema_version"]["const"], LOG_EVENT_SCHEMA_VERSION)
        self.assertEqual(redact_text("token sk-1234567890abcdef"), "token [REDACTED_SECRET]")
        self.assertEqual(redact_value({"github_token": "ghp_1234567890abcdef"})["github_token"], "[REDACTED_SECRET]")


if __name__ == "__main__":
    unittest.main()
