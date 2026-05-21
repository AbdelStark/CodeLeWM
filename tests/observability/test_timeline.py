from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.observability import (
    RUN_TIMELINE_SCHEMA_VERSION,
    RunTimelineError,
    RunTimelineRecorder,
    read_run_timeline_report,
    validate_run_timeline_report_payload,
    write_run_timeline_report,
)


class RunTimelineReportTest(unittest.TestCase):
    def test_recorder_writes_success_timeline_with_redacted_metadata(self) -> None:
        recorder = RunTimelineRecorder(
            run_id="run-fixture",
            command=("codelewm", "llm-demo", "--api-key", "sk-fixture-secret-token-1234567890"),
        )
        with recorder.step("candidate generation", command_id="llm_demo.candidate_generation") as step:
            step.add_artifact("candidate_pack-1")
        with recorder.step(
            "world model scoring",
            command_id="llm_demo.world_model_scoring",
            metadata={"prompt": "line\n" * 25, "token": "sk-fixture-secret-token-1234567890"},
        ):
            pass

        report = recorder.to_report(artifact_ids=("demo_report-1",))

        self.assertEqual(report.schema_version, RUN_TIMELINE_SCHEMA_VERSION)
        self.assertEqual(report.status, "completed")
        self.assertEqual([step.order for step in report.steps], [1, 2])
        self.assertEqual(report.steps[0].artifact_ids, ("candidate_pack-1",))
        payload = report.to_dict()
        self.assertIn("[REDACTED_SECRET]", payload["command"][3])
        self.assertEqual(payload["steps"][1]["metadata"]["token"], "[REDACTED_SECRET]")
        self.assertIn("[REDACTED_LONG_TEXT", payload["steps"][1]["metadata"]["prompt"])
        json.dumps(payload, allow_nan=False, sort_keys=True)

    def test_recorder_captures_typed_failure(self) -> None:
        recorder = RunTimelineRecorder(run_id="run-fail", command=("codelewm", "eval", "latent-matrix"))

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            with recorder.step("candidate generation", command_id="llm_demo.candidate_generation"):
                raise RuntimeError("provider failed with sk-fixture-secret-token-1234567890")

        report = recorder.to_report(status="failed")
        payload = report.to_dict()

        self.assertEqual(report.status, "failed")
        self.assertEqual(payload["steps"][0]["status"], "failed")
        self.assertEqual(payload["steps"][0]["typed_failure"]["error_type"], "RuntimeError")
        self.assertNotIn("sk-fixture", payload["steps"][0]["typed_failure"]["message"])

    def test_report_round_trips_json(self) -> None:
        recorder = RunTimelineRecorder(run_id="run-roundtrip", command=("codelewm", "train"))
        with recorder.step("train", command_id="train"):
            pass
        report = recorder.to_report()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_timeline.json"
            write_run_timeline_report(report, path)
            loaded = read_run_timeline_report(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], RUN_TIMELINE_SCHEMA_VERSION)
        self.assertEqual(loaded.to_dict(), report.to_dict())

    def test_failed_report_requires_typed_failure(self) -> None:
        recorder = RunTimelineRecorder(run_id="bad", command=("codelewm",))
        report = recorder.to_report()
        payload = report.to_dict()
        payload["status"] = "failed"

        with self.assertRaisesRegex(RunTimelineError, "typed failure"):
            validate_run_timeline_report_payload(payload)


if __name__ == "__main__":
    unittest.main()
