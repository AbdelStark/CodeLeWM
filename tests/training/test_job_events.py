"""Tests for CodeLeWM HF job event parsing and status rendering."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.training.job_events import (
    JOB_EVENT_PREFIX,
    parse_job_event_lines,
    summarize_job_events,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_SCRIPT = REPO_ROOT / "scripts" / "hf-job-event-status"


def _event(event: str, fields: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "codelewm.log_event.v1",
        "event": event,
        "level": "info",
        "run_id": "codelewm_execution_v0_8_short_a10g",
        "step": str(fields.get("step", 0)),
        "message": event,
        "fields": fields,
    }


class JobEventParserTest(unittest.TestCase):
    def test_parse_live_prefix_and_persisted_jsonl(self) -> None:
        start = _event(
            "execution_training.start",
            {
                "seed": 42,
                "device": "cuda",
                "max_steps": 12000,
                "enable_pass_head": True,
            },
        )
        progress = _event(
            "execution_training.progress",
            {
                "seed": 42,
                "step": 200,
                "max_steps": 12000,
                "progress": 0.016667,
                "elapsed_seconds": 782.0,
                "eta_seconds": 46141.0,
                "metrics": {
                    "loss_total": 1.43,
                    "loss_p_pass_bce": 0.53,
                    "steps_per_second": 0.255,
                },
            },
        )

        lines = [
            "unrelated runtime line",
            JOB_EVENT_PREFIX + json.dumps(start, sort_keys=True),
            json.dumps(progress, sort_keys=True),
        ]

        parsed = parse_job_event_lines(lines)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["event"], "execution_training.start")
        self.assertEqual(parsed[1]["event"], "execution_training.progress")

    def test_summary_reports_latest_progress_and_collapse_gate(self) -> None:
        events = [
            _event("execution_training.start", {"seed": 1729, "max_steps": 12000}),
            _event(
                "runtime.command_start",
                {
                    "command_name": "codelewm",
                    "command_arg_count": 8,
                },
            ),
            _event(
                "execution_training.progress",
                {
                    "seed": 1729,
                    "step": 1000,
                    "max_steps": 12000,
                    "progress": 0.083333,
                    "eta_seconds": 43000.0,
                    "metrics": {
                        "loss_total": 0.58,
                        "loss_p_pass_bce": 0.24,
                    },
                },
            ),
            _event(
                "execution_training.collapse_diagnostics",
                {
                    "seed": 1729,
                    "step": 1000,
                    "diagnostics": {
                        "z_pred_effective_rank_ratio": 0.2901,
                        "z_target_effective_rank_ratio": 0.2545,
                    },
                },
            ),
            _event(
                "execution_training.checkpoint",
                {
                    "seed": 1729,
                    "step": 4000,
                    "checkpoint_name": "checkpoint_step_00004000.pt",
                },
            ),
        ]

        summary = summarize_job_events(
            events,
            job_id="6a-test",
            job_stage="RUNNING",
            collapse_threshold=0.20,
        )

        self.assertEqual(summary["job"]["id"], "6a-test")
        self.assertEqual(summary["latest_progress"]["step"], 1000)
        self.assertEqual(summary["latest_progress"]["remaining_steps"], 11000)
        self.assertTrue(summary["latest_collapse"]["passed"])
        self.assertEqual(
            summary["latest_checkpoint"]["checkpoint_name"],
            "checkpoint_step_00004000.pt",
        )
        self.assertEqual(summary["latest_runtime"]["event"], "runtime.command_start")
        self.assertEqual(
            summary["latest_runtime"]["fields"]["command_name"], "codelewm"
        )
        self.assertEqual(summary["event_counts"]["execution_training.progress"], 1)

    def test_summary_reports_latest_runtime_phase(self) -> None:
        events = [
            _event(
                "runtime.pack_download_start",
                {
                    "pack_repo_id": "abdelstark/codelewm-execution-pack",
                    "pack_revision": "v0.8.0-rc1",
                    "pack_local_dir": "/workspace/pack",
                },
            ),
            _event(
                "runtime.pack_download_complete",
                {
                    "pack_local_dir": "/workspace/pack",
                    "elapsed_seconds": 71,
                },
            ),
        ]

        summary = summarize_job_events(events, job_id="6a-test", job_stage="RUNNING")

        self.assertEqual(
            summary["latest_runtime"]["event"], "runtime.pack_download_complete"
        )
        self.assertEqual(summary["latest_runtime"]["phase"], "pack_download_complete")
        self.assertEqual(summary["latest_runtime"]["fields"]["elapsed_seconds"], 71)


class JobEventStatusScriptTest(unittest.TestCase):
    def test_script_summarizes_saved_progress_jsonl(self) -> None:
        events = [
            _event("execution_training.start", {"seed": 42, "max_steps": 12000}),
            _event(
                "execution_training.progress",
                {
                    "seed": 42,
                    "step": 200,
                    "max_steps": 12000,
                    "progress": 0.016667,
                    "elapsed_seconds": 782.0,
                    "eta_seconds": 46141.0,
                    "metrics": {
                        "loss_total": 1.43,
                        "loss_p_pass_bce": 0.53,
                        "loss_prediction_mse": 0.64,
                    },
                },
            ),
            _event(
                "execution_training.collapse_diagnostics",
                {
                    "seed": 42,
                    "step": 1000,
                    "diagnostics": {"z_pred_effective_rank_ratio": 0.2739},
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job_progress.jsonl"
            path.write_text(
                "\n".join(json.dumps(event, sort_keys=True) for event in events),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(STATUS_SCRIPT),
                    "--from-file",
                    str(path),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        payload = json.loads(completed.stdout)
        summary = payload["summaries"][0]
        self.assertEqual(payload["schema_version"], "codelewm.hf_job_event_status.v1")
        self.assertEqual(summary["latest_progress"]["step"], 200)
        self.assertTrue(summary["latest_collapse"]["passed"])

    def test_script_human_output_is_compact(self) -> None:
        event = _event(
            "execution_training.progress",
            {
                "seed": 42,
                "step": 200,
                "max_steps": 12000,
                "progress": 0.016667,
                "elapsed_seconds": 782.0,
                "eta_seconds": 46141.0,
                "metrics": {
                    "loss_total": 1.43,
                    "loss_p_pass_bce": 0.53,
                    "steps_per_second": 0.255,
                },
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job_progress.jsonl"
            path.write_text(json.dumps(event, sort_keys=True), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(STATUS_SCRIPT), "--from-file", str(path)],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        self.assertIn("step=200/12000", completed.stdout)
        self.assertIn("loss_p_pass_bce=0.53", completed.stdout)

    def test_script_human_output_includes_runtime_phase(self) -> None:
        event = _event(
            "runtime.upload_start",
            {
                "upload_path_in_repo": "codelewm-v0-8-short-seed-42",
                "elapsed_seconds": 10,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime.jsonl"
            path.write_text(json.dumps(event, sort_keys=True), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(STATUS_SCRIPT), "--from-file", str(path)],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        self.assertIn("runtime.upload_start", completed.stdout)
        self.assertIn("upload_path_in_repo=codelewm-v0-8-short-seed-42", completed.stdout)

    def test_script_accepts_retry_flags_for_saved_progress_jsonl(self) -> None:
        event = _event(
            "execution_training.progress",
            {
                "seed": 42,
                "step": 200,
                "max_steps": 12000,
                "progress": 0.016667,
                "elapsed_seconds": 782.0,
                "eta_seconds": 46141.0,
                "metrics": {"loss_total": 1.43},
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job_progress.jsonl"
            path.write_text(json.dumps(event, sort_keys=True), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(STATUS_SCRIPT),
                    "--from-file",
                    str(path),
                    "--retries",
                    "0",
                    "--retry-sleep",
                    "0",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["summaries"][0]["latest_progress"]["step"], 200)

    def test_retry_helper_reports_final_attempt(self) -> None:
        script_globals = runpy.run_path(str(STATUS_SCRIPT))
        run_hf_command = script_globals["_run_hf_command"]

        completed, error = run_hf_command(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            retries=2,
            retry_sleep=0,
        )

        self.assertIsNone(completed)
        self.assertIsInstance(error, dict)
        self.assertEqual(error["attempt"], 3)
        self.assertEqual(error["attempts"], 3)
        self.assertEqual(error["returncode"], 7)
