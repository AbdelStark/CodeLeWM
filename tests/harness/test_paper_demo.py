from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.harness.paper_demo import (
    PAPER_DEMO_REPORT_SCHEMA_VERSION,
    PAPER_DEMO_SCORE_SOURCE,
    run_paper_demo,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


ROOT = Path(__file__).resolve().parents[2]


class PaperDemoTest(unittest.TestCase):
    def test_runner_writes_report_html_timeline_secret_scan_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "paper-demo"
            result = run_paper_demo(
                source_root=ROOT,
                out=out,
                overwrite=True,
                command=("codelewm", "paper-demo", "--out", str(out)),
            )

            manifest = read_artifact_manifest(out / result.artifact_manifest_path)
            checked = validate_artifact_checksums(manifest, root=out)
            report_text = (out / result.report_path).read_text(encoding="utf-8")
            report = json.loads(report_text)
            claim_gate = json.loads((out / result.claim_gate_path).read_text(encoding="utf-8"))
            timeline = json.loads((out / result.timeline_path).read_text(encoding="utf-8"))
            secret_scan = json.loads(
                (out / result.secret_scan_report_path).read_text(encoding="utf-8")
            )
            html = (out / result.html_path).read_text(encoding="utf-8")
            table = (out / result.table_path).read_text(encoding="utf-8")

        self.assertEqual(report["schema_version"], PAPER_DEMO_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["score_source"], PAPER_DEMO_SCORE_SOURCE)
        self.assertEqual(len(report["slices"]), 4)
        self.assertFalse(report["claim_gate"]["allowed"])
        self.assertFalse(claim_gate["allowed"])
        self.assertTrue(secret_scan["ok"], msg=secret_scan)
        self.assertEqual(timeline["schema_version"], "codelewm.run_timeline.v1")
        self.assertIn("CodeLeWM v1.0 Paper Demo", html)
        self.assertIn("Candidate Rankings", html)
        self.assertIn("Aggregate claim gate: closed", table)
        self.assertEqual(manifest.artifact_kind, "demo_report")
        self.assertEqual(
            set(manifest.parent_artifacts),
            {item["source_artifact_id"] for item in report["slices"]},
        )
        self.assertEqual(
            {path.name for path in checked},
            {
                "paper_demo_report.json",
                "paper_demo_claim_gate.json",
                "paper_demo_table.md",
                "run_timeline.json",
                "demo.html",
                "secret_scan_report.json",
            },
        )

        policy = report["candidate_code_policy"]
        self.assertTrue(policy["candidate_code_is_untrusted"])
        self.assertFalse(policy["imports_candidate_code"])
        self.assertFalse(policy["executes_candidate_code"])
        self.assertFalse(policy["test_runs_candidate_code"])
        self.assertNotIn("completion_text", report_text)
        self.assertNotIn('"code":', report_text)

        first_slice = report["slices"][0]
        first_problem = first_slice["candidate_rankings"][0]
        self.assertIn("rankings_by_baseline", first_problem)
        self.assertIn("codelewm", first_problem["rankings_by_baseline"])
        self.assertIn("score_deltas", first_problem["top_candidates"][0])
        self.assertIn(
            "codelewm_minus_no_action",
            first_problem["top_candidates"][0]["score_deltas"],
        )
        self.assertEqual(
            first_slice["required_baseline_status"][-1]["status"], "not_recorded"
        )

    def test_cli_json_summary_and_manifest_verify_parent_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "paper-demo"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "paper-demo",
                    "--source-root",
                    str(ROOT),
                    "--out",
                    str(out),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(completed.stdout)
            parent_args: list[str] = []
            for parent in payload["parent_manifest_paths"]:
                parent_args.extend(("--parent-manifest", str(ROOT / parent)))
            verify = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "manifest",
                    "verify",
                    "--manifest",
                    str(out / payload["artifact_manifest_path"]),
                    "--json",
                    *parent_args,
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["schema_version"], "codelewm.harness.paper_demo_run.v1")
        self.assertFalse(payload["claim_allowed"])
        self.assertEqual(payload["slice_count"], 4)
        self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
        verify_payload = json.loads(verify.stdout)
        self.assertTrue(verify_payload["ok"])
        self.assertEqual(len(verify_payload["parents_checked"]), 4)


if __name__ == "__main__":
    unittest.main()
