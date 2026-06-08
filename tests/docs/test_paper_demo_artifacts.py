from __future__ import annotations

import json
import unittest
from pathlib import Path

from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "docs" / "benchmark" / "v1_0" / "paper_demo"
RESULTS_NOTE = ROOT / "docs" / "benchmark" / "PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md"


class PaperDemoArtifactsDocsTest(unittest.TestCase):
    def test_committed_paper_demo_artifact_set_is_manifest_backed(self) -> None:
        manifest = read_artifact_manifest(ARTIFACT_DIR / "manifest.json")
        checked = validate_artifact_checksums(manifest, root=ARTIFACT_DIR)
        report = json.loads(
            (ARTIFACT_DIR / "reports" / "paper_demo_report.json").read_text(
                encoding="utf-8"
            )
        )
        claim_gate = json.loads(
            (ARTIFACT_DIR / "reports" / "paper_demo_claim_gate.json").read_text(
                encoding="utf-8"
            )
        )
        secret_scan = json.loads(
            (ARTIFACT_DIR / "reports" / "secret_scan_report.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(manifest.artifact_kind, "demo_report")
        self.assertEqual(manifest.artifact_id, "demo_report-e6fc06c328eed245")
        self.assertEqual(
            set(manifest.parent_artifacts),
            {
                "eval_report-0bc9a04d4a6bfa86",
                "eval_report-7e9fa967ee6356af",
                "eval_report-3cd1cfeeb2fe2c09",
                "eval_report-570bdbfeac5928ef",
            },
        )
        self.assertEqual(len(checked), 6)
        self.assertEqual(
            report["schema_version"], "codelewm.harness.paper_demo_report.v1"
        )
        self.assertEqual(report["score_source"], "replay_existing_scores")
        self.assertEqual(report["aggregate_summary"]["slice_count"], 4)
        self.assertEqual(report["aggregate_summary"]["completion_count_sum"], 768)
        self.assertFalse(report["claim_gate"]["allowed"])
        self.assertFalse(claim_gate["allowed"])
        self.assertTrue(secret_scan["ok"], msg=secret_scan)
        self.assertFalse(report["candidate_code_policy"]["executes_candidate_code"])

    def test_readme_and_results_note_link_committed_artifact_paths(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        note = RESULTS_NOTE.read_text(encoding="utf-8")
        for text in (readme, note):
            self.assertIn("docs/benchmark/v1_0/paper_demo", text)
        self.assertIn("PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md", readme)
        self.assertIn("demo_report-e6fc06c328eed245", note)
        self.assertIn("claim_allowed=false", note)


if __name__ == "__main__":
    unittest.main()
