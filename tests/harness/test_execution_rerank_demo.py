from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.harness.execution_rerank_demo import (
    EXECUTION_RERANK_TOUR_REPORT_SCHEMA_VERSION,
    run_execution_rerank_tour,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


class ExecutionRerankDemoTest(unittest.TestCase):
    def test_tour_writes_fixture_report_html_view_model_and_asciicast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            html_export = root / "public-demo.html"

            result = run_execution_rerank_tour(
                checkpoint=checkpoint,
                out=root / "tour",
                tour_count=2,
                env={
                    "CODELEWM_LLM_PROVIDER": "openrouter",
                    "CODELEWM_LLM_DRY_RUN": "1",
                    "CODELEWM_LLM_MAX_CANDIDATES": "2",
                },
                html_export=html_export,
                allow_unsafe_checkpoint=True,
                require_learned_scorer=False,
                overwrite=True,
                command=("scripts/llm-world-model-demo", "--tour", "2"),
            )

            manifest = read_artifact_manifest(root / "tour" / "manifest.json")
            checked = validate_artifact_checksums(manifest, root=root / "tour")
            report = json.loads((root / "tour" / result.report_path).read_text(encoding="utf-8"))
            view_model = json.loads((root / "tour" / result.view_model_path).read_text(encoding="utf-8"))
            html = (root / "tour" / result.html_path).read_text(encoding="utf-8")
            asciicast = (root / "tour" / result.asciicast_path).read_text(encoding="utf-8")
            html_export_exists = html_export.is_file()

        self.assertEqual(report["schema_version"], EXECUTION_RERANK_TOUR_REPORT_SCHEMA_VERSION)
        self.assertEqual(view_model["schema_version"], "codelewm.harness.execution_rerank_view_model.v1")
        self.assertEqual(result.problem_count, 2)
        self.assertEqual(result.completion_count, 4)
        self.assertFalse(result.claim_allowed)
        self.assertFalse(report["claim_gate"]["allowed"])
        self.assertEqual(report["rerank_report"]["problem_count"], 2)
        self.assertEqual(report["rerank_report"]["completions_per_problem"], 2)
        self.assertIn("CodeLeWM execution-rerank tour", html)
        self.assertIn("CodeLeWM order", html)
        self.assertIn("return n * n", html)
        self.assertIn("square-neg:pass", html)
        self.assertIn("CodeLeWM execution-rerank tour", asciicast)
        self.assertTrue(html_export_exists)
        self.assertEqual(manifest.artifact_kind, "demo_report")
        self.assertEqual(
            {path.name for path in checked},
            {
                "execution_rerank_tour_report.json",
                "execution_rerank_view_model.json",
                "demo.html",
                "execution_rerank_tour.cast",
            },
        )
        self.assertGreaterEqual(len(manifest.parent_artifacts), 2)
        first_problem = report["problems"][0]
        self.assertEqual(first_problem["problem_id"], "mbpp-demo-square")
        self.assertEqual(first_problem["candidates"][0]["passed"], True)
        self.assertEqual(first_problem["candidates"][1]["passed"], False)


if __name__ == "__main__":
    unittest.main()
