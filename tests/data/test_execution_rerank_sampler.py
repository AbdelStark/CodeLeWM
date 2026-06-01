"""Tests for the execution rerank completion sampler."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.data.execution_rerank_sampler import (
    COMPLETION_LABEL_ARTIFACT_SCHEMA_VERSION,
    COMPLETION_LABEL_SCHEMA_VERSION,
    COMPLETION_SAMPLING_REPORT_SCHEMA_VERSION,
    sample_execution_rerank_completions,
)
from codelewm.eval import (
    COMPLETION_LABEL_SCHEMA_VERSION as EVAL_COMPLETION_LABEL_SCHEMA_VERSION,
)
from codelewm.eval import load_completion_labels
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "data" / "execution_sources" / "fixtures"
SCRIPT = ROOT / "scripts" / "sample-execution-rerank-completions"


class ExecutionRerankSamplerTest(unittest.TestCase):
    def test_label_schema_version_matches_eval_loader(self) -> None:
        self.assertEqual(
            COMPLETION_LABEL_SCHEMA_VERSION,
            EVAL_COMPLETION_LABEL_SCHEMA_VERSION,
        )

    def test_dry_run_sampler_writes_manifested_humaneval_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "labels"
            result = sample_execution_rerank_completions(
                benchmark="humaneval",
                source_path=FIXTURES / "humaneval_tiny.jsonl",
                out=out,
                samples_per_problem=2,
                llm_seeds=(17,),
            )

            self.assertEqual(
                result.schema_version, COMPLETION_LABEL_ARTIFACT_SCHEMA_VERSION
            )
            self.assertEqual(result.completion_count, 2)
            self.assertEqual(result.passed_completion_count, 1)

            labels_path = out / result.labels_path
            rows = [
                json.loads(line)
                for line in labels_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["schema_version"] for row in rows},
                {COMPLETION_LABEL_SCHEMA_VERSION},
            )
            self.assertEqual({row["label"] for row in rows}, {"pass", "fail"})
            self.assertTrue(all(row["test_results"] for row in rows))
            self.assertTrue(
                all("expected_output_sha256" in row["test_results"][0] for row in rows)
            )
            self.assertTrue(all(row["scoring_inputs"] for row in rows))
            self.assertTrue(
                all("input_repr" in row["scoring_inputs"][0] for row in rows)
            )

            loaded = load_completion_labels(labels_path, benchmark_id="humaneval")
            self.assertEqual(len(loaded), 2)
            self.assertTrue(any(label.passed for label in loaded))
            self.assertTrue(any(not label.passed for label in loaded))
            self.assertTrue(all(label.scoring_inputs for label in loaded))

            report = json.loads((out / result.report_path).read_text(encoding="utf-8"))
            self.assertEqual(
                report["schema_version"], COMPLETION_SAMPLING_REPORT_SCHEMA_VERSION
            )
            secret_scan = json.loads(
                (out / result.secret_scan_report_path).read_text(encoding="utf-8")
            )
            self.assertTrue(secret_scan["ok"])

            manifest = read_artifact_manifest(out / result.artifact_manifest_path)
            validate_artifact_checksums(manifest, root=out)

    def test_script_dry_run_writes_mbpp_plus_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mbpp"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--benchmark",
                    "mbpp-plus",
                    "--source",
                    str(FIXTURES / "mbpp_plus_tiny.jsonl"),
                    "--out",
                    str(out),
                    "--samples-per-problem",
                    "2",
                    "--llm-seeds",
                    "42",
                    "--json",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["benchmark_id"], "mbpp_plus")
            self.assertEqual(payload["completion_count"], 2)

            labels_path = out / payload["labels_path"]
            labels = load_completion_labels(labels_path, benchmark_id="mbpp-plus")
            self.assertEqual(len(labels), 2)
            self.assertEqual({label.passed for label in labels}, {False, True})
            self.assertTrue(all(label.scoring_inputs for label in labels))

    def test_case_cap_and_short_circuit_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "labels"
            result = sample_execution_rerank_completions(
                benchmark="humaneval",
                source_path=FIXTURES / "humaneval_tiny.jsonl",
                out=out,
                samples_per_problem=2,
                llm_seeds=(17,),
                max_cases_per_problem=1,
                short_circuit_failures=True,
            )

            labels_path = out / result.labels_path
            rows = [
                json.loads(line)
                for line in labels_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(len(row["scoring_inputs"]) == 1 for row in rows))
            failing = [row for row in rows if not row["passed"]]
            self.assertEqual(len(failing), 1)
            self.assertEqual(len(failing[0]["test_results"]), 1)

            report = json.loads((out / result.report_path).read_text(encoding="utf-8"))
            self.assertEqual(report["max_cases_per_problem"], 1)
            self.assertTrue(report["short_circuit_failures"])
            self.assertEqual(report["problem_summaries"][0]["input_case_count"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
