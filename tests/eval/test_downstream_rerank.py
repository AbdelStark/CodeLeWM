from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    DOWNSTREAM_REQUIRED_BASELINES,
    DOWNSTREAM_RERANK_EVAL_RUN_SCHEMA_VERSION,
    DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION,
    build_downstream_benchmark_pack,
    read_downstream_rerank_report,
    run_downstream_rerank_evaluation,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "config" / "benchmark" / "downstream_rerank_fixture.json"


class DownstreamRerankEvalTest(unittest.TestCase):
    def test_eval_writes_report_with_required_baselines_and_closed_claim_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_dir = root / "benchmark"
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            build_downstream_benchmark_pack(
                config_path=FIXTURE_CONFIG,
                out=benchmark_dir,
                command=("codelewm", "eval", "downstream-pack"),
            )
            result = run_downstream_rerank_evaluation(
                benchmark_manifest=benchmark_dir / "manifest.json",
                checkpoint=checkpoint,
                out=root / "rerank",
                allow_unsafe_checkpoint=True,
                bootstrap_samples=0,
                command=("codelewm", "eval", "downstream-rerank"),
            )
            manifest = read_artifact_manifest(root / "rerank" / result.artifact_manifest_path)
            checked = validate_artifact_checksums(manifest, root=root / "rerank")
            report = read_downstream_rerank_report(root / "rerank" / result.report_path)

        self.assertEqual(result.schema_version, DOWNSTREAM_RERANK_EVAL_RUN_SCHEMA_VERSION)
        self.assertFalse(result.claim_allowed)
        self.assertEqual(manifest.artifact_kind, "eval_report")
        self.assertIn("downstream_rerank_report.json", {path.name for path in checked})
        self.assertEqual(report["schema_version"], DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION)
        self.assertEqual(tuple(report["summary"]["required_baselines"]), DOWNSTREAM_REQUIRED_BASELINES)
        self.assertEqual(set(report["metrics"]), set(DOWNSTREAM_REQUIRED_BASELINES))
        self.assertEqual(report["metrics"]["retrieval_prior"]["status"], "blocked")
        self.assertEqual(
            report["metrics"]["retrieval_prior"]["blocked_reason"],
            "no_finite_retrieval_prior_scores",
        )
        self.assertFalse(report["claim_gate"]["allowed"])
        self.assertIn("example_count_below_minimum:1<100", report["claim_gate"]["failure_reasons"])
        self.assertEqual(report["confidence_intervals"]["status"], "skipped")
        self.assertIn("refactor", report["slices"]["by_task_type"])
        self.assertIn("candidate_001_true_after", report["tasks"][0]["rankings"]["llm_order"])
        self.assertIn("Retrieval-prior baseline is unavailable", "\n".join(report["caveats"]))

    def test_cli_runs_downstream_rerank_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_dir = root / "benchmark"
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            build_downstream_benchmark_pack(
                config_path=FIXTURE_CONFIG,
                out=benchmark_dir,
                command=("codelewm", "eval", "downstream-pack"),
            )
            out = root / "rerank"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "eval",
                    "downstream-rerank",
                    "--benchmark-manifest",
                    str(benchmark_dir / "manifest.json"),
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(out),
                    "--allow-unsafe-checkpoint",
                    "--bootstrap-samples",
                    "0",
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(completed.stdout)
            report = read_downstream_rerank_report(out / payload["report_path"])

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(payload["schema_version"], DOWNSTREAM_RERANK_EVAL_RUN_SCHEMA_VERSION)
        self.assertFalse(payload["claim_allowed"])
        self.assertEqual(report["summary"]["example_count"], 1)


if __name__ == "__main__":
    unittest.main()
