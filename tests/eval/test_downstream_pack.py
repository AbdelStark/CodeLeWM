from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    DOWNSTREAM_BENCHMARK_PACK_RUN_SCHEMA_VERSION,
    DOWNSTREAM_BENCHMARK_READINESS_SCHEMA_VERSION,
    DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION,
    DOWNSTREAM_SOURCE_LICENSE_POLICY_SCHEMA_VERSION,
    DOWNSTREAM_SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION,
    build_downstream_benchmark_pack,
    read_downstream_rerank_benchmark,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums
from codelewm.security.secret_scan import SECRET_SCAN_REPORT_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "config" / "benchmark" / "downstream_rerank_fixture.json"


class DownstreamBenchmarkPackTest(unittest.TestCase):
    def test_pack_builder_writes_manifested_claim_blocked_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = build_downstream_benchmark_pack(
                config_path=FIXTURE_CONFIG,
                out=root / "downstream",
                command=("codelewm", "eval", "downstream-pack"),
            )
            out = root / "downstream"
            manifest = read_artifact_manifest(out / result.artifact_manifest_path)
            checked = validate_artifact_checksums(manifest, root=out)
            benchmark = read_downstream_rerank_benchmark(out / result.benchmark_path)
            readiness = json.loads((out / result.readiness_report_path).read_text(encoding="utf-8"))
            source_policy = json.loads((out / result.source_license_policy_path).read_text(encoding="utf-8"))
            split_report = json.loads((out / result.split_leakage_report_path).read_text(encoding="utf-8"))
            secret_report = json.loads((out / result.secret_scan_report_path).read_text(encoding="utf-8"))

        self.assertEqual(result.schema_version, DOWNSTREAM_BENCHMARK_PACK_RUN_SCHEMA_VERSION)
        self.assertEqual(result.example_count, 1)
        self.assertEqual(result.labeled_example_count, 1)
        self.assertFalse(result.scaled_evaluation_ready)
        self.assertFalse(result.downstream_claim_allowed)
        self.assertEqual(manifest.artifact_kind, "downstream_benchmark")
        self.assertEqual(manifest.metadata["example_count"], 1)
        self.assertEqual(benchmark.schema_version, DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION)
        self.assertEqual(benchmark.tasks[0].before_path, "tasks/class-method-fixture/before.py")
        self.assertEqual(len(benchmark.tasks[0].candidates), 4)
        self.assertEqual(readiness["schema_version"], DOWNSTREAM_BENCHMARK_READINESS_SCHEMA_VERSION)
        self.assertIn("labeled_example_count_below_minimum:1<100", readiness["blocked_reasons"])
        self.assertEqual(source_policy["schema_version"], DOWNSTREAM_SOURCE_LICENSE_POLICY_SCHEMA_VERSION)
        self.assertTrue(source_policy["publication_allowed"])
        self.assertEqual(split_report["schema_version"], DOWNSTREAM_SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION)
        self.assertTrue(split_report["ok"])
        self.assertEqual(secret_report["schema_version"], SECRET_SCAN_REPORT_SCHEMA_VERSION)
        self.assertEqual(secret_report["findings"], [])
        self.assertIn("benchmark.json", {path.name for path in checked})

    def test_cli_builds_fixture_pack_and_emits_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "downstream"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "eval",
                    "downstream-pack",
                    "--config",
                    str(FIXTURE_CONFIG),
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
            manifest = read_artifact_manifest(out / payload["artifact_manifest_path"])

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(payload["schema_version"], DOWNSTREAM_BENCHMARK_PACK_RUN_SCHEMA_VERSION)
        self.assertEqual(payload["example_count"], 1)
        self.assertFalse(payload["scaled_evaluation_ready"])
        self.assertEqual(manifest.artifact_kind, "downstream_benchmark")


if __name__ == "__main__":
    unittest.main()
