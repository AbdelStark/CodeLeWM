from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    ANTI_SATURATION_PROFILE,
    DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION,
    DOWNSTREAM_BENCHMARK_PACK_RUN_SCHEMA_VERSION,
    DownstreamBenchmarkPackError,
    build_downstream_benchmark_pack,
    load_downstream_benchmark_pack_config,
    read_downstream_rerank_benchmark,
    validate_anti_saturation_report,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "config" / "benchmark" / "hard_downstream_anti_saturation_fixture.json"
PLAIN_FIXTURE_CONFIG = ROOT / "config" / "benchmark" / "downstream_rerank_fixture.json"


class HardDownstreamPackTest(unittest.TestCase):
    def test_pack_writes_anti_saturation_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pack"
            result = build_downstream_benchmark_pack(
                config_path=FIXTURE_CONFIG,
                out=out,
                command=("codelewm", "eval", "downstream-pack"),
            )
            manifest = read_artifact_manifest(out / result.artifact_manifest_path)
            checked = validate_artifact_checksums(manifest, root=out)
            benchmark = read_downstream_rerank_benchmark(out / result.benchmark_path)
            report = json.loads(
                (out / result.anti_saturation_report_path).read_text(encoding="utf-8")
            )
            readiness = json.loads(
                (out / result.readiness_report_path).read_text(encoding="utf-8")
            )

        self.assertEqual(result.schema_version, DOWNSTREAM_BENCHMARK_PACK_RUN_SCHEMA_VERSION)
        self.assertEqual(result.anti_saturation_report_path, "reports/anti_saturation_report.json")
        self.assertFalse(result.anti_saturation_eligible)
        self.assertFalse(result.scaled_evaluation_ready)

        # Manifest records the profile + report and the report file is checksummed.
        self.assertEqual(manifest.metadata["profile"], ANTI_SATURATION_PROFILE)
        self.assertEqual(
            manifest.metadata["anti_saturation_report"], "reports/anti_saturation_report.json"
        )
        self.assertIn("anti_saturation_report.json", {p.name for p in checked})

        # The report is well-formed and only blocked by the fixture problem count.
        validate_anti_saturation_report(report)
        self.assertEqual(report["schema_version"], DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["profile"], ANTI_SATURATION_PROFILE)
        self.assertFalse(report["eligible"])
        self.assertEqual(report["problem_count"], 1)
        self.assertTrue(report["candidate_pool_size"]["ok"])
        self.assertEqual(report["candidate_pool_size"]["min"], 6)
        self.assertEqual(report["blocked_reasons"], ["problem_count_below_minimum:1<100"])
        # Simple baselines are NOT saturated on this fixture.
        self.assertTrue(report["no_action_below_ceiling"])
        self.assertTrue(report["lexical_below_ceiling"])
        self.assertTrue(report["llm_order_below_ceiling"])
        # Dual hard-negative coverage and class accounting are populated.
        self.assertEqual(report["dual_hard_negative_fraction"], 1.0)
        self.assertEqual(report["hard_negative_class_coverage"]["passing_reference"], 1)
        self.assertEqual(report["parser_apply_failure_rate"], 0.0)

        # Readiness report surfaces the anti-saturation blocker.
        self.assertIn("anti_saturation_slice_not_eligible", readiness["blocked_reasons"])
        self.assertFalse(readiness["anti_saturation_eligible"])

        # Materialized candidates carry the hard_negative_class in their source.
        classes = {
            candidate.source.get("hard_negative_class")
            for candidate in benchmark.tasks[0].candidates
        }
        self.assertEqual(
            classes,
            {
                "no_action_bait",
                "partial_fix",
                "passing_reference",
                "wrong_symbol",
                "over_broad",
                "deterministic_mutant",
            },
        )

    def test_plain_fixture_has_no_anti_saturation_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pack"
            result = build_downstream_benchmark_pack(config_path=PLAIN_FIXTURE_CONFIG, out=out)
            self.assertIsNone(result.anti_saturation_report_path)
            self.assertIsNone(result.anti_saturation_eligible)
            self.assertFalse((out / "reports" / "anti_saturation_report.json").exists())

    def test_config_round_trip_preserves_profile_and_classes(self) -> None:
        config = load_downstream_benchmark_pack_config(FIXTURE_CONFIG)
        self.assertEqual(config.profile, ANTI_SATURATION_PROFILE)
        self.assertTrue(config.is_anti_saturation)
        payload = config.to_dict()
        self.assertEqual(payload["profile"], ANTI_SATURATION_PROFILE)
        classes = {c.get("hard_negative_class") for c in payload["tasks"][0]["candidates"]}
        self.assertIn("no_action_bait", classes)

    def test_unknown_hard_negative_class_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            payload = json.loads(FIXTURE_CONFIG.read_text(encoding="utf-8"))
            payload["tasks"][0]["candidates"][0]["hard_negative_class"] = "not_a_real_class"
            bad.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DownstreamBenchmarkPackError):
                load_downstream_benchmark_pack_config(bad)

    def test_cli_builds_anti_saturation_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pack"
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
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(
                payload["anti_saturation_report_path"], "reports/anti_saturation_report.json"
            )
            self.assertFalse(payload["anti_saturation_eligible"])
            self.assertTrue((out / "reports" / "anti_saturation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
