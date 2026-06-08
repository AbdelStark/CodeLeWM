from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    DOWNSTREAM_ANTI_SATURATION_CLAIM_GATE_SCHEMA_VERSION,
    DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION,
    HARD_DOWNSTREAM_REQUIRED_BASELINES,
    build_anti_saturation_claim_gate,
    build_downstream_benchmark_pack,
    read_downstream_rerank_report,
    run_downstream_rerank_evaluation,
)
from codelewm.observability import read_artifact_manifest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "config" / "benchmark" / "hard_downstream_anti_saturation_fixture.json"
PLAIN_FIXTURE_CONFIG = ROOT / "config" / "benchmark" / "downstream_rerank_fixture.json"


class HardModeEvalTest(unittest.TestCase):
    def _build(self, root: Path):
        benchmark_dir = root / "benchmark"
        checkpoint = root / "checkpoint.bin"
        checkpoint.write_bytes(b"fixture checkpoint")
        build_downstream_benchmark_pack(
            config_path=FIXTURE_CONFIG,
            out=benchmark_dir,
            command=("codelewm", "eval", "downstream-pack"),
        )
        return benchmark_dir, checkpoint

    def test_hard_mode_report_has_all_baselines_and_anti_saturation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_dir, checkpoint = self._build(root)
            result = run_downstream_rerank_evaluation(
                benchmark_manifest=benchmark_dir / "manifest.json",
                checkpoint=checkpoint,
                out=root / "rerank",
                allow_unsafe_checkpoint=True,
                bootstrap_samples=0,
                hard_mode=True,
            )
            report = read_downstream_rerank_report(root / "rerank" / result.report_path)
            manifest = read_artifact_manifest(root / "rerank" / result.artifact_manifest_path)

        self.assertTrue(report["hard_mode"])
        self.assertEqual(report["profile"], "anti_saturation_semantic_v1")
        self.assertEqual(
            set(report["metrics"]), set(HARD_DOWNSTREAM_REQUIRED_BASELINES)
        )
        self.assertEqual(report["metrics"]["shuffled_action"]["status"], "completed")
        self.assertEqual(report["metrics"]["static_heuristic"]["status"], "completed")
        self.assertEqual(report["metrics"]["p_pass"]["status"], "not_recorded")
        self.assertEqual(
            report["anti_saturation_report"]["schema_version"],
            DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["claim_gate"]["schema_version"],
            DOWNSTREAM_ANTI_SATURATION_CLAIM_GATE_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["claim_gate"]["checked_baselines"], ["no_action", "lexical", "llm_order"]
        )
        self.assertFalse(report["claim_gate"]["allowed"])
        self.assertIn(
            "example_count_below_minimum:1<100", report["claim_gate"]["failure_reasons"]
        )
        self.assertEqual(report["lift_confidence_intervals"]["status"], "skipped")
        self.assertFalse(result.claim_allowed)
        self.assertTrue(manifest.metadata["hard_mode"])
        self.assertFalse(manifest.metadata["anti_saturation_eligible"])

    def test_plain_mode_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_dir = root / "benchmark"
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            build_downstream_benchmark_pack(config_path=PLAIN_FIXTURE_CONFIG, out=benchmark_dir)
            result = run_downstream_rerank_evaluation(
                benchmark_manifest=benchmark_dir / "manifest.json",
                checkpoint=checkpoint,
                out=root / "rerank",
                allow_unsafe_checkpoint=True,
                bootstrap_samples=0,
            )
            report = read_downstream_rerank_report(root / "rerank" / result.report_path)
        self.assertNotIn("anti_saturation_report", report)
        self.assertNotIn("hard_mode", report)
        self.assertEqual(report["claim_gate"]["schema_version"], "codelewm.downstream_rerank_claim_gate.v1")

    def test_cli_hard_mode_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_dir, checkpoint = self._build(root)
            out = root / "rerank"
            completed = subprocess.run(
                [
                    sys.executable, "-m", "codelewm.harness.cli", "eval", "downstream-rerank",
                    "--benchmark-manifest", str(benchmark_dir / "manifest.json"),
                    "--checkpoint", str(checkpoint),
                    "--out", str(out),
                    "--allow-unsafe-checkpoint",
                    "--bootstrap-samples", "0",
                    "--hard-mode",
                    "--json",
                ],
                cwd=ROOT, check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            report = read_downstream_rerank_report(out / payload["report_path"])
        self.assertFalse(payload["claim_allowed"])
        self.assertTrue(report["hard_mode"])
        self.assertEqual(report["metrics"]["p_pass"]["status"], "not_recorded")


class HardClaimGateOutcomeTest(unittest.TestCase):
    """RFC-0016 #422 claim-gate outcomes: positive, saturated, missing-baseline,
    invalid-candidate-only, mixed-slice."""

    def _metrics(self, **overrides):
        base = {
            "codelewm": {"pass_at_1": 0.72, "mrr": 0.81},
            "no_action": {"pass_at_1": 0.40, "mrr": 0.55},
            "lexical": {"pass_at_1": 0.42, "mrr": 0.57},
            "llm_order": {"pass_at_1": 0.55, "mrr": 0.66},
        }
        base.update(overrides)
        return base

    def _winning_cis(self):
        return {
            "no_action": {"pass_at_1": {"low": 0.1, "high": 0.3}, "mrr": {"low": 0.1, "high": 0.3}},
            "lexical": {"pass_at_1": {"low": 0.08, "high": 0.28}, "mrr": {"low": 0.08, "high": 0.28}},
            "llm_order": {"pass_at_1": {"low": 0.05, "high": 0.25}, "mrr": {"low": 0.05, "high": 0.25}},
        }

    def test_positive_outcome_opens_gate(self) -> None:
        gate = build_anti_saturation_claim_gate(
            example_count=150,
            metrics=self._metrics(),
            anti_saturation_eligible=True,
            lift_confidence_intervals=self._winning_cis(),
        )
        self.assertTrue(gate["allowed"])

    def test_saturated_slice_blocks(self) -> None:
        gate = build_anti_saturation_claim_gate(
            example_count=150, metrics=self._metrics(), anti_saturation_eligible=False
        )
        self.assertFalse(gate["allowed"])
        self.assertIn("anti_saturation_slice_not_eligible", gate["failure_reasons"])

    def test_missing_baseline_blocks(self) -> None:
        metrics = self._metrics()
        del metrics["lexical"]
        gate = build_anti_saturation_claim_gate(
            example_count=150, metrics=metrics, anti_saturation_eligible=True
        )
        self.assertFalse(gate["allowed"])
        self.assertTrue(any(r.startswith("missing_metric:lexical") for r in gate["failure_reasons"]))

    def test_invalid_candidate_only_blocks(self) -> None:
        # When every candidate is invalid, CodeLeWM cannot beat the baselines
        # (all pass@1 collapse to 0.0) -> not strictly above -> blocked.
        zero = {"pass_at_1": 0.0, "mrr": 0.0}
        metrics = {"codelewm": zero, "no_action": zero, "lexical": zero, "llm_order": zero}
        gate = build_anti_saturation_claim_gate(
            example_count=150, metrics=metrics, anti_saturation_eligible=True
        )
        self.assertFalse(gate["allowed"])
        self.assertTrue(any(r.startswith("not_strictly_above") for r in gate["failure_reasons"]))

    def test_mixed_slice_with_ci_including_zero_blocks(self) -> None:
        cis = self._winning_cis()
        cis["llm_order"]["mrr"] = {"low": -0.02, "high": 0.2}  # CI includes zero
        gate = build_anti_saturation_claim_gate(
            example_count=150,
            metrics=self._metrics(),
            anti_saturation_eligible=True,
            lift_confidence_intervals=cis,
        )
        self.assertFalse(gate["allowed"])
        self.assertTrue(any(r.startswith("lift_ci_includes_zero:llm_order:mrr") for r in gate["failure_reasons"]))


if __name__ == "__main__":
    unittest.main()
