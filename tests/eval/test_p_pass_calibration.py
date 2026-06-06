"""Tests for held-out p_pass calibration reports."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    P_PASS_CALIBRATION_REPORT_SCHEMA_VERSION,
    P_PASS_CALIBRATION_RUN_SCHEMA_VERSION,
    PPassCalibrationError,
    build_p_pass_calibration_report,
)
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


def _row(
    *,
    passed: bool,
    score: float,
    benchmark_id: str = "humaneval",
    split: str = "test",
) -> dict[str, object]:
    return {
        "schema_version": "codelewm.eval.completion_score.v1",
        "completion_id": f"{benchmark_id}-{split}-{score}-{passed}",
        "benchmark_id": benchmark_id,
        "split": split,
        "passed": passed,
        "scores": {"p_pass": score, "codelewm": score},
    }


class PPassCalibrationMetricsTest(unittest.TestCase):
    def test_perfect_scores_emit_auc_average_precision_and_calibration(self) -> None:
        report = build_p_pass_calibration_report(
            (
                _row(passed=True, score=0.95),
                _row(passed=True, score=0.85),
                _row(passed=False, score=0.15),
                _row(passed=False, score=0.05),
            ),
            dataset_kind="downstream_completion",
            baselines=("p_pass",),
        )

        metric = report["overall"]["baselines"]["p_pass"]
        self.assertEqual(report["schema_version"], P_PASS_CALIBRATION_REPORT_SCHEMA_VERSION)
        self.assertEqual(metric["status"], "ok")
        self.assertEqual(metric["roc_auc"], 1.0)
        self.assertEqual(metric["average_precision"], 1.0)
        self.assertLess(metric["brier_score"], 0.03)
        self.assertEqual(metric["thresholded"]["accuracy"], 1.0)
        self.assertFalse(report["claim_allowed"])

    def test_tied_scores_emit_chance_roc_auc(self) -> None:
        report = build_p_pass_calibration_report(
            (
                _row(passed=True, score=0.5),
                _row(passed=False, score=0.5),
                _row(passed=True, score=0.5),
                _row(passed=False, score=0.5),
            ),
            dataset_kind="downstream_completion",
            baselines=("p_pass",),
        )

        metric = report["overall"]["baselines"]["p_pass"]
        self.assertEqual(metric["status"], "ok")
        self.assertEqual(metric["roc_auc"], 0.5)

    def test_single_class_rows_are_not_auc_evaluable(self) -> None:
        report = build_p_pass_calibration_report(
            (
                _row(passed=True, score=0.9),
                _row(passed=True, score=0.8),
            ),
            dataset_kind="training_pack_held_out",
            baselines=("p_pass",),
        )

        metric = report["overall"]["baselines"]["p_pass"]
        self.assertEqual(metric["status"], "single_class")
        self.assertIsNone(metric["roc_auc"])
        self.assertIsNone(metric["average_precision"])
        self.assertEqual(metric["positive_count"], 2)
        self.assertEqual(metric["negative_count"], 0)

    def test_nonfinite_scores_are_counted_and_excluded(self) -> None:
        report = build_p_pass_calibration_report(
            (
                _row(passed=True, score=0.9),
                _row(passed=False, score=0.1),
                _row(passed=True, score=math.nan),
            ),
            dataset_kind="downstream_completion",
            baselines=("p_pass",),
        )

        metric = report["overall"]["baselines"]["p_pass"]
        self.assertEqual(metric["status"], "ok")
        self.assertEqual(metric["nonfinite_score_count"], 1)
        self.assertEqual(metric["usable_row_count"], 2)
        self.assertEqual(metric["roc_auc"], 1.0)

    def test_per_benchmark_slices_keep_coverage_visible(self) -> None:
        report = build_p_pass_calibration_report(
            (
                _row(passed=True, score=0.9, benchmark_id="humaneval"),
                _row(passed=False, score=0.1, benchmark_id="humaneval"),
                _row(passed=True, score=0.8, benchmark_id="mbpp-plus"),
                _row(passed=False, score=0.2, benchmark_id="mbpp-plus"),
            ),
            dataset_kind="downstream_completion",
            baselines=("p_pass",),
        )

        benchmark_slices = report["slices"]["benchmark"]
        self.assertEqual(report["benchmark_counts"], {"humaneval": 2, "mbpp_plus": 2})
        self.assertEqual(benchmark_slices["humaneval"]["row_count"], 2)
        self.assertEqual(benchmark_slices["mbpp_plus"]["row_count"], 2)
        self.assertEqual(
            benchmark_slices["mbpp_plus"]["baselines"]["p_pass"]["roc_auc"],
            1.0,
        )

    def test_invalid_dataset_kind_is_rejected(self) -> None:
        with self.assertRaises(PPassCalibrationError):
            build_p_pass_calibration_report(
                (_row(passed=True, score=0.9),),
                dataset_kind="scratch",
                baselines=("p_pass",),
            )


class PPassCalibrationCliTest(unittest.TestCase):
    def test_cli_writes_manifest_backed_report_with_parent_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores_path = root / "completion_scores.jsonl"
            scores_path.write_text(
                "\n".join(
                    json.dumps(row, sort_keys=True, allow_nan=False)
                    for row in (
                        _row(passed=True, score=0.95),
                        _row(passed=False, score=0.05),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            parent_manifest = build_artifact_manifest(
                artifact_kind="eval_report",
                root=root,
                files=(scores_path,),
                command=("fixture", "completion-scores"),
                config={"scores": "completion_scores.jsonl"},
                metadata={"schema_version": "fixture.completion_scores.v1"},
            )
            parent_manifest_path = root / "parent_manifest.json"
            write_artifact_manifest(parent_manifest, parent_manifest_path)
            out_dir = root / "p_pass_calibration"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "eval",
                    "p-pass-calibration",
                    "--scores",
                    str(scores_path),
                    "--parent-manifest",
                    str(parent_manifest_path),
                    "--dataset-kind",
                    "downstream_completion",
                    "--baseline",
                    "p_pass",
                    "--out",
                    str(out_dir),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(
                result["schema_version"],
                P_PASS_CALIBRATION_RUN_SCHEMA_VERSION,
            )
            self.assertEqual(result["parent_artifacts"], [parent_manifest.artifact_id])
            manifest = read_artifact_manifest(out_dir / "manifest.json")
            validate_artifact_checksums(manifest, root=out_dir)
            report = json.loads(
                (out_dir / "reports" / "p_pass_calibration_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["row_count"], 2)
            self.assertEqual(report["overall"]["baselines"]["p_pass"]["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
