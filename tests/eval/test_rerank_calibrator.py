"""Tests for the Phase-0 rerank calibration probe (RFC-0015 WS-A1)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.eval.rerank_calibrator import (
    RERANK_CALIBRATION_REPORT_SCHEMA_VERSION,
    RerankCalibratorError,
    evaluate_rerank_calibration,
    load_completion_scores,
    roc_auc,
)


def _separable_rows(n_problems: int = 40, seed: int = 0) -> list[dict[str, object]]:
    """Rows where `codelewm` strongly predicts `passed` (separable signal)."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for p in range(n_problems):
        for rank in range(1, 4):
            passed = bool(rng.integers(0, 2))
            rows.append(
                {
                    "completion_id": f"P{p}::rank-{rank}",
                    "problem_id": f"P{p}",
                    "passed": passed,
                    "llm_order_rank": rank,
                    "scores": {
                        # codelewm carries the signal; others are noise.
                        "codelewm": (1.0 if passed else 0.0) + 0.05 * rng.standard_normal(),
                        "no_action": 0.5 + 0.05 * rng.standard_normal(),
                        "lexical": 0.5 + 0.05 * rng.standard_normal(),
                        "shuffled_action": 0.5 + 0.05 * rng.standard_normal(),
                        "random": float(rng.random()),
                    },
                }
            )
    return rows


class RocAucTest(unittest.TestCase):
    def test_perfect_separation_is_one(self) -> None:
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(roc_auc(y, s), 1.0)

    def test_inverted_is_zero(self) -> None:
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertEqual(roc_auc(y, s), 0.0)

    def test_single_class_returns_none(self) -> None:
        self.assertIsNone(roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


class RerankCalibrationTest(unittest.TestCase):
    def test_separable_signal_is_decodable_and_helps_rerank(self) -> None:
        rows = _separable_rows()
        report = evaluate_rerank_calibration(rows, k_folds=4, seed=1, bootstrap_samples=200)
        self.assertEqual(report["schema_version"], RERANK_CALIBRATION_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["completion_count"], len(rows))
        # The calibrator and the codelewm feature recover the planted signal.
        self.assertGreater(report["decodability"]["calibrator_cv_auc"], 0.8)
        self.assertGreater(report["decodability"]["univariate_auc"]["codelewm"], 0.8)
        # On the rerankable subset, the calibrator beats the noise baseline.
        unsat = report["rerank_pass_at_1"]["unsaturated_only"]
        self.assertIsNotNone(unsat["calibrator"])
        self.assertGreaterEqual(unsat["calibrator"], unsat["no_action"])
        # JSON-native.
        json.dumps(report, allow_nan=False, sort_keys=True)

    def test_noise_signal_is_near_chance(self) -> None:
        # passed independent of all features -> AUC near 0.5.
        rng = np.random.default_rng(3)
        rows = []
        for p in range(40):
            for rank in range(1, 4):
                rows.append(
                    {
                        "completion_id": f"P{p}::r{rank}",
                        "problem_id": f"P{p}",
                        "passed": bool(rng.integers(0, 2)),
                        "llm_order_rank": rank,
                        "scores": {k: float(rng.random()) for k in
                                   ("codelewm", "no_action", "lexical", "shuffled_action", "random")},
                    }
                )
        report = evaluate_rerank_calibration(rows, k_folds=4, seed=2, bootstrap_samples=200)
        self.assertLess(abs(report["decodability"]["calibrator_cv_auc"] - 0.5), 0.2)

    def test_structural_ceiling_fields(self) -> None:
        rows = _separable_rows(n_problems=10)
        report = evaluate_rerank_calibration(rows, k_folds=3, seed=0, bootstrap_samples=50)
        self.assertEqual(
            report["rerankable_problem_count"]
            + report["all_pass_problem_count"]
            + report["all_fail_problem_count"],
            report["problem_count"],
        )
        self.assertIn("max_possible_rerank_headroom_pts", report)

    def test_load_round_trip_and_empty_error(self) -> None:
        rows = _separable_rows(n_problems=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completion_scores.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            loaded = load_completion_scores([path])
            self.assertEqual(len(loaded), len(rows))
            empty = Path(tmp) / "empty.jsonl"
            empty.write_text("\n", encoding="utf-8")
            with self.assertRaises(RerankCalibratorError):
                load_completion_scores([empty])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
