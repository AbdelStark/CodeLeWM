from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.eval import (
    KILL_REPORT_SCHEMA_VERSION,
    CollapseThresholds,
    EvaluationGateError,
    compute_collapse_report,
    enforce_collapse_gates,
    evaluate_collapse_gates,
)


class CollapseDiagnosticsTest(unittest.TestCase):
    def test_effective_rank_matches_known_two_dimensional_subspace(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
            ]
        )

        report = compute_collapse_report(embeddings)

        self.assertAlmostEqual(report.effective_rank, 2.0)
        self.assertAlmostEqual(report.effective_rank_ratio, 0.5)
        self.assertEqual(report.embedding_count, 4)
        self.assertEqual(report.latent_dim, 4)

    def test_noncollapsed_embeddings_pass_default_gate(self) -> None:
        rng = np.random.default_rng(7)
        report = compute_collapse_report(rng.normal(size=(32, 8)))

        failures = evaluate_collapse_gates(report)

        self.assertEqual(failures, ())

    def test_forced_collapse_writes_kill_report_and_raises(self) -> None:
        report = compute_collapse_report(np.ones((8, 4)))
        thresholds = CollapseThresholds(
            effective_rank_ratio_min=0.20,
            per_dim_variance_median_min=1e-8,
            nearest_neighbor_entropy_min=0.10,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kill-report.json"
            with self.assertRaisesRegex(EvaluationGateError, "collapse gate failed"):
                enforce_collapse_gates(
                    report,
                    thresholds,
                    kill_report_path=path,
                    command=("codelewm", "train"),
                    config_hash="abc123",
                )

            payload = json.loads(path.read_text())

        self.assertEqual(payload["schema_version"], KILL_REPORT_SCHEMA_VERSION)
        self.assertEqual(payload["reason"], "embedding_collapse")
        self.assertEqual(payload["config_hash"], "abc123")
        self.assertIn("collapse_report", payload)
        self.assertTrue(any(item["metric"] == "effective_rank_ratio" for item in payload["failures"]))
        self.assertTrue(any(item["metric"] == "per_dim_variance_median" for item in payload["failures"]))

    def test_report_rejects_nonfinite_embeddings(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or inf"):
            compute_collapse_report(np.array([[1.0, float("nan")]]))

    def test_three_dimensional_embeddings_are_flattened_over_time_and_batch(self) -> None:
        embeddings = np.zeros((3, 2, 4))
        embeddings[:, :, 0] = np.arange(6).reshape(3, 2)

        report = compute_collapse_report(embeddings)

        self.assertEqual(report.embedding_count, 6)
        self.assertEqual(report.latent_dim, 4)


if __name__ == "__main__":
    unittest.main()
