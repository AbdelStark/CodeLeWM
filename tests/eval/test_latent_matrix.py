from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.eval import (
    LATENT_MATRIX_REPORT_SCHEMA_VERSION,
    LATENT_PROBE_VIEWS,
    LatentMatrixConfig,
    LatentMatrixError,
    LatentProbeConfig,
    LatentProbeRow,
    build_latent_matrix_report,
    build_latent_probe_report,
    read_latent_matrix_report,
    write_latent_matrix_report,
    write_latent_probe_report,
)
from codelewm.eval.latent_probe import LATENT_PROBE_TARGETS


class LatentMatrixReportTest(unittest.TestCase):
    def test_report_builds_finite_matrix_geometry_and_blocked_claim_gates(self) -> None:
        rows = _fixture_rows()
        matrix = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.2],
                [-1.0, 0.0, 0.1, 0.2],
                [0.9, 0.1, 0.0, 0.3],
                [-0.9, 0.1, 0.1, 0.3],
                [0.8, 0.0, 0.0, 0.4],
                [-0.8, 0.0, 0.1, 0.4],
            ],
            dtype=np.float64,
        )

        report = build_latent_matrix_report(
            rows,
            embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
            config=LatentMatrixConfig(
                matrix_dimension_limit=2,
                top_dimensions=2,
                max_pairwise_rows=4,
                seed=9,
            ),
            metadata={"checkpoint": {"sha256": "fixture"}},
        )

        self.assertEqual(report.schema_version, LATENT_MATRIX_REPORT_SCHEMA_VERSION)
        self.assertEqual(report.row_count, 6)
        self.assertEqual(report.split_counts, {"test": 2, "train": 2, "val": 2})
        self.assertEqual(report.views["z_pred_after"]["shape"], {"rows": 6, "dimensions": 4})
        self.assertEqual(len(report.views["z_pred_after"]["dimension_statistics"]), 4)
        self.assertEqual(
            report.views["z_pred_after"]["heatmap_matrices"]["dimension_selection"]["selected_count"],
            2,
        )
        self.assertFalse(report.views["z_pred_after"]["matrix_policy"]["raw_latent_vectors_serialized"])
        self.assertGreater(report.views["z_pred_after"]["effective_rank"], 0.0)
        self.assertIn("edit_class", report.probe_associations["inline_dimension_associations"])
        self.assertFalse(report.claim_boundary["positive_representation_claim_allowed"])
        self.assertFalse(report.claim_boundary["semantic_axis_claim_allowed"])

    def test_report_links_existing_latent_probe_controls(self) -> None:
        rows = _fixture_rows()
        matrix = np.eye(len(rows), 4, dtype=np.float64)
        probe = build_latent_probe_report(
            rows,
            embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
            baselines={
                "random_latent": np.flip(matrix, axis=1),
                "no_action": matrix,
                "shuffled_action": np.roll(matrix, shift=1, axis=0),
            },
            config=LatentProbeConfig(bootstrap_samples=0),
        )

        with tempfile.TemporaryDirectory() as tmp:
            probe_path = Path(tmp) / "latent_probe_report.json"
            write_latent_probe_report(probe, probe_path)
            report = build_latent_matrix_report(
                rows,
                embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
                latent_probe_report=probe,
                latent_probe_report_path=probe_path,
            )

        linked = report.probe_associations["latent_probe_report"]
        self.assertTrue(linked["available"])
        self.assertEqual(linked["schema_version"], probe.schema_version)
        self.assertEqual(len(linked["sha256"]), 64)
        self.assertIn("majority", linked["targets"]["edit_class"]["controls"])
        self.assertEqual(
            report.claim_boundary["semantic_axis_gate"]["controls_beat_status"],
            probe.claim_boundary["semantic_structure_status"],
        )

    def test_report_round_trips_json(self) -> None:
        rows = _fixture_rows()
        matrix = np.eye(len(rows), 3, dtype=np.float64)
        report = build_latent_matrix_report(
            rows,
            embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
            config=LatentMatrixConfig(matrix_dimension_limit=2),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latent_matrix_report.json"
            write_latent_matrix_report(report, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = read_latent_matrix_report(path)

        self.assertEqual(payload["schema_version"], LATENT_MATRIX_REPORT_SCHEMA_VERSION)
        self.assertEqual(loaded.to_dict(), report.to_dict())

    def test_report_rejects_nonfinite_values_and_row_mismatch(self) -> None:
        rows = _fixture_rows()
        matrix = np.zeros((len(rows), 2), dtype=np.float64)
        matrix[0, 0] = np.nan

        with self.assertRaisesRegex(LatentMatrixError, "NaN or inf"):
            build_latent_matrix_report(
                rows,
                embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
            )

        with self.assertRaisesRegex(LatentMatrixError, "row count mismatch"):
            build_latent_matrix_report(
                rows,
                embeddings={view: np.zeros((len(rows) - 1, 2), dtype=np.float64) for view in LATENT_PROBE_VIEWS},
            )


def _fixture_rows() -> tuple[LatentProbeRow, ...]:
    return (
        _row("train-a", "train", "left"),
        _row("train-b", "train", "right"),
        _row("val-a", "val", "left"),
        _row("val-b", "val", "right"),
        _row("test-a", "test", "left"),
        _row("test-b", "test", "right"),
    )


def _row(transition_id: str, split: str, label: str) -> LatentProbeRow:
    labels = {target: f"{target}:{label}" for target in LATENT_PROBE_TARGETS}
    return LatentProbeRow(
        transition_id=transition_id,
        split=split,
        labels=labels,
        metadata_features={
            "source": "synthetic",
            "edit_size_bucket": "0-9",
            "action_cluster": label,
        },
        lexical_tokens=(1 if label == "left" else 2, 10),
    )


if __name__ == "__main__":
    unittest.main()
