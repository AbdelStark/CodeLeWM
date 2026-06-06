from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.eval import (
    LATENT_PROBE_REPORT_SCHEMA_VERSION,
    LATENT_PROBE_TARGETS,
    LATENT_PROBE_VIEWS,
    LatentProbeConfig,
    LatentProbeError,
    LatentProbeRow,
    build_latent_probe_report,
    read_latent_probe_report,
    write_latent_probe_report,
)


class LatentProbeReportTest(unittest.TestCase):
    def test_report_builds_predeclared_targets_controls_and_axis_gates(self) -> None:
        rows = _fixture_rows()
        matrix = np.asarray(
            [
                [1.0, 0.0, 0.1],
                [-1.0, 0.0, 0.1],
                [0.9, 0.1, 0.0],
                [-0.9, 0.1, 0.0],
                [0.8, 0.0, 0.2],
                [-0.8, 0.0, 0.2],
            ],
            dtype=np.float64,
        )

        report = build_latent_probe_report(
            rows,
            embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
            baselines={
                "random_latent": np.flip(matrix, axis=1),
                "no_action": np.zeros_like(matrix),
                "shuffled_action": np.roll(matrix, shift=1, axis=0),
            },
            config=LatentProbeConfig(bootstrap_samples=8, seed=11, top_dimensions=2),
            metadata={"checkpoint": {"sha256": "fixture"}},
        )

        self.assertEqual(report.schema_version, LATENT_PROBE_REPORT_SCHEMA_VERSION)
        self.assertEqual(set(report.target_reports), set(LATENT_PROBE_TARGETS))
        self.assertEqual(report.split_counts, {"test": 2, "train": 2, "val": 2})
        self.assertEqual(report.claim_boundary["available_target_count"], len(LATENT_PROBE_TARGETS))
        self.assertFalse(report.claim_boundary["positive_representation_claim_allowed"])
        for target in LATENT_PROBE_TARGETS:
            with self.subTest(target=target):
                target_report = report.target_reports[target]
                self.assertTrue(target_report["available"])
                self.assertEqual(
                    set(target_report["baselines"]),
                    {"majority", "metadata_only", "lexical", "random_latent", "no_action", "shuffled_action"},
                )
                self.assertIn("z_pred_after", target_report["views"])
                self.assertIn("accuracy_ci95", target_report["views"]["z_pred_after"]["splits"]["test"])
                self.assertFalse(report.axis_diagnostics[target]["dimension_claims_allowed"])

    def test_report_round_trips_json(self) -> None:
        rows = _fixture_rows()
        matrix = np.eye(len(rows), 4, dtype=np.float64)
        report = build_latent_probe_report(
            rows,
            embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
            baselines={
                "random_latent": matrix,
                "no_action": matrix,
                "shuffled_action": np.roll(matrix, shift=1, axis=0),
            },
            config=LatentProbeConfig(bootstrap_samples=0),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latent_probe_report.json"
            write_latent_probe_report(report, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = read_latent_probe_report(path)

        self.assertEqual(payload["schema_version"], LATENT_PROBE_REPORT_SCHEMA_VERSION)
        self.assertEqual(loaded.to_dict(), report.to_dict())

    def test_report_records_unavailable_target_when_train_has_one_class(self) -> None:
        rows = [
            _row("train-a", "train", "only"),
            _row("train-b", "train", "only"),
            _row("val-a", "val", "other"),
            _row("test-a", "test", "other"),
        ]
        matrix = np.ones((len(rows), 2), dtype=np.float64)

        report = build_latent_probe_report(
            rows,
            embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
            baselines={
                "random_latent": matrix,
                "no_action": matrix,
                "shuffled_action": matrix,
            },
        )

        self.assertFalse(report.target_reports["edit_class"]["available"])
        self.assertIn("fewer than two train labels", report.target_reports["edit_class"]["unavailable_reason"])
        self.assertEqual(report.claim_boundary["semantic_structure_status"], "not_evaluable")

    def test_boolean_false_labels_are_not_treated_as_missing(self) -> None:
        rows = (
            _passed_row("train-a", "train", True),
            _passed_row("train-b", "train", False),
            _passed_row("val-a", "val", True),
            _passed_row("val-b", "val", False),
            _passed_row("test-a", "test", True),
            _passed_row("test-b", "test", False),
        )
        matrix = np.asarray(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.1],
                [-1.0, 0.1],
                [1.0, 0.2],
                [-1.0, 0.2],
            ],
            dtype=np.float64,
        )

        report = build_latent_probe_report(
            rows,
            embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
            baselines={
                "random_latent": matrix,
                "no_action": np.zeros_like(matrix),
                "shuffled_action": np.roll(matrix, shift=1, axis=0),
            },
            config=LatentProbeConfig(
                bootstrap_samples=0,
                targets=("passed",),
            ),
        )

        passed_report = report.target_reports["passed"]
        self.assertTrue(passed_report["available"])
        self.assertEqual(
            passed_report["label_counts"],
            {
                "test": {"False": 1, "True": 1},
                "train": {"False": 1, "True": 1},
                "val": {"False": 1, "True": 1},
            },
        )
        self.assertEqual(
            passed_report["split_counts"],
            {"test": 2, "train": 2, "val": 2},
        )

    def test_report_rejects_matrix_row_mismatch(self) -> None:
        rows = _fixture_rows()
        matrix = np.zeros((len(rows) - 1, 2), dtype=np.float64)

        with self.assertRaisesRegex(LatentProbeError, "row count mismatch"):
            build_latent_probe_report(
                rows,
                embeddings={view: matrix for view in LATENT_PROBE_VIEWS},
                baselines={
                    "random_latent": matrix,
                    "no_action": matrix,
                    "shuffled_action": matrix,
                },
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


def _passed_row(transition_id: str, split: str, passed: bool) -> LatentProbeRow:
    return LatentProbeRow(
        transition_id=transition_id,
        split=split,
        labels={"passed": passed},
        metadata_features={
            "source": "synthetic",
            "output_type": "bool",
        },
        lexical_tokens=(1 if passed else 2, 10),
    )


if __name__ == "__main__":
    unittest.main()
