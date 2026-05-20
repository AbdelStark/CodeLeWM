from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.data import (
    ACTION_DISCRIMINATIVE_SHARD_REPORT_SCHEMA_VERSION,
    PackSpec,
    PackedTransition,
    TokenSequence,
    build_action_discriminative_shard_report,
    build_dataset_from_config_path,
    read_dataset_manifest,
    validate_action_discriminative_shard_report_payload,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "tests" / "fixtures" / "dataset_build" / "config.json"


class ActionDiscriminativeDiagnosticsTest(unittest.TestCase):
    def test_report_identifies_action_discriminative_coverage_and_round_trips(self) -> None:
        rows = (
            _transition(
                "heldout-a",
                split="val",
                before=(1, 2, 3),
                after=(1, 2, 4),
                action_abs=(30, 31),
                edit_size=4,
            ),
            _transition(
                "heldout-b",
                split="test",
                before=(1, 2, 3),
                after=(1, 2, 5),
                action_abs=(30, 31),
                edit_size=6,
            ),
            _transition(
                "train-c",
                split="train",
                before=(9, 10),
                after=(9, 11),
                action_abs=(40,),
                path="pkg/other.py",
                edit_size=20,
            ),
        )

        report = build_action_discriminative_shard_report(rows)
        loaded = validate_action_discriminative_shard_report_payload(report)

        self.assertEqual(report["schema_version"], ACTION_DISCRIMINATIVE_SHARD_REPORT_SCHEMA_VERSION)
        self.assertEqual(loaded, json.loads(json.dumps(report, sort_keys=True)))
        self.assertEqual(report["split_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(report["duplicate_pressure"]["same_before_different_after_pair_count"], 1)
        self.assertEqual(report["hard_negative_pools"]["same_before_different_after"]["pair_count"], 1)
        self.assertEqual(report["hard_negative_pools"]["same_file"]["pair_count"], 1)
        self.assertEqual(report["hard_negative_pools"]["action_cluster"]["pair_count"], 1)
        self.assertTrue(report["claim_readiness"]["positive_action_use_claim_ready"])

    def test_dataset_build_writes_manifested_action_discriminative_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"

            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=output_dir,
                command=("codelewm", "dataset", "build"),
            )

            report = json.loads(
                (output_dir / "reports" / "action_discriminative_shard_report.json").read_text(
                    encoding="utf-8"
                )
            )
            artifact_manifest = read_artifact_manifest(output_dir / "manifest.json")
            dataset_manifest = read_dataset_manifest(output_dir / "dataset_manifest.json")
            checked = validate_artifact_checksums(artifact_manifest, root=output_dir)

        self.assertEqual(report["schema_version"], ACTION_DISCRIMINATIVE_SHARD_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["row_count"], 3)
        self.assertFalse(report["claim_readiness"]["positive_action_use_claim_ready"])
        self.assertIn("insufficient_heldout_rows", report["claim_readiness"]["failure_reasons"])
        self.assertIn(
            "reports/action_discriminative_shard_report.json",
            {artifact.path for artifact in dataset_manifest.artifacts},
        )
        self.assertEqual(
            artifact_manifest.metadata["action_discriminative_shard_report"],
            "reports/action_discriminative_shard_report.json",
        )
        self.assertEqual(len(checked), len(artifact_manifest.files))


def _transition(
    transition_id: str,
    *,
    split: str,
    before: tuple[int, ...],
    after: tuple[int, ...],
    action_abs: tuple[int, ...],
    path: str = "pkg/mod.py",
    edit_size: int = 1,
) -> PackedTransition:
    return PackedTransition(
        transition_id=transition_id,
        source="local_repo",
        repo="example/repo",
        commit=transition_id,
        path=path,
        split=split,  # type: ignore[arg-type]
        state_before=TokenSequence(input_ids=before),
        state_after=TokenSequence(input_ids=after),
        action_text=TokenSequence(input_ids=(100, 101)),
        action_abs=TokenSequence(input_ids=action_abs),
        edit_size=edit_size,
        license="mit",
        dedup_keys=("diff_shape:test-shape",),
    )


if __name__ == "__main__":
    unittest.main()
