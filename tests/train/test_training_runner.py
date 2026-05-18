from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.data import SourceUnavailableError
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.training import (
    TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
    TrainingExecutorResult,
    TrainingRunContext,
    TrainingRunError,
    load_train_config,
    read_training_run_manifest,
    train,
    validate_train_config,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "b" * 40
CREATED_AT = "2026-05-18T00:00:00Z"


class ManifestBackedTrainingRunnerTest(unittest.TestCase):
    def test_training_runner_writes_config_metrics_checkpoints_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest = _write_dataset_parent(root)
            config = _training_config(root, dataset_manifest_path=root / "data" / "manifest.json")

            manifest = train(
                config,
                root=root,
                executor=_fixture_executor,
                command=("codelewm", "train", "--config", "fixture.yaml"),
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
            )

            run_dir = root / "runs" / "fixture"
            loaded_training = read_training_run_manifest(run_dir / "training_manifest.json")
            artifact_manifest = read_artifact_manifest(run_dir / "manifest.json")
            validate_artifact_checksums(artifact_manifest, root=run_dir)

            self.assertEqual(loaded_training.to_dict(), manifest.to_dict())
            self.assertEqual(manifest.schema_version, TRAINING_RUN_MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest.parent_artifacts, (dataset_manifest.artifact_id,))
            self.assertEqual(artifact_manifest.artifact_kind, "training_run")
            self.assertEqual(artifact_manifest.parent_artifacts, (dataset_manifest.artifact_id,))
            self.assertEqual(artifact_manifest.source_git_sha, SOURCE_SHA)
            self.assertEqual(manifest.step_count, 1)
            self.assertEqual(manifest.final_metrics["loss/total"], 0.25)
            self.assertEqual(len(manifest.checkpoint_files), 1)
            self.assertEqual(manifest.checkpoint_files[0].path, "checkpoints/checkpoint.state")

            artifact_paths = {file.path for file in artifact_manifest.files}
            self.assertIn("config.json", artifact_paths)
            self.assertIn("metrics.jsonl", artifact_paths)
            self.assertIn("reports/metrics_report.json", artifact_paths)
            self.assertIn("checkpoints/checkpoint.state", artifact_paths)
            self.assertIn("reports/collapse.json", artifact_paths)

            metrics_event = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(metrics_event["metrics"]["loss/total"], 0.25)
            saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_config["wm"]["history_size"], 1)

    def test_training_runner_refuses_missing_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "train.hdf5").write_bytes(b"train")
            (data_dir / "val.hdf5").write_bytes(b"val")
            config = _training_config(root, dataset_manifest_path=data_dir / "missing_manifest.json")

            with self.assertRaisesRegex(SourceUnavailableError, "data.manifest"):
                train(config, root=root, executor=_fixture_executor)

    def test_training_runner_refuses_overwrite_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset_parent(root)
            config = _training_config(root, dataset_manifest_path=root / "data" / "manifest.json")

            train(config, root=root, executor=_fixture_executor, source_git_sha=SOURCE_SHA, created_at=CREATED_AT)

            with self.assertRaisesRegex(TrainingRunError, "overwrite=True"):
                train(config, root=root, executor=_fixture_executor, source_git_sha=SOURCE_SHA, created_at=CREATED_AT)


def _write_dataset_parent(root: Path):
    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "train.hdf5").write_bytes(b"train")
    (data_dir / "val.hdf5").write_bytes(b"val")
    manifest = build_artifact_manifest(
        artifact_kind="dataset",
        root=data_dir,
        files=("train.hdf5", "val.hdf5"),
        command=("codelewm", "dataset", "pack"),
        config={"dataset": "fixture"},
        source_git_sha=SOURCE_SHA,
        created_at=CREATED_AT,
        artifact_id="dataset-fixture",
        metadata={"split_counts": {"train": 1, "val": 1}},
    )
    write_artifact_manifest(manifest, data_dir / "manifest.json")
    return manifest


def _training_config(root: Path, *, dataset_manifest_path: Path):
    payload = load_train_config(ROOT / "config/train/codelewm_tiny.yaml").to_dict()
    payload["name"] = "fixture_train"
    payload["data"]["train"] = str(root / "data" / "train.hdf5")
    payload["data"]["val"] = str(root / "data" / "val.hdf5")
    payload["data"]["manifest"] = str(dataset_manifest_path)
    payload["output"]["run_dir"] = str(root / "runs" / "fixture")
    payload["output"]["checkpoint_dir"] = str(root / "runs" / "fixture" / "checkpoints")
    payload["output"]["metrics_path"] = str(root / "runs" / "fixture" / "metrics.jsonl")
    payload["output"]["manifest_path"] = str(root / "runs" / "fixture" / "training_manifest.json")
    return validate_train_config(payload)


def _fixture_executor(context: TrainingRunContext) -> TrainingExecutorResult:
    checkpoint_path = context.checkpoint_dir / "checkpoint.state"
    checkpoint_path.write_bytes(b"fixture weights")
    report_path = context.run_dir / "reports" / "collapse.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('{"rank": 4.0}\n', encoding="utf-8")
    return TrainingExecutorResult(
        step_count=1,
        metrics={"loss/total": 0.25, "loss/prediction_mse": 0.20},
        checkpoint_paths=(checkpoint_path,),
        report_paths=(report_path,),
        metadata={"device": "cpu", "dtype": "float32"},
    )


if __name__ == "__main__":
    unittest.main()
