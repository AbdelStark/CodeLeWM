from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codelewm.model import (
    CheckpointMetadata,
    LATENT_DIM,
    build_checkpoint_metadata,
    compute_config_hash,
    write_checkpoint_manifest,
)
from codelewm.observability import (
    build_artifact_manifest,
    write_artifact_manifest,
)
from codelewm.training import (
    CheckpointResumePlan,
    TrainingExecutorResult,
    TrainingRunContext,
    TrainingRunError,
    compatibility_config_payload,
    load_train_config,
    prepare_checkpoint_resume,
    read_training_run_manifest,
    train,
    validate_train_config,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "b" * 40
CREATED_AT = "2026-05-18T00:00:00Z"
PARENT_CREATED_AT = "2026-05-17T00:00:00Z"


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


def _training_config(root: Path, *, dataset_manifest_path: Path, run_name: str = "fixture_train"):
    payload = load_train_config(ROOT / "config/train/codelewm_tiny.yaml").to_dict()
    payload["name"] = run_name
    payload["data"]["train"] = str(root / "data" / "train.hdf5")
    payload["data"]["val"] = str(root / "data" / "val.hdf5")
    payload["data"]["manifest"] = str(dataset_manifest_path)
    payload["output"]["run_dir"] = str(root / "runs" / run_name)
    payload["output"]["checkpoint_dir"] = str(root / "runs" / run_name / "checkpoints")
    payload["output"]["metrics_path"] = str(root / "runs" / run_name / "metrics.jsonl")
    payload["output"]["manifest_path"] = str(root / "runs" / run_name / "training_manifest.json")
    return validate_train_config(payload)


def _make_resume_capable_executor(
    *,
    metadata_config_dict: dict | None = None,
    action_view: str | None = None,
    latent_dim: int | None = None,
    record_schema_version: str | None = None,
):
    """Build an executor that writes both a checkpoint file and its manifest.

    Tests pass overrides to construct a parent whose checkpoint manifest
    intentionally disagrees with the new config so the resume validator
    can reject it.
    """

    def _executor(context: TrainingRunContext) -> TrainingExecutorResult:
        checkpoint_path = context.checkpoint_dir / "checkpoint.state"
        checkpoint_path.write_bytes(b"fixture weights")
        if metadata_config_dict is None:
            metadata = build_checkpoint_metadata(
                compatibility_config_payload(context.config),
                action_view=action_view or context.config.wm.action_view,
                latent_dim=latent_dim or context.config.wm.embed_dim,
                record_schema_version=record_schema_version or "codelewm.transition.v1",
            )
        else:
            metadata = CheckpointMetadata(
                config_hash=compute_config_hash(metadata_config_dict),
                action_view=action_view or context.config.wm.action_view,
                latent_dim=latent_dim or context.config.wm.embed_dim,
                record_schema_version=record_schema_version or "codelewm.transition.v1",
            )
        manifest_path = checkpoint_path.with_name(checkpoint_path.name + ".manifest.json")
        write_checkpoint_manifest(
            metadata=metadata,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
        )
        report_path = context.run_dir / "reports" / "collapse.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text('{"rank": 4.0}\n', encoding="utf-8")
        return TrainingExecutorResult(
            step_count=1,
            metrics={"loss/total": 0.25},
            checkpoint_paths=(checkpoint_path, manifest_path),
            report_paths=(report_path,),
            metadata={"device": "cpu"},
        )

    return _executor


def _run_parent(root: Path, **executor_kwargs) -> Path:
    _write_dataset_parent(root)
    config = _training_config(root, dataset_manifest_path=root / "data" / "manifest.json", run_name="parent")
    train(
        config,
        root=root,
        executor=_make_resume_capable_executor(**executor_kwargs),
        source_git_sha=SOURCE_SHA,
        created_at=PARENT_CREATED_AT,
    )
    return root / "runs" / "parent" / "training_manifest.json"


class CompatibleResumeTest(unittest.TestCase):
    def test_resume_records_parent_artifact_in_new_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_manifest_path = _run_parent(root)
            parent_manifest = read_training_run_manifest(parent_manifest_path)
            parent_checkpoint_sha256 = _read_parent_checkpoint_sha256(root, parent_manifest)
            child_config = _training_config(
                root,
                dataset_manifest_path=root / "data" / "manifest.json",
                run_name="child",
            )

            child_manifest = train(
                child_config,
                root=root,
                executor=_make_resume_capable_executor(),
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
                resume_from=parent_manifest_path,
            )
            resume_meta = child_manifest.metadata.get("resume")

        self.assertIn(parent_manifest.artifact_manifest_id, child_manifest.parent_artifacts)
        self.assertIn("dataset-fixture", child_manifest.parent_artifacts)
        self.assertIsInstance(resume_meta, dict)
        self.assertEqual(resume_meta["parent_training_artifact_id"], parent_manifest.artifact_manifest_id)
        self.assertEqual(resume_meta["parent_step_count"], parent_manifest.step_count)
        self.assertEqual(resume_meta["parent_checkpoint_sha256"], parent_checkpoint_sha256)

    def test_prepare_checkpoint_resume_returns_validated_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_manifest_path = _run_parent(root)
            child_config = _training_config(
                root,
                dataset_manifest_path=root / "data" / "manifest.json",
                run_name="child",
            )

            plan = prepare_checkpoint_resume(
                parent_manifest_path,
                config=child_config,
                root=root,
            )

            self.assertIsInstance(plan, CheckpointResumePlan)
            self.assertEqual(plan.parent_training_manifest_path, parent_manifest_path)
            self.assertTrue(plan.parent_checkpoint_path.is_file())
            self.assertTrue(plan.parent_checkpoint_manifest_path.is_file())


class IncompatibleResumeTest(unittest.TestCase):
    def test_resume_rejected_when_config_hash_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_manifest_path = _run_parent(
                root,
                metadata_config_dict={"unrelated": "config"},
            )
            child_config = _training_config(
                root,
                dataset_manifest_path=root / "data" / "manifest.json",
                run_name="child",
            )

            with self.assertRaisesRegex(TrainingRunError, "config_hash"):
                prepare_checkpoint_resume(
                    parent_manifest_path,
                    config=child_config,
                    root=root,
                )

    def test_resume_rejected_when_latent_dim_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_manifest_path = _run_parent(root, latent_dim=LATENT_DIM + 16)
            child_config = _training_config(
                root,
                dataset_manifest_path=root / "data" / "manifest.json",
                run_name="child",
            )

            with self.assertRaisesRegex(TrainingRunError, "latent"):
                prepare_checkpoint_resume(
                    parent_manifest_path,
                    config=child_config,
                    root=root,
                )

    def test_resume_rejected_when_action_view_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_manifest_path = _run_parent(root, action_view="abstract")
            child_config = _training_config(
                root,
                dataset_manifest_path=root / "data" / "manifest.json",
                run_name="child",
            )

            with self.assertRaisesRegex(TrainingRunError, "action_view"):
                prepare_checkpoint_resume(
                    parent_manifest_path,
                    config=child_config,
                    root=root,
                )

    def test_resume_rejected_when_parent_training_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset_parent(root)
            child_config = _training_config(
                root,
                dataset_manifest_path=root / "data" / "manifest.json",
                run_name="child",
            )
            missing_path = root / "runs" / "missing" / "training_manifest.json"

            with self.assertRaisesRegex(TrainingRunError, "does not exist"):
                prepare_checkpoint_resume(
                    missing_path,
                    config=child_config,
                    root=root,
                )

    def test_resume_rejected_when_parent_checkpoint_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_manifest_path = _run_parent(root)
            parent_run_dir = parent_manifest_path.parent
            # Tampering with the checkpoint file breaks the parent artifact
            # manifest checksum, which the resume preparation must catch.
            checkpoint_path = parent_run_dir / "checkpoints" / "checkpoint.state"
            checkpoint_path.write_bytes(b"tampered")
            child_config = _training_config(
                root,
                dataset_manifest_path=root / "data" / "manifest.json",
                run_name="child",
            )

            with self.assertRaisesRegex(TrainingRunError, "mismatch"):
                prepare_checkpoint_resume(
                    parent_manifest_path,
                    config=child_config,
                    root=root,
                )


class TrainingRunnerResumeIntegrationTest(unittest.TestCase):
    def test_train_with_resume_from_writes_parent_lineage_in_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_manifest_path = _run_parent(root)
            parent_manifest = read_training_run_manifest(parent_manifest_path)
            child_config = _training_config(
                root,
                dataset_manifest_path=root / "data" / "manifest.json",
                run_name="child",
            )

            train(
                child_config,
                root=root,
                executor=_make_resume_capable_executor(),
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
                resume_from=parent_manifest_path,
            )
            from codelewm.observability import read_artifact_manifest

            child_artifact_manifest = read_artifact_manifest(
                root / "runs" / "child" / "manifest.json"
            )

        self.assertIn(
            parent_manifest.artifact_manifest_id,
            child_artifact_manifest.parent_artifacts,
        )
        self.assertIn("dataset-fixture", child_artifact_manifest.parent_artifacts)
        resume_meta = child_artifact_manifest.metadata.get("resume")
        self.assertIsInstance(resume_meta, dict)
        self.assertEqual(
            resume_meta["parent_training_artifact_id"],
            parent_manifest.artifact_manifest_id,
        )


def _read_parent_checkpoint_sha256(root: Path, parent_manifest) -> str:
    from codelewm.model.checkpoint import read_checkpoint_manifest

    parent_run_dir = root / "runs" / "parent"
    checkpoint_manifest_path = (
        parent_run_dir / parent_manifest.checkpoint_files[0].path
    )
    checkpoint_manifest_path = checkpoint_manifest_path.with_name(
        checkpoint_manifest_path.name + ".manifest.json"
    )
    return read_checkpoint_manifest(checkpoint_manifest_path).checkpoint_sha256


if __name__ == "__main__":
    unittest.main()
