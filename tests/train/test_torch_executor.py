from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codelewm.data import build_dataset_from_config_path, pack_dataset_from_manifest
from codelewm.model import (
    AbstractActionEncoder,
    CheckpointCompatibilitySpec,
    TextActionEncoder,
    build_torch_transition_model,
    compute_config_hash,
    load_checkpoint_manifest,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums
from codelewm.training import (
    DEFAULT_TRAINING_VOCAB_SIZE,
    TENSORBOARD_EXPORT_SCHEMA_VERSION,
    TORCH_CHECKPOINT_SCHEMA_VERSION,
    TORCH_TRAINING_REPORT_SCHEMA_VERSION,
    PackedTransitionHdf5Dataset,
    TensorBoardExportResult,
    TrainingRunError,
    compatibility_config_payload,
    load_train_config,
    train_torch,
    validate_train_config,
)


TORCH_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("einops") is not None
)
DATA_DEPS_AVAILABLE = (
    importlib.util.find_spec("h5py") is not None
    and importlib.util.find_spec("pyarrow") is not None
)
ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "tests" / "fixtures" / "dataset_build" / "config.json"
SOURCE_SHA = "d" * 40
CREATED_AT = "2026-05-19T00:00:00Z"


@unittest.skipUnless(TORCH_RUNTIME_AVAILABLE and DATA_DEPS_AVAILABLE, "torch/einops/h5py/pyarrow are not installed")
class PackedTransitionHdf5DatasetTest(unittest.TestCase):
    def test_loader_emits_expected_shapes_and_buckets_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pack_dir = _build_and_pack_fixture(Path(tmp))
            dataset = PackedTransitionHdf5Dataset(
                pack_dir / "hdf5" / "train.hdf5",
                action_view="text",
            )

            item = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(tuple(item["state_before"]["input_ids"].shape), (1024,))
        self.assertEqual(tuple(item["state_after"]["input_ids"].shape), (1024,))
        self.assertEqual(tuple(item["action"]["input_ids"].shape), (256,))
        self.assertLess(int(item["state_before"]["input_ids"].max()), DEFAULT_TRAINING_VOCAB_SIZE)
        self.assertLess(int(item["action"]["input_ids"].max()), DEFAULT_TRAINING_VOCAB_SIZE)

    def test_loader_selects_abstract_action_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pack_dir = _build_and_pack_fixture(Path(tmp))
            dataset = PackedTransitionHdf5Dataset(
                pack_dir / "hdf5" / "train.hdf5",
                action_view="abstract",
            )

            item = dataset[0]

        self.assertEqual(tuple(item["action"]["input_ids"].shape), (192,))

    def test_loader_rejects_patch_action_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pack_dir = _build_and_pack_fixture(Path(tmp))

            with self.assertRaisesRegex(TrainingRunError, "patch action"):
                PackedTransitionHdf5Dataset(
                    pack_dir / "hdf5" / "train.hdf5",
                    action_view="patch",
                )

    def test_loader_rejects_incompatible_hdf5_schema(self) -> None:
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.hdf5"
            with h5py.File(path, "w") as handle:
                handle.attrs["schema_version"] = "codelewm.transition.v0"
                handle.attrs["row_count"] = 1
                for group_name in ("state_before", "state_after"):
                    group = handle.create_group(group_name)
                    group.create_dataset("input_ids", data=np.zeros((1, 1024), dtype=np.int32))
                    group.create_dataset("attention_mask", data=np.zeros((1, 1024), dtype=bool))
                    group.create_dataset("segment_ids", data=np.zeros((1, 1024), dtype=np.int16))
                    group.create_dataset("changed_hunk_mask", data=np.zeros((1, 1024), dtype=bool))
                action = handle.create_group("action_text")
                action.create_dataset("input_ids", data=np.zeros((1, 256), dtype=np.int32))
                action.create_dataset("attention_mask", data=np.zeros((1, 256), dtype=bool))

            with self.assertRaisesRegex(TrainingRunError, "schema_version"):
                PackedTransitionHdf5Dataset(path)


@unittest.skipUnless(TORCH_RUNTIME_AVAILABLE and DATA_DEPS_AVAILABLE, "torch/einops/h5py/pyarrow are not installed")
class TorchTrainingExecutorTest(unittest.TestCase):
    def test_action_view_selects_matching_encoder(self) -> None:
        text_config = _train_config_payload(Path("/tmp"), Path("/tmp/pack"))
        text_model = build_torch_transition_model(
            _model_config_from_payload(text_config)
        )
        abstract_payload = _train_config_payload(Path("/tmp"), Path("/tmp/pack"))
        abstract_payload["wm"]["action_view"] = "abstract"
        abstract_payload["wm"]["action_sequence_length"] = 192
        abstract_model = build_torch_transition_model(
            _model_config_from_payload(abstract_payload)
        )

        self.assertIsInstance(text_model.action_encoder, TextActionEncoder)
        self.assertIsInstance(abstract_model.action_encoder, AbstractActionEncoder)

    def test_tiny_packed_fixture_runs_one_torch_step_and_writes_checkpoint_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = _build_and_pack_fixture(root)
            config = validate_train_config(_train_config_payload(root, pack_dir))

            manifest = train_torch(
                config,
                root=root,
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
            )

            run_dir = root / "runs" / "torch_fixture"
            artifact_manifest = read_artifact_manifest(run_dir / "manifest.json")
            validate_artifact_checksums(artifact_manifest, root=run_dir)
            checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
            checkpoint_manifest_path = checkpoint.with_name(checkpoint.name + ".manifest.json")
            checkpoint_manifest = load_checkpoint_manifest(
                checkpoint_manifest_path,
                expected=CheckpointCompatibilitySpec(
                    config_hash=compute_config_hash(compatibility_config_payload(config)),
                    action_view="text",
                ),
            )
            report = json.loads((run_dir / "reports" / "torch_training_report.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest.step_count, 1)
        self.assertEqual(report["schema_version"], TORCH_TRAINING_REPORT_SCHEMA_VERSION)
        self.assertEqual(checkpoint_manifest.metadata.action_view, "text")
        self.assertEqual(checkpoint_manifest.metadata.model_class, "TorchCodeTransitionModel")
        self.assertEqual(checkpoint_manifest.metadata.schema_version, "codelewm.checkpoint.v1")
        self.assertEqual(report["metrics"]["collapse/latent_dim"], 256.0)
        self.assertTrue(math.isfinite(manifest.final_metrics["loss/total"]))
        self.assertGreater(manifest.final_metrics["collapse/per_dim_variance_max"], 0.0)
        self.assertIn("checkpoints/checkpoint.pt", {item.path for item in manifest.checkpoint_files})

    def test_tiny_packed_fixture_can_manifest_tensorboard_export_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = _build_and_pack_fixture(root)
            payload = _train_config_payload(root, pack_dir, name="torch_tensorboard_fixture")
            config = validate_train_config(payload)

            with mock.patch(
                "codelewm.training.torch_executor.export_tensorboard_training_run",
                side_effect=_fake_tensorboard_export,
            ):
                manifest = train_torch(
                    config,
                    root=root,
                    source_git_sha=SOURCE_SHA,
                    created_at=CREATED_AT,
                    tensorboard=True,
                    tensorboard_dir="tb",
                )

            run_dir = root / "runs" / "torch_tensorboard_fixture"
            artifact_manifest = read_artifact_manifest(run_dir / "manifest.json")
            report = json.loads((run_dir / "reports" / "torch_training_report.json").read_text(encoding="utf-8"))
            artifact_paths = {item.path for item in artifact_manifest.files}

        self.assertEqual(manifest.metadata["executor"]["tensorboard_export"]["schema_version"], TENSORBOARD_EXPORT_SCHEMA_VERSION)
        self.assertEqual(report["tensorboard_export"]["schema_version"], TENSORBOARD_EXPORT_SCHEMA_VERSION)
        self.assertIn("reports/tensorboard_export.json", artifact_paths)
        self.assertIn("tb/events.out.tfevents.fixture", artifact_paths)

    def test_action_use_margin_objective_runs_one_torch_step_and_records_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = _build_and_pack_fixture(root)
            payload = _train_config_payload(root, pack_dir, name="torch_action_margin_fixture")
            payload["loss"]["enable_action_use_margin"] = True
            payload["loss"]["action_use_margin_weight"] = 0.25
            payload["loss"]["action_use_margin"] = 0.02
            config = validate_train_config(payload)

            manifest = train_torch(
                config,
                root=root,
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
            )

            run_dir = root / "runs" / "torch_action_margin_fixture"
            report = json.loads((run_dir / "reports" / "torch_training_report.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest.step_count, 1)
        self.assertTrue(math.isfinite(manifest.final_metrics["loss/action_use_margin"]))
        self.assertTrue(math.isfinite(manifest.final_metrics["val/loss/action_use_margin"]))
        self.assertTrue(report["objective"]["enable_action_use_margin"])
        self.assertEqual(report["objective"]["action_use_margin_weight"], 0.25)
        self.assertEqual(report["objective"]["action_use_margin"], 0.02)
        self.assertTrue(manifest.metadata["executor"]["objective"]["enable_action_use_margin"])

    def test_action_swap_inverse_objectives_run_one_torch_step_and_record_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = _build_and_pack_fixture(root)
            payload = _train_config_payload(root, pack_dir, name="torch_action_swap_inverse_fixture")
            payload["wm"]["action_fusion"] = "gated_residual"
            payload["loss"]["enable_action_swap_contrastive"] = True
            payload["loss"]["action_swap_contrastive_weight"] = 0.20
            payload["loss"]["action_swap_contrastive_margin"] = 0.05
            payload["loss"]["enable_inverse_action_reconstruction"] = True
            payload["loss"]["inverse_action_reconstruction_weight"] = 0.10
            config = validate_train_config(payload)

            manifest = train_torch(
                config,
                root=root,
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
            )

            run_dir = root / "runs" / "torch_action_swap_inverse_fixture"
            report = json.loads((run_dir / "reports" / "torch_training_report.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest.step_count, 1)
        self.assertTrue(math.isfinite(manifest.final_metrics["loss/action_swap_contrastive"]))
        self.assertTrue(math.isfinite(manifest.final_metrics["loss/inverse_action_reconstruction"]))
        self.assertTrue(math.isfinite(manifest.final_metrics["action_diagnostics/swap_distance_gap"]))
        self.assertTrue(math.isfinite(manifest.final_metrics["val/loss/action_swap_contrastive"]))
        self.assertTrue(math.isfinite(manifest.final_metrics["val/loss/inverse_action_reconstruction"]))
        self.assertEqual(report["objective"]["action_swap_contrastive_weight"], 0.20)
        self.assertEqual(report["objective"]["action_swap_contrastive_margin"], 0.05)
        self.assertTrue(report["objective"]["enable_inverse_action_reconstruction"])
        self.assertTrue(manifest.metadata["executor"]["objective"]["enable_action_swap_contrastive"])

    def test_p_pass_bce_is_rejected_for_legacy_hdf5_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = _build_and_pack_fixture(root)
            payload = _train_config_payload(root, pack_dir, name="torch_p_pass_fixture")
            payload["loss"]["enable_p_pass_bce"] = True
            payload["loss"]["p_pass_bce_weight"] = 0.25
            config = validate_train_config(payload)

            with self.assertRaisesRegex(TrainingRunError, "pass/fail labels"):
                train_torch(
                    config,
                    root=root,
                    source_git_sha=SOURCE_SHA,
                    created_at=CREATED_AT,
                )


def _build_and_pack_fixture(root: Path) -> Path:
    build_dir = root / "build"
    pack_dir = root / "pack"
    build_dataset_from_config_path(
        config_path=FIXTURE_CONFIG,
        output_dir=build_dir,
        command=("codelewm", "dataset", "build"),
    )
    pack_dataset_from_manifest(
        manifest_path=build_dir / "manifest.json",
        output_dir=pack_dir,
        command=("codelewm", "dataset", "pack"),
    )
    return pack_dir


def _fake_tensorboard_export(**kwargs) -> TensorBoardExportResult:
    run_dir = Path(kwargs["run_dir"])
    log_dir = run_dir / Path(kwargs["log_dir"] or "tensorboard")
    log_dir.mkdir(parents=True, exist_ok=True)
    event_path = log_dir / "events.out.tfevents.fixture"
    event_path.write_bytes(b"fixture tensorboard event\n")
    report_path = run_dir / "reports" / "tensorboard_export.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": TENSORBOARD_EXPORT_SCHEMA_VERSION,
                "run_id": kwargs["run_id"],
                "step_count": kwargs["step_count"],
                "event_files": [{"path": "tb/events.out.tfevents.fixture"}],
                "scalar_tags": ["loss/total"],
                "histogram_tags": ["parameters/encoder.weight"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return TensorBoardExportResult(
        report_path=report_path,
        event_files=(event_path,),
        scalar_tags=("loss/total",),
        histogram_tags=("parameters/encoder.weight",),
    )


def _train_config_payload(root: Path, pack_dir: Path, *, name: str = "torch_fixture") -> dict:
    payload = load_train_config(ROOT / "config" / "train" / "codelewm_tiny.yaml").to_dict()
    payload["name"] = name
    payload["data"]["train"] = str(pack_dir / "hdf5" / "train.hdf5")
    payload["data"]["val"] = str(pack_dir / "hdf5" / "val.hdf5")
    payload["data"]["manifest"] = str(pack_dir / "manifest.json")
    payload["trainer"]["max_steps"] = 1
    payload["loader"]["batch_size"] = 2
    payload["loader"]["shuffle"] = False
    payload["output"]["run_dir"] = str(root / "runs" / name)
    payload["output"]["checkpoint_dir"] = str(root / "runs" / name / "checkpoints")
    payload["output"]["metrics_path"] = str(root / "runs" / name / "metrics.jsonl")
    payload["output"]["manifest_path"] = str(root / "runs" / name / "training_manifest.json")
    return payload


def _model_config_from_payload(payload: dict):
    from codelewm.model import TorchCodeTransitionModelConfig

    config = validate_train_config(payload)
    return TorchCodeTransitionModelConfig(
        action_view=config.wm.action_view,
        latent_dim=config.wm.embed_dim,
        state_sequence_length=config.wm.state_sequence_length,
        action_sequence_length=config.wm.action_sequence_length,
        dropout=0.0,
    )


if __name__ == "__main__":
    unittest.main()
