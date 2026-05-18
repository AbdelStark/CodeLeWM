from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.training import (
    CPU_SMOKE_CHECKPOINT_SCHEMA_VERSION,
    load_train_config,
    train_cpu_smoke,
    validate_train_config,
)


H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None
ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "c" * 40
CREATED_AT = "2026-05-18T00:00:00Z"


@unittest.skipUnless(H5PY_AVAILABLE, "h5py is not installed")
class CpuTrainSmokeTest(unittest.TestCase):
    def test_cpu_smoke_training_writes_finite_loss_and_nonzero_variance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_fixture_hdf5(data_dir / "train.hdf5")
            _write_fixture_hdf5(data_dir / "val.hdf5")
            dataset_manifest = build_artifact_manifest(
                artifact_kind="dataset",
                root=data_dir,
                files=("train.hdf5", "val.hdf5"),
                command=("codelewm", "dataset", "pack"),
                config={"dataset": "cpu-smoke-fixture"},
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
                artifact_id="dataset-cpu-smoke",
                metadata={"rows": 4},
            )
            write_artifact_manifest(dataset_manifest, data_dir / "manifest.json")
            config = _smoke_config(root)

            manifest = train_cpu_smoke(
                config,
                root=root,
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
            )

            run_dir = root / "runs" / "cpu_smoke"
            artifact_manifest = read_artifact_manifest(run_dir / "manifest.json")
            validate_artifact_checksums(artifact_manifest, root=run_dir)

            self.assertEqual(manifest.parent_artifacts, (dataset_manifest.artifact_id,))
            self.assertTrue(math.isfinite(manifest.final_metrics["loss/total"]))
            self.assertGreater(manifest.final_metrics["embedding/variance"], 0.0)
            self.assertGreaterEqual(manifest.step_count, 1)
            checkpoint_path = run_dir / manifest.checkpoint_files[0].path
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["schema_version"], CPU_SMOKE_CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(checkpoint["run_id"], "cpu_smoke_fixture")


def _smoke_config(root: Path):
    payload = load_train_config(ROOT / "config/train/codelewm_tiny.yaml").to_dict()
    payload["name"] = "cpu_smoke_fixture"
    payload["data"]["train"] = str(root / "data" / "train.hdf5")
    payload["data"]["val"] = str(root / "data" / "val.hdf5")
    payload["data"]["manifest"] = str(root / "data" / "manifest.json")
    payload["trainer"]["max_steps"] = 3
    payload["output"]["run_dir"] = str(root / "runs" / "cpu_smoke")
    payload["output"]["checkpoint_dir"] = str(root / "runs" / "cpu_smoke" / "checkpoints")
    payload["output"]["metrics_path"] = str(root / "runs" / "cpu_smoke" / "metrics.jsonl")
    payload["output"]["manifest_path"] = str(root / "runs" / "cpu_smoke" / "training_manifest.json")
    return validate_train_config(payload)


def _write_fixture_hdf5(path: Path) -> None:
    import h5py

    state_before = np.zeros((4, 1024), dtype=np.int32)
    state_after = np.zeros((4, 1024), dtype=np.int32)
    action_text = np.zeros((4, 256), dtype=np.int32)
    action_abs = np.zeros((4, 192), dtype=np.int32)
    for row in range(4):
        base = row * 10
        state_before[row, :5] = np.array([base + 1, base + 2, base + 3, base + 4, base + 5])
        state_after[row, :5] = state_before[row, :5] + np.array([1, 0, 1, 0, 1])
        action_text[row, :3] = np.array([100 + row, 110 + row, 120 + row])
        action_abs[row, :2] = np.array([10 + row, 20 + row])
    with h5py.File(path, "w") as handle:
        handle.create_group("state_before").create_dataset("input_ids", data=state_before)
        handle.create_group("state_after").create_dataset("input_ids", data=state_after)
        handle.create_group("action_text").create_dataset("input_ids", data=action_text)
        handle.create_group("action_abs").create_dataset("input_ids", data=action_abs)


if __name__ == "__main__":
    unittest.main()
