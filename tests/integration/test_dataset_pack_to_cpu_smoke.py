from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from codelewm.data import build_dataset_from_config_path, pack_dataset_from_manifest
from codelewm.training import load_train_config, train_cpu_smoke, validate_train_config


DATA_DEPS_AVAILABLE = (
    importlib.util.find_spec("h5py") is not None
    and importlib.util.find_spec("pyarrow") is not None
)
ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "tests" / "fixtures" / "dataset_build" / "config.json"


@unittest.skipUnless(DATA_DEPS_AVAILABLE, "h5py and pyarrow are not installed")
class DatasetPackToCpuSmokeTest(unittest.TestCase):
    def test_packed_fixture_dataset_can_run_cpu_smoke_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            payload = load_train_config(ROOT / "config" / "train" / "codelewm_tiny.yaml").to_dict()
            payload["name"] = "packed_fixture_cpu_smoke"
            payload["data"]["train"] = str(pack_dir / "hdf5" / "train.hdf5")
            payload["data"]["val"] = str(pack_dir / "hdf5" / "val.hdf5")
            payload["data"]["manifest"] = str(pack_dir / "manifest.json")
            payload["trainer"]["max_steps"] = 2
            payload["output"]["run_dir"] = str(root / "runs" / "packed_fixture")
            payload["output"]["checkpoint_dir"] = str(root / "runs" / "packed_fixture" / "checkpoints")
            payload["output"]["metrics_path"] = str(root / "runs" / "packed_fixture" / "metrics.jsonl")
            payload["output"]["manifest_path"] = str(root / "runs" / "packed_fixture" / "training_manifest.json")

            manifest = train_cpu_smoke(validate_train_config(payload), root=root)

        self.assertGreater(manifest.final_metrics["embedding/variance"], 0.0)
        self.assertEqual(manifest.step_count, 2)
        self.assertEqual(len(manifest.parent_artifacts), 1)


if __name__ == "__main__":
    unittest.main()
