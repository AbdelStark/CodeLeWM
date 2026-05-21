from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.data import build_dataset_from_config_path, pack_dataset_from_manifest
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums
from codelewm.training import TRAINING_RUN_MANIFEST_SCHEMA_VERSION, load_train_config


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


@unittest.skipUnless(TORCH_RUNTIME_AVAILABLE and DATA_DEPS_AVAILABLE, "torch/einops/h5py/pyarrow are not installed")
class TrainCliTorchExecutorTest(unittest.TestCase):
    def test_train_cli_runs_tiny_fixture_and_writes_manifested_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = _build_and_pack_fixture(root)
            config_path = _write_train_config(root, pack_dir)
            run_dir = root / "runs" / "cli_torch"
            log_path = root / "train.jsonl"

            completed = _run_cli(
                "train",
                "--config",
                str(config_path),
                "--out",
                str(run_dir),
                "--executor",
                "torch",
                "--device",
                "cpu",
                "--json",
                "--log-jsonl",
                str(log_path),
            )

            payload = json.loads(completed.stdout)
            artifact_manifest = read_artifact_manifest(run_dir / "manifest.json")
            validate_artifact_checksums(artifact_manifest, root=run_dir)
            log_events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(payload["schema_version"], TRAINING_RUN_MANIFEST_SCHEMA_VERSION)
            self.assertEqual(payload["step_count"], 1)
            self.assertEqual(payload["parent_artifacts"], [artifact_manifest.parent_artifacts[0]])
            self.assertTrue((run_dir / "training_manifest.json").is_file())
            self.assertTrue((run_dir / "metrics.jsonl").is_file())
            self.assertTrue((run_dir / "reports" / "torch_training_report.json").is_file())
            self.assertTrue((run_dir / "checkpoints" / "checkpoint.pt").is_file())
            self.assertTrue((run_dir / "checkpoints" / "checkpoint.pt.manifest.json").is_file())
            self.assertEqual([event["event"] for event in log_events], ["training.start", "training.complete"])

    def test_train_cli_surfaces_resume_errors_as_checkpoint_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = _build_and_pack_fixture(root)
            config_path = _write_train_config(root, pack_dir)
            run_dir = root / "runs" / "resume_error"

            completed = _run_cli(
                "train",
                "--config",
                str(config_path),
                "--out",
                str(run_dir),
                "--executor",
                "torch",
                "--device",
                "cpu",
                "--resume-from",
                str(root / "missing-training-manifest.json"),
                "--json",
            )
            payload = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 5, completed.stderr)
        self.assertEqual(payload["schema_version"], "codelewm.error.v1")
        self.assertEqual(payload["error_type"], "checkpoint_error")


class TrainCliArgumentTest(unittest.TestCase):
    def test_cpu_smoke_rejects_non_cpu_device_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "missing.json"
            config_path.write_text("{}", encoding="utf-8")

            completed = _run_cli(
                "train",
                "--config",
                str(config_path),
                "--executor",
                "cpu-smoke",
                "--device",
                "cuda",
                "--json",
            )
            payload = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(payload["error_type"], "config_error")

    def test_cpu_smoke_rejects_tensorboard_export_before_runtime(self) -> None:
        completed = _run_cli(
            "train",
            "--config",
            str(ROOT / "config" / "train" / "codelewm_tiny.yaml"),
            "--executor",
            "cpu-smoke",
            "--tensorboard",
            "--json",
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(payload["error_type"], "config_error")
        self.assertIn("TensorBoard", payload["message"])


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


def _write_train_config(root: Path, pack_dir: Path) -> Path:
    payload = load_train_config(ROOT / "config" / "train" / "codelewm_tiny.yaml").to_dict()
    payload["name"] = "cli_torch_fixture"
    payload["data"]["train"] = str(pack_dir / "hdf5" / "train.hdf5")
    payload["data"]["val"] = str(pack_dir / "hdf5" / "val.hdf5")
    payload["data"]["manifest"] = str(pack_dir / "manifest.json")
    payload["trainer"]["max_steps"] = 1
    payload["loader"]["batch_size"] = 2
    payload["loader"]["shuffle"] = False
    config_path = root / "train_config.json"
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "codelewm.harness.cli", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    unittest.main()
