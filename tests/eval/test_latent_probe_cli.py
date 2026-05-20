from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.data import build_dataset_from_config_path, pack_dataset_from_manifest
from codelewm.eval import LATENT_PROBE_EVAL_RUN_SCHEMA_VERSION, LATENT_PROBE_REPORT_SCHEMA_VERSION
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums
from codelewm.training import load_train_config, train_torch


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
class LatentProbeCliTest(unittest.TestCase):
    def test_latent_probe_cli_writes_report_and_manifested_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = _build_and_pack_fixture(root)
            train_run_dir = _train_tiny_fixture(root, pack_dir)
            checkpoint = train_run_dir / "checkpoints" / "checkpoint.pt"
            out_dir = root / "runs" / "latent-probe"
            log_path = root / "latent-probe.jsonl"

            completed = _run_cli(
                "eval",
                "latent-probe",
                "--checkpoint",
                str(checkpoint),
                "--data",
                str(pack_dir),
                "--out",
                str(out_dir),
                "--max-examples-per-split",
                "4",
                "--bootstrap-samples",
                "4",
                "--json",
                "--log-jsonl",
                str(log_path),
            )

            payload = json.loads(completed.stdout)
            report = json.loads((out_dir / "reports" / "latent_probe_report.json").read_text(encoding="utf-8"))
            artifact_manifest = read_artifact_manifest(out_dir / "manifest.json")
            checked_files = validate_artifact_checksums(artifact_manifest, root=out_dir)
            log_events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["schema_version"], LATENT_PROBE_EVAL_RUN_SCHEMA_VERSION)
        self.assertEqual(payload["artifact_manifest_path"], "manifest.json")
        self.assertEqual(report["schema_version"], LATENT_PROBE_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["config"]["targets"], [
            "edit_class",
            "ast_node_kind",
            "symbol_kind",
            "edit_size_bucket",
            "action_cluster",
            "source_family",
        ])
        self.assertIn("edit_size_bucket", report["target_reports"])
        self.assertIn("axis_diagnostics", report)
        self.assertIn("unavailable_reason", report["target_reports"]["edit_size_bucket"])
        self.assertFalse(report["claim_boundary"]["positive_representation_claim_allowed"])
        self.assertEqual(artifact_manifest.artifact_kind, "eval_report")
        self.assertEqual(
            {path.name for path in checked_files},
            {"config.json", "latent_probe_report.json"},
        )
        self.assertEqual(
            [event["event"] for event in log_events],
            ["evaluation.latent_probe.start", "evaluation.latent_probe.complete"],
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


def _train_tiny_fixture(root: Path, pack_dir: Path) -> Path:
    payload = load_train_config(ROOT / "config" / "train" / "codelewm_tiny.yaml").to_dict()
    run_dir = root / "runs" / "train"
    payload["name"] = "latent_probe_cli_fixture"
    payload["data"]["train"] = str(pack_dir / "hdf5" / "train.hdf5")
    payload["data"]["val"] = str(pack_dir / "hdf5" / "val.hdf5")
    payload["data"]["manifest"] = str(pack_dir / "manifest.json")
    payload["output"]["run_dir"] = str(run_dir)
    payload["output"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    payload["output"]["metrics_path"] = str(run_dir / "metrics.jsonl")
    payload["output"]["manifest_path"] = str(run_dir / "training_manifest.json")
    payload["trainer"]["max_steps"] = 1
    payload["trainer"]["accelerator"] = "cpu"
    payload["loader"]["batch_size"] = 2
    payload["loader"]["shuffle"] = False
    train_torch(payload, root=ROOT, device="cpu")
    return run_dir


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
