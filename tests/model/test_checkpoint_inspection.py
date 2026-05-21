from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.model import (
    CHECKPOINT_SCHEMA_VERSION,
    TorchCodeTransitionModelConfig,
    build_checkpoint_metadata,
    build_torch_transition_model,
    compute_config_hash,
    write_checkpoint_manifest,
)
from codelewm.model.inspection import (
    MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION,
    CheckpointInspectionResult,
    inspect_checkpoint,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums
from codelewm.security import CheckpointTrustError
from codelewm.training import TORCH_CHECKPOINT_SCHEMA_VERSION


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class CheckpointInspectionTest(unittest.TestCase):
    def test_inspection_writes_manifested_tensor_layer_report(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = _write_tiny_trusted_checkpoint(root, torch)
            output_dir = root / "inspection"

            result = inspect_checkpoint(
                checkpoint=checkpoint_path,
                out=output_dir,
                command=("codelewm", "model", "inspect-checkpoint"),
                max_histogram_tensors=2,
                histogram_bins=8,
            )

            manifest = read_artifact_manifest(output_dir / "manifest.json")
            validate_artifact_checksums(manifest, root=output_dir)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertIsInstance(result, CheckpointInspectionResult)
        self.assertEqual(report["schema_version"], MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION)
        self.assertTrue(report["checkpoint"]["trust_gate"]["trusted"])
        self.assertFalse(report["histogram_policy"]["raw_tensor_values_serialized"])
        self.assertFalse(report["histogram_policy"]["optimizer_state_serialized"])
        self.assertGreater(report["summary"]["tensor_count"], 0)
        self.assertGreater(report["summary"]["module_count"], 0)
        self.assertGreater(report["summary"]["parameter_count"], 0)
        self.assertEqual(report["summary"]["histogrammed_tensor_count"], 2)
        self.assertIn("checkpoint_manifest_metadata", report["compatibility"])
        self.assertIn("modules", report)
        self.assertIn("tensors", report)
        self.assertEqual(manifest.artifact_kind, "eval_report")
        self.assertIn("reports/model_checkpoint_inspection.json", {item.path for item in manifest.files})

    def test_inspection_refuses_unmanifested_checkpoint_by_default(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "checkpoint.pt"
            torch.save(
                {
                    "schema_version": TORCH_CHECKPOINT_SCHEMA_VERSION,
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "model_state_dict": {},
                },
                checkpoint_path,
            )

            with self.assertRaises(CheckpointTrustError):
                inspect_checkpoint(
                    checkpoint=checkpoint_path,
                    out=root / "inspection",
                    command=("codelewm", "model", "inspect-checkpoint"),
                )

    def test_cli_inspect_checkpoint_emits_json_result(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = _write_tiny_trusted_checkpoint(root, torch)
            output_dir = root / "inspection"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "model",
                    "inspect-checkpoint",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["schema_version"], "codelewm.model_checkpoint_inspection_run.v1")
        self.assertEqual(payload["report_path"], "reports/model_checkpoint_inspection.json")
        self.assertGreater(payload["tensor_count"], 0)


def _write_tiny_trusted_checkpoint(root: Path, torch) -> Path:
    config = TorchCodeTransitionModelConfig()
    model = build_torch_transition_model(config)
    compatibility_config = {
        "wm": {
            "history_size": 1,
            "num_preds": 1,
            "embed_dim": config.latent_dim,
            "action_view": config.action_view,
            "state_sequence_length": config.state_sequence_length,
            "action_sequence_length": config.action_sequence_length,
        },
        "loss": {
            "sigreg_weight": 0.1,
            "enable_retrieval_loss": False,
            "enable_action_use_margin": False,
            "enable_action_swap_contrastive": False,
            "enable_inverse_action_reconstruction": False,
        },
    }
    checkpoint_path = root / "checkpoints" / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": TORCH_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "compatibility_config": compatibility_config,
            "compatibility_config_hash": compute_config_hash(compatibility_config),
            "metrics": {"loss/total": 0.25},
        },
        checkpoint_path,
    )
    metadata = build_checkpoint_metadata(
        compatibility_config,
        latent_dim=config.latent_dim,
        action_view=config.action_view,
        model_class="TorchCodeTransitionModel",
    )
    write_checkpoint_manifest(
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        manifest_path=checkpoint_path.with_name(checkpoint_path.name + ".manifest.json"),
    )
    return checkpoint_path


if __name__ == "__main__":
    unittest.main()
