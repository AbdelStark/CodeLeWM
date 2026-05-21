from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.training import (
    TENSORBOARD_EXPORT_SCHEMA_VERSION,
    TensorBoardExportError,
    export_tensorboard_training_run,
)


class TensorBoardExportTest(unittest.TestCase):
    def test_export_writes_event_metadata_without_live_tensorboard_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            model = _FakeModel()

            result = export_tensorboard_training_run(
                run_id="fixture",
                run_dir=run_dir,
                step_count=3,
                metrics={
                    "loss/total": 0.25,
                    "loss/prediction_mse": 0.20,
                    "train/examples_per_second": 12.0,
                },
                model=model,
                embeddings=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
                checkpoint_path=run_dir / "checkpoints" / "checkpoint.pt",
                checkpoint_manifest_path=run_dir / "checkpoints" / "checkpoint.pt.manifest.json",
                log_dir="events",
                writer_factory=_FakeSummaryWriter,
                max_histogram_tensors=2,
                max_histogram_values=4,
            )
            payload = json.loads(result.report_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["schema_version"], TENSORBOARD_EXPORT_SCHEMA_VERSION)
            self.assertEqual(payload["run_id"], "fixture")
            self.assertEqual(payload["step_count"], 3)
            self.assertEqual(payload["log_dir"], "events")
            self.assertEqual(payload["event_files"][0]["path"], "events/events.out.tfevents.fixture")
            self.assertRegex(payload["event_files"][0]["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("loss/total", payload["scalar_tags"])
            self.assertIn("latents/export_std", payload["scalar_tags"])
            self.assertIn("latents/last_embedding_values", payload["histogram_tags"])
            self.assertIn("parameters/encoder.weight", payload["histogram_tags"])
            self.assertFalse(payload["safety_limits"]["raw_checkpoint_serialized"])
            self.assertFalse(payload["safety_limits"]["candidate_code_serialized"])
            self.assertEqual(
                result.to_metadata(root=run_dir)["event_files"][0]["path"],
                "events/events.out.tfevents.fixture",
            )

    def test_export_rejects_event_dir_outside_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            with self.assertRaisesRegex(TensorBoardExportError, "run directory"):
                export_tensorboard_training_run(
                    run_id="fixture",
                    run_dir=run_dir,
                    step_count=1,
                    metrics={"loss/total": 0.25},
                    log_dir=Path(tmp) / "outside",
                    writer_factory=_FakeSummaryWriter,
                )


class _FakeModel:
    def named_parameters(self):
        return iter(
            (
                ("encoder.weight", np.array([0.1, 0.2, 0.3], dtype=np.float32)),
                ("predictor.weight", np.array([0.4, 0.5], dtype=np.float32)),
            )
        )


class _FakeSummaryWriter:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.log_dir / "events.out.tfevents.fixture"
        self.event_path.write_bytes(b"fake tensorboard events\n")

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.event_path.write_bytes(self.event_path.read_bytes() + f"scalar {global_step} {tag} {scalar_value}\n".encode())

    def add_histogram(self, tag: str, values, global_step: int) -> None:
        array = np.asarray(values).reshape(-1)
        self.event_path.write_bytes(
            self.event_path.read_bytes() + f"hist {global_step} {tag} {array.size}\n".encode()
        )

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
