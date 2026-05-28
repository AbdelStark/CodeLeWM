"""Tests for the execution-substrate torch training runner.

These tests are gated on torch being installed because the runner
exercises the full forward+backward pass through the JEPA model. The
lightweight test environment that only installs ``--group dev`` skips
them. The dataset-fixture CI job installs ``--group data --group dev``
which still does not include torch; the runner tests live in their own
opt-in group documented in the smoke report.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


try:
    import torch  # noqa: F401  # pyright: ignore[reportMissingImports]

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "execution_sources"
    / "fixtures"
)
SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "codelewm-execution-train-smoke"
)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed")
class ExecutionTorchRunnerSmokeTest(unittest.TestCase):
    def test_loss_trajectory_satisfies_smoke_gate(self) -> None:
        from codelewm.data.execution_pack import build_execution_pack
        from codelewm.data.execution_sources import load_execution_source
        from codelewm.data.sandbox import SandboxPolicy
        from codelewm.training import (
            EXECUTION_TRAIN_REPORT_SCHEMA_VERSION,
            ExecutionTorchTrainConfig,
            train_execution_smoke,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ingest = tmp / "mbpp.jsonl"
            load_execution_source(
                source="mbpp",
                source_path=FIXTURES / "mbpp_tiny.jsonl",
                output_path=ingest,
            )
            pack_dir = tmp / "pack"
            build_execution_pack(
                ingestion_paths=[ingest],
                output_dir=pack_dir,
                sandbox_policy=SandboxPolicy(
                    timeout_ms=3000,
                    memory_mb=1024,
                    cpu_seconds=2,
                    determinism_check=True,
                ),
                seed=42,
                train_frac=0.5,
                val_frac=0.25,
            )
            report = train_execution_smoke(
                ExecutionTorchTrainConfig(
                    pack_jsonl=pack_dir / "pack.jsonl",
                    output_dir=tmp / "out",
                    batch_size=4,
                    max_steps=80,
                    warmup_steps=10,
                    device="cpu",
                    seed=42,
                )
            )

            self.assertEqual(
                report.schema_version,
                EXECUTION_TRAIN_REPORT_SCHEMA_VERSION,
            )
            # Prediction MSE drops substantially over 80 steps on the
            # tiny pack — overfitting to the fixture is expected.
            self.assertLess(
                report.final_metrics["loss_prediction_mse"],
                report.initial_metrics["loss_prediction_mse"] * 0.5,
                msg=(
                    f"initial={report.initial_metrics['loss_prediction_mse']:.4f} "
                    f"final={report.final_metrics['loss_prediction_mse']:.4f}"
                ),
            )
            # SIGReg drops too (more spread, less collapsed).
            self.assertLess(
                report.final_metrics["loss_sigreg"],
                report.initial_metrics["loss_sigreg"],
            )
            # The substrate-pivot headline: the no-action margin flips
            # from negative (no-action wins) to positive (pred wins).
            self.assertLess(
                report.initial_metrics["margin_no_action_minus_pred"], 0.0
            )
            self.assertGreater(
                report.final_metrics["margin_no_action_minus_pred"], 0.0
            )
            # Latent diagnostics are emitted and well-formed.
            self.assertEqual(
                report.z_diagnostics["latent_dim"], 256
            )
            self.assertGreater(
                report.z_diagnostics["sample_count"], 0
            )
            self.assertGreater(
                report.z_diagnostics["z_pred_effective_rank"], 0.0
            )
            self.assertTrue(
                (tmp / "out" / "execution_train_report.json").is_file()
            )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed")
class ExecutionTorchRunnerCLITest(unittest.TestCase):
    def test_cli_runs_and_emits_pass_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--ingestion",
                    str(FIXTURES / "mbpp_tiny.jsonl"),
                    "--out",
                    str(Path(tmpdir) / "smoke"),
                    "--batch-size",
                    "4",
                    "--max-steps",
                    "80",
                    "--warmup-steps",
                    "10",
                    "--device",
                    "cpu",
                    "--json",
                ],
                env={
                    "PYTHONPATH": str(
                        Path(__file__).resolve().parents[2]
                    ),
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode, 0,
                msg=f"stderr={completed.stderr!r}\nstdout={completed.stdout[:500]!r}",
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(
                payload["schema_version"], "codelewm.execution_train_smoke.v1"
            )
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["device"], "cpu")
            self.assertGreater(payload["training_steps"], 0)


class ExecutionTorchRunnerConfigTest(unittest.TestCase):
    """Config validation tests that don't need torch."""

    def test_invalid_code_sequence_length_rejected(self) -> None:
        from codelewm.training import (
            ExecutionTorchRunnerError,
            ExecutionTorchTrainConfig,
        )

        with self.assertRaises(ExecutionTorchRunnerError):
            ExecutionTorchTrainConfig(
                pack_jsonl=Path("/tmp/pack.jsonl"),
                output_dir=Path("/tmp/out"),
                code_sequence_length=999,
            )

    def test_invalid_output_sequence_length_rejected(self) -> None:
        from codelewm.training import (
            ExecutionTorchRunnerError,
            ExecutionTorchTrainConfig,
        )

        with self.assertRaises(ExecutionTorchRunnerError):
            ExecutionTorchTrainConfig(
                pack_jsonl=Path("/tmp/pack.jsonl"),
                output_dir=Path("/tmp/out"),
                output_sequence_length=2048,  # > STATE_SEQUENCE_LENGTH
            )

    def test_zero_steps_rejected(self) -> None:
        from codelewm.training import (
            ExecutionTorchRunnerError,
            ExecutionTorchTrainConfig,
        )

        with self.assertRaises(ExecutionTorchRunnerError):
            ExecutionTorchTrainConfig(
                pack_jsonl=Path("/tmp/pack.jsonl"),
                output_dir=Path("/tmp/out"),
                max_steps=0,
            )

    def test_missing_pack_jsonl_in_runner_raises(self) -> None:
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not installed")
        from codelewm.training import (
            ExecutionTorchRunnerError,
            ExecutionTorchTrainConfig,
            train_execution_smoke,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ExecutionTorchRunnerError):
                train_execution_smoke(
                    ExecutionTorchTrainConfig(
                        pack_jsonl=Path(tmpdir) / "missing.jsonl",
                        output_dir=Path(tmpdir) / "out",
                        max_steps=1,
                    )
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
