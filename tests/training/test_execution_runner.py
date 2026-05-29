"""Tests for the v0.6 execution-substrate production runner.

The runner exercises the full forward+backward pass, so the heavy tests
are gated on torch. The lightweight tests use the same `mbpp_tiny`
fixture the local smoke does (#288).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401  # pyright: ignore[reportMissingImports]

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "execution_sources" / "fixtures"


def _build_pack(tmpdir: Path) -> Path:
    from codelewm.data.execution_pack import build_execution_pack
    from codelewm.data.execution_sources import load_execution_source
    from codelewm.data.sandbox import SandboxPolicy

    ingest = tmpdir / "mbpp.jsonl"
    load_execution_source(
        source="mbpp",
        source_path=FIXTURE_DIR / "mbpp_tiny.jsonl",
        output_path=ingest,
    )
    pack_dir = tmpdir / "pack"
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
    return pack_dir


def _config(*, max_steps: int = 6, collapse_every: int = 2) -> "ExecutionTrainConfig":
    from codelewm.training import (
        ExecutionTrainClaimBoundaryConfig,
        ExecutionTrainClaimGatesConfig,
        ExecutionTrainConfig,
        ExecutionTrainDataConfig,
        ExecutionTrainHfJobsConfig,
        ExecutionTrainLoaderConfig,
        ExecutionTrainObjectiveConfig,
        ExecutionTrainOptimizerConfig,
        ExecutionTrainTrainerConfig,
        ExecutionTrainWorldModelConfig,
    )

    return ExecutionTrainConfig(
        schema_version="codelewm.execution_train_config.v1",
        name="codelewm_execution_runner_test",
        substrate="execution_trace_v1",
        parent_issue=259,
        implementing_issue=265,
        target_substrate_run="v0.6.0",
        data=ExecutionTrainDataConfig(
            pack_repo_id="abdelstark/codelewm-execution-pack",
            pack_revision="v0.6.0",
            pack_jsonl="pack.jsonl",
            manifest_filename="manifest.json",
            claim_boundary_filename="claim_boundary.md",
            ingestion_sources=("mbpp",),
            held_out_for_eval=("mbpp_plus",),
        ),
        loader=ExecutionTrainLoaderConfig(
            code_sequence_length=1024,
            action_sequence_length=256,
            output_sequence_length=256,
            batch_size=2,
            gradient_accumulation_steps=2,
            effective_batch_size=4,
            shuffle=True,
        ),
        trainer=ExecutionTrainTrainerConfig(
            accelerator="cpu",
            devices=1,
            precision="float32",
            max_steps=max_steps,
            warmup_steps=1,
            cosine_decay_to=0.0,
            gradient_clip_val=1.0,
            checkpoint_every_n_steps=3,
            keep_last_n_checkpoints=1,
            keep_best_by_metric="loss_prediction_mse",
            tensorboard_enabled=False,
            collapse_diagnostics_every_n_steps=collapse_every,
        ),
        optimizer=ExecutionTrainOptimizerConfig(
            name="adamw",
            lr=3.0e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        ),
        wm=ExecutionTrainWorldModelConfig(
            history_size=1, num_preds=1, embed_dim=256
        ),
        objective=ExecutionTrainObjectiveConfig(
            prediction_mse_weight=1.0,
            sigreg_weight=0.09,
            action_swap_contrastive_weight=0.1,
            inverse_action_reconstruction_weight=0.0,
        ),
        seeds=(42,),
        hf_jobs=ExecutionTrainHfJobsConfig(
            flavor="a10g-small",
            region="us-east-1",
            timeout_hours=24,
            run_name_template="codelewm-test-{date}-{sha}-seed-{seed}",
            artifact_repo_id="abdelstark/codelewm-runs",
            checkpoint_repo_id="abdelstark/codelewm-transition-model",
            checkpoint_revision_template="v0.6.0-seed-{seed}",
        ),
        claim_gates=ExecutionTrainClaimGatesConfig(
            retrieval_min_recall_at_1_lift_over_no_action=0.05,
            retrieval_min_mrr_lift_over_no_action=0.05,
            collapse_effective_rank_ratio_min=0.20,
            collapse_per_dim_variance_median_min=1.0e-8,
            collapse_nearest_neighbor_entropy_min=0.10,
            surprise_mutation_auc_min=0.65,
            surprise_same_problem_different_submission_auc_min=0.60,
            surprise_same_code_different_input_auc_min=0.70,
            downstream_rerank_pass_at_1_lift_min=3.0,
            required_seeds=1,
        ),
        claim_boundary=ExecutionTrainClaimBoundaryConfig(
            name="execution_substrate.v1",
            scope="v0_6_runner_test",
        ),
    )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed")
class ExecutionRunnerIntegrationTest(unittest.TestCase):
    def test_runner_writes_manifest_and_checkpoints(self) -> None:
        from codelewm.training import (
            EXECUTION_TRAIN_RUN_REPORT_SCHEMA_VERSION,
            TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
            train_execution_run,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pack_dir = _build_pack(tmp)
            output_dir = tmp / "run-seed-42"

            cfg = _config(max_steps=6, collapse_every=2)
            result = train_execution_run(
                cfg,
                seed=42,
                output_dir=output_dir,
                root=tmp,
                pack_local_dir=pack_dir,
                command=("codelewm", "train", "--seed", "42"),
            )

            # Artifact + training manifests exist and validate.
            self.assertTrue(result.artifact_manifest_path.is_file())
            self.assertTrue(result.training_manifest_path.is_file())
            self.assertTrue(result.metrics_path.is_file())
            self.assertTrue(result.report_path.is_file())

            manifest_dict = json.loads(
                result.training_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest_dict["schema_version"],
                TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
            )
            self.assertEqual(manifest_dict["seed"], 42)
            self.assertEqual(manifest_dict["step_count"], 6)

            # Report carries the v0.6 schema marker and the claim
            # boundary / gates fingerprint.
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["schema_version"],
                EXECUTION_TRAIN_RUN_REPORT_SCHEMA_VERSION,
            )
            self.assertEqual(
                report["claim_boundary"]["name"], "execution_substrate.v1"
            )
            self.assertEqual(
                report["claim_gates"]["required_seeds"], 1
            )

            # Pointer files exist.
            self.assertTrue((output_dir / "checkpoints" / "last.pt").is_file())
            self.assertTrue(
                (output_dir / "checkpoints" / "last.pt.manifest.json").is_file()
            )

            # Metrics JSONL has one row per optimizer step.
            metric_rows = [
                json.loads(line)
                for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(metric_rows), 6)
            for row in metric_rows:
                self.assertIn("metrics", row)
                self.assertIn("loss_prediction_mse", row["metrics"])

            # Collapse diagnostics JSONL has a row per cadence hit.
            assert result.collapse_diagnostics_path is not None
            self.assertTrue(result.collapse_diagnostics_path.is_file())
            collapse_rows = [
                json.loads(line)
                for line in result.collapse_diagnostics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            # cadence=2, max_steps=6 -> rows at steps 2, 4, 6.
            self.assertEqual(len(collapse_rows), 3)
            self.assertIn("z_pred_effective_rank", collapse_rows[0]["diagnostics"])

    def test_runner_respects_env_pack_local_dir(self) -> None:
        from codelewm.training import train_execution_run

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pack_dir = _build_pack(tmp)
            output_dir = tmp / "run-env"
            cfg = _config(max_steps=3, collapse_every=1)
            os.environ["CODELEWM_EXECUTION_PACK_LOCAL_DIR"] = str(pack_dir)
            try:
                result = train_execution_run(
                    cfg,
                    seed=42,
                    output_dir=output_dir,
                    root=tmp,
                )
            finally:
                os.environ.pop("CODELEWM_EXECUTION_PACK_LOCAL_DIR", None)
            self.assertEqual(result.pack_dir.resolve(), pack_dir.resolve())

    def test_runner_rejects_existing_non_empty_output_without_overwrite(
        self,
    ) -> None:
        from codelewm.training import TrainingRunError, train_execution_run

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pack_dir = _build_pack(tmp)
            output_dir = tmp / "run-conflict"
            output_dir.mkdir(parents=True)
            (output_dir / "stamp").write_text("already here", encoding="utf-8")

            cfg = _config(max_steps=3)
            with self.assertRaises(TrainingRunError):
                train_execution_run(
                    cfg,
                    seed=42,
                    output_dir=output_dir,
                    root=tmp,
                    pack_local_dir=pack_dir,
                )

    def test_runner_overwrite_clears_path(self) -> None:
        from codelewm.training import train_execution_run

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pack_dir = _build_pack(tmp)
            output_dir = tmp / "run-overwrite"
            output_dir.mkdir(parents=True)
            (output_dir / "stamp").write_text("already here", encoding="utf-8")

            cfg = _config(max_steps=3)
            result = train_execution_run(
                cfg,
                seed=42,
                output_dir=output_dir,
                root=tmp,
                pack_local_dir=pack_dir,
                overwrite=True,
            )
            self.assertTrue(result.training_manifest_path.is_file())


class ExecutionRunnerWithoutTorchTest(unittest.TestCase):
    def test_invalid_config_type_rejected(self) -> None:
        from codelewm.training import (
            ExecutionTrainConfigError,
            train_execution_run,
        )

        with self.assertRaises(ExecutionTrainConfigError):
            # Calls into the runner with a non-config object so the
            # type-guard fires before any torch import.
            train_execution_run(  # type: ignore[arg-type]
                "not a config",  # type: ignore[arg-type]
                seed=42,
                output_dir=Path("/tmp/codelewm-test-nonexistent"),
            )

    def test_negative_seed_rejected(self) -> None:
        from codelewm.training import TrainingRunError, train_execution_run

        with self.assertRaises(TrainingRunError):
            train_execution_run(
                _config(),
                seed=-1,
                output_dir=Path("/tmp/codelewm-test-nonexistent-seed"),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
