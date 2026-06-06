from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

from codelewm.eval import (
    CRASH_PREDICTION_EVAL_RUN_SCHEMA_VERSION,
    CRASH_PREDICTION_REPORT_SCHEMA_VERSION,
    EXECUTION_PROBE_EVAL_RUN_SCHEMA_VERSION,
    EXECUTION_RETRIEVAL_EVAL_RUN_SCHEMA_VERSION,
    EXECUTION_SURPRISE_EVAL_RUN_SCHEMA_VERSION,
    LATENT_PROBE_REPORT_SCHEMA_VERSION,
    RETRIEVAL_REPORT_SCHEMA_VERSION,
    SURPRISE_REPORT_SCHEMA_VERSION,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


TORCH_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("einops") is not None
)
ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "data" / "execution_sources" / "fixtures"
EXECUTION_CONFIG = ROOT / "config" / "train" / "scaled" / "codelewm_execution_v0_6_a10g.yaml"


@unittest.skipUnless(TORCH_RUNTIME_AVAILABLE, "torch/einops are not installed")
class ExecutionEvalCliTest(unittest.TestCase):
    def test_execution_eval_clis_write_schema_versioned_manifested_reports(self) -> None:
        from codelewm.training import train_execution_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = _build_pack(root)
            train_dir = root / "runs" / "train"
            cfg = _tiny_execution_config()
            train_result = train_execution_run(
                cfg,
                seed=42,
                output_dir=train_dir,
                root=root,
                pack_local_dir=pack_dir,
                command=("codelewm", "train", "--config", str(EXECUTION_CONFIG)),
            )
            checkpoint = train_dir / "checkpoints" / "last.pt"

            cases = (
                _EvalCase(
                    name="execution-retrieval",
                    args=(
                        "eval",
                        "execution-retrieval",
                        "--checkpoint",
                        str(checkpoint),
                        "--pack",
                        str(pack_dir),
                        "--baselines",
                        "random,no_action,shuffled_action",
                        "--out",
                        str(root / "runs" / "execution-retrieval"),
                        "--max-candidates",
                        "5",
                        "--json",
                    ),
                    run_schema=EXECUTION_RETRIEVAL_EVAL_RUN_SCHEMA_VERSION,
                    report_path=Path("reports/retrieval_report.json"),
                    report_schema=RETRIEVAL_REPORT_SCHEMA_VERSION,
                ),
                _EvalCase(
                    name="execution-surprise",
                    args=(
                        "eval",
                        "execution-surprise",
                        "--checkpoint",
                        str(checkpoint),
                        "--pack",
                        str(pack_dir),
                        "--decoys",
                        "mutation,same_problem_different_submission,same_code_different_input",
                        "--out",
                        str(root / "runs" / "execution-surprise"),
                        "--max-examples",
                        "5",
                        "--json",
                    ),
                    run_schema=EXECUTION_SURPRISE_EVAL_RUN_SCHEMA_VERSION,
                    report_path=Path("reports/surprise_report.json"),
                    report_schema=SURPRISE_REPORT_SCHEMA_VERSION,
                ),
                _EvalCase(
                    name="execution-probe",
                    args=(
                        "eval",
                        "execution-probe",
                        "--checkpoint",
                        str(checkpoint),
                        "--pack",
                        str(pack_dir),
                        "--targets",
                        "output_type,will_raise,output_magnitude_bucket,output_length_bucket",
                        "--out",
                        str(root / "runs" / "execution-probe"),
                        "--max-examples-per-split",
                        "5",
                        "--bootstrap-samples",
                        "4",
                        "--json",
                    ),
                    run_schema=EXECUTION_PROBE_EVAL_RUN_SCHEMA_VERSION,
                    report_path=Path("reports/latent_probe_report.json"),
                    report_schema=LATENT_PROBE_REPORT_SCHEMA_VERSION,
                ),
                _EvalCase(
                    name="crash-prediction",
                    args=(
                        "eval",
                        "crash-prediction",
                        "--checkpoint",
                        str(checkpoint),
                        "--pack",
                        str(pack_dir),
                        "--out",
                        str(root / "runs" / "crash-prediction"),
                        "--max-examples",
                        "5",
                        "--json",
                    ),
                    run_schema=CRASH_PREDICTION_EVAL_RUN_SCHEMA_VERSION,
                    report_path=Path("reports/crash_prediction_report.json"),
                    report_schema=CRASH_PREDICTION_REPORT_SCHEMA_VERSION,
                ),
            )

            for case in cases:
                with self.subTest(case=case.name):
                    out_dir = root / "runs" / case.name
                    completed = _run_cli(*case.args)
                    self.assertEqual(
                        completed.returncode,
                        0,
                        completed.stderr or completed.stdout,
                    )
                    payload = json.loads(completed.stdout)
                    report = json.loads(
                        (out_dir / case.report_path).read_text(encoding="utf-8")
                    )
                    artifact_manifest = read_artifact_manifest(out_dir / "manifest.json")
                    checked_files = validate_artifact_checksums(
                        artifact_manifest, root=out_dir
                    )
                    verify = _run_cli(
                        "manifest",
                        "verify",
                        "--manifest",
                        str(out_dir / "manifest.json"),
                        "--parent-manifest",
                        str(train_result.artifact_manifest_path),
                        "--parent-manifest",
                        str(pack_dir / "artifact_manifest.json"),
                        "--json",
                    )
                    verify_payload = json.loads(verify.stdout)

                    self.assertEqual(payload["schema_version"], case.run_schema)
                    self.assertEqual(payload["artifact_manifest_path"], "manifest.json")
                    self.assertEqual(payload["report_path"], case.report_path.as_posix())
                    self.assertEqual(report["schema_version"], case.report_schema)
                    self.assertEqual(artifact_manifest.artifact_kind, "eval_report")
                    self.assertIn(case.report_path.name, {path.name for path in checked_files})
                    self.assertEqual(verify.returncode, 0, verify.stderr)
                    self.assertTrue(verify_payload["ok"])
                    if case.name == "execution-surprise":
                        decoy_report = json.loads(
                            (out_dir / "reports" / "execution_decoy_report.json")
                            .read_text(encoding="utf-8")
                        )
                        self.assertIn("pair_count_summary", decoy_report)
                        self.assertEqual(
                            decoy_report["pair_count_summary"]["schema_version"],
                            "codelewm.eval.execution_decoy_coverage_summary.v1",
                        )
                        gates = report["metadata"]["execution_surprise_claim_gates"]
                        self.assertIn("coverage_blockers", gates)
                        self.assertIn(
                            "scorable_pair_count_by_category",
                            gates["semantic_decoy_category_counts"],
                        )

    def test_execution_eval_loader_accepts_pass_head_checkpoint(self) -> None:
        import torch

        from codelewm.eval.execution_runner import _load_execution_torch_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_pass_head_execution_checkpoint(root)
            model, payload = _load_execution_torch_checkpoint(
                checkpoint, device=torch.device("cpu"), runtime=torch
            )

        self.assertEqual(
            payload["schema_version"], "codelewm.execution_train_checkpoint.v1"
        )
        self.assertIsNotNone(model.pass_head)


@dataclass(frozen=True)
class _EvalCase:
    name: str
    args: tuple[str, ...]
    run_schema: str
    report_path: Path
    report_schema: str


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


def _tiny_execution_config() -> "ExecutionTrainConfig":
    from codelewm.training import load_execution_train_config

    cfg = load_execution_train_config(EXECUTION_CONFIG)
    return replace(
        cfg,
        name="execution_eval_cli_fixture",
        implementing_issue=302,
        loader=replace(
            cfg.loader,
            batch_size=2,
            gradient_accumulation_steps=1,
            effective_batch_size=2,
            shuffle=False,
        ),
        trainer=replace(
            cfg.trainer,
            accelerator="cpu",
            precision="float32",
            max_steps=3,
            warmup_steps=1,
            checkpoint_every_n_steps=3,
            keep_last_n_checkpoints=1,
            keep_best_by_metric="loss_prediction_mse",
            tensorboard_enabled=False,
            collapse_diagnostics_every_n_steps=1,
        ),
        objective=replace(
            cfg.objective,
            inverse_action_reconstruction_weight=0.0,
        ),
        seeds=(42,),
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "codelewm.harness.cli", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_pass_head_execution_checkpoint(root: Path) -> Path:
    import torch

    from codelewm.harness.scorer import EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION
    from codelewm.model import (
        TorchCodeTransitionModelConfig,
        build_torch_transition_model,
        compute_config_hash,
    )
    from codelewm.model.checkpoint import (
        build_checkpoint_metadata,
        write_checkpoint_manifest,
    )
    from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE

    compatibility = {
        "wm": {
            "action_view": "text",
            "embed_dim": 256,
            "state_sequence_length": 1024,
            "action_sequence_length": 256,
            "action_fusion": "conditional_transformer",
            "enable_pass_head": True,
        },
        "objective": {
            "inverse_action_reconstruction_weight": 0.0,
            "p_pass_bce_weight": 0.5,
            "p_pass_bce_pos_weight": 1.0,
        },
        "loader": {"output_sequence_length": 256},
    }
    model = build_torch_transition_model(
        TorchCodeTransitionModelConfig(
            vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
            dropout=0.0,
            enable_pass_head=True,
        )
    )
    checkpoint = root / "pass_head_last.pt"
    torch.save(
        {
            "schema_version": EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION,
            "step": 12,
            "model_state_dict": model.state_dict(),
            "compatibility_config": compatibility,
            "compatibility_config_hash": compute_config_hash(compatibility),
            "metrics": {"fixture": 1.0},
        },
        checkpoint,
    )
    write_checkpoint_manifest(
        metadata=build_checkpoint_metadata(
            compatibility,
            record_schema_version="codelewm.execution_pack_record.v2",
            action_view="text",
            model_class="TorchCodeTransitionModel",
        ),
        checkpoint_path=checkpoint,
        manifest_path=checkpoint.with_name(checkpoint.name + ".manifest.json"),
    )
    return checkpoint


if __name__ == "__main__":
    unittest.main()
