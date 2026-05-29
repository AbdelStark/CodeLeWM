"""End-to-end CLI test for the v0.6 execution-substrate train dispatch.

Exercises ``codelewm train --config <execution-yaml> --seed N`` in-process
so the routing decision in :func:`codelewm.harness.cli._train_command` is
covered without spawning a subprocess.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
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


# A minimal execution-train YAML config with a tiny step budget so the
# CLI can complete the run in under a minute on CPU.
_TINY_CFG_YAML = """
schema_version: codelewm.execution_train_config.v1
name: codelewm_execution_cli_test
substrate: execution_trace_v1
parent_issue: 259
implementing_issue: 265
target_substrate_run: v0.6.0
data:
  pack_repo_id: abdelstark/codelewm-execution-pack
  pack_revision: v0.6.0
  pack_jsonl: pack.jsonl
  manifest_filename: manifest.json
  claim_boundary_filename: claim_boundary.md
  ingestion_sources:
    - mbpp
  held_out_for_eval:
    - mbpp_plus
loader:
  code_sequence_length: 1024
  action_sequence_length: 256
  output_sequence_length: 256
  batch_size: 2
  gradient_accumulation_steps: 2
  effective_batch_size: 4
  shuffle: true
trainer:
  accelerator: cpu
  devices: 1
  precision: float32
  max_steps: 4
  warmup_steps: 1
  cosine_decay_to: 0.0
  gradient_clip_val: 1.0
  checkpoint_every_n_steps: 2
  keep_last_n_checkpoints: 1
  keep_best_by_metric: loss_prediction_mse
  tensorboard_enabled: false
  collapse_diagnostics_every_n_steps: 2
optimizer:
  name: adamw
  lr: 0.0003
  betas:
    - 0.9
    - 0.95
  weight_decay: 0.1
wm:
  history_size: 1
  num_preds: 1
  embed_dim: 256
objective:
  prediction_mse_weight: 1.0
  sigreg_weight: 0.09
  action_swap_contrastive_weight: 0.1
  inverse_action_reconstruction_weight: 0.0
seeds:
  - 42
hf_jobs:
  flavor: a10g-small
  region: us-east-1
  timeout_hours: 24
  run_name_template: codelewm-test-{date}-{sha}-seed-{seed}
  artifact_repo_id: abdelstark/codelewm-runs
  checkpoint_repo_id: abdelstark/codelewm-transition-model
  checkpoint_revision_template: v0.6.0-seed-{seed}
claim_gates:
  retrieval_min_recall_at_1_lift_over_no_action: 0.05
  retrieval_min_mrr_lift_over_no_action: 0.05
  collapse_effective_rank_ratio_min: 0.20
  collapse_per_dim_variance_median_min: 0.00000001
  collapse_nearest_neighbor_entropy_min: 0.10
  surprise_mutation_auc_min: 0.65
  surprise_same_problem_different_submission_auc_min: 0.60
  surprise_same_code_different_input_auc_min: 0.70
  downstream_rerank_pass_at_1_lift_min: 3.0
  required_seeds: 1
claim_boundary:
  name: execution_substrate.v1
  scope: v0_6_cli_test
"""


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed")
class CliExecutionTrainDispatchTest(unittest.TestCase):
    def test_train_dispatches_to_execution_runner(self) -> None:
        from codelewm.harness.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pack_dir = _build_pack(tmp)
            cfg_path = tmp / "execution_v0_6_test.yaml"
            cfg_path.write_text(_TINY_CFG_YAML, encoding="utf-8")
            out_dir = tmp / "out"

            argv = [
                "train",
                "--config",
                str(cfg_path),
                "--seed",
                "42",
                "--pack-local-dir",
                str(pack_dir),
                "--out",
                str(out_dir),
                "--device",
                "cpu",
                "--json",
            ]
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exit_code = main(argv)

            self.assertEqual(
                exit_code,
                0,
                msg=(
                    f"main returned {exit_code}; "
                    f"stderr={stderr_buf.getvalue()!r}; "
                    f"stdout={stdout_buf.getvalue()[:500]!r}"
                ),
            )

            payload = json.loads(stdout_buf.getvalue())
            self.assertEqual(
                payload["schema_version"], "codelewm.training_run.v1"
            )
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["step_count"], 4)
            self.assertTrue((out_dir / "manifest.json").is_file())
            self.assertTrue((out_dir / "training_manifest.json").is_file())
            self.assertTrue((out_dir / "checkpoints" / "last.pt").is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
