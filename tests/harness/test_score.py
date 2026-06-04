from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.harness import (
    SCORE_RESULT_SCHEMA_VERSION,
    ScoreError,
    ScoreResult,
    TransitionIndexEntry,
    build_transition_index,
    load_scorer,
    write_transition_index,
)
from codelewm.harness.scorer import EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION, _hashed_vector
from codelewm.model.checkpoint import build_checkpoint_metadata, write_checkpoint_manifest


ROOT = Path(__file__).resolve().parents[2]


class ConstantBackend:
    model_id = "constant-test-backend"
    warnings = ("constant backend",)

    def transition_energy(self, before: str, instruction: str, candidate: str) -> float:
        return 10.0


class ScoreApiTest(unittest.TestCase):
    def test_load_scorer_scores_fixture_files_with_schema_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            candidate = root / "after.py"
            checkpoint = root / "checkpoint.bin"
            before.write_text("def add(a, b):\n    return a + b\n")
            candidate.write_text("def add(a, b):\n    return a + b + 1\n")
            checkpoint.write_bytes(b"fixture checkpoint")

            scorer = load_scorer(checkpoint, allow_unsafe=True)
            result = scorer.score_files(
                before=before,
                instruction="add one to the returned value",
                candidate=candidate,
            )
            payload = result.to_dict()
            loaded = ScoreResult.from_dict(payload)

        self.assertEqual(payload["schema_version"], SCORE_RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["candidate"], str(candidate))
        self.assertGreaterEqual(payload["transition_energy"], 0.0)
        self.assertEqual(payload["final_score"], payload["transition_energy"])
        self.assertEqual(len(payload["checkpoint_sha256"]), 64)
        self.assertEqual(len(payload["input_digest"]), 64)
        self.assertIn("lightweight scorer backend", payload["warnings"][0])
        self.assertEqual(loaded.to_dict(), payload)

    def test_score_cli_emits_json_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            candidate = root / "after.py"
            instruction = root / "instruction.txt"
            checkpoint = root / "checkpoint.bin"
            before.write_text("value = 1\n")
            candidate.write_text("value = 2\n")
            instruction.write_text("increment value")
            checkpoint.write_bytes(b"fixture checkpoint")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "score",
                    "--before",
                    str(before),
                    "--instruction",
                    str(instruction),
                    "--candidate",
                    str(candidate),
                    "--checkpoint",
                    str(checkpoint),
                    "--json",
                    "--allow-unsafe-checkpoint",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], SCORE_RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["candidate"], str(candidate))
        self.assertGreaterEqual(payload["final_score"], 0.0)

    def test_score_with_index_populates_retrieval_prior_without_schema_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            candidate = root / "after.py"
            checkpoint = root / "checkpoint.bin"
            index_root = root / "index"
            candidate_text = "value = 2\n"
            before.write_text("value = 1\n")
            candidate.write_text(candidate_text)
            checkpoint.write_bytes(b"fixture checkpoint")
            index = build_transition_index(
                name="score-prior-fixture",
                entries=(TransitionIndexEntry(transition_id="t-0", split="train"),),
                vectors=np.asarray([_hashed_vector(candidate_text, dim=8)], dtype=np.float32),
            )
            write_transition_index(index, index_root)

            scorer = load_scorer(
                checkpoint,
                allow_unsafe=True,
                backend=ConstantBackend(),
                index=index_root,
                retrieval_prior_weight=2.0,
                retrieval_prior_k=1,
            )
            result = scorer.score_files(
                before=before,
                instruction="increment value",
                candidate=candidate,
            )

        payload = result.to_dict()
        self.assertEqual(payload["schema_version"], SCORE_RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["retrieval_prior"], 0.0)
        self.assertEqual(payload["final_score"], payload["transition_energy"])
        self.assertIn("retrieval prior computed from local transition index", payload["warnings"][-1])

    def test_require_learned_backend_rejects_manifested_non_torch_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            write_checkpoint_manifest(
                metadata=build_checkpoint_metadata(
                    {"fixture": True},
                    model_class="CodeTransitionModel",
                ),
                checkpoint_path=checkpoint,
                manifest_path=checkpoint.with_name(checkpoint.name + ".manifest.json"),
            )

            with self.assertRaises(ScoreError) as raised:
                load_scorer(checkpoint, require_learned_backend=True)

        self.assertEqual(raised.exception.error_type, "checkpoint_error")
        self.assertIn("learned scorer backend was required", str(raised.exception))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("einops") is not None,
        "torch scoring runtime is unavailable",
    )
    def test_load_scorer_uses_learned_torch_backend_for_trusted_checkpoint(self) -> None:
        import torch

        from codelewm.model import (
            TorchCodeTransitionModelConfig,
            build_torch_transition_model,
            compute_config_hash,
        )
        from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE, TORCH_CHECKPOINT_SCHEMA_VERSION

        compatibility = {
            "wm": {
                "action_view": "text",
                "embed_dim": 256,
                "state_sequence_length": 1024,
                "action_sequence_length": 256,
                "action_fusion": "conditional_transformer",
            },
            "loss": {"enable_inverse_action_reconstruction": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            model = build_torch_transition_model(
                TorchCodeTransitionModelConfig(
                    vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
                    dropout=0.0,
                )
            )
            torch.save(
                {
                    "schema_version": TORCH_CHECKPOINT_SCHEMA_VERSION,
                    "step": 7,
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
                    action_view="text",
                    model_class="TorchCodeTransitionModel",
                ),
                checkpoint_path=checkpoint,
                manifest_path=checkpoint.with_name(checkpoint.name + ".manifest.json"),
            )

            scorer = load_scorer(checkpoint, device="cpu", require_learned_backend=True)
            result = scorer.score_texts(
                before="value = 1\n",
                instruction="increment value",
                candidate="value = 2\n",
            )

        self.assertEqual(result.model_id, "codelewm.torch_transition_scorer.v1")
        self.assertGreaterEqual(result.transition_energy, 0.0)
        self.assertTrue(any("checkpoint_step=7" == warning for warning in result.warnings))
        self.assertFalse(any("lightweight scorer backend" in warning for warning in result.warnings))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("einops") is not None,
        "torch scoring runtime is unavailable",
    )
    def test_load_scorer_accepts_ema_target_encoder_checkpoint(self) -> None:
        import torch

        from codelewm.model import (
            TorchCodeTransitionModelConfig,
            build_torch_transition_model,
            compute_config_hash,
        )
        from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE, TORCH_CHECKPOINT_SCHEMA_VERSION

        compatibility = {
            "wm": {
                "action_view": "text",
                "embed_dim": 256,
                "state_sequence_length": 1024,
                "action_sequence_length": 256,
                "action_fusion": "conditional_transformer",
                "enable_ema_target_encoder": True,
                "ema_target_decay": 0.5,
            },
            "loss": {"enable_inverse_action_reconstruction": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            model = build_torch_transition_model(
                TorchCodeTransitionModelConfig(
                    vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
                    dropout=0.0,
                    enable_ema_target_encoder=True,
                    ema_target_decay=0.5,
                )
            )
            torch.save(
                {
                    "schema_version": TORCH_CHECKPOINT_SCHEMA_VERSION,
                    "step": 8,
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
                    action_view="text",
                    model_class="TorchCodeTransitionModel",
                ),
                checkpoint_path=checkpoint,
                manifest_path=checkpoint.with_name(checkpoint.name + ".manifest.json"),
            )

            scorer = load_scorer(checkpoint, device="cpu", require_learned_backend=True)
            result = scorer.score_texts(
                before="value = 1\n",
                instruction="increment value",
                candidate="value = 2\n",
            )

        self.assertEqual(result.model_id, "codelewm.torch_transition_scorer.v1")
        self.assertTrue(any("checkpoint_step=8" == warning for warning in result.warnings))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("einops") is not None,
        "torch scoring runtime is unavailable",
    )
    def test_load_scorer_accepts_output_value_head_checkpoint(self) -> None:
        import torch

        from codelewm.model import (
            TorchCodeTransitionModelConfig,
            build_torch_transition_model,
            compute_config_hash,
        )
        from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE, TORCH_CHECKPOINT_SCHEMA_VERSION

        compatibility = {
            "wm": {
                "action_view": "text",
                "embed_dim": 256,
                "state_sequence_length": 1024,
                "action_sequence_length": 256,
                "action_fusion": "conditional_transformer",
                "enable_output_value_head": True,
            },
            "loss": {"enable_inverse_action_reconstruction": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            model = build_torch_transition_model(
                TorchCodeTransitionModelConfig(
                    vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
                    dropout=0.0,
                    enable_output_value_head=True,
                )
            )
            torch.save(
                {
                    "schema_version": TORCH_CHECKPOINT_SCHEMA_VERSION,
                    "step": 9,
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
                    action_view="text",
                    model_class="TorchCodeTransitionModel",
                ),
                checkpoint_path=checkpoint,
                manifest_path=checkpoint.with_name(checkpoint.name + ".manifest.json"),
            )

            scorer = load_scorer(checkpoint, device="cpu", require_learned_backend=True)
            result = scorer.score_texts(
                before="value = 1\n",
                instruction="increment value",
                candidate="value = 2\n",
            )

        self.assertEqual(result.model_id, "codelewm.torch_transition_scorer.v1")
        self.assertTrue(any("checkpoint_step=9" == warning for warning in result.warnings))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("einops") is not None,
        "torch scoring runtime is unavailable",
    )
    def test_load_scorer_uses_execution_backend_for_v0_6_checkpoint_manifest(self) -> None:
        import torch

        from codelewm.model import (
            TorchCodeTransitionModelConfig,
            build_torch_transition_model,
            compute_config_hash,
        )
        from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE

        compatibility = {
            "wm": {
                "action_view": "text",
                "embed_dim": 256,
                "state_sequence_length": 1024,
                "action_sequence_length": 256,
                "action_fusion": "conditional_transformer",
            },
            "objective": {"inverse_action_reconstruction_weight": 0.0},
            "loader": {"output_sequence_length": 256},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "last.pt"
            model = build_torch_transition_model(
                TorchCodeTransitionModelConfig(
                    vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
                    dropout=0.0,
                )
            )
            torch.save(
                {
                    "schema_version": EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION,
                    "step": 11,
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
                    record_schema_version="codelewm.execution_pack_record.v1",
                    action_view="text",
                    model_class="TorchCodeTransitionModel",
                ),
                checkpoint_path=checkpoint,
                manifest_path=checkpoint.with_name(checkpoint.name + ".manifest.json"),
            )

            scorer = load_scorer(checkpoint, device="cpu", require_learned_backend=True)
            result = scorer.score_texts(
                before="def compute_square(n):\n    pass\n",
                instruction="[3]",
                candidate="def compute_square(n):\n    return n * n\n",
            )

        self.assertEqual(result.model_id, "codelewm.execution_torch_transition_scorer.v1")
        self.assertGreaterEqual(result.transition_energy, 0.0)
        self.assertTrue(any("checkpoint_step=11" == warning for warning in result.warnings))
        self.assertTrue(any("execution-substrate" in warning for warning in result.warnings))
        self.assertTrue(
            any(
                warning
                == "transition_energy=predicted_output_latent_norm_plus_abs_code_similarity"
                for warning in result.warnings
            )
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("einops") is not None,
        "torch scoring runtime is unavailable",
    )
    def test_execution_backend_uses_negative_pass_logit_for_pass_head_checkpoint(self) -> None:
        import torch

        from codelewm.model import (
            TorchCodeTransitionModelConfig,
            build_torch_transition_model,
            compute_config_hash,
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "last.pt"
            model = build_torch_transition_model(
                TorchCodeTransitionModelConfig(
                    vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
                    dropout=0.0,
                    enable_pass_head=True,
                )
            )
            assert model.pass_head is not None
            with torch.no_grad():
                for param in model.pass_head.parameters():
                    param.zero_()
                model.pass_head[-1].bias.fill_(2.5)
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

            scorer = load_scorer(checkpoint, device="cpu", require_learned_backend=True)
            result = scorer.score_texts(
                before="",
                instruction="[3]",
                candidate="def compute_square(n):\n    return n * n\n",
            )

        self.assertEqual(result.model_id, "codelewm.execution_torch_transition_scorer.v1")
        self.assertAlmostEqual(result.transition_energy, -2.5, places=6)
        self.assertAlmostEqual(result.final_score, -2.5, places=6)
        self.assertTrue(
            any(warning == "transition_energy=neg_pass_logit" for warning in result.warnings)
        )


if __name__ == "__main__":
    unittest.main()
