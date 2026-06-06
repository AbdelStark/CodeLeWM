"""Tests for the v0.6 execution-substrate training config schema."""

from __future__ import annotations

import json
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[2]
V0_8_PASSFAIL_POS_WEIGHT = 0.9145473041709054


_VALID_CONFIG_DICT = {
    "schema_version": "codelewm.execution_train_config.v1",
    "name": "codelewm_execution_v0_6_test",
    "substrate": "execution_trace_v1",
    "parent_issue": 259,
    "implementing_issue": 265,
    "target_substrate_run": "v0.6.0",
    "data": {
        "pack_repo_id": "abdelstark/codelewm-execution-pack",
        "pack_revision": "v0.6.0",
        "pack_jsonl": "pack.jsonl",
        "manifest_filename": "manifest.json",
        "claim_boundary_filename": "claim_boundary.md",
        "ingestion_sources": ["mbpp"],
        "held_out_for_eval": ["mbpp_plus"],
    },
    "loader": {
        "code_sequence_length": 1024,
        "action_sequence_length": 256,
        "output_sequence_length": 256,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 8,
        "shuffle": True,
    },
    "trainer": {
        "accelerator": "cpu",
        "devices": 1,
        "precision": "float32",
        "max_steps": 100,
        "warmup_steps": 10,
        "cosine_decay_to": 0.0,
        "gradient_clip_val": 1.0,
        "checkpoint_every_n_steps": 50,
        "keep_last_n_checkpoints": 2,
        "keep_best_by_metric": "loss_prediction_mse",
        "tensorboard_enabled": False,
        "collapse_diagnostics_every_n_steps": 25,
        "progress_log_every_n_steps": 25,
    },
    "optimizer": {
        "name": "adamw",
        "lr": 3.0e-4,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
    },
    "wm": {"history_size": 1, "num_preds": 1, "embed_dim": 256},
    "objective": {
        "prediction_mse_weight": 1.0,
        "sigreg_weight": 0.09,
        "action_swap_contrastive_weight": 0.1,
        "inverse_action_reconstruction_weight": 0.05,
    },
    "seeds": [42, 1729],
    "hf_jobs": {
        "flavor": "a10g-small",
        "region": "us-east-1",
        "timeout_hours": 24,
        "run_name_template": "codelewm-test-{date}-{sha}-seed-{seed}",
        "artifact_repo_id": "abdelstark/codelewm-runs",
        "checkpoint_repo_id": "abdelstark/codelewm-transition-model",
        "checkpoint_revision_template": "v0.6.0-seed-{seed}",
    },
    "claim_gates": {
        "retrieval_min_recall_at_1_lift_over_no_action": 0.05,
        "retrieval_min_mrr_lift_over_no_action": 0.05,
        "collapse_effective_rank_ratio_min": 0.20,
        "collapse_per_dim_variance_median_min": 1.0e-8,
        "collapse_nearest_neighbor_entropy_min": 0.10,
        "surprise_mutation_auc_min": 0.65,
        "surprise_same_problem_different_submission_auc_min": 0.60,
        "surprise_same_code_different_input_auc_min": 0.70,
        "downstream_rerank_pass_at_1_lift_min": 3.0,
        "required_seeds": 2,
    },
    "claim_boundary": {
        "name": "execution_substrate.v1",
        "scope": "v0_6_full_run",
    },
}


def _config_payload(**overrides):
    payload = json.loads(json.dumps(_VALID_CONFIG_DICT))
    for dotted_key, value in overrides.items():
        head, _, tail = dotted_key.partition(".")
        if tail:
            payload[head][tail] = value
        else:
            payload[head] = value
    return payload


class ExecutionTrainConfigLoadTest(unittest.TestCase):
    def test_loads_checked_in_v0_6_yaml(self) -> None:
        from codelewm.training import (
            EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
            load_execution_train_config,
        )

        path = REPO_ROOT / "config/train/scaled/codelewm_execution_v0_6_a10g.yaml"
        cfg = load_execution_train_config(path)
        self.assertEqual(cfg.schema_version, EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION)
        self.assertEqual(cfg.name, "codelewm_execution_v0_6_a10g")
        self.assertEqual(cfg.seeds, (42, 1729))
        self.assertEqual(cfg.loader.batch_size, 64)
        self.assertEqual(cfg.loader.effective_batch_size, 256)
        self.assertEqual(cfg.trainer.max_steps, 50000)
        self.assertEqual(cfg.trainer.precision, "bf16-mixed")
        self.assertEqual(cfg.wm.embed_dim, 256)
        self.assertEqual(cfg.objective.sigreg_weight, 0.09)
        self.assertEqual(cfg.claim_boundary.name, "execution_substrate.v1")

    def test_loads_checked_in_v0_7_yaml(self) -> None:
        """The v0.7 recipe wires the RFC-0015 WS-C architecture levers."""
        from codelewm.training import (
            EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
            load_execution_train_config,
        )

        path = REPO_ROOT / "config/train/scaled/codelewm_execution_v0_7_a10g.yaml"
        cfg = load_execution_train_config(path)
        self.assertEqual(cfg.schema_version, EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION)
        self.assertEqual(cfg.name, "codelewm_execution_v0_7_a10g")
        self.assertEqual(cfg.seeds, (42, 1729))
        # WS-C1: transformer state encoder replaces the v0.6 bag-of-embeddings.
        self.assertEqual(cfg.wm.state_encoder_type, "transformer")
        self.assertEqual(cfg.wm.state_encoder_layers, 4)
        self.assertEqual(cfg.wm.state_encoder_heads, 8)
        self.assertFalse(cfg.wm.enable_ema_target_encoder)
        self.assertEqual(cfg.wm.ema_target_decay, 0.99)
        # WS-C3: in-batch InfoNCE retrieval term is enabled (capped <= 0.10).
        self.assertEqual(cfg.objective.retrieval_weight, 0.05)
        # WS-C5: prediction_mse_weight is now an explicit, applied lever.
        self.assertEqual(cfg.objective.prediction_mse_weight, 1.0)
        # Consumes the bucket-augmented v0.7 pack + v0.7 runtime container.
        self.assertEqual(cfg.data.pack_revision, "v0.7.0-rc1")
        self.assertEqual(
            cfg.hf_jobs.runtime_image,
            "ghcr.io/abdelstark/codelewm-runtime:v0.7",
        )

    def test_loads_checked_in_v0_7_short_yaml(self) -> None:
        """The short v0.7 recipe = full recipe with a guaranteed-to-finish step budget."""
        from codelewm.training import (
            EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
            load_execution_train_config,
        )

        path = REPO_ROOT / "config/train/scaled/codelewm_execution_v0_7_short_a10g.yaml"
        cfg = load_execution_train_config(path)
        self.assertEqual(cfg.schema_version, EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION)
        self.assertEqual(cfg.name, "codelewm_execution_v0_7_short_a10g")
        # Same WS-C recipe as the full profile.
        self.assertEqual(cfg.wm.state_encoder_type, "transformer")
        self.assertEqual(cfg.objective.retrieval_weight, 0.05)
        self.assertEqual(cfg.objective.prediction_mse_weight, 1.0)
        self.assertEqual(cfg.data.pack_revision, "v0.7.0-rc1")
        # Right-sized step budget that fits the 24h wall.
        self.assertEqual(cfg.trainer.max_steps, 15000)
        self.assertLess(cfg.trainer.max_steps, 50000)
        # Distinct run/checkpoint names so uploads never collide with the 50k run.
        self.assertIn("short", cfg.hf_jobs.run_name_template)
        self.assertIn("short", cfg.hf_jobs.checkpoint_revision_template)
        self.assertEqual(
            cfg.hf_jobs.runtime_image,
            "ghcr.io/abdelstark/codelewm-runtime:v0.7-short",
        )

    def test_loads_checked_in_v0_8_yaml(self) -> None:
        """The v0.8 recipe co-trains correctness heads on the pass/fail pack."""
        from codelewm.training import (
            EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
            load_execution_train_config,
        )

        path = REPO_ROOT / "config/train/scaled/codelewm_execution_v0_8_a10g.yaml"
        cfg = load_execution_train_config(path)
        self.assertEqual(cfg.schema_version, EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION)
        self.assertEqual(cfg.name, "codelewm_execution_v0_8_a10g")
        self.assertEqual(cfg.parent_issue, 364)
        self.assertEqual(cfg.implementing_issue, 370)
        self.assertEqual(cfg.seeds, (42, 1729))
        self.assertEqual(cfg.data.pack_revision, "v0.8.0-rc1")
        self.assertEqual(cfg.data.ingestion_sources, ("humaneval",))
        self.assertEqual(cfg.wm.state_encoder_type, "transformer")
        self.assertEqual(cfg.wm.state_encoder_layers, 4)
        self.assertEqual(cfg.wm.state_encoder_heads, 8)
        self.assertTrue(cfg.wm.enable_ema_target_encoder)
        self.assertEqual(cfg.wm.ema_target_decay, 0.99)
        self.assertEqual(cfg.objective.prediction_mse_weight, 1.0)
        self.assertEqual(cfg.objective.retrieval_weight, 0.05)
        self.assertEqual(cfg.objective.p_pass_bce_weight, 0.5)
        self.assertAlmostEqual(
            cfg.objective.p_pass_bce_pos_weight,
            V0_8_PASSFAIL_POS_WEIGHT,
        )
        self.assertEqual(cfg.objective.output_value_ce_weight, 0.2)
        self.assertEqual(cfg.trainer.progress_log_every_n_steps, 10)
        self.assertEqual(
            cfg.hf_jobs.runtime_image,
            "ghcr.io/abdelstark/codelewm-runtime:v0.8",
        )
        self.assertEqual(cfg.claim_gates.required_seeds, 2)
        self.assertEqual(
            cfg.claim_boundary.scope, "v0_8_correctness_co_training"
        )

    def test_loads_checked_in_v0_8_short_yaml(self) -> None:
        """The short v0.8 profile keeps the same recipe with a smaller budget."""
        from codelewm.training import (
            EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
            load_execution_train_config,
        )

        full_path = REPO_ROOT / "config/train/scaled/codelewm_execution_v0_8_a10g.yaml"
        short_path = (
            REPO_ROOT
            / "config/train/scaled/codelewm_execution_v0_8_short_a10g.yaml"
        )
        full = load_execution_train_config(full_path)
        cfg = load_execution_train_config(short_path)
        self.assertEqual(cfg.schema_version, EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION)
        self.assertEqual(cfg.name, "codelewm_execution_v0_8_short_a10g")
        self.assertEqual(cfg.data.pack_revision, full.data.pack_revision)
        self.assertEqual(cfg.wm, full.wm)
        self.assertEqual(cfg.objective, full.objective)
        self.assertEqual(cfg.hf_jobs.runtime_image, full.hf_jobs.runtime_image)
        self.assertEqual(cfg.trainer.max_steps, 12000)
        self.assertLess(cfg.trainer.max_steps, full.trainer.max_steps)
        self.assertEqual(cfg.trainer.checkpoint_every_n_steps, 4000)
        self.assertEqual(cfg.trainer.progress_log_every_n_steps, 10)
        self.assertIn("short", cfg.hf_jobs.run_name_template)
        self.assertIn("short", cfg.hf_jobs.checkpoint_revision_template)

    def test_loads_checked_in_v0_9_short_yaml(self) -> None:
        """The v0.9 profile points at the cross-benchmark pass/fail pack."""
        from codelewm.training import (
            EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
            load_execution_train_config,
        )

        path = (
            REPO_ROOT
            / "config/train/scaled/codelewm_execution_v0_9_short_a10g.yaml"
        )
        cfg = load_execution_train_config(path)
        self.assertEqual(cfg.schema_version, EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION)
        self.assertEqual(cfg.name, "codelewm_execution_v0_9_short_a10g")
        self.assertEqual(cfg.parent_issue, 385)
        self.assertEqual(cfg.implementing_issue, 391)
        self.assertEqual(cfg.seeds, (42, 1729))
        self.assertEqual(cfg.data.pack_revision, "v0.9.0-rc1")
        self.assertEqual(cfg.data.ingestion_sources, ("humaneval", "mbpp_plus"))
        self.assertEqual(cfg.objective.p_pass_bce_weight, 0.5)
        self.assertEqual(
            cfg.objective.p_pass_bce_pos_weight,
            1.0240518038852915,
        )
        self.assertEqual(cfg.objective.output_value_ce_weight, 0.2)
        self.assertEqual(cfg.trainer.max_steps, 12000)
        self.assertEqual(
            cfg.hf_jobs.runtime_image,
            "ghcr.io/abdelstark/codelewm-runtime:v0.9",
        )
        self.assertEqual(cfg.claim_gates.required_seeds, 2)
        self.assertEqual(
            cfg.claim_boundary.scope,
            "v0_9_cross_benchmark_correctness_co_training_short",
        )

    def test_loads_json_round_trip(self) -> None:
        from codelewm.training import (
            ExecutionTrainConfig,
            load_execution_train_config,
        )

        with TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            cfg_path.write_text(
                json.dumps(_config_payload(), indent=2), encoding="utf-8"
            )
            cfg = load_execution_train_config(cfg_path)
            self.assertIsInstance(cfg, ExecutionTrainConfig)
            # to_dict round-trips
            self.assertEqual(
                cfg.to_dict()["loader"]["effective_batch_size"], 8
            )

    def test_peek_returns_schema_version_for_execution_config(self) -> None:
        from codelewm.training import (
            EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
            peek_train_config_schema_version,
        )

        path = REPO_ROOT / "config/train/scaled/codelewm_execution_v0_6_a10g.yaml"
        self.assertEqual(
            peek_train_config_schema_version(path),
            EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
        )

    def test_peek_returns_legacy_schema_for_v1_config(self) -> None:
        from codelewm.training import peek_train_config_schema_version

        path = REPO_ROOT / "config/train/scaled/codelewm_scaled_cpu.yaml"
        if not path.exists():
            self.skipTest(f"legacy config not checked in at {path}")
        self.assertEqual(
            peek_train_config_schema_version(path),
            "codelewm.train_config.v1",
        )

    def test_peek_returns_none_for_missing_file(self) -> None:
        from codelewm.training import peek_train_config_schema_version

        self.assertIsNone(
            peek_train_config_schema_version("/tmp/nonexistent-codelewm-cfg.yaml")
        )


class ExecutionTrainConfigRejectionTest(unittest.TestCase):
    """Each test poisons one field and asserts the validator rejects it."""

    def setUp(self) -> None:
        from codelewm.training import (
            ExecutionTrainConfigError,
            load_execution_train_config,
        )

        self.Error = ExecutionTrainConfigError
        self.load = load_execution_train_config

    def _load_payload(self, payload):
        with TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return self.load(cfg_path)

    def test_unknown_top_level_key_rejected(self) -> None:
        payload = _config_payload()
        payload["mystery"] = 1
        with self.assertRaisesRegex(self.Error, "unknown key"):
            self._load_payload(payload)

    def test_unknown_loader_key_rejected(self) -> None:
        payload = _config_payload()
        payload["loader"]["mystery"] = True
        with self.assertRaisesRegex(self.Error, "loader.*unknown"):
            self._load_payload(payload)

    def test_wrong_schema_version_rejected(self) -> None:
        with self.assertRaisesRegex(self.Error, "schema_version"):
            self._load_payload(_config_payload(schema_version="codelewm.train_config.v1"))

    def test_empty_seeds_rejected(self) -> None:
        payload = _config_payload()
        payload["seeds"] = []
        with self.assertRaisesRegex(self.Error, "seeds"):
            self._load_payload(payload)

    def test_non_integer_seed_rejected(self) -> None:
        payload = _config_payload()
        payload["seeds"] = [42, "1729"]
        # Plain int(value) coercion happens during build, but the
        # __post_init__ won't accept strings; we surface via the helper.
        with self.assertRaises((self.Error, ValueError)):
            self._load_payload(payload)

    def test_effective_batch_mismatch_rejected(self) -> None:
        payload = _config_payload()
        payload["loader"]["effective_batch_size"] = 32  # != 4 * 2
        with self.assertRaisesRegex(self.Error, "effective_batch_size"):
            self._load_payload(payload)

    def test_warmup_steps_must_be_less_than_max_steps(self) -> None:
        payload = _config_payload()
        payload["trainer"]["warmup_steps"] = 100  # == max_steps
        with self.assertRaisesRegex(self.Error, "warmup_steps"):
            self._load_payload(payload)

    def test_history_size_must_be_one(self) -> None:
        payload = _config_payload()
        payload["wm"]["history_size"] = 2
        with self.assertRaisesRegex(self.Error, "history_size"):
            self._load_payload(payload)

    def test_state_encoder_defaults_to_pool(self) -> None:
        # RFC-0015 WS-C1: legacy configs without the field stay on v0.6 pool.
        cfg = self._load_payload(_config_payload())
        self.assertEqual(cfg.wm.state_encoder_type, "pool")

    def test_transformer_state_encoder_parses_and_round_trips(self) -> None:
        payload = _config_payload()
        payload["wm"]["state_encoder_type"] = "transformer"
        payload["wm"]["state_encoder_layers"] = 6
        cfg = self._load_payload(payload)
        self.assertEqual(cfg.wm.state_encoder_type, "transformer")
        self.assertEqual(cfg.wm.state_encoder_layers, 6)
        cfg2 = self._load_payload(cfg.to_dict())
        self.assertEqual(cfg2.wm.state_encoder_type, "transformer")

    def test_ema_target_encoder_parses_and_round_trips(self) -> None:
        payload = _config_payload()
        payload["wm"]["enable_ema_target_encoder"] = True
        payload["wm"]["ema_target_decay"] = 0.95
        cfg = self._load_payload(payload)

        self.assertTrue(cfg.wm.enable_ema_target_encoder)
        self.assertEqual(cfg.wm.ema_target_decay, 0.95)
        cfg2 = self._load_payload(cfg.to_dict())
        self.assertTrue(cfg2.wm.enable_ema_target_encoder)
        self.assertEqual(cfg2.wm.ema_target_decay, 0.95)

    def test_ema_target_encoder_defaults_off(self) -> None:
        cfg = self._load_payload(_config_payload())

        self.assertFalse(cfg.wm.enable_ema_target_encoder)
        self.assertEqual(cfg.wm.ema_target_decay, 0.99)

    def test_invalid_ema_target_decay_rejected(self) -> None:
        payload = _config_payload()
        payload["wm"]["ema_target_decay"] = 1.0
        with self.assertRaisesRegex(self.Error, "ema_target_decay"):
            self._load_payload(payload)

    def test_invalid_state_encoder_type_rejected(self) -> None:
        payload = _config_payload()
        payload["wm"]["state_encoder_type"] = "mlp"
        with self.assertRaisesRegex(self.Error, "state_encoder_type"):
            self._load_payload(payload)

    def test_state_encoder_heads_must_divide_embed_dim(self) -> None:
        payload = _config_payload()
        payload["wm"]["state_encoder_heads"] = 7  # 256 % 7 != 0
        with self.assertRaisesRegex(self.Error, "state_encoder_heads"):
            self._load_payload(payload)

    def test_negative_objective_weight_rejected(self) -> None:
        payload = _config_payload()
        payload["objective"]["sigreg_weight"] = -0.1
        with self.assertRaisesRegex(self.Error, "sigreg_weight"):
            self._load_payload(payload)

    def test_retrieval_weight_defaults_off(self) -> None:
        # RFC-0015 WS-C3: legacy configs without the field keep InfoNCE off.
        cfg = self._load_payload(_config_payload())
        self.assertEqual(cfg.objective.retrieval_weight, 0.0)

    def test_retrieval_weight_parses_and_round_trips(self) -> None:
        payload = _config_payload()
        payload["objective"]["retrieval_weight"] = 0.05
        cfg = self._load_payload(payload)
        self.assertEqual(cfg.objective.retrieval_weight, 0.05)
        cfg2 = self._load_payload(cfg.to_dict())
        self.assertEqual(cfg2.objective.retrieval_weight, 0.05)

    def test_p_pass_bce_parses_and_round_trips(self) -> None:
        payload = _config_payload()
        payload["objective"]["p_pass_bce_weight"] = 0.4
        payload["objective"]["p_pass_bce_pos_weight"] = 2.5
        cfg = self._load_payload(payload)

        self.assertEqual(cfg.objective.p_pass_bce_weight, 0.4)
        self.assertEqual(cfg.objective.p_pass_bce_pos_weight, 2.5)
        cfg2 = self._load_payload(cfg.to_dict())
        self.assertEqual(cfg2.objective.p_pass_bce_weight, 0.4)
        self.assertEqual(cfg2.objective.p_pass_bce_pos_weight, 2.5)

    def test_p_pass_bce_defaults_off(self) -> None:
        cfg = self._load_payload(_config_payload())

        self.assertEqual(cfg.objective.p_pass_bce_weight, 0.0)
        self.assertEqual(cfg.objective.p_pass_bce_pos_weight, 1.0)

    def test_p_pass_bce_invalid_weights_rejected(self) -> None:
        payload = _config_payload()
        payload["objective"]["p_pass_bce_weight"] = -0.1
        with self.assertRaisesRegex(self.Error, "p_pass_bce_weight"):
            self._load_payload(payload)

        payload = _config_payload()
        payload["objective"]["p_pass_bce_pos_weight"] = 0.0
        with self.assertRaisesRegex(self.Error, "p_pass_bce_pos_weight"):
            self._load_payload(payload)

    def test_output_value_ce_weight_parses_and_round_trips(self) -> None:
        payload = _config_payload()
        payload["objective"]["output_value_ce_weight"] = 0.2
        cfg = self._load_payload(payload)

        self.assertEqual(cfg.objective.output_value_ce_weight, 0.2)
        cfg2 = self._load_payload(cfg.to_dict())
        self.assertEqual(cfg2.objective.output_value_ce_weight, 0.2)

    def test_output_value_ce_weight_defaults_off(self) -> None:
        cfg = self._load_payload(_config_payload())

        self.assertEqual(cfg.objective.output_value_ce_weight, 0.0)

    def test_output_value_ce_invalid_weight_rejected(self) -> None:
        payload = _config_payload()
        payload["objective"]["output_value_ce_weight"] = -0.1
        with self.assertRaisesRegex(self.Error, "output_value_ce_weight"):
            self._load_payload(payload)

    def test_retrieval_weight_over_cap_rejected(self) -> None:
        payload = _config_payload()
        payload["objective"]["retrieval_weight"] = 0.2  # > 0.10 cap
        with self.assertRaisesRegex(self.Error, "retrieval_weight"):
            self._load_payload(payload)

    def test_zero_collapse_diag_cadence_rejected(self) -> None:
        payload = _config_payload()
        payload["trainer"]["collapse_diagnostics_every_n_steps"] = 0
        with self.assertRaisesRegex(self.Error, "collapse_diagnostics_every_n_steps"):
            self._load_payload(payload)

    def test_progress_log_cadence_defaults_and_round_trips(self) -> None:
        payload = _config_payload()
        payload["trainer"].pop("progress_log_every_n_steps")
        cfg = self._load_payload(payload)

        self.assertEqual(cfg.trainer.progress_log_every_n_steps, 100)
        cfg2 = self._load_payload(cfg.to_dict())
        self.assertEqual(cfg2.trainer.progress_log_every_n_steps, 100)

    def test_zero_progress_log_cadence_rejected(self) -> None:
        payload = _config_payload()
        payload["trainer"]["progress_log_every_n_steps"] = 0
        with self.assertRaisesRegex(self.Error, "progress_log_every_n_steps"):
            self._load_payload(payload)

    def test_unsupported_optimizer_rejected(self) -> None:
        payload = _config_payload()
        payload["optimizer"]["name"] = "sgd"
        with self.assertRaisesRegex(self.Error, "optimizer"):
            self._load_payload(payload)

    def test_unsupported_extension_rejected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.txt"
            cfg_path.write_text("schema_version: x\n", encoding="utf-8")
            with self.assertRaisesRegex(self.Error, "extension"):
                self.load(cfg_path)


class ExecutionTrainConfigYamlEdgeTest(unittest.TestCase):
    def test_strict_yaml_subset_parses_minimal_doc(self) -> None:
        from codelewm.training import load_execution_train_config

        yaml_text = textwrap.dedent(
            """
            schema_version: codelewm.execution_train_config.v1
            name: tiny
            substrate: execution_trace_v1
            parent_issue: 1
            implementing_issue: 2
            target_substrate_run: v0.6.0
            data:
              pack_repo_id: a/b
              pack_revision: v1
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
              batch_size: 4
              gradient_accumulation_steps: 2
              effective_batch_size: 8
              shuffle: true
            trainer:
              accelerator: cpu
              devices: 1
              precision: float32
              max_steps: 10
              warmup_steps: 2
              cosine_decay_to: 0.0
              gradient_clip_val: 1.0
              checkpoint_every_n_steps: 5
              keep_last_n_checkpoints: 1
              keep_best_by_metric: loss_prediction_mse
              tensorboard_enabled: false
              collapse_diagnostics_every_n_steps: 5
            optimizer:
              name: adamw
              lr: 0.001
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
              timeout_hours: 1
              run_name_template: r-{seed}
              artifact_repo_id: a/b
              checkpoint_repo_id: a/c
              checkpoint_revision_template: v1-seed-{seed}
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
              scope: v0_6_smoke
            """
        ).strip()
        with TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text(yaml_text + "\n", encoding="utf-8")
            cfg = load_execution_train_config(cfg_path)
            self.assertEqual(cfg.seeds, (42,))
            self.assertEqual(cfg.wm.embed_dim, 256)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
