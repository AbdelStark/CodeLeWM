from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from codelewm.model import LATENT_DIM, compute_config_hash
from codelewm.training import (
    TRAIN_CONFIG_SCHEMA_VERSION,
    TrainConfigError,
    compatibility_config_payload,
    default_train_config_paths,
    load_default_train_configs,
    load_scaled_train_configs,
    load_train_config,
    scaled_train_config_paths,
    validate_train_config,
)


ROOT = Path(__file__).resolve().parents[2]


class TrainConfigLoadTest(unittest.TestCase):
    def test_default_train_configs_load(self) -> None:
        tiny, small = load_default_train_configs(ROOT)

        self.assertEqual(tiny.schema_version, TRAIN_CONFIG_SCHEMA_VERSION)
        self.assertEqual(small.schema_version, TRAIN_CONFIG_SCHEMA_VERSION)
        self.assertEqual(tiny.name, "codelewm_tiny")
        self.assertEqual(small.name, "codelewm_small")
        self.assertEqual(tiny.wm.history_size, 1)
        self.assertEqual(tiny.wm.num_preds, 1)
        self.assertEqual(tiny.wm.embed_dim, LATENT_DIM)
        self.assertEqual(tiny.wm.action_view, "text")
        self.assertEqual(tiny.wm.action_sequence_length, 256)
        self.assertEqual(small.wm.history_size, 1)
        self.assertEqual(small.wm.num_preds, 1)
        self.assertEqual(small.wm.embed_dim, LATENT_DIM)
        self.assertEqual(small.wm.action_view, "text")
        self.assertEqual(small.wm.action_sequence_length, 256)

    def test_tiny_config_is_cpu_smoke_sized(self) -> None:
        tiny = load_train_config(ROOT / "config/train/codelewm_tiny.yaml")

        self.assertEqual(tiny.trainer.accelerator, "cpu")
        self.assertEqual(tiny.trainer.devices, 1)
        self.assertEqual(tiny.trainer.precision, "float32")
        self.assertEqual(tiny.loader.batch_size, 4)
        self.assertEqual(tiny.loader.num_workers, 0)
        self.assertLessEqual(tiny.trainer.max_steps, 16)

    def test_small_config_matches_single_device_training_defaults(self) -> None:
        small = load_train_config(ROOT / "config/train/codelewm_small.yaml")

        self.assertEqual(small.trainer.accelerator, "auto")
        self.assertEqual(small.trainer.devices, 1)
        self.assertEqual(small.trainer.precision, "bf16-mixed")
        self.assertEqual(small.trainer.max_steps, 10000)
        self.assertEqual(small.loader.batch_size, 64)

    def test_default_train_configs_are_json_native(self) -> None:
        tiny, small = load_default_train_configs(ROOT)

        json.dumps(tiny.to_dict(), sort_keys=True, allow_nan=False)
        json.dumps(small.to_dict(), sort_keys=True, allow_nan=False)

    def test_default_paths_helper_points_to_checked_in_configs(self) -> None:
        tiny_path, small_path = default_train_config_paths(ROOT)

        self.assertEqual(tiny_path.name, "codelewm_tiny.yaml")
        self.assertEqual(small_path.name, "codelewm_small.yaml")
        self.assertTrue(tiny_path.exists())
        self.assertTrue(small_path.exists())

    def test_default_configs_do_not_reference_image_control_datasets(self) -> None:
        forbidden = ("pusht", "dmc", "tworoom", "ogbench", "pixels", "proprio")
        for path in default_train_config_paths(ROOT):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8").lower()
                for token in forbidden:
                    self.assertNotIn(token, text)


class ScaledTrainConfigLoadTest(unittest.TestCase):
    def test_scaled_train_configs_load_with_text_action_headline(self) -> None:
        configs = {config.name: config for config in load_scaled_train_configs(ROOT)}

        self.assertEqual(
            set(configs),
            {
                "codelewm_scaled_cpu",
                "codelewm_scaled_mps",
                "codelewm_scaled_gpu_a10g",
                "codelewm_scaled_action_use_margin_gpu_a10g",
                "codelewm_scaled_action_use_margin_retrieval_gpu_a10g",
                "codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g",
            },
        )
        for config in configs.values():
            with self.subTest(config=config.name):
                self.assertEqual(config.schema_version, TRAIN_CONFIG_SCHEMA_VERSION)
                self.assertEqual(config.seed, 240119)
                self.assertEqual(config.wm.history_size, 1)
                self.assertEqual(config.wm.num_preds, 1)
                self.assertEqual(config.wm.embed_dim, LATENT_DIM)
                self.assertEqual(config.wm.action_view, "text")
                self.assertEqual(config.wm.action_sequence_length, 256)

        for name in ("codelewm_scaled_cpu", "codelewm_scaled_mps", "codelewm_scaled_gpu_a10g"):
            with self.subTest(config=name):
                self.assertEqual(configs[name].wm.action_fusion, "conditional_transformer")
                self.assertFalse(configs[name].loss.enable_retrieval_loss)
                self.assertEqual(configs[name].loss.retrieval_weight, 0.0)
                self.assertFalse(configs[name].loss.enable_action_use_margin)
                self.assertEqual(configs[name].loss.action_use_margin_weight, 0.0)
                self.assertEqual(configs[name].loss.action_use_margin, 0.0)
                self.assertFalse(configs[name].loss.enable_action_swap_contrastive)
                self.assertEqual(configs[name].loss.action_swap_contrastive_weight, 0.0)
                self.assertEqual(configs[name].loss.action_swap_contrastive_margin, 0.0)
                self.assertFalse(configs[name].loss.enable_inverse_action_reconstruction)
                self.assertEqual(configs[name].loss.inverse_action_reconstruction_weight, 0.0)

        action_margin = configs["codelewm_scaled_action_use_margin_gpu_a10g"]
        self.assertTrue(action_margin.loss.enable_action_use_margin)
        self.assertEqual(action_margin.loss.action_use_margin_weight, 0.25)
        self.assertEqual(action_margin.loss.action_use_margin, 0.02)
        self.assertFalse(action_margin.loss.enable_retrieval_loss)
        self.assertEqual(action_margin.loss.retrieval_weight, 0.0)

        action_retrieval = configs["codelewm_scaled_action_use_margin_retrieval_gpu_a10g"]
        self.assertTrue(action_retrieval.loss.enable_action_use_margin)
        self.assertEqual(action_retrieval.loss.action_use_margin_weight, 0.25)
        self.assertEqual(action_retrieval.loss.action_use_margin, 0.02)
        self.assertTrue(action_retrieval.loss.enable_retrieval_loss)
        self.assertEqual(action_retrieval.loss.retrieval_weight, 0.05)

        v0_2 = configs["codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g"]
        self.assertEqual(v0_2.wm.action_fusion, "gated_residual")
        self.assertTrue(v0_2.loss.enable_action_use_margin)
        self.assertEqual(v0_2.loss.action_use_margin_weight, 0.25)
        self.assertEqual(v0_2.loss.action_use_margin, 0.02)
        self.assertTrue(v0_2.loss.enable_action_swap_contrastive)
        self.assertEqual(v0_2.loss.action_swap_contrastive_weight, 0.20)
        self.assertEqual(v0_2.loss.action_swap_contrastive_margin, 0.05)
        self.assertTrue(v0_2.loss.enable_inverse_action_reconstruction)
        self.assertEqual(v0_2.loss.inverse_action_reconstruction_weight, 0.10)
        self.assertFalse(v0_2.loss.enable_retrieval_loss)

    def test_scaled_config_budgets_match_hardware_profiles(self) -> None:
        configs = {config.name: config for config in load_scaled_train_configs(ROOT)}

        cpu = configs["codelewm_scaled_cpu"]
        self.assertEqual(cpu.trainer.accelerator, "cpu")
        self.assertEqual(cpu.trainer.precision, "float32")
        self.assertEqual(cpu.loader.batch_size, 8)
        self.assertEqual(cpu.trainer.max_steps, 2048)

        mps = configs["codelewm_scaled_mps"]
        self.assertEqual(mps.trainer.accelerator, "mps")
        self.assertEqual(mps.trainer.precision, "float32")
        self.assertEqual(mps.loader.batch_size, 32)
        self.assertEqual(mps.trainer.max_steps, 10000)

        gpu = configs["codelewm_scaled_gpu_a10g"]
        self.assertEqual(gpu.trainer.accelerator, "gpu")
        self.assertEqual(gpu.trainer.precision, "bf16-mixed")
        self.assertEqual(gpu.loader.batch_size, 64)
        self.assertGreaterEqual(gpu.trainer.max_steps, 60000)
        self.assertLessEqual(gpu.trainer.max_steps, 100000)

        action_margin = configs["codelewm_scaled_action_use_margin_gpu_a10g"]
        self.assertEqual(action_margin.trainer.accelerator, "gpu")
        self.assertEqual(action_margin.trainer.precision, "bf16-mixed")
        self.assertEqual(action_margin.loader.batch_size, 64)
        self.assertEqual(action_margin.trainer.max_steps, 60000)

        action_retrieval = configs["codelewm_scaled_action_use_margin_retrieval_gpu_a10g"]
        self.assertEqual(action_retrieval.trainer.accelerator, "gpu")
        self.assertEqual(action_retrieval.trainer.precision, "bf16-mixed")
        self.assertEqual(action_retrieval.loader.batch_size, 64)
        self.assertEqual(action_retrieval.trainer.max_steps, 60000)

        v0_2 = configs["codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g"]
        self.assertEqual(v0_2.trainer.accelerator, "gpu")
        self.assertEqual(v0_2.trainer.precision, "bf16-mixed")
        self.assertEqual(v0_2.loader.batch_size, 64)
        self.assertEqual(v0_2.trainer.max_steps, 60000)

    def test_scaled_paths_helper_points_to_checked_in_configs(self) -> None:
        paths = scaled_train_config_paths(ROOT)

        self.assertEqual(
            tuple(path.name for path in paths),
            (
                "codelewm_scaled_cpu.yaml",
                "codelewm_scaled_mps.yaml",
                "codelewm_scaled_gpu_a10g.yaml",
                "codelewm_scaled_action_use_margin_gpu_a10g.yaml",
                "codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml",
                "codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml",
            ),
        )
        for path in paths:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())

    def test_scaled_train_configs_are_json_native_and_hashable(self) -> None:
        for config in load_scaled_train_configs(ROOT):
            with self.subTest(config=config.name):
                payload = config.to_dict()
                json.dumps(payload, sort_keys=True, allow_nan=False)
                digest = compute_config_hash(payload)
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_scaled_config_validation_script_reports_hashes_and_seeds(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/validate-training-configs"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema_version"], "codelewm.train_config_validation.v1")
        self.assertEqual(payload["train_config_schema_version"], TRAIN_CONFIG_SCHEMA_VERSION)
        self.assertEqual(len(payload["configs"]), 6)
        for item in payload["configs"]:
            with self.subTest(config=item["name"]):
                self.assertEqual(item["seed"], 240119)
                self.assertEqual(item["action_view"], "text")
                self.assertIn("action_fusion", item)
                self.assertRegex(item["config_sha256"], r"^[0-9a-f]{64}$")
                self.assertIn("action_use_margin_enabled", item)
                self.assertIn("action_use_margin_weight", item)
                self.assertIn("action_use_margin", item)
                self.assertIn("action_swap_contrastive_enabled", item)
                self.assertIn("action_swap_contrastive_weight", item)
                self.assertIn("action_swap_contrastive_margin", item)
                self.assertIn("inverse_action_reconstruction_enabled", item)
                self.assertIn("inverse_action_reconstruction_weight", item)
        by_name = {item["name"]: item for item in payload["configs"]}
        self.assertFalse(by_name["codelewm_scaled_gpu_a10g"]["action_use_margin_enabled"])
        self.assertTrue(by_name["codelewm_scaled_action_use_margin_gpu_a10g"]["action_use_margin_enabled"])
        self.assertTrue(
            by_name["codelewm_scaled_action_use_margin_retrieval_gpu_a10g"]["retrieval_loss_enabled"]
        )
        self.assertEqual(by_name["codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g"]["action_fusion"], "gated_residual")
        self.assertTrue(
            by_name["codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g"][
                "action_swap_contrastive_enabled"
            ]
        )
        self.assertTrue(
            by_name["codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g"][
                "inverse_action_reconstruction_enabled"
            ]
        )


class TrainConfigValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_train_config(ROOT / "config/train/codelewm_tiny.yaml").to_dict()

    def test_validation_rejects_non_v0_1_history(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["wm"]["history_size"] = 3

        with self.assertRaisesRegex(TrainConfigError, "history_size"):
            validate_train_config(payload)

    def test_validation_rejects_multi_step_prediction(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["wm"]["num_preds"] = 2

        with self.assertRaisesRegex(TrainConfigError, "num_preds"):
            validate_train_config(payload)

    def test_validation_rejects_patch_action_as_training_default(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["wm"]["action_view"] = "patch"
        payload["wm"]["action_sequence_length"] = 512

        with self.assertRaisesRegex(TrainConfigError, "patch"):
            validate_train_config(payload)

    def test_validation_rejects_action_sequence_length_mismatch(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["wm"]["action_sequence_length"] = 192

        with self.assertRaisesRegex(TrainConfigError, "action_sequence_length"):
            validate_train_config(payload)

    def test_validation_rejects_unknown_action_fusion(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["wm"]["action_fusion"] = "cross_attention"

        with self.assertRaisesRegex(TrainConfigError, "action_fusion"):
            validate_train_config(payload)

    def test_validation_accepts_abstract_action_length(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["wm"]["action_view"] = "abstract"
        payload["wm"]["action_sequence_length"] = 192

        config = validate_train_config(payload)

        self.assertEqual(config.wm.action_view, "abstract")
        self.assertEqual(config.wm.action_sequence_length, 192)

    def test_validation_rejects_retrieval_loss_without_gate(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["loss"]["retrieval_weight"] = 0.05

        with self.assertRaisesRegex(TrainConfigError, "enable_retrieval_loss"):
            validate_train_config(payload)

    def test_validation_rejects_action_use_margin_without_gate(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["loss"]["action_use_margin_weight"] = 0.25
        payload["loss"]["action_use_margin"] = 0.02

        with self.assertRaisesRegex(TrainConfigError, "enable_action_use_margin"):
            validate_train_config(payload)

    def test_validation_rejects_action_use_margin_gate_without_weight_and_margin(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["loss"]["enable_action_use_margin"] = True

        with self.assertRaisesRegex(TrainConfigError, "action_use_margin_weight"):
            validate_train_config(payload)

    def test_compatibility_hash_omits_disabled_action_margin_defaults(self) -> None:
        config = validate_train_config(self.payload)
        payload = compatibility_config_payload(config)

        self.assertNotIn("action_fusion", payload["wm"])
        self.assertNotIn("enable_action_use_margin", payload["loss"])
        self.assertNotIn("action_use_margin_weight", payload["loss"])
        self.assertNotIn("action_use_margin", payload["loss"])
        self.assertNotIn("enable_action_swap_contrastive", payload["loss"])
        self.assertNotIn("action_swap_contrastive_weight", payload["loss"])
        self.assertNotIn("action_swap_contrastive_margin", payload["loss"])
        self.assertNotIn("enable_inverse_action_reconstruction", payload["loss"])
        self.assertNotIn("inverse_action_reconstruction_weight", payload["loss"])

    def test_compatibility_hash_records_enabled_action_margin_surface(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["loss"]["enable_action_use_margin"] = True
        payload["loss"]["action_use_margin_weight"] = 0.25
        payload["loss"]["action_use_margin"] = 0.02
        config = validate_train_config(payload)
        compatibility_payload = compatibility_config_payload(config)

        self.assertTrue(compatibility_payload["loss"]["enable_action_use_margin"])
        self.assertEqual(compatibility_payload["loss"]["action_use_margin_weight"], 0.25)
        self.assertEqual(compatibility_payload["loss"]["action_use_margin"], 0.02)

    def test_compatibility_hash_records_enabled_v0_2_surfaces(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["wm"]["action_fusion"] = "gated_residual"
        payload["loss"]["enable_action_swap_contrastive"] = True
        payload["loss"]["action_swap_contrastive_weight"] = 0.20
        payload["loss"]["action_swap_contrastive_margin"] = 0.05
        payload["loss"]["enable_inverse_action_reconstruction"] = True
        payload["loss"]["inverse_action_reconstruction_weight"] = 0.10
        config = validate_train_config(payload)
        compatibility_payload = compatibility_config_payload(config)

        self.assertEqual(compatibility_payload["wm"]["action_fusion"], "gated_residual")
        self.assertTrue(compatibility_payload["loss"]["enable_action_swap_contrastive"])
        self.assertEqual(compatibility_payload["loss"]["action_swap_contrastive_weight"], 0.20)
        self.assertEqual(compatibility_payload["loss"]["action_swap_contrastive_margin"], 0.05)
        self.assertTrue(compatibility_payload["loss"]["enable_inverse_action_reconstruction"])
        self.assertEqual(compatibility_payload["loss"]["inverse_action_reconstruction_weight"], 0.10)

    def test_validation_rejects_image_control_data_paths(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["data"]["train"] = "data/pusht/train.hdf5"

        with self.assertRaisesRegex(TrainConfigError, "image-control"):
            validate_train_config(payload)

    def test_validation_allows_forbidden_token_inside_unrelated_path_component(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["data"]["train"] = "/tmp/tmpdmcol33k/data/train.hdf5"
        payload["data"]["val"] = "/tmp/tmpdmcol33k/data/val.hdf5"
        payload["data"]["manifest"] = "/tmp/tmpdmcol33k/data/manifest.json"

        config = validate_train_config(payload)

        self.assertEqual(config.data.train, payload["data"]["train"])

    def test_validation_rejects_unknown_schema_keys(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["img_size"] = 224

        with self.assertRaisesRegex(TrainConfigError, "unknown key"):
            validate_train_config(payload)


if __name__ == "__main__":
    unittest.main()
