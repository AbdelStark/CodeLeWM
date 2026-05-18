from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from codelewm.model import LATENT_DIM
from codelewm.training import (
    TRAIN_CONFIG_SCHEMA_VERSION,
    TrainConfigError,
    default_train_config_paths,
    load_default_train_configs,
    load_train_config,
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

    def test_validation_rejects_image_control_data_paths(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["data"]["train"] = "data/pusht/train.hdf5"

        with self.assertRaisesRegex(TrainConfigError, "image-control"):
            validate_train_config(payload)

    def test_validation_rejects_unknown_schema_keys(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["img_size"] = 224

        with self.assertRaisesRegex(TrainConfigError, "unknown key"):
            validate_train_config(payload)


if __name__ == "__main__":
    unittest.main()
