from __future__ import annotations

import unittest
from pathlib import Path

from codelewm.data import DATASET_BUILD_CONFIG_SCHEMA_VERSION, load_dataset_build_config


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SHARD_CONFIG = ROOT / "config" / "data" / "codelewm_public_shard_commitpackft_python.json"


class PublicShardConfigTest(unittest.TestCase):
    def test_commitpackft_python_public_shard_config_is_valid(self) -> None:
        config = load_dataset_build_config(PUBLIC_SHARD_CONFIG)

        self.assertEqual(config.schema_version, DATASET_BUILD_CONFIG_SCHEMA_VERSION)
        self.assertEqual(config.name, "codelewm-public-shard-commitpackft-python-v0-3")
        self.assertEqual(config.seed, 240119)
        self.assertEqual(len(config.sources), 1)

        source = config.sources[0]
        self.assertEqual(source.source, "commitpackft")
        self.assertEqual(source.name, "bigcode-commitpackft-python")
        self.assertEqual(source.path, "../../.artifacts/hf-sources/commitpackft/data/python/data.jsonl")
        self.assertEqual(source.options, {"language": "Python"})
        self.assertFalse(config.action.include_patch)
        self.assertTrue(config.license.require_license_field)
        self.assertEqual(config.license.artifact_policy, "full_text")
        self.assertIn("mit", config.license.allowed_licenses)
        self.assertIn("apache-2.0", config.license.allowed_licenses)


if __name__ == "__main__":
    unittest.main()
