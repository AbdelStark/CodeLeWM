from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.model import (
    CHECKPOINT_SCHEMA_VERSION,
    LATENT_DIM,
    CheckpointCompatibilityError,
    CheckpointCompatibilitySpec,
    CheckpointManifest,
    CheckpointMetadata,
    build_checkpoint_metadata,
    compute_config_hash,
    load_checkpoint_manifest,
    migrate_checkpoint_manifest,
    read_checkpoint_manifest,
    validate_checkpoint_compatibility,
    write_checkpoint_manifest,
)


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_manifest_round_trip_records_compatibility_fields_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.state"
            manifest_path = root / "manifest.json"
            checkpoint.write_bytes(b"weights")
            config = {"model": {"latent_dim": LATENT_DIM}, "seed": 7}
            metadata = build_checkpoint_metadata(config, action_view="text")

            manifest = write_checkpoint_manifest(
                metadata=metadata,
                checkpoint_path=checkpoint,
                manifest_path=manifest_path,
            )
            loaded = read_checkpoint_manifest(manifest_path)

            self.assertEqual(loaded.to_dict(), manifest.to_dict())
            self.assertEqual(loaded.schema_version, CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(loaded.metadata.record_schema_version, "codelewm.transition.v1")
            self.assertEqual(loaded.metadata.latent_dim, LATENT_DIM)
            self.assertEqual(loaded.metadata.action_view, "text")
            self.assertEqual(loaded.metadata.config_hash, compute_config_hash(config))
            self.assertEqual(loaded.checkpoint_path, "checkpoint.state")

    def test_load_manifest_refuses_incompatible_metadata(self) -> None:
        config_hash = compute_config_hash({"seed": 1})
        manifest = CheckpointManifest(
            metadata=CheckpointMetadata(config_hash=config_hash, action_view="abstract"),
            checkpoint_path="checkpoint.state",
            checkpoint_sha256="0" * 64,
        )

        with self.assertRaisesRegex(CheckpointCompatibilityError, "action_view"):
            validate_checkpoint_compatibility(
                manifest,
                CheckpointCompatibilitySpec(config_hash=config_hash, action_view="text"),
            )

        with self.assertRaisesRegex(CheckpointCompatibilityError, "config_hash"):
            validate_checkpoint_compatibility(
                manifest,
                CheckpointCompatibilitySpec(
                    config_hash=compute_config_hash({"seed": 2}),
                    action_view="abstract",
                ),
            )

    def test_load_manifest_verifies_checksum_before_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.state"
            manifest_path = root / "manifest.json"
            checkpoint.write_bytes(b"first")
            config = {"seed": 3}
            metadata = build_checkpoint_metadata(config)
            write_checkpoint_manifest(
                metadata=metadata,
                checkpoint_path=checkpoint,
                manifest_path=manifest_path,
            )
            checkpoint.write_bytes(b"tampered")

            with self.assertRaisesRegex(CheckpointCompatibilityError, "checksum mismatch"):
                load_checkpoint_manifest(
                    manifest_path,
                    expected=CheckpointCompatibilitySpec(config_hash=metadata.config_hash),
                )

    def test_manifest_rejects_unsafe_checkpoint_paths(self) -> None:
        manifest = CheckpointManifest(
            metadata=CheckpointMetadata(config_hash=compute_config_hash({"seed": 1})),
            checkpoint_path="../checkpoint.state",
            checkpoint_sha256="0" * 64,
        )

        payload = manifest.to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(CheckpointCompatibilityError, "relative"):
                load_checkpoint_manifest(path)

    def test_unsupported_schema_records_migration_hook_without_running_it(self) -> None:
        manifest = CheckpointManifest(
            schema_version="codelewm.checkpoint.v0",
            metadata=CheckpointMetadata(
                schema_version="codelewm.checkpoint.v0",
                config_hash=compute_config_hash({"seed": 1}),
            ),
            checkpoint_path="checkpoint.state",
            checkpoint_sha256="0" * 64,
            migration_hook="upgrade_v0_to_v1",
        )

        with self.assertRaisesRegex(CheckpointCompatibilityError, "not run implicitly"):
            validate_checkpoint_compatibility(
                manifest,
                CheckpointCompatibilitySpec(config_hash=manifest.metadata.config_hash),
            )
        with self.assertRaisesRegex(CheckpointCompatibilityError, "migration is not implemented"):
            migrate_checkpoint_manifest(
                manifest,
                expected=CheckpointCompatibilitySpec(config_hash=manifest.metadata.config_hash),
            )

    def test_config_hash_is_deterministic_for_json_native_configs(self) -> None:
        left = {"b": [2, 3], "a": {"x": True}}
        right = {"a": {"x": True}, "b": [2, 3]}

        self.assertEqual(compute_config_hash(left), compute_config_hash(right))


if __name__ == "__main__":
    unittest.main()
