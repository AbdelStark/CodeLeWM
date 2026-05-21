from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from codelewm.observability import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ArtifactManifestError,
    artifact_manifest_json_schema,
    build_artifact_manifest,
    build_manifest_file,
    compute_json_sha256,
    read_artifact_manifest,
    validate_artifact_checksums,
    validate_artifact_manifest_payload,
    write_artifact_manifest,
)


SOURCE_SHA = "a" * 40
CREATED_AT = "2026-05-18T00:00:00Z"


class ArtifactManifestSchemaTest(unittest.TestCase):
    def test_manifest_round_trip_records_lineage_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports").mkdir()
            (root / "config.yaml").write_text("seed: 1337\n", encoding="utf-8")
            (root / "reports" / "metrics.json").write_text('{"loss": 0.5}\n', encoding="utf-8")

            manifest = build_artifact_manifest(
                artifact_kind="training_run",
                root=root,
                files=("config.yaml", "reports/metrics.json"),
                command=("codelewm", "train", "--config", "config/train/codelewm_tiny.yaml"),
                config={"seed": 1337, "wm": {"history_size": 1}},
                parent_artifacts=("dataset-parent",),
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
                metadata={"python_version": "3.14.3", "device": "cpu"},
            )
            manifest_path = root / "manifest.json"

            write_artifact_manifest(manifest, manifest_path)
            loaded = read_artifact_manifest(manifest_path)

            self.assertEqual(loaded.to_dict(), manifest.to_dict())
            self.assertEqual(loaded.schema_version, ARTIFACT_MANIFEST_SCHEMA_VERSION)
            self.assertEqual(loaded.artifact_kind, "training_run")
            self.assertEqual(loaded.source_git_sha, SOURCE_SHA)
            self.assertEqual(loaded.parent_artifacts, ("dataset-parent",))
            self.assertEqual(len(loaded.files), 2)
            self.assertTrue(all(len(file.sha256) == 64 for file in loaded.files))
            validate_artifact_checksums(loaded, root=root)

    def test_checksum_validation_rejects_tampered_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "report.json"
            artifact.write_text('{"ok": true}\n', encoding="utf-8")
            manifest = build_artifact_manifest(
                artifact_kind="eval_report",
                root=root,
                files=(artifact,),
                command=("codelewm", "eval", "retrieval"),
                config={"eval": "retrieval"},
                source_git_sha=SOURCE_SHA,
                created_at=CREATED_AT,
            )
            artifact.write_text('{"ok": null}\n', encoding="utf-8")

            with self.assertRaisesRegex(ArtifactManifestError, "checksum mismatch"):
                validate_artifact_checksums(manifest, root=root)

    def test_build_manifest_rejects_files_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            outside_file = Path(outside_tmp) / "outside.txt"
            outside_file.write_text("outside\n", encoding="utf-8")

            with self.assertRaisesRegex(ArtifactManifestError, "artifact root"):
                build_manifest_file(outside_file, root=Path(root_tmp))

    def test_payload_validation_rejects_unsafe_paths(self) -> None:
        payload = self._valid_payload()
        payload["files"][0]["path"] = "../secret.txt"

        with self.assertRaisesRegex(ArtifactManifestError, "relative"):
            validate_artifact_manifest_payload(payload)

    def test_payload_validation_rejects_duplicate_parents(self) -> None:
        payload = self._valid_payload()
        payload["parent_artifacts"] = ["dataset-a", "dataset-a"]

        with self.assertRaisesRegex(ArtifactManifestError, "duplicates"):
            validate_artifact_manifest_payload(payload)

    def test_metadata_must_be_json_native(self) -> None:
        payload = self._valid_payload()
        payload["metadata"] = {"bad": float("nan")}

        with self.assertRaisesRegex(ArtifactManifestError, "JSON-native"):
            validate_artifact_manifest_payload(payload)

    def test_metadata_field_is_required(self) -> None:
        payload = self._valid_payload()
        del payload["metadata"]

        with self.assertRaisesRegex(ArtifactManifestError, "metadata"):
            validate_artifact_manifest_payload(payload)

    def test_json_schema_exposes_required_manifest_contract(self) -> None:
        schema = artifact_manifest_json_schema()

        self.assertEqual(schema["$id"], ARTIFACT_MANIFEST_SCHEMA_VERSION)
        self.assertIn("candidate_pack", schema["properties"]["artifact_kind"]["enum"])
        self.assertIn("training_run", schema["properties"]["artifact_kind"]["enum"])
        for field in (
            "artifact_id",
            "source_git_sha",
            "command",
            "config_sha256",
            "parent_artifacts",
            "files",
        ):
            self.assertIn(field, schema["required"])

    def test_config_hash_is_deterministic_for_json_native_payloads(self) -> None:
        left = {"b": [2, 3], "a": {"enabled": True}}
        right = {"a": {"enabled": True}, "b": [2, 3]}

        self.assertEqual(compute_json_sha256(left), compute_json_sha256(right))

    def _valid_payload(self) -> dict[str, object]:
        return copy.deepcopy(
            {
                "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
                "artifact_id": "training_run-abc123",
                "artifact_kind": "training_run",
                "created_at": CREATED_AT,
                "source_git_sha": SOURCE_SHA,
                "command": ["codelewm", "train"],
                "config_sha256": compute_json_sha256({"seed": 1337}),
                "parent_artifacts": ["dataset-a"],
                "files": [
                    {
                        "path": "config.yaml",
                        "sha256": "0" * 64,
                        "bytes": 12,
                    }
                ],
                "metadata": {"device": "cpu"},
            }
        )


if __name__ == "__main__":
    unittest.main()
