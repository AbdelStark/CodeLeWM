from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from codelewm.data import (
    PackError,
    PackSpec,
    PackedTransition,
    TokenSequence,
    read_dataset_manifest,
    validate_manifest_checksums,
    write_dataset_artifacts,
)


H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None
PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


def _transition(**overrides: object) -> PackedTransition:
    values: dict[str, object] = {
        "transition_id": "t0",
        "source": "commitpackft",
        "repo": "example/repo",
        "commit": "abc123",
        "path": "pkg/mod.py",
        "split": "train",
        "state_before": TokenSequence(input_ids=(1, 2, 3)),
        "state_after": TokenSequence(input_ids=(1, 2, 4)),
        "action_text": TokenSequence(input_ids=(10, 11)),
        "action_abs": TokenSequence(input_ids=(20,)),
        "edit_size": 2,
        "license": "mit",
        "dedup_keys": ("abc", "def"),
    }
    values.update(overrides)
    return PackedTransition(**values)  # type: ignore[arg-type]


@unittest.skipUnless(H5PY_AVAILABLE and PYARROW_AVAILABLE, "h5py and pyarrow are not installed")
class DatasetManifestTest(unittest.TestCase):
    def test_dataset_artifacts_manifest_records_checksums_and_feature_flags(self) -> None:
        spec = PackSpec(state_length=4, action_text_length=3, action_abs_length=2)
        rows = [
            _transition(transition_id="t0", split="train"),
            _transition(transition_id="t1", commit="def456", split="test", source="synthetic"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            manifest = write_dataset_artifacts(rows, output_dir, spec=spec, parquet_shard_size=1)
            loaded = read_dataset_manifest(output_dir / "manifest.json")

            self.assertEqual(loaded.to_dict(), manifest.to_dict())
            self.assertEqual(loaded.row_count, 2)
            self.assertEqual(loaded.features, {"action_patch": False})
            self.assertEqual(loaded.split_counts, {"train": 1, "val": 0, "test": 1})
            self.assertEqual(loaded.source_counts["commitpackft"], 1)
            self.assertEqual(loaded.source_counts["synthetic"], 1)
            self.assertEqual({artifact.kind for artifact in loaded.artifacts}, {"parquet", "hdf5"})
            self.assertIn("dataset.h5", {artifact.path for artifact in loaded.artifacts})
            self.assertTrue(all(len(artifact.sha256) == 64 for artifact in loaded.artifacts))
            validate_manifest_checksums(loaded, root=output_dir)

    def test_manifest_checksum_validation_rejects_tampered_artifact(self) -> None:
        spec = PackSpec(state_length=4, action_text_length=3, action_abs_length=2)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest = write_dataset_artifacts([_transition()], output_dir, spec=spec)
            hdf5_artifact = next(artifact for artifact in manifest.artifacts if artifact.kind == "hdf5")
            with (output_dir / hdf5_artifact.path).open("ab") as handle:
                handle.write(b"tamper")

            with self.assertRaisesRegex(PackError, "checksum mismatch"):
                validate_manifest_checksums(manifest, root=output_dir)


if __name__ == "__main__":
    unittest.main()
