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
    write_hdf5_pack,
)


H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None


def _transition(**overrides: object) -> PackedTransition:
    values: dict[str, object] = {
        "transition_id": "t0",
        "source": "commitpackft",
        "repo": "example/repo",
        "commit": "abc123",
        "path": "pkg/mod.py",
        "split": "train",
        "state_before": TokenSequence(
            input_ids=(1, 2, 3),
            segment_ids=(0, 0, 1),
            changed_hunk_mask=(False, True, True),
        ),
        "state_after": TokenSequence(input_ids=(1, 2, 4)),
        "action_text": TokenSequence(input_ids=(10, 11)),
        "action_abs": TokenSequence(input_ids=(20,)),
        "edit_size": 2,
        "license": "mit",
    }
    values.update(overrides)
    return PackedTransition(**values)  # type: ignore[arg-type]


@unittest.skipUnless(H5PY_AVAILABLE, "h5py is not installed")
class HDF5PackTest(unittest.TestCase):
    def test_hdf5_round_trip_preserves_arrays_metadata_and_feature_flags(self) -> None:
        import h5py

        spec = PackSpec(state_length=4, action_text_length=3, action_abs_length=2)
        rows = [
            _transition(transition_id="t0", split="train"),
            _transition(transition_id="t1", commit="def456", split="val"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.h5"

            artifact = write_hdf5_pack(rows, path, spec=spec)

            self.assertEqual(artifact.kind, "hdf5")
            self.assertEqual(artifact.rows, 2)
            self.assertEqual(len(artifact.sha256), 64)
            with h5py.File(path, "r") as handle:
                self.assertEqual(handle.attrs["schema_version"], "codelewm.transition.v1")
                self.assertFalse(bool(handle.attrs["features.action_patch"]))
                self.assertEqual(handle.attrs["row_count"], 2)
                self.assertEqual(handle["state_before/input_ids"].shape, (2, 4))
                self.assertEqual(handle["state_before/input_ids"][0].tolist(), [1, 2, 3, 0])
                self.assertEqual(
                    handle["state_before/changed_hunk_mask"][0].astype(bool).tolist(),
                    [False, True, True, False],
                )
                self.assertEqual(handle["action_text/attention_mask"][0].astype(bool).tolist(), [True, True, False])
                self.assertNotIn("action_patch", handle)
                self.assertEqual(handle["metadata/split"][:].tolist(), [0, 1])
                self.assertEqual(handle["metadata/source"][:].tolist(), [1, 1])
                self.assertEqual(handle["metadata/repo"].asstr()[:].tolist(), ["example/repo", "example/repo"])

    def test_action_patch_group_is_written_when_feature_flag_is_enabled(self) -> None:
        import h5py

        spec = PackSpec(
            state_length=4,
            action_text_length=3,
            action_abs_length=2,
            action_patch_length=3,
            include_action_patch=True,
        )
        row = _transition(action_patch=TokenSequence(input_ids=(30, 31)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.h5"

            write_hdf5_pack([row], path, spec=spec)

            with h5py.File(path, "r") as handle:
                self.assertTrue(bool(handle.attrs["features.action_patch"]))
                self.assertEqual(handle["action_patch/input_ids"][0].tolist(), [30, 31, 0])

    def test_missing_patch_tokens_fail_when_patch_feature_is_enabled(self) -> None:
        spec = PackSpec(
            state_length=4,
            action_text_length=3,
            action_abs_length=2,
            include_action_patch=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.h5"

            with self.assertRaisesRegex(PackError, "missing action_patch"):
                write_hdf5_pack([_transition()], path, spec=spec)

    def test_overlong_sequence_fails_instead_of_truncating(self) -> None:
        spec = PackSpec(state_length=2, action_text_length=3, action_abs_length=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.h5"

            with self.assertRaisesRegex(PackError, "exceeds fixed width"):
                write_hdf5_pack([_transition()], path, spec=spec)


if __name__ == "__main__":
    unittest.main()
