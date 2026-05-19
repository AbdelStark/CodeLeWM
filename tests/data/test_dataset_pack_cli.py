from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codelewm.data import (
    DATASET_PACK_REPORT_SCHEMA_VERSION,
    OptionalDependencyError,
    PackError,
    build_dataset_from_config_path,
    pack_dataset_from_manifest,
    read_dataset_manifest,
    sha256_file,
)
from codelewm.data.dataset_pack import _require_pack_dependencies
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)


H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None
PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
DATA_DEPS_AVAILABLE = H5PY_AVAILABLE and PYARROW_AVAILABLE
ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "tests" / "fixtures" / "dataset_build" / "config.json"


class DatasetPackCliTest(unittest.TestCase):
    def test_missing_optional_dependency_message_names_data_group(self) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object):
            if name == "h5py":
                raise ModuleNotFoundError("No module named 'h5py'")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(OptionalDependencyError, "uv sync --group data --group dev"):
                _require_pack_dependencies()

    @unittest.skipUnless(DATA_DEPS_AVAILABLE, "h5py and pyarrow are not installed")
    def test_pack_cli_writes_split_artifacts_and_verifiable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            pack_dir = root / "pack"
            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=build_dir,
                command=("codelewm", "dataset", "build"),
            )

            completed = _run_dataset_pack(build_dir / "manifest.json", pack_dir)

            build_artifact_id = read_artifact_manifest(build_dir / "manifest.json").artifact_id
            pack_manifest = read_artifact_manifest(pack_dir / "manifest.json")
            checked = validate_artifact_checksums(pack_manifest, root=pack_dir)
            dataset_manifest = read_dataset_manifest(pack_dir / "dataset_manifest.json")

            import h5py

            with h5py.File(pack_dir / "hdf5" / "train.hdf5", "r") as handle:
                train_rows = int(handle.attrs["row_count"])
                train_shape = handle["state_before/input_ids"].shape
            with h5py.File(pack_dir / "hdf5" / "val.hdf5", "r") as handle:
                val_rows = int(handle.attrs["row_count"])

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], DATASET_PACK_REPORT_SCHEMA_VERSION)
        self.assertEqual(payload["row_count"], 3)
        self.assertEqual(pack_manifest.parent_artifacts, (build_artifact_id,))
        self.assertEqual(len(checked), len(pack_manifest.files))
        self.assertEqual(dataset_manifest.row_count, 3)
        self.assertEqual(dataset_manifest.split_counts, {"train": 2, "val": 1, "test": 0})
        self.assertEqual(train_rows, 2)
        self.assertEqual(val_rows, 1)
        self.assertEqual(train_shape[1], 1024)
        self.assertIn("hdf5/train.hdf5", {artifact.path for artifact in dataset_manifest.artifacts})
        self.assertIn("parquet/train/part-00000.parquet", {artifact.path for artifact in dataset_manifest.artifacts})

    @unittest.skipUnless(DATA_DEPS_AVAILABLE, "h5py and pyarrow are not installed")
    def test_manifest_verify_accepts_parent_build_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            pack_dir = root / "pack"
            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=build_dir,
                command=("codelewm", "dataset", "build"),
            )
            pack_dataset_from_manifest(
                manifest_path=build_dir / "manifest.json",
                output_dir=pack_dir,
                command=("codelewm", "dataset", "pack"),
            )

            completed = _run_manifest_verify(pack_dir / "manifest.json", build_dir / "manifest.json")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["parents_checked"]), 1)

    @unittest.skipUnless(DATA_DEPS_AVAILABLE, "h5py and pyarrow are not installed")
    def test_pack_accepts_relative_output_dir_for_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp).relative_to(ROOT)
            build_dir = root / "build"
            pack_dir = root / "pack"
            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=build_dir,
                command=("codelewm", "dataset", "build"),
            )

            pack_dataset_from_manifest(
                manifest_path=build_dir / "manifest.json",
                output_dir=pack_dir,
                command=("codelewm", "dataset", "pack"),
            )
            manifest = read_artifact_manifest(pack_dir / "manifest.json")
            checked = validate_artifact_checksums(manifest, root=pack_dir)

        self.assertEqual(len(checked), len(manifest.files))

    @unittest.skipUnless(DATA_DEPS_AVAILABLE, "h5py and pyarrow are not installed")
    def test_pack_rejects_tampered_build_artifact_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=build_dir,
                command=("codelewm", "dataset", "build"),
            )
            with (build_dir / "transitions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n")

            with self.assertRaisesRegex(Exception, "checksum mismatch"):
                pack_dataset_from_manifest(
                    manifest_path=build_dir / "manifest.json",
                    output_dir=root / "pack",
                    command=("codelewm", "dataset", "pack"),
                )

    @unittest.skipUnless(DATA_DEPS_AVAILABLE, "h5py and pyarrow are not installed")
    def test_pack_rejects_dataset_manifest_row_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=build_dir,
                command=("codelewm", "dataset", "build"),
            )
            dataset_manifest_path = build_dir / "dataset_manifest.json"
            payload = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            payload["row_count"] = 999
            dataset_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            original_manifest = read_artifact_manifest(build_dir / "manifest.json")
            refreshed = build_artifact_manifest(
                artifact_kind="dataset",
                root=build_dir,
                files=[file.path for file in original_manifest.files],
                command=original_manifest.command,
                config={"test": "mismatch"},
                source_git_sha="0" * 40,
                artifact_id=original_manifest.artifact_id,
                metadata=original_manifest.metadata,
            )
            write_artifact_manifest(refreshed, build_dir / "manifest.json")

            with self.assertRaisesRegex(PackError, "row_count"):
                pack_dataset_from_manifest(
                    manifest_path=build_dir / "manifest.json",
                    output_dir=root / "pack",
                    command=("codelewm", "dataset", "pack"),
                )

    @unittest.skipUnless(DATA_DEPS_AVAILABLE, "h5py and pyarrow are not installed")
    def test_pack_outputs_are_deterministic_for_hdf5_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            first = root / "first"
            second = root / "second"
            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=build_dir,
                command=("codelewm", "dataset", "build"),
            )
            pack_dataset_from_manifest(
                manifest_path=build_dir / "manifest.json",
                output_dir=first,
                command=("codelewm", "dataset", "pack"),
            )
            pack_dataset_from_manifest(
                manifest_path=build_dir / "manifest.json",
                output_dir=second,
                command=("codelewm", "dataset", "pack"),
            )

            first_hash = sha256_file(first / "hdf5" / "train.hdf5")
            second_hash = sha256_file(second / "hdf5" / "train.hdf5")

        self.assertEqual(first_hash, second_hash)


def _run_dataset_pack(manifest_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "codelewm.harness.cli",
            "dataset",
            "pack",
            "--manifest",
            str(manifest_path),
            "--out",
            str(output_dir),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_manifest_verify(manifest_path: Path, parent_manifest_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "codelewm.harness.cli",
            "manifest",
            "verify",
            "--manifest",
            str(manifest_path),
            "--parent-manifest",
            str(parent_manifest_path),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    unittest.main()
