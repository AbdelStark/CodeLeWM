from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.data import (
    DATASET_BUILD_CONFIG_SCHEMA_VERSION,
    DATASET_BUILD_REPORT_SCHEMA_VERSION,
    DatasetBuildConfigError,
    build_dataset_from_config_path,
    load_dataset_build_config,
    read_dataset_manifest,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "tests" / "fixtures" / "dataset_build" / "config.json"


class DatasetBuildTest(unittest.TestCase):
    def test_build_dataset_from_fixture_writes_accounted_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"

            result = build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=output_dir,
                command=("codelewm", "dataset", "build"),
            )

            artifact_manifest = read_artifact_manifest(output_dir / "manifest.json")
            dataset_manifest = read_dataset_manifest(output_dir / "dataset_manifest.json")
            filter_report = _read_json(output_dir / "reports" / "filter_report.json")
            row_counts = _read_json(output_dir / "reports" / "row_counts.json")

        self.assertEqual(result.dataset_manifest.row_count, 3)
        self.assertEqual(dataset_manifest.row_count, 3)
        self.assertEqual(artifact_manifest.artifact_kind, "dataset")
        self.assertEqual(filter_report["report"]["total_before"], 4)
        self.assertEqual(filter_report["report"]["total_after"], 3)
        self.assertEqual(filter_report["report"]["drop_reasons"], {"non_python_path": 1})
        self.assertEqual(row_counts["total_loaded"], 4)
        self.assertTrue(all(row_counts["accounting"].values()))
        self.assertEqual(dataset_manifest.metadata["license_gate_report"]["included_rows"], 3)
        self.assertIn("transitions.jsonl", {artifact.path for artifact in dataset_manifest.artifacts})

    def test_artifact_manifest_verifies_dataset_build_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"
            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=output_dir,
                command=("codelewm", "dataset", "build"),
            )
            manifest = read_artifact_manifest(output_dir / "manifest.json")

            checked = validate_artifact_checksums(manifest, root=output_dir)

        self.assertEqual(len(checked), 7)

    def test_build_accepts_relative_output_dir_for_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            relative_output = Path(tmp).relative_to(ROOT) / "dataset"

            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=relative_output,
                command=("codelewm", "dataset", "build"),
            )
            manifest = read_artifact_manifest(relative_output / "manifest.json")
            checked = validate_artifact_checksums(manifest, root=relative_output)

        self.assertEqual(len(checked), 7)

    def test_build_is_deterministic_for_transitions_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=first,
                command=("codelewm", "dataset", "build"),
            )
            build_dataset_from_config_path(
                config_path=FIXTURE_CONFIG,
                output_dir=second,
                command=("codelewm", "dataset", "build"),
            )

            first_rows = (first / "transitions.jsonl").read_text(encoding="utf-8")
            second_rows = (second / "transitions.jsonl").read_text(encoding="utf-8")
            first_manifest = read_dataset_manifest(first / "dataset_manifest.json")
            second_manifest = read_dataset_manifest(second / "dataset_manifest.json")

        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_manifest.split_counts, second_manifest.split_counts)
        self.assertEqual(first_manifest.source_counts, second_manifest.source_counts)

    def test_invalid_config_is_rejected_before_build(self) -> None:
        payload = _fixture_config_payload()
        payload["unexpected"] = True
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "invalid.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(DatasetBuildConfigError, "unexpected"):
                load_dataset_build_config(config_path)

    def test_dataset_build_cli_returns_json_success_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"

            completed = _run_dataset_build(FIXTURE_CONFIG, output_dir)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], DATASET_BUILD_REPORT_SCHEMA_VERSION)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["row_count"], 3)

    def test_invalid_config_cli_returns_structured_error(self) -> None:
        payload = _fixture_config_payload()
        payload["schema_version"] = "wrong"
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "invalid.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            completed = _run_dataset_build(config_path, Path(tmp) / "out")

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "codelewm.error.v1")
        self.assertEqual(payload["error_type"], "config_error")

    def test_missing_source_cli_returns_source_unavailable_error(self) -> None:
        payload = _fixture_config_payload()
        payload["sources"][0]["path"] = "missing.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "missing-source.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            completed = _run_dataset_build(config_path, Path(tmp) / "out")

        self.assertEqual(completed.returncode, 3)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["error_type"], "source_unavailable")
        self.assertIn("missing.jsonl", payload["message"])

    def test_empty_output_cli_returns_empty_dataset_error(self) -> None:
        payload = _fixture_config_payload()
        payload["filter"] = {"min_changed_lines": 999}
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "empty.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            completed = _run_dataset_build(config_path, Path(tmp) / "out")

        self.assertEqual(completed.returncode, 4)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "codelewm.error.v1")
        self.assertEqual(payload["error_type"], "empty_dataset")

    def test_schema_version_constant_matches_fixture_config(self) -> None:
        payload = _fixture_config_payload()

        self.assertEqual(payload["schema_version"], DATASET_BUILD_CONFIG_SCHEMA_VERSION)


def _fixture_config_payload() -> dict[str, object]:
    payload = json.loads(FIXTURE_CONFIG.read_text(encoding="utf-8"))
    payload["sources"][0]["path"] = str(FIXTURE_CONFIG.with_name("records.jsonl"))
    return payload


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected object in {path}")
    return payload


def _run_dataset_build(config_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "codelewm.harness.cli",
            "dataset",
            "build",
            "--config",
            str(config_path),
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


if __name__ == "__main__":
    unittest.main()
