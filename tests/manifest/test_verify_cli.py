from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.observability import build_artifact_manifest, write_artifact_manifest


ROOT = Path(__file__).resolve().parents[2]


class ManifestVerifyCliTest(unittest.TestCase):
    def test_manifest_verify_cli_returns_json_success_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "report.json"
            artifact.write_text('{"ok": true}\n', encoding="utf-8")
            manifest = build_artifact_manifest(
                artifact_kind="eval_report",
                root=root,
                files=(artifact,),
                command=("codelewm", "eval", "retrieval"),
                config={"name": "fixture"},
                source_git_sha="0" * 40,
                artifact_id="eval-report-fixture",
            )
            manifest_path = root / "manifest.json"
            write_artifact_manifest(manifest, manifest_path)

            completed = _run_verify(manifest_path)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "codelewm.manifest_verify.v1")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["artifact_id"], "eval-report-fixture")
        self.assertEqual(payload["files_checked"], 1)

    def test_manifest_verify_cli_returns_json_error_on_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "report.json"
            artifact.write_text('{"ok": true}\n', encoding="utf-8")
            manifest = build_artifact_manifest(
                artifact_kind="eval_report",
                root=root,
                files=(artifact,),
                command=("codelewm", "eval", "retrieval"),
                config={"name": "fixture"},
                source_git_sha="0" * 40,
            )
            manifest_path = root / "manifest.json"
            write_artifact_manifest(manifest, manifest_path)
            artifact.write_text('{"ok": false}\n', encoding="utf-8")

            completed = _run_verify(manifest_path)

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "codelewm.error.v1")
        self.assertEqual(payload["error_type"], "manifest_error")
        self.assertIn("checksum mismatch", payload["message"])

    def test_manifest_verify_cli_requires_declared_parent_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "checkpoint.bin"
            artifact.write_bytes(b"checkpoint")
            manifest = build_artifact_manifest(
                artifact_kind="checkpoint",
                root=root,
                files=(artifact,),
                command=("codelewm", "train"),
                config={"name": "fixture"},
                parent_artifacts=("dataset-parent",),
                source_git_sha="0" * 40,
            )
            manifest_path = root / "manifest.json"
            write_artifact_manifest(manifest, manifest_path)

            completed = _run_verify(manifest_path)

        self.assertEqual(completed.returncode, 2)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["error_type"], "manifest_error")
        self.assertIn("dataset-parent", payload["message"])

    def test_manifest_verify_cli_accepts_passed_parent_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as parent_tmp, tempfile.TemporaryDirectory() as child_tmp:
            parent_root = Path(parent_tmp)
            parent_artifact = parent_root / "dataset.bin"
            parent_artifact.write_bytes(b"parent")
            parent_manifest = build_artifact_manifest(
                artifact_kind="dataset",
                root=parent_root,
                files=(parent_artifact,),
                command=("codelewm", "dataset", "build"),
                config={"name": "parent"},
                source_git_sha="0" * 40,
                artifact_id="dataset-parent",
            )
            parent_manifest_path = parent_root / "manifest.json"
            write_artifact_manifest(parent_manifest, parent_manifest_path)

            child_root = Path(child_tmp)
            child_artifact = child_root / "checkpoint.bin"
            child_artifact.write_bytes(b"checkpoint")
            child_manifest = build_artifact_manifest(
                artifact_kind="checkpoint",
                root=child_root,
                files=(child_artifact,),
                command=("codelewm", "train"),
                config={"name": "child"},
                parent_artifacts=("dataset-parent",),
                source_git_sha="0" * 40,
            )
            child_manifest_path = child_root / "manifest.json"
            write_artifact_manifest(child_manifest, child_manifest_path)

            completed = _run_verify(child_manifest_path, "--parent-manifest", str(parent_manifest_path))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["parents_checked"], ["dataset-parent"])
        self.assertEqual(payload["parent_artifacts"], ["dataset-parent"])

    def test_manifest_verify_cli_reports_missing_artifact_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "report.json"
            artifact.write_text("{}\n", encoding="utf-8")
            manifest = build_artifact_manifest(
                artifact_kind="eval_report",
                root=root,
                files=(artifact,),
                command=("codelewm", "eval", "retrieval"),
                config={"name": "fixture"},
                source_git_sha="0" * 40,
            )
            manifest_path = root / "manifest.json"
            write_artifact_manifest(manifest, manifest_path)
            artifact.unlink()

            completed = _run_verify(manifest_path)

        self.assertEqual(completed.returncode, 2)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["error_type"], "manifest_error")
        self.assertIn("does not exist", payload["message"])

    def test_manifest_verify_cli_rejects_malformed_manifest_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text("not json\n", encoding="utf-8")

            completed = _run_verify(manifest_path)

        self.assertEqual(completed.returncode, 2)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["error_type"], "manifest_error")
        self.assertEqual(payload["schema_version"], "codelewm.error.v1")


def _run_verify(manifest_path: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "codelewm.harness.cli",
            "manifest",
            "verify",
            "--manifest",
            str(manifest_path),
            "--json",
            *extra_args,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    unittest.main()
