from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.release import (
    RELEASE_PROVENANCE_SCHEMA_VERSION,
    ReleaseProvenanceError,
    build_release_provenance,
    validate_release_provenance_payload,
)


ROOT = Path(__file__).resolve().parents[2]


class ReleaseProvenanceTest(unittest.TestCase):
    def test_builds_repository_relative_provenance_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_release_fixture(root)

            payload = build_release_provenance(
                root=root,
                dist_dir="dist",
                audit_report="reports/pip-audit.json",
                include_paths=("docs/release/PACKAGE_PUBLISHING.md",),
                source_git_sha="0" * 40,
                command=("scripts/release-provenance", "--dist", "dist"),
            )

            checked = validate_release_provenance_payload(payload)
            self.assertEqual(checked["schema_version"], RELEASE_PROVENANCE_SCHEMA_VERSION)
            self.assertEqual(checked["source_git_sha"], "0" * 40)
            self.assertFalse(checked["tracked_git_dirty"])
            self.assertEqual(checked["dependency_audit"]["report"]["path"], "reports/pip-audit.json")
            self.assertEqual(checked["lockfile"]["path"], "uv.lock")
            self.assertEqual(
                {entry["path"] for entry in checked["built_distributions"]},
                {"dist/codelewm-0.0.0-py3-none-any.whl", "dist/codelewm-0.0.0.tar.gz"},
            )
            self.assertEqual(checked["included_evidence"][0]["path"], "docs/release/PACKAGE_PUBLISHING.md")

    def test_rejects_missing_distribution_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dist").mkdir()
            (root / "dist" / "codelewm-0.0.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
            (root / "uv.lock").write_text("lock", encoding="utf-8")

            with self.assertRaisesRegex(ReleaseProvenanceError, "source distribution"):
                build_release_provenance(root=root, dist_dir="dist", source_git_sha="0" * 40)

    def test_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_release_fixture(root)
            output = root / "provenance" / "provenance.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "release-provenance"),
                    "--root",
                    str(root),
                    "--dist",
                    "dist",
                    "--audit-report",
                    "reports/pip-audit.json",
                    "--include",
                    "docs/release/PACKAGE_PUBLISHING.md",
                    "--out",
                    str(output),
                    "--source-git-sha",
                    "0" * 40,
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], RELEASE_PROVENANCE_SCHEMA_VERSION)
            self.assertIn(RELEASE_PROVENANCE_SCHEMA_VERSION, completed.stdout)

    @staticmethod
    def _write_release_fixture(root: Path) -> None:
        (root / "dist").mkdir(parents=True)
        (root / "reports").mkdir()
        (root / "docs" / "release").mkdir(parents=True)
        (root / "uv.lock").write_text("lock\n", encoding="utf-8")
        (root / "dist" / "codelewm-0.0.0-py3-none-any.whl").write_text("wheel\n", encoding="utf-8")
        (root / "dist" / "codelewm-0.0.0.tar.gz").write_text("sdist\n", encoding="utf-8")
        (root / "reports" / "pip-audit.json").write_text('{"dependencies": []}\n', encoding="utf-8")
        (root / "docs" / "release" / "PACKAGE_PUBLISHING.md").write_text("package gate\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
