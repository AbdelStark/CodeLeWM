from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


class PackageMetadataTest(unittest.TestCase):
    def test_console_script_entrypoint_is_registered(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())

        self.assertEqual(
            metadata["project"]["scripts"]["codelewm"],
            "codelewm.harness.cli:main",
        )

    def test_cli_module_help_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "codelewm.harness.cli", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CodeLeWM command-line interface", completed.stdout)

    def test_package_exposes_version(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import codelewm; print(codelewm.__version__)",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertRegex(completed.stdout.strip(), r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
