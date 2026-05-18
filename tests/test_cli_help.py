from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliHelpTest(unittest.TestCase):
    def test_cli_help_runs_from_package_module(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "codelewm.harness.cli", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: codelewm", completed.stdout)
        self.assertIn("CodeLeWM command-line interface", completed.stdout)
        self.assertIn("score", completed.stdout)
        self.assertIn("rerank", completed.stdout)


if __name__ == "__main__":
    unittest.main()
