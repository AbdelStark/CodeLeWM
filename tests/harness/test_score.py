from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.harness import SCORE_RESULT_SCHEMA_VERSION, ScoreResult, load_scorer


ROOT = Path(__file__).resolve().parents[2]


class ScoreApiTest(unittest.TestCase):
    def test_load_scorer_scores_fixture_files_with_schema_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            candidate = root / "after.py"
            checkpoint = root / "checkpoint.bin"
            before.write_text("def add(a, b):\n    return a + b\n")
            candidate.write_text("def add(a, b):\n    return a + b + 1\n")
            checkpoint.write_bytes(b"fixture checkpoint")

            scorer = load_scorer(checkpoint)
            result = scorer.score_files(
                before=before,
                instruction="add one to the returned value",
                candidate=candidate,
            )
            payload = result.to_dict()
            loaded = ScoreResult.from_dict(payload)

        self.assertEqual(payload["schema_version"], SCORE_RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["candidate"], str(candidate))
        self.assertGreaterEqual(payload["transition_energy"], 0.0)
        self.assertEqual(payload["final_score"], payload["transition_energy"])
        self.assertEqual(len(payload["checkpoint_sha256"]), 64)
        self.assertEqual(len(payload["input_digest"]), 64)
        self.assertIn("lightweight scorer backend", payload["warnings"][0])
        self.assertEqual(loaded.to_dict(), payload)

    def test_score_cli_emits_json_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            candidate = root / "after.py"
            instruction = root / "instruction.txt"
            checkpoint = root / "checkpoint.bin"
            before.write_text("value = 1\n")
            candidate.write_text("value = 2\n")
            instruction.write_text("increment value")
            checkpoint.write_bytes(b"fixture checkpoint")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "score",
                    "--before",
                    str(before),
                    "--instruction",
                    str(instruction),
                    "--candidate",
                    str(candidate),
                    "--checkpoint",
                    str(checkpoint),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], SCORE_RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["candidate"], str(candidate))
        self.assertGreaterEqual(payload["final_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
