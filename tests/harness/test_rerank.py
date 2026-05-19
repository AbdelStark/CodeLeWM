from __future__ import annotations

import difflib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.harness import (
    ERROR_REPORT_SCHEMA_VERSION,
    RERANK_RESULT_SCHEMA_VERSION,
    SCORE_RESULT_SCHEMA_VERSION,
    ErrorReport,
    ScoreResult,
    TransitionIndexEntry,
    build_transition_index,
    load_scorer,
    validate_rerank_result_payload,
    write_transition_index,
)
from codelewm.harness.scorer import _hashed_vector


ROOT = Path(__file__).resolve().parents[2]


class ConstantBackend:
    model_id = "constant-test-backend"
    warnings = ("constant backend",)

    def transition_energy(self, before: str, instruction: str, candidate: str) -> float:
        return 10.0


class RerankHarnessTest(unittest.TestCase):
    def test_rerank_scores_after_files_and_patches_with_errors_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            checkpoint = root / "checkpoint.bin"
            candidates = root / "candidates"
            candidates.mkdir()
            marker = root / "marker.txt"
            before_text = "def value():\n    return 1\n"
            patch_after = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
                "def value():\n"
                "    return 2\n"
            )
            after_file = candidates / "after_b.py"
            patch_file = candidates / "after_a.patch"
            invalid_file = candidates / "after_c.py"
            before.write_text(before_text)
            checkpoint.write_bytes(b"fixture checkpoint")
            after_file.write_text("def value():\n    return 3\n")
            patch_file.write_text(_unified_diff(before_text, patch_after))
            invalid_file.write_text("def broken(:\n    return 4\n")

            result = load_scorer(checkpoint, allow_unsafe=True).rerank_files(
                before=before,
                instruction="change return value",
                candidates=candidates,
            )

            score_results = [item for item in result.results if isinstance(item, ScoreResult)]
            error_reports = [item for item in result.results if isinstance(item, ErrorReport)]
            self.assertEqual(result.schema_version, RERANK_RESULT_SCHEMA_VERSION)
            self.assertEqual(len(score_results), 2)
            self.assertEqual(len(error_reports), 1)
            self.assertEqual(result.results[-1].schema_version, ERROR_REPORT_SCHEMA_VERSION)
            self.assertEqual(error_reports[0].error_type, "invalid_syntax")
            self.assertEqual(error_reports[0].artifact, str(invalid_file))
            self.assertEqual(
                [item.final_score for item in score_results],
                sorted(item.final_score for item in score_results),
            )
            self.assertEqual(before.read_text(), before_text)
            self.assertFalse(marker.exists())

    def test_rerank_reports_patch_apply_failure_after_valid_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            checkpoint = root / "checkpoint.bin"
            candidates = root / "candidates"
            candidates.mkdir()
            valid = candidates / "after.py"
            bad_patch = candidates / "bad.patch"
            before.write_text("value = 1\n")
            checkpoint.write_bytes(b"fixture checkpoint")
            valid.write_text("value = 2\n")
            bad_patch.write_text(
                "--- before.py\n"
                "+++ after.py\n"
                "@@ -1 +1 @@\n"
                "-missing = 1\n"
                "+value = 3\n"
            )

            result = load_scorer(checkpoint, allow_unsafe=True).rerank_files(
                before=before,
                instruction="change value",
                candidates=candidates,
            )

            self.assertIsInstance(result.results[0], ScoreResult)
            self.assertIsInstance(result.results[-1], ErrorReport)
            self.assertEqual(result.results[-1].error_type, "patch_apply_failed")
            self.assertEqual(result.results[-1].artifact, str(bad_patch))

    def test_rerank_cli_emits_schema_versioned_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            instruction = root / "instruction.txt"
            checkpoint = root / "checkpoint.bin"
            candidates = root / "candidates"
            candidates.mkdir()
            before.write_text("value = 1\n")
            instruction.write_text("increment value")
            checkpoint.write_bytes(b"fixture checkpoint")
            (candidates / "good.py").write_text("value = 2\n")
            (candidates / "bad.py").write_text("def broken(:\n    return 1\n")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "rerank",
                    "--before",
                    str(before),
                    "--instruction",
                    str(instruction),
                    "--candidates",
                    str(candidates),
                    "--checkpoint",
                    str(checkpoint),
                    "--json",
                    "--allow-unsafe-checkpoint",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        result = validate_rerank_result_payload(payload)
        self.assertEqual(result.schema_version, RERANK_RESULT_SCHEMA_VERSION)
        self.assertEqual(result.results[0].schema_version, SCORE_RESULT_SCHEMA_VERSION)
        self.assertEqual(result.results[-1].schema_version, ERROR_REPORT_SCHEMA_VERSION)

    def test_rerank_cli_emits_retrieval_prior_when_index_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            instruction = root / "instruction.txt"
            checkpoint = root / "checkpoint.bin"
            candidates = root / "candidates"
            index_root = root / "index"
            candidates.mkdir()
            near_text = "value = 2\n"
            before.write_text("value = 1\n")
            instruction.write_text("increment value")
            checkpoint.write_bytes(b"fixture checkpoint")
            (candidates / "near.py").write_text(near_text)
            (candidates / "far.py").write_text("def other():\n    return 100\n")
            index = build_transition_index(
                name="rerank-cli-prior-fixture",
                entries=(TransitionIndexEntry(transition_id="t-0", split="train"),),
                vectors=np.asarray([_hashed_vector(near_text, dim=8)], dtype=np.float32),
            )
            write_transition_index(index, index_root)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "rerank",
                    "--before",
                    str(before),
                    "--instruction",
                    str(instruction),
                    "--candidates",
                    str(candidates),
                    "--checkpoint",
                    str(checkpoint),
                    "--index",
                    str(index_root),
                    "--retrieval-prior-weight",
                    "100",
                    "--retrieval-prior-k",
                    "1",
                    "--json",
                    "--allow-unsafe-checkpoint",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        result = validate_rerank_result_payload(payload)
        score_results = [item for item in result.results if isinstance(item, ScoreResult)]
        self.assertTrue(all(item.retrieval_prior is not None for item in score_results))

    def test_rerank_orders_by_retrieval_prior_when_weighted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            checkpoint = root / "checkpoint.bin"
            candidates = root / "candidates"
            index_root = root / "index"
            candidates.mkdir()
            before.write_text("value = 1\n")
            checkpoint.write_bytes(b"fixture checkpoint")
            near = candidates / "near.py"
            far = candidates / "far.py"
            near_text = "value = 2\n"
            near.write_text(near_text)
            far.write_text("def other():\n    return 100\n")
            index = build_transition_index(
                name="rerank-prior-fixture",
                entries=(TransitionIndexEntry(transition_id="t-0", split="train"),),
                vectors=np.asarray([_hashed_vector(near_text, dim=8)], dtype=np.float32),
            )
            write_transition_index(index, index_root)

            result = load_scorer(
                checkpoint,
                allow_unsafe=True,
                backend=ConstantBackend(),
                index=index_root,
                retrieval_prior_weight=1.0,
                retrieval_prior_k=1,
            ).rerank_files(
                before=before,
                instruction="increment value",
                candidates=candidates,
            )

        score_results = [item for item in result.results if isinstance(item, ScoreResult)]
        self.assertEqual([item.candidate for item in score_results], [str(near), str(far)])
        self.assertEqual(score_results[0].retrieval_prior, 0.0)
        self.assertIsNotNone(score_results[1].retrieval_prior)
        self.assertGreater(score_results[1].final_score, score_results[0].final_score)


def _unified_diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before.py",
            tofile="after.py",
        )
    )


if __name__ == "__main__":
    unittest.main()
