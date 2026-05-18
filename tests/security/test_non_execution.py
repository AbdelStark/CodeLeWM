from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from codelewm.data import FilterPolicy, RawEditRecord, extract_codestate_pair, filter_raw_edit_records
from codelewm.harness import load_scorer
from codelewm.security import (
    NonExecutionPolicyError,
    parse_python_source_text,
    reject_code_execution_config,
)
from codelewm.training import TrainConfigError, load_train_config, validate_train_config


ROOT = Path(__file__).resolve().parents[2]


class NonExecutionBoundaryTest(unittest.TestCase):
    def test_text_parser_does_not_execute_top_level_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "marker.txt"
            source = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
                "value = 1\n"
            )

            tree = parse_python_source_text(source, filename="malicious.py")

            self.assertEqual(type(tree).__name__, "Module")
            self.assertFalse(marker.exists())

    def test_dataset_codestate_extraction_does_not_execute_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "dataset-marker.txt"
            before = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('before executed')\n"
                "def value():\n"
                "    return 1\n"
            )
            after = before.replace("return 1", "return 2")
            record = _record(before=before, after=after)

            pair = extract_codestate_pair(record)
            kept = filter_raw_edit_records((record,), policy=FilterPolicy(min_edit_ratio=0.0))

            self.assertEqual(pair.before.kind, "function")
            self.assertEqual(pair.after.kind, "function")
            self.assertEqual(len(kept.kept), 1)
            self.assertFalse(marker.exists())

    def test_harness_score_files_does_not_execute_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            marker = root / "candidate-marker.txt"
            checkpoint = root / "checkpoint.bin"
            before = root / "before.py"
            candidate = root / "after.py"
            checkpoint.write_bytes(b"checkpoint")
            before.write_text("value = 1\n")
            candidate.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('candidate executed')\n"
                "value = 2\n"
            )

            result = load_scorer(checkpoint).score_files(
                before=before,
                instruction="change the value",
                candidate=candidate,
            )

            self.assertEqual(result.candidate, str(candidate))
            self.assertFalse(marker.exists())

    def test_security_policy_rejects_execution_request_configs(self) -> None:
        with self.assertRaisesRegex(NonExecutionPolicyError, "run_tests"):
            reject_code_execution_config(
                {
                    "pipeline": {
                        "parse_only": True,
                        "run_tests": "pytest tests",
                    }
                }
            )

    def test_train_config_rejects_execution_request_before_unknown_key_fallback(self) -> None:
        payload = copy.deepcopy(load_train_config(ROOT / "config/train/codelewm_tiny.yaml").to_dict())
        payload["data"]["run_tests"] = True

        with self.assertRaisesRegex(TrainConfigError, "run_tests"):
            validate_train_config(payload)


def _record(*, before: str, after: str) -> RawEditRecord:
    return RawEditRecord(
        source="local_repo",
        repo="fixture",
        commit="abc123",
        path_before="pkg/module.py",
        path_after="pkg/module.py",
        before=before,
        after=after,
        message="update value function",
        license="mit",
    )


if __name__ == "__main__":
    unittest.main()
