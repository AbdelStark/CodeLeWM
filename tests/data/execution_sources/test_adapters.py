"""Tests for the execution-substrate source adapters.

Each adapter has a tiny checked-in fixture under
``tests/data/execution_sources/fixtures/`` so the test runs offline.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.data.execution_sources import (
    EXECUTION_SOURCE_DATASETS,
    EXECUTION_SOURCE_RECORD_SCHEMA_VERSION,
    ExecutionSourceError,
    InputCase,
    SourceSubmission,
    get_execution_source_adapter,
    load_execution_source,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class RecordValidationTest(unittest.TestCase):
    def test_minimal_record_round_trip(self) -> None:
        case = InputCase(
            input_id="x", input_repr="[1, 2]", input_kind="function_call",
            function_name="f",
        )
        submission = SourceSubmission(
            source_dataset="mbpp",
            source_problem_id="mbpp/1",
            source_submission_id="mbpp/1/r",
            code="def f(a, b):\n    return a + b\n",
            inputs=(case,),
            expected_outputs=("3",),
            judge_verdict="accepted",
            license="MIT",
            license_attribution_url="https://example.test",
        )
        self.assertEqual(submission.source_dataset, "mbpp")
        self.assertTrue(submission.raw_hash)
        payload = submission.as_dict()
        self.assertEqual(payload["source_problem_id"], "mbpp/1")
        self.assertEqual(len(payload["inputs"]), 1)

    def test_input_case_rejects_invalid_kind(self) -> None:
        with self.assertRaises(ValueError):
            InputCase(input_id="x", input_repr="", input_kind="weird")  # type: ignore[arg-type]

    def test_function_call_requires_function_name(self) -> None:
        with self.assertRaises(ValueError):
            InputCase(input_id="x", input_repr="[]", input_kind="function_call")


class MBPPAdapterTest(unittest.TestCase):
    def test_parses_assert_eq_lines_into_function_call_cases(self) -> None:
        adapter = get_execution_source_adapter("mbpp")
        records = list(adapter.iter_submissions(source_path=FIXTURES / "mbpp_tiny.jsonl"))
        self.assertEqual(len(records), 2, msg=[r.source_problem_id for r in records])
        square = records[0]
        self.assertEqual(square.source_problem_id, "mbpp/1")
        self.assertEqual(len(square.inputs), 3)
        self.assertEqual(square.inputs[0].function_name, "square")
        self.assertEqual(json.loads(square.inputs[0].input_repr), [2])
        self.assertEqual(square.expected_outputs, ("4", "9", "0"))
        self.assertFalse(square.held_out_for_eval)
        total = records[1]
        self.assertEqual(total.source_problem_id, "mbpp/2")
        self.assertEqual(total.inputs[0].function_name, "total")
        self.assertEqual(json.loads(total.inputs[0].input_repr), [[1, 2, 3]])

    def test_unparseable_assertions_drop_the_row(self) -> None:
        adapter = get_execution_source_adapter("mbpp")
        ids = [
            r.source_problem_id
            for r in adapter.iter_submissions(source_path=FIXTURES / "mbpp_tiny.jsonl")
        ]
        self.assertNotIn("mbpp/99", ids)


class MBPPPlusAdapterTest(unittest.TestCase):
    def test_emits_function_call_cases_and_is_held_out(self) -> None:
        adapter = get_execution_source_adapter("mbpp_plus")
        records = list(
            adapter.iter_submissions(source_path=FIXTURES / "mbpp_plus_tiny.jsonl")
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_problem_id, "Mbpp/1")
        self.assertEqual(len(records[0].inputs), 4)
        self.assertTrue(records[0].held_out_for_eval)
        self.assertEqual(records[0].license, "Apache-2.0")


class HumanEvalAdapterTest(unittest.TestCase):
    def test_parses_check_body_and_is_held_out(self) -> None:
        adapter = get_execution_source_adapter("humaneval")
        records = list(
            adapter.iter_submissions(source_path=FIXTURES / "humaneval_tiny.jsonl")
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].inputs), 2)
        self.assertEqual(records[0].inputs[0].function_name, "has_pair")
        self.assertTrue(records[0].held_out_for_eval)


class CodeNetAdapterTest(unittest.TestCase):
    def test_filters_non_python_and_keeps_stdin_cases(self) -> None:
        adapter = get_execution_source_adapter("codenet")
        records = list(
            adapter.iter_submissions(source_path=FIXTURES / "codenet_tiny.jsonl")
        )
        ids = {r.source_submission_id for r in records}
        self.assertIn("p00001/s001", ids)
        self.assertIn("p00002/s001", ids)
        self.assertNotIn("p00001/s002", ids, msg="C++ rows must be filtered out")
        accepted = [r for r in records if r.source_submission_id == "p00001/s001"][0]
        self.assertEqual(accepted.inputs[0].input_kind, "stdin")
        self.assertEqual(accepted.judge_verdict, "accepted")


class APPSAdapterTest(unittest.TestCase):
    def test_emits_one_submission_per_solution(self) -> None:
        adapter = get_execution_source_adapter("apps")
        records = list(
            adapter.iter_submissions(source_path=FIXTURES / "apps_tiny.jsonl")
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].source_problem_id, "apps/intro/0001")
        self.assertEqual(records[0].inputs[0].input_kind, "stdin")
        self.assertEqual(records[0].expected_outputs, ("9\n", "25\n"))


class RegistryTest(unittest.TestCase):
    def test_known_dataset_names_registered(self) -> None:
        for name in EXECUTION_SOURCE_DATASETS:
            with self.subTest(name=name):
                adapter = get_execution_source_adapter(name)
                self.assertEqual(adapter.dataset, name)
        for name in EXECUTION_SOURCE_DATASETS:
            if name in {"mbpp_plus", "humaneval"}:
                self.assertTrue(get_execution_source_adapter(name).held_out_for_eval)

    def test_unknown_source_raises(self) -> None:
        with self.assertRaises(ExecutionSourceError):
            get_execution_source_adapter("does-not-exist")


class LoadExecutionSourceTest(unittest.TestCase):
    def test_writes_jsonl_with_schema_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            result = load_execution_source(
                source="mbpp",
                source_path=FIXTURES / "mbpp_tiny.jsonl",
                output_path=out,
            )
            self.assertEqual(result["source"], "mbpp")
            self.assertGreaterEqual(int(result["submission_count"]), 1)
            self.assertEqual(
                result["schema_version"], EXECUTION_SOURCE_RECORD_SCHEMA_VERSION
            )
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), int(result["submission_count"]))
            first = json.loads(lines[0])
            self.assertEqual(
                first["schema_version"], EXECUTION_SOURCE_RECORD_SCHEMA_VERSION
            )
            self.assertIn("raw_hash", first)
            self.assertEqual(first["source_dataset"], "mbpp")

    def test_limit_caps_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            result = load_execution_source(
                source="mbpp",
                source_path=FIXTURES / "mbpp_tiny.jsonl",
                output_path=out,
                limit=1,
            )
            self.assertEqual(result["submission_count"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
