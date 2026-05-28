"""CodeNet source adapter (stdin-style competitive programming submissions).

CodeNet's Project_CodeNet (IBM) tracks ~14M problem submissions. The
upstream layout is complex; this adapter operates on a flattened JSONL
where each row already aggregates one submission with its problem's
test inputs and expected outputs::

    {
      "problem_id": "p00001",
      "submission_id": "s_00001234",
      "language": "Python",
      "license": "MIT",
      "license_url": "https://...",
      "code": "n = int(input()); print(n*n)",
      "judge_verdict": "Accepted",
      "test_cases": [
        {"input_id": "case-1", "input": "3\\n", "expected_output": "9\\n"},
        ...
      ]
    }

The adapter emits one :class:`SourceSubmission` per upstream row. Each
test case becomes a ``stdin`` :class:`InputCase`. Non-Python rows and
rows without test cases are skipped.

Building the flattened JSONL from raw CodeNet downloads is out of scope
for this adapter and lives in ``scripts/`` next to the pack builder.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .base import ExecutionSourceError
from .record import InputCase, SourceSubmission


_DEFAULT_LICENSE = "Apache-2.0"
_DEFAULT_ATTR = "https://github.com/IBM/Project_CodeNet"


class CodeNetSourceAdapter:
    dataset = "codenet"
    license = _DEFAULT_LICENSE
    license_attribution_url = _DEFAULT_ATTR
    held_out_for_eval = False

    def iter_submissions(self, *, source_path: Path) -> Iterator[SourceSubmission]:
        if not source_path.is_file():
            raise ExecutionSourceError(
                f"CodeNet flattened source must be a JSONL file: {source_path}"
            )
        with source_path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ExecutionSourceError(
                        f"{source_path}:{line_no}: invalid JSON"
                    ) from exc
                submission = self._parse_row(row)
                if submission is not None:
                    yield submission

    def _parse_row(self, row: dict[str, object]) -> SourceSubmission | None:
        problem_id = row.get("problem_id")
        submission_id = row.get("submission_id")
        language = row.get("language")
        code = row.get("code")
        test_cases = row.get("test_cases")
        if (
            not isinstance(problem_id, str)
            or not isinstance(submission_id, str)
            or not isinstance(code, str)
            or not isinstance(test_cases, list)
        ):
            return None
        if isinstance(language, str) and language.strip().lower() not in {
            "python",
            "python3",
            "py",
        }:
            return None
        license_name = (
            row.get("license") if isinstance(row.get("license"), str) else _DEFAULT_LICENSE
        )
        attr_url = (
            row.get("license_url")
            if isinstance(row.get("license_url"), str)
            else _DEFAULT_ATTR
        )
        inputs: list[InputCase] = []
        expected_outputs: list[str] = []
        for case_idx, case in enumerate(test_cases):
            if not isinstance(case, dict):
                continue
            input_text = case.get("input")
            expected = case.get("expected_output")
            input_id = case.get("input_id") or f"{submission_id}/case-{case_idx}"
            if not isinstance(input_text, str):
                continue
            inputs.append(
                InputCase(
                    input_id=str(input_id),
                    input_repr=input_text,
                    input_kind="stdin",
                )
            )
            expected_outputs.append(
                expected if isinstance(expected, str) else ""
            )
        if not inputs:
            return None
        verdict_value = row.get("judge_verdict")
        verdict = (
            _normalize_verdict(verdict_value)
            if isinstance(verdict_value, str)
            else None
        )
        return SourceSubmission(
            source_dataset="codenet",
            source_problem_id=problem_id,
            source_submission_id=submission_id,
            code=code,
            inputs=tuple(inputs),
            expected_outputs=tuple(expected_outputs),
            judge_verdict=verdict,
            license=str(license_name),
            license_attribution_url=str(attr_url),
        )


def _normalize_verdict(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    mapping = {
        "accepted": "accepted",
        "wrong_answer": "wrong_answer",
        "runtime_error": "runtime_error",
        "time_limit_exceeded": "time_limit_exceeded",
        "memory_limit_exceeded": "memory_limit_exceeded",
        "compile_error": "compile_error",
    }
    return mapping.get(value, "unknown")
