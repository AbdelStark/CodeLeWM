"""APPS source adapter (stdin-style competitive programming).

APPS (https://huggingface.co/datasets/codeparrot/apps) ships problems
with hidden test cases. The expected flattened JSONL per row::

    {
      "problem_id": "apps/intro/0001",
      "difficulty": "introductory",
      "solutions": ["import sys\\n...", "..."],
      "input_output": {
        "inputs": ["3\\n", "5\\n"],
        "outputs": ["9\\n", "25\\n"]
      },
      "license": "MIT"
    }

Each ``solutions[i]`` becomes one :class:`SourceSubmission`. Test
inputs are shared across all solutions of the same problem (they
describe the problem). Inputs are ``stdin`` cases.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .base import ExecutionSourceError
from .record import InputCase, SourceSubmission


_DEFAULT_LICENSE = "MIT"
_DEFAULT_ATTR = "https://huggingface.co/datasets/codeparrot/apps"


class APPSSourceAdapter:
    dataset = "apps"
    license = _DEFAULT_LICENSE
    license_attribution_url = _DEFAULT_ATTR
    held_out_for_eval = False

    def iter_submissions(self, *, source_path: Path) -> Iterator[SourceSubmission]:
        if not source_path.is_file():
            raise ExecutionSourceError(
                f"APPS flattened source must be a JSONL file: {source_path}"
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
                yield from self._parse_row(row)

    def _parse_row(self, row: dict[str, object]) -> Iterator[SourceSubmission]:
        problem_id = row.get("problem_id")
        solutions = row.get("solutions")
        io = row.get("input_output")
        if (
            not isinstance(problem_id, str)
            or not isinstance(solutions, list)
            or not isinstance(io, dict)
        ):
            return
        inputs_raw = io.get("inputs")
        outputs_raw = io.get("outputs") or []
        if not isinstance(inputs_raw, list) or not inputs_raw:
            return
        license_name = (
            row.get("license") if isinstance(row.get("license"), str) else _DEFAULT_LICENSE
        )
        attr_url = (
            row.get("license_url")
            if isinstance(row.get("license_url"), str)
            else _DEFAULT_ATTR
        )
        cases = [
            InputCase(
                input_id=f"{problem_id}/case-{idx}",
                input_repr=str(inp) if not isinstance(inp, str) else inp,
                input_kind="stdin",
            )
            for idx, inp in enumerate(inputs_raw)
        ]
        expected_outputs = tuple(
            str(o) if not isinstance(o, str) else o for o in outputs_raw
        )
        if not expected_outputs:
            expected_outputs = None  # type: ignore[assignment]
        elif len(expected_outputs) != len(cases):
            # Trim or pad to keep alignment.
            expected_outputs = tuple(
                list(expected_outputs[: len(cases)])
                + [""] * max(0, len(cases) - len(expected_outputs))
            )
        for sol_idx, solution in enumerate(solutions):
            if not isinstance(solution, str) or not solution.strip():
                continue
            yield SourceSubmission(
                source_dataset="apps",
                source_problem_id=problem_id,
                source_submission_id=f"{problem_id}/sol-{sol_idx}",
                code=solution,
                inputs=tuple(cases),
                expected_outputs=expected_outputs,
                judge_verdict=None,
                license=str(license_name),
                license_attribution_url=str(attr_url),
            )
