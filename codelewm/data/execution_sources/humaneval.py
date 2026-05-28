"""HumanEval source adapter (held out for downstream evaluation).

HumanEval (https://huggingface.co/datasets/openai_humaneval) ships as
JSONL where each row has::

    {
      "task_id": "HumanEval/0",
      "prompt": "def has_close_elements(...):\\n    ...",
      "canonical_solution": "...",
      "test": "def check(candidate):\\n    assert candidate(...) == ...",
      "entry_point": "has_close_elements"
    }

The adapter extracts the function name from ``entry_point`` and parses
``check()``'s body for ``assert candidate(args) == expected`` lines, the
same shape MBPP uses. ``prompt + canonical_solution`` form the executable
program text.

The adapter is registered with ``held_out_for_eval=True``.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path

from .base import ExecutionSourceError
from .record import InputCase, SourceSubmission


_LICENSE = "MIT"
_ATTR = "https://huggingface.co/datasets/openai_humaneval"


class HumanEvalSourceAdapter:
    dataset = "humaneval"
    license = _LICENSE
    license_attribution_url = _ATTR
    held_out_for_eval = True

    def iter_submissions(self, *, source_path: Path) -> Iterator[SourceSubmission]:
        if not source_path.is_file():
            raise ExecutionSourceError(
                f"HumanEval source must be a JSONL file: {source_path}"
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
        task_id = row.get("task_id")
        prompt = row.get("prompt")
        canonical = row.get("canonical_solution")
        test_src = row.get("test")
        entry_point = row.get("entry_point")
        if (
            not isinstance(task_id, str)
            or not isinstance(prompt, str)
            or not isinstance(canonical, str)
            or not isinstance(test_src, str)
            or not isinstance(entry_point, str)
        ):
            return None
        code = prompt + canonical
        inputs, expected_outputs = _parse_check_body(test_src, entry_point)
        if not inputs:
            return None
        return SourceSubmission(
            source_dataset="humaneval",
            source_problem_id=task_id,
            source_submission_id=f"{task_id}/canonical",
            code=code,
            inputs=tuple(inputs),
            expected_outputs=tuple(expected_outputs),
            judge_verdict="accepted",
            license=_LICENSE,
            license_attribution_url=_ATTR,
            held_out_for_eval=True,
        )


def _parse_check_body(
    test_src: str, entry_point: str
) -> tuple[list[InputCase], list[str]]:
    """Extract ``assert candidate(args) == expected`` lines from ``check``."""

    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return [], []
    inputs: list[InputCase] = []
    expected_outputs: list[str] = []
    case_idx = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            parsed = _parse_candidate_assert(node)
            if parsed is None:
                continue
            args_value, expected_value = parsed
            inputs.append(
                InputCase(
                    input_id=f"humaneval/case-{case_idx}",
                    input_repr=json.dumps(args_value, ensure_ascii=False),
                    input_kind="function_call",
                    function_name=entry_point,
                )
            )
            expected_outputs.append(repr(expected_value))
            case_idx += 1
    return inputs, expected_outputs


def _parse_candidate_assert(node: ast.Assert) -> tuple[list[object], object] | None:
    test = node.test
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    if not isinstance(test.left, ast.Call):
        return None
    call = test.left
    if not (isinstance(call.func, ast.Name) and call.func.id == "candidate"):
        return None
    if call.keywords:
        return None
    args: list[object] = []
    for arg_node in call.args:
        try:
            args.append(ast.literal_eval(arg_node))
        except (ValueError, SyntaxError):
            return None
    try:
        expected = ast.literal_eval(test.comparators[0])
    except (ValueError, SyntaxError):
        return None
    return args, expected
