"""MBPP-Plus source adapter (held out for downstream evaluation).

MBPP-Plus (https://github.com/evalplus/evalplus) augments MBPP with
``base_input`` and ``plus_input`` arrays plus a canonical solution. The
JSONL ships one row per problem::

    {
      "task_id": "Mbpp/1",
      "prompt": "...",
      "canonical_solution": "def f(...):\\n    ...",
      "entry_point": "f",
      "base_input": [[2], [3]],
      "plus_input": [[10], [99]],
      "expected_output": [4, 9, 100, 9801]
    }

The adapter takes ``canonical_solution`` as the code, ``entry_point`` as
the function name, and emits one input case per element of
``base_input`` + ``plus_input``. ``expected_output`` is aligned by index.

The adapter is registered with ``held_out_for_eval=True`` so the pack
builder refuses to put MBPP-Plus rows into train/val splits.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path

from .base import ExecutionSourceError
from .record import InputCase, SourceSubmission


_LICENSE = "Apache-2.0"
_ATTR = "https://huggingface.co/datasets/evalplus/mbppplus"


class MBPPPlusSourceAdapter:
    dataset = "mbpp_plus"
    license = _LICENSE
    license_attribution_url = _ATTR
    held_out_for_eval = True

    def iter_submissions(self, *, source_path: Path) -> Iterator[SourceSubmission]:
        if not source_path.is_file():
            raise ExecutionSourceError(
                f"MBPP-Plus source must be a JSONL file: {source_path}"
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
        current_schema = _parse_evalplus_current_row(row)
        if current_schema is not None:
            return current_schema
        return _parse_legacy_row(row)


def _parse_legacy_row(row: dict[str, object]) -> SourceSubmission | None:
    task_id = row.get("task_id")
    code = row.get("canonical_solution")
    entry_point = row.get("entry_point")
    base_input = row.get("base_input") or []
    plus_input = row.get("plus_input") or []
    expected_output = row.get("expected_output") or []
    if (
        not isinstance(task_id, str)
        or not isinstance(code, str)
        or not isinstance(entry_point, str)
        or not isinstance(base_input, list)
        or not isinstance(plus_input, list)
        or not isinstance(expected_output, list)
    ):
        return None
    all_inputs = list(base_input) + list(plus_input)
    if not all_inputs:
        return None
    inputs: list[InputCase] = []
    expected_outputs: list[str] = []
    for idx, arg_list in enumerate(all_inputs):
        if not isinstance(arg_list, list):
            continue
        input_repr = json.dumps(arg_list, ensure_ascii=False)
        inputs.append(
            InputCase(
                input_id=f"{task_id}/case-{idx}",
                input_repr=input_repr,
                input_kind="function_call",
                function_name=entry_point,
            )
        )
        if idx < len(expected_output):
            expected_outputs.append(repr(expected_output[idx]))
        else:
            expected_outputs.append("")
    if not inputs:
        return None
    return SourceSubmission(
        source_dataset="mbpp_plus",
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


def _parse_evalplus_current_row(row: dict[str, object]) -> SourceSubmission | None:
    task_id_raw = row.get("task_id")
    code = row.get("code")
    test_src = row.get("test")
    if (
        task_id_raw is None
        or not isinstance(code, str)
        or not isinstance(test_src, str)
    ):
        return None
    task_id = _normalize_task_id(task_id_raw)
    if task_id is None:
        return None
    entry_point = _extract_function_name(code)
    if entry_point is None:
        return None
    parsed_cases = _parse_evalplus_test(test_src)
    if parsed_cases is None:
        return None
    input_values, expected_values = parsed_cases
    if not input_values:
        return None
    inputs: list[InputCase] = []
    expected_outputs: list[str] = []
    for idx, arg_list in enumerate(input_values):
        if not isinstance(arg_list, list):
            continue
        try:
            input_repr = json.dumps(arg_list, ensure_ascii=False, default=_json_default)
        except (TypeError, ValueError):
            continue
        inputs.append(
            InputCase(
                input_id=f"{task_id}/case-{idx}",
                input_repr=input_repr,
                input_kind="function_call",
                function_name=entry_point,
            )
        )
        expected_outputs.append(
            repr(expected_values[idx]) if idx < len(expected_values) else ""
        )
    if not inputs:
        return None
    return SourceSubmission(
        source_dataset="mbpp_plus",
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


def _normalize_task_id(task_id: object) -> str | None:
    if isinstance(task_id, str):
        return task_id
    if isinstance(task_id, int) and not isinstance(task_id, bool):
        return f"Mbpp/{task_id}"
    return None


def _extract_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def _parse_evalplus_test(test_src: str) -> tuple[list[object], list[object]] | None:
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return None
    literals: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"inputs", "results"}:
                try:
                    literals[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return None
    inputs = literals.get("inputs")
    results = literals.get("results")
    if not isinstance(inputs, list) or not isinstance(results, list):
        return None
    normalized_inputs: list[object] = []
    for item in inputs:
        if isinstance(item, tuple):
            normalized_inputs.append(list(item))
        elif isinstance(item, list):
            normalized_inputs.append(item)
        else:
            normalized_inputs.append([item])
    return normalized_inputs, results


def _json_default(value: object) -> object:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"unsupported literal in MBPP-Plus input: {type(value).__name__}")
