"""MBPP source adapter.

MBPP (Mostly Basic Python Problems) ships as JSONL where each row has::

    {
      "task_id": 11,
      "text": "...",
      "code": "def f(...):\\n    ...",
      "test_list": ["assert f(2) == 4", ...],
      "test_setup_code": "",
      "challenge_test_list": []
    }

The adapter extracts the function name from ``code`` and parses each
``assert`` in ``test_list`` into a function-call input case. Only
assertions of the form ``assert <fn>(<args>) == <expected>`` are kept.
The rest are dropped with a warning in the per-record metadata.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path

from .base import ExecutionSourceError
from .record import InputCase, SourceSubmission


_MBPP_LICENSE = "CC-BY-4.0"
_MBPP_ATTR = "https://huggingface.co/datasets/mbpp"


class MBPPSourceAdapter:
    dataset = "mbpp"
    license = _MBPP_LICENSE
    license_attribution_url = _MBPP_ATTR
    held_out_for_eval = False

    def iter_submissions(self, *, source_path: Path) -> Iterator[SourceSubmission]:
        if not source_path.is_file():
            raise ExecutionSourceError(
                f"MBPP source must be a JSONL file: {source_path}"
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
        code = row.get("code")
        test_list = row.get("test_list")
        if task_id is None or not isinstance(code, str) or not isinstance(test_list, list):
            return None
        function_name = _extract_function_name(code)
        if function_name is None:
            return None
        inputs: list[InputCase] = []
        expected_outputs: list[str] = []
        for case_idx, assertion in enumerate(test_list):
            if not isinstance(assertion, str):
                continue
            parsed = _parse_function_assertion(assertion, function_name)
            if parsed is None:
                continue
            args_json, expected_repr = parsed
            inputs.append(
                InputCase(
                    input_id=f"mbpp/{task_id}/{case_idx}",
                    input_repr=args_json,
                    input_kind="function_call",
                    function_name=function_name,
                )
            )
            expected_outputs.append(expected_repr)
        if not inputs:
            return None
        return SourceSubmission(
            source_dataset="mbpp",
            source_problem_id=f"mbpp/{task_id}",
            source_submission_id=f"mbpp/{task_id}/reference",
            code=code,
            inputs=tuple(inputs),
            expected_outputs=tuple(expected_outputs),
            judge_verdict="accepted",
            license=_MBPP_LICENSE,
            license_attribution_url=_MBPP_ATTR,
        )


def _extract_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def _parse_function_assertion(
    assertion: str, function_name: str
) -> tuple[str, str] | None:
    """Parse ``assert <fn>(args) == <expected>`` into (json_args, expected_repr).

    Returns ``None`` for assertions that do not match the expected shape.
    """

    try:
        tree = ast.parse(assertion, mode="exec")
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    test = tree.body[0].test
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    if not isinstance(test.left, ast.Call):
        return None
    call = test.left
    func = call.func
    if isinstance(func, ast.Name) and func.id == function_name:
        pass
    elif isinstance(func, ast.Attribute) and func.attr == function_name:
        pass
    else:
        return None
    if call.keywords:
        return None
    args_value: list[object] = []
    for arg_node in call.args:
        literal = _try_literal_eval(arg_node)
        if literal is _Sentinel:
            return None
        args_value.append(literal)
    expected_node = test.comparators[0]
    expected_literal = _try_literal_eval(expected_node)
    if expected_literal is _Sentinel:
        return None
    args_json = json.dumps(args_value, ensure_ascii=False)
    expected_repr = repr(expected_literal)
    return args_json, expected_repr


class _SentinelType:
    pass


_Sentinel: object = _SentinelType()


def _try_literal_eval(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return _Sentinel
