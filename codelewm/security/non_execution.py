"""Non-execution parsing and configuration guards."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any


FORBIDDEN_EXECUTION_CONFIG_KEYS = frozenset(
    {
        "allow_user_code_execution",
        "execute_user_code",
        "run_user_code",
        "eval_user_code",
        "exec_user_code",
        "import_user_module",
        "import_user_modules",
        "load_user_module",
        "load_user_modules",
        "preprocess_with_user_module",
        "run_tests",
        "test_command",
    }
)
_FALSE_STRINGS = frozenset({"", "0", "false", "no", "off", "none", "null"})


class NonExecutionPolicyError(ValueError):
    """Raised when a parser or config crosses the non-execution boundary."""


def parse_python_source_text(source: str, *, filename: str = "<unknown>") -> ast.Module:
    """Parse Python source text without importing, compiling, or executing it."""

    return ast.parse(source, filename=filename)


def reject_code_execution_config(payload: Mapping[str, Any], *, context: str = "config") -> None:
    """Reject config payloads that explicitly request untrusted code execution."""

    _reject_code_execution_config(payload, path=context)


def _reject_code_execution_config(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized_key = str(key).strip().lower().replace("-", "_")
            if normalized_key in FORBIDDEN_EXECUTION_CONFIG_KEYS and _requests_execution(child):
                raise NonExecutionPolicyError(
                    f"{child_path} requests untrusted code execution, which CodeLeWM forbids"
                )
            _reject_code_execution_config(child, path=child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_code_execution_config(child, path=f"{path}[{index}]")


def _requests_execution(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return True
