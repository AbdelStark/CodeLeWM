"""Non-execution parsing and configuration guards.

These helpers cover the **training, inference, scoring, indexing, evaluation,
and dataset-construction** paths. They make sure those paths never compile,
import, evaluate, or run untrusted Python.

The execution-substrate data builder introduced by RFC-0014 is a separate,
named subsystem that lives at :mod:`codelewm.data.sandbox` (added in #260).
The sandbox executes licensed public Python submissions at data-build time
inside an isolated subprocess to capture deterministic outputs. It is **not**
governed by the helpers in this module; it has its own claim boundary at
``codelewm/security/claim_boundaries/execution_substrate.v1.md`` and its own
operations policy at ``docs/operations/sandbox_policy.md``.

The two contracts are complementary, not contradictory:

- :mod:`codelewm.security.non_execution` keeps untrusted code from running on
  the model paths.
- :mod:`codelewm.data.sandbox` is the one-shot, operator-controlled data-prep
  component that the spec section 06 anticipates ("CodeLeWM may support
  sandboxed execution later, but that must be a separate opt-in subsystem with
  its own isolation, manifest, and logging contract").
"""

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
