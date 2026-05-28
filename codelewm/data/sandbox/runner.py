"""Sandbox runner — spawns the child process, enforces the wall-clock budget.

The runner builds a subprocess invocation that runs
``python -m codelewm.data.sandbox._child``. The child reads the job spec
from stdin and writes a single JSON result line on stdout. The runner
applies the wall-clock timeout via ``subprocess.run(timeout=...)`` and
performs the optional determinism re-run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .policy import (
    DEFAULT_SANDBOX_POLICY,
    SandboxExitCode,
    SandboxPolicy,
    SandboxPolicyError,
)


SANDBOX_RESULT_SCHEMA_VERSION = "codelewm.sandbox_result.v1"
_CHILD_MODULE = "codelewm.data.sandbox._child"


class SandboxRunnerError(RuntimeError):
    """Raised when the runner cannot complete an invocation cleanly.

    These errors indicate a problem with the runner itself (cannot spawn
    the child, malformed runner inputs, broken child protocol). Normal
    policy failures from the child are reported via the
    :attr:`SandboxResult.exit_code` field, not via this exception.
    """


@dataclass(frozen=True)
class SandboxResult:
    """Result of one ``run_one`` invocation.

    The schema version is :data:`SANDBOX_RESULT_SCHEMA_VERSION`.
    """

    schema_version: str
    exit_code: SandboxExitCode
    output_repr: str | None
    output_truncated: bool
    output_kind: str
    output_type: str
    stdout: str
    stdout_truncated: bool
    exception_class: str | None
    exception_message: str | None
    policy_violations: tuple[str, ...]
    wall_time_ms: float
    peak_rss_kb: int
    determinism_check: bool
    policy: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""

        return {
            "schema_version": self.schema_version,
            "exit_code": self.exit_code.value,
            "output_repr": self.output_repr,
            "output_truncated": self.output_truncated,
            "output_kind": self.output_kind,
            "output_type": self.output_type,
            "stdout": self.stdout,
            "stdout_truncated": self.stdout_truncated,
            "exception_class": self.exception_class,
            "exception_message": self.exception_message,
            "policy_violations": list(self.policy_violations),
            "wall_time_ms": self.wall_time_ms,
            "peak_rss_kb": self.peak_rss_kb,
            "determinism_check": self.determinism_check,
            "policy": dict(self.policy),
        }

    @property
    def ok(self) -> bool:
        return self.exit_code is SandboxExitCode.OK


def classify_output_type(value: Any) -> str:
    """Public re-export of the child's type classifier for use in tests."""

    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, set):
        return "set"
    return "other"


def run_one(
    code: str,
    *,
    input_repr: str | None = None,
    function_name: str | None = None,
    stdin_text: str = "",
    policy: SandboxPolicy | None = None,
    scratch_dir: Path | None = None,
) -> SandboxResult:
    """Run one ``(code, input)`` pair inside the sandbox.

    Args:
        code: Python source to execute. Must compile as a module.
        input_repr: JSON-encoded argument value for ``function_name``. If
            the decoded value is a list, it is splatted as positional
            args; a dict splats as keyword args; any other value is
            passed as a single positional arg. Required when
            ``function_name`` is provided.
        function_name: name of a function to call after executing the
            module. ``None`` runs the code as a script and treats
            captured stdout as the output.
        stdin_text: text fed to the child's stdin (used by script-style
            payloads).
        policy: a :class:`SandboxPolicy`. Defaults to the
            ``stdlib_only`` policy from the operations doc.
        scratch_dir: directory the child may write to. A fresh
            temporary directory is used when omitted.

    Returns:
        A :class:`SandboxResult` with the child's exit code, output, and
        policy-violation list. When ``policy.determinism_check`` is
        true, the runner executes the payload twice and downgrades the
        result to :attr:`SandboxExitCode.NONDETERMINISTIC` if outputs
        differ.

    Raises:
        SandboxRunnerError: when the child cannot be spawned or the
            child protocol is violated.
    """

    effective_policy = policy or DEFAULT_SANDBOX_POLICY
    if function_name is not None and input_repr is None:
        raise SandboxPolicyError(
            "input_repr is required when function_name is provided"
        )

    with _scratch(scratch_dir) as scratch:
        first = _invoke(
            code=code,
            input_repr=input_repr,
            function_name=function_name,
            stdin_text=stdin_text,
            policy=effective_policy,
            scratch_dir=scratch,
        )
        determinism_check = False
        if effective_policy.determinism_check and first["exit_code"] in ("ok", "raised"):
            second = _invoke(
                code=code,
                input_repr=input_repr,
                function_name=function_name,
                stdin_text=stdin_text,
                policy=effective_policy,
                scratch_dir=scratch,
            )
            determinism_check = _outputs_match(first, second)
            if not determinism_check:
                first["exit_code"] = SandboxExitCode.NONDETERMINISTIC.value

    return _to_result(first, effective_policy, determinism_check)


def _outputs_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = (
        "exit_code",
        "output_repr",
        "output_type",
        "output_kind",
        "stdout",
        "exception_class",
        "exception_message",
    )
    return all(a.get(k) == b.get(k) for k in keys)


def _invoke(
    *,
    code: str,
    input_repr: str | None,
    function_name: str | None,
    stdin_text: str,
    policy: SandboxPolicy,
    scratch_dir: Path,
) -> dict[str, Any]:
    job = {
        "code": code,
        "input_repr": input_repr,
        "function_name": function_name,
        "stdin": stdin_text,
        "policy": policy.as_dict(),
        "scratch_dir": str(scratch_dir),
    }
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(policy.python_hash_seed)
    # Inherit PYTHONPATH so the child can resolve the codelewm package.
    cmd = [sys.executable, "-I", "-m", _CHILD_MODULE]
    # ``-I`` enables isolated mode but also strips PYTHONPATH; restore the
    # current sys.path through PYTHONPATH so the child can locate the
    # codelewm package without needing it installed.
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    timeout_s = policy.timeout_ms / 1000.0
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            input=json.dumps(job),
            text=True,
            cwd=str(scratch_dir),
            env=env,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": SandboxExitCode.TIMEOUT.value,
            "output_repr": None,
            "output_truncated": False,
            "output_type": "none",
            "output_kind": "timeout",
            "stdout": (exc.stdout or "")[: policy.stdout_truncation_bytes]
            if isinstance(exc.stdout, str)
            else "",
            "stdout_truncated": False,
            "exception_class": "Timeout",
            "exception_message": f"wall_time_ms>{policy.timeout_ms}",
            "policy_violations": [],
            "wall_time_ms": (time.perf_counter() - start) * 1000.0,
            "peak_rss_kb": 0,
        }
    except FileNotFoundError as exc:
        raise SandboxRunnerError(f"cannot spawn sandbox child: {exc}") from exc

    if not completed.stdout.strip():
        raise SandboxRunnerError(
            "sandbox child produced no result line. "
            f"stderr={completed.stderr!r} return_code={completed.returncode}"
        )

    last_line = completed.stdout.strip().splitlines()[-1]
    try:
        payload = json.loads(last_line)
    except ValueError as exc:
        raise SandboxRunnerError(
            f"sandbox child returned non-JSON output: {last_line!r}"
        ) from exc

    # Map child-side internal_error from oom-style returncode to oom.
    if completed.returncode != 0 and payload.get("exit_code") == "ok":
        payload["exit_code"] = SandboxExitCode.INTERNAL_ERROR.value
        payload["exception_class"] = payload.get("exception_class") or "ChildProcessError"
        payload["exception_message"] = (
            payload.get("exception_message") or f"return_code={completed.returncode}"
        )

    if payload.get("policy_violations"):
        payload["exit_code"] = SandboxExitCode.POLICY_VIOLATION.value

    return payload


def _to_result(
    payload: dict[str, Any], policy: SandboxPolicy, determinism_check: bool
) -> SandboxResult:
    try:
        exit_code = SandboxExitCode(payload["exit_code"])
    except (KeyError, ValueError) as exc:
        raise SandboxRunnerError(
            f"sandbox child returned invalid exit_code: {payload.get('exit_code')!r}"
        ) from exc
    return SandboxResult(
        schema_version=SANDBOX_RESULT_SCHEMA_VERSION,
        exit_code=exit_code,
        output_repr=payload.get("output_repr"),
        output_truncated=bool(payload.get("output_truncated", False)),
        output_kind=str(payload.get("output_kind", "value")),
        output_type=str(payload.get("output_type", "none")),
        stdout=str(payload.get("stdout", "")),
        stdout_truncated=bool(payload.get("stdout_truncated", False)),
        exception_class=payload.get("exception_class"),
        exception_message=payload.get("exception_message"),
        policy_violations=tuple(payload.get("policy_violations", ())),
        wall_time_ms=float(payload.get("wall_time_ms", 0.0)),
        peak_rss_kb=int(payload.get("peak_rss_kb", 0)),
        determinism_check=determinism_check,
        policy=policy.as_dict(),
    )


class _scratch:
    """Context manager that yields a scratch directory.

    Either uses the caller-provided path (without deleting it) or creates a
    fresh temp dir that is removed on exit.
    """

    def __init__(self, override: Path | None) -> None:
        self._override = override
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        if self._override is not None:
            self._override.mkdir(parents=True, exist_ok=True)
            return self._override
        self._tmp = tempfile.TemporaryDirectory(prefix="codelewm-sandbox-")
        return Path(self._tmp.name)

    def __exit__(self, *_: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
