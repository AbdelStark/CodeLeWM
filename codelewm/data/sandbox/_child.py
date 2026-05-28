"""In-subprocess sandbox bootstrap.

This module is the entry point of the child process spawned by
:mod:`codelewm.data.sandbox.runner`. It reads a JSON job spec from stdin,
installs the policy guards, executes the payload, and writes a single JSON
result to stdout.

The module imports only stdlib modules at startup and only installs the
guards *after* its own imports have completed, so the guards do not
interfere with the bootstrap itself.

Public protocol (stdin):

::

    {
      "code": "<python source>",
      "input_repr": "<json-encoded value or null>",
      "function_name": "<name or null>",
      "stdin": "<string or empty>",
      "policy": {
        "import_allowlist": "stdlib_only",
        "deny_network": true,
        "deny_subprocess": true,
        "deny_filesystem_writes_outside_scratch": true,
        "memory_mb": 256,
        "cpu_seconds": 10,
        "output_truncation_bytes": 4096,
        "stdout_truncation_bytes": 4096,
        "python_hash_seed": 0
      },
      "scratch_dir": "/path/to/scratch"
    }

Public protocol (stdout, exactly one line):

::

    {
      "exit_code": "ok|raised|timeout|nondeterministic|policy_violation|oom|internal_error",
      "output_repr": "<truncated repr or null>",
      "output_truncated": false,
      "output_type": "int|float|...|exception|null",
      "stdout": "<truncated stdout>",
      "stdout_truncated": false,
      "stderr": "",
      "exception_class": "<name or null>",
      "exception_message": "<str or null>",
      "policy_violations": ["<event>", ...],
      "wall_time_ms": 17.4,
      "peak_rss_kb": 0
    }

Stderr from the child is captured by the runner and surfaced for debug
only; the result is the single JSON line on stdout.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
from typing import Any


def _stdlib_top_levels() -> frozenset[str]:
    """Return the set of stdlib top-level module names.

    ``sys.stdlib_module_names`` is the canonical answer on Python 3.10+.
    A small set of built-in module names is always permitted.
    """

    return frozenset(sys.stdlib_module_names) | frozenset(sys.builtin_module_names)


_STDLIB_NAMES: frozenset[str] = _stdlib_top_levels()
_POLICY_VIOLATIONS: list[str] = []


class _PolicyViolation(RuntimeError):
    """Raised when a sandboxed payload crosses a policy boundary."""


class _StdlibOnlyFinder:
    """Import meta-path finder that refuses non-stdlib top-level imports."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        top = fullname.partition(".")[0]
        if top in _STDLIB_NAMES:
            return None
        # Allow re-imports of names that are already loaded; the bootstrap
        # itself does not load any non-stdlib module before installing the
        # finder, so this allowance only covers built-ins.
        if fullname in sys.modules:
            return None
        message = f"import_denied:{fullname}"
        _POLICY_VIOLATIONS.append(message)
        raise _PolicyViolation(message)


def _install_import_hook() -> None:
    finder = _StdlibOnlyFinder()
    sys.meta_path.insert(0, finder)


def _install_module_guards(
    *,
    deny_network: bool,
    deny_subprocess: bool,
) -> None:
    """Belt-and-suspenders monkey-patches for network and subprocess primitives.

    Audit events (``socket.connect``, ``subprocess.Popen``, …) are the
    primary defense but their firing semantics vary slightly across Python
    builds and OSes. Replacing the canonical constructors with stubs that
    raise :class:`_PolicyViolation` guarantees the same observable
    behavior across CPython versions, kernels, and CI environments.
    """

    if deny_network:
        import socket as _socket

        def _denied_socket(*args: object, **kwargs: object) -> object:
            _POLICY_VIOLATIONS.append("network_denied:socket")
            raise _PolicyViolation("network_denied:socket")

        _socket.socket = _denied_socket  # type: ignore[assignment]
        # gethostbyname, create_connection, getaddrinfo round out the
        # high-level entry points that bypass socket.socket.
        for name in ("create_connection", "gethostbyname", "gethostbyname_ex"):
            if hasattr(_socket, name):
                def _denied_net(*args: object, _name: str = name, **kwargs: object) -> object:
                    _POLICY_VIOLATIONS.append(f"network_denied:socket.{_name}")
                    raise _PolicyViolation(f"network_denied:socket.{_name}")

                setattr(_socket, name, _denied_net)

    if deny_subprocess:
        import subprocess as _sp

        def _denied_popen(*args: object, **kwargs: object) -> object:
            _POLICY_VIOLATIONS.append("subprocess_denied:Popen")
            raise _PolicyViolation("subprocess_denied:Popen")

        _sp.Popen = _denied_popen  # type: ignore[assignment]
        # os.system is the most common exec entry point that bypasses
        # subprocess. The remaining os.exec* family is left to the audit
        # hook; patching them too aggressively can interfere with the
        # interpreter's own shutdown sequence.
        import os as _os

        def _denied_system(*args: object, **kwargs: object) -> object:
            _POLICY_VIOLATIONS.append("subprocess_denied:os.system")
            raise _PolicyViolation("subprocess_denied:os.system")

        _os.system = _denied_system  # type: ignore[assignment]


def _install_audit_hook(
    *,
    deny_network: bool,
    deny_subprocess: bool,
    deny_filesystem_writes_outside_scratch: bool,
    scratch_dir: str,
) -> None:
    forbidden_open_modes = ("w", "a", "x", "+")
    # Resolve symlinks once so the comparison is stable across macOS
    # ``/tmp`` -> ``/private/tmp`` rewrites and similar host quirks.
    scratch_real = os.path.realpath(scratch_dir)
    scratch_prefix = scratch_real + os.sep

    def hook(event: str, args: tuple[object, ...]) -> None:
        if deny_network:
            if event in {"socket.connect", "socket.bind", "socket.gethostbyname", "urllib.Request"}:
                _POLICY_VIOLATIONS.append(f"network_denied:{event}")
                raise _PolicyViolation(f"network_denied:{event}")
        if deny_subprocess:
            if event in {"subprocess.Popen", "os.system", "os.exec"}:
                _POLICY_VIOLATIONS.append(f"subprocess_denied:{event}")
                raise _PolicyViolation(f"subprocess_denied:{event}")
        if deny_filesystem_writes_outside_scratch and event == "open":
            # args: (path, mode, flags)
            if not args:
                return
            path = args[0]
            mode = args[1] if len(args) > 1 else ""
            if path is None or not isinstance(path, (str, bytes, os.PathLike)):
                return
            try:
                path_str = os.fspath(path)
            except TypeError:
                return
            if isinstance(path_str, bytes):
                try:
                    path_str = path_str.decode()
                except UnicodeDecodeError:
                    return
            mode_str = (
                mode.decode() if isinstance(mode, (bytes, bytearray)) else (mode or "")
            )
            if any(ch in mode_str for ch in forbidden_open_modes):
                resolved = os.path.realpath(path_str)
                allowed = resolved == scratch_real or resolved.startswith(scratch_prefix)
                # stdin/stdout/stderr-style file descriptors do not go through "open".
                # We also tolerate writes to /dev/null which are not state-bearing.
                if not allowed and resolved != os.devnull:
                    _POLICY_VIOLATIONS.append(f"filesystem_denied:{resolved}")
                    raise _PolicyViolation(f"filesystem_denied:{resolved}")

    sys.addaudithook(hook)


def _apply_rlimits(*, memory_mb: int, cpu_seconds: int) -> None:
    """Apply CPU and memory rlimits, best-effort.

    ``RLIMIT_DATA`` caps the heap-style data segment, which correlates with
    real memory usage. ``RLIMIT_AS`` caps the entire virtual address space,
    which on 64-bit Linux is misleading because Python's libraries reserve
    far more VAS than they actually use — capping VAS at 128 MB triggers
    a spurious ``MemoryError`` during interpreter startup. We prefer
    ``RLIMIT_DATA`` and fall back to ``RLIMIT_AS`` only when it is not
    available.
    """

    try:
        import resource  # type: ignore[import-not-found]
    except ImportError:
        return
    bytes_cap = memory_mb * 1024 * 1024
    set_memory = False
    if hasattr(resource, "RLIMIT_DATA"):
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (bytes_cap, bytes_cap))
            set_memory = True
        except (ValueError, OSError):
            pass
    if not set_memory:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (bytes_cap, bytes_cap))
        except (ValueError, OSError):
            # macOS often refuses RLIMIT_AS; treat as best-effort.
            pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value.encode("utf-8")) <= limit:
        return value, False
    truncated = value.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
    return truncated, True


def _classify_output_type(value: Any) -> str:
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


def _emit(result: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(result, default=str) + "\n")
    sys.stdout.flush()


def _peak_rss_kb() -> int:
    try:
        import resource  # type: ignore[import-not-found]
    except ImportError:
        return 0
    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
    except OSError:
        return 0
    # ru_maxrss is bytes on macOS, KB on Linux.
    value = int(rusage.ru_maxrss)
    if sys.platform == "darwin":
        return max(0, value // 1024)
    return max(0, value)


def _execute(job: dict[str, Any]) -> dict[str, object]:
    code = job["code"]
    function_name = job.get("function_name") or None
    input_repr = job.get("input_repr")
    stdin_text = job.get("stdin") or ""
    policy = job["policy"]
    scratch_dir = job["scratch_dir"]

    _apply_rlimits(memory_mb=int(policy["memory_mb"]), cpu_seconds=int(policy["cpu_seconds"]))
    _install_module_guards(
        deny_network=bool(policy.get("deny_network", True)),
        deny_subprocess=bool(policy.get("deny_subprocess", True)),
    )
    _install_audit_hook(
        deny_network=bool(policy.get("deny_network", True)),
        deny_subprocess=bool(policy.get("deny_subprocess", True)),
        deny_filesystem_writes_outside_scratch=bool(
            policy.get("deny_filesystem_writes_outside_scratch", True)
        ),
        scratch_dir=str(scratch_dir),
    )
    _install_import_hook()

    stdout_capture = io.StringIO()
    saved_stdin = sys.stdin
    saved_stdout = sys.stdout
    sys.stdin = io.StringIO(stdin_text)
    sys.stdout = stdout_capture

    start = time.perf_counter()
    namespace: dict[str, Any] = {"__name__": "__main__", "__builtins__": __builtins__}
    output_value: Any = None
    output_kind = "value"
    exception_class: str | None = None
    exception_message: str | None = None
    exit_code = "ok"

    try:
        compiled = compile(code, "<sandboxed>", "exec")
        exec(compiled, namespace)
        if function_name is not None:
            if function_name not in namespace:
                raise NameError(f"sandboxed code does not define {function_name!r}")
            fn = namespace[function_name]
            args: Any = json.loads(input_repr) if input_repr is not None else None
            if isinstance(args, list):
                output_value = fn(*args)
            elif isinstance(args, dict):
                output_value = fn(**args)
            elif args is None:
                output_value = fn()
            else:
                output_value = fn(args)
        else:
            # Script-style execution: code already executed at exec() time.
            # Stdout is the output.
            output_value = None
            output_kind = "stdout"
    except _PolicyViolation as exc:
        exit_code = "policy_violation"
        exception_class = "PolicyViolation"
        exception_message = str(exc)
    except MemoryError:
        exit_code = "oom"
        exception_class = "MemoryError"
        exception_message = ""
    except SystemExit as exc:
        # Treat sys.exit(non-zero) as a raised exception.
        if exc.code in (0, None):
            exit_code = "ok"
        else:
            exit_code = "raised"
            exception_class = "SystemExit"
            exception_message = str(exc.code)
    except BaseException as exc:  # noqa: BLE001
        exit_code = "raised"
        exception_class = type(exc).__name__
        exception_message = str(exc)
    finally:
        sys.stdin = saved_stdin
        sys.stdout = saved_stdout

    wall_ms = (time.perf_counter() - start) * 1000.0

    raw_stdout = stdout_capture.getvalue()
    stdout_truncated_to, stdout_was_truncated = _truncate(
        raw_stdout, int(policy["stdout_truncation_bytes"])
    )

    if exit_code == "ok":
        if output_kind == "stdout":
            # Script-style execution: the "output" the model has to learn
            # is the captured stdout (the deterministic effect of running
            # the program on stdin). ``output_value`` stays ``None`` and
            # is not the object the pack should record.
            output_repr = stdout_truncated_to
            output_was_truncated = stdout_was_truncated
            output_truncated_to = output_repr
            output_type = "str"
        else:
            try:
                output_repr = repr(output_value)
            except BaseException as exc:  # noqa: BLE001 - repr of pathological objects
                output_repr = f"<unreprable: {type(exc).__name__}>"
            output_truncated_to, output_was_truncated = _truncate(
                output_repr, int(policy["output_truncation_bytes"])
            )
            output_type = _classify_output_type(output_value)
    else:
        output_truncated_to, output_was_truncated = None, False
        output_type = "exception" if exit_code == "raised" else "none"

    return {
        "exit_code": exit_code,
        "output_repr": output_truncated_to,
        "output_truncated": output_was_truncated,
        "output_type": output_type,
        "output_kind": output_kind,
        "stdout": stdout_truncated_to,
        "stdout_truncated": stdout_was_truncated,
        "exception_class": exception_class,
        "exception_message": exception_message,
        "policy_violations": list(_POLICY_VIOLATIONS),
        "wall_time_ms": wall_ms,
        "peak_rss_kb": _peak_rss_kb(),
    }


def main() -> int:
    try:
        job = json.loads(sys.stdin.read())
    except (ValueError, OSError) as exc:
        _emit(
            {
                "exit_code": "internal_error",
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
                "policy_violations": [],
                "wall_time_ms": 0.0,
                "peak_rss_kb": 0,
                "output_repr": None,
                "output_truncated": False,
                "output_type": "none",
                "output_kind": "value",
                "stdout": "",
                "stdout_truncated": False,
            }
        )
        return 0

    try:
        result = _execute(job)
    except BaseException as exc:  # noqa: BLE001
        result = {
            "exit_code": "internal_error",
            "exception_class": type(exc).__name__,
            "exception_message": str(exc),
            "policy_violations": list(_POLICY_VIOLATIONS),
            "wall_time_ms": 0.0,
            "peak_rss_kb": 0,
            "output_repr": None,
            "output_truncated": False,
            "output_type": "none",
            "output_kind": "value",
            "stdout": "",
            "stdout_truncated": False,
            "traceback": traceback.format_exc(),
        }
    _emit(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    _exit_code = main()
    # ``os._exit`` skips finalizers / atexit, so anything we have
    # monkey-patched cannot fire during shutdown. The JSON result line
    # is already on stdout by the time we reach this call.
    os._exit(_exit_code)
