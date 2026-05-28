"""Sandbox policy definitions.

The policy is plain data. It is consumed by :mod:`codelewm.data.sandbox.runner`
to build a subprocess invocation and by :mod:`codelewm.data.sandbox._child` to
install the in-subprocess guards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class SandboxPolicyError(ValueError):
    """Raised when a policy field is invalid."""


class SandboxExitCode(str, Enum):
    """Outcome of a single sandbox invocation."""

    OK = "ok"
    RAISED = "raised"
    TIMEOUT = "timeout"
    NONDETERMINISTIC = "nondeterministic"
    POLICY_VIOLATION = "policy_violation"
    OOM = "oom"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class SandboxPolicy:
    """One sandbox invocation's policy.

    All fields are validated in :meth:`__post_init__`. The defaults match
    the operations doc (``docs/operations/sandbox_policy.md``).

    Attributes:
        import_allowlist: ``"stdlib_only"`` is the only supported value.
            The child uses :data:`sys.stdlib_module_names` to enforce it.
        timeout_ms: wall-clock timeout in milliseconds. Range [10, 60_000].
        memory_mb: RSS cap in megabytes. Range [16, 4096]. Enforced via
            ``RLIMIT_AS`` on POSIX. Ignored on platforms without
            ``resource``.
        cpu_seconds: CPU-time cap in seconds. Range [1, 120]. Enforced via
            ``RLIMIT_CPU`` on POSIX. Ignored on platforms without
            ``resource``.
        deny_network: when true, the child denies socket creation.
        deny_subprocess: when true, the child denies subprocess spawning.
        deny_filesystem_writes_outside_scratch: when true, the child
            denies writes outside the scratch directory.
        determinism_check: when true, the runner executes the same payload
            twice with identical ``PYTHONHASHSEED`` and rejects records
            whose outputs differ.
        output_truncation_bytes: ``repr(output)`` truncation cap. Range
            [256, 65_536].
        stdout_truncation_bytes: stdout capture cap. Range [256, 65_536].
        python_hash_seed: deterministic ``PYTHONHASHSEED`` value for the
            child. ``0`` disables hash randomization in the child.
    """

    import_allowlist: Literal["stdlib_only"] = "stdlib_only"
    timeout_ms: int = 5_000
    memory_mb: int = 256
    cpu_seconds: int = 10
    deny_network: bool = True
    deny_subprocess: bool = True
    deny_filesystem_writes_outside_scratch: bool = True
    determinism_check: bool = True
    output_truncation_bytes: int = 4 * 1024
    stdout_truncation_bytes: int = 4 * 1024
    python_hash_seed: int = 0

    POLICY_VERSION: str = field(default="codelewm.sandbox_policy.v1", init=False)

    def __post_init__(self) -> None:
        if self.import_allowlist != "stdlib_only":
            raise SandboxPolicyError(
                f"unsupported import_allowlist {self.import_allowlist!r}; only 'stdlib_only' is implemented"
            )
        _check_range("timeout_ms", self.timeout_ms, 10, 60_000)
        _check_range("memory_mb", self.memory_mb, 16, 4096)
        _check_range("cpu_seconds", self.cpu_seconds, 1, 120)
        _check_range("output_truncation_bytes", self.output_truncation_bytes, 256, 65_536)
        _check_range("stdout_truncation_bytes", self.stdout_truncation_bytes, 256, 65_536)
        if self.python_hash_seed < 0:
            raise SandboxPolicyError(
                f"python_hash_seed must be non-negative, got {self.python_hash_seed}"
            )

    def as_dict(self) -> dict[str, object]:
        """Serialize the policy to a JSON-friendly dict."""

        return {
            "policy_version": self.POLICY_VERSION,
            "import_allowlist": self.import_allowlist,
            "timeout_ms": self.timeout_ms,
            "memory_mb": self.memory_mb,
            "cpu_seconds": self.cpu_seconds,
            "deny_network": self.deny_network,
            "deny_subprocess": self.deny_subprocess,
            "deny_filesystem_writes_outside_scratch": self.deny_filesystem_writes_outside_scratch,
            "determinism_check": self.determinism_check,
            "output_truncation_bytes": self.output_truncation_bytes,
            "stdout_truncation_bytes": self.stdout_truncation_bytes,
            "python_hash_seed": self.python_hash_seed,
        }


def _check_range(name: str, value: int, lo: int, hi: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SandboxPolicyError(f"{name} must be int, got {type(value).__name__}")
    if value < lo or value > hi:
        raise SandboxPolicyError(f"{name} must be in [{lo}, {hi}], got {value}")


DEFAULT_SANDBOX_POLICY = SandboxPolicy()
