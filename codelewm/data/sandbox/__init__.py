"""Sandboxed deterministic Python execution for data-prep only.

The sandbox is a *one-shot data builder* used to capture
``(code, input, output)`` triples from licensed public Python submissions at
data-build time. It is not part of the training or inference paths.

The contract is documented in:

- ``codelewm/security/claim_boundaries/execution_substrate.v1.md`` — the
  claim boundary every artifact that depends on the sandbox must reference;
- ``docs/operations/sandbox_policy.md`` — the operations doc that lists the
  policy, threat model, and audit-trail format;
- ``docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`` — the RFC
  that motivates the substrate pivot;
- ``docs/spec/06-security.md`` — the non-execution policy that anticipates
  this subsystem.

The public surface is intentionally narrow:

- :class:`SandboxPolicy` describes the policy applied to one invocation.
- :func:`run_one` executes one ``(code, input)`` pair under a policy and
  returns a :class:`SandboxResult` that the pack builder consumes.
- :data:`SANDBOX_RESULT_SCHEMA_VERSION` versions the serialized result.

Only the data-prep pipeline imports this module. Training, scoring,
indexing, and evaluation paths are forbidden from importing it; a
structural test in ``tests/security/`` enforces that boundary.
"""

from __future__ import annotations

from .policy import (
    DEFAULT_SANDBOX_POLICY,
    SandboxExitCode,
    SandboxPolicy,
    SandboxPolicyError,
)
from .runner import (
    SANDBOX_RESULT_SCHEMA_VERSION,
    SandboxResult,
    SandboxRunnerError,
    classify_output_type,
    run_one,
)

__all__ = [
    "DEFAULT_SANDBOX_POLICY",
    "SANDBOX_RESULT_SCHEMA_VERSION",
    "SandboxExitCode",
    "SandboxPolicy",
    "SandboxPolicyError",
    "SandboxResult",
    "SandboxRunnerError",
    "classify_output_type",
    "run_one",
]
