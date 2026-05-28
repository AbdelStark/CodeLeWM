# Sandbox Policy (Execution-Substrate Data Builder)

This document describes the policy and operating model for the sandbox
that the execution-substrate data pipeline uses. The sandbox is introduced
by RFC-0014 and tracked under #259.

The sandbox lives at `codelewm.data.sandbox` (added in #260). It is not
part of the training or inference paths. It is a data-prep component that
turns licensed public Python submissions into deterministic
`(code, input, output)` records.

## Policy Summary

| Property | Default |
|----------|---------|
| Import allowlist | stdlib only (`sys.stdlib_module_names`) |
| Network | denied |
| Filesystem writes outside scratch dir | denied |
| CPU timeout | 5 seconds |
| Memory cap | 256 MB |
| Determinism check | required; two runs with same `PYTHONHASHSEED` |
| Output truncation | 4 KB canonical `repr()` of return value |
| Stdout capture | enabled, separate from return value |
| Audit hook | `sys.addaudithook` watches filesystem writes and subprocess calls |
| Exit codes | `ok`, `raised`, `timeout`, `nondeterministic`, `policy_violation`, `oom` |

The policy is enforced inside the subprocess via an import hook installed
in the bootstrap and an audit hook registered before user code is loaded.
Resource limits are applied via `resource.setrlimit` on POSIX systems.

## Build-Host Requirements

The sandbox is supported on Linux and macOS build hosts. Windows is not
supported initially.

Build hosts must:

- run a current OS with kernel-level resource accounting;
- have at least 8 GB of RAM (sandbox memory cap × concurrency);
- be controlled by a CodeLeWM operator;
- not be shared with user-facing inference or scoring workloads;
- record per-sandbox-invocation logs to the run manifest.

## Threat Model

See the execution-substrate claim boundary
(`codelewm/security/claim_boundaries/execution_substrate.v1.md`) for the
governing language.

The short version:

- the source code in each `(code, input)` record is untrusted;
- a determined attacker who controls the source code may escape the
  Python-level sandbox, but the build host is the trust boundary;
- inputs come from licensed public datasets, so the attacker-controlled-
  source case is a data-quality risk, not a production risk;
- the published artifact is data, not an executable.

## Escape Mitigations

The sandbox is defense-in-depth, not a guarantee:

- subprocess isolation prevents in-process leaks;
- import allowlist limits the reachable Python surface;
- audit hook denies `open(..., "w"|"a"|"x")` outside the scratch dir;
- `resource.setrlimit(RLIMIT_AS)` caps memory; `RLIMIT_CPU` caps CPU;
- a wall-clock timeout kills runaway processes;
- the subprocess runs as an unprivileged user where available;
- network is denied by replacing `socket` with a deny-list shim in the
  bootstrap.

Operators who want stronger isolation may run the sandbox under
`firejail`, `nsjail`, `bubblewrap`, or a container; the sandbox writes
all per-run state to a single scratch directory to make those wrappers
straightforward.

## Audit Trail

Each sandbox invocation produces a JSON record in the run manifest:

```json
{
  "source_dataset": "mbpp",
  "source_problem_id": "mbpp/123",
  "source_submission_id": "mbpp/123/sol-a",
  "input_id": "case-001",
  "policy": "stdlib-only",
  "timeout_ms": 5000,
  "memory_mb": 256,
  "wall_time_ms": 17.4,
  "peak_rss_kb": 23408,
  "exit_code": "ok",
  "determinism_check": true,
  "policy_violations": [],
  "stdout_truncated": false,
  "output_repr_truncated": false
}
```

These records are aggregated into the execution-pack manifest under
`sandbox_audit_summary` and are part of the published artifact provenance.

## Out of Scope

- third-party imports (numpy, pandas, requests, ...) — deferred;
- multi-step traces (intermediate variable states) — deferred to v0.6.1;
- Windows host support;
- any training-time or inference-time execution.

## References

- Execution-substrate claim boundary:
  `codelewm/security/claim_boundaries/execution_substrate.v1.md`
- RFC-0014: `docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`
- Substrate roadmap:
  `docs/roadmap/EXECUTION_TRACE_WORLD_MODEL.md`
- Non-execution policy: `docs/spec/06-security.md`
- Implementation issue (sandbox): #260
- Tracker: #259
