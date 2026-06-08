"""Sandbox label construction for hard-negative candidate pools (RFC-0016 #420).

This is a data-prep component: it is the *only* part of the hard-downstream
benchmark path allowed to execute candidate code, and it does so exclusively
through the allowlisted disposable-checkout sandbox
(:mod:`codelewm.data.sandbox`) under the stdlib-only policy with timeouts,
output limits, and a determinism check. It assigns trustworthy ``pass`` /
``fail`` labels to generated candidates (e.g. the ``"unknown"`` mutants from
:mod:`codelewm.eval.hard_negative_pool`) so the benchmark never asserts an
unverified label.

The model-scoring path must never import this module; it lives under
``codelewm/data`` for exactly that reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from codelewm.data.sandbox import (
    DEFAULT_SANDBOX_POLICY,
    SandboxPolicy,
    SandboxPolicyError,
    run_one,
)
from codelewm.data.sandbox.runner import SandboxRunnerError
from codelewm.observability.manifest import compute_json_sha256


HARD_NEGATIVE_LABELED_CANDIDATE_SCHEMA_VERSION = "codelewm.hard_negative_labeled_candidate.v1"
SANDBOX_LABEL_CONSTRUCTION_REPORT_SCHEMA_VERSION = (
    "codelewm.downstream_label_construction_report.v1"
)


class HardNegativeLabelerError(ValueError):
    """Raised when sandbox label construction cannot run."""


@dataclass(frozen=True)
class LabelTestCase:
    """One hidden-test input/expected-output pair for label construction."""

    input_id: str
    input_repr: str
    expected_output: str
    function_name: str | None = None
    input_kind: str = "function_call"

    def __post_init__(self) -> None:
        if self.input_kind not in {"function_call", "stdin"}:
            raise HardNegativeLabelerError("input_kind must be function_call or stdin")
        if self.input_kind == "function_call" and not self.function_name:
            raise HardNegativeLabelerError(
                "function_call test cases require a function_name"
            )


@dataclass(frozen=True)
class CandidateLabel:
    """The constructed label for one candidate plus per-case sandbox evidence."""

    candidate_id: str
    label: str  # "pass" | "fail"
    case_results: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HARD_NEGATIVE_LABELED_CANDIDATE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "label": self.label,
            "case_results": [dict(result) for result in self.case_results],
        }


def label_candidate(
    *,
    candidate_id: str,
    after_text: str,
    test_cases: Sequence[LabelTestCase],
    sandbox_policy: SandboxPolicy | None = None,
) -> CandidateLabel:
    """Construct a pass/fail label for one candidate by sandbox execution.

    A candidate is ``"pass"`` only when every test case runs to ``ok`` and the
    sandboxed output repr matches the expected output. Any non-ok exit
    (raised/timeout/policy violation/...) or mismatch yields ``"fail"``. The
    sandbox enforces the disposable-checkout, timeout, output-limit, and
    determinism contract from ``codelewm.data.sandbox``.
    """

    if not test_cases:
        raise HardNegativeLabelerError("at least one test case is required to label a candidate")
    policy = sandbox_policy or DEFAULT_SANDBOX_POLICY
    case_results: list[dict[str, Any]] = []
    all_passed = True
    for case in test_cases:
        try:
            if case.input_kind == "function_call":
                result = run_one(
                    after_text,
                    input_repr=case.input_repr,
                    function_name=case.function_name,
                    policy=policy,
                )
            else:
                result = run_one(after_text, stdin_text=case.input_repr, policy=policy)
        except (SandboxPolicyError, SandboxRunnerError) as exc:
            raise HardNegativeLabelerError(
                f"sandbox labeling failed for {candidate_id}: {exc}"
            ) from exc
        passed = bool(result.ok and result.output_repr == case.expected_output)
        all_passed = all_passed and passed
        case_results.append(
            {
                "input_id": case.input_id,
                "input_kind": case.input_kind,
                "function_name": case.function_name,
                "exit_code": result.exit_code.value,
                "passed": passed,
                "output_repr_sha256": (
                    None if result.output_repr is None else compute_json_sha256(result.output_repr)
                ),
                "expected_output_sha256": compute_json_sha256(case.expected_output),
                "policy_violations": list(result.policy_violations),
                "wall_time_ms": result.wall_time_ms,
                "determinism_check": result.determinism_check,
            }
        )
    return CandidateLabel(
        candidate_id=candidate_id,
        label="pass" if all_passed else "fail",
        case_results=tuple(case_results),
    )


def label_candidates(
    candidates: Mapping[str, str],
    *,
    test_cases: Sequence[LabelTestCase],
    sandbox_policy: SandboxPolicy | None = None,
) -> dict[str, CandidateLabel]:
    """Label many ``candidate_id -> after_text`` candidates against shared tests."""

    return {
        candidate_id: label_candidate(
            candidate_id=candidate_id,
            after_text=after_text,
            test_cases=test_cases,
            sandbox_policy=sandbox_policy,
        )
        for candidate_id, after_text in candidates.items()
    }


def build_sandbox_label_construction_report(
    labels: Mapping[str, CandidateLabel],
    *,
    sandbox_policy: SandboxPolicy | None = None,
) -> dict[str, Any]:
    """Build a sandbox-verified label-construction report."""

    policy = sandbox_policy or DEFAULT_SANDBOX_POLICY
    label_counts: dict[str, int] = {}
    for candidate_label in labels.values():
        label_counts[candidate_label.label] = label_counts.get(candidate_label.label, 0) + 1
    return {
        "schema_version": SANDBOX_LABEL_CONSTRUCTION_REPORT_SCHEMA_VERSION,
        "candidate_count": len(labels),
        "sandbox_used": True,
        "sandbox_policy_version": policy.POLICY_VERSION,
        "label_counts": dict(sorted(label_counts.items())),
        "label_source_counts": {"sandbox": len(labels)},
        "labels": {
            candidate_id: candidate_label.label
            for candidate_id, candidate_label in sorted(labels.items())
        },
    }
