"""Deterministic public-safe hard-negative candidate pools (RFC-0016 #420).

This module is the *non-executing* candidate generator. It derives plausible
wrong candidates from an accepted reference using single-point AST mutations
(:func:`codelewm.data.wsd_mutations.generate_mutants`) plus definitional
no-action baits. It never imports the sandbox and never runs candidate code:
the only code-touching operation is ``ast.parse`` for the static-check status.

Trustworthy pass/fail labels for mutant candidates are constructed separately
in the data-prep layer (:mod:`codelewm.data.hard_negative_labeler`) using the
allowlisted sandbox. Mutants produced here default to ``label="unknown"`` so a
caller never asserts a label that was not verified; the two definitional baits
(no-action and near-no-action of a task that requires a change) are ``"fail"``
and the reference is ``"pass"``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from codelewm.data.wsd_mutations import generate_mutants
from codelewm.observability.manifest import compute_json_sha256
from codelewm.security.non_execution import parse_python_source_text

from .downstream_anti_saturation import (
    MAX_CANDIDATES_PER_PROBLEM,
    MIN_CANDIDATES_PER_PROBLEM,
    validate_hard_negative_class,
)


HARD_NEGATIVE_POOL_SCHEMA_VERSION = "codelewm.hard_negative_pool.v1"
LABEL_CONSTRUCTION_REPORT_SCHEMA_VERSION = "codelewm.downstream_label_construction_report.v1"

# Map a wsd-mutation operator family onto an RFC-0016 hard-negative class.
_MUTANT_OPERATOR_CLASS: dict[str, str] = {
    "compare": "wrong_branch",
    "boolop": "wrong_branch",
    "binop": "wrong_symbol",
    "augop": "wrong_symbol",
    "const": "deterministic_mutant",
}
_DEFAULT_MUTANT_CLASS = "deterministic_mutant"


class HardNegativePoolError(ValueError):
    """Raised when a hard-negative candidate pool cannot be generated."""


@dataclass(frozen=True)
class HardNegativeCandidate:
    """One generated candidate with class, label provenance, and a checksum."""

    candidate_id: str
    hard_negative_class: str
    after_text: str
    label: str  # "pass" | "fail" | "unknown"
    static_check: str  # "pass" | "fail"
    checksum: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_hard_negative_class(self.hard_negative_class)
        if self.label not in {"pass", "fail", "unknown"}:
            raise HardNegativePoolError("candidate label must be pass, fail, or unknown")
        if self.static_check not in {"pass", "fail"}:
            raise HardNegativePoolError("static_check must be pass or fail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "hard_negative_class": self.hard_negative_class,
            "label": self.label,
            "static_check": self.static_check,
            "checksum": self.checksum,
            "provenance": dict(self.provenance),
        }


def _static_check(after_text: str) -> str:
    try:
        parse_python_source_text(after_text, filename="<hard_negative_candidate>")
        return "pass"
    except Exception:  # noqa: BLE001 - any parse failure is a static-check fail
        return "fail"


def _checksum(*, hard_negative_class: str, after_text: str) -> str:
    return compute_json_sha256(
        {"hard_negative_class": hard_negative_class, "after_text": after_text}
    )


def _candidate(
    *,
    index: int,
    hard_negative_class: str,
    after_text: str,
    label: str,
    provenance: Mapping[str, Any],
) -> HardNegativeCandidate:
    candidate_id = f"hn_{index:03d}_{hard_negative_class}"
    return HardNegativeCandidate(
        candidate_id=candidate_id,
        hard_negative_class=hard_negative_class,
        after_text=after_text,
        label=label,
        static_check=_static_check(after_text),
        checksum=_checksum(hard_negative_class=hard_negative_class, after_text=after_text),
        provenance=dict(provenance),
    )


def generate_hard_negative_pool(
    *,
    before_text: str,
    reference_after_text: str,
    seed: int = 0,
    pool_size: int = MIN_CANDIDATES_PER_PROBLEM,
) -> tuple[HardNegativeCandidate, ...]:
    """Generate a deterministic public-safe hard-negative candidate pool.

    The pool always contains the passing reference, a no-action bait, and a
    near-no-action bait, then fills the remainder with single-point AST mutants
    of the reference. Output is deterministic given ``(reference, seed,
    pool_size)``. Raises if ``pool_size`` is outside the RFC-0016 6-12 range or
    the reference does not parse.
    """

    if not (MIN_CANDIDATES_PER_PROBLEM <= pool_size <= MAX_CANDIDATES_PER_PROBLEM):
        raise HardNegativePoolError(
            f"pool_size must be in {MIN_CANDIDATES_PER_PROBLEM}-{MAX_CANDIDATES_PER_PROBLEM}"
        )
    if _static_check(reference_after_text) != "pass":
        raise HardNegativePoolError("reference_after_text must be parseable Python")

    candidates: list[HardNegativeCandidate] = []
    candidates.append(
        _candidate(
            index=len(candidates),
            hard_negative_class="passing_reference",
            after_text=reference_after_text,
            label="pass",
            provenance={"origin": "reference", "label_source": "definitional"},
        )
    )
    candidates.append(
        _candidate(
            index=len(candidates),
            hard_negative_class="no_action_bait",
            after_text=before_text,
            label="fail",
            provenance={"origin": "before_state", "label_source": "definitional"},
        )
    )
    near_no_action = before_text.rstrip("\n") + "\n# no-op clarifying comment, no behavior change\n"
    candidates.append(
        _candidate(
            index=len(candidates),
            hard_negative_class="near_no_action_bait",
            after_text=near_no_action,
            label="fail",
            provenance={"origin": "before_state_comment", "label_source": "definitional"},
        )
    )

    remaining = pool_size - len(candidates)
    mutants = generate_mutants(reference_after_text, count=remaining, seed=seed)
    for mutant in mutants:
        hard_negative_class = _MUTANT_OPERATOR_CLASS.get(mutant.operator, _DEFAULT_MUTANT_CLASS)
        candidates.append(
            _candidate(
                index=len(candidates),
                hard_negative_class=hard_negative_class,
                after_text=mutant.source,
                label="unknown",
                provenance={
                    "origin": "single_point_mutant",
                    "label_source": "unverified",
                    "mutation_operator": mutant.operator,
                    "mutation_description": mutant.description,
                },
            )
        )

    return tuple(candidates)


def build_label_construction_report(
    candidates: Sequence[HardNegativeCandidate],
    *,
    sandbox_used: bool = False,
    sandbox_policy_version: str | None = None,
) -> dict[str, Any]:
    """Build the ``codelewm.downstream_label_construction_report.v1`` payload.

    This is a pure accounting report: it records how each candidate's label was
    constructed (definitional vs sandbox-verified vs unverified) without running
    any candidate code. The sandbox path itself lives in
    :mod:`codelewm.data.hard_negative_labeler`.
    """

    label_counts: dict[str, int] = {}
    label_source_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    for candidate in candidates:
        label_counts[candidate.label] = label_counts.get(candidate.label, 0) + 1
        source = str(candidate.provenance.get("label_source", "unknown"))
        label_source_counts[source] = label_source_counts.get(source, 0) + 1
        class_counts[candidate.hard_negative_class] = (
            class_counts.get(candidate.hard_negative_class, 0) + 1
        )
    return {
        "schema_version": LABEL_CONSTRUCTION_REPORT_SCHEMA_VERSION,
        "candidate_count": len(candidates),
        "sandbox_used": bool(sandbox_used),
        "sandbox_policy_version": sandbox_policy_version,
        "label_counts": dict(sorted(label_counts.items())),
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "hard_negative_class_counts": dict(sorted(class_counts.items())),
        "unverified_label_count": label_counts.get("unknown", 0),
    }


def _ensure_json_native(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HardNegativePoolError(f"{field_name} must be JSON-native: {exc}") from exc
