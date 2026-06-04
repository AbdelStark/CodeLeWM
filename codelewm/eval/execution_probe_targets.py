"""Latent probe target label extractors for the execution substrate.

Each probe target maps a packed execution record (or its dict form) to
a label that a downstream linear-classifier evaluator can train against.
The label extractors are pure functions — they make no assumptions about
the underlying probe model and never touch latents directly.

The targets here are the execution-substrate analogues of the
commit-edit targets in :mod:`codelewm.eval.latent_probe`. The set is
intentionally minimal and well-grounded:

- ``output_type`` — multi-class over the stable
  :data:`codelewm.training.OUTPUT_TYPE_VOCAB`. Tests *what* the program
  computes.
- ``will_raise`` — binary. Tests whether the latent encodes whether a
  program raises on this input.
- ``output_magnitude_bucket`` — multi-class over numeric outputs only.
  Distinguishes negative / zero / small / medium / large.
- ``output_length_bucket`` — multi-class over sequence/string outputs.
  Distinguishes empty / short / medium / long / huge.
- ``arithmetic_vs_string_vs_collection`` — 3-class coarse semantic
  shape.
- ``judge_verdict`` — 3-class on records where the source carried a
  judge verdict (currently only CodeNet).
- ``passed`` — binary over v0.8 pass/fail records that explicitly carry a
  sandboxed completion-level correctness label.

For records that do not belong to a target's domain (e.g. numeric
magnitude on a string output), the extractor returns ``None`` and the
classifier excludes the record from that target's eval. This matches
the "applicable-only" policy in the existing probe runner.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


EXECUTION_PROBE_TARGET_SCHEMA_VERSION = "codelewm.eval.execution_probe_target.v1"

EXECUTION_PROBE_TARGETS: tuple[str, ...] = (
    "output_type",
    "will_raise",
    "output_magnitude_bucket",
    "output_length_bucket",
    "arithmetic_vs_string_vs_collection",
    "judge_verdict",
    "passed",
)

ExecutionProbeTarget = Literal[
    "output_type",
    "will_raise",
    "output_magnitude_bucket",
    "output_length_bucket",
    "arithmetic_vs_string_vs_collection",
    "judge_verdict",
    "passed",
]


class ExecutionProbeTargetError(ValueError):
    """Raised when a probe target is unknown or a record is malformed."""


@dataclass(frozen=True)
class LabelExtraction:
    """One target's labels across a record stream.

    ``labels[i]`` is ``None`` when the record is outside the target's
    applicable domain.
    """

    target: str
    labels: tuple[object | None, ...]
    applicable_count: int
    class_distribution: dict[str, int]


_MAGNITUDE_BUCKETS = ("negative", "zero", "small", "medium", "large")
_LENGTH_BUCKETS = ("empty", "short", "medium", "long", "huge")


def label_record(record: Mapping[str, Any], target: str) -> object | None:
    """Compute the label for one record under ``target``.

    Returns ``None`` when the record is outside the target's domain.
    """

    if target == "output_type":
        return _label_output_type(record)
    if target == "will_raise":
        return _label_will_raise(record)
    if target == "output_magnitude_bucket":
        return _label_output_magnitude(record)
    if target == "output_length_bucket":
        return _label_output_length(record)
    if target == "arithmetic_vs_string_vs_collection":
        return _label_coarse_kind(record)
    if target == "judge_verdict":
        return _label_judge_verdict(record)
    if target == "passed":
        return _label_passed(record)
    raise ExecutionProbeTargetError(f"unsupported probe target: {target!r}")


def extract_labels(
    records: list[Mapping[str, Any]], *, target: str
) -> LabelExtraction:
    """Vectorize ``label_record`` over a list of records."""

    labels: list[object | None] = []
    distribution: dict[str, int] = {}
    applicable = 0
    for record in records:
        value = label_record(record, target)
        labels.append(value)
        if value is None:
            continue
        applicable += 1
        key = str(value)
        distribution[key] = distribution.get(key, 0) + 1
    return LabelExtraction(
        target=target,
        labels=tuple(labels),
        applicable_count=applicable,
        class_distribution=dict(sorted(distribution.items())),
    )


# --- internal extractors ------------------------------------------------


def _label_output_type(record: Mapping[str, Any]) -> str | None:
    value = record.get("output_type")
    if not isinstance(value, str) or not value:
        return None
    return value


def _label_will_raise(record: Mapping[str, Any]) -> bool | None:
    # Execution-pack records mark a raising run with execution_status="raised"
    # and output_type="exception" (output_kind stays "value"); the previous
    # check on output_kind=="exception" never matched, so this target always
    # collapsed to a single class. (RFC-0015 WS-B4.)
    status = record.get("execution_status")
    if status == "raised" or record.get("output_type") == "exception":
        return True
    output_kind = record.get("output_kind")
    if status == "ok" or output_kind in {"value", "stdout"}:
        return False
    return None


def _label_output_magnitude(record: Mapping[str, Any]) -> str | None:
    # Prefer the privacy-safe bucket precomputed at build time (RFC-0015 WS-B4);
    # fall back to parsing a raw output_repr when present (legacy/local packs).
    precomputed = record.get("output_magnitude_bucket")
    if isinstance(precomputed, str) and precomputed:
        return precomputed
    output_type = record.get("output_type")
    if output_type not in {"int", "float"}:
        return None
    raw = record.get("output_repr")
    if not isinstance(raw, str):
        return None
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value < 0:
        return "negative"
    if value == 0:
        return "zero"
    if value <= 10:
        return "small"
    if value <= 1000:
        return "medium"
    return "large"


def _label_output_length(record: Mapping[str, Any]) -> str | None:
    precomputed = record.get("output_length_bucket")
    if isinstance(precomputed, str) and precomputed:
        return precomputed
    output_type = record.get("output_type")
    if output_type not in {"str", "list", "tuple", "dict", "set", "bytes"}:
        return None
    raw = record.get("output_repr")
    if not isinstance(raw, str):
        return None
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    try:
        n = len(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if n == 0:
        return "empty"
    if n <= 5:
        return "short"
    if n <= 50:
        return "medium"
    if n <= 1000:
        return "long"
    return "huge"


def _label_coarse_kind(record: Mapping[str, Any]) -> str | None:
    output_type = record.get("output_type")
    output_kind = record.get("output_kind")
    if output_kind == "exception" or output_type in {None, "none"}:
        return None
    if output_type in {"int", "float", "bool"}:
        return "arithmetic"
    if output_type in {"str", "bytes"}:
        return "string"
    if output_type in {"list", "tuple", "dict", "set"}:
        return "collection"
    return None


def _label_judge_verdict(record: Mapping[str, Any]) -> str | None:
    verdict = record.get("judge_verdict")
    if not isinstance(verdict, str):
        return None
    normalized = verdict.strip().lower()
    if normalized in {"accepted"}:
        return "accepted"
    if normalized in {"wrong_answer"}:
        return "wrong_answer"
    if normalized in {"runtime_error"}:
        return "runtime_error"
    return None


def _label_passed(record: Mapping[str, Any]) -> bool | None:
    passed = record.get("passed")
    if isinstance(passed, bool):
        return passed
    return None
