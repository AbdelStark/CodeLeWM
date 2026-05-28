"""Normalized records emitted by the execution-substrate source adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal


EXECUTION_SOURCE_DATASETS = (
    "codenet",
    "mbpp",
    "mbpp_plus",
    "apps",
    "humaneval",
)
ExecutionSourceDataset = Literal["codenet", "mbpp", "mbpp_plus", "apps", "humaneval"]
InputKind = Literal["stdin", "argv", "function_call"]


@dataclass(frozen=True)
class InputCase:
    """One input case for a submission.

    ``input_repr`` is the canonical serialization of the argument. For
    ``function_call`` cases it is a JSON-encoded list (positional args),
    dict (kwargs), or scalar value. For ``stdin`` cases it is the raw
    text fed to the program's standard input. For ``argv`` cases it is
    a JSON-encoded list of strings used as additional argv after the
    program name.
    """

    input_id: str
    input_repr: str
    input_kind: InputKind
    function_name: str | None = None

    def __post_init__(self) -> None:
        if not self.input_id:
            raise ValueError("input_id must be non-empty")
        if self.input_kind not in {"stdin", "argv", "function_call"}:
            raise ValueError(f"unsupported input_kind: {self.input_kind!r}")
        if self.input_kind == "function_call" and not self.function_name:
            raise ValueError("function_call input requires function_name")

    def as_dict(self) -> dict[str, object]:
        return {
            "input_id": self.input_id,
            "input_repr": self.input_repr,
            "input_kind": self.input_kind,
            "function_name": self.function_name,
        }


@dataclass(frozen=True)
class SourceSubmission:
    """One ``(code, inputs, expected_outputs, license)`` record.

    The record is the unit consumed by the sandbox runner and the pack
    builder. Adapters normalize upstream rows into this shape; the rest
    of the pipeline never sees the dataset-specific schema.
    """

    source_dataset: ExecutionSourceDataset
    source_problem_id: str
    source_submission_id: str
    code: str
    inputs: tuple[InputCase, ...]
    expected_outputs: tuple[str, ...] | None
    judge_verdict: str | None
    license: str
    license_attribution_url: str
    held_out_for_eval: bool = False
    raw_hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.source_dataset not in EXECUTION_SOURCE_DATASETS:
            raise ValueError(
                f"unsupported source_dataset: {self.source_dataset!r}"
            )
        if not self.source_problem_id:
            raise ValueError("source_problem_id must be non-empty")
        if not self.source_submission_id:
            raise ValueError("source_submission_id must be non-empty")
        if not self.code:
            raise ValueError("code must be non-empty")
        if not self.inputs:
            raise ValueError("inputs must contain at least one InputCase")
        if not self.license:
            raise ValueError("license must be non-empty")
        if (
            self.expected_outputs is not None
            and len(self.expected_outputs) != len(self.inputs)
        ):
            raise ValueError(
                "expected_outputs, when provided, must align with inputs by index"
            )
        if not self.raw_hash:
            object.__setattr__(self, "raw_hash", self._compute_raw_hash())

    def _compute_raw_hash(self) -> str:
        payload = {
            "source_dataset": self.source_dataset,
            "source_problem_id": self.source_problem_id,
            "source_submission_id": self.source_submission_id,
            "code": self.code,
            "inputs": [c.as_dict() for c in self.inputs],
            "expected_outputs": list(self.expected_outputs)
            if self.expected_outputs is not None
            else None,
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "source_dataset": self.source_dataset,
            "source_problem_id": self.source_problem_id,
            "source_submission_id": self.source_submission_id,
            "code": self.code,
            "inputs": [c.as_dict() for c in self.inputs],
            "expected_outputs": list(self.expected_outputs)
            if self.expected_outputs is not None
            else None,
            "judge_verdict": self.judge_verdict,
            "license": self.license,
            "license_attribution_url": self.license_attribution_url,
            "held_out_for_eval": self.held_out_for_eval,
            "raw_hash": self.raw_hash,
        }
