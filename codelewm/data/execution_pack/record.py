"""Packed record schema for the execution substrate."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal


SplitName = Literal["train", "val", "test"]
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class PackedExecutionRecord:
    """One tokenized ``(code, input, output)`` triple.

    All fields are JSON-serializable. The token lists are variable-length;
    the consuming training executor pads or truncates to the configured
    sequence length at load time.
    """

    source_dataset: str
    source_problem_id: str
    source_submission_id: str
    input_id: str
    split: SplitName

    code_tokens: tuple[int, ...]
    code_checksum: str

    input_tokens: tuple[int, ...]
    input_repr_checksum: str
    input_kind: str
    function_name: str | None

    output_tokens: tuple[int, ...]
    output_repr_checksum: str
    output_kind: str
    output_type: str
    execution_status: str

    judge_verdict: str | None
    wall_time_ms: float
    peak_rss_kb: int
    determinism_check: bool

    license: str
    license_attribution_url: str
    held_out_for_eval: bool

    @property
    def record_id(self) -> str:
        return f"{self.source_problem_id}::{self.source_submission_id}::{self.input_id}"

    def as_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source_dataset": self.source_dataset,
            "source_problem_id": self.source_problem_id,
            "source_submission_id": self.source_submission_id,
            "input_id": self.input_id,
            "split": self.split,
            "code_tokens": list(self.code_tokens),
            "code_checksum": self.code_checksum,
            "input_tokens": list(self.input_tokens),
            "input_repr_checksum": self.input_repr_checksum,
            "input_kind": self.input_kind,
            "function_name": self.function_name,
            "output_tokens": list(self.output_tokens),
            "output_repr_checksum": self.output_repr_checksum,
            "output_kind": self.output_kind,
            "output_type": self.output_type,
            "execution_status": self.execution_status,
            "judge_verdict": self.judge_verdict,
            "wall_time_ms": self.wall_time_ms,
            "peak_rss_kb": self.peak_rss_kb,
            "determinism_check": self.determinism_check,
            "license": self.license,
            "license_attribution_url": self.license_attribution_url,
            "held_out_for_eval": self.held_out_for_eval,
        }


def classify_record_kind(record: PackedExecutionRecord) -> str:
    """Coarse category used for output-type balancing.

    Returns ``"value-int"``, ``"value-numeric"``, ``"value-string"``,
    ``"value-collection"``, ``"value-bool"``, ``"value-none"``,
    ``"exception"``, ``"stdout"``, or ``"other"``.
    """

    if record.output_kind == "exception":
        return "exception"
    if record.output_kind == "stdout":
        return "stdout"
    if record.output_type == "int":
        return "value-int"
    if record.output_type == "float":
        return "value-numeric"
    if record.output_type == "str":
        return "value-string"
    if record.output_type in {"list", "tuple", "dict", "set", "bytes"}:
        return "value-collection"
    if record.output_type == "bool":
        return "value-bool"
    if record.output_type == "none":
        return "value-none"
    return "other"


def tokenize_text(text: str) -> tuple[int, ...]:
    """Whitespace + punctuation tokenizer matching the existing pipeline.

    Each token is hashed via blake2b into a non-negative 31-bit int. The
    hash is stable across runs and Python versions because ``blake2b``
    has a deterministic, documented output for a given seed.
    """

    return tuple(_stable_token_id(t) for t in _TOKEN_PATTERN.findall(text))


def _stable_token_id(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
