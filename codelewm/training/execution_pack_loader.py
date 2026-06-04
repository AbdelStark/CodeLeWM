"""Batch loader for the execution-substrate pack.

Reads the ``pack.jsonl`` produced by
:func:`codelewm.data.execution_pack.build_execution_pack`, pads or
truncates each tokenized field to the configured sequence length, and
yields batches as NumPy arrays. The torch training executor (#265)
wraps this loader to expose ``torch.Tensor`` batches.

This loader is deliberately torch-free so the smoke runner (#264) and
unit tests can exercise the pack-to-batch path on the lightweight test
environment without installing the train extras.

Schema notes:

- The pack records carry variable-length token lists. The loader pads
  with ``pad_token_id`` (default 0) and writes a parallel boolean
  ``attention_mask``.
- ``code_tokens`` is padded to ``code_sequence_length``.
- ``input_tokens`` is padded to ``action_sequence_length``.
- ``output_tokens`` is padded to ``output_sequence_length``.
- Records longer than the cap are truncated and a per-batch counter is
  emitted as a diagnostic.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


EXECUTION_PACK_BATCH_SCHEMA_VERSION = "codelewm.execution_pack_batch.v2"


@dataclass(frozen=True)
class ExecutionPackBatch:
    """One batch of tokenized execution-substrate records.

    All arrays have dtype ``int32`` for token ids and ``bool`` for masks,
    matching the conventions of the existing CommitPackFT loader.
    """

    schema_version: str
    code_tokens: np.ndarray  # (B, code_sequence_length) int32
    code_attention_mask: np.ndarray  # (B, code_sequence_length) bool
    input_tokens: np.ndarray  # (B, action_sequence_length) int32
    input_attention_mask: np.ndarray  # (B, action_sequence_length) bool
    output_tokens: np.ndarray  # (B, output_sequence_length) int32
    output_attention_mask: np.ndarray  # (B, output_sequence_length) bool
    output_type_index: np.ndarray  # (B,) int32 — encoded OUTPUT_TYPE_VOCAB index
    will_raise: np.ndarray  # (B,) bool — derived from output_kind == "exception"
    passed: np.ndarray | None  # (B,) bool when v0.8 pass/fail labels are present
    record_ids: tuple[str, ...]
    splits: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.code_tokens.shape[0])

    def shape_summary(self) -> dict[str, tuple[int, ...]]:
        return {
            "code_tokens": tuple(self.code_tokens.shape),
            "input_tokens": tuple(self.input_tokens.shape),
            "output_tokens": tuple(self.output_tokens.shape),
            "output_type_index": tuple(self.output_type_index.shape),
            "will_raise": tuple(self.will_raise.shape),
            "passed": () if self.passed is None else tuple(self.passed.shape),
        }


# Stable vocabulary for output_type; the index is used by the
# `output_type` probe target (#266).
OUTPUT_TYPE_VOCAB: tuple[str, ...] = (
    "none",
    "bool",
    "int",
    "float",
    "str",
    "bytes",
    "tuple",
    "list",
    "dict",
    "set",
    "exception",
    "other",
)
_OUTPUT_TYPE_INDEX: dict[str, int] = {kind: i for i, kind in enumerate(OUTPUT_TYPE_VOCAB)}


@dataclass(frozen=True)
class ExecutionPackLoaderConfig:
    pack_jsonl: Path
    code_sequence_length: int = 1024
    action_sequence_length: int = 256
    output_sequence_length: int = 256
    batch_size: int = 8
    pad_token_id: int = 0
    split: str | None = None  # restrict to one split when set
    drop_last_partial: bool = False
    shuffle: bool = False
    shuffle_seed: int = 0


@dataclass
class LoaderDiagnostics:
    """Running tally of pack statistics seen by the loader."""

    record_count: int = 0
    truncated_code: int = 0
    truncated_input: int = 0
    truncated_output: int = 0
    output_type_histogram: dict[str, int] = field(default_factory=dict)
    output_kind_histogram: dict[str, int] = field(default_factory=dict)
    execution_status_histogram: dict[str, int] = field(default_factory=dict)
    split_histogram: dict[str, int] = field(default_factory=dict)
    pass_label_histogram: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "truncated_code": self.truncated_code,
            "truncated_input": self.truncated_input,
            "truncated_output": self.truncated_output,
            "output_type_histogram": dict(sorted(self.output_type_histogram.items())),
            "output_kind_histogram": dict(sorted(self.output_kind_histogram.items())),
            "execution_status_histogram": dict(
                sorted(self.execution_status_histogram.items())
            ),
            "split_histogram": dict(sorted(self.split_histogram.items())),
            "pass_label_histogram": dict(sorted(self.pass_label_histogram.items())),
        }


def iter_records(config: ExecutionPackLoaderConfig) -> Iterator[dict[str, Any]]:
    """Yield filtered records from ``pack.jsonl``."""

    if not config.pack_jsonl.is_file():
        raise FileNotFoundError(f"pack.jsonl not found: {config.pack_jsonl}")
    with config.pack_jsonl.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{config.pack_jsonl}:{line_no}: invalid JSON"
                ) from exc
            if config.split is not None and row.get("split") != config.split:
                continue
            yield row


def iter_batches(
    config: ExecutionPackLoaderConfig,
    *,
    diagnostics: LoaderDiagnostics | None = None,
) -> Iterator[ExecutionPackBatch]:
    """Yield :class:`ExecutionPackBatch` instances from the pack JSONL."""

    diagnostics = diagnostics if diagnostics is not None else LoaderDiagnostics()
    rng = np.random.default_rng(config.shuffle_seed) if config.shuffle else None
    records = list(iter_records(config))
    if rng is not None:
        rng.shuffle(records)

    batch_size = max(1, int(config.batch_size))
    buffer: list[dict[str, Any]] = []
    for record in records:
        buffer.append(record)
        if len(buffer) == batch_size:
            yield _to_batch(buffer, config, diagnostics)
            buffer = []
    if buffer and not config.drop_last_partial:
        yield _to_batch(buffer, config, diagnostics)


def collect_diagnostics(config: ExecutionPackLoaderConfig) -> LoaderDiagnostics:
    """Stream the pack once and return a :class:`LoaderDiagnostics` summary."""

    diagnostics = LoaderDiagnostics()
    for _ in iter_batches(config, diagnostics=diagnostics):
        pass
    return diagnostics


def _to_batch(
    rows: list[dict[str, Any]],
    config: ExecutionPackLoaderConfig,
    diagnostics: LoaderDiagnostics,
) -> ExecutionPackBatch:
    batch_size = len(rows)
    code = np.full(
        (batch_size, config.code_sequence_length), config.pad_token_id, dtype=np.int32
    )
    code_mask = np.zeros((batch_size, config.code_sequence_length), dtype=bool)
    inputs = np.full(
        (batch_size, config.action_sequence_length),
        config.pad_token_id,
        dtype=np.int32,
    )
    input_mask = np.zeros((batch_size, config.action_sequence_length), dtype=bool)
    outputs = np.full(
        (batch_size, config.output_sequence_length),
        config.pad_token_id,
        dtype=np.int32,
    )
    output_mask = np.zeros((batch_size, config.output_sequence_length), dtype=bool)
    output_type_index = np.zeros((batch_size,), dtype=np.int32)
    will_raise = np.zeros((batch_size,), dtype=bool)
    passed = np.zeros((batch_size,), dtype=bool)
    pass_label_seen = False
    pass_label_missing = False
    record_ids: list[str] = []
    splits: list[str] = []

    for i, row in enumerate(rows):
        code_tokens = list(row.get("code_tokens") or [])
        input_tokens = list(row.get("input_tokens") or [])
        output_tokens = list(row.get("output_tokens") or [])
        if len(code_tokens) > config.code_sequence_length:
            diagnostics.truncated_code += 1
            code_tokens = code_tokens[: config.code_sequence_length]
        if len(input_tokens) > config.action_sequence_length:
            diagnostics.truncated_input += 1
            input_tokens = input_tokens[: config.action_sequence_length]
        if len(output_tokens) > config.output_sequence_length:
            diagnostics.truncated_output += 1
            output_tokens = output_tokens[: config.output_sequence_length]
        code[i, : len(code_tokens)] = code_tokens
        code_mask[i, : len(code_tokens)] = True
        inputs[i, : len(input_tokens)] = input_tokens
        input_mask[i, : len(input_tokens)] = True
        outputs[i, : len(output_tokens)] = output_tokens
        output_mask[i, : len(output_tokens)] = True
        output_type = str(row.get("output_type") or "none")
        output_type_index[i] = _OUTPUT_TYPE_INDEX.get(
            output_type, _OUTPUT_TYPE_INDEX["other"]
        )
        will_raise[i] = str(row.get("output_kind") or "") == "exception"
        pass_value = row.get("passed")
        if pass_value is None:
            pass_label_missing = True
        elif isinstance(pass_value, bool):
            passed[i] = pass_value
            pass_label_seen = True
            diagnostics.pass_label_histogram[str(pass_value).lower()] = (
                diagnostics.pass_label_histogram.get(str(pass_value).lower(), 0) + 1
            )
        else:
            raise ValueError(
                f"record {row.get('record_id') or i}: passed must be bool or null"
            )
        record_ids.append(str(row.get("record_id") or ""))
        splits.append(str(row.get("split") or "train"))

        diagnostics.record_count += 1
        diagnostics.output_type_histogram[output_type] = (
            diagnostics.output_type_histogram.get(output_type, 0) + 1
        )
        kind = str(row.get("output_kind") or "")
        diagnostics.output_kind_histogram[kind] = (
            diagnostics.output_kind_histogram.get(kind, 0) + 1
        )
        status = str(row.get("execution_status") or "")
        diagnostics.execution_status_histogram[status] = (
            diagnostics.execution_status_histogram.get(status, 0) + 1
        )
        split = str(row.get("split") or "")
        diagnostics.split_histogram[split] = (
            diagnostics.split_histogram.get(split, 0) + 1
        )

    if pass_label_seen and pass_label_missing:
        raise ValueError(
            "execution-pack batch mixes rows with and without passed labels"
        )

    return ExecutionPackBatch(
        schema_version=EXECUTION_PACK_BATCH_SCHEMA_VERSION,
        code_tokens=code,
        code_attention_mask=code_mask,
        input_tokens=inputs,
        input_attention_mask=input_mask,
        output_tokens=outputs,
        output_attention_mask=output_mask,
        output_type_index=output_type_index,
        will_raise=will_raise,
        passed=passed if pass_label_seen else None,
        record_ids=tuple(record_ids),
        splits=tuple(splits),
    )
