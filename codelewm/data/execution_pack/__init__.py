"""Execution-substrate pack builder.

Consumes a normalized ingestion JSONL (output of
:mod:`codelewm.data.execution_sources`), runs each
``(code, input)`` pair through :func:`codelewm.data.sandbox.run_one`,
filters records that fail the determinism / policy / timeout gates,
tokenizes the code, input ``repr``, and output ``repr``, partitions the
surviving records by ``source_problem_id``, and writes:

- ``pack.jsonl`` — one :class:`PackedExecutionRecord` per line, with the
  tokenized code/input/output and the per-record audit-trail metadata.
- ``manifest.json`` — :data:`EXECUTION_PACK_MANIFEST_SCHEMA_VERSION` summary
  with split counts, license breakdown, sandbox-reject counts, claim
  boundary fingerprint, and parent ingestion-artifact references.
- ``attribution.json`` — per-source attribution URLs.
- ``sandbox_audit_summary.json`` — aggregated reject reasons per source.

The HDF5 mirror is intentionally deferred. The training executor and
the eval harness can consume the JSONL directly for the substrate
pivot's first run; the HDF5 layout will follow once the JSONL gate
results are in.

This package never imports :mod:`codelewm.data.sandbox` at training,
scoring, or evaluation time. The structural import guard in
``tests/security/test_sandbox_import_boundary.py`` continues to hold.
"""

from __future__ import annotations

from .builder import (
    ExecutionPackBuilderError,
    ExecutionPackResult,
    build_execution_pack,
)
from .manifest import (
    EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
    EXECUTION_PACK_RECORD_SCHEMA_VERSION,
    ExecutionPackManifest,
)
from .record import (
    PackedExecutionRecord,
    SplitName,
    classify_record_kind,
)


__all__ = [
    "EXECUTION_PACK_MANIFEST_SCHEMA_VERSION",
    "EXECUTION_PACK_RECORD_SCHEMA_VERSION",
    "ExecutionPackBuilderError",
    "ExecutionPackManifest",
    "ExecutionPackResult",
    "PackedExecutionRecord",
    "SplitName",
    "build_execution_pack",
    "classify_record_kind",
]
