"""Source adapters for the execution-substrate data pipeline.

The execution-substrate world model (RFC-0014, tracker #259) needs a uniform
intermediate representation across CodeNet, MBPP, MBPP-Plus, APPS, and
HumanEval. Each upstream dataset has its own layout, license terms, and
input-case shape. The adapters in this package normalize them into a
single :class:`SourceSubmission` record so the sandbox (#260) and pack
builder (#262) can consume one schema.

Adapters parse **already-downloaded** upstream files. Network fetch is
intentionally out of scope: the actual HF Hub download flow lives in
``scripts/`` and is exercised by the data builder. Keeping the adapters
file-driven keeps tests offline and reproducible.

The public surface is:

- :class:`SourceSubmission` — one normalized record.
- :class:`InputCase` — one ``(input_id, input_repr, input_kind, function_name)``.
- :class:`ExecutionSourceAdapter` — the protocol every adapter satisfies.
- :func:`get_execution_source_adapter` — look an adapter up by name.
- :func:`load_execution_source` — top-level entry that writes JSONL.

Five adapter implementations are registered at import time: ``codenet``,
``mbpp``, ``mbpp_plus``, ``apps``, ``humaneval``. The latter two are
flagged ``held_out_for_eval=True`` so the pack builder refuses to put
them in train/val splits.
"""

from __future__ import annotations

from .apps import APPSSourceAdapter
from .base import (
    EXECUTION_SOURCE_RECORD_SCHEMA_VERSION,
    ExecutionSourceAdapter,
    ExecutionSourceError,
    ExecutionSourceKind,
    get_execution_source_adapter,
    load_execution_source,
    register_execution_source_adapter,
)
from .codenet import CodeNetSourceAdapter
from .humaneval import HumanEvalSourceAdapter
from .mbpp import MBPPSourceAdapter
from .mbpp_plus import MBPPPlusSourceAdapter
from .record import EXECUTION_SOURCE_DATASETS, InputCase, SourceSubmission

# Register concrete adapters with the lookup table.
register_execution_source_adapter("codenet", CodeNetSourceAdapter())
register_execution_source_adapter("mbpp", MBPPSourceAdapter())
register_execution_source_adapter("mbpp_plus", MBPPPlusSourceAdapter())
register_execution_source_adapter("apps", APPSSourceAdapter())
register_execution_source_adapter("humaneval", HumanEvalSourceAdapter())

__all__ = [
    "APPSSourceAdapter",
    "CodeNetSourceAdapter",
    "EXECUTION_SOURCE_DATASETS",
    "EXECUTION_SOURCE_RECORD_SCHEMA_VERSION",
    "ExecutionSourceAdapter",
    "ExecutionSourceError",
    "ExecutionSourceKind",
    "HumanEvalSourceAdapter",
    "InputCase",
    "MBPPPlusSourceAdapter",
    "MBPPSourceAdapter",
    "SourceSubmission",
    "get_execution_source_adapter",
    "load_execution_source",
    "register_execution_source_adapter",
]
