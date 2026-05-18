"""Data source contracts for CodeLeWM."""

from __future__ import annotations

from .sources import (
    AdapterKind,
    FixtureSourceAdapter,
    RawEditRecord,
    SourceAdapter,
    SourceKind,
    SourceRecordError,
    SourceSpec,
    SourceUnavailableError,
    get_source_adapter,
    load_source,
)

__all__ = [
    "AdapterKind",
    "FixtureSourceAdapter",
    "RawEditRecord",
    "SourceAdapter",
    "SourceKind",
    "SourceRecordError",
    "SourceSpec",
    "SourceUnavailableError",
    "get_source_adapter",
    "load_source",
]
