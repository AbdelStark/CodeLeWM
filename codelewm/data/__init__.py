"""Data source contracts for CodeLeWM."""

from __future__ import annotations

from .sources import (
    AdapterKind,
    CommitPackFTSourceAdapter,
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
from .filters import (
    DropReason,
    DroppedRecord,
    FilteredRecords,
    FilterPolicy,
    FilterReport,
    evaluate_raw_edit_record,
    filter_raw_edit_records,
)

__all__ = [
    "AdapterKind",
    "CommitPackFTSourceAdapter",
    "DropReason",
    "DroppedRecord",
    "FixtureSourceAdapter",
    "FilteredRecords",
    "FilterPolicy",
    "FilterReport",
    "RawEditRecord",
    "SourceAdapter",
    "SourceKind",
    "SourceRecordError",
    "SourceSpec",
    "SourceUnavailableError",
    "evaluate_raw_edit_record",
    "filter_raw_edit_records",
    "get_source_adapter",
    "load_source",
]
