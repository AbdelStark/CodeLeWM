"""Source adapter interfaces for raw edit records."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


SourceKind = Literal["commitpackft", "commitpack", "agentpack", "synthetic", "local_repo"]
AdapterKind = SourceKind | Literal["fixture"]


class SourceUnavailableError(RuntimeError):
    """Raised when a configured source cannot be opened or resolved."""


class SourceRecordError(ValueError):
    """Raised when a source row does not satisfy the raw edit record schema."""


@dataclass(frozen=True)
class SourceSpec:
    """Configuration for one raw edit source."""

    source: AdapterKind
    path: Path | None = None
    name: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path is not None and not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class RawEditRecord:
    """Raw before/action/after edit record emitted by source adapters."""

    source: SourceKind
    repo: str
    commit: str
    path_before: str
    path_after: str
    before: str
    after: str
    message: str
    license: str | None = None
    timestamp: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, default_source: SourceKind | None = None) -> "RawEditRecord":
        source = row.get("source", default_source)
        if source is None:
            raise SourceRecordError("Raw edit row is missing required field: source")
        if source not in {"commitpackft", "commitpack", "agentpack", "synthetic", "local_repo"}:
            raise SourceRecordError(f"Unsupported raw edit source: {source}")

        required = (
            "repo",
            "commit",
            "path_before",
            "path_after",
            "before",
            "after",
            "message",
        )
        missing = [field_name for field_name in required if field_name not in row]
        if missing:
            raise SourceRecordError(
                "Raw edit row is missing required field(s): " + ", ".join(missing)
            )

        metadata = row.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise SourceRecordError("Raw edit row metadata must be a mapping")

        return cls(
            source=source,
            repo=str(row["repo"]),
            commit=str(row["commit"]),
            path_before=str(row["path_before"]),
            path_after=str(row["path_after"]),
            before=str(row["before"]),
            after=str(row["after"]),
            message=str(row["message"]),
            license=None if row.get("license") is None else str(row.get("license")),
            timestamp=None if row.get("timestamp") is None else str(row.get("timestamp")),
            metadata=dict(metadata),
        )


class SourceAdapter(Protocol):
    """Protocol implemented by raw edit source adapters."""

    source: AdapterKind

    def iter_records(self, spec: SourceSpec) -> Iterator[RawEditRecord]:
        ...


class FixtureSourceAdapter:
    """JSON/JSONL adapter used for deterministic tests and local smoke fixtures."""

    source: AdapterKind = "fixture"

    def iter_records(self, spec: SourceSpec) -> Iterator[RawEditRecord]:
        if spec.path is None:
            raise SourceUnavailableError("Fixture source requires a path")
        if not spec.path.exists():
            raise SourceUnavailableError(f"Fixture source does not exist: {spec.path}")
        if not spec.path.is_file():
            raise SourceUnavailableError(f"Fixture source is not a file: {spec.path}")

        default_source = spec.options.get("record_source", "local_repo")
        if default_source not in {"commitpackft", "commitpack", "agentpack", "synthetic", "local_repo"}:
            raise SourceUnavailableError(f"Unsupported fixture record_source: {default_source}")

        for row in _iter_json_rows(spec.path):
            yield RawEditRecord.from_mapping(row, default_source=default_source)


def get_source_adapter(source: AdapterKind) -> SourceAdapter:
    if source == "fixture":
        return FixtureSourceAdapter()
    raise SourceUnavailableError(f"No source adapter is registered for: {source}")


def load_source(spec: SourceSpec) -> Iterator[RawEditRecord]:
    """Load raw edit records from a configured source."""

    return get_source_adapter(spec.source).iter_records(spec)


def _iter_json_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise SourceRecordError("Fixture JSON source must contain a list of records")
        for row in data:
            if not isinstance(row, Mapping):
                raise SourceRecordError("Fixture JSON row must be an object")
            yield row
        return

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, Mapping):
                raise SourceRecordError(f"Fixture JSONL row {line_number} must be an object")
            yield row
