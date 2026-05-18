"""Source adapter interfaces for raw edit records."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


SourceKind = Literal["commitpackft", "commitpack", "agentpack", "synthetic", "local_repo"]
AdapterKind = SourceKind | Literal["fixture"]
RAW_SOURCE_KINDS = {"commitpackft", "commitpack", "agentpack", "synthetic", "local_repo"}


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
        if source not in RAW_SOURCE_KINDS:
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
        if default_source not in RAW_SOURCE_KINDS:
            raise SourceUnavailableError(f"Unsupported fixture record_source: {default_source}")

        for row in _iter_json_rows(spec.path):
            yield RawEditRecord.from_mapping(row, default_source=default_source)


class CommitPackFTSourceAdapter:
    """Streaming adapter for local CommitPackFT-compatible JSONL shards."""

    source: AdapterKind = "commitpackft"

    def iter_records(self, spec: SourceSpec) -> Iterator[RawEditRecord]:
        if spec.path is None:
            raise SourceUnavailableError("CommitPackFT source requires a path")

        for shard_path in _iter_commitpackft_shards(spec.path):
            for line_number, row in _iter_jsonl_objects(shard_path):
                yield _commitpackft_row_to_raw_record(
                    row,
                    shard_path=shard_path,
                    line_number=line_number,
                    source_name=spec.name,
                    expected_language=spec.options.get("language", "Python"),
                )


def get_source_adapter(source: AdapterKind) -> SourceAdapter:
    if source == "fixture":
        return FixtureSourceAdapter()
    if source == "commitpackft":
        return CommitPackFTSourceAdapter()
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


def _iter_commitpackft_shards(path: Path) -> Iterator[Path]:
    if not path.exists():
        raise SourceUnavailableError(f"CommitPackFT source does not exist: {path}")

    if path.is_file():
        if not _is_jsonl_path(path):
            raise SourceUnavailableError(
                f"CommitPackFT source must be a .jsonl or .jsonl.gz file: {path}"
            )
        yield path
        return

    if not path.is_dir():
        raise SourceUnavailableError(f"CommitPackFT source is not a file or directory: {path}")

    shards = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and _is_jsonl_path(candidate)
    )
    if not shards:
        raise SourceUnavailableError(
            f"CommitPackFT source directory contains no .jsonl or .jsonl.gz shards: {path}"
        )
    yield from shards


def _is_jsonl_path(path: Path) -> bool:
    return path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")


def _iter_jsonl_objects(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SourceRecordError(
                    f"CommitPackFT row {path}:{line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise SourceRecordError(f"CommitPackFT row {path}:{line_number} must be an object")
            yield line_number, row


def _commitpackft_row_to_raw_record(
    row: Mapping[str, Any],
    *,
    shard_path: Path,
    line_number: int,
    source_name: str | None,
    expected_language: str | None,
) -> RawEditRecord:
    row_location = f"{shard_path}:{line_number}"
    language = _require_text(row, ("lang", "language"), row_location)
    if expected_language is not None and language.casefold() != str(expected_language).casefold():
        raise SourceRecordError(
            f"CommitPackFT row {row_location} has language {language!r}, expected {expected_language!r}"
        )

    repo = _normalize_repo(_require_value(row, ("repos", "repo", "repository", "repo_name"), row_location))
    commit = _require_text(row, ("commit", "commit_hash", "sha"), row_location).strip()
    path_before = _normalize_path(_require_text(row, ("old_file", "old_path", "path_before"), row_location))
    path_after = _normalize_path(_require_text(row, ("new_file", "new_path", "path_after"), row_location))
    before = _require_text(row, ("old_contents", "old_content", "before"), row_location, strip=False)
    after = _require_text(row, ("new_contents", "new_content", "after"), row_location, strip=False)
    message = _require_text(row, ("message", "subject"), row_location).strip()

    license_value = _optional_text(row, ("license", "repo_license"))
    subject = _optional_text(row, ("subject",))
    timestamp = _optional_text(row, ("timestamp", "date", "committed_at"))
    metadata = {
        "adapter": "commitpackft",
        "language": language,
        "shard": str(shard_path),
        "line_number": line_number,
    }
    if source_name:
        metadata["source_name"] = source_name
    if subject is not None:
        metadata["subject"] = subject.strip()

    return RawEditRecord(
        source="commitpackft",
        repo=repo,
        commit=commit,
        path_before=path_before,
        path_after=path_after,
        before=before,
        after=after,
        message=message,
        license=None if license_value is None else license_value.strip().lower(),
        timestamp=None if timestamp is None else timestamp.strip(),
        metadata=metadata,
    )


def _require_value(row: Mapping[str, Any], fields: tuple[str, ...], row_location: str) -> Any:
    for field_name in fields:
        if field_name in row and row[field_name] is not None:
            return row[field_name]
    raise SourceRecordError(
        f"CommitPackFT row {row_location} is missing required field: {fields[0]}"
    )


def _require_text(
    row: Mapping[str, Any],
    fields: tuple[str, ...],
    row_location: str,
    *,
    strip: bool = True,
) -> str:
    value = _require_value(row, fields, row_location)
    if not isinstance(value, str):
        raise SourceRecordError(
            f"CommitPackFT row {row_location} field {fields[0]} must be a string"
        )
    return value.strip() if strip else value


def _optional_text(row: Mapping[str, Any], fields: tuple[str, ...]) -> str | None:
    for field_name in fields:
        value = row.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise SourceRecordError(f"CommitPackFT optional field {field_name} must be a string")
        return value
    return None


def _normalize_repo(value: Any) -> str:
    if isinstance(value, str):
        raw_repos = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw_repos = [str(repo) for repo in value]
    else:
        raw_repos = [str(value)]
    repos = [repo.strip() for repo in raw_repos if repo.strip()]
    if not repos:
        raise SourceRecordError("CommitPackFT row has an empty repository field")
    return ",".join(repos)


def _normalize_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if not path:
        raise SourceRecordError("CommitPackFT row has an empty file path")
    return path
