"""Local transition index for nearest-historical-edit lookups.

The index stores latent embeddings of training transitions in a flat
on-disk layout: a `vectors.npy` array (rows are entries), an
`entries.jsonl` file with one record per entry, and an
`index.json` header (schema-versioned). The artifact manifest for the
index uses the shared `codelewm.artifact_manifest.v1` contract from
`codelewm.observability`, so existing manifest tooling (the
`codelewm manifest verify` CLI, the artifact lineage stack) applies
without change.

The index is intentionally local and brute-force. RFC-0008 calls it
"optionally include nearest historical edits"; remote vector
databases are explicitly out of scope. The brute-force search uses
plain NumPy and hits the v0.1 performance target of < 500 ms for
top-100 retrieval at the documented 250k-transition scale.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from codelewm.observability import (
    ArtifactManifest,
    ArtifactManifestError,
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)


TRANSITION_INDEX_SCHEMA_VERSION = "codelewm.transition_index.v1"
_DEFAULT_VECTORS_FILENAME = "vectors.npy"
_DEFAULT_ENTRIES_FILENAME = "entries.jsonl"
_DEFAULT_INDEX_FILENAME = "index.json"
_DEFAULT_MANIFEST_FILENAME = "manifest.json"

IndexDistance = Literal["l2", "cosine"]
_SUPPORTED_DISTANCES: frozenset[str] = frozenset({"l2", "cosine"})


class TransitionIndexError(ValueError):
    """Raised when a transition index is malformed or unverifiable."""


@dataclass(frozen=True)
class TransitionIndexEntry:
    """One record stored in a transition index."""

    transition_id: str
    split: str
    source: str = "unknown"
    repo: str = ""
    path: str = ""
    edit_size: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise TransitionIndexError("entry transition_id must not be empty")
        if not self.split:
            raise TransitionIndexError(
                f"entry {self.transition_id!r} must declare split"
            )
        if self.edit_size < 0:
            raise TransitionIndexError(
                f"entry {self.transition_id!r} edit_size must be non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "split": self.split,
            "source": self.source,
            "repo": self.repo,
            "path": self.path,
            "edit_size": self.edit_size,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionIndexEntry":
        if "transition_id" not in payload:
            raise TransitionIndexError("entry payload must include transition_id")
        if "split" not in payload:
            raise TransitionIndexError(
                f"entry {payload['transition_id']!r} must declare split"
            )
        return cls(
            transition_id=str(payload["transition_id"]),
            split=str(payload["split"]),
            source=str(payload.get("source", "unknown")),
            repo=str(payload.get("repo", "")),
            path=str(payload.get("path", "")),
            edit_size=int(payload.get("edit_size", 0) or 0),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class TransitionIndexSearchHit:
    """One nearest-neighbor result returned by :meth:`TransitionIndex.search`."""

    entry: TransitionIndexEntry
    distance: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "distance": float(self.distance),
            "rank": int(self.rank),
        }


@dataclass(frozen=True)
class TransitionIndex:
    """Validated, schema-versioned local transition index."""

    name: str
    vectors: np.ndarray
    entries: tuple[TransitionIndexEntry, ...]
    distance: IndexDistance = "l2"
    schema_version: str = TRANSITION_INDEX_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != TRANSITION_INDEX_SCHEMA_VERSION:
            raise TransitionIndexError(
                f"schema_version must be {TRANSITION_INDEX_SCHEMA_VERSION!r}"
            )
        if not self.name:
            raise TransitionIndexError("index name must not be empty")
        if self.distance not in _SUPPORTED_DISTANCES:
            allowed = ", ".join(sorted(_SUPPORTED_DISTANCES))
            raise TransitionIndexError(
                f"distance must be one of: {allowed}; got {self.distance!r}"
            )
        if not isinstance(self.vectors, np.ndarray):
            raise TransitionIndexError("vectors must be a numpy.ndarray")
        if self.vectors.ndim != 2:
            raise TransitionIndexError("vectors must be 2-dimensional")
        if self.vectors.dtype != np.float32:
            raise TransitionIndexError("vectors must use dtype float32")
        if not np.isfinite(self.vectors).all():
            raise TransitionIndexError("vectors must be finite")
        if self.vectors.shape[0] != len(self.entries):
            raise TransitionIndexError(
                "vectors row count must equal len(entries); "
                f"got {self.vectors.shape[0]} vs {len(self.entries)}"
            )
        seen: set[str] = set()
        for entry in self.entries:
            if entry.transition_id in seen:
                raise TransitionIndexError(
                    f"duplicate transition_id in index: {entry.transition_id!r}"
                )
            seen.add(entry.transition_id)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0])

    def search(
        self,
        query: np.ndarray,
        *,
        k: int = 10,
        exclude_ids: Iterable[str] = (),
    ) -> tuple[TransitionIndexSearchHit, ...]:
        """Return the ``k`` nearest entries for ``query``.

        ``query`` must be a 1-D float32 array whose length matches the index
        dimension. ``exclude_ids`` lets callers drop a query's own
        transition id from results.
        """

        if not isinstance(query, np.ndarray):
            raise TransitionIndexError("query must be a numpy.ndarray")
        if query.ndim != 1:
            raise TransitionIndexError("query must be 1-dimensional")
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)
        if query.shape[0] != self.dim:
            raise TransitionIndexError(
                f"query dim {query.shape[0]} does not match index dim {self.dim}"
            )
        if k <= 0:
            raise TransitionIndexError("k must be positive")

        distances = self._pairwise_distance(query)
        exclude = set(exclude_ids)
        candidate_indices = np.argsort(distances, kind="stable")
        hits: list[TransitionIndexSearchHit] = []
        for index in candidate_indices:
            entry = self.entries[int(index)]
            if entry.transition_id in exclude:
                continue
            hits.append(
                TransitionIndexSearchHit(
                    entry=entry,
                    distance=float(distances[int(index)]),
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= k:
                break
        return tuple(hits)

    def _pairwise_distance(self, query: np.ndarray) -> np.ndarray:
        if self.distance == "l2":
            diff = self.vectors - query
            return np.sum(diff * diff, axis=1)
        if self.distance == "cosine":
            vectors_norm = np.linalg.norm(self.vectors, axis=1)
            query_norm = float(np.linalg.norm(query))
            if query_norm == 0.0:
                raise TransitionIndexError(
                    "cosine search requires a non-zero query vector"
                )
            similarities = np.zeros(self.vectors.shape[0], dtype=np.float32)
            valid = vectors_norm > 0.0
            similarities[valid] = (
                self.vectors[valid] @ query
            ) / (vectors_norm[valid] * query_norm)
            return 1.0 - similarities
        raise TransitionIndexError(f"unsupported distance: {self.distance!r}")

    def header_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "count": self.count,
            "dim": self.dim,
            "distance": self.distance,
            "vectors_path": _DEFAULT_VECTORS_FILENAME,
            "entries_path": _DEFAULT_ENTRIES_FILENAME,
            "metadata": dict(self.metadata),
        }


def build_transition_index(
    *,
    name: str,
    entries: Sequence[TransitionIndexEntry],
    vectors: np.ndarray,
    distance: IndexDistance = "l2",
    metadata: Mapping[str, Any] | None = None,
) -> TransitionIndex:
    """Construct a validated transition index from in-memory data."""

    vectors_array = np.asarray(vectors, dtype=np.float32)
    return TransitionIndex(
        name=name,
        vectors=vectors_array,
        entries=tuple(entries),
        distance=distance,
        metadata=dict(metadata or {}),
    )


def write_transition_index(
    index: TransitionIndex,
    root: Path | str,
    *,
    command: Sequence[str] = ("codelewm", "index"),
    config: Any | None = None,
    parent_artifacts: Sequence[str] = (),
    source_git_sha: str | None = None,
    created_at: str | None = None,
    artifact_id: str | None = None,
) -> tuple[ArtifactManifest, Path]:
    """Write the index to ``root`` and emit an artifact manifest.

    Returns ``(artifact_manifest, manifest_path)``.
    """

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    vectors_path = root_path / _DEFAULT_VECTORS_FILENAME
    entries_path = root_path / _DEFAULT_ENTRIES_FILENAME
    header_path = root_path / _DEFAULT_INDEX_FILENAME
    manifest_path = root_path / _DEFAULT_MANIFEST_FILENAME

    np.save(vectors_path, index.vectors, allow_pickle=False)
    with entries_path.open("w", encoding="utf-8") as handle:
        for entry in index.entries:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    header_path.write_text(
        json.dumps(index.header_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = build_artifact_manifest(
        artifact_kind="index",
        root=root_path,
        files=(vectors_path, entries_path, header_path),
        command=command,
        config=config if config is not None else index.header_dict(),
        parent_artifacts=parent_artifacts,
        source_git_sha=source_git_sha,
        created_at=created_at,
        artifact_id=artifact_id,
        metadata={
            "schema_version": TRANSITION_INDEX_SCHEMA_VERSION,
            "name": index.name,
            "count": index.count,
            "dim": index.dim,
            "distance": index.distance,
            "index_metadata": dict(index.metadata),
        },
    )
    write_artifact_manifest(manifest, manifest_path)
    return manifest, manifest_path


def read_transition_index(
    root: Path | str,
    *,
    verify_manifest: bool = True,
) -> TransitionIndex:
    """Read a transition index from ``root`` and validate its manifest."""

    root_path = Path(root)
    manifest_path = root_path / _DEFAULT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise TransitionIndexError(
            f"transition index manifest not found: {manifest_path}"
        )

    if verify_manifest:
        try:
            artifact_manifest = read_artifact_manifest(manifest_path)
            validate_artifact_checksums(artifact_manifest, root=root_path)
        except ArtifactManifestError as exc:
            raise TransitionIndexError(
                f"transition index manifest failed validation: {exc}"
            ) from exc

    header_path = root_path / _DEFAULT_INDEX_FILENAME
    if not header_path.is_file():
        raise TransitionIndexError(
            f"transition index header not found: {header_path}"
        )
    header = json.loads(header_path.read_text(encoding="utf-8"))
    if not isinstance(header, Mapping):
        raise TransitionIndexError("transition index header must be a JSON object")
    if header.get("schema_version") != TRANSITION_INDEX_SCHEMA_VERSION:
        raise TransitionIndexError(
            f"unsupported index schema_version: {header.get('schema_version')!r}"
        )

    vectors_path = root_path / header.get("vectors_path", _DEFAULT_VECTORS_FILENAME)
    entries_path = root_path / header.get("entries_path", _DEFAULT_ENTRIES_FILENAME)
    vectors = np.load(vectors_path, allow_pickle=False)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    entries: list[TransitionIndexEntry] = []
    with entries_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            entries.append(TransitionIndexEntry.from_dict(json.loads(line)))

    return TransitionIndex(
        name=str(header["name"]),
        vectors=vectors,
        entries=tuple(entries),
        distance=str(header.get("distance", "l2")),  # type: ignore[arg-type]
        metadata=dict(header.get("metadata", {})),
    )


def transition_index_header_json_schema() -> dict[str, Any]:
    """Return the JSON schema for the index header (`index.json`)."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": TRANSITION_INDEX_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "name",
            "count",
            "dim",
            "distance",
            "vectors_path",
            "entries_path",
            "metadata",
        ],
        "properties": {
            "schema_version": {"const": TRANSITION_INDEX_SCHEMA_VERSION},
            "name": {"type": "string", "minLength": 1},
            "count": {"type": "integer", "minimum": 0},
            "dim": {"type": "integer", "minimum": 1},
            "distance": {"type": "string", "enum": sorted(_SUPPORTED_DISTANCES)},
            "vectors_path": {"type": "string", "minLength": 1},
            "entries_path": {"type": "string", "minLength": 1},
            "metadata": {"type": "object"},
        },
    }


__all__ = [
    "TRANSITION_INDEX_SCHEMA_VERSION",
    "IndexDistance",
    "TransitionIndex",
    "TransitionIndexEntry",
    "TransitionIndexError",
    "TransitionIndexSearchHit",
    "build_transition_index",
    "read_transition_index",
    "transition_index_header_json_schema",
    "write_transition_index",
]
