"""Dataset staging, HDF5 packing, and manifest contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.data.sources import SourceKind
from codelewm.data.split_dedup import SplitName


DATASET_SCHEMA_VERSION = "codelewm.transition.v1"
SOURCE_CODES: dict[SourceKind, int] = {
    "commitpackft": 1,
    "commitpack": 2,
    "agentpack": 3,
    "synthetic": 4,
    "local_repo": 5,
}
SPLIT_CODES: dict[SplitName, int] = {"train": 0, "val": 1, "test": 2}


class PackError(ValueError):
    """Raised when a transition cannot be packed without losing data."""


class OptionalDependencyError(RuntimeError):
    """Raised when a requested artifact format needs an optional dependency."""


@dataclass(frozen=True)
class TokenSequence:
    """Token ids plus optional masks for one fixed-width packed array."""

    input_ids: Sequence[int]
    attention_mask: Sequence[bool] | None = None
    segment_ids: Sequence[int] | None = None
    changed_hunk_mask: Sequence[bool] | None = None


@dataclass(frozen=True)
class PackedTransition:
    """Tokenized transition ready for staging and HDF5 packing."""

    transition_id: str
    source: SourceKind
    repo: str
    commit: str
    path: str
    split: SplitName
    state_before: TokenSequence
    state_after: TokenSequence
    action_text: TokenSequence
    action_abs: TokenSequence
    action_patch: TokenSequence | None = None
    edit_size: int = 0
    token_count_before: int | None = None
    token_count_after: int | None = None
    license: str | None = None
    filter_flags: tuple[str, ...] = ()
    dedup_keys: tuple[str, ...] = ()

    def to_parquet_row(self) -> dict[str, object]:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "transition_id": self.transition_id,
            "source": self.source,
            "repo": self.repo,
            "commit": self.commit,
            "path": self.path,
            "split": self.split,
            "state_before_input_ids": list(self.state_before.input_ids),
            "state_after_input_ids": list(self.state_after.input_ids),
            "action_text_input_ids": list(self.action_text.input_ids),
            "action_abs_input_ids": list(self.action_abs.input_ids),
            "action_patch_input_ids": None
            if self.action_patch is None
            else list(self.action_patch.input_ids),
            "edit_size": self.edit_size,
            "token_count_before": self.token_count_before,
            "token_count_after": self.token_count_after,
            "license": self.license,
            "filter_flags": list(self.filter_flags),
            "dedup_keys": list(self.dedup_keys),
        }


@dataclass(frozen=True)
class PackSpec:
    """Fixed array lengths and feature flags for a dataset pack."""

    schema_version: str = DATASET_SCHEMA_VERSION
    state_length: int = 1024
    action_text_length: int = 256
    action_abs_length: int = 192
    action_patch_length: int = 512
    include_action_patch: bool = False

    def __post_init__(self) -> None:
        lengths = (
            self.state_length,
            self.action_text_length,
            self.action_abs_length,
            self.action_patch_length,
        )
        if any(length <= 0 for length in lengths):
            raise ValueError("pack sequence lengths must be positive")


@dataclass(frozen=True)
class ArtifactInfo:
    """Checksum-bearing dataset artifact descriptor."""

    path: str
    kind: str
    rows: int
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "rows": self.rows,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class DatasetManifest:
    """JSON-native dataset manifest."""

    schema_version: str
    row_count: int
    features: dict[str, bool]
    artifacts: tuple[ArtifactInfo, ...]
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "features": dict(self.features),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "split_counts": dict(self.split_counts),
            "source_counts": dict(self.source_counts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(
            schema_version=str(payload["schema_version"]),
            row_count=int(payload["row_count"]),
            features={str(key): bool(value) for key, value in payload["features"].items()},
            artifacts=tuple(ArtifactInfo(**artifact) for artifact in payload["artifacts"]),
            split_counts={str(key): int(value) for key, value in payload["split_counts"].items()},
            source_counts={str(key): int(value) for key, value in payload["source_counts"].items()},
            metadata=dict(payload.get("metadata", {})),
        )


def write_hdf5_pack(
    transitions: Iterable[PackedTransition],
    path: Path,
    *,
    spec: PackSpec = PackSpec(),
) -> ArtifactInfo:
    h5py = _require_h5py()
    rows = tuple(transitions)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = spec.schema_version
        handle.attrs["features.action_patch"] = bool(spec.include_action_patch)
        handle.attrs["row_count"] = len(rows)

        _write_state_group(handle, "state_before", [row.state_before for row in rows], spec.state_length)
        _write_state_group(handle, "state_after", [row.state_after for row in rows], spec.state_length)
        _write_action_group(handle, "action_text", [row.action_text for row in rows], spec.action_text_length)
        _write_action_group(handle, "action_abs", [row.action_abs for row in rows], spec.action_abs_length)
        if spec.include_action_patch:
            _write_action_group(
                handle,
                "action_patch",
                [_required_action_patch(row) for row in rows],
                spec.action_patch_length,
            )

        metadata = handle.create_group("metadata")
        _write_string_dataset(metadata, "repo", [row.repo for row in rows])
        _write_string_dataset(metadata, "path", [row.path for row in rows])
        _write_string_dataset(metadata, "commit", [row.commit for row in rows])
        _write_numeric_dataset(metadata, "source", [SOURCE_CODES[row.source] for row in rows], np.int8)
        _write_numeric_dataset(metadata, "split", [SPLIT_CODES[row.split] for row in rows], np.int8)
        _write_numeric_dataset(metadata, "edit_size", [row.edit_size for row in rows], np.int32)
        _write_numeric_dataset(
            metadata,
            "token_count_before",
            [_token_count(row.state_before, row.token_count_before) for row in rows],
            np.int32,
        )
        _write_numeric_dataset(
            metadata,
            "token_count_after",
            [_token_count(row.state_after, row.token_count_after) for row in rows],
            np.int32,
        )

    return _artifact_info(path, kind="hdf5", rows=len(rows))


def write_parquet_staging_shards(
    transitions: Iterable[PackedTransition],
    directory: Path,
    *,
    shard_size: int = 1000,
) -> tuple[ArtifactInfo, ...]:
    if shard_size <= 0:
        raise ValueError("parquet shard_size must be positive")
    pq, pa = _require_pyarrow()
    rows = tuple(transitions)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[ArtifactInfo] = []

    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = [row.to_parquet_row() for row in rows[start : start + shard_size]]
        shard_path = directory / f"part-{shard_index:05d}.parquet"
        table = pa.Table.from_pylist(shard_rows)
        pq.write_table(table, shard_path)
        artifacts.append(_artifact_info(shard_path, kind="parquet", rows=len(shard_rows)))

    if not rows:
        shard_path = directory / "part-00000.parquet"
        table = pa.Table.from_pylist([])
        pq.write_table(table, shard_path)
        artifacts.append(_artifact_info(shard_path, kind="parquet", rows=0))

    return tuple(artifacts)


def write_dataset_artifacts(
    transitions: Iterable[PackedTransition],
    output_dir: Path,
    *,
    spec: PackSpec = PackSpec(),
    parquet_shard_size: int = 1000,
) -> DatasetManifest:
    rows = tuple(transitions)
    parquet_artifacts = write_parquet_staging_shards(
        rows,
        output_dir / "parquet",
        shard_size=parquet_shard_size,
    )
    hdf5_artifact = write_hdf5_pack(rows, output_dir / "dataset.h5", spec=spec)
    artifacts = tuple(_relative_artifact(artifact, output_dir) for artifact in (*parquet_artifacts, hdf5_artifact))
    manifest = build_dataset_manifest(
        rows,
        artifacts=artifacts,
        spec=spec,
    )
    write_dataset_manifest(manifest, output_dir / "manifest.json")
    return manifest


def build_dataset_manifest(
    transitions: Iterable[PackedTransition],
    *,
    artifacts: Iterable[ArtifactInfo],
    spec: PackSpec = PackSpec(),
    metadata: dict[str, object] | None = None,
) -> DatasetManifest:
    rows = tuple(transitions)
    split_counts = {"train": 0, "val": 0, "test": 0}
    source_counts = {source: 0 for source in SOURCE_CODES}
    for row in rows:
        split_counts[row.split] += 1
        source_counts[row.source] += 1

    return DatasetManifest(
        schema_version=spec.schema_version,
        row_count=len(rows),
        features={"action_patch": spec.include_action_patch},
        artifacts=tuple(artifacts),
        split_counts=split_counts,
        source_counts=source_counts,
        metadata={} if metadata is None else dict(metadata),
    )


def write_dataset_manifest(manifest: DatasetManifest, path: Path) -> ArtifactInfo:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return _artifact_info(path, kind="manifest", rows=manifest.row_count)


def read_dataset_manifest(path: Path) -> DatasetManifest:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise PackError("dataset manifest must be a JSON object")
    return DatasetManifest.from_dict(payload)


def validate_manifest_checksums(manifest: DatasetManifest, *, root: Path) -> None:
    for artifact in manifest.artifacts:
        artifact_path = root / artifact.path
        if not artifact_path.exists():
            raise PackError(f"manifest artifact does not exist: {artifact.path}")
        observed = sha256_file(artifact_path)
        if observed != artifact.sha256:
            raise PackError(f"checksum mismatch for {artifact.path}: expected {artifact.sha256}, got {observed}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_info(path: Path, *, kind: str, rows: int) -> ArtifactInfo:
    return ArtifactInfo(
        path=path.name if path.parent == Path(".") else str(path),
        kind=kind,
        rows=rows,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
    )


def _write_state_group(handle: Any, name: str, sequences: Sequence[TokenSequence], length: int) -> None:
    group = handle.create_group(name)
    _write_numeric_matrix(group, "input_ids", [seq.input_ids for seq in sequences], length, np.int32)
    _write_bool_matrix(group, "attention_mask", [_mask(seq, length) for seq in sequences], length)
    _write_numeric_matrix(group, "segment_ids", [_segment_ids(seq, length) for seq in sequences], length, np.int16)
    _write_bool_matrix(
        group,
        "changed_hunk_mask",
        [_changed_hunk_mask(seq, length) for seq in sequences],
        length,
    )


def _write_action_group(handle: Any, name: str, sequences: Sequence[TokenSequence], length: int) -> None:
    group = handle.create_group(name)
    _write_numeric_matrix(group, "input_ids", [seq.input_ids for seq in sequences], length, np.int32)
    _write_bool_matrix(group, "attention_mask", [_mask(seq, length) for seq in sequences], length)


def _write_numeric_matrix(
    group: Any,
    name: str,
    rows: Sequence[Sequence[int]],
    length: int,
    dtype: Any,
) -> None:
    if not rows:
        matrix = np.zeros((0, length), dtype=dtype)
    else:
        matrix = np.vstack([_pad_int(row, length, name=name) for row in rows]).astype(dtype)
    group.create_dataset(name, data=matrix)


def _write_bool_matrix(group: Any, name: str, rows: Sequence[Sequence[bool]], length: int) -> None:
    if not rows:
        matrix = np.zeros((0, length), dtype=bool)
    else:
        matrix = np.vstack([_pad_bool(row, length, name=name) for row in rows]).astype(bool)
    group.create_dataset(name, data=matrix)


def _write_numeric_dataset(group: Any, name: str, values: Sequence[int], dtype: Any) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=dtype))


def _write_string_dataset(group: Any, name: str, values: Sequence[str]) -> None:
    h5py = _require_h5py()
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=dtype), dtype=dtype)


def _pad_int(values: Sequence[int], length: int, *, name: str) -> np.ndarray:
    if len(values) > length:
        raise PackError(f"{name} length {len(values)} exceeds fixed width {length}")
    output = np.zeros(length, dtype=np.int64)
    output[: len(values)] = values
    return output


def _pad_bool(values: Sequence[bool], length: int, *, name: str) -> np.ndarray:
    if len(values) > length:
        raise PackError(f"{name} length {len(values)} exceeds fixed width {length}")
    output = np.zeros(length, dtype=bool)
    output[: len(values)] = values
    return output


def _mask(sequence: TokenSequence, length: int) -> Sequence[bool]:
    if sequence.attention_mask is not None:
        return sequence.attention_mask
    return [True] * len(sequence.input_ids) + [False] * (length - len(sequence.input_ids))


def _segment_ids(sequence: TokenSequence, length: int) -> Sequence[int]:
    if sequence.segment_ids is not None:
        return sequence.segment_ids
    return [0] * min(len(sequence.input_ids), length)


def _changed_hunk_mask(sequence: TokenSequence, length: int) -> Sequence[bool]:
    if sequence.changed_hunk_mask is not None:
        return sequence.changed_hunk_mask
    return [False] * min(len(sequence.input_ids), length)


def _token_count(sequence: TokenSequence, explicit_count: int | None) -> int:
    return len(sequence.input_ids) if explicit_count is None else explicit_count


def _required_action_patch(row: PackedTransition) -> TokenSequence:
    if row.action_patch is None:
        raise PackError(f"transition {row.transition_id} is missing action_patch while feature is enabled")
    return row.action_patch


def _require_h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("HDF5 packing requires h5py; install codelewm[data]") from exc
    return h5py


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("Parquet staging requires pyarrow; install codelewm[data]") from exc
    return pq, pa


def _relative_artifact(artifact: ArtifactInfo, root: Path) -> ArtifactInfo:
    path = Path(artifact.path)
    try:
        artifact_path = str(path.relative_to(root))
    except ValueError:
        artifact_path = artifact.path
    return ArtifactInfo(
        path=artifact_path,
        kind=artifact.kind,
        rows=artifact.rows,
        sha256=artifact.sha256,
        bytes=artifact.bytes,
    )
