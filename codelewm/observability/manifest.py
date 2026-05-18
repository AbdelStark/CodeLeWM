"""Schema-versioned artifact manifest helpers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


ARTIFACT_MANIFEST_SCHEMA_VERSION = "codelewm.artifact_manifest.v1"
ArtifactKind = Literal[
    "dataset",
    "checkpoint",
    "training_run",
    "index",
    "eval_report",
    "score_report",
]
ARTIFACT_KINDS: tuple[str, ...] = (
    "dataset",
    "checkpoint",
    "training_run",
    "index",
    "eval_report",
    "score_report",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class ArtifactManifestError(ValueError):
    """Raised when an artifact manifest is malformed or unverifiable."""


@dataclass(frozen=True)
class ManifestFile:
    """Checksum-bearing file entry within an artifact directory."""

    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _validate_relative_manifest_path(self.path, field_name="files[].path")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ArtifactManifestError("files[].sha256 must be a lowercase 64-character SHA-256 hex digest")
        if self.bytes < 0:
            raise ArtifactManifestError("files[].bytes must be non-negative")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifestFile":
        _reject_unknown(payload, {"path", "sha256", "bytes"}, "manifest file")
        return cls(
            path=_require_string(payload, "path", "manifest file"),
            sha256=_require_string(payload, "sha256", "manifest file"),
            bytes=_require_int(payload, "bytes", "manifest file"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class ArtifactManifest:
    """JSON-native artifact lineage manifest."""

    artifact_id: str
    artifact_kind: ArtifactKind
    created_at: str
    source_git_sha: str
    command: tuple[str, ...]
    config_sha256: str
    parent_artifacts: tuple[str, ...]
    files: tuple[ManifestFile, ...]
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ArtifactManifestError(
                "schema_version must be "
                f"{ARTIFACT_MANIFEST_SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if not isinstance(self.artifact_id, str):
            raise ArtifactManifestError("artifact_id must be a string")
        if not _ARTIFACT_ID_RE.fullmatch(self.artifact_id):
            raise ArtifactManifestError("artifact_id must be non-empty and contain only stable id characters")
        if not isinstance(self.artifact_kind, str):
            raise ArtifactManifestError("artifact_kind must be a string")
        if self.artifact_kind not in ARTIFACT_KINDS:
            allowed = ", ".join(ARTIFACT_KINDS)
            raise ArtifactManifestError(f"artifact_kind must be one of: {allowed}")
        if not isinstance(self.created_at, str):
            raise ArtifactManifestError("created_at must be a string")
        _parse_utc_timestamp(self.created_at)
        if not isinstance(self.source_git_sha, str):
            raise ArtifactManifestError("source_git_sha must be a string")
        if self.source_git_sha != "unknown" and not _GIT_SHA_RE.fullmatch(self.source_git_sha):
            raise ArtifactManifestError("source_git_sha must be a 40-character git SHA or 'unknown'")
        if not self.command or any(not isinstance(item, str) or not item for item in self.command):
            raise ArtifactManifestError("command must contain at least one non-empty argv item")
        if not isinstance(self.config_sha256, str):
            raise ArtifactManifestError("config_sha256 must be a string")
        if not _SHA256_RE.fullmatch(self.config_sha256):
            raise ArtifactManifestError("config_sha256 must be a lowercase 64-character SHA-256 hex digest")
        if any(not isinstance(parent, str) or not parent for parent in self.parent_artifacts):
            raise ArtifactManifestError("parent_artifacts must not contain empty IDs")
        if len(set(self.parent_artifacts)) != len(self.parent_artifacts):
            raise ArtifactManifestError("parent_artifacts must not contain duplicates")
        if any(not isinstance(file, ManifestFile) for file in self.files):
            raise ArtifactManifestError("files must contain ManifestFile entries")
        file_paths = [file.path for file in self.files]
        if len(set(file_paths)) != len(file_paths):
            raise ArtifactManifestError("files must not contain duplicate paths")
        if not isinstance(self.metadata, Mapping):
            raise ArtifactManifestError("metadata must be a JSON object")
        _ensure_json_native(self.metadata, field_name="metadata")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactManifest":
        return validate_artifact_manifest_payload(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "created_at": self.created_at,
            "source_git_sha": self.source_git_sha,
            "command": list(self.command),
            "config_sha256": self.config_sha256,
            "parent_artifacts": list(self.parent_artifacts),
            "files": [file.to_dict() for file in self.files],
            "metadata": dict(self.metadata),
        }


def artifact_manifest_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for CodeLeWM artifact manifests."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "artifact_id",
            "artifact_kind",
            "created_at",
            "source_git_sha",
            "command",
            "config_sha256",
            "parent_artifacts",
            "files",
            "metadata",
        ],
        "properties": {
            "schema_version": {"const": ARTIFACT_MANIFEST_SCHEMA_VERSION},
            "artifact_id": {"type": "string", "pattern": _ARTIFACT_ID_RE.pattern},
            "artifact_kind": {"type": "string", "enum": list(ARTIFACT_KINDS)},
            "created_at": {"type": "string", "format": "date-time"},
            "source_git_sha": {
                "type": "string",
                "pattern": f"^({_GIT_SHA_RE.pattern[1:-1]}|unknown)$",
            },
            "command": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "config_sha256": {"type": "string", "pattern": _SHA256_RE.pattern},
            "parent_artifacts": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "sha256", "bytes"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "sha256": {"type": "string", "pattern": _SHA256_RE.pattern},
                        "bytes": {"type": "integer", "minimum": 0},
                    },
                },
                "uniqueItems": True,
            },
            "metadata": {"type": "object"},
        },
    }


def build_artifact_manifest(
    *,
    artifact_kind: ArtifactKind,
    root: Path | str,
    files: Iterable[Path | str],
    command: Sequence[str],
    config: Any,
    parent_artifacts: Sequence[str] = (),
    source_git_sha: str | None = None,
    created_at: str | None = None,
    artifact_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactManifest:
    """Build an artifact manifest from files under one artifact root."""

    root_path = Path(root)
    manifest_files = tuple(build_manifest_file(path, root=root_path) for path in files)
    timestamp = created_at or utc_now()
    git_sha = source_git_sha or detect_source_git_sha(root_path)
    config_sha256 = compute_json_sha256(config)
    metadata_payload = {} if metadata is None else dict(metadata)
    base_payload = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "created_at": timestamp,
        "source_git_sha": git_sha,
        "command": list(command),
        "config_sha256": config_sha256,
        "parent_artifacts": list(parent_artifacts),
        "files": [file.to_dict() for file in manifest_files],
        "metadata": metadata_payload,
    }
    resolved_artifact_id = artifact_id or _derive_artifact_id(artifact_kind, base_payload)
    return ArtifactManifest(
        schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        artifact_id=resolved_artifact_id,
        artifact_kind=artifact_kind,
        created_at=timestamp,
        source_git_sha=git_sha,
        command=tuple(command),
        config_sha256=config_sha256,
        parent_artifacts=tuple(parent_artifacts),
        files=manifest_files,
        metadata=metadata_payload,
    )


def build_manifest_file(path: Path | str, *, root: Path | str) -> ManifestFile:
    """Build a manifest file entry and reject paths outside the artifact root."""

    root_path = Path(root).resolve()
    raw_path = Path(path)
    resolved_path = raw_path.resolve() if raw_path.is_absolute() else (root_path / raw_path).resolve()
    try:
        relative_path = resolved_path.relative_to(root_path)
    except ValueError as exc:
        raise ArtifactManifestError(f"manifest file must live under artifact root: {path}") from exc
    if not resolved_path.is_file():
        raise ArtifactManifestError(f"manifest file does not exist or is not a file: {relative_path.as_posix()}")
    return ManifestFile(
        path=relative_path.as_posix(),
        sha256=sha256_file(resolved_path),
        bytes=resolved_path.stat().st_size,
    )


def write_artifact_manifest(manifest: ArtifactManifest, path: Path | str) -> ArtifactManifest:
    """Write a validated artifact manifest JSON file."""

    manifest = validate_artifact_manifest_payload(manifest.to_dict())
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_artifact_manifest(path: Path | str) -> ArtifactManifest:
    """Read and validate an artifact manifest JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ArtifactManifestError("artifact manifest must be a JSON object")
    return validate_artifact_manifest_payload(payload)


def validate_artifact_manifest_payload(payload: Mapping[str, Any]) -> ArtifactManifest:
    """Validate a manifest payload against the CodeLeWM artifact schema."""

    _reject_unknown(
        payload,
        {
            "schema_version",
            "artifact_id",
            "artifact_kind",
            "created_at",
            "source_git_sha",
            "command",
            "config_sha256",
            "parent_artifacts",
            "files",
            "metadata",
        },
        "artifact manifest",
    )
    files_payload = _require_sequence(payload, "files", "artifact manifest")
    if "metadata" not in payload:
        raise ArtifactManifestError("artifact manifest.metadata is required")
    metadata_payload = payload["metadata"]
    if not isinstance(metadata_payload, Mapping):
        raise ArtifactManifestError("artifact manifest.metadata must be an object")
    return ArtifactManifest(
        schema_version=_require_string(payload, "schema_version", "artifact manifest"),
        artifact_id=_require_string(payload, "artifact_id", "artifact manifest"),
        artifact_kind=_require_string(payload, "artifact_kind", "artifact manifest"),  # type: ignore[arg-type]
        created_at=_require_string(payload, "created_at", "artifact manifest"),
        source_git_sha=_require_string(payload, "source_git_sha", "artifact manifest"),
        command=tuple(_require_string_items(payload, "command", "artifact manifest")),
        config_sha256=_require_string(payload, "config_sha256", "artifact manifest"),
        parent_artifacts=tuple(_require_string_items(payload, "parent_artifacts", "artifact manifest")),
        files=tuple(ManifestFile.from_dict(_require_mapping_item(item, "files")) for item in files_payload),
        metadata=dict(metadata_payload),
    )


def validate_artifact_checksums(manifest: ArtifactManifest, *, root: Path | str) -> tuple[Path, ...]:
    """Validate file existence, byte sizes, and checksums for a manifest."""

    root_path = Path(root).resolve()
    resolved_paths: list[Path] = []
    for file in manifest.files:
        path = _resolve_manifest_path(file.path, root=root_path)
        if not path.is_file():
            raise ArtifactManifestError(f"manifest file does not exist: {file.path}")
        observed_bytes = path.stat().st_size
        if observed_bytes != file.bytes:
            raise ArtifactManifestError(
                f"byte size mismatch for {file.path}: expected {file.bytes}, got {observed_bytes}"
            )
        observed_sha256 = sha256_file(path)
        if observed_sha256 != file.sha256:
            raise ArtifactManifestError(
                f"checksum mismatch for {file.path}: expected {file.sha256}, got {observed_sha256}"
            )
        resolved_paths.append(path)
    return tuple(resolved_paths)


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_json_sha256(payload: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-native data."""

    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactManifestError(f"payload is not JSON-native: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def detect_source_git_sha(root: Path | str = ".") -> str:
    """Return the current git HEAD SHA for root, or 'unknown' outside git."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    sha = result.stdout.strip().lower()
    return sha if _GIT_SHA_RE.fullmatch(sha) else "unknown"


def utc_now() -> str:
    """Return a second-resolution UTC timestamp for manifests."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _derive_artifact_id(artifact_kind: str, payload: Mapping[str, Any]) -> str:
    return f"{artifact_kind}-{compute_json_sha256(payload)[:16]}"


def _resolve_manifest_path(path: str, *, root: Path) -> Path:
    _validate_relative_manifest_path(path, field_name="files[].path")
    resolved_root = root.resolve()
    resolved_path = (resolved_root / path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactManifestError(f"manifest file path escapes artifact root: {path}") from exc
    return resolved_path


def _validate_relative_manifest_path(path: str, *, field_name: str) -> None:
    if not path:
        raise ArtifactManifestError(f"{field_name} must not be empty")
    raw_path = Path(path)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ArtifactManifestError(f"{field_name} must be relative and stay inside the artifact root")
    if any(part in {"", "."} for part in raw_path.parts):
        raise ArtifactManifestError(f"{field_name} must not contain empty or current-directory parts")


def _parse_utc_timestamp(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ArtifactManifestError("created_at must be an ISO-8601 UTC timestamp ending in 'Z'")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ArtifactManifestError("created_at must be a valid ISO-8601 UTC timestamp") from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ArtifactManifestError(f"{section} contains unknown key(s): {joined}")


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    if key not in payload:
        raise ArtifactManifestError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise ArtifactManifestError(f"{section}.{key} must be a string")
    return value


def _require_int(payload: Mapping[str, Any], key: str, section: str) -> int:
    if key not in payload:
        raise ArtifactManifestError(f"{section}.{key} is required")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactManifestError(f"{section}.{key} must be an integer")
    return value


def _require_sequence(payload: Mapping[str, Any], key: str, section: str) -> Sequence[Any]:
    if key not in payload:
        raise ArtifactManifestError(f"{section}.{key} is required")
    value = payload[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactManifestError(f"{section}.{key} must be a JSON array")
    return value


def _require_string_items(payload: Mapping[str, Any], key: str, section: str) -> tuple[str, ...]:
    values = _require_sequence(payload, key, section)
    for value in values:
        if not isinstance(value, str):
            raise ArtifactManifestError(f"{section}.{key} must contain only strings")
    return tuple(values)


def _require_mapping_item(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactManifestError(f"{section} entries must be JSON objects")
    return value


def _ensure_json_native(payload: Any, *, field_name: str) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArtifactManifestError(f"{field_name} must be JSON-native: {exc}") from exc
