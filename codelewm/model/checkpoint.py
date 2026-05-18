"""Checkpoint compatibility metadata and manifest checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from codelewm.model.transition import ActionView, LATENT_DIM


CHECKPOINT_SCHEMA_VERSION = "codelewm.checkpoint.v1"
DEFAULT_RECORD_SCHEMA_VERSION = "codelewm.transition.v1"


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot be safely used with the requested contract."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Compatibility metadata recorded beside checkpoint weights."""

    config_hash: str
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    record_schema_version: str = DEFAULT_RECORD_SCHEMA_VERSION
    latent_dim: int = LATENT_DIM
    action_view: ActionView = "text"
    model_class: str = "CodeTransitionModel"

    def __post_init__(self) -> None:
        if not self.config_hash:
            raise ValueError("config_hash must not be empty")
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        if not self.record_schema_version:
            raise ValueError("record_schema_version must not be empty")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.action_view not in ("text", "abstract", "patch"):
            raise ValueError(f"unsupported action_view: {self.action_view}")
        if not self.model_class:
            raise ValueError("model_class must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_schema_version": self.record_schema_version,
            "latent_dim": self.latent_dim,
            "action_view": self.action_view,
            "config_hash": self.config_hash,
            "model_class": self.model_class,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointMetadata":
        return cls(
            schema_version=str(payload["schema_version"]),
            record_schema_version=str(payload["record_schema_version"]),
            latent_dim=int(payload["latent_dim"]),
            action_view=payload["action_view"],
            config_hash=str(payload["config_hash"]),
            model_class=str(payload.get("model_class", "CodeTransitionModel")),
        )


@dataclass(frozen=True)
class CheckpointCompatibilitySpec:
    """Expected checkpoint contract for a load or resume request."""

    config_hash: str
    record_schema_version: str = DEFAULT_RECORD_SCHEMA_VERSION
    latent_dim: int = LATENT_DIM
    action_view: ActionView = "text"

    def __post_init__(self) -> None:
        if not self.config_hash:
            raise ValueError("config_hash must not be empty")
        if not self.record_schema_version:
            raise ValueError("record_schema_version must not be empty")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.action_view not in ("text", "abstract", "patch"):
            raise ValueError(f"unsupported action_view: {self.action_view}")


@dataclass(frozen=True)
class CheckpointManifest:
    """JSON-native checkpoint manifest used before loading serialized weights."""

    metadata: CheckpointMetadata
    checkpoint_path: str
    checkpoint_sha256: str
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    migration_hook: str | None = None

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path must not be empty")
        if not self.checkpoint_sha256:
            raise ValueError("checkpoint_sha256 must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "metadata": self.metadata.to_dict(),
            "migration_hook": self.migration_hook,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointManifest":
        metadata_payload = payload.get("metadata")
        if not isinstance(metadata_payload, dict):
            raise CheckpointCompatibilityError("checkpoint manifest metadata must be an object")
        return cls(
            schema_version=str(payload["schema_version"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            metadata=CheckpointMetadata.from_dict(metadata_payload),
            migration_hook=(
                None if payload.get("migration_hook") is None else str(payload["migration_hook"])
            ),
        )


def compute_config_hash(config: Any) -> str:
    """Return a deterministic SHA-256 hash for a JSON-native config object."""

    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_checkpoint_metadata(
    config: Any,
    *,
    record_schema_version: str = DEFAULT_RECORD_SCHEMA_VERSION,
    latent_dim: int = LATENT_DIM,
    action_view: ActionView = "text",
    model_class: str = "CodeTransitionModel",
) -> CheckpointMetadata:
    """Build checkpoint metadata from the resolved training config."""

    return CheckpointMetadata(
        config_hash=compute_config_hash(config),
        record_schema_version=record_schema_version,
        latent_dim=latent_dim,
        action_view=action_view,
        model_class=model_class,
    )


def write_checkpoint_manifest(
    *,
    metadata: CheckpointMetadata,
    checkpoint_path: Path,
    manifest_path: Path,
    migration_hook: str | None = None,
) -> CheckpointManifest:
    """Write a checkpoint manifest after recording the checkpoint file checksum."""

    manifest_root = manifest_path.parent
    relative_checkpoint = _safe_relative_path(checkpoint_path, root=manifest_root)
    manifest = CheckpointManifest(
        metadata=metadata,
        checkpoint_path=relative_checkpoint.as_posix(),
        checkpoint_sha256=sha256_file(checkpoint_path),
        migration_hook=migration_hook,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return manifest


def read_checkpoint_manifest(path: Path) -> CheckpointManifest:
    """Read a checkpoint manifest without loading serialized weight objects."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("checkpoint manifest must be a JSON object")
    return CheckpointManifest.from_dict(payload)


def load_checkpoint_manifest(
    path: Path,
    *,
    expected: CheckpointCompatibilitySpec | None = None,
    verify_checksum: bool = True,
) -> CheckpointManifest:
    """Read, verify, and optionally compatibility-check a checkpoint manifest."""

    manifest = read_checkpoint_manifest(path)
    if verify_checksum:
        validate_checkpoint_file(manifest, root=path.parent)
    if expected is not None:
        validate_checkpoint_compatibility(manifest, expected)
    return manifest


def validate_checkpoint_file(manifest: CheckpointManifest, *, root: Path) -> Path:
    """Return the safe checkpoint path after verifying existence and checksum."""

    checkpoint_path = resolve_checkpoint_path(manifest, root=root)
    if not checkpoint_path.exists():
        raise CheckpointCompatibilityError(f"checkpoint file does not exist: {manifest.checkpoint_path}")
    actual = sha256_file(checkpoint_path)
    if actual != manifest.checkpoint_sha256:
        raise CheckpointCompatibilityError(
            "checkpoint checksum mismatch; "
            f"expected {manifest.checkpoint_sha256}, got {actual}"
        )
    return checkpoint_path


def validate_checkpoint_compatibility(
    manifest: CheckpointManifest,
    expected: CheckpointCompatibilitySpec,
) -> None:
    """Refuse checkpoints whose metadata does not match the expected contract."""

    if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION:
        _raise_schema_mismatch("manifest schema_version", manifest.schema_version, manifest)
    metadata = manifest.metadata
    if metadata.schema_version != CHECKPOINT_SCHEMA_VERSION:
        _raise_schema_mismatch("metadata schema_version", metadata.schema_version, manifest)

    checks = {
        "record_schema_version": (metadata.record_schema_version, expected.record_schema_version),
        "latent_dim": (metadata.latent_dim, expected.latent_dim),
        "action_view": (metadata.action_view, expected.action_view),
        "config_hash": (metadata.config_hash, expected.config_hash),
    }
    for field, (actual, wanted) in checks.items():
        if actual != wanted:
            raise CheckpointCompatibilityError(
                f"checkpoint {field} mismatch; expected {wanted!r}, got {actual!r}"
            )


def resolve_checkpoint_path(manifest: CheckpointManifest, *, root: Path) -> Path:
    """Resolve a manifest checkpoint path and reject absolute or escaping paths."""

    raw = Path(manifest.checkpoint_path)
    if raw.is_absolute() or ".." in raw.parts:
        raise CheckpointCompatibilityError(
            f"checkpoint_path must be relative to the manifest directory: {manifest.checkpoint_path}"
        )
    resolved_root = root.resolve()
    resolved_path = (resolved_root / raw).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint_path must stay inside the manifest directory: {manifest.checkpoint_path}"
        ) from exc
    return resolved_path


def migrate_checkpoint_manifest(
    manifest: CheckpointManifest,
    *,
    expected: CheckpointCompatibilitySpec,
) -> NoReturn:
    """Placeholder for explicit future migrations; never migrates implicitly."""

    hook = manifest.migration_hook or "none"
    raise CheckpointCompatibilityError(
        "checkpoint migration is not implemented; "
        f"hook={hook!r}, expected={expected.record_schema_version}/{expected.action_view}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(path: Path, *, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint file must live under manifest directory: {path}"
        ) from exc


def _raise_schema_mismatch(field: str, actual: str, manifest: CheckpointManifest) -> NoReturn:
    if manifest.migration_hook:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint {field} {actual!r}; "
            f"explicit migration hook {manifest.migration_hook!r} is recorded but not run implicitly"
        )
    raise CheckpointCompatibilityError(
        f"unsupported checkpoint {field} {actual!r}; expected {CHECKPOINT_SCHEMA_VERSION!r}"
    )
