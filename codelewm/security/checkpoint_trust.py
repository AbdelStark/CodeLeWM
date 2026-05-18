"""Checkpoint trust boundary: refuse unmanifested loads by default.

Backends that deserialize Python objects from a checkpoint file (pickle,
``torch.load``, etc.) must call :func:`require_trusted_checkpoint` before
opening the file. The default policy is to refuse any checkpoint that is not
accompanied by a valid manifest with matching checksums; callers that work in
a trusted local environment can pass ``allow_unsafe=True`` to bypass the
check, but only after surfacing the override at the user-facing CLI or API
boundary.
"""

from __future__ import annotations

from pathlib import Path

from codelewm.model.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointManifest,
    load_checkpoint_manifest,
)


class CheckpointTrustError(RuntimeError):
    """Raised when a checkpoint cannot be loaded under the trust boundary."""


def default_checkpoint_manifest_path(checkpoint: Path | str) -> Path:
    """Return the default manifest location for a checkpoint file."""

    checkpoint_path = Path(checkpoint)
    return checkpoint_path.with_name(checkpoint_path.name + ".manifest.json")


def require_trusted_checkpoint(
    checkpoint: Path | str,
    *,
    manifest_path: Path | str | None = None,
) -> CheckpointManifest:
    """Return a verified checkpoint manifest, or raise ``CheckpointTrustError``.

    The manifest defaults to ``<checkpoint>.manifest.json``. The manifest must
    exist, must reference the same checkpoint file, and must match the
    recorded SHA-256.
    """

    checkpoint_file = Path(checkpoint)
    resolved_manifest = (
        Path(manifest_path) if manifest_path is not None else default_checkpoint_manifest_path(checkpoint_file)
    )
    if not resolved_manifest.exists():
        raise CheckpointTrustError(
            "checkpoint manifest is required but was not found: "
            f"{resolved_manifest}; generate the manifest or opt in to "
            "unmanifested loads explicitly"
        )
    if not resolved_manifest.is_file():
        raise CheckpointTrustError(
            f"checkpoint manifest path is not a file: {resolved_manifest}"
        )
    try:
        manifest = load_checkpoint_manifest(resolved_manifest)
    except CheckpointCompatibilityError as exc:
        raise CheckpointTrustError(f"checkpoint manifest failed validation: {exc}") from exc
    expected_path = (resolved_manifest.parent / manifest.checkpoint_path).resolve()
    actual_path = checkpoint_file.resolve()
    if expected_path != actual_path:
        raise CheckpointTrustError(
            "checkpoint manifest does not reference the requested checkpoint: "
            f"manifest points at {manifest.checkpoint_path!r}, requested "
            f"{actual_path}"
        )
    return manifest
