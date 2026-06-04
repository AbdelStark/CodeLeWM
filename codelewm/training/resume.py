"""Checkpoint resume compatibility for the manifest-backed training runner.

A resume reuses the parent run's checkpoint as the warm-start. Per RFC-0006
and ``docs/spec/09-release-and-versioning.md``, a resume is only allowed when
the schema, config, and latent dimension agree with the parent. Silent
partial resumes are refused; every failure raises ``TrainingRunError`` with
a structured message that identifies the parent and the disagreeing field.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from codelewm.model.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointCompatibilitySpec,
    CheckpointManifest,
    compute_config_hash,
    load_checkpoint_manifest,
)
from codelewm.observability import (
    ArtifactManifest,
    ArtifactManifestError,
    read_artifact_manifest,
    validate_artifact_checksums,
)

from .config import TrainConfig
from .runner import TrainingRunError, TrainingRunManifest, read_training_run_manifest


@dataclass(frozen=True)
class CheckpointResumePlan:
    """Validated plan describing how a new training run resumes a parent.

    The plan is produced by :func:`prepare_checkpoint_resume` and consumed by
    the manifest-backed training runner. Callers should not mutate it.
    """

    parent_training_manifest: TrainingRunManifest
    parent_training_manifest_path: Path
    parent_artifact_manifest: ArtifactManifest
    parent_artifact_manifest_path: Path
    parent_checkpoint_manifest: CheckpointManifest
    parent_checkpoint_manifest_path: Path
    parent_checkpoint_path: Path

    @property
    def parent_artifact_id(self) -> str:
        return self.parent_artifact_manifest.artifact_id


def prepare_checkpoint_resume(
    parent_training_manifest_path: Path | str,
    *,
    config: TrainConfig,
    root: Path | str = ".",
) -> CheckpointResumePlan:
    """Validate a resume from ``parent_training_manifest_path`` against ``config``.

    The function:

    - reads the parent ``codelewm.training_run.v1`` manifest;
    - reads and checksum-validates the parent ``codelewm.artifact_manifest.v1``;
    - reads and checksum-validates the parent ``codelewm.checkpoint.v1``;
    - checks that the parent's checkpoint metadata matches the new config's
      record schema, latent dimension, action view, and config hash;
    - returns a :class:`CheckpointResumePlan` for the runner.

    Any incompatibility raises :class:`TrainingRunError` with a message
    that names the mismatched field. The parent's artifact id is intended
    to be added to the new run's ``parent_artifacts`` list by the runner.
    """

    parent_manifest_path = Path(parent_training_manifest_path)
    if not parent_manifest_path.is_file():
        raise TrainingRunError(
            f"parent training run manifest does not exist: {parent_manifest_path}"
        )
    try:
        parent_training_manifest = read_training_run_manifest(parent_manifest_path)
    except (json.JSONDecodeError, OSError) as exc:
        raise TrainingRunError(
            f"parent training run manifest could not be read: {parent_manifest_path}"
        ) from exc

    parent_run_root = parent_manifest_path.parent
    parent_artifact_manifest_path = (
        parent_run_root / parent_training_manifest.artifact_manifest_path
    ).resolve()
    if not parent_artifact_manifest_path.is_file():
        raise TrainingRunError(
            "parent artifact manifest does not exist: "
            f"{parent_artifact_manifest_path}"
        )
    try:
        parent_artifact_manifest = read_artifact_manifest(parent_artifact_manifest_path)
        validate_artifact_checksums(
            parent_artifact_manifest, root=parent_artifact_manifest_path.parent
        )
    except ArtifactManifestError as exc:
        raise TrainingRunError(
            f"parent artifact manifest failed validation: {exc}"
        ) from exc

    parent_checkpoint_manifest_path, parent_checkpoint_path = _locate_parent_checkpoint(
        parent_training_manifest,
        parent_run_root=parent_run_root,
    )
    try:
        parent_checkpoint_manifest = load_checkpoint_manifest(
            parent_checkpoint_manifest_path
        )
    except CheckpointCompatibilityError as exc:
        raise TrainingRunError(
            f"parent checkpoint manifest failed validation: {exc}"
        ) from exc

    if parent_checkpoint_manifest.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise TrainingRunError(
            "parent checkpoint schema_version is unsupported for resume: "
            f"{parent_checkpoint_manifest.schema_version!r}; expected "
            f"{CHECKPOINT_SCHEMA_VERSION!r}"
        )

    new_config_hash = compute_config_hash(_compatibility_config_payload(config))
    parent_meta = parent_checkpoint_manifest.metadata
    expected_record_schema = parent_meta.record_schema_version
    if config.wm.embed_dim != parent_meta.latent_dim:
        raise TrainingRunError(
            "resume rejected: latent dimension differs from parent checkpoint; "
            f"parent latent_dim={parent_meta.latent_dim}, "
            f"new wm.embed_dim={config.wm.embed_dim}"
        )
    if config.wm.action_view != parent_meta.action_view:
        raise TrainingRunError(
            "resume rejected: action_view differs from parent checkpoint; "
            f"parent action_view={parent_meta.action_view!r}, "
            f"new wm.action_view={config.wm.action_view!r}"
        )
    if new_config_hash != parent_meta.config_hash:
        raise TrainingRunError(
            "resume rejected: config_hash differs from parent checkpoint; "
            f"parent config_hash={parent_meta.config_hash}, "
            f"new config_hash={new_config_hash}"
        )

    # Cross-check via the existing compatibility spec to share one source of truth.
    spec = CheckpointCompatibilitySpec(
        config_hash=new_config_hash,
        record_schema_version=expected_record_schema,
        latent_dim=parent_meta.latent_dim,
        action_view=parent_meta.action_view,
    )
    try:
        _validate_via_spec(parent_checkpoint_manifest, spec)
    except CheckpointCompatibilityError as exc:
        raise TrainingRunError(f"resume rejected: {exc}") from exc

    return CheckpointResumePlan(
        parent_training_manifest=parent_training_manifest,
        parent_training_manifest_path=parent_manifest_path,
        parent_artifact_manifest=parent_artifact_manifest,
        parent_artifact_manifest_path=parent_artifact_manifest_path,
        parent_checkpoint_manifest=parent_checkpoint_manifest,
        parent_checkpoint_manifest_path=parent_checkpoint_manifest_path,
        parent_checkpoint_path=parent_checkpoint_path,
    )


def _locate_parent_checkpoint(
    manifest: TrainingRunManifest,
    *,
    parent_run_root: Path,
) -> tuple[Path, Path]:
    """Return ``(checkpoint_manifest_path, checkpoint_path)`` for the parent.

    The checkpoint manifest is looked up beside each declared checkpoint file
    at ``<checkpoint>.manifest.json``. Raises ``TrainingRunError`` if a
    matching pair cannot be found.
    """

    if not manifest.checkpoint_files:
        raise TrainingRunError(
            "parent training run has no checkpoint files; cannot resume from "
            f"{manifest.artifact_manifest_id}"
        )
    for entry in manifest.checkpoint_files:
        checkpoint_path = (parent_run_root / entry.path).resolve()
        candidate_manifest = checkpoint_path.with_name(
            checkpoint_path.name + ".manifest.json"
        )
        if candidate_manifest.is_file():
            return candidate_manifest, checkpoint_path
    raise TrainingRunError(
        "parent training run does not include a checkpoint manifest beside any "
        "checkpoint file; expected <checkpoint>.manifest.json"
    )


def _validate_via_spec(
    manifest: CheckpointManifest, spec: CheckpointCompatibilitySpec
) -> None:
    from codelewm.model.checkpoint import validate_checkpoint_compatibility

    validate_checkpoint_compatibility(manifest, spec)


def _compatibility_config_payload(config: TrainConfig) -> dict:
    """Return the subset of ``config`` that defines checkpoint compatibility.

    A resume requires the same architecture and loss surface so that the
    deserialized weights stay meaningful. Run-specific fields (name, seed,
    output paths, dataset paths) deliberately do not participate so a user
    can resume into a fresh run directory.
    """

    wm_payload = config.wm.to_compatibility_dict()
    if config.loss.enable_p_pass_bce or config.loss.p_pass_bce_weight != 0.0:
        wm_payload["enable_pass_head"] = True
    return {
        "wm": wm_payload,
        "loss": config.loss.to_compatibility_dict(),
    }


__all__ = [
    "CheckpointResumePlan",
    "prepare_checkpoint_resume",
    "compatibility_config_payload",
]


def compatibility_config_payload(config: TrainConfig) -> dict:
    """Public wrapper returning the resume-compatibility config payload.

    Executors writing a checkpoint manifest should pass this payload to
    :func:`codelewm.model.build_checkpoint_metadata` so the recorded
    ``config_hash`` agrees with the resume validator.
    """

    return _compatibility_config_payload(config)
