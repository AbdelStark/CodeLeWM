"""Manifest-backed transition-index builder for the public CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.eval.retrieval_runner import (
    _EvalRow,
    _PackPaths,
    _display_path,
    _infer_training_artifact_manifest_path,
    _load_split_rows,
    _load_torch_checkpoint,
    _read_verified_artifact_manifest,
    _require_torch_runtime,
    _resolve_device,
    _resolve_pack_paths,
    _state_batch,
)
from codelewm.observability import (
    ArtifactManifestError,
    read_artifact_manifest,
    validate_artifact_checksums,
)
from codelewm.security import require_trusted_checkpoint
from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE

from .transition_index import (
    TRANSITION_INDEX_SCHEMA_VERSION,
    IndexDistance,
    TransitionIndexEntry,
    TransitionIndexError,
    build_transition_index,
    write_transition_index,
)


INDEX_BUILD_RESULT_SCHEMA_VERSION = "codelewm.index_build.v1"


@dataclass(frozen=True)
class IndexBuildResult:
    """CLI-facing summary for a transition-index build."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    index_path: str
    vectors_path: str
    entries_path: str
    parent_artifacts: tuple[str, ...]
    count: int
    dim: int
    distance: str
    metadata: Mapping[str, Any]
    schema_version: str = INDEX_BUILD_RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "index_path": self.index_path,
            "vectors_path": self.vectors_path,
            "entries_path": self.entries_path,
            "parent_artifacts": list(self.parent_artifacts),
            "count": self.count,
            "dim": self.dim,
            "distance": self.distance,
            "metadata": dict(self.metadata),
        }


def build_transition_index_artifact(
    *,
    checkpoint: Path | str,
    data: Path | str,
    out: Path | str,
    device: str = "cpu",
    distance: IndexDistance = "l2",
    name: str = "codelewm-train-index",
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "index"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> IndexBuildResult:
    """Build a verified train-split transition index artifact."""

    if distance not in {"l2", "cosine"}:
        raise TransitionIndexError("distance must be l2 or cosine")
    if not name:
        raise TransitionIndexError("index name must not be empty")

    checkpoint_path = Path(checkpoint).resolve()
    out_dir = Path(out).resolve()
    _reject_existing_index_outputs(out_dir, overwrite=overwrite)

    pack_paths = _resolve_pack_paths(data)
    dataset_artifact = _read_verified_artifact_manifest(
        pack_paths.artifact_manifest_path,
        root=pack_paths.root,
    )
    if dataset_artifact.artifact_kind != "dataset":
        raise ArtifactManifestError("index --data manifest must be a dataset artifact")
    training_artifact_path = _infer_training_artifact_manifest_path(checkpoint_path)
    training_artifact = read_artifact_manifest(training_artifact_path)
    if training_artifact.artifact_kind != "training_run":
        raise ArtifactManifestError("checkpoint parent manifest must be a training_run artifact")
    checkpoint_manifest = require_trusted_checkpoint(checkpoint_path)
    validate_artifact_checksums(training_artifact, root=training_artifact_path.parent)

    runtime = _require_torch_runtime()
    selected_device = _resolve_device(device, runtime)
    model, checkpoint_payload = _load_torch_checkpoint(
        checkpoint_path,
        device=selected_device,
        runtime=runtime,
    )
    action_view = str(checkpoint_manifest.metadata.action_view)
    if action_view != str(model.config.action_view):
        raise TransitionIndexError(
            "checkpoint manifest action_view does not match checkpoint payload: "
            f"{action_view!r} != {model.config.action_view!r}"
        )

    rows = _load_train_rows(pack_paths, action_view=model.config.action_view)
    vectors = _embed_after_states(rows, model=model, runtime=runtime, device=selected_device)
    entries = tuple(_entry_from_row(row) for row in rows)
    config_payload = {
        "schema_version": INDEX_BUILD_RESULT_SCHEMA_VERSION,
        "checkpoint": _display_path(checkpoint_path),
        "data": _display_path(pack_paths.root),
        "out": _display_path(out_dir),
        "device": str(selected_device),
        "distance": distance,
        "name": name,
        "indexed_splits": ["train"],
        "action_view": model.config.action_view,
    }
    parent_artifacts = (training_artifact.artifact_id, dataset_artifact.artifact_id)
    index = build_transition_index(
        name=name,
        entries=entries,
        vectors=vectors,
        distance=distance,
        metadata={
            "schema_version": TRANSITION_INDEX_SCHEMA_VERSION,
            "dataset_artifact_id": dataset_artifact.artifact_id,
            "training_artifact_id": training_artifact.artifact_id,
            "checkpoint_sha256": checkpoint_manifest.checkpoint_sha256,
            "checkpoint_step": _optional_int(checkpoint_payload.get("step"), "checkpoint.step"),
            "checkpoint_action_view": model.config.action_view,
            "indexed_splits": ("train",),
            "source_row_count": len(rows),
        },
    )
    manifest, manifest_path = write_transition_index(
        index,
        out_dir,
        command=command,
        config=config_payload,
        parent_artifacts=parent_artifacts,
        source_git_sha=source_git_sha,
        created_at=created_at,
    )
    return IndexBuildResult(
        artifact_manifest_id=manifest.artifact_id,
        artifact_manifest_path=manifest_path.name,
        index_path="index.json",
        vectors_path="vectors.npy",
        entries_path="entries.jsonl",
        parent_artifacts=parent_artifacts,
        count=index.count,
        dim=index.dim,
        distance=index.distance,
        metadata={
            "action_view": model.config.action_view,
            "indexed_splits": ("train",),
            "checkpoint_step": index.metadata.get("checkpoint_step"),
            "dataset_artifact_id": dataset_artifact.artifact_id,
            "training_artifact_id": training_artifact.artifact_id,
        },
    )


def _load_train_rows(pack_paths: _PackPaths, *, action_view: str) -> tuple[_EvalRow, ...]:
    rows = _load_split_rows(
        pack_paths,
        split="train",
        action_view=action_view,
        vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
    )
    if not rows:
        raise TransitionIndexError("packed dataset has no train rows for index build")
    return rows


def _embed_after_states(rows: tuple[_EvalRow, ...], *, model: Any, runtime: Any, device: Any) -> np.ndarray:
    state_after = _state_batch(tuple(row.state_after for row in rows), runtime=runtime, device=device)
    was_training = bool(model.training)
    model.eval()
    with runtime.no_grad():
        z_after = model.encode_state(state_after).float().detach().cpu().numpy()
    if was_training:
        model.train()
    if not np.isfinite(z_after).all():
        raise TransitionIndexError("index vectors must be finite")
    return np.asarray(z_after, dtype=np.float32)


def _entry_from_row(row: _EvalRow) -> TransitionIndexEntry:
    return TransitionIndexEntry(
        transition_id=row.transition_id,
        split=row.split,
        source=row.source,
        repo=row.repo,
        path=row.path,
        edit_size=row.edit_size,
        metadata={
            "commit": str(row.metadata.get("commit", "")),
            "action_cluster": str(row.metadata.get("action_cluster", "")),
            "token_count_before": int(row.metadata.get("token_count_before", 0) or 0),
            "token_count_after": int(row.metadata.get("token_count_after", 0) or 0),
        },
    )


def _reject_existing_index_outputs(out_dir: Path, *, overwrite: bool) -> None:
    for path in (
        out_dir / "vectors.npy",
        out_dir / "entries.jsonl",
        out_dir / "index.json",
        out_dir / "manifest.json",
    ):
        if path.exists() and not overwrite:
            raise TransitionIndexError(f"output already exists; pass --overwrite to replace: {path}")


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TransitionIndexError(f"{name} must be an integer")
    result = int(value)
    if result != value or result < 0:
        raise TransitionIndexError(f"{name} must be a non-negative integer")
    return result


__all__ = [
    "INDEX_BUILD_RESULT_SCHEMA_VERSION",
    "IndexBuildResult",
    "build_transition_index_artifact",
]
