"""Dependency-light CPU smoke training executor."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.data import OptionalDependencyError

from .runner import (
    TrainingExecutorResult,
    TrainingRunContext,
    TrainingRunError,
    train,
)


CPU_SMOKE_CHECKPOINT_SCHEMA_VERSION = "codelewm.cpu_smoke_checkpoint.v1"
CPU_SMOKE_REPORT_SCHEMA_VERSION = "codelewm.cpu_smoke_report.v1"


def train_cpu_smoke(
    config: Any,
    *,
    root: Path | str = ".",
    source_git_sha: str | None = None,
    created_at: str | None = None,
    overwrite: bool = False,
):
    """Run the manifest-backed runner with the CPU smoke executor."""

    return train(
        config,
        executor=cpu_smoke_training_executor,
        root=root,
        command=("codelewm", "train", "--cpu-smoke"),
        source_git_sha=source_git_sha,
        created_at=created_at,
        overwrite=overwrite,
    )


def cpu_smoke_training_executor(context: TrainingRunContext) -> TrainingExecutorResult:
    """Run a bounded NumPy training smoke over a packed HDF5 transition file."""

    batch = _load_hdf5_batch(Path(context.config.data.train), action_view=context.config.wm.action_view)
    state_before = _sequence_features(batch.state_before)
    state_after = _sequence_features(batch.state_after)
    action = _sequence_features(batch.action)
    embeddings = np.concatenate((state_before, action), axis=1)
    targets = state_after
    embedding_variance = float(np.var(embeddings, axis=0).mean())
    if not math.isfinite(embedding_variance) or embedding_variance <= 0.0:
        raise TrainingRunError("CPU smoke embeddings must have nonzero finite variance")

    max_steps = max(1, min(int(context.config.trainer.max_steps), 8))
    lr = min(float(context.config.optimizer.lr), 1e-3)
    weights = np.zeros((embeddings.shape[1], targets.shape[1]), dtype=np.float64)
    loss = math.inf
    for _ in range(max_steps):
        predictions = embeddings @ weights
        error = predictions - targets
        loss = float(np.mean(np.square(error)))
        if not math.isfinite(loss):
            raise TrainingRunError("CPU smoke loss became non-finite")
        gradient = (embeddings.T @ error) / max(1, embeddings.shape[0])
        weights -= lr * gradient

    final_predictions = embeddings @ weights
    final_loss = float(np.mean(np.square(final_predictions - targets)))
    if not math.isfinite(final_loss):
        raise TrainingRunError("CPU smoke final loss is non-finite")

    checkpoint_path = context.checkpoint_dir / "cpu_smoke_checkpoint.json"
    checkpoint_payload = {
        "schema_version": CPU_SMOKE_CHECKPOINT_SCHEMA_VERSION,
        "run_id": context.config.name,
        "steps": max_steps,
        "weights": weights.tolist(),
        "feature_dim": embeddings.shape[1],
        "target_dim": targets.shape[1],
    }
    _write_json(checkpoint_payload, checkpoint_path)

    report_path = context.run_dir / "reports" / "cpu_smoke_report.json"
    metrics = {
        "loss/total": final_loss,
        "loss/prediction_mse": final_loss,
        "embedding/variance": embedding_variance,
        "train/examples": float(embeddings.shape[0]),
    }
    report_payload = {
        "schema_version": CPU_SMOKE_REPORT_SCHEMA_VERSION,
        "run_id": context.config.name,
        "steps": max_steps,
        "metrics": metrics,
    }
    _write_json(report_payload, report_path)

    return TrainingExecutorResult(
        step_count=max_steps,
        metrics=metrics,
        checkpoint_paths=(checkpoint_path,),
        report_paths=(report_path,),
        metadata={
            "executor": "cpu_smoke",
            "device": "cpu",
            "dtype": "float64",
            "examples": int(embeddings.shape[0]),
        },
    )


class _SmokeBatch:
    def __init__(self, *, state_before: np.ndarray, state_after: np.ndarray, action: np.ndarray) -> None:
        self.state_before = state_before
        self.state_after = state_after
        self.action = action


def _load_hdf5_batch(path: Path, *, action_view: str) -> _SmokeBatch:
    h5py = _require_h5py()
    action_group = "action_text" if action_view == "text" else "action_abs"
    with h5py.File(path, "r") as handle:
        state_before = _read_matrix(handle, "state_before/input_ids")
        state_after = _read_matrix(handle, "state_after/input_ids")
        action = _read_matrix(handle, f"{action_group}/input_ids")
    if state_before.shape[0] == 0:
        raise TrainingRunError("CPU smoke HDF5 train split must contain at least one row")
    if state_before.shape[0] != state_after.shape[0] or state_before.shape[0] != action.shape[0]:
        raise TrainingRunError("CPU smoke HDF5 arrays must have matching row counts")
    return _SmokeBatch(state_before=state_before, state_after=state_after, action=action)


def _read_matrix(handle: Any, key: str) -> np.ndarray:
    if key not in handle:
        raise TrainingRunError(f"CPU smoke HDF5 is missing dataset {key!r}")
    matrix = np.asarray(handle[key], dtype=np.float64)
    if matrix.ndim != 2:
        raise TrainingRunError(f"CPU smoke HDF5 dataset {key!r} must be rank 2")
    if not np.isfinite(matrix).all():
        raise TrainingRunError(f"CPU smoke HDF5 dataset {key!r} contains non-finite values")
    return matrix


def _sequence_features(values: np.ndarray) -> np.ndarray:
    active = values != 0.0
    counts = active.sum(axis=1, keepdims=True).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    sums = values.sum(axis=1, keepdims=True)
    means = sums / safe_counts
    centered = np.where(active, values - means, 0.0)
    variances = np.square(centered).sum(axis=1, keepdims=True) / safe_counts
    first = values[:, :1]
    last = np.take_along_axis(values, np.maximum(counts.astype(int) - 1, 0), axis=1)
    return np.concatenate((means, np.sqrt(variances), counts, first, last), axis=1) / 1000.0


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _require_h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("CPU smoke training requires h5py; install codelewm[data]") from exc
    return h5py
