"""Manifest-backed training runner for CodeLeWM."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codelewm.data import SourceUnavailableError
from codelewm.observability import (
    ArtifactManifest,
    ManifestFile,
    build_artifact_manifest,
    build_manifest_file,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)

from .config import TrainConfig, load_train_config, validate_train_config


TRAINING_RUN_MANIFEST_SCHEMA_VERSION = "codelewm.training_run.v1"
TRAINING_METRICS_SCHEMA_VERSION = "codelewm.training_metrics.v1"

TrainingExecutor = Callable[["TrainingRunContext"], "TrainingExecutorResult"]


class TrainingRunError(RuntimeError):
    """Raised when a training run cannot be completed safely."""


@dataclass(frozen=True)
class TrainingRunContext:
    """Filesystem and config context passed to a concrete training executor."""

    config: TrainConfig
    root: Path
    run_dir: Path
    checkpoint_dir: Path
    metrics_path: Path
    config_path: Path
    dataset_manifest_path: Path
    parent_dataset_manifest: ArtifactManifest


@dataclass(frozen=True)
class TrainingExecutorResult:
    """Artifacts and final metrics produced by a concrete training executor."""

    step_count: int
    metrics: Mapping[str, float]
    checkpoint_paths: tuple[Path, ...]
    report_paths: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_count < 0:
            raise TrainingRunError("executor step_count must be non-negative")
        if not self.checkpoint_paths:
            raise TrainingRunError("executor must produce at least one checkpoint file")
        _validate_metrics(self.metrics)
        _ensure_json_native(self.metadata, field_name="executor metadata")


@dataclass(frozen=True)
class TrainingRunManifest:
    """CodeLeWM training run manifest returned by the package runner."""

    run_id: str
    config_sha256: str
    artifact_manifest_id: str
    artifact_manifest_path: str
    parent_artifacts: tuple[str, ...]
    dataset_manifest_path: str
    config_path: str
    metrics_path: str
    metrics_report_path: str
    checkpoint_files: tuple[ManifestFile, ...]
    report_files: tuple[ManifestFile, ...]
    final_metrics: Mapping[str, float]
    step_count: int
    seed: int
    schema_version: str = TRAINING_RUN_MANIFEST_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_RUN_MANIFEST_SCHEMA_VERSION:
            raise TrainingRunError(
                "schema_version must be "
                f"{TRAINING_RUN_MANIFEST_SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if not self.run_id:
            raise TrainingRunError("run_id must not be empty")
        if not self.artifact_manifest_id:
            raise TrainingRunError("artifact_manifest_id must not be empty")
        if not self.parent_artifacts:
            raise TrainingRunError("parent_artifacts must include the dataset artifact id")
        if self.step_count < 0:
            raise TrainingRunError("step_count must be non-negative")
        _validate_metrics(self.final_metrics)
        _ensure_json_native(self.metadata, field_name="training run metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "parent_artifacts": list(self.parent_artifacts),
            "dataset_manifest_path": self.dataset_manifest_path,
            "config_path": self.config_path,
            "metrics_path": self.metrics_path,
            "metrics_report_path": self.metrics_report_path,
            "checkpoint_files": [file.to_dict() for file in self.checkpoint_files],
            "report_files": [file.to_dict() for file in self.report_files],
            "final_metrics": dict(self.final_metrics),
            "step_count": self.step_count,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingRunManifest":
        _reject_unknown(
            payload,
            {
                "schema_version",
                "run_id",
                "config_sha256",
                "artifact_manifest_id",
                "artifact_manifest_path",
                "parent_artifacts",
                "dataset_manifest_path",
                "config_path",
                "metrics_path",
                "metrics_report_path",
                "checkpoint_files",
                "report_files",
                "final_metrics",
                "step_count",
                "seed",
                "metadata",
            },
            "training run manifest",
        )
        return cls(
            schema_version=_require_string(payload, "schema_version", "training run manifest"),
            run_id=_require_string(payload, "run_id", "training run manifest"),
            config_sha256=_require_string(payload, "config_sha256", "training run manifest"),
            artifact_manifest_id=_require_string(
                payload,
                "artifact_manifest_id",
                "training run manifest",
            ),
            artifact_manifest_path=_require_string(
                payload,
                "artifact_manifest_path",
                "training run manifest",
            ),
            parent_artifacts=tuple(_require_string_items(payload, "parent_artifacts")),
            dataset_manifest_path=_require_string(
                payload,
                "dataset_manifest_path",
                "training run manifest",
            ),
            config_path=_require_string(payload, "config_path", "training run manifest"),
            metrics_path=_require_string(payload, "metrics_path", "training run manifest"),
            metrics_report_path=_require_string(
                payload,
                "metrics_report_path",
                "training run manifest",
            ),
            checkpoint_files=tuple(
                ManifestFile.from_dict(_require_mapping_item(item, "checkpoint_files"))
                for item in _require_sequence(payload, "checkpoint_files")
            ),
            report_files=tuple(
                ManifestFile.from_dict(_require_mapping_item(item, "report_files"))
                for item in _require_sequence(payload, "report_files")
            ),
            final_metrics={
                str(key): _require_float(value, f"final_metrics.{key}")
                for key, value in _require_mapping(payload, "final_metrics").items()
            },
            step_count=_require_int(payload, "step_count", "training run manifest"),
            seed=_require_int(payload, "seed", "training run manifest"),
            metadata=dict(_require_mapping(payload, "metadata")),
        )


def train(
    config: TrainConfig | Mapping[str, Any] | Path | str,
    *,
    executor: TrainingExecutor,
    root: Path | str = ".",
    command: Sequence[str] = ("codelewm", "train"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
    overwrite: bool = False,
) -> TrainingRunManifest:
    """Run a manifest-backed training job and return its run manifest.

    The concrete model step is supplied by ``executor``. This keeps the artifact
    and lineage contract independent from the CPU/GPU training implementation
    that lands in a separate issue.
    """

    root_path = Path(root).resolve()
    cfg = _coerce_config(config, root=root_path)
    paths = _resolve_output_paths(cfg, root=root_path)
    if not overwrite:
        _reject_existing_outputs(paths)

    dataset_manifest_path = _resolve_required_file(cfg.data.manifest, root=root_path, field="data.manifest")
    _resolve_required_file(cfg.data.train, root=root_path, field="data.train")
    _resolve_required_file(cfg.data.val, root=root_path, field="data.val")
    parent_dataset_manifest = read_artifact_manifest(dataset_manifest_path)
    validate_artifact_checksums(parent_dataset_manifest, root=dataset_manifest_path.parent)

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    paths.training_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    _write_json(cfg.to_dict(), paths.config_path)

    context = TrainingRunContext(
        config=cfg,
        root=root_path,
        run_dir=paths.run_dir,
        checkpoint_dir=paths.checkpoint_dir,
        metrics_path=paths.metrics_path,
        config_path=paths.config_path,
        dataset_manifest_path=dataset_manifest_path,
        parent_dataset_manifest=parent_dataset_manifest,
    )
    result = executor(context)
    if not isinstance(result, TrainingExecutorResult):
        raise TrainingRunError("executor must return TrainingExecutorResult")

    _write_metrics_jsonl(result.metrics, path=paths.metrics_path, run_id=cfg.name, step=result.step_count)
    metrics_report_path = paths.run_dir / "reports" / "metrics_report.json"
    metrics_report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        {
            "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
            "run_id": cfg.name,
            "step_count": result.step_count,
            "final_metrics": dict(result.metrics),
        },
        metrics_report_path,
    )

    checkpoint_files = tuple(build_manifest_file(path, root=paths.run_dir) for path in result.checkpoint_paths)
    report_files = tuple(build_manifest_file(path, root=paths.run_dir) for path in result.report_paths)
    artifact_manifest = build_artifact_manifest(
        artifact_kind="training_run",
        root=paths.run_dir,
        files=(
            paths.config_path,
            paths.metrics_path,
            metrics_report_path,
            *result.checkpoint_paths,
            *result.report_paths,
        ),
        command=command,
        config=cfg.to_dict(),
        parent_artifacts=(parent_dataset_manifest.artifact_id,),
        source_git_sha=source_git_sha,
        created_at=created_at,
        metadata={
            "schema_version": TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": cfg.name,
            "seed": cfg.seed,
            "step_count": result.step_count,
            "final_metrics": dict(result.metrics),
            "dataset_manifest_path": _relative_to_root(dataset_manifest_path, root_path),
            "executor": dict(result.metadata),
        },
    )
    artifact_manifest_path = paths.run_dir / "manifest.json"
    write_artifact_manifest(artifact_manifest, artifact_manifest_path)

    manifest = TrainingRunManifest(
        run_id=cfg.name,
        config_sha256=artifact_manifest.config_sha256,
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path=_relative_to_root(artifact_manifest_path, paths.run_dir),
        parent_artifacts=(parent_dataset_manifest.artifact_id,),
        dataset_manifest_path=_relative_to_root(dataset_manifest_path, root_path),
        config_path=_relative_to_root(paths.config_path, paths.run_dir),
        metrics_path=_relative_to_root(paths.metrics_path, paths.run_dir),
        metrics_report_path=_relative_to_root(metrics_report_path, paths.run_dir),
        checkpoint_files=checkpoint_files,
        report_files=report_files,
        final_metrics=dict(result.metrics),
        step_count=result.step_count,
        seed=cfg.seed,
        metadata={"executor": dict(result.metadata)},
    )
    _write_json(manifest.to_dict(), paths.training_manifest_path)
    return manifest


def read_training_run_manifest(path: Path | str) -> TrainingRunManifest:
    """Read and validate a training run manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TrainingRunError("training run manifest must be a JSON object")
    return TrainingRunManifest.from_dict(payload)


@dataclass(frozen=True)
class _OutputPaths:
    run_dir: Path
    checkpoint_dir: Path
    metrics_path: Path
    training_manifest_path: Path
    config_path: Path


def _coerce_config(config: TrainConfig | Mapping[str, Any] | Path | str, *, root: Path) -> TrainConfig:
    if isinstance(config, TrainConfig):
        return validate_train_config(config)
    if isinstance(config, Mapping):
        return validate_train_config(config)
    path = Path(config)
    if not path.is_absolute():
        path = root / path
    return load_train_config(path)


def _resolve_output_paths(config: TrainConfig, *, root: Path) -> _OutputPaths:
    run_dir = _resolve_config_path(config.output.run_dir, root=root)
    checkpoint_dir = _resolve_config_path(config.output.checkpoint_dir, root=root)
    metrics_path = _resolve_config_path(config.output.metrics_path, root=root)
    training_manifest_path = _resolve_config_path(config.output.manifest_path, root=root)
    config_path = run_dir / "config.json"
    for path, field in (
        (checkpoint_dir, "output.checkpoint_dir"),
        (metrics_path, "output.metrics_path"),
        (training_manifest_path, "output.manifest_path"),
        (config_path, "output.run_dir"),
    ):
        _require_under(path, root=run_dir, field=field)
    return _OutputPaths(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        metrics_path=metrics_path,
        training_manifest_path=training_manifest_path,
        config_path=config_path,
    )


def _reject_existing_outputs(paths: _OutputPaths) -> None:
    for path in (
        paths.config_path,
        paths.metrics_path,
        paths.training_manifest_path,
        paths.run_dir / "manifest.json",
    ):
        if path.exists():
            raise TrainingRunError(f"output already exists; pass overwrite=True to replace: {path}")


def _resolve_required_file(value: str | None, *, root: Path, field: str) -> Path:
    if value is None:
        raise SourceUnavailableError(f"{field} is required for manifest-backed training")
    path = _resolve_config_path(value, root=root)
    if not path.is_file():
        raise SourceUnavailableError(f"{field} does not exist: {path}")
    return path


def _resolve_config_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_under(path: Path, *, root: Path, field: str) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise TrainingRunError(f"{field} must stay under output.run_dir") from exc


def _write_metrics_jsonl(metrics: Mapping[str, float], *, path: Path, run_id: str, step: int) -> None:
    payload = {
        "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
        "run_id": run_id,
        "step": step,
        "metrics": dict(metrics),
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    _ensure_json_native(payload, field_name=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _validate_metrics(metrics: Mapping[str, float]) -> None:
    if not metrics:
        raise TrainingRunError("metrics must not be empty")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise TrainingRunError("metric names must be non-empty strings")
        _require_float(value, key)


def _require_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingRunError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise TrainingRunError(f"{field} must be finite")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise TrainingRunError(f"{section} contains unknown key(s): {', '.join(unknown)}")


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    if key not in payload:
        raise TrainingRunError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise TrainingRunError(f"{section}.{key} must be a string")
    return value


def _require_int(payload: Mapping[str, Any], key: str, section: str) -> int:
    if key not in payload:
        raise TrainingRunError(f"{section}.{key} is required")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingRunError(f"{section}.{key} must be an integer")
    return value


def _require_sequence(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    if key not in payload:
        raise TrainingRunError(f"{key} is required")
    value = payload[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TrainingRunError(f"{key} must be a JSON array")
    return value


def _require_string_items(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = _require_sequence(payload, key)
    for value in values:
        if not isinstance(value, str):
            raise TrainingRunError(f"{key} must contain only strings")
    return tuple(values)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in payload:
        raise TrainingRunError(f"{key} is required")
    value = payload[key]
    if not isinstance(value, Mapping):
        raise TrainingRunError(f"{key} must be a JSON object")
    return value


def _require_mapping_item(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingRunError(f"{section} entries must be JSON objects")
    return value


def _ensure_json_native(payload: Any, *, field_name: str) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TrainingRunError(f"{field_name} must be JSON-native: {exc}") from exc
