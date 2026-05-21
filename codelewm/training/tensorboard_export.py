"""TensorBoard-compatible training export helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.data import OptionalDependencyError
from codelewm.observability import ManifestFile, build_manifest_file


TENSORBOARD_EXPORT_SCHEMA_VERSION = "codelewm.training.tensorboard_export.v1"
DEFAULT_TENSORBOARD_MAX_HISTOGRAM_TENSORS = 8
DEFAULT_TENSORBOARD_MAX_HISTOGRAM_VALUES = 4096


class TensorBoardExportError(RuntimeError):
    """Raised when TensorBoard-compatible export cannot be written safely."""


@dataclass(frozen=True)
class TensorBoardExportResult:
    """Paths and tags written by a TensorBoard-compatible export."""

    report_path: Path
    event_files: tuple[Path, ...]
    scalar_tags: tuple[str, ...]
    histogram_tags: tuple[str, ...]
    summary_tags: tuple[str, ...] = ()

    def to_metadata(self, *, root: Path | str) -> dict[str, Any]:
        root_path = Path(root)
        return {
            "enabled": True,
            "schema_version": TENSORBOARD_EXPORT_SCHEMA_VERSION,
            "report_path": _relative_to_root(self.report_path, root_path),
            "event_files": [_manifest_file_payload(path, root=root_path) for path in self.event_files],
            "scalar_tags": list(self.scalar_tags),
            "histogram_tags": list(self.histogram_tags),
            "summary_tags": list(self.summary_tags),
        }


TensorBoardWriterFactory = Callable[[Path], Any]


def export_tensorboard_training_run(
    *,
    run_id: str,
    run_dir: Path | str,
    step_count: int,
    metrics: Mapping[str, float],
    model: Any | None = None,
    embeddings: Any | None = None,
    checkpoint_path: Path | str | None = None,
    checkpoint_manifest_path: Path | str | None = None,
    log_dir: Path | str | None = None,
    writer_factory: TensorBoardWriterFactory | None = None,
    max_histogram_tensors: int = DEFAULT_TENSORBOARD_MAX_HISTOGRAM_TENSORS,
    max_histogram_values: int = DEFAULT_TENSORBOARD_MAX_HISTOGRAM_VALUES,
) -> TensorBoardExportResult:
    """Write TensorBoard-compatible event files plus a manifestable metadata report.

    The event file is a diagnostic surface only. JSONL metrics, JSON reports,
    checkpoint manifests, and artifact manifests remain the release contract.
    """

    if not run_id:
        raise TensorBoardExportError("run_id must not be empty")
    if step_count < 0:
        raise TensorBoardExportError("step_count must be non-negative")
    if max_histogram_tensors < 0:
        raise TensorBoardExportError("max_histogram_tensors must be non-negative")
    if max_histogram_values <= 0:
        raise TensorBoardExportError("max_histogram_values must be positive")

    run_root = Path(run_dir).resolve()
    event_dir = _resolve_log_dir(log_dir, run_dir=run_root)
    event_dir.mkdir(parents=True, exist_ok=True)

    writer = _make_summary_writer(event_dir, writer_factory=writer_factory)
    scalar_tags: list[str] = []
    histogram_tags: list[str] = []
    summary_tags: list[str] = []
    try:
        scalar_tags.extend(_write_scalars(writer, metrics, step_count=step_count))
        latent_scalars, latent_histogram = _write_latent_summaries(
            writer,
            embeddings,
            step_count=step_count,
            max_values=max_histogram_values,
        )
        scalar_tags.extend(latent_scalars)
        histogram_tags.extend(latent_histogram)
        histogram_tags.extend(
            _write_parameter_histograms(
                writer,
                model,
                step_count=step_count,
                max_tensors=max_histogram_tensors,
                max_values=max_histogram_values,
            )
        )
        _call_writer(writer, "flush")
    finally:
        _call_writer(writer, "close")

    event_files = tuple(sorted(path for path in event_dir.iterdir() if path.is_file()))
    if not event_files:
        raise TensorBoardExportError(f"TensorBoard export produced no event files under {event_dir}")

    report_path = run_root / "reports" / "tensorboard_export.json"
    report_payload = {
        "schema_version": TENSORBOARD_EXPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "step_count": step_count,
        "log_dir": _relative_to_root(event_dir, run_root),
        "event_files": [_manifest_file_payload(path, root=run_root) for path in event_files],
        "scalar_tags": scalar_tags,
        "histogram_tags": histogram_tags,
        "summary_tags": summary_tags,
        "checkpoint": {
            "checkpoint_path": None
            if checkpoint_path is None
            else _relative_to_root(Path(checkpoint_path), run_root),
            "checkpoint_manifest_path": None
            if checkpoint_manifest_path is None
            else _relative_to_root(Path(checkpoint_manifest_path), run_root),
        },
        "safety_limits": {
            "max_histogram_tensors": max_histogram_tensors,
            "max_histogram_values": max_histogram_values,
            "raw_checkpoint_serialized": False,
            "candidate_code_serialized": False,
        },
        "writer": {
            "class": writer.__class__.__name__,
            "module": writer.__class__.__module__,
        },
    }
    _write_json(report_payload, report_path)
    return TensorBoardExportResult(
        report_path=report_path,
        event_files=event_files,
        scalar_tags=tuple(scalar_tags),
        histogram_tags=tuple(histogram_tags),
        summary_tags=tuple(summary_tags),
    )


def _write_scalars(writer: Any, metrics: Mapping[str, float], *, step_count: int) -> tuple[str, ...]:
    tags: list[str] = []
    for tag, value in sorted(metrics.items()):
        scalar = _finite_float(value, field=tag)
        writer.add_scalar(tag, scalar, step_count)
        tags.append(tag)
    return tuple(tags)


def _write_latent_summaries(
    writer: Any,
    embeddings: Any | None,
    *,
    step_count: int,
    max_values: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    values = _safe_array(embeddings, max_values=max_values)
    if values is None:
        return (), ()
    scalar_values = {
        "latents/export_mean": float(values.mean()),
        "latents/export_std": float(values.std()),
        "latents/export_min": float(values.min()),
        "latents/export_max": float(values.max()),
    }
    scalar_tags = []
    for tag, scalar in scalar_values.items():
        writer.add_scalar(tag, scalar, step_count)
        scalar_tags.append(tag)
    histogram_tag = "latents/last_embedding_values"
    writer.add_histogram(histogram_tag, values, step_count)
    return tuple(scalar_tags), (histogram_tag,)


def _write_parameter_histograms(
    writer: Any,
    model: Any | None,
    *,
    step_count: int,
    max_tensors: int,
    max_values: int,
) -> tuple[str, ...]:
    if model is None or max_tensors == 0 or not hasattr(model, "named_parameters"):
        return ()

    tags: list[str] = []
    for index, (name, value) in enumerate(model.named_parameters()):
        if index >= max_tensors:
            break
        values = _safe_array(value, max_values=max_values)
        if values is None:
            continue
        tag = f"parameters/{_sanitize_tag(str(name))}"
        writer.add_histogram(tag, values, step_count)
        tags.append(tag)
    return tuple(tags)


def _make_summary_writer(
    log_dir: Path,
    *,
    writer_factory: TensorBoardWriterFactory | None,
) -> Any:
    if writer_factory is not None:
        return writer_factory(log_dir)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
        raise OptionalDependencyError(
            "TensorBoard export requires the optional observability group; "
            "install it with `uv sync --group dev --group train --group data --group observability`"
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))


def _resolve_log_dir(log_dir: Path | str | None, *, run_dir: Path) -> Path:
    if log_dir is None:
        path = run_dir / "tensorboard"
    else:
        path = Path(log_dir)
        if not path.is_absolute():
            path = run_dir / path
    resolved = path.resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise TensorBoardExportError("TensorBoard log_dir must stay under the training run directory") from exc
    return resolved


def _safe_array(value: Any | None, *, max_values: int) -> np.ndarray | None:
    if value is None:
        return None
    candidate = value
    for method_name in ("detach", "float", "cpu"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            candidate = method()
    if hasattr(candidate, "reshape"):
        try:
            candidate = candidate.reshape(-1)
        except TypeError:
            candidate = np.asarray(candidate).reshape(-1)
    try:
        if hasattr(candidate, "numel") and int(candidate.numel()) == 0:
            return None
        if hasattr(candidate, "numpy"):
            values = candidate.numpy()
        else:
            values = np.asarray(candidate)
    except (TypeError, ValueError):
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    if array.size > max_values:
        array = array[:max_values]
    return array


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TensorBoardExportError(f"{field} must be numeric")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TensorBoardExportError(f"{field} must be finite")
    return scalar


def _manifest_file_payload(path: Path, *, root: Path) -> dict[str, Any]:
    entry: ManifestFile = build_manifest_file(path, root=root)
    return entry.to_dict()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sanitize_tag(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._/-" else "_" for character in value)
    return cleaned.strip("/") or "unnamed"


def _call_writer(writer: Any, method_name: str) -> None:
    method = getattr(writer, method_name, None)
    if callable(method):
        method()


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    _ensure_json_native(payload, field_name=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_json_native(payload: Any, *, field_name: str) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TensorBoardExportError(f"{field_name} must be JSON-native: {exc}") from exc
