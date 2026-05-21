"""Safe checkpoint inspection reports for CodeLeWM models."""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codelewm.model.checkpoint import (
    CheckpointManifest,
    sha256_file,
)
from codelewm.observability import (
    ArtifactManifestError,
    build_artifact_manifest,
    compute_json_sha256,
    read_artifact_manifest,
    redact_text,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security import CheckpointTrustError, require_trusted_checkpoint


MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION = "codelewm.model_checkpoint_inspection.v1"
MODEL_CHECKPOINT_INSPECTION_RUN_SCHEMA_VERSION = "codelewm.model_checkpoint_inspection_run.v1"
DEFAULT_HISTOGRAM_BINS = 16
DEFAULT_MAX_HISTOGRAM_TENSORS = 24
DEFAULT_MAX_HISTOGRAM_VALUES = 8192


class CheckpointInspectionError(RuntimeError):
    """Raised when a checkpoint inspection report cannot be written safely."""


@dataclass(frozen=True)
class CheckpointInspectionResult:
    """Manifest-backed checkpoint inspection run result."""

    output_dir: Path
    report_path: Path
    artifact_manifest_id: str
    artifact_manifest_path: str
    parent_artifacts: tuple[str, ...]
    tensor_count: int
    module_count: int
    parameter_count: int
    schema_version: str = MODEL_CHECKPOINT_INSPECTION_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": _relative_to_root(self.report_path, self.output_dir),
            "parent_artifacts": list(self.parent_artifacts),
            "tensor_count": self.tensor_count,
            "module_count": self.module_count,
            "parameter_count": self.parameter_count,
        }


def inspect_checkpoint(
    *,
    checkpoint: Path | str,
    out: Path | str,
    command: Sequence[str],
    checkpoint_manifest: Path | str | None = None,
    parent_manifests: Sequence[Path | str] = (),
    allow_unsafe_checkpoint: bool = False,
    overwrite: bool = False,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    max_histogram_tensors: int = DEFAULT_MAX_HISTOGRAM_TENSORS,
    max_histogram_values: int = DEFAULT_MAX_HISTOGRAM_VALUES,
) -> CheckpointInspectionResult:
    """Inspect a checkpoint without serializing raw tensor values into reports."""

    if histogram_bins <= 0:
        raise CheckpointInspectionError("histogram_bins must be positive")
    if max_histogram_tensors < 0:
        raise CheckpointInspectionError("max_histogram_tensors must be non-negative")
    if max_histogram_values <= 0:
        raise CheckpointInspectionError("max_histogram_values must be positive")
    if not command:
        raise CheckpointInspectionError("command must not be empty")

    checkpoint_path = Path(checkpoint)
    output_dir = Path(out).resolve()
    report_path = output_dir / "reports" / "model_checkpoint_inspection.json"
    artifact_manifest_path = output_dir / "manifest.json"
    if not overwrite and (report_path.exists() or artifact_manifest_path.exists()):
        raise CheckpointInspectionError(
            f"checkpoint inspection output already exists under {output_dir}; pass --overwrite to replace it"
        )

    parent_artifact_ids = _validate_parent_manifests(parent_manifests)
    trusted_manifest: CheckpointManifest | None = None
    trust_payload: dict[str, Any]
    if allow_unsafe_checkpoint:
        trust_payload = {
            "trusted": False,
            "allow_unsafe_checkpoint": True,
            "status": "unsafe_override_selected",
            "checkpoint_manifest_path": None,
            "checkpoint_manifest_sha256": None,
        }
    else:
        trusted_manifest = require_trusted_checkpoint(
            checkpoint_path,
            manifest_path=checkpoint_manifest,
        )
        resolved_manifest_path = (
            Path(checkpoint_manifest)
            if checkpoint_manifest is not None
            else checkpoint_path.with_name(checkpoint_path.name + ".manifest.json")
        )
        trust_payload = {
            "trusted": True,
            "allow_unsafe_checkpoint": False,
            "status": "checkpoint_manifest_verified",
            "checkpoint_manifest_path": redact_text(str(resolved_manifest_path)),
            "checkpoint_manifest_sha256": sha256_file(resolved_manifest_path),
        }

    runtime = _require_torch()
    payload = _load_torch_checkpoint(checkpoint_path, runtime=runtime)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise CheckpointInspectionError("checkpoint payload must contain a model_state_dict mapping")

    tensor_entries, module_entries, summary = _inspect_state_dict(
        model_state,
        runtime=runtime,
        histogram_bins=histogram_bins,
        max_histogram_tensors=max_histogram_tensors,
        max_histogram_values=max_histogram_values,
    )
    checkpoint_sha256 = (
        trusted_manifest.checkpoint_sha256
        if trusted_manifest is not None
        else sha256_file(checkpoint_path)
    )
    checkpoint_manifest_payload = (
        None if trusted_manifest is None else trusted_manifest.to_dict()
    )
    report = {
        "schema_version": MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION,
        "checkpoint": {
            "checkpoint_path": redact_text(str(checkpoint_path)),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_manifest": checkpoint_manifest_payload,
            "trust_gate": trust_payload,
        },
        "payload": _payload_summary(payload),
        "compatibility": {
            "checkpoint_manifest_metadata": None
            if trusted_manifest is None
            else trusted_manifest.metadata.to_dict(),
            "compatibility_config_hash": _optional_string(payload.get("compatibility_config_hash")),
            "compatibility_config": _json_native_mapping(payload.get("compatibility_config")),
        },
        "summary": summary,
        "modules": module_entries,
        "tensors": tensor_entries,
        "histogram_policy": {
            "histogram_bins": histogram_bins,
            "max_histogram_tensors": max_histogram_tensors,
            "max_histogram_values": max_histogram_values,
            "raw_tensor_values_serialized": False,
            "optimizer_state_serialized": False,
        },
        "claim_boundary": {
            "allowed": False,
            "reason": "checkpoint_inspection_is_diagnostic_only",
            "blocked_claims": [
                "semantic_latent_axes",
                "action_conditioned_quality",
                "downstream_coding_usefulness",
            ],
        },
    }
    _write_json(report, report_path)
    artifact_manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=output_dir,
        files=(report_path,),
        command=command,
        config={
            "schema_version": MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_manifest_sha256": trust_payload["checkpoint_manifest_sha256"],
            "histogram_bins": histogram_bins,
            "max_histogram_tensors": max_histogram_tensors,
            "max_histogram_values": max_histogram_values,
        },
        parent_artifacts=parent_artifact_ids,
        metadata={
            "report_schema_version": MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION,
            "report_path": _relative_to_root(report_path, output_dir),
            "checkpoint_sha256": checkpoint_sha256,
            "tensor_count": summary["tensor_count"],
            "module_count": summary["module_count"],
            "parameter_count": summary["parameter_count"],
            "trusted_checkpoint": trust_payload["trusted"],
        },
    )
    write_artifact_manifest(artifact_manifest, artifact_manifest_path)
    return CheckpointInspectionResult(
        output_dir=output_dir,
        report_path=report_path,
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path=artifact_manifest_path.name,
        parent_artifacts=tuple(parent_artifact_ids),
        tensor_count=int(summary["tensor_count"]),
        module_count=int(summary["module_count"]),
        parameter_count=int(summary["parameter_count"]),
    )


def _inspect_state_dict(
    state_dict: Mapping[str, Any],
    *,
    runtime: Any,
    histogram_bins: int,
    max_histogram_tensors: int,
    max_histogram_values: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tensor_entries: list[dict[str, Any]] = []
    module_index: dict[str, dict[str, Any]] = {}
    histogrammed = 0
    finite_tensors = 0
    nonfinite_tensors = 0
    parameter_count = 0

    for name, value in sorted(state_dict.items()):
        if not runtime.is_tensor(value):
            tensor_entries.append(
                {
                    "name": str(name),
                    "module": _module_name(str(name)),
                    "is_tensor": False,
                    "skipped_reason": "state_dict_value_is_not_tensor",
                }
            )
            continue
        include_histogram = histogrammed < max_histogram_tensors
        entry = _tensor_entry(
            str(name),
            value,
            runtime=runtime,
            histogram_bins=histogram_bins,
            max_histogram_values=max_histogram_values,
            include_histogram=include_histogram,
        )
        if entry.get("histogram") is not None:
            histogrammed += 1
        tensor_entries.append(entry)
        parameter_count += int(entry["element_count"])
        if entry["finite"]:
            finite_tensors += 1
        else:
            nonfinite_tensors += 1
        _accumulate_modules(module_index, entry)

    module_entries = [
        {
            "name": name,
            "depth": int(module["depth"]),
            "tensor_count": int(module["tensor_count"]),
            "parameter_count": int(module["parameter_count"]),
        }
        for name, module in sorted(module_index.items())
    ]
    summary = {
        "tensor_count": len([entry for entry in tensor_entries if entry.get("is_tensor", True)]),
        "non_tensor_state_entries": len([entry for entry in tensor_entries if not entry.get("is_tensor", True)]),
        "module_count": len(module_entries),
        "parameter_count": parameter_count,
        "finite_tensor_count": finite_tensors,
        "nonfinite_tensor_count": nonfinite_tensors,
        "histogrammed_tensor_count": histogrammed,
        "all_tensors_finite": nonfinite_tensors == 0,
    }
    return tensor_entries, module_entries, summary


def _tensor_entry(
    name: str,
    value: Any,
    *,
    runtime: Any,
    histogram_bins: int,
    max_histogram_values: int,
    include_histogram: bool,
) -> dict[str, Any]:
    detached = value.detach().cpu()
    shape = tuple(int(item) for item in detached.shape)
    element_count = int(detached.numel())
    module = _module_name(name)
    numeric = _numeric_tensor(detached, runtime=runtime)
    stats = _finite_stats(numeric, runtime=runtime)
    histogram = (
        _histogram(numeric, runtime=runtime, bins=histogram_bins, max_values=max_histogram_values)
        if include_histogram and stats["finite_count"] > 0
        else None
    )
    return {
        "name": name,
        "module": module,
        "is_tensor": True,
        "shape": list(shape),
        "rank": len(shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "element_count": element_count,
        "finite": stats["nonfinite_count"] == 0,
        "finite_count": stats["finite_count"],
        "nonfinite_count": stats["nonfinite_count"],
        "min": stats["min"],
        "mean": stats["mean"],
        "std": stats["std"],
        "max": stats["max"],
        "l2_norm": stats["l2_norm"],
        "max_abs": stats["max_abs"],
        "histogram": histogram,
    }


def _numeric_tensor(value: Any, *, runtime: Any) -> Any:
    if runtime.is_complex(value):
        return value.abs().to(dtype=runtime.float64)
    if not runtime.is_floating_point(value):
        return value.to(dtype=runtime.float64)
    return value.to(dtype=runtime.float64)


def _finite_stats(value: Any, *, runtime: Any) -> dict[str, Any]:
    element_count = int(value.numel())
    if element_count == 0:
        return {
            "finite_count": 0,
            "nonfinite_count": 0,
            "min": None,
            "mean": None,
            "std": None,
            "max": None,
            "l2_norm": None,
            "max_abs": None,
        }
    flattened = value.reshape(-1)
    finite_mask = runtime.isfinite(flattened)
    finite_values = flattened[finite_mask]
    finite_count = int(finite_values.numel())
    nonfinite_count = element_count - finite_count
    if finite_count == 0:
        return {
            "finite_count": 0,
            "nonfinite_count": nonfinite_count,
            "min": None,
            "mean": None,
            "std": None,
            "max": None,
            "l2_norm": None,
            "max_abs": None,
        }
    return {
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "min": _finite_float(finite_values.min().item()),
        "mean": _finite_float(finite_values.mean().item()),
        "std": _finite_float(finite_values.std(unbiased=False).item()),
        "max": _finite_float(finite_values.max().item()),
        "l2_norm": _finite_float(runtime.linalg.vector_norm(finite_values).item()),
        "max_abs": _finite_float(finite_values.abs().max().item()),
    }


def _histogram(
    value: Any,
    *,
    runtime: Any,
    bins: int,
    max_values: int,
) -> dict[str, Any] | None:
    flattened = value.reshape(-1)
    finite_values = flattened[runtime.isfinite(flattened)]
    if int(finite_values.numel()) == 0:
        return None
    if int(finite_values.numel()) > max_values:
        finite_values = finite_values[:max_values]
    minimum = float(finite_values.min().item())
    maximum = float(finite_values.max().item())
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        return None
    if minimum == maximum:
        counts = [0 for _ in range(bins)]
        counts[0] = int(finite_values.numel())
        edges = [minimum + float(index) for index in range(bins + 1)]
    else:
        counts_tensor = runtime.histc(finite_values, bins=bins, min=minimum, max=maximum)
        counts = [int(item) for item in counts_tensor.to(dtype=runtime.int64).tolist()]
        width = (maximum - minimum) / bins
        edges = [minimum + width * index for index in range(bins + 1)]
    return {
        "sampled_count": int(finite_values.numel()),
        "bin_count": bins,
        "min": _finite_float(minimum),
        "max": _finite_float(maximum),
        "edges": [_finite_float(item) for item in edges],
        "counts": counts,
    }


def _accumulate_modules(module_index: dict[str, dict[str, Any]], entry: Mapping[str, Any]) -> None:
    name = str(entry["module"])
    parts = [] if name == "__root__" else name.split(".")
    prefixes = ["__root__"] + [".".join(parts[:index]) for index in range(1, len(parts) + 1)]
    for prefix in prefixes:
        module = module_index.setdefault(
            prefix,
            {
                "depth": 0 if prefix == "__root__" else prefix.count(".") + 1,
                "tensor_count": 0,
                "parameter_count": 0,
            },
        )
        module["tensor_count"] += 1
        module["parameter_count"] += int(entry["element_count"])


def _module_name(tensor_name: str) -> str:
    if "." not in tensor_name:
        return "__root__"
    return tensor_name.rsplit(".", maxsplit=1)[0]


def _payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics")
    metric_keys = sorted(str(key) for key in metrics) if isinstance(metrics, Mapping) else []
    return {
        "schema_version": _optional_string(payload.get("schema_version")),
        "checkpoint_schema_version": _optional_string(payload.get("checkpoint_schema_version")),
        "step": _optional_int(payload.get("step")),
        "has_model_state_dict": isinstance(payload.get("model_state_dict"), Mapping),
        "has_optimizer_state_dict": isinstance(payload.get("optimizer_state_dict"), Mapping),
        "optimizer_state_serialized": False,
        "metric_count": len(metric_keys),
        "metric_keys": metric_keys,
    }


def _validate_parent_manifests(parent_manifests: Sequence[Path | str]) -> list[str]:
    parent_ids: list[str] = []
    for manifest_path in parent_manifests:
        manifest = read_artifact_manifest(manifest_path)
        validate_artifact_checksums(manifest, root=Path(manifest_path).parent)
        parent_ids.append(manifest.artifact_id)
    if len(set(parent_ids)) != len(parent_ids):
        raise ArtifactManifestError("parent manifests must not contain duplicate artifact ids")
    return parent_ids


def _load_torch_checkpoint(path: Path, *, runtime: Any) -> Mapping[str, Any]:
    try:
        payload = runtime.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility.
        payload = runtime.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise CheckpointInspectionError("checkpoint payload must be a JSON-like mapping")
    return payload


def _require_torch() -> Any:
    if importlib.util.find_spec("torch") is None:
        from codelewm.data import OptionalDependencyError

        raise OptionalDependencyError(
            "checkpoint inspection requires torch; install it with `uv sync --group train --group dev`"
        )
    import torch

    return torch


def _json_native_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return {"sha256": compute_json_sha256(_stringify_jsonish(value))}
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _stringify_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stringify_jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_jsonish(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _finite_float(value: Any) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise CheckpointInspectionError("tensor statistic must be finite")
    return scalar


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return redact_text(str(path))


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CheckpointInspectionError(f"checkpoint inspection report must be JSON-native: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
