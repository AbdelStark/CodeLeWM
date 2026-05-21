"""Latent matrix diagnostics for CodeLeWM representation geometry."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .latent_probe import (
    LATENT_PROBE_TARGETS,
    LATENT_PROBE_VIEWS,
    LatentProbeReport,
    LatentProbeRow,
    read_latent_probe_report,
)


LATENT_MATRIX_REPORT_SCHEMA_VERSION = "codelewm.eval.latent_matrix_report.v1"
LATENT_MATRIX_VIEWS = LATENT_PROBE_VIEWS


class LatentMatrixError(ValueError):
    """Raised when latent matrix inputs or reports are invalid."""


@dataclass(frozen=True)
class LatentMatrixConfig:
    """Configuration for bounded latent matrix diagnostics."""

    matrix_dimension_limit: int = 32
    top_dimensions: int = 16
    max_pairwise_rows: int = 512
    seed: int = 0
    views: tuple[str, ...] = LATENT_MATRIX_VIEWS

    def __post_init__(self) -> None:
        _positive_int(self.matrix_dimension_limit, "matrix_dimension_limit")
        _positive_int(self.top_dimensions, "top_dimensions")
        _positive_int(self.max_pairwise_rows, "max_pairwise_rows")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise LatentMatrixError("seed must be an integer")
        if not self.views:
            raise LatentMatrixError("latent matrix views must not be empty")
        unsupported_views = tuple(view for view in self.views if view not in LATENT_MATRIX_VIEWS)
        if unsupported_views:
            raise LatentMatrixError("unsupported latent matrix views: " + ", ".join(unsupported_views))

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_dimension_limit": self.matrix_dimension_limit,
            "top_dimensions": self.top_dimensions,
            "max_pairwise_rows": self.max_pairwise_rows,
            "seed": self.seed,
            "views": list(self.views),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentMatrixConfig":
        return cls(
            matrix_dimension_limit=_positive_int(
                payload.get("matrix_dimension_limit", 32), "matrix_dimension_limit"
            ),
            top_dimensions=_positive_int(payload.get("top_dimensions", 16), "top_dimensions"),
            max_pairwise_rows=_positive_int(
                payload.get("max_pairwise_rows", 512), "max_pairwise_rows"
            ),
            seed=int(payload.get("seed", 0)),
            views=tuple(str(view) for view in payload.get("views", LATENT_MATRIX_VIEWS)),
        )


@dataclass(frozen=True)
class LatentMatrixReport:
    """Schema-versioned latent matrix diagnostic report."""

    row_count: int
    split_counts: Mapping[str, int]
    source_counts: Mapping[str, int]
    views: Mapping[str, Any]
    probe_associations: Mapping[str, Any]
    claim_boundary: Mapping[str, Any]
    config: LatentMatrixConfig = field(default_factory=LatentMatrixConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = LATENT_MATRIX_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_latent_matrix_report(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "split_counts": dict(sorted(self.split_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "views": dict(sorted(self.views.items())),
            "probe_associations": dict(self.probe_associations),
            "claim_boundary": dict(self.claim_boundary),
            "config": self.config.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentMatrixReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            row_count=_non_negative_int(payload["row_count"], "row_count"),
            split_counts={
                str(split): _non_negative_int(count, f"split_counts.{split}")
                for split, count in _require_mapping(payload["split_counts"], "split_counts").items()
            },
            source_counts={
                str(source): _non_negative_int(count, f"source_counts.{source}")
                for source, count in _require_mapping(payload["source_counts"], "source_counts").items()
            },
            views=dict(_require_mapping(payload["views"], "views")),
            probe_associations=dict(_require_mapping(payload["probe_associations"], "probe_associations")),
            claim_boundary=dict(_require_mapping(payload["claim_boundary"], "claim_boundary")),
            config=LatentMatrixConfig.from_dict(_require_mapping(payload["config"], "config")),
            metadata=dict(_require_mapping(payload.get("metadata", {}), "metadata")),
        )


def build_latent_matrix_report(
    rows: Sequence[LatentProbeRow],
    *,
    embeddings: Mapping[str, np.ndarray],
    config: LatentMatrixConfig | None = None,
    latent_probe_report: LatentProbeReport | None = None,
    latent_probe_report_path: Path | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LatentMatrixReport:
    """Build bounded latent geometry diagnostics from frozen latent matrices."""

    config = LatentMatrixConfig() if config is None else config
    if not rows:
        raise LatentMatrixError("latent matrix report requires at least one row")
    _validate_matrix_map(embeddings, row_count=len(rows), names=config.views)

    view_reports = {
        view: _view_report(np.asarray(embeddings[view], dtype=np.float64), config=config)
        for view in config.views
    }
    probe_associations = _probe_associations(
        rows,
        embeddings=embeddings,
        config=config,
        latent_probe_report=latent_probe_report,
        latent_probe_report_path=latent_probe_report_path,
    )
    claim_boundary = _claim_boundary(probe_associations)
    return LatentMatrixReport(
        row_count=len(rows),
        split_counts=_split_counts(rows),
        source_counts=_source_counts(rows),
        views=view_reports,
        probe_associations=probe_associations,
        claim_boundary=claim_boundary,
        config=config,
        metadata={} if metadata is None else dict(metadata),
    )


def validate_latent_matrix_report(report: LatentMatrixReport) -> LatentMatrixReport:
    """Validate a latent matrix report object."""

    if report.schema_version != LATENT_MATRIX_REPORT_SCHEMA_VERSION:
        raise LatentMatrixError(
            "unsupported latent matrix report schema; "
            f"expected {LATENT_MATRIX_REPORT_SCHEMA_VERSION!r}, got {report.schema_version!r}"
        )
    _non_negative_int(report.row_count, "row_count")
    if set(report.views) != set(report.config.views):
        raise LatentMatrixError("latent matrix views must cover every configured view")
    for split, count in report.split_counts.items():
        if split not in {"train", "val", "test"}:
            raise LatentMatrixError(f"unsupported latent matrix split count: {split}")
        _non_negative_int(count, f"split_counts.{split}")
    _require_json_native(report.to_dict(), "latent matrix report")
    return report


def validate_latent_matrix_report_payload(payload: Mapping[str, Any]) -> LatentMatrixReport:
    """Return a validated latent matrix report from a JSON payload."""

    return LatentMatrixReport.from_dict(payload)


def write_latent_matrix_report(report: LatentMatrixReport, path: Path) -> None:
    """Write a latent matrix report JSON file."""

    validate_latent_matrix_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_latent_matrix_report(path: Path) -> LatentMatrixReport:
    """Read and validate a latent matrix report."""

    return validate_latent_matrix_report_payload(json.loads(path.read_text(encoding="utf-8")))


def read_optional_latent_probe_report(path: Path | str | None) -> LatentProbeReport | None:
    """Read an optional latent-probe report for report linkage."""

    if path is None:
        return None
    return read_latent_probe_report(Path(path))


def _view_report(matrix: np.ndarray, *, config: LatentMatrixConfig) -> dict[str, Any]:
    finite_mask = np.isfinite(matrix)
    if not finite_mask.all():
        raise LatentMatrixError("latent matrix contains NaN or inf")
    row_count, dimension_count = matrix.shape
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(row_count - 1, 1)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = (covariance + covariance.T) / 2.0
    variances = np.diag(covariance)
    correlation = _correlation_from_covariance(covariance)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    effective_rank = _effective_rank(eigenvalues)
    selected_dimensions = _selected_dimensions(
        variances,
        limit=min(config.matrix_dimension_limit, dimension_count),
    )
    dimension_statistics = _dimension_statistics(matrix, variances)
    return {
        "shape": {
            "rows": int(row_count),
            "dimensions": int(dimension_count),
        },
        "finite": {
            "all_finite": True,
            "nonfinite_count": int(matrix.size - int(finite_mask.sum())),
        },
        "norms": _norm_stats(matrix),
        "effective_rank": effective_rank,
        "effective_rank_ratio": 0.0 if dimension_count == 0 else effective_rank / dimension_count,
        "mean_pairwise_cosine": _mean_pairwise_cosine(
            matrix,
            max_rows=config.max_pairwise_rows,
            seed=config.seed,
        ),
        "dimension_statistics": dimension_statistics,
        "covariance_summary": _covariance_summary(covariance, eigenvalues),
        "correlation_summary": _correlation_summary(correlation),
        "heatmap_matrices": {
            "dimension_selection": {
                "policy": "top_variance_dimensions",
                "limit": config.matrix_dimension_limit,
                "selected_count": len(selected_dimensions),
                "dimensions": selected_dimensions,
            },
            "covariance": _matrix_preview(covariance, selected_dimensions),
            "correlation": _matrix_preview(correlation, selected_dimensions),
            "dimension_statistics": [dimension_statistics[index] for index in selected_dimensions],
        },
        "matrix_policy": {
            "raw_latent_vectors_serialized": False,
            "full_covariance_serialized": False,
            "bounded_heatmap_dimension_limit": config.matrix_dimension_limit,
        },
    }


def _dimension_statistics(matrix: np.ndarray, variances: np.ndarray) -> list[dict[str, Any]]:
    total_variance = float(np.sum(variances))
    rows: list[dict[str, Any]] = []
    for dimension in range(matrix.shape[1]):
        column = matrix[:, dimension]
        variance = float(variances[dimension])
        rows.append(
            {
                "dimension": dimension,
                "mean": float(np.mean(column)),
                "std": float(math.sqrt(max(variance, 0.0))),
                "variance": variance,
                "min": float(np.min(column)),
                "max": float(np.max(column)),
                "l2_norm": float(np.linalg.norm(column)),
                "abs_mean": float(np.mean(np.abs(column))),
                "nonzero_fraction": float(np.mean(column != 0.0)),
                "variance_share": 0.0 if total_variance <= 0.0 else variance / total_variance,
            }
        )
    return rows


def _probe_associations(
    rows: Sequence[LatentProbeRow],
    *,
    embeddings: Mapping[str, np.ndarray],
    config: LatentMatrixConfig,
    latent_probe_report: LatentProbeReport | None,
    latent_probe_report_path: Path | str | None,
) -> dict[str, Any]:
    inline = {
        target: _inline_target_associations(rows, target=target, embeddings=embeddings, config=config)
        for target in LATENT_PROBE_TARGETS
    }
    return {
        "inline_dimension_associations": inline,
        "latent_probe_report": _latent_probe_report_summary(
            latent_probe_report,
            latent_probe_report_path=latent_probe_report_path,
        ),
        "caveat": (
            "Dimension-label associations are exploratory diagnostics. They do not name "
            "semantic axes unless stable across seeds/splits and stronger than controls."
        ),
    }


def _inline_target_associations(
    rows: Sequence[LatentProbeRow],
    *,
    target: str,
    embeddings: Mapping[str, np.ndarray],
    config: LatentMatrixConfig,
) -> dict[str, Any]:
    labeled_indices = tuple(
        index for index, row in enumerate(rows) if _label_for(row, target) is not None
    )
    split_counts = Counter(rows[index].split for index in labeled_indices)
    labels = tuple(str(_label_for(rows[index], target)) for index in labeled_indices)
    if len(set(labels)) < 2:
        return {
            "available": False,
            "unavailable_reason": "fewer than two labels available for this target",
            "sample_count": len(labeled_indices),
            "split_counts": dict(sorted(split_counts.items())),
            "views": {},
        }
    views: dict[str, Any] = {}
    for view in config.views:
        matrix = np.asarray(embeddings[view], dtype=np.float64)[list(labeled_indices)]
        views[view] = {
            "top_dimensions": _dimension_associations(matrix, labels, top_k=config.top_dimensions),
            "sample_count": len(labeled_indices),
            "class_count": len(set(labels)),
        }
    return {
        "available": True,
        "sample_count": len(labeled_indices),
        "split_counts": dict(sorted(split_counts.items())),
        "class_count": len(set(labels)),
        "views": views,
    }


def _latent_probe_report_summary(
    report: LatentProbeReport | None,
    *,
    latent_probe_report_path: Path | str | None,
) -> dict[str, Any]:
    if report is None:
        return {
            "available": False,
            "unavailable_reason": "latent probe report was not provided",
        }
    path = None if latent_probe_report_path is None else Path(latent_probe_report_path)
    target_summaries = {}
    for target, target_report in report.target_reports.items():
        target_summaries[target] = {
            "available": bool(target_report.get("available")),
            "best_view_test_accuracy": _best_test_accuracy(target_report.get("views", {})),
            "best_control_test_accuracy": _best_test_accuracy(target_report.get("baselines", {})),
            "controls": _control_summaries(target_report.get("baselines", {})),
            "views": _control_summaries(target_report.get("views", {})),
            "axis_diagnostics": _axis_summary(report.axis_diagnostics.get(target, {})),
        }
    return {
        "available": True,
        "path": None if path is None else str(path),
        "sha256": None if path is None else _sha256_file(path),
        "schema_version": report.schema_version,
        "row_count": report.row_count,
        "split_counts": dict(report.split_counts),
        "claim_boundary": dict(report.claim_boundary),
        "targets": target_summaries,
    }


def _claim_boundary(probe_associations: Mapping[str, Any]) -> dict[str, Any]:
    probe_summary = probe_associations.get("latent_probe_report", {})
    probe_claim = probe_summary.get("claim_boundary", {}) if probe_summary.get("available") else {}
    controls_beat_status = str(probe_claim.get("semantic_structure_status", "not_evaluable"))
    return {
        "positive_representation_claim_allowed": False,
        "semantic_axis_claim_allowed": False,
        "action_conditioned_quality_claim_allowed": False,
        "downstream_coding_usefulness_claim_allowed": False,
        "reason": (
            "latent_matrix_report_is_diagnostic_only; semantic-axis claims require "
            "stable dimensions across multiple seeds and held-out splits plus "
            "control-beating probe metrics"
        ),
        "semantic_axis_gate": {
            "split_stability_evidence": _has_split_stability(probe_associations),
            "seed_stability_evidence": False,
            "controls_beat_status": controls_beat_status,
            "passed": False,
        },
        "blocked_claims": [
            "semantic_latent_axes",
            "action_conditioned_quality",
            "downstream_coding_usefulness",
        ],
    }


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance)
    std = np.sqrt(np.clip(diagonal, 0.0, None))
    denominator = np.outer(std, std)
    correlation = np.zeros_like(covariance, dtype=np.float64)
    np.divide(covariance, denominator, out=correlation, where=denominator > 0.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    for index, value in enumerate(std):
        correlation[index, index] = 1.0 if value > 0.0 else 0.0
    return correlation


def _effective_rank(eigenvalues: np.ndarray) -> float:
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def _norm_stats(matrix: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "row_l2_min": float(np.min(norms)),
        "row_l2_mean": float(np.mean(norms)),
        "row_l2_std": float(np.std(norms)),
        "row_l2_max": float(np.max(norms)),
    }


def _mean_pairwise_cosine(matrix: np.ndarray, *, max_rows: int, seed: int) -> dict[str, Any]:
    row_count = matrix.shape[0]
    if row_count < 2:
        return {
            "value": 0.0,
            "row_sample_count": row_count,
            "sampled": False,
        }
    if row_count > max_rows:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(row_count), max_rows))
        sample = matrix[indices]
        sampled = True
    else:
        sample = matrix
        sampled = False
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    normalized = np.zeros_like(sample, dtype=np.float64)
    np.divide(sample, norms, out=normalized, where=norms > 0.0)
    similarity = normalized @ normalized.T
    off_diagonal_count = sample.shape[0] * (sample.shape[0] - 1)
    value = 0.0 if off_diagonal_count == 0 else float((similarity.sum() - np.trace(similarity)) / off_diagonal_count)
    return {
        "value": value,
        "row_sample_count": int(sample.shape[0]),
        "sampled": sampled,
    }


def _covariance_summary(covariance: np.ndarray, eigenvalues: np.ndarray) -> dict[str, Any]:
    diagonal = np.diag(covariance)
    return {
        "trace": float(np.trace(covariance)),
        "diag_min": float(np.min(diagonal)),
        "diag_median": float(np.median(diagonal)),
        "diag_max": float(np.max(diagonal)),
        "mean_abs_covariance": float(np.mean(np.abs(covariance))),
        "top_eigenvalues": [float(value) for value in sorted(eigenvalues, reverse=True)[:16]],
    }


def _correlation_summary(correlation: np.ndarray) -> dict[str, Any]:
    dimension_count = correlation.shape[0]
    if dimension_count < 2:
        off = np.asarray([], dtype=np.float64)
    else:
        mask = ~np.eye(dimension_count, dtype=bool)
        off = np.abs(correlation[mask])
    top_pairs = []
    for i in range(dimension_count):
        for j in range(i + 1, dimension_count):
            top_pairs.append(
                {
                    "dimension_i": i,
                    "dimension_j": j,
                    "correlation": float(correlation[i, j]),
                    "abs_correlation": float(abs(correlation[i, j])),
                }
            )
    top_pairs.sort(key=lambda item: (-float(item["abs_correlation"]), item["dimension_i"], item["dimension_j"]))
    return {
        "mean_abs_off_diagonal": 0.0 if off.size == 0 else float(np.mean(off)),
        "max_abs_off_diagonal": 0.0 if off.size == 0 else float(np.max(off)),
        "top_abs_pairs": top_pairs[:16],
    }


def _selected_dimensions(variances: np.ndarray, *, limit: int) -> list[int]:
    indices = list(range(variances.shape[0]))
    indices.sort(key=lambda index: (-float(variances[index]), index))
    return sorted(indices[:limit])


def _matrix_preview(matrix: np.ndarray, dimensions: Sequence[int]) -> list[list[float]]:
    return [
        [float(matrix[row_dimension, column_dimension]) for column_dimension in dimensions]
        for row_dimension in dimensions
    ]


def _dimension_associations(matrix: np.ndarray, labels: Sequence[str], *, top_k: int) -> list[dict[str, Any]]:
    if matrix.ndim != 2:
        raise LatentMatrixError("dimension association matrix must be rank 2")
    if len(labels) != matrix.shape[0]:
        raise LatentMatrixError("dimension association labels must align to rows")
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped_indices[label].append(index)
    scores = []
    for dimension in range(matrix.shape[1]):
        column = np.asarray(matrix[:, dimension], dtype=np.float64)
        total = float(np.sum((column - float(np.mean(column))) ** 2))
        if total <= 0.0:
            eta = 0.0
        else:
            between = 0.0
            global_mean = float(np.mean(column))
            for indices in grouped_indices.values():
                group = column[indices]
                between += len(indices) * (float(np.mean(group)) - global_mean) ** 2
            eta = float(max(0.0, min(1.0, between / total)))
        dominant_label = max(
            grouped_indices,
            key=lambda label: (float(np.mean(column[grouped_indices[label]])), label),
        )
        scores.append(
            {
                "dimension": dimension,
                "eta_squared": eta,
                "dominant_label": dominant_label,
            }
        )
    scores.sort(key=lambda item: (-float(item["eta_squared"]), int(item["dimension"])))
    return scores[:top_k]


def _control_summaries(reports: Mapping[str, Any]) -> dict[str, Any]:
    summaries = {}
    for name, report in reports.items():
        test = report.get("splits", {}).get("test", {})
        summaries[name] = {
            "test_accuracy": float(test.get("accuracy", 0.0) or 0.0),
            "test_macro_f1": float(test.get("macro_f1", 0.0) or 0.0),
            "test_accuracy_ci95": list(test.get("accuracy_ci95", [])),
        }
    return summaries


def _axis_summary(axis: Mapping[str, Any]) -> dict[str, Any]:
    if not axis:
        return {
            "available": False,
            "dimension_claims_allowed": False,
        }
    return {
        "available": bool(axis.get("available")),
        "dimension_claims_allowed": bool(axis.get("dimension_claims_allowed")),
        "claim_block_reason": axis.get("claim_block_reason"),
        "views": {
            view: {
                "stable_dimensions_across_splits": list(payload.get("stable_dimensions_across_splits", [])),
                "stable_across_splits": bool(payload.get("stable_across_splits")),
                "stable_across_seeds": bool(payload.get("stable_across_seeds")),
            }
            for view, payload in axis.get("views", {}).items()
        },
    }


def _has_split_stability(probe_associations: Mapping[str, Any]) -> bool:
    latent_probe = probe_associations.get("latent_probe_report", {})
    for target in latent_probe.get("targets", {}).values():
        for view in target.get("axis_diagnostics", {}).get("views", {}).values():
            if view.get("stable_across_splits"):
                return True
    return False


def _best_test_accuracy(reports: Mapping[str, Any]) -> float:
    best = 0.0
    for report in reports.values():
        split = report.get("splits", {}).get("test", {})
        best = max(best, float(split.get("accuracy", 0.0) or 0.0))
    return best


def _split_counts(rows: Sequence[LatentProbeRow]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row.split] += 1
    return counts


def _source_counts(rows: Sequence[LatentProbeRow]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.metadata_features.get("source", "<unknown>"))] += 1
    return dict(sorted(counts.items()))


def _label_for(row: LatentProbeRow, target: str) -> str | None:
    value = row.labels.get(target)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_matrix_map(
    matrices: Mapping[str, np.ndarray],
    *,
    row_count: int,
    names: Sequence[str],
) -> None:
    for name in names:
        if name not in matrices:
            raise LatentMatrixError(f"missing latent matrix: {name}")
        matrix = np.asarray(matrices[name])
        if matrix.ndim != 2:
            raise LatentMatrixError(f"latent matrix {name!r} must be rank 2")
        if matrix.shape[0] != row_count:
            raise LatentMatrixError(
                f"latent matrix {name!r} row count mismatch: {matrix.shape[0]} != {row_count}"
            )
        if matrix.shape[1] == 0:
            raise LatentMatrixError(f"latent matrix {name!r} must have at least one dimension")
        if not np.isfinite(matrix).all():
            raise LatentMatrixError(f"latent matrix {name!r} contains NaN or inf")


def _require_json_native(value: Any, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise LatentMatrixError(f"{name} must be JSON native") from exc


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LatentMatrixError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LatentMatrixError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LatentMatrixError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise LatentMatrixError(f"{name} must be a positive integer")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LatentMatrixError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LatentMatrixError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise LatentMatrixError(f"{name} must be a non-negative integer")
    return parsed
