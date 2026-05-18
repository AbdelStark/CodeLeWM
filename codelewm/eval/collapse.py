"""Collapse diagnostics and hard-gate reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when torch is installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight local env.
    torch = None


COLLAPSE_REPORT_SCHEMA_VERSION = "codelewm.eval.collapse_report.v1"
KILL_REPORT_SCHEMA_VERSION = "codelewm.eval.kill_report.v1"


class EvaluationGateError(RuntimeError):
    """Raised when an evaluation or training gate must hard-stop."""


@dataclass(frozen=True)
class CollapseReport:
    """Embedding collapse metrics for a validation batch or report slice."""

    effective_rank: float
    effective_rank_ratio: float
    per_dim_variance_min: float
    per_dim_variance_median: float
    per_dim_variance_max: float
    pairwise_cosine_mean: float
    embedding_norm_mean: float
    nearest_neighbor_entropy: float
    embedding_count: int
    latent_dim: int
    schema_version: str = COLLAPSE_REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "effective_rank": self.effective_rank,
            "effective_rank_ratio": self.effective_rank_ratio,
            "per_dim_variance_min": self.per_dim_variance_min,
            "per_dim_variance_median": self.per_dim_variance_median,
            "per_dim_variance_max": self.per_dim_variance_max,
            "pairwise_cosine_mean": self.pairwise_cosine_mean,
            "embedding_norm_mean": self.embedding_norm_mean,
            "nearest_neighbor_entropy": self.nearest_neighbor_entropy,
            "embedding_count": self.embedding_count,
            "latent_dim": self.latent_dim,
        }


@dataclass(frozen=True)
class CollapseThresholds:
    """Hard thresholds for collapse gate failures."""

    effective_rank_ratio_min: float = 0.20
    per_dim_variance_median_min: float = 1e-8
    nearest_neighbor_entropy_min: float = 0.10


@dataclass(frozen=True)
class CollapseFailure:
    """One failed collapse threshold."""

    metric: str
    observed: float
    threshold: float
    comparison: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "observed": self.observed,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "message": self.message,
        }


@dataclass(frozen=True)
class KillReport:
    """JSON-native hard-stop report for collapse gate failures."""

    collapse_report: CollapseReport
    failures: tuple[CollapseFailure, ...]
    command: tuple[str, ...] = ()
    config_hash: str | None = None
    reason: str = "embedding_collapse"
    suggested_action: str = "inspect collapse diagnostics before continuing training"
    schema_version: str = KILL_REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reason": self.reason,
            "config_hash": self.config_hash,
            "command": list(self.command),
            "suggested_action": self.suggested_action,
            "collapse_report": self.collapse_report.to_dict(),
            "failures": [failure.to_dict() for failure in self.failures],
        }


def compute_collapse_report(embeddings: Any) -> CollapseReport:
    """Compute rank, variance, cosine, norm, and nearest-neighbor diagnostics."""

    values = _as_numpy(embeddings)
    if values.ndim == 3:
        values = values.reshape((-1, values.shape[-1]))
    if values.ndim != 2:
        raise ValueError(f"embeddings must have shape [batch, dim] or [time, batch, dim]; got {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must not be empty")
    _check_finite(values, "embeddings")

    variances = values.var(axis=0)
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigvals = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    eig_sum = float(eigvals.sum())
    if eig_sum <= 0.0:
        effective_rank = 0.0
    else:
        probs = eigvals[eigvals > 0.0] / eig_sum
        effective_rank = float(np.exp(-(probs * np.log(probs)).sum()))

    return CollapseReport(
        effective_rank=effective_rank,
        effective_rank_ratio=effective_rank / values.shape[1],
        per_dim_variance_min=float(variances.min()),
        per_dim_variance_median=float(np.median(variances)),
        per_dim_variance_max=float(variances.max()),
        pairwise_cosine_mean=_pairwise_cosine_mean(values),
        embedding_norm_mean=float(np.linalg.norm(values, axis=1).mean()),
        nearest_neighbor_entropy=_nearest_neighbor_entropy(values),
        embedding_count=int(values.shape[0]),
        latent_dim=int(values.shape[1]),
    )


def evaluate_collapse_gates(
    report: CollapseReport,
    thresholds: CollapseThresholds = CollapseThresholds(),
) -> tuple[CollapseFailure, ...]:
    """Return all hard gate failures for a collapse report."""

    failures: list[CollapseFailure] = []
    _append_min_failure(
        failures,
        metric="effective_rank_ratio",
        observed=report.effective_rank_ratio,
        threshold=thresholds.effective_rank_ratio_min,
        message="effective rank ratio is below the collapse threshold",
    )
    _append_min_failure(
        failures,
        metric="per_dim_variance_median",
        observed=report.per_dim_variance_median,
        threshold=thresholds.per_dim_variance_median_min,
        message="median per-dimension variance is below the collapse threshold",
    )
    _append_min_failure(
        failures,
        metric="nearest_neighbor_entropy",
        observed=report.nearest_neighbor_entropy,
        threshold=thresholds.nearest_neighbor_entropy_min,
        message="nearest-neighbor entropy is below the collapse threshold",
    )
    for metric, observed in report.to_dict().items():
        if isinstance(observed, (int, float)) and not math.isfinite(float(observed)):
            failures.append(
                CollapseFailure(
                    metric=str(metric),
                    observed=float(observed),
                    threshold=float("nan"),
                    comparison="finite",
                    message=f"{metric} is not finite",
                )
            )
    return tuple(failures)


def enforce_collapse_gates(
    report: CollapseReport,
    thresholds: CollapseThresholds = CollapseThresholds(),
    *,
    kill_report_path: Path | None = None,
    command: tuple[str, ...] = (),
    config_hash: str | None = None,
) -> None:
    """Raise and optionally write a kill report when collapse thresholds fail."""

    failures = evaluate_collapse_gates(report, thresholds)
    if not failures:
        return

    kill_report = KillReport(
        collapse_report=report,
        failures=failures,
        command=command,
        config_hash=config_hash,
    )
    if kill_report_path is not None:
        write_kill_report(kill_report, kill_report_path)
    failed_metrics = ", ".join(failure.metric for failure in failures)
    raise EvaluationGateError(f"collapse gate failed: {failed_metrics}")


def write_kill_report(report: KillReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def _append_min_failure(
    failures: list[CollapseFailure],
    *,
    metric: str,
    observed: float,
    threshold: float,
    message: str,
) -> None:
    if observed < threshold:
        failures.append(
            CollapseFailure(
                metric=metric,
                observed=float(observed),
                threshold=float(threshold),
                comparison=">=",
                message=message,
            )
        )


def _pairwise_cosine_mean(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    cosine = normalized @ normalized.T
    mask = ~np.eye(values.shape[0], dtype=bool)
    return float(cosine[mask].mean())


def _nearest_neighbor_entropy(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argmax(similarity, axis=1)
    counts = np.bincount(nearest, minlength=values.shape[0]).astype(float)
    probs = counts[counts > 0.0] / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def _as_numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _check_finite(value: np.ndarray, name: str) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or inf")
