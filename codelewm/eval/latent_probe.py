"""Latent representation probes and claim-safe axis diagnostics."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


LATENT_PROBE_REPORT_SCHEMA_VERSION = "codelewm.eval.latent_probe_report.v1"
LATENT_PROBE_TARGETS = (
    "edit_class",
    "ast_node_kind",
    "symbol_kind",
    "edit_size_bucket",
    "action_cluster",
    "source_family",
)
SUPPORTED_LATENT_PROBE_TARGETS = (
    *LATENT_PROBE_TARGETS,
    "output_type",
    "will_raise",
    "output_magnitude_bucket",
    "output_length_bucket",
    "arithmetic_vs_string_vs_collection",
    "judge_verdict",
)
LATENT_PROBE_VIEWS = ("z_before", "z_after", "z_pred_after")
LATENT_PROBE_BASELINES = (
    "majority",
    "metadata_only",
    "lexical",
    "random_latent",
    "no_action",
    "shuffled_action",
)


class LatentProbeError(ValueError):
    """Raised when latent probe inputs or reports are invalid."""


@dataclass(frozen=True)
class LatentProbeConfig:
    """Configuration for deterministic latent representation probes."""

    bootstrap_samples: int = 200
    seed: int = 0
    top_dimensions: int = 8
    min_train_classes: int = 2
    min_eval_rows: int = 1
    targets: tuple[str, ...] = LATENT_PROBE_TARGETS
    views: tuple[str, ...] = LATENT_PROBE_VIEWS

    def __post_init__(self) -> None:
        _non_negative_int(self.bootstrap_samples, "bootstrap_samples")
        _non_negative_int(self.top_dimensions, "top_dimensions")
        _positive_int(self.min_train_classes, "min_train_classes")
        _positive_int(self.min_eval_rows, "min_eval_rows")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise LatentProbeError("seed must be an integer")
        if not self.targets:
            raise LatentProbeError("latent probe targets must not be empty")
        if not self.views:
            raise LatentProbeError("latent probe views must not be empty")
        unsupported_targets = tuple(target for target in self.targets if target not in SUPPORTED_LATENT_PROBE_TARGETS)
        if unsupported_targets:
            raise LatentProbeError("unsupported latent probe targets: " + ", ".join(unsupported_targets))
        unsupported_views = tuple(view for view in self.views if view not in LATENT_PROBE_VIEWS)
        if unsupported_views:
            raise LatentProbeError("unsupported latent probe views: " + ", ".join(unsupported_views))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bootstrap_samples": self.bootstrap_samples,
            "seed": self.seed,
            "top_dimensions": self.top_dimensions,
            "min_train_classes": self.min_train_classes,
            "min_eval_rows": self.min_eval_rows,
            "targets": list(self.targets),
            "views": list(self.views),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentProbeConfig":
        return cls(
            bootstrap_samples=_non_negative_int(
                payload.get("bootstrap_samples", 200), "bootstrap_samples"
            ),
            seed=int(payload.get("seed", 0)),
            top_dimensions=_non_negative_int(payload.get("top_dimensions", 8), "top_dimensions"),
            min_train_classes=_positive_int(payload.get("min_train_classes", 2), "min_train_classes"),
            min_eval_rows=_positive_int(payload.get("min_eval_rows", 1), "min_eval_rows"),
            targets=tuple(str(target) for target in payload.get("targets", LATENT_PROBE_TARGETS)),
            views=tuple(str(view) for view in payload.get("views", LATENT_PROBE_VIEWS)),
        )


@dataclass(frozen=True)
class LatentProbeRow:
    """One split-aware row used by latent probe reports."""

    transition_id: str
    split: str
    labels: Mapping[str, str | None]
    metadata_features: Mapping[str, str] = field(default_factory=dict)
    lexical_tokens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise LatentProbeError("latent probe row transition_id must not be empty")
        if self.split not in {"train", "val", "test"}:
            raise LatentProbeError(f"latent probe row split is unsupported: {self.split!r}")
        for target in self.labels:
            if target not in SUPPORTED_LATENT_PROBE_TARGETS:
                raise LatentProbeError(f"unsupported latent probe target: {target}")
        _require_json_native(dict(self.metadata_features), "latent probe metadata_features")


@dataclass(frozen=True)
class LatentProbeReport:
    """Schema-versioned report for frozen latent representation probes."""

    row_count: int
    split_counts: Mapping[str, int]
    target_reports: Mapping[str, Any]
    axis_diagnostics: Mapping[str, Any]
    claim_boundary: Mapping[str, Any]
    config: LatentProbeConfig = field(default_factory=LatentProbeConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = LATENT_PROBE_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_latent_probe_report(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "split_counts": dict(sorted(self.split_counts.items())),
            "target_reports": dict(sorted(self.target_reports.items())),
            "axis_diagnostics": dict(sorted(self.axis_diagnostics.items())),
            "claim_boundary": dict(self.claim_boundary),
            "config": self.config.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentProbeReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            row_count=_non_negative_int(payload["row_count"], "row_count"),
            split_counts={
                str(split): _non_negative_int(count, f"split_counts.{split}")
                for split, count in _require_mapping(payload["split_counts"], "split_counts").items()
            },
            target_reports=dict(_require_mapping(payload["target_reports"], "target_reports")),
            axis_diagnostics=dict(_require_mapping(payload["axis_diagnostics"], "axis_diagnostics")),
            claim_boundary=dict(_require_mapping(payload["claim_boundary"], "claim_boundary")),
            config=LatentProbeConfig.from_dict(_require_mapping(payload["config"], "config")),
            metadata=dict(_require_mapping(payload.get("metadata", {}), "metadata")),
        )


def build_latent_probe_report(
    rows: Sequence[LatentProbeRow],
    *,
    embeddings: Mapping[str, np.ndarray],
    baselines: Mapping[str, np.ndarray],
    config: LatentProbeConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LatentProbeReport:
    """Fit deterministic linear-free probes over frozen latent views."""

    config = LatentProbeConfig() if config is None else config
    if not rows:
        raise LatentProbeError("latent probe report requires at least one row")
    _validate_matrix_map(embeddings, row_count=len(rows), names=config.views, kind="embedding")
    _validate_matrix_map(
        baselines,
        row_count=len(rows),
        names=("random_latent", "no_action", "shuffled_action"),
        kind="baseline",
    )
    split_counts = _split_counts(rows)
    target_reports: dict[str, Any] = {}
    axis_diagnostics: dict[str, Any] = {}
    for target in config.targets:
        target_report, target_axis = _build_target_report(
            rows,
            target=target,
            embeddings=embeddings,
            baselines=baselines,
            config=config,
        )
        target_reports[target] = target_report
        axis_diagnostics[target] = target_axis
    claim_boundary = _build_claim_boundary(target_reports, axis_diagnostics, config=config)
    return LatentProbeReport(
        row_count=len(rows),
        split_counts=split_counts,
        target_reports=target_reports,
        axis_diagnostics=axis_diagnostics,
        claim_boundary=claim_boundary,
        config=config,
        metadata={} if metadata is None else dict(metadata),
    )


def validate_latent_probe_report(report: LatentProbeReport) -> LatentProbeReport:
    """Validate a latent probe report object."""

    if report.schema_version != LATENT_PROBE_REPORT_SCHEMA_VERSION:
        raise LatentProbeError(
            "unsupported latent probe report schema; "
            f"expected {LATENT_PROBE_REPORT_SCHEMA_VERSION!r}, got {report.schema_version!r}"
        )
    _non_negative_int(report.row_count, "row_count")
    if set(report.target_reports) != set(report.config.targets):
        raise LatentProbeError("latent probe target_reports must cover every configured target")
    for split, count in report.split_counts.items():
        if split not in {"train", "val", "test"}:
            raise LatentProbeError(f"unsupported latent probe split count: {split}")
        _non_negative_int(count, f"split_counts.{split}")
    _require_json_native(report.to_dict(), "latent probe report")
    return report


def validate_latent_probe_report_payload(payload: Mapping[str, Any]) -> LatentProbeReport:
    """Return a validated latent probe report from JSON payload."""

    return LatentProbeReport.from_dict(payload)


def write_latent_probe_report(report: LatentProbeReport, path: Path) -> None:
    """Write a latent probe report JSON file."""

    validate_latent_probe_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_latent_probe_report(path: Path) -> LatentProbeReport:
    """Read and validate a latent probe report."""

    return validate_latent_probe_report_payload(json.loads(path.read_text(encoding="utf-8")))


def _build_target_report(
    rows: Sequence[LatentProbeRow],
    *,
    target: str,
    embeddings: Mapping[str, np.ndarray],
    baselines: Mapping[str, np.ndarray],
    config: LatentProbeConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = [_label_for(row, target) for row in rows]
    split_indices = {
        split: tuple(index for index, row in enumerate(rows) if row.split == split and labels[index])
        for split in ("train", "val", "test")
    }
    label_counts = {
        split: dict(sorted(Counter(labels[index] for index in indices).items()))
        for split, indices in split_indices.items()
    }
    train_indices = split_indices["train"]
    train_labels = tuple(str(labels[index]) for index in train_indices)
    train_classes = sorted(set(train_labels))
    target_summary: dict[str, Any] = {
        "available": False,
        "label_counts": label_counts,
        "split_counts": {split: len(indices) for split, indices in split_indices.items()},
        "views": {},
        "baselines": {},
        "unavailable_reason": "",
    }
    if len(train_classes) < config.min_train_classes:
        target_summary["unavailable_reason"] = "fewer than two train labels available for this target"
        return target_summary, _unavailable_axis_diagnostic(target_summary["unavailable_reason"])
    if len(split_indices["val"]) < config.min_eval_rows or len(split_indices["test"]) < config.min_eval_rows:
        target_summary["unavailable_reason"] = "val and test splits must each contain target labels"
        return target_summary, _unavailable_axis_diagnostic(target_summary["unavailable_reason"])

    target_summary["available"] = True
    target_summary["unavailable_reason"] = None
    for view in config.views:
        target_summary["views"][view] = _evaluate_centroid_probe(
            embeddings[view],
            labels,
            train_indices=train_indices,
            eval_indices_by_split={
                "val": split_indices["val"],
                "test": split_indices["test"],
            },
            config=config,
            seed_offset=_stable_seed_offset(f"{target}:{view}"),
        )
    target_summary["baselines"] = {
        "majority": _evaluate_majority_baseline(
            train_labels,
            labels,
            eval_indices_by_split={
                "val": split_indices["val"],
                "test": split_indices["test"],
            },
            config=config,
            seed_offset=_stable_seed_offset(f"{target}:majority"),
        ),
        "metadata_only": _evaluate_metadata_baseline(
            rows,
            target_labels=labels,
            train_indices=train_indices,
            eval_indices_by_split={
                "val": split_indices["val"],
                "test": split_indices["test"],
            },
            config=config,
            seed_offset=_stable_seed_offset(f"{target}:metadata"),
        ),
        "lexical": _evaluate_centroid_probe(
            _lexical_matrix(rows),
            labels,
            train_indices=train_indices,
            eval_indices_by_split={
                "val": split_indices["val"],
                "test": split_indices["test"],
            },
            config=config,
            seed_offset=_stable_seed_offset(f"{target}:lexical"),
        ),
        "random_latent": _evaluate_centroid_probe(
            baselines["random_latent"],
            labels,
            train_indices=train_indices,
            eval_indices_by_split={
                "val": split_indices["val"],
                "test": split_indices["test"],
            },
            config=config,
            seed_offset=_stable_seed_offset(f"{target}:random_latent"),
        ),
        "no_action": _evaluate_centroid_probe(
            baselines["no_action"],
            labels,
            train_indices=train_indices,
            eval_indices_by_split={
                "val": split_indices["val"],
                "test": split_indices["test"],
            },
            config=config,
            seed_offset=_stable_seed_offset(f"{target}:no_action"),
        ),
        "shuffled_action": _evaluate_centroid_probe(
            baselines["shuffled_action"],
            labels,
            train_indices=train_indices,
            eval_indices_by_split={
                "val": split_indices["val"],
                "test": split_indices["test"],
            },
            config=config,
            seed_offset=_stable_seed_offset(f"{target}:shuffled_action"),
        ),
    }
    return target_summary, _axis_diagnostics_for_target(
        embeddings,
        labels,
        split_indices=split_indices,
        config=config,
    )


def _evaluate_centroid_probe(
    matrix: np.ndarray,
    labels: Sequence[str | None],
    *,
    train_indices: Sequence[int],
    eval_indices_by_split: Mapping[str, Sequence[int]],
    config: LatentProbeConfig,
    seed_offset: int,
) -> dict[str, Any]:
    train_labels = tuple(str(labels[index]) for index in train_indices)
    centroids = _centroids(matrix, train_indices, train_labels)
    result = {"model": "nearest_centroid", "splits": {}}
    for split, eval_indices in eval_indices_by_split.items():
        actual = tuple(str(labels[index]) for index in eval_indices)
        predicted = tuple(_predict_centroid(matrix[index], centroids) for index in eval_indices)
        result["splits"][split] = _classification_metrics(
            actual,
            predicted,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed + seed_offset + _stable_seed_offset(split),
        )
    return result


def _evaluate_majority_baseline(
    train_labels: Sequence[str],
    labels: Sequence[str | None],
    *,
    eval_indices_by_split: Mapping[str, Sequence[int]],
    config: LatentProbeConfig,
    seed_offset: int,
) -> dict[str, Any]:
    majority = _majority_label(train_labels)
    result = {"model": "train_majority", "majority_label": majority, "splits": {}}
    for split, eval_indices in eval_indices_by_split.items():
        actual = tuple(str(labels[index]) for index in eval_indices)
        predicted = tuple(majority for _ in eval_indices)
        result["splits"][split] = _classification_metrics(
            actual,
            predicted,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed + seed_offset + _stable_seed_offset(split),
        )
    return result


def _evaluate_metadata_baseline(
    rows: Sequence[LatentProbeRow],
    *,
    target_labels: Sequence[str | None],
    train_indices: Sequence[int],
    eval_indices_by_split: Mapping[str, Sequence[int]],
    config: LatentProbeConfig,
    seed_offset: int,
) -> dict[str, Any]:
    default = _majority_label(tuple(str(target_labels[index]) for index in train_indices))
    table: dict[str, str] = {}
    buckets: dict[str, list[str]] = defaultdict(list)
    for index in train_indices:
        buckets[_metadata_key(rows[index])].append(str(target_labels[index]))
    for key, values in buckets.items():
        table[key] = _majority_label(values)
    result = {"model": "metadata_lookup_majority", "default_label": default, "splits": {}}
    for split, eval_indices in eval_indices_by_split.items():
        actual = tuple(str(target_labels[index]) for index in eval_indices)
        predicted = tuple(table.get(_metadata_key(rows[index]), default) for index in eval_indices)
        result["splits"][split] = _classification_metrics(
            actual,
            predicted,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed + seed_offset + _stable_seed_offset(split),
        )
    return result


def _axis_diagnostics_for_target(
    embeddings: Mapping[str, np.ndarray],
    labels: Sequence[str | None],
    *,
    split_indices: Mapping[str, Sequence[int]],
    config: LatentProbeConfig,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "available": True,
        "dimension_claims_allowed": False,
        "claim_block_reason": "axis names require stability across seeds; this report only checks split stability",
        "views": {},
    }
    for view in config.views:
        per_split: dict[str, Any] = {}
        top_sets: list[set[int]] = []
        for split, indices in split_indices.items():
            split_labels = tuple(str(labels[index]) for index in indices if labels[index])
            if not split_labels:
                per_split[split] = []
                continue
            split_matrix = embeddings[view][list(indices)]
            top = _dimension_associations(split_matrix, split_labels, top_k=config.top_dimensions)
            per_split[split] = top
            top_sets.append({int(item["dimension"]) for item in top})
        stable = sorted(set.intersection(*top_sets)) if top_sets else []
        diagnostics["views"][view] = {
            "top_dimensions_by_split": per_split,
            "stable_dimensions_across_splits": stable,
            "stable_across_splits": bool(stable),
            "stable_across_seeds": False,
        }
    return diagnostics


def _unavailable_axis_diagnostic(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "dimension_claims_allowed": False,
        "claim_block_reason": reason,
        "views": {},
    }


def _build_claim_boundary(
    target_reports: Mapping[str, Any],
    axis_diagnostics: Mapping[str, Any],
    *,
    config: LatentProbeConfig,
) -> dict[str, Any]:
    available_targets = tuple(target for target, report in target_reports.items() if report.get("available"))
    dimension_claims_allowed = all(
        bool(axis_diagnostics.get(target, {}).get("dimension_claims_allowed"))
        for target in available_targets
    ) if available_targets else False
    probe_advantages = []
    for target in available_targets:
        target_report = target_reports[target]
        best_view = _best_test_accuracy(target_report.get("views", {}))
        best_control = _best_test_accuracy(target_report.get("baselines", {}))
        probe_advantages.append(best_view - best_control)
    if len(available_targets) < min(5, len(config.targets)):
        status = "not_evaluable"
        reason = "fewer than five predeclared probe targets have train/val/test labels"
    elif dimension_claims_allowed and probe_advantages and min(probe_advantages) > 0.05:
        status = "supported"
        reason = "latent probes beat listed controls and dimension diagnostics allow semantic axis claims"
    elif probe_advantages and min(probe_advantages) > 0.05:
        status = "weakly_indicated"
        reason = "latent probes beat listed controls on available targets, but axis names remain blocked"
    else:
        status = "unsupported"
        reason = "latent probes do not consistently beat listed controls on available targets"
    return {
        "semantic_structure_status": status,
        "positive_representation_claim_allowed": False,
        "dimension_claims_allowed": dimension_claims_allowed,
        "available_target_count": len(available_targets),
        "available_targets": list(available_targets),
        "required_targets": list(config.targets),
        "reason": reason,
        "multiple_comparison_caveat": (
            "Per-dimension associations are exploratory and cannot name semantic axes "
            "without stable dimensions across seeds and splits."
        ),
    }


def _best_test_accuracy(reports: Mapping[str, Any]) -> float:
    best = 0.0
    for report in reports.values():
        split = report.get("splits", {}).get("test", {})
        best = max(best, float(split.get("accuracy", 0.0) or 0.0))
    return best


def _centroids(matrix: np.ndarray, indices: Sequence[int], labels: Sequence[str]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for index, label in zip(indices, labels):
        grouped[label].append(np.asarray(matrix[index], dtype=np.float64))
    return {
        label: np.mean(np.stack(vectors, axis=0), axis=0)
        for label, vectors in sorted(grouped.items())
    }


def _predict_centroid(vector: np.ndarray, centroids: Mapping[str, np.ndarray]) -> str:
    best_label = ""
    best_distance = math.inf
    for label, centroid in sorted(centroids.items()):
        distance = float(np.sum((np.asarray(vector, dtype=np.float64) - centroid) ** 2))
        if distance < best_distance:
            best_label = label
            best_distance = distance
    return best_label


def _classification_metrics(
    actual: Sequence[str],
    predicted: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(actual) != len(predicted):
        raise LatentProbeError("actual and predicted label streams must align")
    if not actual:
        return {
            "sample_count": 0,
            "class_count": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "accuracy_ci95": [0.0, 0.0],
        }
    correct = tuple(1.0 if left == right else 0.0 for left, right in zip(actual, predicted))
    accuracy = float(sum(correct) / len(correct))
    macro_f1 = _macro_f1(actual, predicted)
    return {
        "sample_count": len(actual),
        "class_count": len(set(actual)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "accuracy_ci95": _bootstrap_accuracy_ci(correct, bootstrap_samples=bootstrap_samples, seed=seed),
    }


def _macro_f1(actual: Sequence[str], predicted: Sequence[str]) -> float:
    labels = sorted(set(actual) | set(predicted))
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for left, right in zip(actual, predicted) if left == label and right == label)
        fp = sum(1 for left, right in zip(actual, predicted) if left != label and right == label)
        fn = sum(1 for left, right in zip(actual, predicted) if left == label and right != label)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _bootstrap_accuracy_ci(correct: Sequence[float], *, bootstrap_samples: int, seed: int) -> list[float]:
    if not correct:
        return [0.0, 0.0]
    if bootstrap_samples <= 0:
        value = float(sum(correct) / len(correct))
        return [value, value]
    rng = random.Random(seed)
    values = []
    for _ in range(bootstrap_samples):
        sample = [correct[rng.randrange(len(correct))] for _ in range(len(correct))]
        values.append(sum(sample) / len(sample))
    values.sort()
    lo_index = int(0.025 * (len(values) - 1))
    hi_index = int(0.975 * (len(values) - 1))
    return [float(values[lo_index]), float(values[hi_index])]


def _dimension_associations(matrix: np.ndarray, labels: Sequence[str], *, top_k: int) -> list[dict[str, Any]]:
    if matrix.ndim != 2:
        raise LatentProbeError("dimension association matrix must be rank 2")
    if len(labels) != matrix.shape[0]:
        raise LatentProbeError("dimension association labels must align to rows")
    if top_k <= 0:
        return []
    scores = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped_indices[label].append(index)
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


def _lexical_matrix(rows: Sequence[LatentProbeRow], *, width: int = 256) -> np.ndarray:
    matrix = np.zeros((len(rows), width), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for token in row.lexical_tokens:
            matrix[row_index, int(token) % width] += 1.0
        norm = float(np.linalg.norm(matrix[row_index]))
        if norm > 0.0:
            matrix[row_index] /= norm
    return matrix


def _metadata_key(row: LatentProbeRow) -> str:
    if not row.metadata_features:
        return "__empty__"
    return "|".join(f"{key}={value}" for key, value in sorted(row.metadata_features.items()))


def _label_for(row: LatentProbeRow, target: str) -> str | None:
    value = row.labels.get(target)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _majority_label(labels: Sequence[str]) -> str:
    if not labels:
        raise LatentProbeError("majority baseline requires labels")
    counts = Counter(labels)
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def _split_counts(rows: Sequence[LatentProbeRow]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row.split] += 1
    return counts


def _validate_matrix_map(
    matrices: Mapping[str, np.ndarray],
    *,
    row_count: int,
    names: Sequence[str],
    kind: str,
) -> None:
    for name in names:
        if name not in matrices:
            raise LatentProbeError(f"missing {kind} matrix: {name}")
        matrix = np.asarray(matrices[name])
        if matrix.ndim != 2:
            raise LatentProbeError(f"{kind} matrix {name!r} must be rank 2")
        if matrix.shape[0] != row_count:
            raise LatentProbeError(
                f"{kind} matrix {name!r} row count mismatch: {matrix.shape[0]} != {row_count}"
            )
        if not np.isfinite(matrix).all():
            raise LatentProbeError(f"{kind} matrix {name!r} contains NaN or inf")


def _require_json_native(value: Any, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise LatentProbeError(f"{name} must be JSON native") from exc


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LatentProbeError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LatentProbeError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LatentProbeError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise LatentProbeError(f"{name} must be a positive integer")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LatentProbeError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LatentProbeError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise LatentProbeError(f"{name} must be a non-negative integer")
    return parsed


def _stable_seed_offset(value: str) -> int:
    digest = 0
    for byte in value.encode("utf-8"):
        digest = ((digest * 131) + byte) % (2**31 - 1)
    return digest
