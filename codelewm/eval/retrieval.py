"""Retrieval metrics, candidate pools, and JSON reports."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RETRIEVAL_METRICS_SCHEMA_VERSION = "codelewm.eval.retrieval_metrics.v1"
RETRIEVAL_REPORT_SCHEMA_VERSION = "codelewm.eval.retrieval_report.v1"
CANDIDATE_POOL_SCHEMA_VERSION = "codelewm.eval.candidate_pool.v1"
TRAIN_SPLITS = frozenset({"train"})


class RetrievalEvalError(ValueError):
    """Raised when retrieval inputs or report payloads are invalid."""


@dataclass(frozen=True)
class RetrievalMetrics:
    """Rank-based retrieval metrics for a set of query targets."""

    query_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    median_rank: float
    candidate_count_min: int | None = None
    candidate_count_max: int | None = None
    schema_version: str = RETRIEVAL_METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_count": self.query_count,
            "candidate_count_min": self.candidate_count_min,
            "candidate_count_max": self.candidate_count_max,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "median_rank": self.median_rank,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RetrievalMetrics":
        metrics = cls(
            schema_version=str(payload["schema_version"]),
            query_count=int(payload["query_count"]),
            candidate_count_min=_optional_int(payload.get("candidate_count_min"), "candidate_count_min"),
            candidate_count_max=_optional_int(payload.get("candidate_count_max"), "candidate_count_max"),
            recall_at_1=float(payload["recall_at_1"]),
            recall_at_5=float(payload["recall_at_5"]),
            recall_at_10=float(payload["recall_at_10"]),
            mrr=float(payload["mrr"]),
            median_rank=float(payload["median_rank"]),
        )
        validate_retrieval_metrics(metrics)
        return metrics


@dataclass(frozen=True)
class CandidatePoolEntry:
    """One held-out after-state candidate referenced by transition id."""

    transition_id: str
    split: str
    source: str = "unknown"
    repo: str = ""
    path: str = ""
    edit_size: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise RetrievalEvalError("candidate transition_id must not be empty")
        if not self.split:
            raise RetrievalEvalError(f"candidate {self.transition_id!r} must declare split")
        if self.edit_size < 0:
            raise RetrievalEvalError(f"candidate {self.transition_id!r} edit_size must be non-negative")
        _require_json_native(self.metadata, "candidate metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "split": self.split,
            "source": self.source,
            "repo": self.repo,
            "path": self.path,
            "edit_size": self.edit_size,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidatePoolEntry":
        if payload.get("transition_id") is None:
            raise RetrievalEvalError("candidate row must include transition_id")
        if payload.get("split") is None:
            raise RetrievalEvalError(f"candidate {payload['transition_id']!r} must declare split")
        return cls(
            transition_id=str(payload["transition_id"]),
            split=_split_name(payload["split"]),
            source=str(payload.get("source", "unknown")),
            repo=str(payload.get("repo", "")),
            path=str(payload.get("path", "")),
            edit_size=int(payload.get("edit_size", 0) or 0),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class CandidatePool:
    """Schema-versioned set of held-out candidate after-states."""

    name: str
    entries: tuple[CandidatePoolEntry, ...]
    seed: int | None = None
    max_size: int | None = None
    excluded_splits: tuple[str, ...] = ("train",)
    schema_version: str = CANDIDATE_POOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_candidate_pool(self)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(entry.transition_id for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "entry_count": len(self.entries),
            "seed": self.seed,
            "max_size": self.max_size,
            "excluded_splits": list(self.excluded_splits),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidatePool":
        entries = tuple(CandidatePoolEntry.from_dict(entry) for entry in payload["entries"])
        if payload.get("entry_count") is not None and int(payload["entry_count"]) != len(entries):
            raise RetrievalEvalError("candidate pool entry_count does not match entries")
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            seed=_optional_int(payload.get("seed"), "seed"),
            max_size=_optional_int(payload.get("max_size"), "max_size"),
            excluded_splits=tuple(_split_name(split) for split in payload.get("excluded_splits", ("train",))),
            entries=entries,
        )


@dataclass(frozen=True)
class RetrievalReport:
    """JSON-native retrieval report matching the v0.1 metric contract."""

    metrics: RetrievalMetrics
    candidate_pool: CandidatePool | None = None
    baselines: Mapping[str, RetrievalMetrics] = field(default_factory=dict)
    slices: Mapping[str, RetrievalMetrics] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RETRIEVAL_REPORT_SCHEMA_VERSION

    @property
    def recall_at_1(self) -> float:
        return self.metrics.recall_at_1

    @property
    def recall_at_5(self) -> float:
        return self.metrics.recall_at_5

    @property
    def recall_at_10(self) -> float:
        return self.metrics.recall_at_10

    @property
    def mrr(self) -> float:
        return self.metrics.mrr

    @property
    def median_rank(self) -> float:
        return self.metrics.median_rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "median_rank": self.median_rank,
            "metrics": self.metrics.to_dict(),
            "candidate_pool": None if self.candidate_pool is None else self.candidate_pool.to_dict(),
            "baselines": {name: metrics.to_dict() for name, metrics in sorted(self.baselines.items())},
            "slices": {name: metrics.to_dict() for name, metrics in sorted(self.slices.items())},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RetrievalReport":
        candidate_pool_payload = payload.get("candidate_pool")
        metrics = RetrievalMetrics.from_dict(payload["metrics"])
        _validate_top_level_metrics(payload, metrics)
        report = cls(
            schema_version=str(payload["schema_version"]),
            metrics=metrics,
            candidate_pool=None
            if candidate_pool_payload is None
            else CandidatePool.from_dict(candidate_pool_payload),
            baselines={
                str(name): RetrievalMetrics.from_dict(metrics)
                for name, metrics in payload.get("baselines", {}).items()
            },
            slices={
                str(name): RetrievalMetrics.from_dict(metrics)
                for name, metrics in payload.get("slices", {}).items()
            },
            metadata=dict(payload.get("metadata", {})),
        )
        validate_retrieval_report(report)
        return report


def compute_retrieval_metrics(
    ranks: Iterable[int],
    *,
    candidate_counts: Iterable[int] | None = None,
) -> RetrievalMetrics:
    """Compute Recall@1/5/10, MRR, and median rank from 1-based target ranks."""

    rank_values = tuple(_positive_int(rank, "rank") for rank in ranks)
    if not rank_values:
        raise RetrievalEvalError("retrieval metrics require at least one rank")

    count_values: tuple[int, ...] | None = None
    if candidate_counts is not None:
        count_values = tuple(_positive_int(count, "candidate_count") for count in candidate_counts)
        if len(count_values) != len(rank_values):
            raise RetrievalEvalError("candidate_counts length must match ranks length")
        for rank, candidate_count in zip(rank_values, count_values):
            if rank > candidate_count:
                raise RetrievalEvalError("rank cannot exceed candidate_count")

    query_count = len(rank_values)
    metrics = RetrievalMetrics(
        query_count=query_count,
        recall_at_1=_recall_at(rank_values, 1),
        recall_at_5=_recall_at(rank_values, 5),
        recall_at_10=_recall_at(rank_values, 10),
        mrr=sum(1.0 / rank for rank in rank_values) / query_count,
        median_rank=float(statistics.median(rank_values)),
        candidate_count_min=None if count_values is None else min(count_values),
        candidate_count_max=None if count_values is None else max(count_values),
    )
    validate_retrieval_metrics(metrics)
    return metrics


def rank_targets(
    score_rows: Sequence[Sequence[float]],
    candidate_ids_by_query: Sequence[Sequence[str]],
    target_ids: Sequence[str],
    *,
    larger_is_better: bool = True,
) -> tuple[int, ...]:
    """Rank each target id in its candidate row using deterministic tie order."""

    if len(score_rows) != len(candidate_ids_by_query) or len(score_rows) != len(target_ids):
        raise RetrievalEvalError("score_rows, candidate_ids_by_query, and target_ids must have equal length")

    ranks: list[int] = []
    for row_index, (scores, candidate_ids, target_id) in enumerate(
        zip(score_rows, candidate_ids_by_query, target_ids)
    ):
        score_values = tuple(float(score) for score in scores)
        candidate_values = tuple(str(candidate_id) for candidate_id in candidate_ids)
        target_value = str(target_id)
        if len(score_values) != len(candidate_values):
            raise RetrievalEvalError(f"score row {row_index} length does not match candidate ids")
        if not score_values:
            raise RetrievalEvalError(f"score row {row_index} must not be empty")
        if len(set(candidate_values)) != len(candidate_values):
            raise RetrievalEvalError(f"candidate ids for row {row_index} must be unique")
        if candidate_values.count(target_value) != 1:
            raise RetrievalEvalError(f"target id {target_value!r} must appear exactly once in row {row_index}")
        if any(not math.isfinite(score) for score in score_values):
            raise RetrievalEvalError(f"score row {row_index} contains NaN or inf")

        indexed_scores = tuple(enumerate(score_values))
        if larger_is_better:
            ordered = sorted(indexed_scores, key=lambda item: (-item[1], item[0]))
        else:
            ordered = sorted(indexed_scores, key=lambda item: (item[1], item[0]))
        for rank, (candidate_index, _) in enumerate(ordered, start=1):
            if candidate_values[candidate_index] == target_value:
                ranks.append(rank)
                break
    return tuple(ranks)


def build_easy_candidate_pool(
    rows: Iterable[Any],
    *,
    max_size: int = 1000,
    seed: int = 0,
    name: str = "easy-1k",
    candidate_splits: Sequence[str] = ("val", "test"),
    exclude_splits: Sequence[str] = ("train",),
) -> CandidatePool:
    """Build a deterministic random held-out pool for easy retrieval."""

    if max_size <= 0:
        raise RetrievalEvalError("max_size must be positive")
    excluded = tuple(_split_name(split) for split in exclude_splits)
    allowed = frozenset(_split_name(split) for split in candidate_splits)
    entries = [
        entry
        for entry in (_coerce_pool_entry(row) for row in rows)
        if entry.split not in excluded and entry.split in allowed
    ]
    if not entries:
        raise RetrievalEvalError("candidate pool has no held-out rows after filtering")

    ordered = sorted(entries, key=lambda entry: entry.transition_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return CandidatePool(
        name=name,
        entries=tuple(ordered[:max_size]),
        seed=seed,
        max_size=max_size,
        excluded_splits=excluded,
    )


def build_fixture_candidate_pool(
    rows: Iterable[Any],
    *,
    candidate_ids: Sequence[str] | None = None,
    name: str = "fixture",
    exclude_splits: Sequence[str] = ("train",),
) -> CandidatePool:
    """Build a small explicit fixture pool and fail on split leakage."""

    by_id: dict[str, CandidatePoolEntry] = {}
    for row in rows:
        entry = _coerce_pool_entry(row)
        if entry.transition_id in by_id:
            raise RetrievalEvalError(f"duplicate candidate row id: {entry.transition_id}")
        by_id[entry.transition_id] = entry
    if candidate_ids is None:
        entries = tuple(by_id[transition_id] for transition_id in sorted(by_id))
    else:
        ids = tuple(str(candidate_id) for candidate_id in candidate_ids)
        missing = [candidate_id for candidate_id in ids if candidate_id not in by_id]
        if missing:
            raise RetrievalEvalError(f"candidate ids not found: {', '.join(missing)}")
        entries = tuple(by_id[candidate_id] for candidate_id in ids)
    return CandidatePool(
        name=name,
        entries=entries,
        excluded_splits=tuple(_split_name(split) for split in exclude_splits),
        max_size=len(entries),
    )


def build_retrieval_report(
    ranks: Iterable[int],
    *,
    candidate_pool: CandidatePool | None = None,
    candidate_counts: Iterable[int] | None = None,
    baselines: Mapping[str, RetrievalMetrics | Iterable[int]] | None = None,
    slices: Mapping[str, RetrievalMetrics | Iterable[int]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RetrievalReport:
    """Build and validate a retrieval report from query ranks."""

    report = RetrievalReport(
        metrics=compute_retrieval_metrics(ranks, candidate_counts=candidate_counts),
        candidate_pool=candidate_pool,
        baselines=_coerce_metrics_mapping(baselines),
        slices=_coerce_metrics_mapping(slices),
        metadata={} if metadata is None else dict(metadata),
    )
    validate_retrieval_report(report)
    return report


def validate_candidate_pool(pool: CandidatePool) -> CandidatePool:
    """Validate that a candidate pool is non-empty, unique, and held-out."""

    if pool.schema_version != CANDIDATE_POOL_SCHEMA_VERSION:
        raise RetrievalEvalError(
            "unsupported candidate pool schema; "
            f"expected {CANDIDATE_POOL_SCHEMA_VERSION!r}, got {pool.schema_version!r}"
        )
    if not pool.name:
        raise RetrievalEvalError("candidate pool name must not be empty")
    if not pool.entries:
        raise RetrievalEvalError("candidate pool must contain at least one entry")
    if pool.max_size is not None and pool.max_size <= 0:
        raise RetrievalEvalError("candidate pool max_size must be positive when set")

    ids = [entry.transition_id for entry in pool.entries]
    if len(set(ids)) != len(ids):
        raise RetrievalEvalError("candidate pool transition ids must be unique")

    excluded = frozenset(pool.excluded_splits) | TRAIN_SPLITS
    leaked = [entry.transition_id for entry in pool.entries if entry.split in excluded]
    if leaked:
        raise RetrievalEvalError(f"candidate pool includes training rows: {', '.join(leaked)}")
    return pool


def validate_retrieval_metrics(metrics: RetrievalMetrics) -> RetrievalMetrics:
    """Validate metric values and schema."""

    if metrics.schema_version != RETRIEVAL_METRICS_SCHEMA_VERSION:
        raise RetrievalEvalError(
            "unsupported retrieval metrics schema; "
            f"expected {RETRIEVAL_METRICS_SCHEMA_VERSION!r}, got {metrics.schema_version!r}"
        )
    if metrics.query_count <= 0:
        raise RetrievalEvalError("query_count must be positive")
    for name in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"):
        value = float(getattr(metrics, name))
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise RetrievalEvalError(f"{name} must be finite and within [0, 1]")
    if not math.isfinite(metrics.median_rank) or metrics.median_rank < 1.0:
        raise RetrievalEvalError("median_rank must be finite and at least 1")
    if metrics.candidate_count_min is not None and metrics.candidate_count_min <= 0:
        raise RetrievalEvalError("candidate_count_min must be positive when set")
    if metrics.candidate_count_max is not None and metrics.candidate_count_max <= 0:
        raise RetrievalEvalError("candidate_count_max must be positive when set")
    if (
        metrics.candidate_count_min is not None
        and metrics.candidate_count_max is not None
        and metrics.candidate_count_min > metrics.candidate_count_max
    ):
        raise RetrievalEvalError("candidate_count_min cannot exceed candidate_count_max")
    return metrics


def validate_retrieval_report(report: RetrievalReport) -> RetrievalReport:
    """Validate a retrieval report object."""

    if report.schema_version != RETRIEVAL_REPORT_SCHEMA_VERSION:
        raise RetrievalEvalError(
            "unsupported retrieval report schema; "
            f"expected {RETRIEVAL_REPORT_SCHEMA_VERSION!r}, got {report.schema_version!r}"
        )
    validate_retrieval_metrics(report.metrics)
    if report.candidate_pool is not None:
        validate_candidate_pool(report.candidate_pool)
    for name, metrics in (*report.baselines.items(), *report.slices.items()):
        if not name:
            raise RetrievalEvalError("baseline and slice names must not be empty")
        validate_retrieval_metrics(metrics)
    _require_json_native(report.metadata, "retrieval report metadata")
    return report


def validate_retrieval_report_payload(payload: Mapping[str, Any]) -> RetrievalReport:
    """Return a validated report from a JSON-native payload."""

    return RetrievalReport.from_dict(payload)


def write_retrieval_report(report: RetrievalReport, path: Path) -> None:
    """Write a retrieval report JSON file."""

    validate_retrieval_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_retrieval_report(path: Path) -> RetrievalReport:
    """Read and validate a retrieval report JSON file."""

    return validate_retrieval_report_payload(json.loads(path.read_text()))


def _coerce_metrics_mapping(
    value: Mapping[str, RetrievalMetrics | Iterable[int]] | None,
) -> dict[str, RetrievalMetrics]:
    if value is None:
        return {}
    metrics: dict[str, RetrievalMetrics] = {}
    for name, item in value.items():
        metrics[str(name)] = item if isinstance(item, RetrievalMetrics) else compute_retrieval_metrics(item)
    return metrics


def _coerce_pool_entry(value: Any) -> CandidatePoolEntry:
    if isinstance(value, CandidatePoolEntry):
        return value
    if isinstance(value, Mapping):
        transition_id = value.get("transition_id", value.get("id"))
        if transition_id is None:
            raise RetrievalEvalError("candidate row must include transition_id")
        if value.get("split") is None:
            raise RetrievalEvalError(f"candidate {transition_id!r} must declare split")
        return CandidatePoolEntry.from_dict(
            {
                "transition_id": transition_id,
                "split": value["split"],
                "source": value.get("source", "unknown"),
                "repo": value.get("repo", ""),
                "path": value.get("path", ""),
                "edit_size": value.get("edit_size", 0) or 0,
                "metadata": value.get("metadata", {}),
            }
        )
    try:
        return CandidatePoolEntry(
            transition_id=str(getattr(value, "transition_id")),
            split=_split_name(getattr(value, "split")),
            source=str(getattr(value, "source", "unknown")),
            repo=str(getattr(value, "repo", "")),
            path=str(getattr(value, "path", "")),
            edit_size=int(getattr(value, "edit_size", 0) or 0),
            metadata=dict(getattr(value, "metadata", {}) or {}),
        )
    except AttributeError as exc:
        raise RetrievalEvalError("candidate row must include transition_id and split") from exc


def _recall_at(ranks: Sequence[int], cutoff: int) -> float:
    return sum(1 for rank in ranks if rank <= cutoff) / len(ranks)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RetrievalEvalError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RetrievalEvalError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise RetrievalEvalError(f"{name} must be a positive integer")
    return result


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _split_name(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, int):
        return {0: "train", 1: "val", 2: "test"}.get(value, str(value))
    return str(value)


def _validate_top_level_metrics(payload: Mapping[str, Any], metrics: RetrievalMetrics) -> None:
    for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "median_rank"):
        if key not in payload:
            raise RetrievalEvalError(f"retrieval report payload missing top-level {key}")
        observed = float(payload[key])
        expected = float(getattr(metrics, key))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RetrievalEvalError(f"retrieval report top-level {key} does not match metrics")


def _require_json_native(value: Any, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RetrievalEvalError(f"{name} must be JSON-native") from exc
