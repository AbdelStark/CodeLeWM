"""Retrieval metrics, candidate pools, and JSON reports."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RETRIEVAL_METRICS_SCHEMA_VERSION = "codelewm.eval.retrieval_metrics.v1"
RETRIEVAL_REPORT_SCHEMA_VERSION = "codelewm.eval.retrieval_report.v1"
ACTION_USE_BASELINE_DELTA_SCHEMA_VERSION = "codelewm.eval.action_use_baseline_delta.v1"
ACTION_USE_CLAIM_GATE_SCHEMA_VERSION = "codelewm.eval.action_use_claim_gate.v1"
CANDIDATE_POOL_SCHEMA_VERSION = "codelewm.eval.candidate_pool.v1"
HARD_NEGATIVE_SAMPLE_SCHEMA_VERSION = "codelewm.eval.hard_negative_sample.v1"
HARD_NEGATIVE_SAMPLER_REPORT_SCHEMA_VERSION = "codelewm.eval.hard_negative_sampler_report.v1"
ACTION_CONTRAST_POOL_REPORT_SCHEMA_VERSION = "codelewm.eval.action_contrast_pool_report.v1"
TRAIN_SPLITS = frozenset({"train"})
REQUIRED_HEADLINE_BASELINES = ("random", "lexical", "no_action", "shuffled_action")
ACTION_USE_REQUIRED_METRICS = ("recall_at_1", "mrr")
ACTION_CONTRAST_POOL_NAMES = (
    "exact_same_before",
    "near_before",
    "same_file",
    "action_cluster",
    "edit_shape",
    "mutation",
    "random",
)
_LEXICAL_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")


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
class ActionUseBaselineDelta:
    """Metric deltas between text-action retrieval and one baseline."""

    baseline: str
    recall_at_1_delta: float
    recall_at_5_delta: float
    recall_at_10_delta: float
    mrr_delta: float
    median_rank_improvement: float
    text_action_beats_baseline: bool
    schema_version: str = ACTION_USE_BASELINE_DELTA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_action_use_baseline_delta(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline": self.baseline,
            "recall_at_1_delta": self.recall_at_1_delta,
            "recall_at_5_delta": self.recall_at_5_delta,
            "recall_at_10_delta": self.recall_at_10_delta,
            "mrr_delta": self.mrr_delta,
            "median_rank_improvement": self.median_rank_improvement,
            "text_action_beats_baseline": self.text_action_beats_baseline,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionUseBaselineDelta":
        return cls(
            schema_version=str(payload["schema_version"]),
            baseline=str(payload["baseline"]),
            recall_at_1_delta=float(payload["recall_at_1_delta"]),
            recall_at_5_delta=float(payload["recall_at_5_delta"]),
            recall_at_10_delta=float(payload["recall_at_10_delta"]),
            mrr_delta=float(payload["mrr_delta"]),
            median_rank_improvement=float(payload["median_rank_improvement"]),
            text_action_beats_baseline=bool(payload["text_action_beats_baseline"]),
        )


@dataclass(frozen=True)
class ActionUseClaimGate:
    """Machine-readable gate for positive action-conditioning claims."""

    claim_allowed: bool
    checked_baselines: tuple[str, ...]
    required_metrics: tuple[str, ...]
    baseline_deltas: Mapping[str, ActionUseBaselineDelta]
    failure_reasons: tuple[str, ...] = ()
    schema_version: str = ACTION_USE_CLAIM_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_action_use_claim_gate(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claim_allowed": self.claim_allowed,
            "checked_baselines": list(self.checked_baselines),
            "required_metrics": list(self.required_metrics),
            "baseline_deltas": {
                name: delta.to_dict() for name, delta in sorted(self.baseline_deltas.items())
            },
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionUseClaimGate":
        return cls(
            schema_version=str(payload["schema_version"]),
            claim_allowed=bool(payload["claim_allowed"]),
            checked_baselines=tuple(str(name) for name in payload["checked_baselines"]),
            required_metrics=tuple(str(name) for name in payload["required_metrics"]),
            baseline_deltas={
                str(name): ActionUseBaselineDelta.from_dict(_require_mapping(delta, "baseline_deltas"))
                for name, delta in payload.get("baseline_deltas", {}).items()
            },
            failure_reasons=tuple(str(reason) for reason in payload.get("failure_reasons", ())),
        )


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
    metadata: Mapping[str, Any] = field(default_factory=dict)
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
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidatePool":
        entries = tuple(CandidatePoolEntry.from_dict(entry) for entry in payload["entries"])
        if payload.get("entry_count") is not None and int(payload["entry_count"]) != len(entries):
            raise RetrievalEvalError("candidate pool entry_count does not match entries")
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            seed=_optional_non_negative_int(payload.get("seed"), "seed"),
            max_size=_optional_int(payload.get("max_size"), "max_size"),
            excluded_splits=tuple(_split_name(split) for split in payload.get("excluded_splits", ("train",))),
            entries=entries,
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class HardNegativeSamplerConfig:
    """Configuration for deterministic hard-negative sampling."""

    max_negatives: int = 1000
    seed: int = 0
    edit_size_bucket_width: int = 10
    near_before_hamming_threshold: int = 3
    exclude_splits: tuple[str, ...] = ("train",)
    pool_name: str = "hard-1k"

    def __post_init__(self) -> None:
        _positive_int(self.max_negatives, "max_negatives")
        _positive_int(self.edit_size_bucket_width, "edit_size_bucket_width")
        _non_negative_int(self.near_before_hamming_threshold, "near_before_hamming_threshold")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RetrievalEvalError("seed must be an integer")
        if not self.pool_name:
            raise RetrievalEvalError("pool_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_negatives": self.max_negatives,
            "seed": self.seed,
            "edit_size_bucket_width": self.edit_size_bucket_width,
            "near_before_hamming_threshold": self.near_before_hamming_threshold,
            "exclude_splits": list(self.exclude_splits),
            "pool_name": self.pool_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardNegativeSamplerConfig":
        return cls(
            max_negatives=_positive_int(payload.get("max_negatives", 1000), "max_negatives"),
            seed=int(payload.get("seed", 0)),
            edit_size_bucket_width=_positive_int(
                payload.get("edit_size_bucket_width", 10),
                "edit_size_bucket_width",
            ),
            near_before_hamming_threshold=_non_negative_int(
                payload.get("near_before_hamming_threshold", 3),
                "near_before_hamming_threshold",
            ),
            exclude_splits=tuple(_split_name(split) for split in payload.get("exclude_splits", ("train",))),
            pool_name=str(payload.get("pool_name", "hard-1k")),
        )


@dataclass(frozen=True)
class HardNegativeSample:
    """Hard-negative selection and composition for one query target."""

    query_id: str
    target_id: str
    negative_ids: tuple[str, ...]
    requested_negatives: int
    available_negatives: int
    composition: Mapping[str, int]
    rejected: Mapping[str, int]
    schema_version: str = HARD_NEGATIVE_SAMPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HARD_NEGATIVE_SAMPLE_SCHEMA_VERSION:
            raise RetrievalEvalError("unsupported hard-negative sample schema")
        if not self.query_id or not self.target_id:
            raise RetrievalEvalError("hard-negative sample requires query_id and target_id")
        if self.target_id in self.negative_ids:
            raise RetrievalEvalError("hard-negative sample cannot include the true target")
        if len(set(self.negative_ids)) != len(self.negative_ids):
            raise RetrievalEvalError("hard-negative sample ids must be unique")
        _non_negative_int(self.requested_negatives, "requested_negatives")
        _non_negative_int(self.available_negatives, "available_negatives")
        _validate_count_mapping(self.composition, "composition")
        _validate_count_mapping(self.rejected, "rejected")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "target_id": self.target_id,
            "negative_ids": list(self.negative_ids),
            "requested_negatives": self.requested_negatives,
            "available_negatives": self.available_negatives,
            "returned_negatives": len(self.negative_ids),
            "composition": dict(self.composition),
            "rejected": dict(self.rejected),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardNegativeSample":
        return cls(
            schema_version=str(payload["schema_version"]),
            query_id=str(payload["query_id"]),
            target_id=str(payload["target_id"]),
            negative_ids=tuple(str(negative_id) for negative_id in payload["negative_ids"]),
            requested_negatives=_non_negative_int(payload["requested_negatives"], "requested_negatives"),
            available_negatives=_non_negative_int(payload["available_negatives"], "available_negatives"),
            composition={
                str(key): _non_negative_int(value, f"composition.{key}")
                for key, value in payload["composition"].items()
            },
            rejected={
                str(key): _non_negative_int(value, f"rejected.{key}")
                for key, value in payload["rejected"].items()
            },
        )


@dataclass(frozen=True)
class HardNegativeSamplerReport:
    """Aggregate sampler report over one or more query selections."""

    samples: tuple[HardNegativeSample, ...]
    config: HardNegativeSamplerConfig = field(default_factory=HardNegativeSamplerConfig)
    schema_version: str = HARD_NEGATIVE_SAMPLER_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HARD_NEGATIVE_SAMPLER_REPORT_SCHEMA_VERSION:
            raise RetrievalEvalError("unsupported hard-negative sampler report schema")
        if not self.samples:
            raise RetrievalEvalError("hard-negative sampler report requires at least one sample")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_count": len(self.samples),
            "requested_negatives": sum(sample.requested_negatives for sample in self.samples),
            "available_negatives": sum(sample.available_negatives for sample in self.samples),
            "returned_negatives": sum(len(sample.negative_ids) for sample in self.samples),
            "composition": _sum_count_mappings(sample.composition for sample in self.samples),
            "rejected": _sum_count_mappings(sample.rejected for sample in self.samples),
            "config": self.config.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardNegativeSamplerReport":
        samples = tuple(HardNegativeSample.from_dict(sample) for sample in payload["samples"])
        if payload.get("sample_count") is not None and int(payload["sample_count"]) != len(samples):
            raise RetrievalEvalError("hard-negative sampler sample_count does not match samples")
        return cls(
            schema_version=str(payload["schema_version"]),
            config=HardNegativeSamplerConfig.from_dict(payload["config"]),
            samples=samples,
        )


@dataclass(frozen=True)
class ActionContrastPoolConfig:
    """Configuration for deterministic v0.2 action-contrast pools."""

    max_queries: int = 1000
    max_candidates_per_pool: int = 16
    seed: int = 0
    near_before_hamming_threshold: int = 3
    exclude_splits: tuple[str, ...] = ("train",)
    pool_names: tuple[str, ...] = ACTION_CONTRAST_POOL_NAMES

    def __post_init__(self) -> None:
        _positive_int(self.max_queries, "max_queries")
        _positive_int(self.max_candidates_per_pool, "max_candidates_per_pool")
        _non_negative_int(self.near_before_hamming_threshold, "near_before_hamming_threshold")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RetrievalEvalError("seed must be an integer")
        if not self.pool_names:
            raise RetrievalEvalError("action-contrast pool_names must not be empty")
        unsupported = tuple(name for name in self.pool_names if name not in ACTION_CONTRAST_POOL_NAMES)
        if unsupported:
            raise RetrievalEvalError(
                "unsupported action-contrast pool names: " + ", ".join(unsupported)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_queries": self.max_queries,
            "max_candidates_per_pool": self.max_candidates_per_pool,
            "seed": self.seed,
            "near_before_hamming_threshold": self.near_before_hamming_threshold,
            "exclude_splits": list(self.exclude_splits),
            "pool_names": list(self.pool_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionContrastPoolConfig":
        return cls(
            max_queries=_positive_int(payload.get("max_queries", 1000), "max_queries"),
            max_candidates_per_pool=_positive_int(
                payload.get("max_candidates_per_pool", 16),
                "max_candidates_per_pool",
            ),
            seed=int(payload.get("seed", 0)),
            near_before_hamming_threshold=_non_negative_int(
                payload.get("near_before_hamming_threshold", 3),
                "near_before_hamming_threshold",
            ),
            exclude_splits=tuple(_split_name(split) for split in payload.get("exclude_splits", ("train",))),
            pool_names=tuple(str(name) for name in payload.get("pool_names", ACTION_CONTRAST_POOL_NAMES)),
        )


@dataclass(frozen=True)
class ActionContrastPoolSample:
    """Per-query action-contrast candidate ids."""

    query_id: str
    target_id: str
    split: str
    pools: Mapping[str, tuple[str, ...]]
    unavailable_pools: Mapping[str, str]
    no_action_challenge: bool

    def __post_init__(self) -> None:
        if not self.query_id or not self.target_id:
            raise RetrievalEvalError("action-contrast sample requires query_id and target_id")
        if not self.split:
            raise RetrievalEvalError(f"action-contrast sample {self.query_id!r} requires split")
        for name, ids in self.pools.items():
            if name not in ACTION_CONTRAST_POOL_NAMES:
                raise RetrievalEvalError(f"unsupported action-contrast pool name: {name}")
            if len(set(ids)) != len(ids):
                raise RetrievalEvalError(f"action-contrast pool {name!r} has duplicate ids")
            if self.target_id in ids:
                raise RetrievalEvalError(f"action-contrast pool {name!r} cannot include the target id")
            if any(not candidate_id for candidate_id in ids):
                raise RetrievalEvalError(f"action-contrast pool {name!r} contains an empty candidate id")
        _require_json_native(self.unavailable_pools, "action-contrast unavailable_pools")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "target_id": self.target_id,
            "split": self.split,
            "pools": {
                name: list(ids)
                for name, ids in sorted(self.pools.items())
            },
            "unavailable_pools": dict(sorted(self.unavailable_pools.items())),
            "no_action_challenge": self.no_action_challenge,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionContrastPoolSample":
        return cls(
            query_id=str(payload["query_id"]),
            target_id=str(payload["target_id"]),
            split=_split_name(payload["split"]),
            pools={
                str(name): tuple(str(candidate_id) for candidate_id in ids)
                for name, ids in _require_mapping(payload["pools"], "pools").items()
            },
            unavailable_pools={
                str(name): str(reason)
                for name, reason in _require_mapping(
                    payload.get("unavailable_pools", {}),
                    "unavailable_pools",
                ).items()
            },
            no_action_challenge=bool(payload.get("no_action_challenge", False)),
        )


@dataclass(frozen=True)
class ActionContrastPoolReport:
    """Schema-versioned v0.2 action-contrast pool report."""

    samples: tuple[ActionContrastPoolSample, ...]
    pool_counts: Mapping[str, int]
    available_pools: tuple[str, ...]
    unavailable_pools: Mapping[str, str]
    leakage: Mapping[str, Any]
    split_membership_proofs: Mapping[str, Any]
    no_action_challenge: Mapping[str, Any]
    config: ActionContrastPoolConfig = field(default_factory=ActionContrastPoolConfig)
    schema_version: str = ACTION_CONTRAST_POOL_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_action_contrast_pool_report(self)

    @property
    def query_count(self) -> int:
        return len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_count": self.query_count,
            "pool_counts": dict(sorted(self.pool_counts.items())),
            "available_pools": list(self.available_pools),
            "unavailable_pools": dict(sorted(self.unavailable_pools.items())),
            "leakage": dict(self.leakage),
            "split_membership_proofs": dict(self.split_membership_proofs),
            "no_action_challenge": dict(self.no_action_challenge),
            "config": self.config.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
        }

    def summary_dict(self) -> dict[str, Any]:
        """Return a compact JSON-native summary safe to embed in retrieval metadata."""

        return {
            "schema_version": self.schema_version,
            "query_count": self.query_count,
            "pool_counts": dict(sorted(self.pool_counts.items())),
            "available_pools": list(self.available_pools),
            "unavailable_pools": dict(sorted(self.unavailable_pools.items())),
            "leakage": dict(self.leakage),
            "no_action_challenge": dict(self.no_action_challenge),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionContrastPoolReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            config=ActionContrastPoolConfig.from_dict(_require_mapping(payload["config"], "config")),
            samples=tuple(
                ActionContrastPoolSample.from_dict(_require_mapping(sample, "samples[]"))
                for sample in payload["samples"]
            ),
            pool_counts={
                str(name): _non_negative_int(count, f"pool_counts.{name}")
                for name, count in _require_mapping(payload["pool_counts"], "pool_counts").items()
            },
            available_pools=tuple(str(name) for name in payload.get("available_pools", ())),
            unavailable_pools={
                str(name): str(reason)
                for name, reason in _require_mapping(
                    payload.get("unavailable_pools", {}),
                    "unavailable_pools",
                ).items()
            },
            leakage=dict(_require_mapping(payload["leakage"], "leakage")),
            split_membership_proofs=dict(
                _require_mapping(payload["split_membership_proofs"], "split_membership_proofs")
            ),
            no_action_challenge=dict(
                _require_mapping(payload["no_action_challenge"], "no_action_challenge")
            ),
        )


@dataclass(frozen=True)
class RetrievalReport:
    """JSON-native retrieval report matching the v0.1 metric contract."""

    metrics: RetrievalMetrics
    candidate_pool: CandidatePool | None = None
    baselines: Mapping[str, RetrievalMetrics] = field(default_factory=dict)
    slices: Mapping[str, RetrievalMetrics] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    action_use_claim_gate: ActionUseClaimGate | None = None
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
            "baseline_deltas": {
                name: delta.to_dict()
                for name, delta in sorted(self.baseline_deltas.items())
            },
            "action_use_claim_gate": None
            if self.action_use_claim_gate is None
            else self.action_use_claim_gate.to_dict(),
            "slices": {name: metrics.to_dict() for name, metrics in sorted(self.slices.items())},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RetrievalReport":
        candidate_pool_payload = payload.get("candidate_pool")
        metrics = RetrievalMetrics.from_dict(payload["metrics"])
        _validate_top_level_metrics(payload, metrics)
        baselines = {
            str(name): RetrievalMetrics.from_dict(metrics)
            for name, metrics in payload.get("baselines", {}).items()
        }
        gate_payload = payload.get("action_use_claim_gate")
        action_use_claim_gate = (
            None
            if gate_payload is None
            else ActionUseClaimGate.from_dict(_require_mapping(gate_payload, "action_use_claim_gate"))
        )
        report = cls(
            schema_version=str(payload["schema_version"]),
            metrics=metrics,
            candidate_pool=None
            if candidate_pool_payload is None
            else CandidatePool.from_dict(candidate_pool_payload),
            baselines=baselines,
            slices={
                str(name): RetrievalMetrics.from_dict(metrics)
                for name, metrics in payload.get("slices", {}).items()
            },
            metadata=dict(payload.get("metadata", {})),
            action_use_claim_gate=action_use_claim_gate
            if action_use_claim_gate is not None
            else _default_action_use_claim_gate(metrics, baselines),
        )
        validate_retrieval_report(report)
        return report

    @property
    def baseline_deltas(self) -> Mapping[str, ActionUseBaselineDelta]:
        if self.action_use_claim_gate is None:
            return {}
        return self.action_use_claim_gate.baseline_deltas


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


def random_baseline_ranks(
    candidate_ids_by_query: Sequence[Sequence[str]],
    target_ids: Sequence[str],
    *,
    seed: int = 0,
) -> tuple[int, ...]:
    """Return deterministic random-rank baseline targets for each query."""

    if len(candidate_ids_by_query) != len(target_ids):
        raise RetrievalEvalError("candidate_ids_by_query and target_ids must have equal length")
    ranks: list[int] = []
    rng = random.Random(seed)
    for row_index, (candidate_ids, target_id) in enumerate(zip(candidate_ids_by_query, target_ids)):
        candidate_values = tuple(str(candidate_id) for candidate_id in candidate_ids)
        target_value = str(target_id)
        if not candidate_values:
            raise RetrievalEvalError(f"candidate row {row_index} must not be empty")
        if len(set(candidate_values)) != len(candidate_values):
            raise RetrievalEvalError(f"candidate ids for row {row_index} must be unique")
        if candidate_values.count(target_value) != 1:
            raise RetrievalEvalError(f"target id {target_value!r} must appear exactly once in row {row_index}")
        shuffled = list(candidate_values)
        rng.shuffle(shuffled)
        ranks.append(shuffled.index(target_value) + 1)
    return tuple(ranks)


def lexical_baseline_ranks(
    query_texts: Sequence[str],
    candidate_texts_by_query: Sequence[Sequence[str]],
    candidate_ids_by_query: Sequence[Sequence[str]],
    target_ids: Sequence[str],
) -> tuple[int, ...]:
    """Rank targets with a lightweight lexical cosine baseline."""

    if (
        len(query_texts) != len(candidate_texts_by_query)
        or len(query_texts) != len(candidate_ids_by_query)
        or len(query_texts) != len(target_ids)
    ):
        raise RetrievalEvalError("query_texts, candidate_texts, candidate_ids, and target_ids must align")
    score_rows: list[tuple[float, ...]] = []
    for row_index, (query_text, candidate_texts, candidate_ids) in enumerate(
        zip(query_texts, candidate_texts_by_query, candidate_ids_by_query)
    ):
        if len(candidate_texts) != len(candidate_ids):
            raise RetrievalEvalError(f"candidate text row {row_index} length does not match candidate ids")
        score_rows.append(tuple(_lexical_cosine(query_text, candidate_text) for candidate_text in candidate_texts))
    return rank_targets(score_rows, candidate_ids_by_query, target_ids)


def no_action_baseline_ranks(
    score_rows: Sequence[Sequence[float]],
    candidate_ids_by_query: Sequence[Sequence[str]],
    target_ids: Sequence[str],
    *,
    larger_is_better: bool = True,
) -> tuple[int, ...]:
    """Rank targets from a scoring path that intentionally ignores actions."""

    return rank_targets(
        score_rows,
        candidate_ids_by_query,
        target_ids,
        larger_is_better=larger_is_better,
    )


def shuffled_action_baseline_ranks(
    score_rows: Sequence[Sequence[float]],
    candidate_ids_by_query: Sequence[Sequence[str]],
    target_ids: Sequence[str],
    *,
    seed: int = 0,
    larger_is_better: bool = True,
) -> tuple[int, ...]:
    """Rank targets after shuffling action-conditioned score rows across queries."""

    if len(score_rows) != len(candidate_ids_by_query) or len(score_rows) != len(target_ids):
        raise RetrievalEvalError("score_rows, candidate_ids_by_query, and target_ids must have equal length")
    order = _deranged_order(len(score_rows), seed=seed)
    shuffled_rows = tuple(score_rows[index] for index in order)
    return rank_targets(
        shuffled_rows,
        candidate_ids_by_query,
        target_ids,
        larger_is_better=larger_is_better,
    )


def build_baseline_metrics(
    baseline_ranks: Mapping[str, Iterable[int]],
    *,
    candidate_counts: Mapping[str, Iterable[int]] | None = None,
) -> dict[str, RetrievalMetrics]:
    """Build retrieval metrics for named baseline rank streams."""

    metrics: dict[str, RetrievalMetrics] = {}
    for name, ranks in baseline_ranks.items():
        counts = None if candidate_counts is None else candidate_counts.get(name)
        metrics[str(name)] = compute_retrieval_metrics(ranks, candidate_counts=counts)
    return metrics


def validate_required_headline_baselines(
    report: RetrievalReport,
    *,
    required: Sequence[str] = REQUIRED_HEADLINE_BASELINES,
) -> RetrievalReport:
    """Reject headline retrieval reports that omit required baseline metrics."""

    missing = tuple(name for name in required if name not in report.baselines)
    if missing:
        raise RetrievalEvalError(f"headline retrieval report missing required baselines: {', '.join(missing)}")
    for name in required:
        validate_retrieval_metrics(report.baselines[name])
    return report


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


def sample_hard_negatives(
    query: Any,
    candidates: Iterable[Any],
    *,
    target_id: str | None = None,
    config: HardNegativeSamplerConfig | None = None,
) -> HardNegativeSample:
    """Select deterministic hard negatives for a held-out query."""

    config = HardNegativeSamplerConfig() if config is None else config
    query_entry = _coerce_pool_entry(query)
    target_value = query_entry.transition_id if target_id is None else str(target_id)
    excluded = frozenset(config.exclude_splits) | TRAIN_SPLITS
    if query_entry.split in excluded:
        raise RetrievalEvalError(f"hard-negative query {query_entry.transition_id!r} must be held out")

    rejected = {"true_target": 0, "train_leakage": 0}
    candidate_entries = _unique_candidate_entries(candidates)
    scored: list[_ScoredHardNegative] = []
    rng = random.Random(config.seed + _stable_seed_offset(query_entry.transition_id))

    for candidate in sorted(candidate_entries, key=lambda entry: entry.transition_id):
        if candidate.transition_id == target_value:
            rejected["true_target"] += 1
            continue
        if candidate.split in excluded:
            rejected["train_leakage"] += 1
            continue
        scored.append(_score_hard_negative(query_entry, candidate, config, tie_breaker=rng.random()))

    scored.sort(key=lambda item: item.sort_key)
    selected = tuple(item for item in scored[: config.max_negatives])
    return HardNegativeSample(
        query_id=query_entry.transition_id,
        target_id=target_value,
        negative_ids=tuple(item.entry.transition_id for item in selected),
        requested_negatives=config.max_negatives,
        available_negatives=len(scored),
        composition=_hard_negative_composition(selected),
        rejected=rejected,
    )


def build_hard_candidate_pool(
    query: Any,
    candidates: Iterable[Any],
    *,
    target_id: str | None = None,
    config: HardNegativeSamplerConfig | None = None,
) -> tuple[CandidatePool, HardNegativeSample]:
    """Return a query-specific hard pool containing target plus sampled negatives."""

    config = HardNegativeSamplerConfig() if config is None else config
    query_entry = _coerce_pool_entry(query)
    target_value = query_entry.transition_id if target_id is None else str(target_id)
    candidate_entries = _unique_candidate_entries(candidates)
    by_id = {entry.transition_id: entry for entry in candidate_entries}
    if target_value not in by_id:
        if target_value == query_entry.transition_id:
            by_id[target_value] = query_entry
        else:
            raise RetrievalEvalError(f"target id not found in candidates: {target_value}")

    sample = sample_hard_negatives(
        query_entry,
        candidate_entries,
        target_id=target_value,
        config=config,
    )
    entries = (by_id[target_value], *(by_id[negative_id] for negative_id in sample.negative_ids))
    pool = CandidatePool(
        name=config.pool_name,
        entries=entries,
        seed=config.seed,
        max_size=config.max_negatives + 1,
        excluded_splits=config.exclude_splits,
        metadata={
            "query_id": sample.query_id,
            "target_id": sample.target_id,
            "negative_count": len(sample.negative_ids),
            "sampler": sample.to_dict(),
        },
    )
    return pool, sample


def build_hard_negative_sampler_report(
    samples: Iterable[HardNegativeSample],
    *,
    config: HardNegativeSamplerConfig | None = None,
) -> HardNegativeSamplerReport:
    """Build an aggregate report for hard-negative sampler composition."""

    config = HardNegativeSamplerConfig() if config is None else config
    return HardNegativeSamplerReport(samples=tuple(samples), config=config)


def build_action_contrast_pool_report(
    rows: Iterable[Any],
    *,
    query_ids: Sequence[str] | None = None,
    config: ActionContrastPoolConfig | None = None,
) -> ActionContrastPoolReport:
    """Build deterministic v0.2 pools where action text should matter."""

    config = ActionContrastPoolConfig() if config is None else config
    entries = _unique_candidate_entries(rows)
    by_id = {entry.transition_id: entry for entry in entries}
    excluded = frozenset(config.exclude_splits) | TRAIN_SPLITS
    heldout_entries = tuple(entry for entry in entries if entry.split not in excluded)
    if not heldout_entries:
        raise RetrievalEvalError("action-contrast pools require at least one held-out row")

    if query_ids is None:
        ordered_queries = sorted(heldout_entries, key=lambda entry: entry.transition_id)
        rng = random.Random(config.seed)
        rng.shuffle(ordered_queries)
        queries = tuple(ordered_queries[: config.max_queries])
    else:
        ids = tuple(str(query_id) for query_id in query_ids)
        missing = [query_id for query_id in ids if query_id not in by_id]
        if missing:
            raise RetrievalEvalError(f"action-contrast query ids not found: {', '.join(missing)}")
        queries = tuple(by_id[query_id] for query_id in ids[: config.max_queries])

    samples: list[ActionContrastPoolSample] = []
    for query in queries:
        if query.split in excluded:
            raise RetrievalEvalError(f"action-contrast query {query.transition_id!r} must be held out")
        pools, unavailable = _action_contrast_pools_for_query(query, heldout_entries, config=config)
        samples.append(
            ActionContrastPoolSample(
                query_id=query.transition_id,
                target_id=query.transition_id,
                split=query.split,
                pools=pools,
                unavailable_pools=unavailable,
                no_action_challenge=_is_no_action_challenge(query, pools, by_id),
            )
        )

    pool_counts = {
        pool_name: sum(len(sample.pools.get(pool_name, ())) for sample in samples)
        for pool_name in config.pool_names
    }
    available_pools = tuple(pool_name for pool_name in config.pool_names if pool_counts[pool_name] > 0)
    unavailable_pools = {
        pool_name: "no held-out candidates matched this action-contrast predicate"
        for pool_name in config.pool_names
        if pool_counts[pool_name] == 0
    }
    split_membership_proofs = _action_contrast_split_membership_proofs(
        samples,
        entries_by_id=by_id,
        pool_names=config.pool_names,
    )
    selected_train_rows = sum(
        int(proof["split_counts"].get("train", 0))
        for proof in split_membership_proofs.values()
        if isinstance(proof, Mapping)
    )
    if selected_train_rows:
        raise RetrievalEvalError("action-contrast pool selected train split rows")
    leakage = {
        "excluded_splits": list(sorted(excluded)),
        "input_train_rows": sum(1 for entry in entries if entry.split in excluded),
        "query_train_rows": sum(1 for query in queries if query.split in excluded),
        "selected_train_rows": selected_train_rows,
        "leakage_detected": bool(selected_train_rows),
    }
    no_action_challenge = {
        "same_before_multi_action_query_count": sum(
            1 for sample in samples if sample.no_action_challenge
        ),
        "exact_same_before_query_count": sum(
            1 for sample in samples if sample.pools.get("exact_same_before")
        ),
        "synthetic_controlled_same_before_query_count": sum(
            1
            for sample in samples
            if sample.no_action_challenge and _sample_has_synthetic_entries(sample, by_id)
        ),
    }
    no_action_challenge["no_action_prior_insufficient"] = (
        no_action_challenge["same_before_multi_action_query_count"] > 0
    )
    return ActionContrastPoolReport(
        samples=tuple(samples),
        pool_counts=pool_counts,
        available_pools=available_pools,
        unavailable_pools=unavailable_pools,
        leakage=leakage,
        split_membership_proofs=split_membership_proofs,
        no_action_challenge=no_action_challenge,
        config=config,
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

    metrics = compute_retrieval_metrics(ranks, candidate_counts=candidate_counts)
    baseline_metrics = _coerce_metrics_mapping(baselines)
    report = RetrievalReport(
        metrics=metrics,
        candidate_pool=candidate_pool,
        baselines=baseline_metrics,
        slices=_coerce_metrics_mapping(slices),
        metadata={} if metadata is None else dict(metadata),
        action_use_claim_gate=_default_action_use_claim_gate(metrics, baseline_metrics),
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
    if pool.seed is not None:
        _non_negative_int(pool.seed, "candidate pool seed")
    if pool.max_size is not None and pool.max_size <= 0:
        raise RetrievalEvalError("candidate pool max_size must be positive when set")

    ids = [entry.transition_id for entry in pool.entries]
    if len(set(ids)) != len(ids):
        raise RetrievalEvalError("candidate pool transition ids must be unique")

    excluded = frozenset(pool.excluded_splits) | TRAIN_SPLITS
    leaked = [entry.transition_id for entry in pool.entries if entry.split in excluded]
    if leaked:
        raise RetrievalEvalError(f"candidate pool includes training rows: {', '.join(leaked)}")
    _require_json_native(pool.metadata, "candidate pool metadata")
    return pool


def validate_action_contrast_pool_report(report: ActionContrastPoolReport) -> ActionContrastPoolReport:
    """Validate v0.2 action-contrast pool report payloads."""

    if report.schema_version != ACTION_CONTRAST_POOL_REPORT_SCHEMA_VERSION:
        raise RetrievalEvalError(
            "unsupported action-contrast pool report schema; "
            f"expected {ACTION_CONTRAST_POOL_REPORT_SCHEMA_VERSION!r}, got {report.schema_version!r}"
        )
    if not report.samples:
        raise RetrievalEvalError("action-contrast pool report requires at least one sample")
    if set(report.pool_counts) != set(report.config.pool_names):
        raise RetrievalEvalError("action-contrast pool_counts must cover every configured pool")
    for pool_name, count in report.pool_counts.items():
        _non_negative_int(count, f"pool_counts.{pool_name}")
    for pool_name in report.available_pools:
        if pool_name not in report.config.pool_names:
            raise RetrievalEvalError(f"unknown available action-contrast pool: {pool_name}")
        if report.pool_counts.get(pool_name, 0) <= 0:
            raise RetrievalEvalError(f"available action-contrast pool has zero count: {pool_name}")
    for pool_name in report.unavailable_pools:
        if pool_name not in report.config.pool_names:
            raise RetrievalEvalError(f"unknown unavailable action-contrast pool: {pool_name}")
    for sample in report.samples:
        if sample.split in (set(report.config.exclude_splits) | TRAIN_SPLITS):
            raise RetrievalEvalError(f"action-contrast sample {sample.query_id!r} is not held out")
    leakage = _require_mapping(report.leakage, "action-contrast leakage")
    selected_train_rows = _non_negative_int(
        leakage.get("selected_train_rows", 0),
        "leakage.selected_train_rows",
    )
    if selected_train_rows:
        raise RetrievalEvalError("action-contrast leakage selected train rows")
    _require_json_native(report.to_dict(), "action-contrast pool report")
    return report


def validate_action_contrast_pool_report_payload(payload: Mapping[str, Any]) -> ActionContrastPoolReport:
    """Return a validated action-contrast pool report from a JSON payload."""

    return ActionContrastPoolReport.from_dict(payload)


def write_action_contrast_pool_report(report: ActionContrastPoolReport, path: Path) -> None:
    """Write a v0.2 action-contrast pool report JSON file."""

    validate_action_contrast_pool_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_action_contrast_pool_report(path: Path) -> ActionContrastPoolReport:
    """Read and validate a v0.2 action-contrast pool report."""

    return validate_action_contrast_pool_report_payload(json.loads(path.read_text()))


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


def build_action_use_baseline_deltas(
    metrics: RetrievalMetrics,
    baselines: Mapping[str, RetrievalMetrics],
) -> dict[str, ActionUseBaselineDelta]:
    """Compute text-action minus baseline retrieval deltas."""

    return {
        name: ActionUseBaselineDelta(
            baseline=name,
            recall_at_1_delta=metrics.recall_at_1 - baseline.recall_at_1,
            recall_at_5_delta=metrics.recall_at_5 - baseline.recall_at_5,
            recall_at_10_delta=metrics.recall_at_10 - baseline.recall_at_10,
            mrr_delta=metrics.mrr - baseline.mrr,
            median_rank_improvement=baseline.median_rank - metrics.median_rank,
            text_action_beats_baseline=(
                metrics.recall_at_1 > baseline.recall_at_1
                and metrics.mrr > baseline.mrr
            ),
        )
        for name, baseline in baselines.items()
    }


def build_action_use_claim_gate(
    metrics: RetrievalMetrics,
    baselines: Mapping[str, RetrievalMetrics],
    *,
    required_baselines: Sequence[str] = REQUIRED_HEADLINE_BASELINES,
    required_metrics: Sequence[str] = ACTION_USE_REQUIRED_METRICS,
    additional_failure_reasons: Sequence[str] = (),
) -> ActionUseClaimGate:
    """Build the positive action-conditioning claim gate for retrieval evidence."""

    required = tuple(str(name) for name in required_baselines)
    required_metric_names = tuple(str(name) for name in required_metrics)
    unsupported_metrics = tuple(
        name for name in required_metric_names if name not in ACTION_USE_REQUIRED_METRICS
    )
    if unsupported_metrics:
        raise RetrievalEvalError(
            "unsupported action-use claim metrics: " + ", ".join(unsupported_metrics)
        )

    deltas = build_action_use_baseline_deltas(metrics, baselines)
    failure_reasons: list[str] = []
    for name in required:
        if name not in baselines:
            failure_reasons.append(f"missing_baseline:{name}")
            continue
        delta = deltas[name]
        if not delta.text_action_beats_baseline:
            if name == "no_action":
                failure_reasons.append(
                    "no_action_dominance:"
                    "text_action_recall_at_1_or_mrr_not_strictly_above_no_action"
                )
            else:
                failure_reasons.append(f"baseline_not_beaten:{name}")
    for reason in additional_failure_reasons:
        reason_text = str(reason)
        if not reason_text:
            raise RetrievalEvalError("action-use claim failure reasons must not be empty")
        failure_reasons.append(reason_text)
    return ActionUseClaimGate(
        claim_allowed=not failure_reasons,
        checked_baselines=required,
        required_metrics=required_metric_names,
        baseline_deltas=deltas,
        failure_reasons=tuple(failure_reasons),
    )


def validate_action_use_baseline_delta(delta: ActionUseBaselineDelta) -> ActionUseBaselineDelta:
    """Validate one action-use baseline delta."""

    if delta.schema_version != ACTION_USE_BASELINE_DELTA_SCHEMA_VERSION:
        raise RetrievalEvalError(
            "unsupported action-use baseline delta schema; "
            f"expected {ACTION_USE_BASELINE_DELTA_SCHEMA_VERSION!r}, got {delta.schema_version!r}"
        )
    if not delta.baseline:
        raise RetrievalEvalError("action-use baseline delta name must not be empty")
    for name in (
        "recall_at_1_delta",
        "recall_at_5_delta",
        "recall_at_10_delta",
        "mrr_delta",
        "median_rank_improvement",
    ):
        value = float(getattr(delta, name))
        if not math.isfinite(value):
            raise RetrievalEvalError(f"{name} must be finite")
    return delta


def validate_action_use_claim_gate(gate: ActionUseClaimGate) -> ActionUseClaimGate:
    """Validate action-use claim gate payloads."""

    if gate.schema_version != ACTION_USE_CLAIM_GATE_SCHEMA_VERSION:
        raise RetrievalEvalError(
            "unsupported action-use claim gate schema; "
            f"expected {ACTION_USE_CLAIM_GATE_SCHEMA_VERSION!r}, got {gate.schema_version!r}"
        )
    if not gate.checked_baselines:
        raise RetrievalEvalError("action-use claim gate must check at least one baseline")
    if not gate.required_metrics:
        raise RetrievalEvalError("action-use claim gate must list required metrics")
    for name in gate.checked_baselines:
        if not name:
            raise RetrievalEvalError("action-use claim gate checked baseline names must not be empty")
    for name in gate.required_metrics:
        if name not in ACTION_USE_REQUIRED_METRICS:
            raise RetrievalEvalError(f"unsupported action-use required metric: {name}")
    for name, delta in gate.baseline_deltas.items():
        if not name:
            raise RetrievalEvalError("action-use baseline delta map names must not be empty")
        if name != delta.baseline:
            raise RetrievalEvalError("action-use baseline delta map key must match delta.baseline")
        validate_action_use_baseline_delta(delta)
    for reason in gate.failure_reasons:
        if not reason:
            raise RetrievalEvalError("action-use claim gate failure reasons must not be empty")
    if gate.claim_allowed and gate.failure_reasons:
        raise RetrievalEvalError("claim_allowed cannot be true when failure_reasons are present")
    if not gate.claim_allowed and not gate.failure_reasons:
        raise RetrievalEvalError("claim_allowed=false requires at least one failure reason")
    return gate


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
    if report.action_use_claim_gate is not None:
        validate_action_use_claim_gate(report.action_use_claim_gate)
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


def _default_action_use_claim_gate(
    metrics: RetrievalMetrics,
    baselines: Mapping[str, RetrievalMetrics],
) -> ActionUseClaimGate | None:
    if not baselines:
        return None
    return build_action_use_claim_gate(metrics, baselines)


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


@dataclass(frozen=True)
class _ScoredHardNegative:
    entry: CandidatePoolEntry
    primary_reasons: tuple[str, ...]
    similarity: float | None
    tie_breaker: float

    @property
    def sort_key(self) -> tuple[int, float, float, str]:
        similarity = float("-inf") if self.similarity is None else self.similarity
        return (-len(self.primary_reasons), -similarity, self.tie_breaker, self.entry.transition_id)


def _score_hard_negative(
    query: CandidatePoolEntry,
    candidate: CandidatePoolEntry,
    config: HardNegativeSamplerConfig,
    *,
    tie_breaker: float,
) -> _ScoredHardNegative:
    reasons: list[str] = []
    query_before_hash = _metadata_string(query.metadata, "state_before_hash")
    candidate_before_hash = _metadata_string(candidate.metadata, "state_before_hash")
    query_after_hash = _metadata_string(query.metadata, "state_after_hash")
    candidate_after_hash = _metadata_string(candidate.metadata, "state_after_hash")
    different_after = (
        query_after_hash is not None
        and candidate_after_hash is not None
        and query_after_hash != candidate_after_hash
    )
    if (
        query_before_hash is not None
        and candidate_before_hash is not None
        and query_before_hash == candidate_before_hash
        and different_after
    ):
        reasons.append("same_before_different_after")
    elif different_after and _near_before_match(query, candidate, config=config):
        reasons.append("near_before_different_after")
    if query.path and candidate.path == query.path and (not query.repo or candidate.repo == query.repo):
        reasons.append("same_file")
    if candidate.source == query.source:
        reasons.append("same_source")
    if _edit_size_bucket(candidate.edit_size, config.edit_size_bucket_width) == _edit_size_bucket(
        query.edit_size,
        config.edit_size_bucket_width,
    ):
        reasons.append("same_edit_size_bucket")
    query_cluster = _action_cluster(query)
    candidate_cluster = _action_cluster(candidate)
    if query_cluster is not None and query_cluster == candidate_cluster:
        reasons.append("same_action_cluster")
    return _ScoredHardNegative(
        entry=candidate,
        primary_reasons=tuple(reasons),
        similarity=_entry_similarity(candidate),
        tie_breaker=tie_breaker,
    )


def _action_contrast_pools_for_query(
    query: CandidatePoolEntry,
    candidates: Sequence[CandidatePoolEntry],
    *,
    config: ActionContrastPoolConfig,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    pools: dict[str, tuple[str, ...]] = {}
    unavailable: dict[str, str] = {}
    for pool_name in config.pool_names:
        pool_candidates = _select_action_contrast_candidates(
            query,
            candidates,
            pool_name=pool_name,
            config=config,
        )
        if pool_candidates:
            pools[pool_name] = pool_candidates
        else:
            unavailable[pool_name] = "no held-out candidates matched this action-contrast predicate"
    return pools, unavailable


def _select_action_contrast_candidates(
    query: CandidatePoolEntry,
    candidates: Sequence[CandidatePoolEntry],
    *,
    pool_name: str,
    config: ActionContrastPoolConfig,
) -> tuple[str, ...]:
    if pool_name == "random":
        matches = [
            candidate
            for candidate in candidates
            if candidate.transition_id != query.transition_id
        ]
    else:
        matches = [
            candidate
            for candidate in candidates
            if candidate.transition_id != query.transition_id
            and _matches_action_contrast_pool(query, candidate, pool_name=pool_name, config=config)
        ]
    ordered = sorted(matches, key=lambda entry: entry.transition_id)
    if len(ordered) > config.max_candidates_per_pool:
        rng = random.Random(config.seed + _stable_seed_offset(f"{query.transition_id}:{pool_name}"))
        rng.shuffle(ordered)
        ordered = sorted(ordered[: config.max_candidates_per_pool], key=lambda entry: entry.transition_id)
    return tuple(entry.transition_id for entry in ordered)


def _matches_action_contrast_pool(
    query: CandidatePoolEntry,
    candidate: CandidatePoolEntry,
    *,
    pool_name: str,
    config: ActionContrastPoolConfig,
) -> bool:
    different_after = _different_after(query, candidate)
    if pool_name == "exact_same_before":
        return _same_before(query, candidate) and different_after
    if pool_name == "near_before":
        return not _same_before(query, candidate) and different_after and _near_before_match(
            query,
            candidate,
            config=HardNegativeSamplerConfig(
                near_before_hamming_threshold=config.near_before_hamming_threshold,
            ),
        )
    if pool_name == "same_file":
        return (
            bool(query.path)
            and candidate.path == query.path
            and (not query.repo or candidate.repo == query.repo)
            and different_after
        )
    if pool_name == "action_cluster":
        query_cluster = _action_cluster(query)
        return query_cluster is not None and query_cluster == _action_cluster(candidate) and different_after
    if pool_name == "edit_shape":
        query_shape = _edit_shape(query)
        return query_shape is not None and query_shape == _edit_shape(candidate) and different_after
    if pool_name == "mutation":
        return _is_mutation_candidate(candidate) and different_after
    raise RetrievalEvalError(f"unsupported action-contrast pool name: {pool_name}")


def _action_contrast_split_membership_proofs(
    samples: Sequence[ActionContrastPoolSample],
    *,
    entries_by_id: Mapping[str, CandidatePoolEntry],
    pool_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    proofs: dict[str, dict[str, Any]] = {}
    for pool_name in pool_names:
        ids: list[str] = []
        split_counts: dict[str, int] = {}
        query_count = 0
        for sample in samples:
            pool_ids = sample.pools.get(pool_name, ())
            if not pool_ids:
                continue
            query_count += 1
            ids.extend(pool_ids)
            for candidate_id in pool_ids:
                entry = entries_by_id[candidate_id]
                split_counts[entry.split] = split_counts.get(entry.split, 0) + 1
        proofs[pool_name] = {
            "query_count": query_count,
            "candidate_count": len(ids),
            "split_counts": dict(sorted(split_counts.items())),
            "example_transition_ids": sorted(set(ids))[:10],
        }
    return proofs


def _is_no_action_challenge(
    query: CandidatePoolEntry,
    pools: Mapping[str, Sequence[str]],
    entries_by_id: Mapping[str, CandidatePoolEntry],
) -> bool:
    exact_ids = pools.get("exact_same_before", ())
    if not exact_ids:
        return False
    action_clusters = {_action_cluster(query)}
    action_clusters.update(_action_cluster(entries_by_id[candidate_id]) for candidate_id in exact_ids)
    action_clusters.discard(None)
    return len(action_clusters) >= 2


def _sample_has_synthetic_entries(
    sample: ActionContrastPoolSample,
    entries_by_id: Mapping[str, CandidatePoolEntry],
) -> bool:
    ids = (sample.query_id, *(candidate_id for ids in sample.pools.values() for candidate_id in ids))
    return any(entries_by_id[transition_id].source == "synthetic" for transition_id in ids)


def _same_before(query: CandidatePoolEntry, candidate: CandidatePoolEntry) -> bool:
    query_before_hash = _metadata_string(query.metadata, "state_before_hash")
    candidate_before_hash = _metadata_string(candidate.metadata, "state_before_hash")
    return (
        query_before_hash is not None
        and candidate_before_hash is not None
        and query_before_hash == candidate_before_hash
    )


def _different_after(query: CandidatePoolEntry, candidate: CandidatePoolEntry) -> bool:
    query_after_hash = _metadata_string(query.metadata, "state_after_hash")
    candidate_after_hash = _metadata_string(candidate.metadata, "state_after_hash")
    if query_after_hash is None or candidate_after_hash is None:
        return query.transition_id != candidate.transition_id
    return query_after_hash != candidate_after_hash


def _edit_shape(entry: CandidatePoolEntry) -> str | None:
    for key in ("diff_shape", "edit_shape", "edit_size_bucket"):
        value = entry.metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _is_mutation_candidate(entry: CandidatePoolEntry) -> bool:
    kind = str(entry.metadata.get("candidate_kind", "")).casefold()
    if kind in {"mutation", "mutated_after", "bug_injection"}:
        return True
    return bool(entry.metadata.get("is_mutation", False))


def _hard_negative_composition(selected: Sequence[_ScoredHardNegative]) -> dict[str, int]:
    composition = {
        "same_before_different_after": 0,
        "near_before_different_after": 0,
        "same_file": 0,
        "same_source": 0,
        "same_edit_size_bucket": 0,
        "same_action_cluster": 0,
        "action_discriminative": 0,
        "similarity_ranked": 0,
        "fallback": 0,
    }
    for item in selected:
        for reason in item.primary_reasons:
            composition[reason] += 1
        if any(
            reason in item.primary_reasons
            for reason in (
                "same_before_different_after",
                "near_before_different_after",
                "same_file",
                "same_action_cluster",
            )
        ):
            composition["action_discriminative"] += 1
        if item.similarity is not None:
            composition["similarity_ranked"] += 1
        if not item.primary_reasons and item.similarity is None:
            composition["fallback"] += 1
    return composition


def _near_before_match(
    query: CandidatePoolEntry,
    candidate: CandidatePoolEntry,
    *,
    config: HardNegativeSamplerConfig,
) -> bool:
    query_simhash = _metadata_string(query.metadata, "state_before_simhash")
    candidate_simhash = _metadata_string(candidate.metadata, "state_before_simhash")
    if query_simhash is None or candidate_simhash is None:
        return False
    try:
        distance = (int(query_simhash, 16) ^ int(candidate_simhash, 16)).bit_count()
    except ValueError:
        return False
    return distance <= config.near_before_hamming_threshold


def _unique_candidate_entries(rows: Iterable[Any]) -> tuple[CandidatePoolEntry, ...]:
    entries: dict[str, CandidatePoolEntry] = {}
    for row in rows:
        entry = _coerce_pool_entry(row)
        if entry.transition_id in entries:
            raise RetrievalEvalError(f"duplicate candidate row id: {entry.transition_id}")
        entries[entry.transition_id] = entry
    return tuple(entries.values())


def _lexical_cosine(query_text: str, candidate_text: str) -> float:
    query_counts = _term_counts(query_text)
    candidate_counts = _term_counts(candidate_text)
    if not query_counts or not candidate_counts:
        return 0.0
    dot = sum(count * candidate_counts.get(term, 0) for term, count in query_counts.items())
    query_norm = math.sqrt(sum(count * count for count in query_counts.values()))
    candidate_norm = math.sqrt(sum(count * count for count in candidate_counts.values()))
    return dot / max(query_norm * candidate_norm, 1e-12)


def _term_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in _LEXICAL_TOKEN_RE.findall(text.lower()):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _deranged_order(length: int, *, seed: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    order = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(order)
    if length > 1 and any(index == value for index, value in enumerate(order)):
        order = order[1:] + order[:1]
        if any(index == value for index, value in enumerate(order)):
            order = list(range(1, length)) + [0]
    return tuple(order)


def _action_cluster(entry: CandidatePoolEntry) -> str | None:
    for key in ("action_cluster", "weak_action_cluster", "action_abs_cluster"):
        value = entry.metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _metadata_string(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _entry_similarity(entry: CandidatePoolEntry) -> float | None:
    for key in ("similarity", "similarity_to_query", "state_similarity", "lexical_similarity"):
        if key not in entry.metadata or entry.metadata[key] is None:
            continue
        value = float(entry.metadata[key])
        if not math.isfinite(value):
            raise RetrievalEvalError(f"candidate {entry.transition_id!r} has non-finite similarity")
        return value
    return None


def _edit_size_bucket(edit_size: int, bucket_width: int) -> int:
    return int(edit_size) // bucket_width


def _stable_seed_offset(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _validate_count_mapping(value: Mapping[str, int], name: str) -> None:
    for key, count in value.items():
        if not key:
            raise RetrievalEvalError(f"{name} keys must not be empty")
        _non_negative_int(count, f"{name}.{key}")


def _sum_count_mappings(values: Iterable[Mapping[str, int]]) -> dict[str, int]:
    summed: dict[str, int] = {}
    for value in values:
        for key, count in value.items():
            summed[key] = summed.get(key, 0) + int(count)
    return dict(sorted(summed.items()))


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


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name)


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RetrievalEvalError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RetrievalEvalError(f"{name} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise RetrievalEvalError(f"{name} must be a non-negative integer")
    return result


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


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetrievalEvalError(f"{name} must be a JSON object")
    return value


def _require_json_native(value: Any, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RetrievalEvalError(f"{name} must be JSON-native") from exc
