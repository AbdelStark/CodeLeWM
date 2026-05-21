"""Downstream candidate-reranking benchmark schema and claim gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION = "codelewm.downstream_rerank_benchmark.v1"
DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION = "codelewm.downstream_rerank_report.v1"
DOWNSTREAM_RERANK_CLAIM_GATE_SCHEMA_VERSION = "codelewm.downstream_rerank_claim_gate.v1"
DOWNSTREAM_MIN_LABELED_EXAMPLES = 100
DOWNSTREAM_REQUIRED_BASELINES: tuple[str, ...] = (
    "llm_order",
    "random",
    "lexical",
    "no_action",
    "codelewm",
    "retrieval_prior",
    "score_ensemble",
)
DOWNSTREAM_REQUIRED_METRICS: tuple[str, ...] = (
    "pass_at_1",
    "pass_at_k",
    "mrr",
    "valid_patch_rate",
    "check_pass_rate",
)
DownstreamCandidateLabel = Literal["pass", "fail", "unknown"]
DownstreamCheckStatus = Literal["pass", "fail", "not_run", "not_applicable"]


class DownstreamBenchmarkError(ValueError):
    """Raised when downstream benchmark payloads or claim gates are invalid."""


@dataclass(frozen=True)
class DownstreamCandidate:
    """One candidate patch or after-state in a labeled downstream task."""

    candidate_id: str
    llm_rank: int
    label: DownstreamCandidateLabel
    patch_path: str | None = None
    after_state_path: str | None = None
    static_check: DownstreamCheckStatus = "not_run"
    test_check: DownstreamCheckStatus = "not_run"
    source: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise DownstreamBenchmarkError("candidate_id must not be empty")
        if self.llm_rank < 1:
            raise DownstreamBenchmarkError("llm_rank must be >= 1")
        if self.label not in {"pass", "fail", "unknown"}:
            raise DownstreamBenchmarkError("candidate label must be pass, fail, or unknown")
        if not self.patch_path and not self.after_state_path:
            raise DownstreamBenchmarkError("candidate must include patch_path or after_state_path")
        if self.static_check not in {"pass", "fail", "not_run", "not_applicable"}:
            raise DownstreamBenchmarkError("static_check has unsupported status")
        if self.test_check not in {"pass", "fail", "not_run", "not_applicable"}:
            raise DownstreamBenchmarkError("test_check has unsupported status")
        _ensure_json_native(self.source, "source")
        _ensure_json_native(self.provenance, "provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "llm_rank": self.llm_rank,
            "label": self.label,
            "patch_path": self.patch_path,
            "after_state_path": self.after_state_path,
            "static_check": self.static_check,
            "test_check": self.test_check,
            "source": dict(self.source),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class DownstreamTask:
    """One labeled reranking task."""

    task_id: str
    task_type: str
    prompt: str
    before_path: str
    candidates: tuple[DownstreamCandidate, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise DownstreamBenchmarkError("task_id must not be empty")
        if not self.task_type:
            raise DownstreamBenchmarkError("task_type must not be empty")
        if not self.prompt:
            raise DownstreamBenchmarkError("prompt must not be empty")
        if not self.before_path:
            raise DownstreamBenchmarkError("before_path must not be empty")
        if len(self.candidates) < 2:
            raise DownstreamBenchmarkError("each downstream task needs at least two candidates")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise DownstreamBenchmarkError("candidate IDs must be unique within a task")
        _ensure_json_native(self.provenance, "provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "before_path": self.before_path,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class DownstreamRerankBenchmark:
    """Schema-versioned downstream reranking benchmark payload."""

    benchmark_id: str
    tasks: tuple[DownstreamTask, ...]
    required_baselines: tuple[str, ...] = DOWNSTREAM_REQUIRED_BASELINES
    required_metrics: tuple[str, ...] = DOWNSTREAM_REQUIRED_METRICS
    min_labeled_examples: int = DOWNSTREAM_MIN_LABELED_EXAMPLES
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION:
            raise DownstreamBenchmarkError("unsupported downstream benchmark schema_version")
        if not self.benchmark_id:
            raise DownstreamBenchmarkError("benchmark_id must not be empty")
        if not self.tasks:
            raise DownstreamBenchmarkError("benchmark must include at least one task")
        missing_baselines = set(DOWNSTREAM_REQUIRED_BASELINES) - set(self.required_baselines)
        if missing_baselines:
            raise DownstreamBenchmarkError(
                "required_baselines is missing: " + ", ".join(sorted(missing_baselines))
            )
        missing_metrics = set(DOWNSTREAM_REQUIRED_METRICS) - set(self.required_metrics)
        if missing_metrics:
            raise DownstreamBenchmarkError(
                "required_metrics is missing: " + ", ".join(sorted(missing_metrics))
            )
        if self.min_labeled_examples < DOWNSTREAM_MIN_LABELED_EXAMPLES:
            raise DownstreamBenchmarkError("min_labeled_examples must be at least 100")
        _ensure_json_native(self.provenance, "provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "tasks": [task.to_dict() for task in self.tasks],
            "required_baselines": list(self.required_baselines),
            "required_metrics": list(self.required_metrics),
            "min_labeled_examples": self.min_labeled_examples,
            "provenance": dict(self.provenance),
        }


def downstream_rerank_benchmark_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for downstream reranking benchmark payloads."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION,
        "type": "object",
        "required": [
            "schema_version",
            "benchmark_id",
            "tasks",
            "required_baselines",
            "required_metrics",
            "min_labeled_examples",
            "provenance",
        ],
        "properties": {
            "schema_version": {"const": DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION},
            "benchmark_id": {"type": "string", "minLength": 1},
            "min_labeled_examples": {"type": "integer", "minimum": DOWNSTREAM_MIN_LABELED_EXAMPLES},
            "required_baselines": {
                "type": "array",
                "items": {"type": "string", "enum": list(DOWNSTREAM_REQUIRED_BASELINES)},
            },
            "required_metrics": {
                "type": "array",
                "items": {"type": "string", "enum": list(DOWNSTREAM_REQUIRED_METRICS)},
            },
            "tasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "task_id",
                        "task_type",
                        "prompt",
                        "before_path",
                        "candidates",
                        "provenance",
                    ],
                    "properties": {
                        "task_id": {"type": "string", "minLength": 1},
                        "task_type": {"type": "string", "minLength": 1},
                        "prompt": {"type": "string", "minLength": 1},
                        "before_path": {"type": "string", "minLength": 1},
                        "candidates": {
                            "type": "array",
                            "minItems": 2,
                            "items": {
                                "type": "object",
                                "required": [
                                    "candidate_id",
                                    "llm_rank",
                                    "label",
                                    "patch_path",
                                    "after_state_path",
                                    "static_check",
                                    "test_check",
                                    "source",
                                    "provenance",
                                ],
                                "properties": {
                                    "candidate_id": {"type": "string", "minLength": 1},
                                    "llm_rank": {"type": "integer", "minimum": 1},
                                    "label": {"type": "string", "enum": ["pass", "fail", "unknown"]},
                                    "patch_path": {"type": ["string", "null"]},
                                    "after_state_path": {"type": ["string", "null"]},
                                    "static_check": {
                                        "type": "string",
                                        "enum": ["pass", "fail", "not_run", "not_applicable"],
                                    },
                                    "test_check": {
                                        "type": "string",
                                        "enum": ["pass", "fail", "not_run", "not_applicable"],
                                    },
                                    "source": {"type": "object"},
                                    "provenance": {"type": "object"},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "provenance": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
            },
            "provenance": {"type": "object"},
        },
        "additionalProperties": False,
    }


def downstream_rerank_report_template() -> dict[str, Any]:
    """Return an empty report template with the required metrics and baselines."""

    return {
        "schema_version": DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION,
        "benchmark_id": "<benchmark-id>",
        "example_count": 0,
        "metrics": {
            baseline: {metric: None for metric in DOWNSTREAM_REQUIRED_METRICS}
            for baseline in DOWNSTREAM_REQUIRED_BASELINES
        },
        "slices": {
            "by_task_type": {},
            "by_candidate_source": {},
            "by_failure_type": {},
        },
        "claim_gate": build_downstream_rerank_claim_gate(
            example_count=0,
            metrics={},
        ),
        "falsification": {
            "hypothesis": "CodeLeWM reranking improves useful candidate selection over LLM order and no-action baselines.",
            "would_falsify": [
                "fewer than 100 labeled examples",
                "CodeLeWM pass_at_1 or MRR not strictly above LLM order",
                "CodeLeWM pass_at_1 or MRR not strictly above no-action",
                "improvement isolated to invalid or unchecked candidates",
            ],
        },
    }


def build_downstream_rerank_claim_gate(
    *,
    example_count: int,
    metrics: Mapping[str, Mapping[str, float]],
    min_labeled_examples: int = DOWNSTREAM_MIN_LABELED_EXAMPLES,
) -> dict[str, Any]:
    """Build the downstream usefulness claim gate from headline metrics."""

    failure_reasons: list[str] = []
    if example_count < min_labeled_examples:
        failure_reasons.append(
            f"example_count_below_minimum:{example_count}<{min_labeled_examples}"
        )
    codelewm = metrics.get("codelewm", {})
    for baseline_name in ("llm_order", "no_action"):
        baseline = metrics.get(baseline_name, {})
        for metric_name in ("pass_at_1", "mrr"):
            codelewm_value = codelewm.get(metric_name)
            baseline_value = baseline.get(metric_name)
            if codelewm_value is None or baseline_value is None:
                failure_reasons.append(f"missing_metric:{baseline_name}:{metric_name}")
                continue
            if codelewm_value <= baseline_value:
                failure_reasons.append(
                    f"not_strictly_above:{baseline_name}:{metric_name}:{codelewm_value}<={baseline_value}"
                )
    return {
        "schema_version": DOWNSTREAM_RERANK_CLAIM_GATE_SCHEMA_VERSION,
        "allowed": not failure_reasons,
        "min_labeled_examples": min_labeled_examples,
        "example_count": example_count,
        "checked_baselines": ["llm_order", "no_action"],
        "required_metrics": ["pass_at_1", "mrr"],
        "failure_reasons": failure_reasons,
    }


def validate_downstream_rerank_claim_gate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema_version") != DOWNSTREAM_RERANK_CLAIM_GATE_SCHEMA_VERSION:
        raise DownstreamBenchmarkError("unsupported downstream claim gate schema_version")
    if not isinstance(payload.get("allowed"), bool):
        raise DownstreamBenchmarkError("claim gate allowed must be boolean")
    failure_reasons = payload.get("failure_reasons")
    if not isinstance(failure_reasons, list):
        raise DownstreamBenchmarkError("claim gate failure_reasons must be a list")
    return payload


def _ensure_json_native(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DownstreamBenchmarkError(f"{field_name} must be JSON-native: {exc}") from exc
