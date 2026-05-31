"""Patch-surprise evaluation for CodeLeWM.

The surprise evaluation scores the true after-state of a transition against
decoys drawn from four categories — random, same-file, mutation, and
action-cluster — and reports the pairwise AUC and the true-after rank as
described in RFC-0007. Scoring uses the transition energy returned by a
caller-supplied callable so the evaluation is independent of the concrete
model runtime.

The module is intentionally self-contained: decoy construction is purely
deterministic (seeded) and operates on plain Python dataclasses, the
scoring oracle is a `Callable[[str, str], float]`, and the report is
JSON-native with `codelewm.eval.surprise_report.v1` as its schema version.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


SURPRISE_REPORT_SCHEMA_VERSION = "codelewm.eval.surprise_report.v1"
SURPRISE_DECOY_CATEGORIES: tuple[str, ...] = (
    "random",
    "same_file",
    "mutation",
    "action_cluster",
)
EXECUTION_SURPRISE_DECOY_CATEGORIES: tuple[str, ...] = (
    "same_problem_different_submission",
    "same_code_different_input",
)
ALLOWED_SURPRISE_DECOY_CATEGORIES: tuple[str, ...] = (
    *SURPRISE_DECOY_CATEGORIES,
    *EXECUTION_SURPRISE_DECOY_CATEGORIES,
)
SurpriseCategory = Literal["random", "same_file", "mutation", "action_cluster"]


class SurpriseEvalError(ValueError):
    """Raised when surprise inputs or outputs violate the public contract."""


@dataclass(frozen=True)
class SurpriseExampleInput:
    """A single transition used as input to the surprise scorer."""

    transition_id: str
    repo: str
    path: str
    action_cluster: str
    true_after: str
    same_file_after_states: tuple[str, ...] = ()
    action_cluster_after_states: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise SurpriseEvalError("transition_id must not be empty")
        if not self.true_after:
            raise SurpriseEvalError(
                f"transition {self.transition_id!r} must declare a non-empty true_after"
            )
        if not self.path:
            raise SurpriseEvalError(
                f"transition {self.transition_id!r} must declare a non-empty path"
            )


@dataclass(frozen=True)
class SurpriseDecoy:
    """One decoy after-state for a given transition."""

    transition_id: str
    decoy_id: str
    category: str
    after: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in ALLOWED_SURPRISE_DECOY_CATEGORIES:
            allowed = ", ".join(ALLOWED_SURPRISE_DECOY_CATEGORIES)
            raise SurpriseEvalError(
                f"decoy category must be one of: {allowed}; got {self.category!r}"
            )
        if not self.decoy_id:
            raise SurpriseEvalError("decoy_id must not be empty")
        if not self.after:
            raise SurpriseEvalError(
                f"decoy {self.decoy_id!r} must declare a non-empty after"
            )


@dataclass(frozen=True)
class SurpriseExampleResult:
    """Scored true and decoy after-states for one transition."""

    transition_id: str
    true_score: float
    decoy_scores_by_category: Mapping[str, tuple[float, ...]]
    true_rank: int
    candidate_count: int

    def __post_init__(self) -> None:
        if self.true_rank < 1:
            raise SurpriseEvalError("true_rank must be >= 1")
        if self.candidate_count < 1:
            raise SurpriseEvalError("candidate_count must be >= 1")
        if not math.isfinite(self.true_score):
            raise SurpriseEvalError("true_score must be finite")
        for category, scores in self.decoy_scores_by_category.items():
            if category not in ALLOWED_SURPRISE_DECOY_CATEGORIES:
                allowed = ", ".join(ALLOWED_SURPRISE_DECOY_CATEGORIES)
                raise SurpriseEvalError(
                    f"decoy category must be one of: {allowed}; got {category!r}"
                )
            for score in scores:
                if not math.isfinite(score):
                    raise SurpriseEvalError("decoy scores must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "true_score": self.true_score,
            "decoy_scores_by_category": {
                category: list(scores)
                for category, scores in sorted(self.decoy_scores_by_category.items())
            },
            "true_rank": self.true_rank,
            "candidate_count": self.candidate_count,
        }


@dataclass(frozen=True)
class SurpriseMetrics:
    """Aggregated surprise metrics across all evaluated transitions."""

    pairwise_auc_overall: float
    pairwise_auc_by_category: Mapping[str, float]
    mean_true_rank: float
    median_true_rank: float
    recall_at_1: float
    decoy_counts: Mapping[str, int]
    example_count: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.pairwise_auc_overall, "pairwise_auc_overall"),
            (self.mean_true_rank, "mean_true_rank"),
            (self.median_true_rank, "median_true_rank"),
            (self.recall_at_1, "recall_at_1"),
        ):
            if not math.isfinite(value):
                raise SurpriseEvalError(f"{name} must be finite")
        for category, auc in self.pairwise_auc_by_category.items():
            if category not in ALLOWED_SURPRISE_DECOY_CATEGORIES:
                allowed = ", ".join(ALLOWED_SURPRISE_DECOY_CATEGORIES)
                raise SurpriseEvalError(
                    f"pairwise_auc_by_category key must be one of: {allowed}; "
                    f"got {category!r}"
                )
            if not math.isfinite(auc):
                raise SurpriseEvalError(f"pairwise_auc[{category}] must be finite")
        for category, count in self.decoy_counts.items():
            if category not in ALLOWED_SURPRISE_DECOY_CATEGORIES:
                allowed = ", ".join(ALLOWED_SURPRISE_DECOY_CATEGORIES)
                raise SurpriseEvalError(
                    f"decoy_counts key must be one of: {allowed}; got {category!r}"
                )
            if count < 0:
                raise SurpriseEvalError(f"decoy_counts[{category}] must be non-negative")
        if self.example_count < 0:
            raise SurpriseEvalError("example_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairwise_auc_overall": self.pairwise_auc_overall,
            "pairwise_auc_by_category": dict(sorted(self.pairwise_auc_by_category.items())),
            "mean_true_rank": self.mean_true_rank,
            "median_true_rank": self.median_true_rank,
            "recall_at_1": self.recall_at_1,
            "decoy_counts": dict(sorted(self.decoy_counts.items())),
            "example_count": self.example_count,
        }


@dataclass(frozen=True)
class SurpriseReport:
    """Schema-versioned report of one surprise-evaluation run."""

    schema_version: str
    metrics: SurpriseMetrics
    examples: tuple[SurpriseExampleResult, ...]
    decoy_seed: int
    score_direction: Literal["lower_is_better", "higher_is_better"]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SURPRISE_REPORT_SCHEMA_VERSION:
            raise SurpriseEvalError(
                f"schema_version must be {SURPRISE_REPORT_SCHEMA_VERSION!r}"
            )
        if self.score_direction not in {"lower_is_better", "higher_is_better"}:
            raise SurpriseEvalError(
                "score_direction must be 'lower_is_better' or 'higher_is_better'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metrics": self.metrics.to_dict(),
            "examples": [example.to_dict() for example in self.examples],
            "decoy_seed": self.decoy_seed,
            "score_direction": self.score_direction,
            "metadata": dict(self.metadata),
        }


def build_decoys(
    example: SurpriseExampleInput,
    *,
    corpus: Sequence[SurpriseExampleInput],
    seed: int,
    random_count: int = 1,
    same_file_count: int = 1,
    mutation_count: int = 1,
    action_cluster_count: int = 1,
) -> tuple[SurpriseDecoy, ...]:
    """Construct decoys for one example from each of the four categories.

    The function never returns the true after-state as a decoy and never
    samples within the example's own ``same_file_after_states`` /
    ``action_cluster_after_states`` if those sets are empty. ``mutation``
    decoys are deterministic byte-level perturbations of the true after.
    """

    decoys: list[SurpriseDecoy] = []
    decoys.extend(
        _sample_random_decoys(
            example=example,
            corpus=corpus,
            count=random_count,
            seed=seed,
        )
    )
    decoys.extend(
        _sample_same_file_decoys(
            example=example,
            count=same_file_count,
            seed=seed,
        )
    )
    decoys.extend(
        _build_mutation_decoys(
            example=example,
            count=mutation_count,
            seed=seed,
        )
    )
    decoys.extend(
        _sample_action_cluster_decoys(
            example=example,
            corpus=corpus,
            count=action_cluster_count,
            seed=seed,
        )
    )
    return tuple(decoys)


def score_surprise_example(
    example: SurpriseExampleInput,
    *,
    decoys: Sequence[SurpriseDecoy],
    score_fn: Callable[[SurpriseExampleInput, str], float],
    score_direction: str = "lower_is_better",
) -> SurpriseExampleResult:
    """Score one example's true and decoy after-states with ``score_fn``.

    The scorer is `(example, candidate_after) -> float`. ``score_direction``
    declares whether smaller scores are better (`lower_is_better`, the
    default for transition-energy scorers) or larger scores are better
    (`higher_is_better`).
    """

    if score_direction not in {"lower_is_better", "higher_is_better"}:
        raise SurpriseEvalError(
            "score_direction must be 'lower_is_better' or 'higher_is_better'"
        )

    true_score = float(score_fn(example, example.true_after))
    decoy_scores_by_category: dict[str, list[float]] = {
        category: [] for category in SURPRISE_DECOY_CATEGORIES
    }
    for decoy in decoys:
        if decoy.transition_id != example.transition_id:
            raise SurpriseEvalError(
                f"decoy {decoy.decoy_id!r} does not belong to transition "
                f"{example.transition_id!r}"
            )
        decoy_scores_by_category[decoy.category].append(
            float(score_fn(example, decoy.after))
        )

    all_scores = [true_score]
    for scores in decoy_scores_by_category.values():
        all_scores.extend(scores)
    true_rank = _rank_of_true(all_scores, true_index=0, direction=score_direction)
    candidate_count = len(all_scores)

    return SurpriseExampleResult(
        transition_id=example.transition_id,
        true_score=true_score,
        decoy_scores_by_category={
            category: tuple(scores)
            for category, scores in decoy_scores_by_category.items()
        },
        true_rank=true_rank,
        candidate_count=candidate_count,
    )


def compute_surprise_metrics(
    results: Iterable[SurpriseExampleResult],
    *,
    score_direction: str = "lower_is_better",
) -> SurpriseMetrics:
    """Aggregate per-example results into the public metrics object."""

    results_tuple = tuple(results)
    if not results_tuple:
        raise SurpriseEvalError("results must not be empty")

    categories = tuple(
        dict.fromkeys(
            (
                *SURPRISE_DECOY_CATEGORIES,
                *(
                    category
                    for result in results_tuple
                    for category in result.decoy_scores_by_category
                ),
            )
        )
    )
    pairwise_wins: dict[str, list[int]] = {category: [] for category in categories}
    pairwise_total: dict[str, list[int]] = {category: [] for category in categories}
    decoy_counts: dict[str, int] = {category: 0 for category in categories}
    true_ranks: list[int] = []
    recall_at_1_hits = 0

    for result in results_tuple:
        true_ranks.append(result.true_rank)
        if result.true_rank == 1:
            recall_at_1_hits += 1
        for category, scores in result.decoy_scores_by_category.items():
            decoy_counts[category] += len(scores)
            wins = sum(
                1
                for score in scores
                if _true_beats_decoy(result.true_score, score, direction=score_direction)
            )
            ties = sum(1 for score in scores if score == result.true_score)
            pairwise_wins[category].append(wins + ties * 0.5)
            pairwise_total[category].append(len(scores))

    pairwise_auc_by_category: dict[str, float] = {}
    for category in categories:
        total = sum(pairwise_total[category])
        if total == 0:
            continue
        pairwise_auc_by_category[category] = float(sum(pairwise_wins[category])) / float(total)

    overall_wins = sum(sum(pairwise_wins[category]) for category in categories)
    overall_total = sum(sum(pairwise_total[category]) for category in categories)
    if overall_total == 0:
        raise SurpriseEvalError(
            "surprise metrics require at least one decoy across the corpus"
        )
    pairwise_auc_overall = float(overall_wins) / float(overall_total)

    return SurpriseMetrics(
        pairwise_auc_overall=pairwise_auc_overall,
        pairwise_auc_by_category=pairwise_auc_by_category,
        mean_true_rank=statistics.fmean(true_ranks),
        median_true_rank=statistics.median(true_ranks),
        recall_at_1=recall_at_1_hits / len(results_tuple),
        decoy_counts=decoy_counts,
        example_count=len(results_tuple),
    )


def build_surprise_report(
    results: Iterable[SurpriseExampleResult],
    *,
    decoy_seed: int,
    score_direction: str = "lower_is_better",
    metadata: Mapping[str, Any] | None = None,
) -> SurpriseReport:
    """Build a schema-versioned surprise report from per-example results."""

    results_tuple = tuple(results)
    metrics = compute_surprise_metrics(results_tuple, score_direction=score_direction)
    return SurpriseReport(
        schema_version=SURPRISE_REPORT_SCHEMA_VERSION,
        metrics=metrics,
        examples=results_tuple,
        decoy_seed=decoy_seed,
        score_direction=score_direction,  # type: ignore[arg-type]
        metadata=dict(metadata or {}),
    )


def surprise_report_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for a :class:`SurpriseReport` payload."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SURPRISE_REPORT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "metrics",
            "examples",
            "decoy_seed",
            "score_direction",
            "metadata",
        ],
        "properties": {
            "schema_version": {"const": SURPRISE_REPORT_SCHEMA_VERSION},
            "decoy_seed": {"type": "integer"},
            "score_direction": {
                "type": "string",
                "enum": ["lower_is_better", "higher_is_better"],
            },
            "metrics": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "pairwise_auc_overall",
                    "pairwise_auc_by_category",
                    "mean_true_rank",
                    "median_true_rank",
                    "recall_at_1",
                    "decoy_counts",
                    "example_count",
                ],
            },
            "examples": {"type": "array"},
            "metadata": {"type": "object"},
        },
    }


def validate_surprise_report_payload(payload: Mapping[str, Any]) -> SurpriseReport:
    """Validate and return a :class:`SurpriseReport` from a JSON payload."""

    required = {"schema_version", "metrics", "examples", "decoy_seed", "score_direction", "metadata"}
    missing = sorted(required - set(payload))
    if missing:
        raise SurpriseEvalError(
            f"surprise report missing required key(s): {', '.join(missing)}"
        )
    schema_version = payload["schema_version"]
    if schema_version != SURPRISE_REPORT_SCHEMA_VERSION:
        raise SurpriseEvalError(
            f"unsupported surprise schema_version: {schema_version!r}"
        )
    metrics_payload = payload["metrics"]
    if not isinstance(metrics_payload, Mapping):
        raise SurpriseEvalError("metrics must be a JSON object")
    metrics = SurpriseMetrics(
        pairwise_auc_overall=float(metrics_payload["pairwise_auc_overall"]),
        pairwise_auc_by_category={
            str(k): float(v) for k, v in metrics_payload["pairwise_auc_by_category"].items()
        },
        mean_true_rank=float(metrics_payload["mean_true_rank"]),
        median_true_rank=float(metrics_payload["median_true_rank"]),
        recall_at_1=float(metrics_payload["recall_at_1"]),
        decoy_counts={str(k): int(v) for k, v in metrics_payload["decoy_counts"].items()},
        example_count=int(metrics_payload["example_count"]),
    )
    examples_payload = payload["examples"]
    if not isinstance(examples_payload, Sequence) or isinstance(examples_payload, (str, bytes)):
        raise SurpriseEvalError("examples must be a JSON array")
    examples = tuple(_example_from_dict(item) for item in examples_payload)
    return SurpriseReport(
        schema_version=schema_version,
        metrics=metrics,
        examples=examples,
        decoy_seed=int(payload["decoy_seed"]),
        score_direction=str(payload["score_direction"]),  # type: ignore[arg-type]
        metadata=dict(payload.get("metadata", {})),
    )


def write_surprise_report(report: SurpriseReport, path: Path | str) -> SurpriseReport:
    """Write a validated surprise report to ``path``."""

    payload = report.to_dict()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validate_surprise_report_payload(payload)


def read_surprise_report(path: Path | str) -> SurpriseReport:
    """Read and validate a surprise report from ``path``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SurpriseEvalError("surprise report must be a JSON object")
    return validate_surprise_report_payload(payload)


def _sample_random_decoys(
    *,
    example: SurpriseExampleInput,
    corpus: Sequence[SurpriseExampleInput],
    count: int,
    seed: int,
) -> tuple[SurpriseDecoy, ...]:
    if count <= 0:
        return ()
    pool = [
        candidate
        for candidate in corpus
        if candidate.transition_id != example.transition_id
        and candidate.true_after != example.true_after
    ]
    if not pool:
        return ()
    rng = _category_rng(seed, example.transition_id, "random")
    pool_sorted = sorted(pool, key=lambda candidate: candidate.transition_id)
    selected = rng.sample(pool_sorted, k=min(count, len(pool_sorted)))
    return tuple(
        SurpriseDecoy(
            transition_id=example.transition_id,
            decoy_id=f"{example.transition_id}::random::{index}",
            category="random",
            after=candidate.true_after,
            metadata={"source_transition_id": candidate.transition_id},
        )
        for index, candidate in enumerate(selected)
    )


def _sample_same_file_decoys(
    *,
    example: SurpriseExampleInput,
    count: int,
    seed: int,
) -> tuple[SurpriseDecoy, ...]:
    if count <= 0 or not example.same_file_after_states:
        return ()
    rng = _category_rng(seed, example.transition_id, "same_file")
    candidates = sorted(
        {state for state in example.same_file_after_states if state != example.true_after}
    )
    if not candidates:
        return ()
    selected = rng.sample(candidates, k=min(count, len(candidates)))
    return tuple(
        SurpriseDecoy(
            transition_id=example.transition_id,
            decoy_id=f"{example.transition_id}::same_file::{index}",
            category="same_file",
            after=after,
            metadata={"path": example.path},
        )
        for index, after in enumerate(selected)
    )


def _sample_action_cluster_decoys(
    *,
    example: SurpriseExampleInput,
    corpus: Sequence[SurpriseExampleInput],
    count: int,
    seed: int,
) -> tuple[SurpriseDecoy, ...]:
    if count <= 0:
        return ()
    intra_candidates = sorted(
        {
            state
            for state in example.action_cluster_after_states
            if state != example.true_after
        }
    )
    if not intra_candidates:
        intra_candidates = sorted(
            {
                candidate.true_after
                for candidate in corpus
                if candidate.transition_id != example.transition_id
                and candidate.action_cluster == example.action_cluster
                and candidate.true_after != example.true_after
            }
        )
    if not intra_candidates:
        return ()
    rng = _category_rng(seed, example.transition_id, "action_cluster")
    selected = rng.sample(intra_candidates, k=min(count, len(intra_candidates)))
    return tuple(
        SurpriseDecoy(
            transition_id=example.transition_id,
            decoy_id=f"{example.transition_id}::action_cluster::{index}",
            category="action_cluster",
            after=after,
            metadata={"action_cluster": example.action_cluster},
        )
        for index, after in enumerate(selected)
    )


def _build_mutation_decoys(
    *,
    example: SurpriseExampleInput,
    count: int,
    seed: int,
) -> tuple[SurpriseDecoy, ...]:
    if count <= 0:
        return ()
    rng = _category_rng(seed, example.transition_id, "mutation")
    decoys: list[SurpriseDecoy] = []
    for index in range(count):
        mutated = _mutate_after(example.true_after, rng=rng)
        if not mutated or mutated == example.true_after:
            continue
        decoys.append(
            SurpriseDecoy(
                transition_id=example.transition_id,
                decoy_id=f"{example.transition_id}::mutation::{index}",
                category="mutation",
                after=mutated,
                metadata={"strategy": "token_substitution"},
            )
        )
    return tuple(decoys)


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_REPLACEMENT_TOKENS = (
    "value",
    "result",
    "tmp",
    "delta",
    "counter",
    "result_v2",
)


def _mutate_after(text: str, *, rng: random.Random) -> str:
    tokens = list(_IDENTIFIER_RE.finditer(text))
    if not tokens:
        return text + " # mutation"
    match = rng.choice(tokens)
    replacement = rng.choice(_REPLACEMENT_TOKENS)
    while replacement == match.group(0):
        replacement = rng.choice(_REPLACEMENT_TOKENS)
    return text[: match.start()] + replacement + text[match.end() :]


def _category_rng(seed: int, transition_id: str, category: str) -> random.Random:
    digest = hashlib.sha256(
        f"{seed}|{transition_id}|{category}".encode("utf-8")
    ).digest()
    derived = int.from_bytes(digest[:8], "big")
    return random.Random(derived)


def _rank_of_true(scores: Sequence[float], *, true_index: int, direction: str) -> int:
    true_score = scores[true_index]
    if direction == "lower_is_better":
        better = sum(1 for index, score in enumerate(scores) if index != true_index and score < true_score)
        ties = sum(1 for index, score in enumerate(scores) if index != true_index and score == true_score)
    else:
        better = sum(1 for index, score in enumerate(scores) if index != true_index and score > true_score)
        ties = sum(1 for index, score in enumerate(scores) if index != true_index and score == true_score)
    return 1 + better + ties // 2


def _true_beats_decoy(true_score: float, decoy_score: float, *, direction: str) -> bool:
    if direction == "lower_is_better":
        return true_score < decoy_score
    if direction == "higher_is_better":
        return true_score > decoy_score
    raise SurpriseEvalError("score_direction must be 'lower_is_better' or 'higher_is_better'")


def _example_from_dict(payload: Any) -> SurpriseExampleResult:
    if not isinstance(payload, Mapping):
        raise SurpriseEvalError("each example must be a JSON object")
    decoy_scores_payload = payload.get("decoy_scores_by_category", {})
    if not isinstance(decoy_scores_payload, Mapping):
        raise SurpriseEvalError("decoy_scores_by_category must be an object")
    decoy_scores_by_category = {
        str(category): tuple(float(score) for score in scores)
        for category, scores in decoy_scores_payload.items()
    }
    return SurpriseExampleResult(
        transition_id=str(payload["transition_id"]),
        true_score=float(payload["true_score"]),
        decoy_scores_by_category=decoy_scores_by_category,
        true_rank=int(payload["true_rank"]),
        candidate_count=int(payload["candidate_count"]),
    )


__all__ = [
    "ALLOWED_SURPRISE_DECOY_CATEGORIES",
    "EXECUTION_SURPRISE_DECOY_CATEGORIES",
    "SURPRISE_DECOY_CATEGORIES",
    "SURPRISE_REPORT_SCHEMA_VERSION",
    "SurpriseCategory",
    "SurpriseDecoy",
    "SurpriseEvalError",
    "SurpriseExampleInput",
    "SurpriseExampleResult",
    "SurpriseMetrics",
    "SurpriseReport",
    "build_decoys",
    "build_surprise_report",
    "compute_surprise_metrics",
    "read_surprise_report",
    "score_surprise_example",
    "surprise_report_json_schema",
    "validate_surprise_report_payload",
    "write_surprise_report",
]
