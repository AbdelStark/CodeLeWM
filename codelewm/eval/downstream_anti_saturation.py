"""Hard anti-saturation downstream reranking schema, diagnostics, and gates.

This module implements the RFC-0016 ``anti_saturation_semantic_v1`` contract.
It is import-light and never executes candidate code: the only candidate-text
transformation it performs is a text-only unified-diff apply (lazy-imported) and
``ast.parse`` for parseability accounting. Model scoring lives in
``downstream_rerank`` (issue #422); this module owns the model-independent
diagnostics that decide whether a slice is *eligible* for a headline claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION = (
    "codelewm.downstream_anti_saturation_report.v1"
)
DOWNSTREAM_ANTI_SATURATION_CLAIM_GATE_SCHEMA_VERSION = (
    "codelewm.downstream_anti_saturation_claim_gate.v1"
)

ANTI_SATURATION_PROFILE = "anti_saturation_semantic_v1"

# Saturation ceilings (RFC-0016 §Anti-Saturation Filter). A slice is eligible
# only when the simple baselines stay strictly below these values.
NO_ACTION_SATURATION_CEILING = 0.85
LEXICAL_SATURATION_CEILING = 0.85
LLM_ORDER_SATURATION_CEILING = 0.90

ANTI_SATURATION_MIN_PROBLEMS = 100
MIN_CANDIDATES_PER_PROBLEM = 6
MAX_CANDIDATES_PER_PROBLEM = 12
MIN_DUAL_HARD_NEGATIVE_FRACTION = 0.70

PASSING_CANDIDATE_CLASS = "passing_reference"
HARD_NEGATIVE_CLASSES: tuple[str, ...] = (
    "passing_reference",
    "no_action_bait",
    "near_no_action_bait",
    "partial_fix",
    "wrong_symbol",
    "wrong_branch",
    "over_broad",
    "deterministic_mutant",
    "llm_generated",
    "parser_apply_failure",
)
# Classes that count as distinct *failing* hard-negative classes for the
# dual-coverage requirement (everything except the passing reference).
FAILING_HARD_NEGATIVE_CLASSES: tuple[str, ...] = tuple(
    name for name in HARD_NEGATIVE_CLASSES if name != PASSING_CANDIDATE_CLASS
)

# Model-independent baselines diagnosed before any CodeLeWM scoring.
ANTI_SATURATION_BASELINES: tuple[str, ...] = ("no_action", "lexical", "llm_order", "random")
# Baselines whose pass@1 gates eligibility (random is reported but never gates).
ANTI_SATURATION_GATING_BASELINES: tuple[str, ...] = ("no_action", "lexical", "llm_order")
ANTI_SATURATION_CLAIM_BASELINES: tuple[str, ...] = ("no_action", "lexical", "llm_order")
ANTI_SATURATION_CLAIM_METRICS: tuple[str, ...] = ("pass_at_1", "mrr")

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")


class DownstreamAntiSaturationError(ValueError):
    """Raised when anti-saturation payloads or hard-negative classes are invalid."""


def validate_hard_negative_class(value: str) -> str:
    """Validate a hard-negative candidate class against the RFC-0016 enumeration."""

    if value not in HARD_NEGATIVE_CLASSES:
        raise DownstreamAntiSaturationError(
            "unsupported hard_negative_class: "
            f"{value!r} (expected one of {', '.join(HARD_NEGATIVE_CLASSES)})"
        )
    return value


def lexical_similarity(left: str, right: str) -> float:
    """Jaccard overlap over identifier/number tokens.

    Identical definition to ``downstream_rerank._lexical_similarity`` so that the
    pack-time anti-saturation diagnostics match the eval-time baseline scores.
    """

    left_tokens = set(_TOKEN_RE.findall(left.lower()))
    right_tokens = set(_TOKEN_RE.findall(right.lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def stable_random_key(task_id: str, candidate_id: str) -> str:
    """Deterministic per-candidate key for the random baseline ordering."""

    return hashlib.sha256(f"{task_id}\0{candidate_id}".encode("utf-8")).hexdigest()


def build_anti_saturation_report(
    *,
    profile: str,
    problem_count: int,
    pool_sizes: Sequence[int],
    baseline_pass_at_1: Mapping[str, float],
    dual_hard_negative_fraction: float,
    hard_negative_class_coverage: Mapping[str, int],
    parser_apply_failure_rate: float,
    expected_random_pass_at_1: float | None = None,
    source_license_ok: bool = True,
    split_leakage_ok: bool = True,
    manifest_ok: bool = True,
    secret_scan_ok: bool = True,
    min_problem_count: int = ANTI_SATURATION_MIN_PROBLEMS,
) -> dict[str, Any]:
    """Build the ``codelewm.downstream_anti_saturation_report.v1`` payload.

    The report is pure data: it accepts pre-computed diagnostics and decides
    ``eligible`` from the RFC-0016 gates. Saturated, under-covered, or
    missing-baseline slices are preserved with ``eligible=False`` and a typed
    ``blocked_reasons`` entry; they are never dropped.
    """

    pool_sizes = list(int(size) for size in pool_sizes)
    blocked_reasons: list[str] = []

    if problem_count < min_problem_count:
        blocked_reasons.append(
            f"problem_count_below_minimum:{problem_count}<{min_problem_count}"
        )

    pool_min = min(pool_sizes) if pool_sizes else 0
    pool_max = max(pool_sizes) if pool_sizes else 0
    pool_size_ok = bool(pool_sizes) and all(
        MIN_CANDIDATES_PER_PROBLEM <= size <= MAX_CANDIDATES_PER_PROBLEM
        for size in pool_sizes
    )
    if not pool_size_ok:
        blocked_reasons.append(
            "candidate_pool_size_out_of_range:"
            f"{pool_min}-{pool_max} not in "
            f"{MIN_CANDIDATES_PER_PROBLEM}-{MAX_CANDIDATES_PER_PROBLEM}"
        )

    ceilings = {
        "no_action": NO_ACTION_SATURATION_CEILING,
        "lexical": LEXICAL_SATURATION_CEILING,
        "llm_order": LLM_ORDER_SATURATION_CEILING,
    }
    below_ceiling: dict[str, bool | None] = {}
    for baseline in ANTI_SATURATION_GATING_BASELINES:
        value = baseline_pass_at_1.get(baseline)
        if value is None:
            below_ceiling[baseline] = None
            blocked_reasons.append(f"missing_baseline:{baseline}")
            continue
        ceiling = ceilings[baseline]
        is_below = float(value) < ceiling
        below_ceiling[baseline] = is_below
        if not is_below:
            blocked_reasons.append(f"{baseline}_saturated:{float(value)}>={ceiling}")

    random_value = baseline_pass_at_1.get("random")
    if expected_random_pass_at_1 is None or random_value is None:
        random_within_pool_expectation: bool | None = None
    else:
        random_within_pool_expectation = (
            abs(float(random_value) - float(expected_random_pass_at_1)) <= 0.5
        )

    dual_fraction = float(dual_hard_negative_fraction)
    if dual_fraction < MIN_DUAL_HARD_NEGATIVE_FRACTION:
        blocked_reasons.append(
            "dual_hard_negative_coverage_below:"
            f"{dual_fraction}<{MIN_DUAL_HARD_NEGATIVE_FRACTION}"
        )

    if not source_license_ok:
        blocked_reasons.append("source_license_gate_open")
    if not split_leakage_ok:
        blocked_reasons.append("split_leakage_gate_open")
    if not manifest_ok:
        blocked_reasons.append("manifest_gate_open")
    if not secret_scan_ok:
        blocked_reasons.append("secret_scan_gate_open")

    return {
        "schema_version": DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION,
        "profile": profile,
        "eligible": not blocked_reasons,
        "problem_count": problem_count,
        "min_problem_count": min_problem_count,
        "candidate_pool_size": {
            "min": pool_min,
            "max": pool_max,
            "min_required": MIN_CANDIDATES_PER_PROBLEM,
            "max_allowed": MAX_CANDIDATES_PER_PROBLEM,
            "ok": pool_size_ok,
        },
        "baseline_pass_at_1": {
            baseline: (
                None
                if baseline_pass_at_1.get(baseline) is None
                else float(baseline_pass_at_1[baseline])
            )
            for baseline in ANTI_SATURATION_BASELINES
        },
        "saturation_ceilings": ceilings,
        "no_action_below_ceiling": below_ceiling["no_action"],
        "lexical_below_ceiling": below_ceiling["lexical"],
        "llm_order_below_ceiling": below_ceiling["llm_order"],
        "expected_random_pass_at_1": (
            None if expected_random_pass_at_1 is None else float(expected_random_pass_at_1)
        ),
        "random_within_pool_expectation": random_within_pool_expectation,
        "dual_hard_negative_fraction": dual_fraction,
        "min_dual_hard_negative_fraction": MIN_DUAL_HARD_NEGATIVE_FRACTION,
        "hard_negative_class_coverage": {
            str(name): int(count)
            for name, count in sorted(hard_negative_class_coverage.items())
        },
        "parser_apply_failure_rate": float(parser_apply_failure_rate),
        "gates": {
            "source_license_ok": bool(source_license_ok),
            "split_leakage_ok": bool(split_leakage_ok),
            "manifest_ok": bool(manifest_ok),
            "secret_scan_ok": bool(secret_scan_ok),
        },
        "blocked_reasons": blocked_reasons,
    }


def build_anti_saturation_claim_gate(
    *,
    example_count: int,
    metrics: Mapping[str, Mapping[str, float]],
    anti_saturation_eligible: bool,
    lift_confidence_intervals: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
    min_labeled_examples: int = ANTI_SATURATION_MIN_PROBLEMS,
) -> dict[str, Any]:
    """Build the hard-downstream claim gate from headline metrics.

    The gate opens only when, on an eligible slice with at least
    ``min_labeled_examples`` problems, CodeLeWM is strictly above no-action,
    lexical, AND llm-order on both pass@1 and MRR, and (when supplied) every
    lift confidence interval excludes zero.
    """

    failure_reasons: list[str] = []
    if example_count < min_labeled_examples:
        failure_reasons.append(
            f"example_count_below_minimum:{example_count}<{min_labeled_examples}"
        )
    if not anti_saturation_eligible:
        failure_reasons.append("anti_saturation_slice_not_eligible")

    codelewm = metrics.get("codelewm", {})
    for baseline_name in ANTI_SATURATION_CLAIM_BASELINES:
        baseline = metrics.get(baseline_name, {})
        for metric_name in ANTI_SATURATION_CLAIM_METRICS:
            codelewm_value = codelewm.get(metric_name)
            baseline_value = baseline.get(metric_name)
            if codelewm_value is None or baseline_value is None:
                failure_reasons.append(f"missing_metric:{baseline_name}:{metric_name}")
                continue
            if codelewm_value <= baseline_value:
                failure_reasons.append(
                    "not_strictly_above:"
                    f"{baseline_name}:{metric_name}:{codelewm_value}<={baseline_value}"
                )

    if lift_confidence_intervals is not None:
        for baseline_name in ANTI_SATURATION_CLAIM_BASELINES:
            baseline_intervals = lift_confidence_intervals.get(baseline_name, {})
            for metric_name in ANTI_SATURATION_CLAIM_METRICS:
                interval = baseline_intervals.get(metric_name)
                if not isinstance(interval, Mapping) or interval.get("low") is None:
                    failure_reasons.append(
                        f"missing_lift_ci:{baseline_name}:{metric_name}"
                    )
                    continue
                if float(interval["low"]) <= 0.0:
                    failure_reasons.append(
                        "lift_ci_includes_zero:"
                        f"{baseline_name}:{metric_name}:{float(interval['low'])}"
                    )

    return {
        "schema_version": DOWNSTREAM_ANTI_SATURATION_CLAIM_GATE_SCHEMA_VERSION,
        "allowed": not failure_reasons,
        "min_labeled_examples": min_labeled_examples,
        "example_count": example_count,
        "anti_saturation_eligible": bool(anti_saturation_eligible),
        "checked_baselines": list(ANTI_SATURATION_CLAIM_BASELINES),
        "required_metrics": list(ANTI_SATURATION_CLAIM_METRICS),
        "lift_confidence_intervals_checked": lift_confidence_intervals is not None,
        "failure_reasons": failure_reasons,
    }


def validate_anti_saturation_report(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema_version") != DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION:
        raise DownstreamAntiSaturationError(
            "unsupported anti-saturation report schema_version"
        )
    if not isinstance(payload.get("eligible"), bool):
        raise DownstreamAntiSaturationError("anti-saturation report eligible must be boolean")
    if not isinstance(payload.get("blocked_reasons"), list):
        raise DownstreamAntiSaturationError(
            "anti-saturation report blocked_reasons must be a list"
        )
    return payload


def validate_anti_saturation_claim_gate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema_version") != DOWNSTREAM_ANTI_SATURATION_CLAIM_GATE_SCHEMA_VERSION:
        raise DownstreamAntiSaturationError(
            "unsupported anti-saturation claim gate schema_version"
        )
    if not isinstance(payload.get("allowed"), bool):
        raise DownstreamAntiSaturationError("claim gate allowed must be boolean")
    if not isinstance(payload.get("failure_reasons"), list):
        raise DownstreamAntiSaturationError("claim gate failure_reasons must be a list")
    return payload


def anti_saturation_report_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for anti-saturation report payloads."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION,
        "type": "object",
        "required": [
            "schema_version",
            "profile",
            "eligible",
            "problem_count",
            "candidate_pool_size",
            "baseline_pass_at_1",
            "dual_hard_negative_fraction",
            "hard_negative_class_coverage",
            "parser_apply_failure_rate",
            "gates",
            "blocked_reasons",
        ],
        "properties": {
            "schema_version": {"const": DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION},
            "profile": {"type": "string", "minLength": 1},
            "eligible": {"type": "boolean"},
            "problem_count": {"type": "integer", "minimum": 0},
            "candidate_pool_size": {"type": "object"},
            "baseline_pass_at_1": {"type": "object"},
            "dual_hard_negative_fraction": {"type": "number"},
            "hard_negative_class_coverage": {"type": "object"},
            "parser_apply_failure_rate": {"type": "number"},
            "gates": {"type": "object"},
            "blocked_reasons": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    }


def compute_model_independent_baselines(
    benchmark: Any,
    *,
    root: Path | str,
) -> dict[str, Any]:
    """Compute pre-scoring diagnostics from a materialized benchmark pack.

    Returns the keyword arguments for :func:`build_anti_saturation_report` that
    depend on candidate text: model-independent baseline pass@1 (no_action,
    lexical, llm_order, random), the expected random pass@1 from the pool
    composition, the per-task pool sizes, the aggregate hard-negative-class
    coverage, the dual-failing-class fraction, and the parser/apply failure
    rate. No candidate code is executed: patch application is text-only and
    parseability uses ``ast.parse``.
    """

    from codelewm.security.non_execution import parse_python_source_text

    root_path = Path(root)

    baseline_pass_counts = {name: 0.0 for name in ANTI_SATURATION_BASELINES}
    expected_random_sum = 0.0
    pool_sizes: list[int] = []
    class_coverage: dict[str, int] = {}
    dual_coverage_tasks = 0
    parser_apply_failures = 0
    candidate_total = 0
    task_total = 0

    for task in benchmark.tasks:
        task_total += 1
        before_text = (root_path / task.before_path).read_text(encoding="utf-8")
        rows: list[dict[str, Any]] = []
        failing_classes: set[str] = set()
        pass_candidates = 0
        for candidate in task.candidates:
            candidate_total += 1
            rel = candidate.after_state_path or candidate.patch_path
            assert rel is not None
            candidate_text = (root_path / rel).read_text(encoding="utf-8")
            if candidate.after_state_path is not None:
                after_text: str | None = candidate_text
            else:
                # Lazy import: only patch candidates need the (heavy) scorer module
                # for its text-only unified-diff applier. Pure after-state packs
                # never import torch on the build path.
                from codelewm.harness.scorer import ScoreError, _apply_unified_diff

                try:
                    after_text = _apply_unified_diff(
                        before_text, candidate_text, artifact=candidate.patch_path or "<patch>"
                    )
                except ScoreError:
                    after_text = None
            apply_failed = candidate.patch_path is not None and after_text is None
            parseable = False
            if after_text is not None:
                try:
                    parse_python_source_text(after_text, filename=candidate.candidate_id)
                    parseable = True
                except Exception:  # noqa: BLE001 - parse failure is a diagnostic, not fatal
                    parseable = False
            if apply_failed or not parseable:
                parser_apply_failures += 1

            hard_negative_class = str(candidate.source.get("hard_negative_class", "")) or None
            if hard_negative_class:
                class_coverage[hard_negative_class] = (
                    class_coverage.get(hard_negative_class, 0) + 1
                )
            if candidate.label == "pass":
                pass_candidates += 1
            elif candidate.label == "fail" and hard_negative_class:
                failing_classes.add(hard_negative_class)

            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "label": candidate.label,
                    "llm_rank": candidate.llm_rank,
                    "text": after_text if after_text is not None else candidate_text,
                }
            )

        pool_sizes.append(len(rows))
        if len(failing_classes) >= 2:
            dual_coverage_tasks += 1
        if rows:
            expected_random_sum += pass_candidates / len(rows)

        orders = {
            "llm_order": sorted(rows, key=lambda r: (r["llm_rank"], r["candidate_id"])),
            "random": sorted(
                rows, key=lambda r: stable_random_key(task.task_id, r["candidate_id"])
            ),
            "lexical": sorted(
                rows,
                key=lambda r: (-lexical_similarity(task.prompt, r["text"]), r["candidate_id"]),
            ),
            "no_action": sorted(
                rows,
                key=lambda r: (-lexical_similarity(before_text, r["text"]), r["candidate_id"]),
            ),
        }
        for name, ordered in orders.items():
            if ordered and ordered[0]["label"] == "pass":
                baseline_pass_counts[name] += 1.0

    baseline_pass_at_1 = {
        name: (baseline_pass_counts[name] / task_total if task_total else 0.0)
        for name in ANTI_SATURATION_BASELINES
    }
    expected_random_pass_at_1 = (
        expected_random_sum / task_total if task_total else 0.0
    )
    dual_hard_negative_fraction = (
        dual_coverage_tasks / task_total if task_total else 0.0
    )
    parser_apply_failure_rate = (
        parser_apply_failures / candidate_total if candidate_total else 0.0
    )

    return {
        "problem_count": task_total,
        "pool_sizes": pool_sizes,
        "baseline_pass_at_1": baseline_pass_at_1,
        "expected_random_pass_at_1": expected_random_pass_at_1,
        "dual_hard_negative_fraction": dual_hard_negative_fraction,
        "hard_negative_class_coverage": class_coverage,
        "parser_apply_failure_rate": parser_apply_failure_rate,
    }


def _ensure_json_native(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DownstreamAntiSaturationError(f"{field_name} must be JSON-native: {exc}") from exc
