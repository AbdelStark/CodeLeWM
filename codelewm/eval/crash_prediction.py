"""Crash-prediction binary classification eval.

This is the scoped fallback gate for the v0.6 substrate pivot (#269,
RFC-0014). Given a held-out eval set of packed records, the classifier
predicts whether each ``(code, input)`` will raise an exception.

The eval is model-agnostic: the caller supplies per-record decision
scores from each method (linear probe on ``z_code``, on
``z_code + z_input``, on ``z_pred_output``, lexical n-gram baseline,
static-analysis baseline, random baseline). The evaluator computes
accuracy, AUC-ROC, AUC-PR, F1, and the per-exception-class slice.

The scoped claim is allowed when the best latent-based method beats
every non-latent baseline by ≥0.05 absolute AUC.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


CRASH_PREDICTION_REPORT_SCHEMA_VERSION = "codelewm.eval.crash_prediction_report.v1"

LATENT_METHODS: frozenset[str] = frozenset(
    {"linear_code", "linear_code_input", "linear_predicted_output"}
)
NON_LATENT_METHODS: frozenset[str] = frozenset(
    {"lexical", "static", "random"}
)


class CrashPredictionError(ValueError):
    """Raised when the inputs to the crash-prediction eval are invalid."""


@dataclass(frozen=True)
class CrashSample:
    """One labeled eval sample."""

    record_id: str
    will_raise: bool
    exception_class: str | None
    source_dataset: str
    scores: dict[str, float]  # method -> score (higher = "will raise" likelihood)


@dataclass(frozen=True)
class MethodMetrics:
    method: str
    accuracy: float
    auc_roc: float
    auc_pr: float
    f1: float
    positives: int
    negatives: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "f1": self.f1,
            "positives": self.positives,
            "negatives": self.negatives,
        }


@dataclass(frozen=True)
class CrashPredictionReport:
    schema_version: str
    sample_count: int
    methods: tuple[MethodMetrics, ...]
    best_latent_method: str | None
    best_latent_auc: float
    best_non_latent_method: str | None
    best_non_latent_auc: float
    latent_lift_auc: float
    per_exception_class_auc: dict[str, dict[str, float]]
    per_source_dataset_auc: dict[str, dict[str, float]]
    claim_allowed: bool
    claim_reason: str
    min_latent_lift_for_claim: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "methods": [m.as_dict() for m in self.methods],
            "best_latent_method": self.best_latent_method,
            "best_latent_auc": self.best_latent_auc,
            "best_non_latent_method": self.best_non_latent_method,
            "best_non_latent_auc": self.best_non_latent_auc,
            "latent_lift_auc": self.latent_lift_auc,
            "per_exception_class_auc": {
                k: dict(sorted(v.items()))
                for k, v in sorted(self.per_exception_class_auc.items())
            },
            "per_source_dataset_auc": {
                k: dict(sorted(v.items()))
                for k, v in sorted(self.per_source_dataset_auc.items())
            },
            "claim_allowed": self.claim_allowed,
            "claim_reason": self.claim_reason,
            "min_latent_lift_for_claim": self.min_latent_lift_for_claim,
        }


def evaluate_crash_prediction(
    samples: Sequence[CrashSample],
    *,
    min_latent_lift_for_claim: float = 0.05,
) -> CrashPredictionReport:
    """Compute per-method metrics and the scoped claim gate."""

    if not samples:
        raise CrashPredictionError("samples sequence is empty")
    method_names: set[str] = set()
    for s in samples:
        method_names.update(s.scores.keys())
    if not method_names:
        raise CrashPredictionError("no scoring methods present in samples")

    positives = sum(1 for s in samples if s.will_raise)
    negatives = len(samples) - positives
    if positives == 0 or negatives == 0:
        raise CrashPredictionError(
            f"need both positive and negative samples; got {positives} pos / {negatives} neg"
        )

    metrics: list[MethodMetrics] = []
    for method in sorted(method_names):
        accuracy, auc_roc, auc_pr, f1 = _per_method_metrics(samples, method)
        metrics.append(
            MethodMetrics(
                method=method,
                accuracy=accuracy,
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                f1=f1,
                positives=positives,
                negatives=negatives,
            )
        )

    best_latent_method: str | None = None
    best_latent_auc = -math.inf
    best_non_latent_method: str | None = None
    best_non_latent_auc = -math.inf
    for m in metrics:
        if m.method in LATENT_METHODS and m.auc_roc > best_latent_auc:
            best_latent_auc = m.auc_roc
            best_latent_method = m.method
        if m.method in NON_LATENT_METHODS and m.auc_roc > best_non_latent_auc:
            best_non_latent_auc = m.auc_roc
            best_non_latent_method = m.method

    if best_latent_auc == -math.inf:
        best_latent_auc = 0.0
    if best_non_latent_auc == -math.inf:
        best_non_latent_auc = 0.0

    lift = best_latent_auc - best_non_latent_auc

    per_exception = _per_slice_auc(samples, method_names, slice_key="exception_class")
    per_source = _per_slice_auc(samples, method_names, slice_key="source_dataset")

    claim_allowed = (
        best_latent_method is not None
        and best_non_latent_method is not None
        and lift >= min_latent_lift_for_claim
    )
    if claim_allowed:
        claim_reason = (
            f"latent_auc={best_latent_auc:.3f} beats best non-latent "
            f"{best_non_latent_method}={best_non_latent_auc:.3f} by {lift:.3f} (>= {min_latent_lift_for_claim})"
        )
    else:
        claim_reason = (
            f"lift={lift:.3f} below threshold {min_latent_lift_for_claim} "
            f"(best_latent={best_latent_method}@{best_latent_auc:.3f}, "
            f"best_non_latent={best_non_latent_method}@{best_non_latent_auc:.3f})"
        )

    return CrashPredictionReport(
        schema_version=CRASH_PREDICTION_REPORT_SCHEMA_VERSION,
        sample_count=len(samples),
        methods=tuple(metrics),
        best_latent_method=best_latent_method,
        best_latent_auc=best_latent_auc,
        best_non_latent_method=best_non_latent_method,
        best_non_latent_auc=best_non_latent_auc,
        latent_lift_auc=lift,
        per_exception_class_auc=per_exception,
        per_source_dataset_auc=per_source,
        claim_allowed=claim_allowed,
        claim_reason=claim_reason,
        min_latent_lift_for_claim=min_latent_lift_for_claim,
    )


def _per_method_metrics(
    samples: Sequence[CrashSample], method: str
) -> tuple[float, float, float, float]:
    labels = [s.will_raise for s in samples if method in s.scores]
    scores = [s.scores[method] for s in samples if method in s.scores]
    if not labels:
        return 0.0, 0.0, 0.0, 0.0

    # Accuracy at threshold 0.5
    predictions = [score >= 0.5 for score in scores]
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels)

    auc_roc = _auc_roc(labels, scores)
    auc_pr = _auc_pr(labels, scores)

    # F1 at threshold 0.5
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if (not p) and l)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return accuracy, auc_roc, auc_pr, f1


def _auc_roc(labels: list[bool], scores: list[float]) -> float:
    """Compute AUC-ROC using the Mann-Whitney U formula."""

    pos_scores = [s for s, l in zip(scores, labels) if l]
    neg_scores = [s for s, l in zip(scores, labels) if not l]
    if not pos_scores or not neg_scores:
        return 0.0
    n_compare = len(pos_scores) * len(neg_scores)
    wins = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / n_compare


def _auc_pr(labels: list[bool], scores: list[float]) -> float:
    """Compute AUC-PR via trapezoidal precision-recall integration."""

    if not labels or not any(labels):
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n_positives = sum(1 for _, l in pairs if l)
    tp = 0
    fp = 0
    prev_recall = 0.0
    auc = 0.0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / n_positives if n_positives else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        auc += (recall - prev_recall) * precision
        prev_recall = recall
    return auc


def _per_slice_auc(
    samples: Sequence[CrashSample],
    method_names: set[str],
    *,
    slice_key: str,
) -> dict[str, dict[str, float]]:
    by_slice: dict[str, list[CrashSample]] = {}
    for s in samples:
        if slice_key == "exception_class":
            key = s.exception_class or "none"
        elif slice_key == "source_dataset":
            key = s.source_dataset or "unknown"
        else:
            continue
        by_slice.setdefault(key, []).append(s)
    out: dict[str, dict[str, float]] = {}
    for slice_name, batch in by_slice.items():
        slice_metrics: dict[str, float] = {}
        labels = [s.will_raise for s in batch]
        if not any(labels) or all(labels):
            # Slice is degenerate; record but skip.
            slice_metrics["sample_count"] = float(len(batch))
            out[slice_name] = slice_metrics
            continue
        for method in sorted(method_names):
            method_scores = [
                s.scores[method] for s in batch if method in s.scores
            ]
            method_labels = [
                s.will_raise for s in batch if method in s.scores
            ]
            if not method_scores:
                continue
            slice_metrics[method] = _auc_roc(method_labels, method_scores)
        slice_metrics["sample_count"] = float(len(batch))
        out[slice_name] = slice_metrics
    return out
