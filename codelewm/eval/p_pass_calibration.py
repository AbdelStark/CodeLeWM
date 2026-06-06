"""Held-out pass-probability calibration reports.

This module consumes completion-level score rows, such as
``codelewm.eval.completion_score.v1`` rows emitted by execution reranking, and
turns them into a manifest-backed correctness-calibration artifact. It does not
load checkpoints or execute candidate code.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security.secret_scan import scan_paths


P_PASS_CALIBRATION_REPORT_SCHEMA_VERSION = (
    "codelewm.eval.p_pass_calibration_report.v1"
)
P_PASS_CALIBRATION_RUN_SCHEMA_VERSION = "codelewm.eval.p_pass_calibration_run.v1"
P_PASS_DEFAULT_BASELINES: tuple[str, ...] = (
    "p_pass",
    "codelewm",
    "no_action",
    "shuffled_action",
    "lexical",
    "random",
    "llm_order",
)
P_PASS_DATASET_KINDS: tuple[str, ...] = (
    "training_pack_held_out",
    "downstream_completion",
)
PPassDatasetKind = Literal["training_pack_held_out", "downstream_completion"]


class PPassCalibrationError(ValueError):
    """Raised when pass-probability calibration cannot be computed or written."""


@dataclass(frozen=True)
class PPassScoreRow:
    """One labeled score row used by the p-pass calibration evaluator."""

    row_id: str
    passed: bool
    scores: Mapping[str, Any]
    benchmark_id: str
    split: str
    schema_version: str
    source_path: str
    source_line: int


@dataclass(frozen=True)
class PPassCalibrationResult:
    """CLI-facing summary for a written p-pass calibration report."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    parent_artifacts: tuple[str, ...]
    dataset_kind: str
    row_count: int
    claim_allowed: bool
    schema_version: str = P_PASS_CALIBRATION_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "parent_artifacts": list(self.parent_artifacts),
            "dataset_kind": self.dataset_kind,
            "row_count": self.row_count,
            "claim_allowed": self.claim_allowed,
        }


def build_p_pass_calibration_report(
    rows: Sequence[Mapping[str, Any] | PPassScoreRow],
    *,
    dataset_kind: str,
    baselines: Sequence[str] = P_PASS_DEFAULT_BASELINES,
    default_benchmark_id: str | None = None,
    score_paths: Sequence[str] = (),
    calibration_bin_count: int = 10,
) -> dict[str, Any]:
    """Build a JSON-native p-pass calibration report from labeled score rows."""

    if dataset_kind not in P_PASS_DATASET_KINDS:
        allowed = ", ".join(P_PASS_DATASET_KINDS)
        raise PPassCalibrationError(f"dataset_kind must be one of: {allowed}")
    if calibration_bin_count < 2:
        raise PPassCalibrationError("calibration_bin_count must be >= 2")
    normalized_baselines = _normalize_baselines(baselines)
    normalized_rows = tuple(
        row
        if isinstance(row, PPassScoreRow)
        else _row_from_mapping(
            row,
            source_path="<memory>",
            source_line=index + 1,
            default_benchmark_id=default_benchmark_id,
        )
        for index, row in enumerate(rows)
    )
    if not normalized_rows:
        raise PPassCalibrationError("at least one p-pass score row is required")

    label_counts = _label_counts(normalized_rows)
    benchmark_counts = Counter(row.benchmark_id for row in normalized_rows)
    split_counts = Counter(row.split for row in normalized_rows)
    schema_versions = Counter(row.schema_version for row in normalized_rows)
    overall = _slice_payload(
        normalized_rows,
        baselines=normalized_baselines,
        calibration_bin_count=calibration_bin_count,
    )
    benchmark_slices = {
        benchmark_id: _slice_payload(
            tuple(row for row in normalized_rows if row.benchmark_id == benchmark_id),
            baselines=normalized_baselines,
            calibration_bin_count=calibration_bin_count,
        )
        for benchmark_id in sorted(benchmark_counts)
    }
    split_slices = {
        split: _slice_payload(
            tuple(row for row in normalized_rows if row.split == split),
            baselines=normalized_baselines,
            calibration_bin_count=calibration_bin_count,
        )
        for split in sorted(split_counts)
    }
    primary_baseline = _select_primary_baseline(overall["baselines"])
    claim_gate = _diagnostic_claim_gate(
        dataset_kind=dataset_kind,
        primary_baseline=primary_baseline,
    )
    return {
        "schema_version": P_PASS_CALIBRATION_REPORT_SCHEMA_VERSION,
        "dataset_kind": dataset_kind,
        "score_direction": "higher_score_means_more_likely_pass",
        "row_count": len(normalized_rows),
        "label_counts": label_counts,
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "score_row_schema_versions": dict(sorted(schema_versions.items())),
        "score_paths": list(score_paths),
        "baselines_requested": list(normalized_baselines),
        "calibration_bin_count": calibration_bin_count,
        "overall": overall,
        "slices": {
            "benchmark": benchmark_slices,
            "split": split_slices,
        },
        "primary_baseline": primary_baseline,
        "claim_gate": claim_gate,
        "claim_allowed": bool(claim_gate["allowed"]),
        "claim_reason": str(claim_gate["reason"]),
        "interpretation": {
            "threshold_free": (
                "ROC-AUC and average_precision evaluate ranking of passed "
                "completions above failed completions."
            ),
            "calibration": (
                "Brier score and expected_calibration_error are computed after "
                "mapping scores to probabilities with identity for [0, 1] "
                "scores and sigmoid otherwise."
            ),
            "claim_boundary": (
                "This report is diagnostic evidence. Positive model-quality "
                "claims require the full v0.9 gate suite, parent artifact "
                "verification, per-benchmark coverage, and declared controls."
            ),
        },
    }


def load_p_pass_score_rows(
    score_paths: Sequence[Path | str],
    *,
    default_benchmark_id: str | None = None,
) -> tuple[PPassScoreRow, ...]:
    """Load p-pass score rows from JSONL files."""

    if not score_paths:
        raise PPassCalibrationError("at least one --scores path is required")
    loaded: list[PPassScoreRow] = []
    for path_value in score_paths:
        path = Path(path_value)
        if not path.is_file():
            raise PPassCalibrationError(f"scores file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise PPassCalibrationError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                loaded.append(
                    _row_from_mapping(
                        payload,
                        source_path=str(path),
                        source_line=line_number,
                        default_benchmark_id=default_benchmark_id,
                    )
                )
    if not loaded:
        raise PPassCalibrationError("score files did not contain any rows")
    return tuple(loaded)


def run_p_pass_calibration_evaluation(
    *,
    scores: Sequence[Path | str],
    out: Path | str,
    dataset_kind: str,
    parent_manifests: Sequence[Path | str],
    baselines: Sequence[str] = P_PASS_DEFAULT_BASELINES,
    benchmark: str | None = None,
    calibration_bin_count: int = 10,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "p-pass-calibration"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> PPassCalibrationResult:
    """Write a manifest-backed p-pass calibration report artifact."""

    output_dir = Path(out).resolve()
    report_path = output_dir / "reports" / "p_pass_calibration_report.json"
    config_path = output_dir / "config.json"
    secret_scan_path = output_dir / "reports" / "secret_scan_report.json"
    manifest_path = output_dir / "manifest.json"
    _reject_existing(
        (report_path, config_path, secret_scan_path, manifest_path),
        overwrite=overwrite,
        output_dir=output_dir,
    )

    parent_artifact_ids = _verified_parent_artifact_ids(parent_manifests)
    score_paths = tuple(Path(path) for path in scores)
    rows = load_p_pass_score_rows(
        score_paths,
        default_benchmark_id=benchmark,
    )
    report = build_p_pass_calibration_report(
        rows,
        dataset_kind=dataset_kind,
        baselines=baselines,
        default_benchmark_id=benchmark,
        score_paths=tuple(str(path) for path in score_paths),
        calibration_bin_count=calibration_bin_count,
    )
    config_payload = {
        "schema_version": P_PASS_CALIBRATION_RUN_SCHEMA_VERSION,
        "scores": [str(path) for path in score_paths],
        "dataset_kind": dataset_kind,
        "parent_manifests": [str(path) for path in parent_manifests],
        "baselines": list(_normalize_baselines(baselines)),
        "benchmark": benchmark,
        "calibration_bin_count": calibration_bin_count,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(config_payload, config_path)
    _write_json(report, report_path)
    scan = scan_paths(
        (report_path, config_path),
        include_suffixes=(),
        recursive=False,
    )
    scan_payload = scan.to_dict()
    _write_json(scan_payload, secret_scan_path)
    if scan_payload.get("findings"):
        raise PPassCalibrationError(
            "p-pass calibration artifact contains secret-scan findings"
        )

    artifact_manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=output_dir,
        files=(report_path, config_path, secret_scan_path),
        command=command,
        config=config_payload,
        parent_artifacts=parent_artifact_ids,
        source_git_sha=source_git_sha,
        created_at=created_at,
        metadata={
            "schema_version": P_PASS_CALIBRATION_RUN_SCHEMA_VERSION,
            "report_schema_version": P_PASS_CALIBRATION_REPORT_SCHEMA_VERSION,
            "dataset_kind": dataset_kind,
            "row_count": report["row_count"],
            "benchmark_counts": report["benchmark_counts"],
            "split_counts": report["split_counts"],
            "claim_allowed": report["claim_allowed"],
        },
    )
    write_artifact_manifest(artifact_manifest, manifest_path)
    return PPassCalibrationResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path=_relative_to_root(manifest_path, output_dir),
        report_path=_relative_to_root(report_path, output_dir),
        parent_artifacts=parent_artifact_ids,
        dataset_kind=dataset_kind,
        row_count=int(report["row_count"]),
        claim_allowed=bool(report["claim_allowed"]),
    )


def _slice_payload(
    rows: Sequence[PPassScoreRow],
    *,
    baselines: Sequence[str],
    calibration_bin_count: int,
) -> dict[str, Any]:
    labels = [row.passed for row in rows]
    return {
        "row_count": len(rows),
        "label_counts": _label_counts(rows),
        "baselines": {
            baseline: _baseline_metrics(
                rows,
                baseline=baseline,
                calibration_bin_count=calibration_bin_count,
            )
            for baseline in baselines
        },
        "positive_rate": _safe_div(sum(1 for label in labels if label), len(labels)),
    }


def _baseline_metrics(
    rows: Sequence[PPassScoreRow],
    *,
    baseline: str,
    calibration_bin_count: int,
) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    missing_count = 0
    invalid_count = 0
    nonfinite_count = 0
    for row in rows:
        if baseline not in row.scores:
            missing_count += 1
            continue
        value = row.scores[baseline]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid_count += 1
            continue
        score = float(value)
        if not math.isfinite(score):
            nonfinite_count += 1
            continue
        labels.append(1 if row.passed else 0)
        scores.append(score)

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    payload: dict[str, Any] = {
        "baseline": baseline,
        "status": "ok",
        "score_direction": "higher_score_means_more_likely_pass",
        "row_count": len(rows),
        "usable_row_count": len(scores),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "missing_score_count": missing_count,
        "invalid_score_count": invalid_count,
        "nonfinite_score_count": nonfinite_count,
        "probability_transform": "identity_0_1_else_sigmoid",
        "roc_auc": None,
        "average_precision": None,
        "brier_score": None,
        "expected_calibration_error": None,
        "threshold": 0.5,
        "thresholded": None,
        "calibration_bins": [],
    }
    if len(scores) == 0:
        payload["status"] = "missing" if missing_count == len(rows) else "no_finite_scores"
        return payload
    probabilities = [_score_to_probability(score) for score in scores]
    payload["brier_score"] = _round_float(_brier_score(probabilities, labels))
    ece, bins = _calibration_bins(
        probabilities,
        labels,
        bin_count=calibration_bin_count,
    )
    payload["expected_calibration_error"] = _round_float(ece)
    payload["calibration_bins"] = bins
    payload["thresholded"] = _thresholded_metrics(probabilities, labels, threshold=0.5)
    if positive_count == 0 or negative_count == 0:
        payload["status"] = "single_class"
        return payload
    payload["roc_auc"] = _round_float(_roc_auc(scores, labels))
    payload["average_precision"] = _round_float(_average_precision(scores, labels))
    return payload


def _roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    n = len(scores)
    ranks = [0.0] * n
    ordered = sorted(range(n), key=lambda index: scores[index])
    cursor = 0
    while cursor < n:
        end = cursor + 1
        while end < n and scores[ordered[end]] == scores[ordered[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for ordered_index in ordered[cursor:end]:
            ranks[ordered_index] = average_rank
        cursor = end
    positive_count = sum(labels)
    negative_count = n - positive_count
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    positive_count = sum(labels)
    if positive_count == 0:
        return 0.0
    true_positive_count = 0
    precision_sum = 0.0
    for rank, index in enumerate(
        sorted(range(len(scores)), key=lambda item: scores[item], reverse=True),
        start=1,
    ):
        if labels[index]:
            true_positive_count += 1
            precision_sum += true_positive_count / rank
    return precision_sum / positive_count


def _thresholded_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    threshold: float,
) -> dict[str, Any]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "accuracy": _round_float(_safe_div(tp + tn, len(labels))),
        "precision": _round_float(precision),
        "recall": _round_float(recall),
        "f1": _round_float(f1),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "predicted_positive_count": sum(predictions),
    }


def _calibration_bins(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bin_count: int,
) -> tuple[float, list[dict[str, Any]]]:
    bins: list[dict[str, Any]] = []
    weighted_error = 0.0
    total = len(labels)
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = [
            (probability, label)
            for probability, label in zip(probabilities, labels)
            if min(int(probability * bin_count), bin_count - 1) == index
        ]
        count = len(selected)
        if count:
            mean_probability = sum(probability for probability, _ in selected) / count
            empirical_rate = sum(label for _, label in selected) / count
            error = abs(mean_probability - empirical_rate)
        else:
            mean_probability = None
            empirical_rate = None
            error = None
        if error is not None:
            weighted_error += (count / total) * error
        bins.append(
            {
                "lower": _round_float(lower),
                "upper": _round_float(upper),
                "count": count,
                "mean_predicted_probability": _round_float(mean_probability)
                if mean_probability is not None
                else None,
                "empirical_positive_rate": _round_float(empirical_rate)
                if empirical_rate is not None
                else None,
                "absolute_calibration_error": _round_float(error)
                if error is not None
                else None,
            }
        )
    return weighted_error, bins


def _brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)


def _score_to_probability(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    if score >= 0:
        denominator = 1.0 + math.exp(-score)
        return 1.0 / denominator
    numerator = math.exp(score)
    return numerator / (1.0 + numerator)


def _row_from_mapping(
    payload: Mapping[str, Any],
    *,
    source_path: str,
    source_line: int,
    default_benchmark_id: str | None,
) -> PPassScoreRow:
    passed = payload.get("passed")
    if not isinstance(passed, bool):
        raise PPassCalibrationError(
            f"{source_path}:{source_line} field 'passed' must be a boolean"
        )
    scores_payload = payload.get("scores", {})
    if scores_payload is None:
        scores_payload = {}
    if not isinstance(scores_payload, Mapping):
        raise PPassCalibrationError(
            f"{source_path}:{source_line} field 'scores' must be a JSON object"
        )
    scores = dict(scores_payload)
    for baseline in P_PASS_DEFAULT_BASELINES:
        if baseline in payload and baseline not in scores:
            scores[baseline] = payload[baseline]
    row_id = _first_string(
        payload,
        "completion_id",
        "row_id",
        "record_id",
        "example_id",
        fallback=f"{source_path}:{source_line}",
    )
    benchmark_id = _first_string(
        payload,
        "benchmark_id",
        "benchmark",
        "source_dataset",
        fallback=default_benchmark_id or "unknown",
    )
    split = _first_string(payload, "split", fallback="unspecified")
    schema_version = _first_string(payload, "schema_version", fallback="unknown")
    return PPassScoreRow(
        row_id=row_id,
        passed=passed,
        scores=scores,
        benchmark_id=_normalize_id(benchmark_id),
        split=_normalize_id(split),
        schema_version=schema_version,
        source_path=source_path,
        source_line=source_line,
    )


def _first_string(
    payload: Mapping[str, Any],
    *keys: str,
    fallback: str,
) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _label_counts(rows: Sequence[PPassScoreRow]) -> dict[str, int]:
    positives = sum(1 for row in rows if row.passed)
    return {
        "passed": positives,
        "failed": len(rows) - positives,
    }


def _normalize_baselines(baselines: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(baseline) for baseline in baselines if str(baseline)))
    if not normalized:
        raise PPassCalibrationError("at least one baseline must be selected")
    return normalized


def _normalize_id(value: str) -> str:
    return value.strip().replace("-", "_").lower() or "unknown"


def _select_primary_baseline(baseline_payloads: Mapping[str, Mapping[str, Any]]) -> str | None:
    for candidate in ("p_pass", "codelewm"):
        payload = baseline_payloads.get(candidate)
        if payload is not None and payload.get("status") == "ok":
            return candidate
    for name, payload in baseline_payloads.items():
        if payload.get("status") == "ok":
            return name
    return None


def _diagnostic_claim_gate(
    *,
    dataset_kind: str,
    primary_baseline: str | None,
) -> dict[str, Any]:
    reasons = [
        "p_pass_calibration_is_diagnostic_until_full_v0_9_gate_suite_passes",
    ]
    if primary_baseline is None:
        reasons.append("no_evaluable_primary_p_pass_or_codelewm_score")
    return {
        "allowed": False,
        "reason": reasons[0],
        "reasons": reasons,
        "dataset_kind": dataset_kind,
        "primary_baseline": primary_baseline,
        "required_followups": [
            "verify parent manifests and score-row lineage",
            "compare p_pass/codelewm against declared controls on held-out rows",
            "check per-benchmark slices for missing or single-class coverage",
            "rerun the full v0.9 gate suite before publishing positive claims",
        ],
    }


def _verified_parent_artifact_ids(
    parent_manifests: Sequence[Path | str],
) -> tuple[str, ...]:
    if not parent_manifests:
        raise PPassCalibrationError("at least one --parent-manifest is required")
    artifact_ids: list[str] = []
    seen: set[str] = set()
    for path_value in parent_manifests:
        path = Path(path_value)
        artifact = read_artifact_manifest(path)
        validate_artifact_checksums(artifact, root=path.parent)
        if artifact.artifact_id not in seen:
            artifact_ids.append(artifact.artifact_id)
            seen.add(artifact.artifact_id)
    return tuple(artifact_ids)


def _reject_existing(
    paths: Sequence[Path],
    *,
    overwrite: bool,
    output_dir: Path,
) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        rel = ", ".join(_relative_to_root(path, output_dir) for path in existing)
        raise PPassCalibrationError(
            f"refusing to overwrite existing p-pass calibration artifact file(s): {rel}; "
            "pass --overwrite to replace them"
        )


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "P_PASS_CALIBRATION_REPORT_SCHEMA_VERSION",
    "P_PASS_CALIBRATION_RUN_SCHEMA_VERSION",
    "P_PASS_DATASET_KINDS",
    "P_PASS_DEFAULT_BASELINES",
    "PPassCalibrationError",
    "PPassCalibrationResult",
    "PPassScoreRow",
    "build_p_pass_calibration_report",
    "load_p_pass_score_rows",
    "run_p_pass_calibration_evaluation",
]
