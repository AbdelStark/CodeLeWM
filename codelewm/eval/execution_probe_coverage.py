"""Execution-probe label coverage and representation gate summaries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .execution_probe_targets import extract_labels


EXECUTION_PROBE_LABEL_COVERAGE_SCHEMA_VERSION = (
    "codelewm.eval.execution_probe_label_coverage.v1"
)
EXECUTION_PROBE_LABEL_BLOCKER_SCHEMA_VERSION = (
    "codelewm.eval.execution_probe_label_blocker.v1"
)
EXECUTION_PROBE_REPRESENTATION_GATE_TABLE_SCHEMA_VERSION = (
    "codelewm.eval.execution_probe_representation_gate_table.v1"
)


def build_execution_probe_label_coverage(
    records: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str],
    min_train_classes: int = 2,
    min_eval_rows: int = 1,
) -> dict[str, Any]:
    """Report whether requested execution probe targets are split-evaluable."""

    selected_targets = tuple(dict.fromkeys(str(target) for target in targets if str(target)))
    if not selected_targets:
        raise ValueError("at least one execution probe target is required")
    split_counts = Counter(_split(record) for record in records)
    target_reports: dict[str, Any] = {}
    blockers: list[dict[str, Any]] = []
    for target in selected_targets:
        labels = extract_labels(list(records), target=target).labels
        split_label_counts: dict[str, dict[str, int]] = {}
        split_applicable_counts: dict[str, int] = {}
        for split in ("train", "val", "test"):
            split_values = [
                labels[index]
                for index, record in enumerate(records)
                if _split(record) == split and labels[index] is not None
            ]
            split_label_counts[split] = dict(
                sorted(Counter(str(value) for value in split_values).items())
            )
            split_applicable_counts[split] = len(split_values)

        target_blockers: list[dict[str, Any]] = []
        train_class_count = len(split_label_counts["train"])
        if train_class_count < min_train_classes:
            target_blockers.append(
                _blocker(
                    "probe_label_train_class_blocker",
                    target=target,
                    split="train",
                    observed=train_class_count,
                    required=min_train_classes,
                    detail="train split must contain at least two label classes",
                )
            )
        for split in ("val", "test"):
            observed = split_applicable_counts[split]
            if observed < min_eval_rows:
                target_blockers.append(
                    _blocker(
                        "probe_label_eval_split_blocker",
                        target=target,
                        split=split,
                        observed=observed,
                        required=min_eval_rows,
                        detail=f"{split} split must contain target labels",
                    )
                )
        blockers.extend(target_blockers)
        target_reports[target] = {
            "target": target,
            "status": "available" if not target_blockers else "blocked",
            "available": not target_blockers,
            "split_label_counts": split_label_counts,
            "split_applicable_counts": split_applicable_counts,
            "train_class_count": train_class_count,
            "min_train_classes": min_train_classes,
            "min_eval_rows": min_eval_rows,
            "blockers": target_blockers,
        }

    return {
        "schema_version": EXECUTION_PROBE_LABEL_COVERAGE_SCHEMA_VERSION,
        "targets": list(selected_targets),
        "row_count": len(records),
        "split_counts": {
            split: int(split_counts.get(split, 0)) for split in ("train", "val", "test")
        },
        "min_train_classes": min_train_classes,
        "min_eval_rows": min_eval_rows,
        "coverage_ready": not blockers,
        "blockers": blockers,
        "target_coverage": target_reports,
        "interpretation": (
            "Probe availability is a data/split property. A blocked target is "
            "not model-quality evidence and must be fixed in the pack or split "
            "before latent probe scores are interpreted."
        ),
    }


def build_execution_probe_representation_gate_table(
    reports: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] | None = None,
    required_targets: Sequence[str] = ("passed", "output_magnitude_bucket"),
) -> dict[str, Any]:
    """Summarize pass/fail and magnitude probe gates across multiple seeds."""

    seed_values = tuple(seeds or range(len(reports)))
    if len(seed_values) != len(reports):
        raise ValueError("seeds length must match reports length")
    rows: list[dict[str, Any]] = []
    for seed, report in zip(seed_values, reports):
        target_reports = _mapping(report.get("target_reports", {}))
        claim_boundary = _mapping(report.get("claim_boundary", {}))
        for target in required_targets:
            target_report = _mapping(target_reports.get(target, {}))
            best_probe = _best_accuracy(_mapping(target_report.get("views", {})))
            best_control = _best_accuracy(_mapping(target_report.get("baselines", {})))
            available = bool(target_report.get("available"))
            rows.append(
                {
                    "seed": int(seed),
                    "target": target,
                    "available": available,
                    "status": "available" if available else "not_evaluable",
                    "best_probe_test_accuracy": best_probe,
                    "best_control_test_accuracy": best_control,
                    "probe_lift_over_best_control": None
                    if best_probe is None or best_control is None
                    else round(best_probe - best_control, 12),
                    "unavailable_reason": target_report.get("unavailable_reason"),
                    "claim_boundary_status": claim_boundary.get(
                        "semantic_structure_status"
                    ),
                    "positive_representation_claim_allowed": bool(
                        claim_boundary.get("positive_representation_claim_allowed")
                    ),
                }
            )
    blockers = [
        {
            "type": "representation_gate_target_unavailable",
            "seed": row["seed"],
            "target": row["target"],
            "reason": row["unavailable_reason"] or "target unavailable",
        }
        for row in rows
        if not row["available"]
    ]
    return {
        "schema_version": EXECUTION_PROBE_REPRESENTATION_GATE_TABLE_SCHEMA_VERSION,
        "required_targets": list(required_targets),
        "seed_count": len(reports),
        "rows": rows,
        "claim_allowed": False,
        "claim_reason": (
            "blocked:" + ";".join(item["reason"] for item in blockers)
            if blockers
            else "representation_gate_table_is_diagnostic_until_full_v0_9_suite_passes"
        ),
        "blockers": blockers,
        "interpretation": (
            "This table separates target availability, probe accuracy, control "
            "accuracy, and claim-boundary status across seeds."
        ),
    }


def _blocker(
    blocker_type: str,
    *,
    target: str,
    split: str,
    observed: int,
    required: int,
    detail: str,
) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_PROBE_LABEL_BLOCKER_SCHEMA_VERSION,
        "type": blocker_type,
        "target": target,
        "split": split,
        "observed": int(observed),
        "required": int(required),
        "reason": f"{blocker_type}:{target}:{split}:{observed}<{required}",
        "detail": detail,
    }


def _split(record: Mapping[str, Any]) -> str:
    value = record.get("split")
    return str(value) if value in {"train", "val", "test"} else "train"


def _best_accuracy(payload: Mapping[str, Any]) -> float | None:
    best: float | None = None
    for report in payload.values():
        report_mapping = _mapping(report)
        splits = _mapping(report_mapping.get("splits", {}))
        test = _mapping(splits.get("test", {}))
        value = test.get("accuracy")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            best = score if best is None else max(best, score)
    return None if best is None else round(best, 12)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "EXECUTION_PROBE_LABEL_BLOCKER_SCHEMA_VERSION",
    "EXECUTION_PROBE_LABEL_COVERAGE_SCHEMA_VERSION",
    "EXECUTION_PROBE_REPRESENTATION_GATE_TABLE_SCHEMA_VERSION",
    "build_execution_probe_label_coverage",
    "build_execution_probe_representation_gate_table",
]
