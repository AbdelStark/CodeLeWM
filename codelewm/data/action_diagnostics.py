"""Action-discriminative shard diagnostics for CodeLeWM datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .pack import PackedTransition, TokenSequence
from .split_dedup import hamming_distance_hex


ACTION_DISCRIMINATIVE_SHARD_REPORT_SCHEMA_VERSION = (
    "codelewm.data.action_discriminative_shard_report.v1"
)
DEFAULT_ACTION_DISCRIMINATIVE_THRESHOLDS: dict[str, float | int] = {
    "min_heldout_rows": 2,
    "min_same_file_or_near_before_pairs": 1,
    "min_action_cluster_pairs": 1,
    "min_edit_size_bucket_pairs": 1,
    "min_action_text_nonempty_ratio": 0.95,
    "min_action_abs_nonempty_ratio": 0.95,
}
DEFAULT_NEAR_BEFORE_HAMMING_THRESHOLD = 3
DEFAULT_MAX_NEAR_BEFORE_GROUP_PAIRS = 100_000
_TOP_COUNT_LIMIT = 20


class ActionDiscriminativeDiagnosticsError(ValueError):
    """Raised when an action-discriminative diagnostic report is malformed."""


def build_action_discriminative_shard_report(
    transitions: Iterable[PackedTransition],
    *,
    thresholds: Mapping[str, float | int] | None = None,
    near_before_hamming_threshold: int = DEFAULT_NEAR_BEFORE_HAMMING_THRESHOLD,
    max_near_before_group_pairs: int = DEFAULT_MAX_NEAR_BEFORE_GROUP_PAIRS,
) -> dict[str, Any]:
    """Build a schema-versioned report for action-discriminative shard coverage."""

    rows = tuple(transitions)
    if near_before_hamming_threshold < 0:
        raise ActionDiscriminativeDiagnosticsError(
            "near_before_hamming_threshold must be non-negative"
        )
    if max_near_before_group_pairs <= 0:
        raise ActionDiscriminativeDiagnosticsError(
            "max_near_before_group_pairs must be positive"
        )

    threshold_payload = {
        **DEFAULT_ACTION_DISCRIMINATIVE_THRESHOLDS,
        **dict(thresholds or {}),
    }
    split_counts: Counter[str] = Counter(row.split for row in rows)
    source_counts: Counter[str] = Counter(row.source for row in rows)
    edit_size_buckets: Counter[str] = Counter(_edit_size_bucket(row.edit_size) for row in rows)
    action_text_token_buckets: Counter[str] = Counter(
        _token_count_bucket(_active_token_count(row.action_text)) for row in rows
    )
    action_abs_token_buckets: Counter[str] = Counter(
        _token_count_bucket(_active_token_count(row.action_abs)) for row in rows
    )
    action_clusters: Counter[str] = Counter(action_signature(row) for row in rows)
    diff_shapes: Counter[str] = Counter(_dedup_key(row, "diff_shape") for row in rows)

    action_text_nonempty = sum(1 for row in rows if _active_token_count(row.action_text) > 0)
    action_abs_nonempty = sum(1 for row in rows if _active_token_count(row.action_abs) > 0)
    metadata_quality = {
        "rows_with_repo": sum(1 for row in rows if bool(row.repo)),
        "rows_with_path": sum(1 for row in rows if bool(row.path)),
        "rows_with_commit": sum(1 for row in rows if bool(row.commit)),
        "rows_with_license": sum(1 for row in rows if bool(row.license)),
        "rows_with_filter_flags": sum(1 for row in rows if bool(row.filter_flags)),
        "rows_with_dedup_keys": sum(1 for row in rows if bool(row.dedup_keys)),
        "action_text_nonempty_rows": action_text_nonempty,
        "action_text_nonempty_ratio": _ratio(action_text_nonempty, len(rows)),
        "action_abs_nonempty_rows": action_abs_nonempty,
        "action_abs_nonempty_ratio": _ratio(action_abs_nonempty, len(rows)),
    }

    before_groups = _group_rows(rows, key=state_before_hash)
    file_groups = _group_rows(rows, key=lambda row: f"{row.repo}\0{row.path}")
    action_groups = _group_rows(rows, key=action_signature)
    edit_bucket_groups = _group_rows(rows, key=lambda row: _edit_size_bucket(row.edit_size))
    diff_shape_groups = _group_rows(rows, key=lambda row: _dedup_key(row, "diff_shape"))
    exact_before_pair_count = _same_group_pair_count(before_groups)
    same_before_different_after_pairs = _same_group_different_after_pairs(before_groups)
    same_file_pairs = _same_group_different_after_pairs(file_groups)
    action_cluster_pairs = _same_group_different_after_pairs(action_groups)
    edit_size_bucket_pairs = _same_group_different_after_pairs(edit_bucket_groups)
    diff_shape_pairs = _same_group_different_after_pairs(diff_shape_groups)
    near_before = _near_before_pair_report(
        rows,
        hamming_threshold=near_before_hamming_threshold,
        max_group_pairs=max_near_before_group_pairs,
    )

    hard_negative_pools = {
        "same_before_different_after": {
            "available": same_before_different_after_pairs > 0,
            "pair_count": same_before_different_after_pairs,
        },
        "near_before_different_after": {
            "available": near_before["different_after_pair_count"] > 0,
            "pair_count": near_before["different_after_pair_count"],
            "scan_truncated": near_before["scan_truncated"],
        },
        "same_file": {
            "available": same_file_pairs > 0,
            "pair_count": same_file_pairs,
        },
        "action_cluster": {
            "available": action_cluster_pairs > 0,
            "pair_count": action_cluster_pairs,
        },
        "edit_size_controlled": {
            "available": edit_size_bucket_pairs > 0,
            "pair_count": edit_size_bucket_pairs,
        },
        "diff_shape_controlled": {
            "available": diff_shape_pairs > 0,
            "pair_count": diff_shape_pairs,
        },
    }
    unavailable_pools = tuple(
        name for name, payload in sorted(hard_negative_pools.items()) if not payload["available"]
    )
    heldout_rows = split_counts.get("val", 0) + split_counts.get("test", 0)
    duplicate_pressure = {
        "exact_before_unique_count": len(before_groups),
        "exact_before_multirow_group_count": sum(1 for group in before_groups.values() if len(group) > 1),
        "exact_before_pair_count": exact_before_pair_count,
        "same_before_different_after_pair_count": same_before_different_after_pairs,
        "near_before_hamming_threshold": near_before_hamming_threshold,
        "near_before_group_pair_scan_count": near_before["group_pair_scan_count"],
        "near_before_pair_count": near_before["pair_count"],
        "near_before_different_after_pair_count": near_before["different_after_pair_count"],
        "near_before_scan_truncated": near_before["scan_truncated"],
    }
    claim_readiness = _claim_readiness(
        heldout_rows=heldout_rows,
        metadata_quality=metadata_quality,
        hard_negative_pools=hard_negative_pools,
        thresholds=threshold_payload,
    )
    report = {
        "schema_version": ACTION_DISCRIMINATIVE_SHARD_REPORT_SCHEMA_VERSION,
        "row_count": len(rows),
        "split_counts": _ordered_counts(split_counts, preferred=("train", "val", "test")),
        "source_counts": _ordered_counts(source_counts),
        "edit_size_buckets": _ordered_counts(edit_size_buckets),
        "action_class_distribution": {
            "unique_action_signature_count": len(action_clusters),
            "top_action_signatures": _top_counts(action_clusters),
            "action_text_token_count_buckets": _ordered_counts(action_text_token_buckets),
            "action_abs_token_count_buckets": _ordered_counts(action_abs_token_buckets),
            "diff_shape_top_counts": _top_counts(diff_shapes),
        },
        "metadata_quality": metadata_quality,
        "duplicate_pressure": duplicate_pressure,
        "hard_negative_pools": hard_negative_pools,
        "unavailable_hard_negative_pools": list(unavailable_pools),
        "claim_readiness": claim_readiness,
    }
    return validate_action_discriminative_shard_report_payload(report)


def validate_action_discriminative_shard_report_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an action-discriminative shard report payload."""

    if payload.get("schema_version") != ACTION_DISCRIMINATIVE_SHARD_REPORT_SCHEMA_VERSION:
        raise ActionDiscriminativeDiagnosticsError("unsupported action-discriminative report schema")
    row_count = _non_negative_int(payload.get("row_count"), "row_count")
    for key in (
        "split_counts",
        "source_counts",
        "edit_size_buckets",
        "metadata_quality",
        "duplicate_pressure",
        "hard_negative_pools",
        "claim_readiness",
    ):
        if not isinstance(payload.get(key), Mapping):
            raise ActionDiscriminativeDiagnosticsError(f"{key} must be a JSON object")
    split_total = sum(_non_negative_int(value, f"split_counts.{key}") for key, value in payload["split_counts"].items())
    if split_total != row_count:
        raise ActionDiscriminativeDiagnosticsError("split_counts must sum to row_count")
    _ensure_json_native(payload)
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def action_diagnostic_metadata(row: PackedTransition) -> dict[str, str]:
    """Return row metadata consumed by hard-negative and surprise diagnostics."""

    return {
        "state_before_hash": state_before_hash(row),
        "state_after_hash": state_after_hash(row),
        "state_before_simhash": state_before_simhash(row),
        "action_cluster": action_signature(row),
        "action_abs_cluster": action_signature(row),
        "edit_size_bucket": _edit_size_bucket(row.edit_size),
    }


def state_before_hash(row: PackedTransition) -> str:
    return token_sequence_hash(row.state_before)


def state_after_hash(row: PackedTransition) -> str:
    return token_sequence_hash(row.state_after)


def state_before_simhash(row: PackedTransition) -> str:
    return token_sequence_simhash(row.state_before)


def action_signature(row: PackedTransition) -> str:
    ids = _active_ids(row.action_abs) or _active_ids(row.action_text)
    if not ids:
        return "empty-action"
    return "act-" + _hash_ints(ids)[:16]


def token_sequence_hash(sequence: TokenSequence | Sequence[int]) -> str:
    return _hash_ints(_active_ids(sequence))


def token_sequence_simhash(sequence: TokenSequence | Sequence[int]) -> str:
    ids = _active_ids(sequence)
    if not ids:
        return "0" * 16
    weights = [0] * 64
    for token_id in ids:
        token_hash = int.from_bytes(
            hashlib.blake2b(str(token_id).encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for bit_index in range(64):
            if token_hash & (1 << bit_index):
                weights[bit_index] += 1
            else:
                weights[bit_index] -= 1
    result = 0
    for bit_index, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit_index
    return f"{result:016x}"


def _claim_readiness(
    *,
    heldout_rows: int,
    metadata_quality: Mapping[str, Any],
    hard_negative_pools: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    failures: list[str] = []
    same_file_or_near = (
        int(hard_negative_pools["same_file"]["pair_count"])
        + int(hard_negative_pools["same_before_different_after"]["pair_count"])
        + int(hard_negative_pools["near_before_different_after"]["pair_count"])
    )
    checks = {
        "heldout_rows": heldout_rows,
        "same_file_or_near_before_pairs": same_file_or_near,
        "action_cluster_pairs": int(hard_negative_pools["action_cluster"]["pair_count"]),
        "edit_size_bucket_pairs": int(hard_negative_pools["edit_size_controlled"]["pair_count"]),
        "action_text_nonempty_ratio": float(metadata_quality["action_text_nonempty_ratio"]),
        "action_abs_nonempty_ratio": float(metadata_quality["action_abs_nonempty_ratio"]),
    }
    if checks["heldout_rows"] < int(thresholds["min_heldout_rows"]):
        failures.append("insufficient_heldout_rows")
    if checks["same_file_or_near_before_pairs"] < int(thresholds["min_same_file_or_near_before_pairs"]):
        failures.append("missing_same_file_or_near_before_negatives")
    if checks["action_cluster_pairs"] < int(thresholds["min_action_cluster_pairs"]):
        failures.append("missing_action_cluster_negatives")
    if checks["edit_size_bucket_pairs"] < int(thresholds["min_edit_size_bucket_pairs"]):
        failures.append("missing_edit_size_controlled_negatives")
    if checks["action_text_nonempty_ratio"] < float(thresholds["min_action_text_nonempty_ratio"]):
        failures.append("low_action_text_coverage")
    if checks["action_abs_nonempty_ratio"] < float(thresholds["min_action_abs_nonempty_ratio"]):
        failures.append("low_action_abs_coverage")
    return {
        "positive_action_use_claim_ready": not failures,
        "thresholds": dict(thresholds),
        "checks": checks,
        "failure_reasons": failures,
    }


def _near_before_pair_report(
    rows: Sequence[PackedTransition],
    *,
    hamming_threshold: int,
    max_group_pairs: int,
) -> dict[str, Any]:
    simhash_groups = _group_rows(rows, key=state_before_simhash)
    keys = sorted(simhash_groups)
    pair_count = 0
    different_after_pair_count = 0
    group_pair_scan_count = 0
    scan_truncated = False

    for index, left_key in enumerate(keys):
        left_group = simhash_groups[left_key]
        same_group_pairs = _different_after_pair_count(left_group)
        pair_count += _choose2(len(left_group))
        different_after_pair_count += same_group_pairs
        for right_key in keys[index + 1 :]:
            group_pair_scan_count += 1
            if group_pair_scan_count > max_group_pairs:
                scan_truncated = True
                break
            if hamming_distance_hex(left_key, right_key) <= hamming_threshold:
                right_group = simhash_groups[right_key]
                pair_count += len(left_group) * len(right_group)
                different_after_pair_count += _cross_different_after_pair_count(left_group, right_group)
        if scan_truncated:
            break

    return {
        "pair_count": pair_count,
        "different_after_pair_count": different_after_pair_count,
        "group_pair_scan_count": min(group_pair_scan_count, max_group_pairs),
        "scan_truncated": scan_truncated,
    }


def _group_rows(
    rows: Sequence[PackedTransition],
    *,
    key: Any,
) -> dict[str, tuple[PackedTransition, ...]]:
    grouped: dict[str, list[PackedTransition]] = {}
    for row in rows:
        grouped.setdefault(str(key(row)), []).append(row)
    return {group_key: tuple(group) for group_key, group in grouped.items()}


def _same_group_pair_count(groups: Mapping[str, Sequence[PackedTransition]]) -> int:
    return sum(_choose2(len(group)) for group in groups.values())


def _same_group_different_after_pairs(groups: Mapping[str, Sequence[PackedTransition]]) -> int:
    return sum(_different_after_pair_count(group) for group in groups.values())


def _different_after_pair_count(group: Sequence[PackedTransition]) -> int:
    total = _choose2(len(group))
    same_after = Counter(state_after_hash(row) for row in group)
    return total - sum(_choose2(count) for count in same_after.values())


def _cross_different_after_pair_count(
    left_group: Sequence[PackedTransition],
    right_group: Sequence[PackedTransition],
) -> int:
    total = len(left_group) * len(right_group)
    left_after = Counter(state_after_hash(row) for row in left_group)
    right_after = Counter(state_after_hash(row) for row in right_group)
    same_after = sum(left_after[key] * right_after.get(key, 0) for key in left_after)
    return total - same_after


def _dedup_key(row: PackedTransition, key: str) -> str:
    prefix = f"{key}:"
    for value in row.dedup_keys:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return "unavailable"


def _active_token_count(sequence: TokenSequence) -> int:
    return len(_active_ids(sequence))


def _active_ids(sequence: TokenSequence | Sequence[int]) -> tuple[int, ...]:
    if isinstance(sequence, TokenSequence):
        ids = tuple(int(value) for value in sequence.input_ids)
        if sequence.attention_mask is None:
            return tuple(value for value in ids if value != 0)
        masks = tuple(bool(value) for value in sequence.attention_mask)
        return tuple(value for value, keep in zip(ids, masks) if keep and value != 0)
    return tuple(int(value) for value in sequence if int(value) != 0)


def _hash_ints(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _edit_size_bucket(edit_size: int) -> str:
    if edit_size <= 2:
        return "xs:0-2"
    if edit_size <= 10:
        return "s:3-10"
    if edit_size <= 50:
        return "m:11-50"
    if edit_size <= 150:
        return "l:51-150"
    return "xl:151+"


def _token_count_bucket(count: int) -> str:
    if count == 0:
        return "empty"
    if count <= 4:
        return "xs:1-4"
    if count <= 16:
        return "s:5-16"
    if count <= 64:
        return "m:17-64"
    if count <= 256:
        return "l:65-256"
    return "xl:257+"


def _top_counts(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:_TOP_COUNT_LIMIT]
    ]


def _ordered_counts(counter: Counter[str], *, preferred: Sequence[str] = ()) -> dict[str, int]:
    ordered: dict[str, int] = {}
    for key in preferred:
        ordered[key] = int(counter.get(key, 0))
    for key, count in sorted(counter.items()):
        if key not in ordered:
            ordered[key] = int(count)
    return ordered


def _choose2(count: int) -> int:
    return count * (count - 1) // 2


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ActionDiscriminativeDiagnosticsError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ActionDiscriminativeDiagnosticsError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ActionDiscriminativeDiagnosticsError(f"{name} must be a non-negative integer")
    return result


def _ensure_json_native(payload: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ActionDiscriminativeDiagnosticsError("report must be finite JSON-native data") from exc
    _ensure_finite(decoded)


def _ensure_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ActionDiscriminativeDiagnosticsError("report contains non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _ensure_finite(item)
    elif isinstance(value, list):
        for item in value:
            _ensure_finite(item)
