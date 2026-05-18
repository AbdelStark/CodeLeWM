"""Deterministic split assignment and deduplication for raw edit records."""

from __future__ import annotations

import difflib
import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from codelewm.data.sources import RawEditRecord


SplitName = Literal["train", "val", "test"]
DedupDropReasonCode = Literal["exact_duplicate", "train_leakage"]

_SPLITS = {"train", "val", "test"}
_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")


@dataclass(frozen=True)
class SplitPolicy:
    """Deterministic repository-level split policy."""

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: str = "codelewm.transition.v1"
    split_overrides: Mapping[str, SplitName] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not 0.999999 <= total <= 1.000001:
            raise ValueError("split ratios must sum to 1.0")
        if min(self.train_ratio, self.val_ratio, self.test_ratio) < 0:
            raise ValueError("split ratios must be non-negative")
        invalid = set(self.split_overrides.values()) - _SPLITS
        if invalid:
            raise ValueError(f"invalid split override(s): {sorted(invalid)}")


@dataclass(frozen=True)
class DedupPolicy:
    """Exact and near-duplicate rejection policy."""

    near_duplicate_hamming_threshold: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.near_duplicate_hamming_threshold <= 64:
            raise ValueError("near duplicate Hamming threshold must be in [0, 64]")


@dataclass(frozen=True)
class DedupKeys:
    """Stable keys used for exact and near-duplicate checks."""

    exact_transition: str
    exact_before_after: str
    near_state: str
    diff_shape: str

    def to_dict(self) -> dict[str, str]:
        return {
            "exact_transition": self.exact_transition,
            "exact_before_after": self.exact_before_after,
            "near_state": self.near_state,
            "diff_shape": self.diff_shape,
        }


@dataclass(frozen=True)
class SplitAssignment:
    """Record plus its split and deduplication keys."""

    record: RawEditRecord
    split: SplitName
    split_key: str
    dedup_keys: DedupKeys

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": record_id(self.record),
            "split": self.split,
            "split_key": self.split_key,
            "dedup_keys": self.dedup_keys.to_dict(),
        }


@dataclass(frozen=True)
class DedupDroppedRecord:
    """Machine-readable split/dedup drop record."""

    record_id: str
    reason: DedupDropReasonCode
    split: SplitName
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "reason": self.reason,
            "split": self.split,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class SplitDedupReport:
    """Aggregate counts for split assignment and deduplication."""

    total_before: int
    total_after: int
    per_split: Mapping[str, int]
    drop_reasons: Mapping[str, int]

    @property
    def total_dropped(self) -> int:
        return self.total_before - self.total_after

    def to_dict(self) -> dict[str, object]:
        return {
            "total_before": self.total_before,
            "total_after": self.total_after,
            "total_dropped": self.total_dropped,
            "per_split": dict(self.per_split),
            "drop_reasons": dict(self.drop_reasons),
        }


@dataclass(frozen=True)
class SplitDedupResult:
    """Result of split assignment and deduplication."""

    kept: tuple[SplitAssignment, ...]
    dropped: tuple[DedupDroppedRecord, ...]
    report: SplitDedupReport

    def to_dict(self) -> dict[str, object]:
        return {
            "kept": [assignment.to_dict() for assignment in self.kept],
            "dropped": [record.to_dict() for record in self.dropped],
            "report": self.report.to_dict(),
        }


def assign_split(
    record: RawEditRecord,
    *,
    policy: SplitPolicy = SplitPolicy(),
) -> tuple[SplitName, str]:
    """Assign a row to a deterministic split before tokenization."""

    inherited = record.metadata.get("source_split")
    if record.source == "synthetic" and inherited in _SPLITS:
        return cast(SplitName, inherited), split_key(record)

    key = split_key(record)
    if key in policy.split_overrides:
        return policy.split_overrides[key], key

    bucket = _stable_unit_interval(f"{policy.seed}:{key}")
    if bucket < policy.train_ratio:
        return "train", key
    if bucket < policy.train_ratio + policy.val_ratio:
        return "val", key
    return "test", key


def split_key(record: RawEditRecord) -> str:
    repo = _normalize_repo(record.repo)
    if repo:
        return repo
    source_identity = record.metadata.get("source_identity") or record.metadata.get("shard")
    if source_identity:
        return f"{record.source}:{source_identity}"
    return f"{record.source}:{record.commit}:{record.path_after}"


def compute_dedup_keys(record: RawEditRecord) -> DedupKeys:
    before = _normalize_code(record.before)
    after = _normalize_code(record.after)
    action = _normalize_action(record.message)
    return DedupKeys(
        exact_transition=_sha256_parts(before, action, after),
        exact_before_after=_sha256_parts(before, after),
        near_state=_simhash_hex(before),
        diff_shape=_diff_shape_hash(before, after),
    )


def split_and_deduplicate(
    records: Iterable[RawEditRecord],
    *,
    split_policy: SplitPolicy = SplitPolicy(),
    dedup_policy: DedupPolicy = DedupPolicy(),
) -> SplitDedupResult:
    assignments: list[SplitAssignment] = []
    for record in records:
        split, key = assign_split(record, policy=split_policy)
        assignments.append(
            SplitAssignment(
                record=record,
                split=split,
                split_key=key,
                dedup_keys=compute_dedup_keys(record),
            )
        )
    train_assignments = [assignment for assignment in assignments if assignment.split == "train"]
    train_exact_before_after = {
        assignment.dedup_keys.exact_before_after: assignment for assignment in train_assignments
    }
    train_near_states = [(int(assignment.dedup_keys.near_state, 16), assignment) for assignment in train_assignments]

    kept: list[SplitAssignment] = []
    dropped: list[DedupDroppedRecord] = []
    seen_transition: dict[str, SplitAssignment] = {}
    seen_before_after: dict[str, SplitAssignment] = {}

    for assignment in assignments:
        leakage = _train_leakage_drop(
            assignment,
            train_exact_before_after=train_exact_before_after,
            train_near_states=train_near_states,
            dedup_policy=dedup_policy,
        )
        if leakage is not None:
            dropped.append(leakage)
            continue

        duplicate = _exact_duplicate_drop(
            assignment,
            seen_transition=seen_transition,
            seen_before_after=seen_before_after,
        )
        if duplicate is not None:
            dropped.append(duplicate)
            continue

        kept.append(assignment)
        seen_transition[assignment.dedup_keys.exact_transition] = assignment
        seen_before_after[assignment.dedup_keys.exact_before_after] = assignment

    per_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for assignment in kept:
        per_split[assignment.split] += 1

    drop_reasons: dict[str, int] = {}
    for drop in dropped:
        drop_reasons[drop.reason] = drop_reasons.get(drop.reason, 0) + 1

    return SplitDedupResult(
        kept=tuple(kept),
        dropped=tuple(dropped),
        report=SplitDedupReport(
            total_before=len(assignments),
            total_after=len(kept),
            per_split=per_split,
            drop_reasons=drop_reasons,
        ),
    )


def record_id(record: RawEditRecord) -> str:
    return ":".join((record.source, record.repo, record.commit, record.path_after))


def hamming_distance_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _train_leakage_drop(
    assignment: SplitAssignment,
    *,
    train_exact_before_after: Mapping[str, SplitAssignment],
    train_near_states: Iterable[tuple[int, SplitAssignment]],
    dedup_policy: DedupPolicy,
) -> DedupDroppedRecord | None:
    if assignment.split == "train":
        return None

    exact_match = train_exact_before_after.get(assignment.dedup_keys.exact_before_after)
    if exact_match is not None:
        return DedupDroppedRecord(
            record_id=record_id(assignment.record),
            reason="train_leakage",
            split=assignment.split,
            details={
                "match_type": "exact_before_after",
                "matched_record_id": record_id(exact_match.record),
            },
        )

    near_state = int(assignment.dedup_keys.near_state, 16)
    for train_near_state, train_assignment in train_near_states:
        distance = (near_state ^ train_near_state).bit_count()
        if distance <= dedup_policy.near_duplicate_hamming_threshold:
            return DedupDroppedRecord(
                record_id=record_id(assignment.record),
                reason="train_leakage",
                split=assignment.split,
                details={
                    "match_type": "near_state",
                    "hamming_distance": distance,
                    "threshold": dedup_policy.near_duplicate_hamming_threshold,
                    "matched_record_id": record_id(train_assignment.record),
                },
            )
    return None


def _exact_duplicate_drop(
    assignment: SplitAssignment,
    *,
    seen_transition: Mapping[str, SplitAssignment],
    seen_before_after: Mapping[str, SplitAssignment],
) -> DedupDroppedRecord | None:
    exact_transition = seen_transition.get(assignment.dedup_keys.exact_transition)
    if exact_transition is not None:
        return DedupDroppedRecord(
            record_id=record_id(assignment.record),
            reason="exact_duplicate",
            split=assignment.split,
            details={
                "match_type": "exact_transition",
                "matched_record_id": record_id(exact_transition.record),
            },
        )

    exact_before_after = seen_before_after.get(assignment.dedup_keys.exact_before_after)
    if exact_before_after is not None:
        return DedupDroppedRecord(
            record_id=record_id(assignment.record),
            reason="exact_duplicate",
            split=assignment.split,
            details={
                "match_type": "exact_before_after",
                "matched_record_id": record_id(exact_before_after.record),
            },
        )
    return None


def _stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _sha256_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalize_repo(repo: str) -> str:
    return repo.strip().casefold()


def _normalize_code(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _normalize_action(value: str) -> str:
    return " ".join(value.split()).casefold()


def _simhash_hex(value: str) -> str:
    weights = [0] * 64
    tokens = _TOKEN_PATTERN.findall(value)
    if not tokens:
        return "0" * 16
    for token in tokens:
        token_hash = int.from_bytes(
            hashlib.blake2b(token.casefold().encode("utf-8"), digest_size=8).digest(),
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


def _diff_shape_hash(before: str, after: str) -> str:
    matcher = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines())
    histogram = {"replace": 0, "delete": 0, "insert": 0}
    changed_lines = 0
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        histogram[tag] = histogram.get(tag, 0) + 1
        changed_lines += (before_end - before_start) + (after_end - after_start)
    size_bucket = _size_bucket(changed_lines)
    shape = (
        f"r={histogram['replace']};d={histogram['delete']};"
        f"i={histogram['insert']};bucket={size_bucket}"
    )
    return _sha256_parts(shape)


def _size_bucket(changed_lines: int) -> str:
    if changed_lines <= 2:
        return "xs"
    if changed_lines <= 10:
        return "s"
    if changed_lines <= 50:
        return "m"
    if changed_lines <= 150:
        return "l"
    return "xl"
