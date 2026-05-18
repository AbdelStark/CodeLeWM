"""Raw edit filtering and report generation."""

from __future__ import annotations

import ast
import difflib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

from codelewm.data.sources import RawEditRecord
from codelewm.security.license_policy import (
    DEFAULT_PUBLIC_LICENSE_POLICY,
    LicenseDecision,
    SourceLicensePolicy,
    decide_license,
)


MappingDetails = dict[str, Any]
IterableRawRecords = Iterable[RawEditRecord]

DropReasonCode = Literal[
    "parse_error",
    "non_python_path",
    "empty_state",
    "whitespace_only_change",
    "edit_size",
    "edit_ratio",
    "message_length",
    "generated_file",
    "license_denied",
]


@dataclass(frozen=True)
class FilterPolicy:
    """Deterministic policy for raw edit row filtering."""

    min_changed_lines: int = 1
    max_changed_lines: int = 150
    min_edit_ratio: float = 0.02
    max_edit_ratio: float = 0.60
    min_message_chars: int = 8
    max_message_chars: int = 512
    generated_path_markers: tuple[str, ...] = (
        "generated",
        "vendor",
        "third_party",
        "node_modules",
        "__pycache__",
    )
    generated_filename_suffixes: tuple[str, ...] = (
        "_pb2.py",
        "_pb2_grpc.py",
        ".generated.py",
        ".gen.py",
    )


@dataclass(frozen=True)
class DropReason:
    """Machine-readable reason for dropping one raw edit record."""

    code: DropReasonCode
    message: str
    details: MappingDetails = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DroppedRecord:
    """Dropped row plus the reason and license decision that caused it."""

    record_id: str
    reason: DropReason
    license_decision: LicenseDecision | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "record_id": self.record_id,
            "reason": self.reason.to_dict(),
        }
        if self.license_decision is not None:
            payload["license_decision"] = self.license_decision.to_dict()
        return payload


@dataclass(frozen=True)
class FilterReport:
    """Aggregate counts for one filter pass."""

    total_before: int
    total_after: int
    drop_reasons: MappingDetails

    @property
    def total_dropped(self) -> int:
        return self.total_before - self.total_after

    def to_dict(self) -> dict[str, object]:
        return {
            "total_before": self.total_before,
            "total_after": self.total_after,
            "total_dropped": self.total_dropped,
            "drop_reasons": dict(self.drop_reasons),
        }


@dataclass(frozen=True)
class FilteredRecords:
    """Result of applying raw edit filters."""

    kept: tuple[RawEditRecord, ...]
    dropped: tuple[DroppedRecord, ...]
    report: FilterReport

    def to_dict(self) -> dict[str, object]:
        return {
            "kept": len(self.kept),
            "dropped": [record.to_dict() for record in self.dropped],
            "report": self.report.to_dict(),
        }


def filter_raw_edit_records(
    records: IterableRawRecords,
    *,
    policy: FilterPolicy = FilterPolicy(),
    license_policy: SourceLicensePolicy = DEFAULT_PUBLIC_LICENSE_POLICY,
) -> FilteredRecords:
    kept: list[RawEditRecord] = []
    dropped: list[DroppedRecord] = []
    reason_counts: dict[str, int] = {}

    for index, record in enumerate(records):
        drop = evaluate_raw_edit_record(record, policy=policy, license_policy=license_policy)
        if drop is None:
            kept.append(record)
            continue
        dropped.append(
            DroppedRecord(
                record_id=_record_id(record, index),
                reason=drop.reason,
                license_decision=drop.license_decision,
            )
        )
        reason_counts[drop.reason.code] = reason_counts.get(drop.reason.code, 0) + 1

    report = FilterReport(
        total_before=len(kept) + len(dropped),
        total_after=len(kept),
        drop_reasons=reason_counts,
    )
    return FilteredRecords(kept=tuple(kept), dropped=tuple(dropped), report=report)


def evaluate_raw_edit_record(
    record: RawEditRecord,
    *,
    policy: FilterPolicy = FilterPolicy(),
    license_policy: SourceLicensePolicy = DEFAULT_PUBLIC_LICENSE_POLICY,
) -> DroppedRecord | None:
    reason = _first_drop_reason(record, policy=policy)
    license_decision = decide_license(
        source=record.source,
        license=record.license,
        policy=license_policy,
    )
    if reason is None and not license_decision.allowed:
        reason = DropReason(
            code="license_denied",
            message="row license is not allowed for the requested artifact policy",
            details=license_decision.to_dict(),
        )

    if reason is None:
        return None

    return DroppedRecord(
        record_id=_record_id(record, 0),
        reason=reason,
        license_decision=license_decision,
    )


def _first_drop_reason(record: RawEditRecord, *, policy: FilterPolicy) -> DropReason | None:
    if not record.path_before.endswith(".py") or not record.path_after.endswith(".py"):
        return DropReason(
            code="non_python_path",
            message="old and new paths must both be Python files",
            details={"path_before": record.path_before, "path_after": record.path_after},
        )

    generated_path = _generated_path(record, policy)
    if generated_path is not None:
        return DropReason(
            code="generated_file",
            message="generated or vendor file path is excluded",
            details={"path": generated_path},
        )

    if not record.before.strip() or not record.after.strip():
        return DropReason(
            code="empty_state",
            message="before and after source text must be non-empty",
            details={},
        )

    parse_error = _parse_error(record)
    if parse_error is not None:
        return parse_error

    if _without_whitespace(record.before) == _without_whitespace(record.after):
        return DropReason(
            code="whitespace_only_change",
            message="whitespace-only changes are excluded",
            details={},
        )

    changed_lines = _changed_line_count(record.before, record.after)
    if changed_lines < policy.min_changed_lines or changed_lines > policy.max_changed_lines:
        return DropReason(
            code="edit_size",
            message="changed line count is outside the configured bounds",
            details={
                "changed_lines": changed_lines,
                "min_changed_lines": policy.min_changed_lines,
                "max_changed_lines": policy.max_changed_lines,
            },
        )

    edit_ratio = _edit_ratio(record.before, record.after)
    if edit_ratio < policy.min_edit_ratio or edit_ratio > policy.max_edit_ratio:
        return DropReason(
            code="edit_ratio",
            message="edit ratio is outside the configured bounds",
            details={
                "edit_ratio": edit_ratio,
                "min_edit_ratio": policy.min_edit_ratio,
                "max_edit_ratio": policy.max_edit_ratio,
            },
        )

    message_length = len(record.message.strip())
    if message_length < policy.min_message_chars or message_length > policy.max_message_chars:
        return DropReason(
            code="message_length",
            message="action text length is outside the configured bounds",
            details={
                "message_length": message_length,
                "min_message_chars": policy.min_message_chars,
                "max_message_chars": policy.max_message_chars,
            },
        )

    return None


def _generated_path(record: RawEditRecord, policy: FilterPolicy) -> str | None:
    for path in (record.path_before, record.path_after):
        normalized = path.replace("\\", "/").casefold()
        parts = PurePosixPath(normalized).parts
        filename = PurePosixPath(normalized).name
        if any(marker in parts for marker in policy.generated_path_markers):
            return path
        if any(filename.endswith(suffix) for suffix in policy.generated_filename_suffixes):
            return path
    return None


def _parse_error(record: RawEditRecord) -> DropReason | None:
    for field_name, source in (("before", record.before), ("after", record.after)):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return DropReason(
                code="parse_error",
                message="before and after source text must parse as Python",
                details={
                    "field": field_name,
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "error": exc.msg,
                },
            )
    return None


def _changed_line_count(before: str, after: str) -> int:
    matcher = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines())
    changed = 0
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += (before_end - before_start) + (after_end - after_start)
    return changed


def _edit_ratio(before: str, after: str) -> float:
    return 1.0 - difflib.SequenceMatcher(None, before, after).ratio()


def _without_whitespace(value: str) -> str:
    return "".join(value.split())


def _record_id(record: RawEditRecord, fallback_index: int) -> str:
    parts = [record.source, record.repo, record.commit, record.path_after]
    if all(parts):
        return ":".join(parts)
    return f"{record.source}:{fallback_index}"
