"""Run timeline reports for multi-step CodeLeWM workflows."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from .logging import redact_text, redact_value


RUN_TIMELINE_SCHEMA_VERSION = "codelewm.run_timeline.v1"
RUN_TIMELINE_STATUSES = ("completed", "failed")
RUN_TIMELINE_STEP_STATUSES = ("completed", "failed", "skipped")


class RunTimelineError(ValueError):
    """Raised when a run timeline report is malformed."""


@dataclass(frozen=True)
class RunTimelineStep:
    """One ordered step in a CodeLeWM run timeline."""

    step_id: str
    name: str
    order: int
    status: str
    started_at: str
    completed_at: str
    duration_ms: float
    command_id: str | None = None
    artifact_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    typed_failure: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id:
            raise RunTimelineError("timeline step_id must not be empty")
        if not self.name:
            raise RunTimelineError("timeline step name must not be empty")
        _non_negative_int(self.order, "timeline step order")
        if self.status not in RUN_TIMELINE_STEP_STATUSES:
            raise RunTimelineError("timeline step status must be completed, failed, or skipped")
        _non_negative_float(self.duration_ms, "timeline step duration_ms")
        if self.command_id is not None and not self.command_id:
            raise RunTimelineError("timeline step command_id must be null or non-empty")
        if self.typed_failure is not None and not isinstance(self.typed_failure, Mapping):
            raise RunTimelineError("timeline step typed_failure must be null or a mapping")
        _require_json_native(self.to_dict(), "timeline step")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "order": self.order,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "command_id": self.command_id,
            "artifact_ids": list(self.artifact_ids),
            "warnings": [redact_text(warning) for warning in self.warnings],
            "typed_failure": None if self.typed_failure is None else redact_value(dict(self.typed_failure)),
            "metadata": redact_value(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunTimelineStep":
        return cls(
            step_id=_require_string(payload, "step_id", "timeline step"),
            name=_require_string(payload, "name", "timeline step"),
            order=_non_negative_int(payload.get("order"), "timeline step order"),
            status=_require_string(payload, "status", "timeline step"),
            started_at=_require_string(payload, "started_at", "timeline step"),
            completed_at=_require_string(payload, "completed_at", "timeline step"),
            duration_ms=_non_negative_float(payload.get("duration_ms"), "timeline step duration_ms"),
            command_id=None
            if payload.get("command_id") is None
            else _require_string(payload, "command_id", "timeline step"),
            artifact_ids=tuple(str(item) for item in payload.get("artifact_ids", ())),
            warnings=tuple(str(item) for item in payload.get("warnings", ())),
            typed_failure=None
            if payload.get("typed_failure") is None
            else dict(_require_mapping(payload["typed_failure"], "timeline step typed_failure")),
            metadata=dict(_require_mapping(payload.get("metadata", {}), "timeline step metadata")),
        )


@dataclass(frozen=True)
class RunTimelineReport:
    """Schema-versioned timeline report for a CodeLeWM run."""

    run_id: str
    status: str
    command: tuple[str, ...]
    started_at: str
    completed_at: str
    duration_ms: float
    steps: tuple[RunTimelineStep, ...]
    artifact_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    typed_failure: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RUN_TIMELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_run_timeline_report(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status,
            "command": [redact_text(item) for item in self.command],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "steps": [step.to_dict() for step in self.steps],
            "artifact_ids": list(self.artifact_ids),
            "warnings": [redact_text(warning) for warning in self.warnings],
            "typed_failure": None if self.typed_failure is None else redact_value(dict(self.typed_failure)),
            "metadata": redact_value(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunTimelineReport":
        return cls(
            schema_version=_require_string(payload, "schema_version", "timeline report"),
            run_id=_require_string(payload, "run_id", "timeline report"),
            status=_require_string(payload, "status", "timeline report"),
            command=tuple(str(item) for item in payload.get("command", ())),
            started_at=_require_string(payload, "started_at", "timeline report"),
            completed_at=_require_string(payload, "completed_at", "timeline report"),
            duration_ms=_non_negative_float(payload.get("duration_ms"), "timeline report duration_ms"),
            steps=tuple(
                RunTimelineStep.from_dict(_require_mapping(item, "timeline step"))
                for item in payload.get("steps", ())
            ),
            artifact_ids=tuple(str(item) for item in payload.get("artifact_ids", ())),
            warnings=tuple(str(item) for item in payload.get("warnings", ())),
            typed_failure=None
            if payload.get("typed_failure") is None
            else dict(_require_mapping(payload["typed_failure"], "timeline report typed_failure")),
            metadata=dict(_require_mapping(payload.get("metadata", {}), "timeline report metadata")),
        )


class RunTimelineRecorder:
    """Small in-process helper for recording ordered run steps."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        command: Sequence[str] = (),
        started_at: str | None = None,
    ) -> None:
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.command = tuple(str(item) for item in command)
        self.started_at = started_at or _utc_now()
        self._started_perf = time.perf_counter()
        self._steps: list[RunTimelineStep] = []
        self._warnings: list[str] = []

    def step(
        self,
        name: str,
        *,
        command_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "_RunTimelineStepContext":
        return _RunTimelineStepContext(
            recorder=self,
            name=name,
            command_id=command_id,
            metadata={} if metadata is None else dict(metadata),
        )

    def add_warning(self, warning: str) -> None:
        self._warnings.append(redact_text(str(warning)))

    def to_report(
        self,
        *,
        status: str | None = None,
        artifact_ids: Sequence[str] = (),
        typed_failure: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunTimelineReport:
        completed_at = _utc_now()
        inferred_status = "failed" if any(step.status == "failed" for step in self._steps) else "completed"
        return RunTimelineReport(
            run_id=self.run_id,
            status=status or inferred_status,
            command=self.command,
            started_at=self.started_at,
            completed_at=completed_at,
            duration_ms=_duration_ms(self._started_perf),
            steps=tuple(self._steps),
            artifact_ids=tuple(str(item) for item in artifact_ids),
            warnings=tuple(self._warnings),
            typed_failure=None if typed_failure is None else dict(typed_failure),
            metadata={} if metadata is None else dict(metadata),
        )

    def _append_step(self, step: RunTimelineStep) -> None:
        self._steps.append(step)


class _RunTimelineStepContext(AbstractContextManager["_RunTimelineStepContext"]):
    def __init__(
        self,
        *,
        recorder: RunTimelineRecorder,
        name: str,
        command_id: str | None,
        metadata: Mapping[str, Any],
    ) -> None:
        self.recorder = recorder
        self.name = name
        self.command_id = command_id
        self.metadata = dict(metadata)
        self.artifact_ids: list[str] = []
        self.warnings: list[str] = []
        self._started_at = ""
        self._started_perf = 0.0

    def __enter__(self) -> "_RunTimelineStepContext":
        self._started_at = _utc_now()
        self._started_perf = time.perf_counter()
        return self

    def add_artifact(self, artifact_id: str) -> None:
        if artifact_id:
            self.artifact_ids.append(str(artifact_id))

    def add_warning(self, warning: str) -> None:
        self.warnings.append(redact_text(str(warning)))

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool | None:
        status = "failed" if exc is not None else "completed"
        failure = None if exc is None else typed_failure_from_exception(exc)
        self.recorder._append_step(
            RunTimelineStep(
                step_id=_slug(self.name),
                name=self.name,
                order=len(self.recorder._steps) + 1,
                status=status,
                started_at=self._started_at,
                completed_at=_utc_now(),
                duration_ms=_duration_ms(self._started_perf),
                command_id=self.command_id,
                artifact_ids=tuple(self.artifact_ids),
                warnings=tuple(self.warnings),
                typed_failure=failure,
                metadata=self.metadata,
            )
        )
        return None


def typed_failure_from_exception(exc: BaseException) -> dict[str, Any]:
    """Return a redacted typed failure payload for timeline reports."""

    return {
        "error_type": exc.__class__.__name__,
        "message": redact_text(str(exc)),
    }


def validate_run_timeline_report(report: RunTimelineReport) -> RunTimelineReport:
    """Validate a run timeline report object."""

    if report.schema_version != RUN_TIMELINE_SCHEMA_VERSION:
        raise RunTimelineError(
            "unsupported run timeline schema; "
            f"expected {RUN_TIMELINE_SCHEMA_VERSION!r}, got {report.schema_version!r}"
        )
    if not report.run_id:
        raise RunTimelineError("timeline run_id must not be empty")
    if report.status not in RUN_TIMELINE_STATUSES:
        raise RunTimelineError("timeline status must be completed or failed")
    _non_negative_float(report.duration_ms, "timeline duration_ms")
    orders = [step.order for step in report.steps]
    if orders != list(range(1, len(report.steps) + 1)):
        raise RunTimelineError("timeline step order must be contiguous from 1")
    if report.status == "failed" and report.typed_failure is None and not any(
        step.typed_failure is not None for step in report.steps
    ):
        raise RunTimelineError("failed timeline must include a typed failure")
    _require_json_native(report.to_dict(), "timeline report")
    return report


def validate_run_timeline_report_payload(payload: Mapping[str, Any]) -> RunTimelineReport:
    """Return a validated timeline report from JSON payload."""

    return RunTimelineReport.from_dict(payload)


def write_run_timeline_report(report: RunTimelineReport, path: Path) -> None:
    """Write a run timeline report JSON file."""

    validate_run_timeline_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_run_timeline_report(path: Path | str) -> RunTimelineReport:
    """Read and validate a run timeline report."""

    return validate_run_timeline_report_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _duration_ms(started_perf: float) -> float:
    return max(0.0, float((time.perf_counter() - started_perf) * 1000.0))


def _slug(value: str) -> str:
    chars = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "_":
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "step"


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunTimelineError(f"{name} must be a mapping")
    return value


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    if key not in payload:
        raise RunTimelineError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise RunTimelineError(f"{section}.{key} must be a string")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RunTimelineError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RunTimelineError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise RunTimelineError(f"{name} must be a non-negative integer")
    return parsed


def _non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise RunTimelineError(f"{name} must be a non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RunTimelineError(f"{name} must be a non-negative number") from exc
    if parsed < 0.0:
        raise RunTimelineError(f"{name} must be a non-negative number")
    return parsed


def _require_json_native(value: Any, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise RunTimelineError(f"{name} must be JSON native") from exc
