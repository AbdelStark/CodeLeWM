"""Utilities for summarizing execution-training HF Jobs event logs."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


JOB_EVENT_PREFIX = "CODELEWM_JOB_EVENT "
JOB_EVENT_SUMMARY_SCHEMA_VERSION = "codelewm.hf_job_event_summary.v1"


def parse_job_event_line(line: str, *, strict: bool = False) -> dict[str, Any] | None:
    """Parse one live HF log line or persisted JSONL event row."""

    raw = line.strip()
    if not raw:
        return None
    prefix_at = raw.find(JOB_EVENT_PREFIX)
    if prefix_at >= 0:
        raw = raw[prefix_at + len(JOB_EVENT_PREFIX) :].strip()
    elif not raw.startswith("{"):
        if strict:
            raise ValueError("line does not contain a CodeLeWM job event")
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        if strict:
            raise
        return None
    if not isinstance(payload, dict):
        if strict:
            raise ValueError("CodeLeWM job event payload must be a JSON object")
        return None
    if not isinstance(payload.get("event"), str):
        if strict:
            raise ValueError("CodeLeWM job event payload is missing event")
        return None
    return payload


def parse_job_event_lines(
    lines: Iterable[str], *, strict: bool = False
) -> tuple[dict[str, Any], ...]:
    """Return parsed CodeLeWM job events from live logs or artifact JSONL."""

    events: list[dict[str, Any]] = []
    for line in lines:
        event = parse_job_event_line(line, strict=strict)
        if event is not None:
            events.append(event)
    return tuple(events)


def summarize_job_events(
    events: Iterable[Mapping[str, Any]],
    *,
    job_id: str | None = None,
    job_stage: str | None = None,
    job_name: str | None = None,
    job_created_at: str | None = None,
    job_message: str | None = None,
    collapse_threshold: float = 0.20,
) -> dict[str, Any]:
    """Build a compact JSON-native status summary for a job event stream."""

    event_list = [dict(event) for event in events]
    counts = Counter(
        str(event.get("event")) for event in event_list if event.get("event")
    )
    latest_progress = _latest_progress(event_list)
    latest_collapse = _latest_collapse(event_list, collapse_threshold)
    latest_checkpoint = _latest_fields(event_list, "execution_training.checkpoint")
    completion = _latest_fields(event_list, "execution_training.complete")

    summary = {
        "schema_version": JOB_EVENT_SUMMARY_SCHEMA_VERSION,
        "job": {
            "id": job_id,
            "stage": job_stage,
            "name": job_name,
            "created_at": job_created_at,
            "message": job_message,
        },
        "event_count": len(event_list),
        "event_counts": dict(sorted(counts.items())),
        "first_event": _event_name(event_list[0]) if event_list else None,
        "latest_event": _event_name(event_list[-1]) if event_list else None,
        "latest_start": _latest_fields(event_list, "execution_training.start"),
        "latest_progress": latest_progress,
        "latest_collapse": latest_collapse,
        "latest_checkpoint": latest_checkpoint,
        "completion": completion,
        "health": {
            "has_events": bool(event_list),
            "has_progress": latest_progress is not None or completion is not None,
            "complete": completion is not None or job_stage == "COMPLETED",
            "collapse_threshold": collapse_threshold,
            "collapse_ok": (
                True
                if latest_collapse is None
                else bool(latest_collapse.get("passed"))
            ),
        },
    }
    if latest_progress is not None:
        summary["health"]["remaining_steps"] = latest_progress.get("remaining_steps")
        summary["health"]["eta_seconds"] = latest_progress.get("eta_seconds")
    return _json_safe_mapping(summary)


def _latest_progress(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    fields = _latest_fields(events, "execution_training.progress")
    if fields is None:
        return None
    step = _as_int(fields.get("step"))
    max_steps = _as_int(fields.get("max_steps"))
    progress = _as_float(fields.get("progress"))
    if progress is None and step is not None and max_steps:
        progress = step / max_steps
    payload = {
        "seed": fields.get("seed"),
        "step": step,
        "max_steps": max_steps,
        "progress": progress,
        "progress_percent": None if progress is None else progress * 100.0,
        "remaining_steps": (
            None
            if step is None or max_steps is None
            else max(max_steps - step, 0)
        ),
        "elapsed_seconds": _as_float(fields.get("elapsed_seconds")),
        "eta_seconds": _as_float(fields.get("eta_seconds")),
        "metrics": (
            fields.get("metrics")
            if isinstance(fields.get("metrics"), Mapping)
            else {}
        ),
    }
    return _json_safe_mapping(payload)


def _latest_collapse(
    events: list[dict[str, Any]], threshold: float
) -> dict[str, Any] | None:
    fields = _latest_fields(events, "execution_training.collapse_diagnostics")
    if fields is None:
        return None
    diagnostics = fields.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    ratio = _as_float(diagnostics.get("z_pred_effective_rank_ratio"))
    payload = {
        "seed": fields.get("seed"),
        "step": _as_int(fields.get("step")),
        "z_pred_effective_rank_ratio": ratio,
        "z_target_effective_rank_ratio": _as_float(
            diagnostics.get("z_target_effective_rank_ratio")
        ),
        "threshold": threshold,
        "passed": None if ratio is None else ratio >= threshold,
        "diagnostics": dict(diagnostics),
    }
    return _json_safe_mapping(payload)


def _latest_fields(
    events: list[dict[str, Any]], event_name: str
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") != event_name:
            continue
        fields = event.get("fields")
        if isinstance(fields, Mapping):
            return dict(fields)
        return {}
    return None


def _event_name(event: Mapping[str, Any]) -> str | None:
    name = event.get("event")
    return name if isinstance(name, str) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _json_safe_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(value) for key, value in payload.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, list | tuple):
        return [_json_safe_value(item) for item in value]
    return str(value)
