"""Structured JSONL logging and redaction helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


LOG_EVENT_SCHEMA_VERSION = "codelewm.log_event.v1"
LogLevel = Literal["debug", "info", "warning", "error"]
LOG_LEVELS: tuple[str, ...] = ("debug", "info", "warning", "error")

_SECRET_KEY_RE = re.compile(r"(token|api[_-]?key|password|passwd|credential|secret)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r"(?i)("
    r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{12,}|"
    r"xox[baprs]-[A-Za-z0-9-]{12,}|"
    r"AKIA[0-9A-Z]{12,}"
    r")"
)
_MAX_TEXT_LINES = 20
_MAX_TEXT_CHARS = 4000


class LogEventError(ValueError):
    """Raised when a structured log event is malformed."""


@dataclass(frozen=True)
class LogEvent:
    """Schema-versioned local log event."""

    event: str
    level: LogLevel
    run_id: str
    step: str
    message: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    schema_version: str = LOG_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOG_EVENT_SCHEMA_VERSION:
            raise LogEventError(f"schema_version must be {LOG_EVENT_SCHEMA_VERSION!r}")
        if not self.event:
            raise LogEventError("event must not be empty")
        if self.level not in LOG_LEVELS:
            raise LogEventError("level must be debug, info, warning, or error")
        if not self.run_id:
            raise LogEventError("run_id must not be empty")
        if not self.step:
            raise LogEventError("step must not be empty")
        if not self.message:
            raise LogEventError("message must not be empty")
        if self.artifact_id is not None and not self.artifact_id:
            raise LogEventError("artifact_id must be null or non-empty")
        if not isinstance(self.fields, Mapping):
            raise LogEventError("fields must be a JSON object")
        _ensure_json_native(redact_value(self.fields), field_name="fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event": self.event,
            "level": self.level,
            "run_id": self.run_id,
            "artifact_id": self.artifact_id,
            "step": self.step,
            "message": redact_text(self.message),
            "fields": redact_value(self.fields),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogEvent":
        _reject_unknown(
            payload,
            {
                "schema_version",
                "event",
                "level",
                "run_id",
                "artifact_id",
                "step",
                "message",
                "fields",
            },
            "log event",
        )
        if "fields" not in payload:
            raise LogEventError("log event.fields is required")
        fields = payload["fields"]
        if not isinstance(fields, Mapping):
            raise LogEventError("log event.fields must be an object")
        return cls(
            schema_version=_require_string(payload, "schema_version", "log event"),
            event=_require_string(payload, "event", "log event"),
            level=_require_string(payload, "level", "log event"),  # type: ignore[arg-type]
            run_id=_require_string(payload, "run_id", "log event"),
            artifact_id=None
            if payload.get("artifact_id") is None
            else _require_string(payload, "artifact_id", "log event"),
            step=_require_string(payload, "step", "log event"),
            message=_require_string(payload, "message", "log event"),
            fields=redact_value(dict(fields)),
        )


def log_event_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for structured log events."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": LOG_EVENT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "event",
            "level",
            "run_id",
            "artifact_id",
            "step",
            "message",
            "fields",
        ],
        "properties": {
            "schema_version": {"const": LOG_EVENT_SCHEMA_VERSION},
            "event": {"type": "string", "minLength": 1},
            "level": {"type": "string", "enum": list(LOG_LEVELS)},
            "run_id": {"type": "string", "minLength": 1},
            "artifact_id": {"type": ["string", "null"]},
            "step": {"type": "string", "minLength": 1},
            "message": {"type": "string", "minLength": 1},
            "fields": {"type": "object"},
        },
    }


def validate_log_event_payload(payload: Mapping[str, Any]) -> LogEvent:
    """Validate and normalize a log event payload."""

    return LogEvent.from_dict(payload)


def write_log_event_jsonl(event: LogEvent, path: Path | str) -> LogEvent:
    """Append one validated event to a JSONL log file."""

    event = validate_log_event_payload(event.to_dict())
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    return event


def redact_value(value: Any) -> Any:
    """Return a JSON-native copy with secrets, home paths, and long text redacted."""

    return _redact_value(value, key=None)


def redact_text(value: str) -> str:
    """Redact one string value without dropping useful short context."""

    value = _redact_home_path(value)
    value = _SECRET_VALUE_RE.sub("[REDACTED_SECRET]", value)
    line_count = len(value.splitlines())
    if line_count > _MAX_TEXT_LINES or len(value) > _MAX_TEXT_CHARS:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"[REDACTED_LONG_TEXT sha256={digest} lines={line_count} chars={len(value)}]"
    return value


def _redact_value(value: Any, *, key: str | None) -> Any:
    if key is not None and _SECRET_KEY_RE.search(key):
        return "[REDACTED_SECRET]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {str(item_key): _redact_value(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, key=None) for item in value]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return redact_text(str(value))


def _redact_home_path(value: str) -> str:
    home = str(Path.home())
    if home and value.startswith(home):
        return "~" + value[len(home) :]
    return value.replace(home + "/", "~/") if home else value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise LogEventError(f"{section} contains unknown key(s): {', '.join(unknown)}")


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    if key not in payload:
        raise LogEventError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise LogEventError(f"{section}.{key} must be a string")
    return value


def _ensure_json_native(payload: Any, *, field_name: str) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LogEventError(f"{field_name} must be JSON-native: {exc}") from exc
