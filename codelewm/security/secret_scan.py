"""Local secret-pattern scanning for reports, logs, and configs.

The scanner emits a schema-versioned report listing every match by file and
line. Matches are redacted before they leave the scanner so secret values are
never re-published in scan output or logs.

Patterns are deliberately aligned with the redaction patterns in
``codelewm.observability.logging``. Adding a pattern here should be mirrored
there so any secret the scanner can flag is also redacted from log output.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SECRET_SCAN_REPORT_SCHEMA_VERSION = "codelewm.secret_scan.v1"

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_api_key", re.compile(r"sk-[A-Za-z0-9_-]{12,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{12,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{12,}")),
    ("aws_access_key_id", re.compile(r"AKIA[0-9A-Z]{12,}")),
    (
        "generic_assigned_secret",
        re.compile(
            r"(?ix)\b"
            r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|"
            r"client[_-]?secret|private[_-]?key)\s*[:=]\s*"
            r"['\"]?"
            r"(?P<value>[A-Za-z0-9_+/.\-]{16,})"
            r"['\"]?"
        ),
    ),
)
_DEFAULT_SCAN_SUFFIXES: frozenset[str] = frozenset(
    {".json", ".jsonl", ".yaml", ".yml", ".toml", ".txt", ".md", ".log", ".env", ".cfg", ".ini"}
)


class SecretScanError(RuntimeError):
    """Raised when a scan request cannot be honored."""


@dataclass(frozen=True)
class SecretFinding:
    """A single secret match in a single file."""

    path: str
    line: int
    pattern: str
    redacted: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "pattern": self.pattern,
            "redacted": self.redacted,
        }


@dataclass(frozen=True)
class SecretScanReport:
    """Schema-versioned report of a secret scan over one or more files."""

    schema_version: str
    ok: bool
    paths_scanned: tuple[str, ...]
    findings: tuple[SecretFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "paths_scanned": list(self.paths_scanned),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def scan_text(text: str, *, path: str = "<text>") -> tuple[SecretFinding, ...]:
    """Return every secret-pattern match in ``text``.

    Matches are anchored to line numbers so triage points at the offending
    line without reading the full document.
    """

    findings: list[SecretFinding] = []
    for line_index, line in enumerate(text.splitlines(), start=1):
        for pattern_name, pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(line):
                token = match.group("value") if "value" in match.groupdict() else match.group(0)
                findings.append(
                    SecretFinding(
                        path=path,
                        line=line_index,
                        pattern=pattern_name,
                        redacted=_redact_token(token),
                    )
                )
    return tuple(findings)


def scan_file(path: Path | str) -> tuple[SecretFinding, ...]:
    """Scan one file for secret patterns; binary files are skipped quietly."""

    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ()
    return scan_text(text, path=str(file_path))


def scan_paths(
    paths: Iterable[Path | str],
    *,
    include_suffixes: Iterable[str] | None = None,
    recursive: bool = True,
) -> SecretScanReport:
    """Scan files under ``paths`` and return a schema-versioned report.

    Directories are walked recursively when ``recursive`` is True. Files are
    filtered by ``include_suffixes``; pass ``None`` to use the default set,
    or an empty iterable to scan every readable file.
    """

    if include_suffixes is None:
        suffixes = _DEFAULT_SCAN_SUFFIXES
    else:
        suffixes = frozenset(suffix.lower() for suffix in include_suffixes)

    scanned: list[str] = []
    findings: list[SecretFinding] = []
    for candidate in _expand_paths(paths, recursive=recursive):
        if suffixes and candidate.suffix.lower() not in suffixes:
            continue
        scanned.append(str(candidate))
        findings.extend(scan_file(candidate))

    return SecretScanReport(
        schema_version=SECRET_SCAN_REPORT_SCHEMA_VERSION,
        ok=not findings,
        paths_scanned=tuple(scanned),
        findings=tuple(findings),
    )


def secret_scan_report_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``SecretScanReport`` payloads."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SECRET_SCAN_REPORT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "ok", "paths_scanned", "findings"],
        "properties": {
            "schema_version": {"const": SECRET_SCAN_REPORT_SCHEMA_VERSION},
            "ok": {"type": "boolean"},
            "paths_scanned": {"type": "array", "items": {"type": "string"}},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "line", "pattern", "redacted"],
                    "properties": {
                        "path": {"type": "string"},
                        "line": {"type": "integer", "minimum": 1},
                        "pattern": {"type": "string"},
                        "redacted": {"type": "string"},
                    },
                },
            },
        },
    }


def validate_secret_scan_report_payload(payload: Mapping[str, Any]) -> SecretScanReport:
    """Validate a JSON payload against the v1 secret-scan report schema."""

    _require_keys(payload, {"schema_version", "ok", "paths_scanned", "findings"})
    schema_version = _require_str(payload, "schema_version")
    if schema_version != SECRET_SCAN_REPORT_SCHEMA_VERSION:
        raise SecretScanError(
            f"unsupported secret-scan schema_version: {schema_version!r}"
        )
    ok = payload["ok"]
    if not isinstance(ok, bool):
        raise SecretScanError("ok must be a boolean")
    paths_value = payload["paths_scanned"]
    if not isinstance(paths_value, Sequence) or isinstance(paths_value, (str, bytes)):
        raise SecretScanError("paths_scanned must be a JSON array")
    paths_scanned: tuple[str, ...] = tuple(_require_str_item(item) for item in paths_value)
    findings_value = payload["findings"]
    if not isinstance(findings_value, Sequence) or isinstance(findings_value, (str, bytes)):
        raise SecretScanError("findings must be a JSON array")
    findings = tuple(_finding_from_mapping(item) for item in findings_value)
    return SecretScanReport(
        schema_version=schema_version,
        ok=ok,
        paths_scanned=paths_scanned,
        findings=findings,
    )


def _expand_paths(paths: Iterable[Path | str], *, recursive: bool) -> Iterable[Path]:
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            yield from sorted(path.rglob("*") if recursive else path.iterdir())
        elif path.is_file():
            yield path


def _redact_token(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"[REDACTED_SECRET sha256={digest} length={len(token)}]"


def _require_keys(payload: Mapping[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise SecretScanError(f"secret-scan report missing required key(s): {', '.join(missing)}")
    extra = sorted(set(payload) - required)
    if extra:
        raise SecretScanError(f"secret-scan report contains unknown key(s): {', '.join(extra)}")


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise SecretScanError(f"secret-scan report {key} must be a string")
    return value


def _require_str_item(value: Any) -> str:
    if not isinstance(value, str):
        raise SecretScanError("secret-scan paths_scanned must contain strings")
    return value


def _finding_from_mapping(payload: Any) -> SecretFinding:
    if not isinstance(payload, Mapping):
        raise SecretScanError("each secret-scan finding must be a JSON object")
    _require_keys(payload, {"path", "line", "pattern", "redacted"})
    path = payload["path"]
    line = payload["line"]
    pattern = payload["pattern"]
    redacted = payload["redacted"]
    if not isinstance(path, str):
        raise SecretScanError("finding.path must be a string")
    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        raise SecretScanError("finding.line must be a positive integer")
    if not isinstance(pattern, str):
        raise SecretScanError("finding.pattern must be a string")
    if not isinstance(redacted, str):
        raise SecretScanError("finding.redacted must be a string")
    return SecretFinding(path=path, line=line, pattern=pattern, redacted=redacted)
