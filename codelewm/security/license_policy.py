"""License policy decisions for public CodeLeWM artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from typing import Literal


ArtifactPolicy = Literal["exclude", "metadata_only", "embeddings", "full_text"]
SourceKind = Literal["commitpackft", "commitpack", "agentpack", "synthetic", "local_repo"]
PUBLIC_LICENSE_GATE_SCHEMA_VERSION = "codelewm.public_license_gate.v1"

PERMISSIVE_PUBLIC_LICENSES = (
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc0-1.0",
    "isc",
    "mit",
    "unlicense",
)

_LICENSE_ALIASES = {
    "apache-license-2.0": "apache-2.0",
    "apache-2": "apache-2.0",
    "cc0": "cc0-1.0",
    "mit-license": "mit",
    "the-unlicense": "unlicense",
}


class LicenseGateError(ValueError):
    """Raised when a public artifact license gate fails."""


@dataclass(frozen=True)
class SourceLicensePolicy:
    """License allowlist for a source and artifact class."""

    source: SourceKind | None = None
    allowed_licenses: tuple[str, ...] = PERMISSIVE_PUBLIC_LICENSES
    require_license_field: bool = True
    redistribution_allowed: bool = True
    derived_artifact_policy: ArtifactPolicy = "full_text"


@dataclass(frozen=True)
class LicenseDecision:
    """Machine-readable decision for one source row."""

    allowed: bool
    reason: str
    source: SourceKind
    license: str | None
    artifact_policy: ArtifactPolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "source": self.source,
            "license": self.license,
            "artifact_policy": self.artifact_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LicenseDecision":
        _reject_unknown(
            payload,
            {"allowed", "reason", "source", "license", "artifact_policy"},
            "license decision",
        )
        return cls(
            allowed=_require_bool(payload, "allowed", "license decision"),
            reason=_require_string(payload, "reason", "license decision"),
            source=_require_string(payload, "source", "license decision"),  # type: ignore[arg-type]
            license=None
            if payload.get("license") is None
            else _require_string(payload, "license", "license decision"),
            artifact_policy=_require_string(payload, "artifact_policy", "license decision"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PublicLicenseGateReport:
    """Machine-readable license gate report for public artifacts."""

    artifact_policy: ArtifactPolicy
    included_rows: int
    excluded_rows: int
    blocked_rows: int
    release_allowed: bool
    included_licenses: Mapping[str, int]
    excluded_licenses: Mapping[str, int]
    included_sources: Mapping[str, int]
    excluded_sources: Mapping[str, int]
    excluded_reasons: Mapping[str, int]
    schema_version: str = PUBLIC_LICENSE_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PUBLIC_LICENSE_GATE_SCHEMA_VERSION:
            raise LicenseGateError(f"schema_version must be {PUBLIC_LICENSE_GATE_SCHEMA_VERSION!r}")
        if self.artifact_policy not in {"exclude", "metadata_only", "embeddings", "full_text"}:
            raise LicenseGateError("artifact_policy is invalid")
        for field_name, value in (
            ("included_rows", self.included_rows),
            ("excluded_rows", self.excluded_rows),
            ("blocked_rows", self.blocked_rows),
        ):
            if value < 0:
                raise LicenseGateError(f"{field_name} must be non-negative")
        if self.blocked_rows > self.included_rows:
            raise LicenseGateError("blocked_rows cannot exceed included_rows")
        for field_name, mapping in (
            ("included_licenses", self.included_licenses),
            ("excluded_licenses", self.excluded_licenses),
            ("included_sources", self.included_sources),
            ("excluded_sources", self.excluded_sources),
            ("excluded_reasons", self.excluded_reasons),
        ):
            _validate_count_mapping(mapping, field_name)
        if sum(self.included_licenses.values()) != self.included_rows:
            raise LicenseGateError("included_licenses count must match included_rows")
        if sum(self.included_sources.values()) != self.included_rows:
            raise LicenseGateError("included_sources count must match included_rows")
        if sum(self.excluded_licenses.values()) != self.excluded_rows:
            raise LicenseGateError("excluded_licenses count must match excluded_rows")
        if sum(self.excluded_sources.values()) != self.excluded_rows:
            raise LicenseGateError("excluded_sources count must match excluded_rows")
        if sum(self.excluded_reasons.values()) != self.excluded_rows:
            raise LicenseGateError("excluded_reasons count must match excluded_rows")
        if self.release_allowed != (self.blocked_rows == 0):
            raise LicenseGateError("release_allowed must match blocked_rows")
        _ensure_json_native(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_policy": self.artifact_policy,
            "included_rows": self.included_rows,
            "excluded_rows": self.excluded_rows,
            "blocked_rows": self.blocked_rows,
            "release_allowed": self.release_allowed,
            "included_licenses": dict(self.included_licenses),
            "excluded_licenses": dict(self.excluded_licenses),
            "included_sources": dict(self.included_sources),
            "excluded_sources": dict(self.excluded_sources),
            "excluded_reasons": dict(self.excluded_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicLicenseGateReport":
        return validate_public_license_gate_report(payload)


DEFAULT_PUBLIC_LICENSE_POLICY = SourceLicensePolicy()


def decide_license(
    *,
    source: SourceKind,
    license: str | None,
    policy: SourceLicensePolicy = DEFAULT_PUBLIC_LICENSE_POLICY,
) -> LicenseDecision:
    """Return the public-artifact license decision for one row."""

    artifact_policy: ArtifactPolicy = policy.derived_artifact_policy
    if not policy.redistribution_allowed:
        return LicenseDecision(
            allowed=False,
            reason="redistribution_not_allowed",
            source=source,
            license=normalize_license(license),
            artifact_policy="exclude",
        )

    normalized = normalize_license(license)
    if normalized is None:
        return LicenseDecision(
            allowed=not policy.require_license_field,
            reason="missing_license" if policy.require_license_field else "source_policy_allows_missing_license",
            source=source,
            license=None,
            artifact_policy=artifact_policy if not policy.require_license_field else "exclude",
        )

    allowed = normalized in {normalize_license(item) for item in policy.allowed_licenses}
    return LicenseDecision(
        allowed=allowed,
        reason="allowed" if allowed else "license_not_allowed",
        source=source,
        license=normalized,
        artifact_policy=artifact_policy if allowed else "exclude",
    )


def build_public_license_gate_report(
    *,
    included: Iterable[LicenseDecision],
    excluded: Iterable[LicenseDecision] = (),
    artifact_policy: ArtifactPolicy = "full_text",
) -> PublicLicenseGateReport:
    """Build a public artifact gate report from included and excluded decisions."""

    included_decisions = tuple(included)
    excluded_decisions = tuple(excluded)
    blocked_rows = sum(
        1
        for decision in included_decisions
        if not decision.allowed or decision.artifact_policy != artifact_policy
    )
    return PublicLicenseGateReport(
        artifact_policy=artifact_policy,
        included_rows=len(included_decisions),
        excluded_rows=len(excluded_decisions),
        blocked_rows=blocked_rows,
        release_allowed=blocked_rows == 0,
        included_licenses=_license_counts(included_decisions),
        excluded_licenses=_license_counts(excluded_decisions),
        included_sources=_source_counts(included_decisions),
        excluded_sources=_source_counts(excluded_decisions),
        excluded_reasons=_reason_counts(excluded_decisions),
    )


def validate_public_license_gate_report(payload: Mapping[str, Any]) -> PublicLicenseGateReport:
    """Validate a public license gate report payload."""

    _reject_unknown(
        payload,
        {
            "schema_version",
            "artifact_policy",
            "included_rows",
            "excluded_rows",
            "blocked_rows",
            "release_allowed",
            "included_licenses",
            "excluded_licenses",
            "included_sources",
            "excluded_sources",
            "excluded_reasons",
        },
        "public license gate report",
    )
    return PublicLicenseGateReport(
        schema_version=_require_string(payload, "schema_version", "public license gate report"),
        artifact_policy=_require_string(payload, "artifact_policy", "public license gate report"),  # type: ignore[arg-type]
        included_rows=_require_int(payload, "included_rows", "public license gate report"),
        excluded_rows=_require_int(payload, "excluded_rows", "public license gate report"),
        blocked_rows=_require_int(payload, "blocked_rows", "public license gate report"),
        release_allowed=_require_bool(payload, "release_allowed", "public license gate report"),
        included_licenses=_require_count_mapping(payload, "included_licenses"),
        excluded_licenses=_require_count_mapping(payload, "excluded_licenses"),
        included_sources=_require_count_mapping(payload, "included_sources"),
        excluded_sources=_require_count_mapping(payload, "excluded_sources"),
        excluded_reasons=_require_count_mapping(payload, "excluded_reasons"),
    )


def enforce_public_license_gate(
    report: PublicLicenseGateReport | Mapping[str, Any],
) -> PublicLicenseGateReport:
    """Raise when a public artifact includes rows outside the declared license policy."""

    if isinstance(report, PublicLicenseGateReport):
        gate = report
    else:
        gate = validate_public_license_gate_report(report)
    if not gate.release_allowed or gate.blocked_rows:
        raise LicenseGateError(
            "public artifact license gate failed: "
            f"{gate.blocked_rows} included row(s) violate {gate.artifact_policy!r} policy"
        )
    return gate


def normalize_license(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().casefold().replace("_", "-")
    while "  " in normalized:
        normalized = normalized.replace("  ", " ")
    normalized = normalized.replace(" ", "-")
    normalized = _LICENSE_ALIASES.get(normalized, normalized)
    if normalized in {"", "none", "null", "unknown", "other", "n/a"}:
        return None
    return normalized


def _license_counts(decisions: Iterable[LicenseDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        key = decision.license or "missing"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _source_counts(decisions: Iterable[LicenseDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.source] = counts.get(decision.source, 0) + 1
    return counts


def _reason_counts(decisions: Iterable[LicenseDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.reason] = counts.get(decision.reason, 0) + 1
    return counts


def _validate_count_mapping(mapping: Mapping[str, int], field_name: str) -> None:
    if not isinstance(mapping, Mapping):
        raise LicenseGateError(f"{field_name} must be a mapping")
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise LicenseGateError(f"{field_name} keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LicenseGateError(f"{field_name} values must be non-negative integers")


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise LicenseGateError(f"{section} contains unknown key(s): {', '.join(unknown)}")


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    if key not in payload:
        raise LicenseGateError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise LicenseGateError(f"{section}.{key} must be a string")
    return value


def _require_int(payload: Mapping[str, Any], key: str, section: str) -> int:
    if key not in payload:
        raise LicenseGateError(f"{section}.{key} is required")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise LicenseGateError(f"{section}.{key} must be an integer")
    return value


def _require_bool(payload: Mapping[str, Any], key: str, section: str) -> bool:
    if key not in payload:
        raise LicenseGateError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, bool):
        raise LicenseGateError(f"{section}.{key} must be true or false")
    return value


def _require_count_mapping(payload: Mapping[str, Any], key: str) -> dict[str, int]:
    if key not in payload:
        raise LicenseGateError(f"public license gate report.{key} is required")
    value = payload[key]
    if not isinstance(value, Mapping):
        raise LicenseGateError(f"public license gate report.{key} must be a mapping")
    mapping = {str(item_key): _mapping_int(item_value, key) for item_key, item_value in value.items()}
    _validate_count_mapping(mapping, key)
    return mapping


def _mapping_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LicenseGateError(f"{field_name} values must be integers")
    return value


def _ensure_json_native(payload: Any) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LicenseGateError(f"payload must be JSON-native: {exc}") from exc
