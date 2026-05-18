"""License policy decisions for public CodeLeWM artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ArtifactPolicy = Literal["exclude", "metadata_only", "embeddings", "full_text"]
SourceKind = Literal["commitpackft", "commitpack", "agentpack", "synthetic", "local_repo"]

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
