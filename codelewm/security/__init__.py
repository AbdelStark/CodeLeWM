"""Security and licensing contracts for CodeLeWM."""

from __future__ import annotations

from .checkpoint_trust import (
    CheckpointTrustError,
    default_checkpoint_manifest_path,
    require_trusted_checkpoint,
)
from .claim_boundaries import (
    ClaimBoundaryError,
    available_claim_boundaries,
    claim_boundary_fingerprint,
    load_claim_boundary,
)
from .license_policy import (
    DEFAULT_PUBLIC_LICENSE_POLICY,
    LicenseDecision,
    LicenseGateError,
    PUBLIC_LICENSE_GATE_SCHEMA_VERSION,
    PublicLicenseGateReport,
    SourceLicensePolicy,
    build_public_license_gate_report,
    decide_license,
    enforce_public_license_gate,
    normalize_license,
    validate_public_license_gate_report,
)
from .non_execution import (
    FORBIDDEN_EXECUTION_CONFIG_KEYS,
    NonExecutionPolicyError,
    parse_python_source_text,
    reject_code_execution_config,
)
from .secret_scan import (
    SECRET_SCAN_REPORT_SCHEMA_VERSION,
    SecretFinding,
    SecretScanError,
    SecretScanReport,
    scan_file,
    scan_paths,
    scan_text,
    secret_scan_report_json_schema,
    validate_secret_scan_report_payload,
)

__all__ = [
    "CheckpointTrustError",
    "ClaimBoundaryError",
    "DEFAULT_PUBLIC_LICENSE_POLICY",
    "FORBIDDEN_EXECUTION_CONFIG_KEYS",
    "LicenseDecision",
    "LicenseGateError",
    "NonExecutionPolicyError",
    "PUBLIC_LICENSE_GATE_SCHEMA_VERSION",
    "PublicLicenseGateReport",
    "SECRET_SCAN_REPORT_SCHEMA_VERSION",
    "SecretFinding",
    "SecretScanError",
    "SecretScanReport",
    "SourceLicensePolicy",
    "available_claim_boundaries",
    "build_public_license_gate_report",
    "claim_boundary_fingerprint",
    "decide_license",
    "default_checkpoint_manifest_path",
    "enforce_public_license_gate",
    "load_claim_boundary",
    "normalize_license",
    "parse_python_source_text",
    "reject_code_execution_config",
    "require_trusted_checkpoint",
    "scan_file",
    "scan_paths",
    "scan_text",
    "secret_scan_report_json_schema",
    "validate_public_license_gate_report",
    "validate_secret_scan_report_payload",
]
