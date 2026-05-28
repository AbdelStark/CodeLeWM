"""Pre-publish gate and dataset-card rendering for the execution pack.

The publish flow uses the existing ``scripts/hf-publish-codelewm-artifacts``
pattern: dry-run by default, requires ``HF_TOKEN`` to upload, refuses to
upload unless the pre-publish gate passes.

This module implements the gate logic and the card rendering so they
are testable without network access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codelewm.security.claim_boundaries import (
    claim_boundary_fingerprint,
    load_claim_boundary,
)

from .manifest import (
    EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
    sha256_file,
)


CLAIM_BOUNDARY_NAME = "execution_substrate.v1"
PERMISSIVE_LICENSES = frozenset(
    {"MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0"}
)


class PrePublishGateError(RuntimeError):
    """Raised when the pre-publish gate refuses to release a pack."""


@dataclass(frozen=True)
class PrePublishReport:
    """Structured report of every gate check the pack passed or failed."""

    pack_dir: str
    manifest_path: str
    checks: dict[str, bool]
    findings: tuple[str, ...]
    record_count: int
    held_out_excluded_count: int
    license_breakdown: dict[str, int]
    attribution_urls: dict[str, str]

    @property
    def allowed(self) -> bool:
        return all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "pack_dir": self.pack_dir,
            "manifest_path": self.manifest_path,
            "allowed": self.allowed,
            "checks": dict(self.checks),
            "findings": list(self.findings),
            "record_count": self.record_count,
            "held_out_excluded_count": self.held_out_excluded_count,
            "license_breakdown": dict(self.license_breakdown),
            "attribution_urls": dict(self.attribution_urls),
        }


@dataclass(frozen=True)
class DatasetCardContext:
    """Inputs used to render the dataset card from the manifest."""

    pack_id: str
    revision: str
    repo_id: str
    record_count: int
    split_counts: dict[str, int]
    output_type_distribution: dict[str, int]
    output_kind_distribution: dict[str, int]
    execution_status_distribution: dict[str, int]
    source_breakdown: dict[str, int]
    license_breakdown: dict[str, int]
    held_out_excluded_count: int
    sandbox_policy: dict[str, Any]
    sandbox_reject_counts: dict[str, int]
    parent_artifacts: tuple[dict[str, str], ...]
    pack_jsonl_checksum: str
    claim_boundary_fingerprint: str
    attribution_urls: dict[str, str] = field(default_factory=dict)


def run_pre_publish_gate(pack_dir: Path) -> PrePublishReport:
    """Verify every contract a release-bound execution pack must satisfy."""

    pack_dir = Path(pack_dir)
    manifest_path = pack_dir / "manifest.json"
    pack_jsonl = pack_dir / "pack.jsonl"
    attribution_path = pack_dir / "attribution.json"
    audit_path = pack_dir / "sandbox_audit_summary.json"
    boundary_path = pack_dir / "claim_boundary.md"

    findings: list[str] = []
    checks: dict[str, bool] = {}

    if not manifest_path.is_file():
        raise PrePublishGateError(f"manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    checks["manifest_schema_version_supported"] = (
        manifest.get("schema_version") == EXECUTION_PACK_MANIFEST_SCHEMA_VERSION
    )
    if not checks["manifest_schema_version_supported"]:
        findings.append(
            f"manifest schema_version mismatch: expected {EXECUTION_PACK_MANIFEST_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )

    checks["pack_jsonl_present"] = pack_jsonl.is_file()
    if not checks["pack_jsonl_present"]:
        findings.append(f"pack.jsonl is missing at {pack_jsonl}")

    checks["attribution_json_present"] = attribution_path.is_file()
    if not checks["attribution_json_present"]:
        findings.append("attribution.json is missing")

    checks["sandbox_audit_summary_present"] = audit_path.is_file()
    if not checks["sandbox_audit_summary_present"]:
        findings.append("sandbox_audit_summary.json is missing")

    checks["claim_boundary_embedded"] = boundary_path.is_file()
    if not checks["claim_boundary_embedded"]:
        findings.append("claim_boundary.md is missing")
    elif boundary_path.read_text(encoding="utf-8") != load_claim_boundary(
        CLAIM_BOUNDARY_NAME
    ):
        checks["claim_boundary_embedded"] = False
        findings.append("claim_boundary.md does not match the registered text")

    if pack_jsonl.is_file():
        actual_checksum = sha256_file(pack_jsonl)
        checks["pack_jsonl_checksum_matches"] = (
            actual_checksum == manifest.get("pack_jsonl_checksum")
        )
        if not checks["pack_jsonl_checksum_matches"]:
            findings.append(
                f"pack.jsonl checksum mismatch: file={actual_checksum} manifest={manifest.get('pack_jsonl_checksum')}"
            )
    else:
        checks["pack_jsonl_checksum_matches"] = False

    cb = manifest.get("claim_boundary") or {}
    checks["claim_boundary_fingerprint_matches"] = cb.get(
        "fingerprint"
    ) == claim_boundary_fingerprint(CLAIM_BOUNDARY_NAME)
    if not checks["claim_boundary_fingerprint_matches"]:
        findings.append(
            "claim_boundary fingerprint in manifest does not match the registered file"
        )

    license_breakdown = dict(manifest.get("license_breakdown") or {})
    non_permissive = [
        lic for lic in license_breakdown if lic not in PERMISSIVE_LICENSES
    ]
    checks["all_licenses_permissive"] = not non_permissive
    if non_permissive:
        findings.append(
            f"non-permissive licenses found: {non_permissive}; "
            "add to PERMISSIVE_LICENSES or filter at ingestion"
        )

    attribution_urls: dict[str, str] = {}
    if attribution_path.is_file():
        attribution_urls = json.loads(
            attribution_path.read_text(encoding="utf-8")
        )
    sources_in_pack = set((manifest.get("source_breakdown") or {}).keys())
    missing_attribution = sources_in_pack - set(attribution_urls.keys())
    checks["every_source_has_attribution"] = not missing_attribution
    if missing_attribution:
        findings.append(
            f"sources missing attribution URL: {sorted(missing_attribution)}"
        )

    checks["record_count_nonzero"] = int(manifest.get("record_count", 0)) > 0
    if not checks["record_count_nonzero"]:
        findings.append("record_count is zero; refusing to publish an empty pack")

    return PrePublishReport(
        pack_dir=str(pack_dir),
        manifest_path=str(manifest_path),
        checks=checks,
        findings=tuple(findings),
        record_count=int(manifest.get("record_count", 0)),
        held_out_excluded_count=int(manifest.get("held_out_eval_excluded_count", 0)),
        license_breakdown=license_breakdown,
        attribution_urls=attribution_urls,
    )


def render_dataset_card(*, context: DatasetCardContext) -> str:
    """Render the dataset card markdown from the manifest context."""

    parts: list[str] = []
    parts.append(f"# {context.repo_id}\n")
    parts.append(
        f"- Pack id: `{context.pack_id}`\n"
        f"- Revision: `{context.revision}`\n"
        f"- Schema: `{EXECUTION_PACK_MANIFEST_SCHEMA_VERSION}`\n"
        f"- Records: {context.record_count}\n"
        f"- Held-out (MBPP-Plus/HumanEval) records excluded at ingestion: "
        f"{context.held_out_excluded_count}\n"
        f"- pack.jsonl SHA-256: `{context.pack_jsonl_checksum}`\n"
        f"- Claim boundary SHA-256: `{context.claim_boundary_fingerprint}`\n"
    )

    parts.append("\n## Summary\n")
    parts.append(
        "This dataset is the execution-substrate pack for CodeLeWM "
        "(RFC-0014, tracker #259). Each line of `pack.jsonl` carries a "
        "tokenized `(code, input, output)` triple captured by running a "
        "licensed public Python submission in a sandboxed deterministic "
        "executor. The dataset is published as research evidence; see the "
        "claim boundary for what claims it does and does not support.\n"
    )

    parts.append("\n## Provenance\n")
    parts.append("| Source | Records |\n| --- | ---: |\n")
    for source, count in sorted(context.source_breakdown.items()):
        parts.append(f"| `{source}` | {count} |\n")

    parts.append("\n## License Summary\n")
    parts.append("| License | Records |\n| --- | ---: |\n")
    for lic, count in sorted(context.license_breakdown.items()):
        parts.append(f"| `{lic}` | {count} |\n")

    parts.append("\n## Attribution\n")
    if context.attribution_urls:
        for source, url in sorted(context.attribution_urls.items()):
            parts.append(f"- `{source}`: {url}\n")
    else:
        parts.append("- (no upstream attribution recorded)\n")

    parts.append("\n## Sandbox Policy\n")
    parts.append("```json\n")
    parts.append(json.dumps(context.sandbox_policy, indent=2, sort_keys=True))
    parts.append("\n```\n")

    parts.append("\n## Determinism And Reject Counts\n")
    parts.append("| Reject reason | Count |\n| --- | ---: |\n")
    for reason, count in sorted(context.sandbox_reject_counts.items()):
        parts.append(f"| `{reason}` | {count} |\n")
    if not context.sandbox_reject_counts:
        parts.append("| (no rejects recorded) | 0 |\n")

    parts.append("\n## Split Policy\n")
    parts.append("Records are partitioned by `source_problem_id`; no problem leaks across splits.\n\n")
    parts.append("| Split | Records |\n| --- | ---: |\n")
    for split, count in sorted(context.split_counts.items()):
        parts.append(f"| `{split}` | {count} |\n")

    parts.append("\n## Output Distribution\n")
    parts.append("Per `output_type`:\n\n")
    parts.append("| output_type | Records |\n| --- | ---: |\n")
    for kind, count in sorted(context.output_type_distribution.items()):
        parts.append(f"| `{kind}` | {count} |\n")

    parts.append("\nPer `output_kind`:\n\n")
    parts.append("| output_kind | Records |\n| --- | ---: |\n")
    for kind, count in sorted(context.output_kind_distribution.items()):
        parts.append(f"| `{kind}` | {count} |\n")

    parts.append("\nPer `execution_status`:\n\n")
    parts.append("| execution_status | Records |\n| --- | ---: |\n")
    for status, count in sorted(context.execution_status_distribution.items()):
        parts.append(f"| `{status}` | {count} |\n")

    parts.append("\n## Parent Artifacts\n")
    if context.parent_artifacts:
        parts.append("| Path | SHA-256 |\n| --- | --- |\n")
        for parent in context.parent_artifacts:
            parts.append(
                f"| `{parent.get('path', '')}` | `{parent.get('sha256', '')}` |\n"
            )
    else:
        parts.append("- (no parent artifacts recorded)\n")

    parts.append("\n## Claim Boundary\n")
    parts.append("This pack is governed by the execution-substrate claim boundary "
                 "(`execution_substrate.v1`). The verbatim text is included in "
                 "`claim_boundary.md` and the SHA-256 is recorded above. ")
    parts.append("Refer to that file for what claims this pack supports and what "
                 "claims it forbids.\n")

    parts.append("\n## How To Verify\n")
    parts.append(
        "```bash\n"
        f"hf download {context.repo_id} --revision {context.revision} \\\n"
        "  --local-dir <download-dir>\n"
        "uv run codelewm manifest verify --manifest <download-dir>/manifest.json --json\n"
        "uv run codelewm secret-scan <download-dir> --json\n"
        "```\n"
    )

    return "".join(parts)


def context_from_manifest(
    *,
    manifest_path: Path,
    attribution_path: Path,
    repo_id: str,
    revision: str,
) -> DatasetCardContext:
    """Build a :class:`DatasetCardContext` from an on-disk manifest."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    attribution: dict[str, str] = {}
    if Path(attribution_path).is_file():
        attribution = json.loads(
            Path(attribution_path).read_text(encoding="utf-8")
        )
    return DatasetCardContext(
        pack_id=str(manifest.get("pack_id", "")),
        revision=revision,
        repo_id=repo_id,
        record_count=int(manifest.get("record_count", 0)),
        split_counts=dict(manifest.get("split_counts") or {}),
        output_type_distribution=dict(manifest.get("output_type_distribution") or {}),
        output_kind_distribution=dict(manifest.get("output_kind_distribution") or {}),
        execution_status_distribution=dict(
            manifest.get("execution_status_distribution") or {}
        ),
        source_breakdown=dict(manifest.get("source_breakdown") or {}),
        license_breakdown=dict(manifest.get("license_breakdown") or {}),
        held_out_excluded_count=int(manifest.get("held_out_eval_excluded_count", 0)),
        sandbox_policy=dict(manifest.get("sandbox_policy") or {}),
        sandbox_reject_counts=dict(manifest.get("sandbox_reject_counts") or {}),
        parent_artifacts=tuple(manifest.get("parent_artifacts") or ()),
        pack_jsonl_checksum=str(manifest.get("pack_jsonl_checksum", "")),
        claim_boundary_fingerprint=str(
            (manifest.get("claim_boundary") or {}).get("fingerprint", "")
        ),
        attribution_urls=attribution,
    )
