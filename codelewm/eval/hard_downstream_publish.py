"""Publish the hard downstream benchmark artifact set and claim audit (RFC-0016 #423).

Given a built anti-saturation benchmark pack and a hard-mode rerank evaluation,
this assembles a self-contained, manifest-backed, secret-scanned publication
artifact: the source/license, split-leakage, anti-saturation, label-construction,
LLM-ingest, benchmark, rerank, and claim-gate reports, plus an artifact index
with checksums and a claim audit. The claim audit's public wording follows the
RFC-0016 claim gate exactly: it never asserts a broad coding-improvement claim
unless the gate opened.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.observability.manifest import sha256_file
from codelewm.security.secret_scan import scan_paths

from .downstream_anti_saturation import DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION
from .downstream_rerank import read_downstream_rerank_report


HARD_DOWNSTREAM_PUBLICATION_SCHEMA_VERSION = "codelewm.hard_downstream_publication.v1"
HARD_DOWNSTREAM_ARTIFACT_INDEX_SCHEMA_VERSION = "codelewm.hard_downstream_artifact_index.v1"
HARD_DOWNSTREAM_CLAIM_AUDIT_SCHEMA_VERSION = "codelewm.hard_downstream_claim_audit.v1"

# Exact RFC-0016 wording. The diagnostic fallback is used whenever the claim
# gate is closed; the positive wording only when the gate opens.
DIAGNOSTIC_FALLBACK_WORDING = (
    "The hard downstream benchmark executed and identified which baselines or "
    "slices block a positive claim; CodeLeWM does not yet support a broad "
    "downstream coding-usefulness claim."
)
POSITIVE_CLAIM_WORDING = (
    "On the locked anti-saturation test split, CodeLeWM downstream reranking "
    "beats no-action, lexical, and LLM-order baselines on pass@1 and MRR with "
    "bootstrap lift confidence intervals excluding zero."
)

_PUBLISH_SCAN_SUFFIXES = (".json", ".jsonl", ".md", ".txt", ".html", ".log")
# Pack manifest metadata keys that point at a publishable report file.
_PACK_REPORT_METADATA_KEYS = (
    "source_license_policy",
    "split_leakage_report",
    "secret_scan_report",
    "anti_saturation_report",
    "label_construction_report",
    "llm_candidate_ingest_report",
)


class HardDownstreamPublishError(ValueError):
    """Raised when the hard downstream artifact set cannot be assembled."""


@dataclass(frozen=True)
class HardDownstreamPublicationResult:
    artifact_manifest_id: str
    artifact_manifest_path: str
    claim_audit_path: str
    artifact_index_path: str
    claim_allowed: bool
    anti_saturation_eligible: bool
    broad_coding_improvement_claim_allowed: bool
    schema_version: str = HARD_DOWNSTREAM_PUBLICATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "claim_audit_path": self.claim_audit_path,
            "artifact_index_path": self.artifact_index_path,
            "claim_allowed": self.claim_allowed,
            "anti_saturation_eligible": self.anti_saturation_eligible,
            "broad_coding_improvement_claim_allowed": self.broad_coding_improvement_claim_allowed,
        }


def build_hard_downstream_claim_audit(
    *,
    anti_saturation_report: Mapping[str, Any],
    rerank_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the claim audit recording eligibility, missing baselines, CIs, and wording."""

    claim_gate = rerank_report.get("claim_gate", {})
    metrics = rerank_report.get("metrics", {})
    benchmark_id = str(rerank_report.get("benchmark_id", "unknown"))
    eligible = bool(anti_saturation_report.get("eligible"))
    allowed = bool(claim_gate.get("allowed"))

    missing_baselines = sorted(
        name
        for name, values in metrics.items()
        if isinstance(values, Mapping) and values.get("status") in {"blocked", "not_recorded"}
    )
    slice_record = {
        "benchmark_id": benchmark_id,
        "problem_count": anti_saturation_report.get("problem_count"),
        "eligible": eligible,
        "blocked_reasons": list(anti_saturation_report.get("blocked_reasons", [])),
        "baseline_pass_at_1": anti_saturation_report.get("baseline_pass_at_1"),
    }
    return {
        "schema_version": HARD_DOWNSTREAM_CLAIM_AUDIT_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "claim_allowed": allowed,
        "anti_saturation_eligible": eligible,
        "broad_coding_improvement_claim_allowed": allowed,
        "eligible_slices": [benchmark_id] if eligible else [],
        "saturated_slices": [] if eligible else [slice_record],
        "headline_slice": slice_record,
        "missing_baselines": missing_baselines,
        "claim_gate_failure_reasons": list(claim_gate.get("failure_reasons", [])),
        "checked_baselines": list(claim_gate.get("checked_baselines", [])),
        "lift_confidence_intervals": rerank_report.get("lift_confidence_intervals"),
        "public_wording": POSITIVE_CLAIM_WORDING if allowed else DIAGNOSTIC_FALLBACK_WORDING,
    }


def _verify_manifest(path: Path) -> tuple[Any, Path]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise HardDownstreamPublishError(f"manifest does not exist: {path}")
    manifest = read_artifact_manifest(manifest_path)
    validate_artifact_checksums(manifest, root=manifest_path.parent)
    return manifest, manifest_path.parent


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def assemble_hard_downstream_artifact_set(
    *,
    pack_manifest: Path | str,
    rerank_manifest: Path | str,
    out: Path | str,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "hard-downstream-publish"),
) -> HardDownstreamPublicationResult:
    """Assemble + verify + secret-scan the publishable hard downstream artifact set."""

    pack_manifest_obj, pack_dir = _verify_manifest(Path(pack_manifest))
    rerank_manifest_obj, rerank_dir = _verify_manifest(Path(rerank_manifest))
    if pack_manifest_obj.artifact_kind != "downstream_benchmark":
        raise HardDownstreamPublishError(
            "pack_manifest must have artifact_kind='downstream_benchmark'"
        )
    if rerank_manifest_obj.artifact_kind != "eval_report":
        raise HardDownstreamPublishError("rerank_manifest must have artifact_kind='eval_report'")

    output_dir = Path(out).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise HardDownstreamPublishError(
                f"output already exists; pass overwrite=True to replace: {output_dir}"
            )
        shutil.rmtree(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    published_files: list[Path] = []

    # Copy the upstream manifests for lineage.
    for label, manifest_path in (
        ("benchmark_manifest.json", Path(pack_manifest)),
        ("rerank_manifest.json", Path(rerank_manifest)),
    ):
        dest = output_dir / label
        shutil.copyfile(manifest_path, dest)
        published_files.append(dest)

    # Copy the pack reports recorded in the pack manifest metadata.
    for key in _PACK_REPORT_METADATA_KEYS:
        rel = pack_manifest_obj.metadata.get(key)
        if not isinstance(rel, str) or not rel:
            continue
        source = pack_dir / rel
        if not source.is_file():
            raise HardDownstreamPublishError(f"pack report missing: {rel}")
        dest = reports_dir / Path(rel).name
        shutil.copyfile(source, dest)
        published_files.append(dest)

    # Copy the benchmark payload + rerank report.
    benchmark_source = pack_dir / "benchmark.json"
    if benchmark_source.is_file():
        dest = output_dir / "benchmark.json"
        shutil.copyfile(benchmark_source, dest)
        published_files.append(dest)

    rerank_report_rel = _rerank_report_relpath(rerank_manifest_obj)
    rerank_report_source = rerank_dir / rerank_report_rel
    rerank_report = read_downstream_rerank_report(rerank_report_source)
    rerank_report_dest = reports_dir / "downstream_rerank_report.json"
    shutil.copyfile(rerank_report_source, rerank_report_dest)
    published_files.append(rerank_report_dest)

    anti_saturation_report = _resolve_anti_saturation_report(rerank_report, pack_dir, pack_manifest_obj)

    # Claim audit.
    claim_audit = build_hard_downstream_claim_audit(
        anti_saturation_report=anti_saturation_report,
        rerank_report=rerank_report,
    )
    claim_audit_path = output_dir / "claim_audit.json"
    _write_json(claim_audit_path, claim_audit)
    published_files.append(claim_audit_path)

    # Artifact index over everything written so far + a secret scan over the set.
    artifact_index_path = output_dir / "artifact_index.json"
    index_payload = _build_artifact_index(
        output_dir,
        published_files,
        pack_artifact_id=pack_manifest_obj.artifact_id,
        rerank_artifact_id=rerank_manifest_obj.artifact_id,
    )
    _write_json(artifact_index_path, index_payload)
    published_files.append(artifact_index_path)

    scan_report = scan_paths([output_dir], include_suffixes=_PUBLISH_SCAN_SUFFIXES)
    secret_scan_path = reports_dir / "publication_secret_scan_report.json"
    _write_json(secret_scan_path, _relativize_secret_scan(scan_report.to_dict(), output_dir))
    published_files.append(secret_scan_path)
    if not scan_report.ok:
        raise HardDownstreamPublishError(
            "secret scan found publish-blocking findings in the hard downstream artifact set"
        )

    manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=output_dir,
        files=published_files,
        command=command,
        config={"pack_manifest": str(pack_manifest), "rerank_manifest": str(rerank_manifest)},
        parent_artifacts=(pack_manifest_obj.artifact_id, rerank_manifest_obj.artifact_id),
        metadata={
            "schema_version": HARD_DOWNSTREAM_PUBLICATION_SCHEMA_VERSION,
            "benchmark_id": claim_audit["benchmark_id"],
            "claim_allowed": claim_audit["claim_allowed"],
            "anti_saturation_eligible": claim_audit["anti_saturation_eligible"],
            "broad_coding_improvement_claim_allowed": claim_audit[
                "broad_coding_improvement_claim_allowed"
            ],
        },
    )
    manifest_path = output_dir / "manifest.json"
    write_artifact_manifest(manifest, manifest_path)

    return HardDownstreamPublicationResult(
        artifact_manifest_id=manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        claim_audit_path="claim_audit.json",
        artifact_index_path="artifact_index.json",
        claim_allowed=bool(claim_audit["claim_allowed"]),
        anti_saturation_eligible=bool(claim_audit["anti_saturation_eligible"]),
        broad_coding_improvement_claim_allowed=bool(
            claim_audit["broad_coding_improvement_claim_allowed"]
        ),
    )


def read_hard_downstream_claim_audit(path: Path | str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise HardDownstreamPublishError("claim audit must be a JSON object")
    if payload.get("schema_version") != HARD_DOWNSTREAM_CLAIM_AUDIT_SCHEMA_VERSION:
        raise HardDownstreamPublishError("unsupported claim audit schema_version")
    return payload


def _rerank_report_relpath(rerank_manifest_obj: Any) -> str:
    for file in rerank_manifest_obj.files:
        if file.path.endswith("downstream_rerank_report.json"):
            return file.path
    raise HardDownstreamPublishError("rerank manifest does not list downstream_rerank_report.json")


def _resolve_anti_saturation_report(
    rerank_report: Mapping[str, Any],
    pack_dir: Path,
    pack_manifest_obj: Any,
) -> Mapping[str, Any]:
    embedded = rerank_report.get("anti_saturation_report")
    if isinstance(embedded, Mapping):
        return embedded
    rel = pack_manifest_obj.metadata.get("anti_saturation_report")
    if isinstance(rel, str) and rel:
        payload = json.loads((pack_dir / rel).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            return payload
    raise HardDownstreamPublishError(
        "no anti-saturation report found; run the rerank evaluation in --hard-mode "
        f"or build the pack with the {DOWNSTREAM_ANTI_SATURATION_REPORT_SCHEMA_VERSION} profile"
    )


def _build_artifact_index(
    output_dir: Path,
    files: Sequence[Path],
    *,
    pack_artifact_id: str,
    rerank_artifact_id: str,
) -> dict[str, Any]:
    entries = []
    for file in sorted(set(files)):
        entries.append(
            {
                "path": file.resolve().relative_to(output_dir).as_posix(),
                "sha256": sha256_file(file),
                "bytes": file.stat().st_size,
            }
        )
    return {
        "schema_version": HARD_DOWNSTREAM_ARTIFACT_INDEX_SCHEMA_VERSION,
        "parent_artifacts": [pack_artifact_id, rerank_artifact_id],
        "file_count": len(entries),
        "files": entries,
    }


def _relativize_secret_scan(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    def rel(path_value: Any) -> str:
        try:
            return Path(str(path_value)).resolve().relative_to(root).as_posix()
        except ValueError:
            return str(path_value)

    return {
        "schema_version": payload["schema_version"],
        "ok": payload["ok"],
        "paths_scanned": [rel(path) for path in payload.get("paths_scanned", [])],
        "findings": [
            {**dict(finding), "path": rel(finding.get("path"))}
            for finding in payload.get("findings", [])
        ],
    }
