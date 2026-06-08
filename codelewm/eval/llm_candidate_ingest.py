"""Ingest ``codelewm.llm_candidate_pack.v1`` artifacts into the hard benchmark.

This is the consumer side of the OpenRouter/BYOK candidate-pack contract
(RFC-0016 #421). It reads a manifest-backed LLM candidate pack, verifies its
checksums, re-scans every patch for secrets, and maps each candidate onto a
benchmark candidate spec carrying provider/model/SDK provenance, checksum,
parser/apply status, redaction status, and source-manifest lineage.

The module is import-light: it never imports the sandbox, the scorer, or the
OpenRouter SDK. LLM candidate code is never executed here; the full patch text
is read from the pack's ``candidates/<id>.patch`` files (the JSON only carries a
redacted preview) and is applied text-only later, on the scoring path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codelewm.observability import read_artifact_manifest, validate_artifact_checksums
from codelewm.security.secret_scan import scan_text

from .downstream_anti_saturation import DownstreamAntiSaturationError, validate_hard_negative_class


# Mirrors codelewm.harness.openrouter_adapter.LLM_CANDIDATE_PACK_SCHEMA_VERSION.
# Re-declared locally so ingestion does not import the (torch-heavy) adapter.
LLM_CANDIDATE_PACK_SCHEMA_VERSION = "codelewm.llm_candidate_pack.v1"
LLM_CANDIDATE_INGEST_REPORT_SCHEMA_VERSION = "codelewm.llm_candidate_ingest_report.v1"
LLM_CANDIDATE_HARD_NEGATIVE_CLASS = "llm_generated"

_HEADER_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:a/|b/)?(\S+)", re.MULTILINE)


class LLMCandidateIngestError(ValueError):
    """Raised on programmer error; pack-content problems become typed blockers."""


@dataclass(frozen=True)
class IngestedLLMCandidate:
    """One LLM candidate ready to be materialized into a benchmark task."""

    candidate_id: str
    patch_text: str
    label: str  # always "unknown": LLM candidates require sandbox labeling to verify
    source: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMCandidateIngestResult:
    """Outcome of ingesting one candidate pack: usable candidates + blockers."""

    source_manifest_id: str | None
    source_manifest_path: str
    candidates: tuple[IngestedLLMCandidate, ...] = ()
    blockers: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_manifest_id": self.source_manifest_id,
            "source_manifest_path": self.source_manifest_path,
            "candidate_count": len(self.candidates),
            "blocker_count": len(self.blockers),
            "blockers": [dict(blocker) for blocker in self.blockers],
        }


def read_llm_candidate_pack(manifest_path: Path | str) -> tuple[Mapping[str, Any], str, Path]:
    """Read and verify a candidate-pack artifact.

    Returns ``(candidate_pack_payload, artifact_id, pack_dir)``. Raises
    :class:`LLMCandidateIngestError` when the manifest is missing, has the wrong
    kind, fails checksum verification, or the pack JSON is malformed.
    """

    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise LLMCandidateIngestError(f"candidate pack manifest does not exist: {manifest_path}")
    try:
        manifest = read_artifact_manifest(manifest_file)
    except Exception as exc:  # noqa: BLE001 - normalize to a typed ingest error
        raise LLMCandidateIngestError(f"candidate pack manifest is invalid: {exc}") from exc
    if manifest.artifact_kind != "candidate_pack":
        raise LLMCandidateIngestError(
            "candidate pack manifest must have artifact_kind='candidate_pack'"
        )
    pack_dir = manifest_file.parent
    validate_artifact_checksums(manifest, root=pack_dir)
    pack_file = pack_dir / "candidate_pack.json"
    if not pack_file.is_file():
        raise LLMCandidateIngestError("candidate pack is missing candidate_pack.json")
    payload = json.loads(pack_file.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LLMCandidateIngestError("candidate_pack.json must be a JSON object")
    if payload.get("schema_version") != LLM_CANDIDATE_PACK_SCHEMA_VERSION:
        raise LLMCandidateIngestError(
            "unsupported candidate pack schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    return payload, manifest.artifact_id, pack_dir


def _safe_patch_paths(patch_text: str) -> bool:
    for path in _HEADER_RE.findall(patch_text):
        if path == "/dev/null":
            continue
        candidate = path.strip()
        if candidate.startswith("/") or ".." in Path(candidate).parts:
            return False
    return True


def ingest_llm_candidate_pack(
    manifest_path: Path | str,
    *,
    base_llm_rank: int = 0,
    candidate_id_prefix: str = "llm",
) -> LLMCandidateIngestResult:
    """Ingest one candidate pack into benchmark candidate specs.

    Pack-level problems (missing/malformed/unscanned manifest) and per-candidate
    problems (secret-positive, redacted/oversized patch, unsafe patch paths,
    missing patch file) are recorded as typed blockers rather than raised, so a
    referencing benchmark build can surface them without crashing. Secret-positive
    candidates are never materialized.
    """

    manifest_str = str(manifest_path)
    try:
        payload, artifact_id, pack_dir = read_llm_candidate_pack(manifest_path)
    except LLMCandidateIngestError as exc:
        return LLMCandidateIngestResult(
            source_manifest_id=None,
            source_manifest_path=manifest_str,
            candidates=(),
            blockers=({"reason": "pack_unreadable", "detail": str(exc)},),
        )

    generator = payload.get("generator", {})
    provider = generator.get("provider")
    model = generator.get("model")
    sdk = generator.get("sdk")
    sdk_version = generator.get("sdk_version")

    candidates: list[IngestedLLMCandidate] = []
    blockers: list[Mapping[str, Any]] = []
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, Sequence):
        return LLMCandidateIngestResult(
            source_manifest_id=artifact_id,
            source_manifest_path=manifest_str,
            candidates=(),
            blockers=({"reason": "candidates_not_a_list", "detail": "candidate_pack.json"},),
        )

    for offset, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            blockers.append({"reason": "candidate_not_object", "index": offset})
            continue
        source_candidate_id = str(raw.get("candidate_id", f"candidate_{offset:03d}"))
        ingest_id = f"{candidate_id_prefix}_{source_candidate_id}"
        patch_path = raw.get("patch_path")
        if not isinstance(patch_path, str) or not patch_path:
            blockers.append({"reason": "missing_patch_path", "candidate_id": source_candidate_id})
            continue
        if not patch_path.endswith(".patch"):
            blockers.append(
                {"reason": "patch_redacted_or_oversized", "candidate_id": source_candidate_id}
            )
            continue
        patch_file = (pack_dir / patch_path).resolve()
        try:
            patch_file.relative_to(pack_dir.resolve())
        except ValueError:
            blockers.append({"reason": "patch_path_escapes_pack", "candidate_id": source_candidate_id})
            continue
        if not patch_file.is_file():
            blockers.append({"reason": "patch_file_missing", "candidate_id": source_candidate_id})
            continue
        patch_text = patch_file.read_text(encoding="utf-8")

        redaction = raw.get("redaction", {})
        if not isinstance(redaction, Mapping) or "secret_scan_ok" not in redaction:
            blockers.append({"reason": "unscanned_candidate", "candidate_id": source_candidate_id})
            continue
        if not redaction.get("secret_scan_ok", False):
            blockers.append({"reason": "secret_positive_in_pack", "candidate_id": source_candidate_id})
            continue
        # Defense in depth: re-scan the full patch we are about to materialize.
        if scan_text(patch_text, path=f"{source_candidate_id}.patch"):
            blockers.append({"reason": "secret_positive_on_rescan", "candidate_id": source_candidate_id})
            continue
        if not _safe_patch_paths(patch_text):
            blockers.append({"reason": "unsafe_patch_path", "candidate_id": source_candidate_id})
            continue

        candidates.append(
            IngestedLLMCandidate(
                candidate_id=ingest_id,
                patch_text=patch_text,
                label="unknown",
                source={
                    "hard_negative_class": LLM_CANDIDATE_HARD_NEGATIVE_CLASS,
                    "candidate_kind": LLM_CANDIDATE_HARD_NEGATIVE_CLASS,
                    "origin": "llm",
                    "provider": provider,
                    "model_slug": model,
                    "sdk": sdk,
                    "sdk_version": sdk_version,
                    "checksum": raw.get("content_sha256"),
                },
                provenance={
                    "generated": True,
                    "source_candidate_id": source_candidate_id,
                    "source_manifest_id": artifact_id,
                    "parser_status": raw.get("parser_status"),
                    "dry_run_patch_status": raw.get("dry_run_patch_status"),
                    "normalized_patch_sha256": raw.get("normalized_patch_sha256"),
                    "secret_scan_ok": True,
                    "llm_rank": base_llm_rank + offset + 1,
                },
            )
        )

    return LLMCandidateIngestResult(
        source_manifest_id=artifact_id,
        source_manifest_path=manifest_str,
        candidates=tuple(candidates),
        blockers=tuple(blockers),
    )


def build_llm_candidate_ingest_report(
    results: Sequence[LLMCandidateIngestResult],
) -> dict[str, Any]:
    """Aggregate ingestion results into a typed report."""

    total_candidates = sum(len(result.candidates) for result in results)
    total_blockers = sum(len(result.blockers) for result in results)
    return {
        "schema_version": LLM_CANDIDATE_INGEST_REPORT_SCHEMA_VERSION,
        "pack_count": len(results),
        "ingested_candidate_count": total_candidates,
        "blocker_count": total_blockers,
        "ok": total_blockers == 0,
        "source_manifest_ids": [
            result.source_manifest_id for result in results if result.source_manifest_id
        ],
        "packs": [result.to_dict() for result in results],
    }


def _validate_llm_class() -> None:
    # Guard: the LLM hard-negative class must remain part of the RFC-0016 set.
    try:
        validate_hard_negative_class(LLM_CANDIDATE_HARD_NEGATIVE_CLASS)
    except DownstreamAntiSaturationError as exc:  # pragma: no cover - defensive
        raise LLMCandidateIngestError(str(exc)) from exc


_validate_llm_class()
