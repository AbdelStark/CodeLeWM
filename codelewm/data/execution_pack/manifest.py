"""Execution-pack manifest schema and helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


EXECUTION_PACK_MANIFEST_SCHEMA_VERSION = "codelewm.execution_pack_manifest.v1"
EXECUTION_PACK_RECORD_SCHEMA_VERSION = "codelewm.execution_pack_record.v1"


@dataclass(frozen=True)
class ExecutionPackManifest:
    """Manifest schema for one execution-pack artifact.

    The manifest is the contract between the pack builder and every
    downstream consumer (training executor, latent probes, manifest
    verify, scorer quality). It records the source ingestion artifacts,
    sandbox policy applied, per-stage reject counts, split counts, and
    the claim-boundary fingerprint that scopes any future claim about
    this pack.
    """

    schema_version: str
    created_at: str
    pack_id: str
    pack_dir: str
    record_count: int
    split_counts: dict[str, int]
    split_by: str
    output_type_distribution: dict[str, int]
    output_kind_distribution: dict[str, int]
    execution_status_distribution: dict[str, int]
    source_breakdown: dict[str, int]
    license_breakdown: dict[str, int]
    sandbox_policy: dict[str, Any]
    sandbox_reject_counts: dict[str, int]
    parent_artifacts: tuple[dict[str, str], ...]
    held_out_eval_excluded_count: int
    claim_boundary: dict[str, str]
    pack_jsonl_checksum: str
    tokenizer: dict[str, Any]
    seed: int
    train_frac: float
    val_frac: float
    max_inputs_per_problem: int | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Stable ordering for downstream consumers.
        for key in (
            "split_counts",
            "output_type_distribution",
            "output_kind_distribution",
            "execution_status_distribution",
            "source_breakdown",
            "license_breakdown",
            "sandbox_reject_counts",
        ):
            payload[key] = dict(sorted(payload[key].items()))
        return payload


def write_execution_pack_manifest(
    manifest: ExecutionPackManifest, path: Path
) -> None:
    """Write a manifest as pretty-printed JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
