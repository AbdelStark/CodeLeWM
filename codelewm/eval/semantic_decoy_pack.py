"""Manifest-backed semantic decoy packs for execution-surprise evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codelewm.data.execution_pack.manifest import (
    EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
    SUPPORTED_EXECUTION_PACK_RECORD_SCHEMA_VERSIONS,
)
from codelewm.observability import (
    ArtifactManifest,
    ArtifactManifestError,
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security.secret_scan import scan_paths

from .execution_surprise_decoys import (
    EXECUTION_SURPRISE_DECOY_SCHEMA_VERSION,
    DecoyGenerationReport,
    DecoyPair,
    generate_same_code_different_input_pairs,
    generate_same_problem_different_submission_pairs,
)


SEMANTIC_DECOY_PACK_RUN_SCHEMA_VERSION = "codelewm.eval.semantic_decoy_pack_run.v1"
SEMANTIC_DECOY_PACK_SCHEMA_VERSION = "codelewm.eval.semantic_decoy_pack.v1"
SEMANTIC_DECOY_PAIR_SCHEMA_VERSION = "codelewm.eval.semantic_decoy_pair.v1"
SEMANTIC_DECOY_SUMMARY_SCHEMA_VERSION = "codelewm.eval.semantic_decoy_summary.v1"
DEFAULT_SEMANTIC_DECOY_CATEGORY = "same_problem_different_submission"


class SemanticDecoyPackError(ValueError):
    """Raised when a semantic decoy pack cannot be built or loaded."""


@dataclass(frozen=True)
class SemanticDecoyPackResult:
    """CLI-facing summary for a written semantic decoy pack."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    pair_rows_path: str
    summary_path: str
    parent_artifacts: tuple[str, ...]
    pair_count: int
    distinct_problem_count: int
    claim_allowed: bool
    schema_version: str = SEMANTIC_DECOY_PACK_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "pair_rows_path": self.pair_rows_path,
            "summary_path": self.summary_path,
            "parent_artifacts": list(self.parent_artifacts),
            "pair_count": self.pair_count,
            "distinct_problem_count": self.distinct_problem_count,
            "claim_allowed": self.claim_allowed,
        }


@dataclass(frozen=True)
class LoadedSemanticDecoyPack:
    """Verified semantic decoy pack consumed by execution-surprise."""

    artifact_manifest: ArtifactManifest
    pair_rows_path: Path
    summary_path: Path
    pairs: tuple[DecoyPair, ...]
    summary: Mapping[str, Any]

    def generation_reports(
        self, categories: Sequence[str] | None = None
    ) -> tuple[DecoyGenerationReport, ...]:
        selected = set(categories or {pair.category for pair in self.pairs})
        by_category = Counter(pair.category for pair in self.pairs)
        report_payloads = self.summary.get("generator_reports", [])
        reports_by_category: dict[str, Mapping[str, Any]] = {}
        if isinstance(report_payloads, Sequence):
            for report in report_payloads:
                if isinstance(report, Mapping) and isinstance(report.get("category"), str):
                    reports_by_category[str(report["category"])] = report
        out: list[DecoyGenerationReport] = []
        for category in sorted(selected):
            report = reports_by_category.get(category, {})
            skipped = report.get("skipped_reasons", {}) if isinstance(report, Mapping) else {}
            out.append(
                DecoyGenerationReport(
                    schema_version=EXECUTION_SURPRISE_DECOY_SCHEMA_VERSION,
                    category=category,
                    pair_count=by_category.get(category, 0),
                    eligible_query_count=int(report.get("eligible_query_count", 0))
                    if isinstance(report.get("eligible_query_count", 0), int)
                    else 0,
                    skipped_reasons={
                        str(key): int(value)
                        for key, value in dict(skipped).items()
                        if isinstance(value, int)
                    }
                    if isinstance(skipped, Mapping)
                    else {},
                )
            )
        return tuple(out)

    def metadata(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_manifest.artifact_id,
            "manifest_path": str(self.pair_rows_path.parent / "manifest.json"),
            "pair_count": len(self.pairs),
            "summary": dict(self.summary),
        }


def build_semantic_decoy_pack(
    *,
    pack: Path | str,
    out: Path | str,
    splits: Sequence[str] = ("val", "test"),
    seed: int = 0,
    max_pairs_per_query: int = 3,
    min_pairs_for_claim: int = 100,
    min_distinct_problems_for_claim: int = 30,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "semantic-decoy-pack"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> SemanticDecoyPackResult:
    """Build a strengthened same-problem semantic decoy pack.

    The artifact contains record IDs and output fingerprints only. It does not
    serialize source code, input reprs, or raw outputs beyond the fingerprints
    already present in execution-pack metadata.
    """

    if max_pairs_per_query < 1:
        raise SemanticDecoyPackError("max_pairs_per_query must be >= 1")
    if min_pairs_for_claim < 1:
        raise SemanticDecoyPackError("min_pairs_for_claim must be >= 1")
    if min_distinct_problems_for_claim < 1:
        raise SemanticDecoyPackError(
            "min_distinct_problems_for_claim must be >= 1"
        )
    selected_splits = tuple(dict.fromkeys(str(split) for split in splits if str(split)))
    if not selected_splits:
        raise SemanticDecoyPackError("at least one split must be selected")

    paths = _resolve_pack_paths(pack)
    pack_artifact, execution_manifest = _read_verified_execution_pack(paths)
    records = _load_pack_records(paths.pack_jsonl_path)
    out_dir = Path(out).resolve()
    pair_rows_path = out_dir / "semantic_decoy_pairs.jsonl"
    summary_path = out_dir / "reports" / "semantic_decoy_summary.json"
    config_path = out_dir / "config.json"
    secret_scan_path = out_dir / "reports" / "secret_scan_report.json"
    manifest_path = out_dir / "manifest.json"
    _reject_existing(
        (pair_rows_path, summary_path, config_path, secret_scan_path, manifest_path),
        overwrite=overwrite,
        output_dir=out_dir,
    )

    selected, split_summary = _filter_records_for_splits(records, selected_splits)
    same_problem_pairs, same_problem_report = generate_same_problem_different_submission_pairs(
        selected,
        seed=seed,
        max_pairs_per_query=max_pairs_per_query,
        same_input_only=False,
    )
    same_code_pairs, same_code_report = generate_same_code_different_input_pairs(
        selected,
        seed=seed,
        max_pairs_per_query=max_pairs_per_query,
    )
    row_by_id = {str(row.get("record_id", "")): row for row in selected}
    generator_reports = (same_problem_report, same_code_report)
    generated_pairs = (*same_problem_pairs, *same_code_pairs)
    pair_rows, leak_summary = _pair_rows(generated_pairs, row_by_id=row_by_id)
    distinct_problem_count = len({str(row["source_problem_id"]) for row in pair_rows})
    claim_gate = _claim_gate(
        pair_count=len(pair_rows),
        distinct_problem_count=distinct_problem_count,
        min_pairs=min_pairs_for_claim,
        min_problems=min_distinct_problems_for_claim,
    )
    config_payload = {
        "schema_version": SEMANTIC_DECOY_PACK_RUN_SCHEMA_VERSION,
        "pack": str(paths.root),
        "out": str(out_dir),
        "splits": list(selected_splits),
        "seed": seed,
        "max_pairs_per_query": max_pairs_per_query,
        "min_pairs_for_claim": min_pairs_for_claim,
        "min_distinct_problems_for_claim": min_distinct_problems_for_claim,
    }
    summary_payload = {
        "schema_version": SEMANTIC_DECOY_SUMMARY_SCHEMA_VERSION,
        "pack_schema_version": SEMANTIC_DECOY_PACK_SCHEMA_VERSION,
        "pair_schema_version": SEMANTIC_DECOY_PAIR_SCHEMA_VERSION,
        "category": "semantic_same_problem",
        "categories": [report.category for report in generator_reports],
        "pair_count": len(pair_rows),
        "pair_count_by_category": dict(
            sorted(Counter(str(row["category"]) for row in pair_rows).items())
        ),
        "distinct_problem_count": distinct_problem_count,
        "eligible_query_count": sum(report.eligible_query_count for report in generator_reports),
        "benchmark_counts": dict(
            sorted(Counter(str(row.get("source_dataset", "unknown")) for row in selected).items())
        ),
        "record_schema_versions": dict(
            sorted(Counter(str(row.get("schema_version", "unknown")) for row in selected).items())
        ),
        "generator_reports": [report.as_dict() for report in generator_reports],
        "filtering_summary": {
            "split_policy": split_summary,
            "leak_checks": leak_summary,
        },
        "source_license_policy": {
            "source_breakdown": execution_manifest.get("source_breakdown", {}),
            "license_breakdown": execution_manifest.get("license_breakdown", {}),
            "derived_artifact_policy": (
                "metadata_only: record IDs, category labels, rationales, and "
                "output fingerprints; no raw source code or provider secrets"
            ),
        },
        "claim_gate": claim_gate,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(config_payload, config_path)
    _write_json(summary_payload, summary_path)
    _write_jsonl(pair_rows_path, pair_rows)
    scan = scan_paths(
        (pair_rows_path, summary_path, config_path),
        include_suffixes=(),
        recursive=False,
    )
    scan_payload = scan.to_dict()
    _write_json(scan_payload, secret_scan_path)
    if scan_payload.get("findings"):
        raise SemanticDecoyPackError(
            "semantic decoy pack contains secret-scan findings; refusing artifact"
        )

    parent_artifacts = (pack_artifact.artifact_id,)
    artifact_manifest = build_artifact_manifest(
        artifact_kind="downstream_benchmark",
        root=out_dir,
        files=(pair_rows_path, summary_path, config_path, secret_scan_path),
        command=command,
        config=config_payload,
        parent_artifacts=parent_artifacts,
        source_git_sha=source_git_sha,
        created_at=created_at,
        metadata={
            "schema_version": SEMANTIC_DECOY_PACK_SCHEMA_VERSION,
            "pair_schema_version": SEMANTIC_DECOY_PAIR_SCHEMA_VERSION,
            "summary_schema_version": SEMANTIC_DECOY_SUMMARY_SCHEMA_VERSION,
            "pair_rows_path": "semantic_decoy_pairs.jsonl",
            "summary_path": "reports/semantic_decoy_summary.json",
            "category": "semantic_same_problem",
            "categories": [report.category for report in generator_reports],
            "benchmark_counts": dict(
                sorted(Counter(str(row.get("source_dataset", "unknown")) for row in selected).items())
            ),
            "record_schema_versions": dict(
                sorted(Counter(str(row.get("schema_version", "unknown")) for row in selected).items())
            ),
            "pair_count": len(pair_rows),
            "pair_count_by_category": dict(
                sorted(Counter(str(row["category"]) for row in pair_rows).items())
            ),
            "distinct_problem_count": distinct_problem_count,
            "claim_allowed": bool(claim_gate["claim_allowed"]),
        },
    )
    write_artifact_manifest(artifact_manifest, manifest_path)
    return SemanticDecoyPackResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        pair_rows_path="semantic_decoy_pairs.jsonl",
        summary_path="reports/semantic_decoy_summary.json",
        parent_artifacts=parent_artifacts,
        pair_count=len(pair_rows),
        distinct_problem_count=distinct_problem_count,
        claim_allowed=bool(claim_gate["claim_allowed"]),
    )


def load_semantic_decoy_pack(manifest_path: Path | str) -> LoadedSemanticDecoyPack:
    """Load and checksum-verify a semantic decoy pack artifact."""

    path = Path(manifest_path)
    artifact = read_artifact_manifest(path)
    if artifact.artifact_kind != "downstream_benchmark":
        raise SemanticDecoyPackError(
            "semantic decoy pack manifest must have artifact_kind='downstream_benchmark'"
        )
    if artifact.metadata.get("schema_version") != SEMANTIC_DECOY_PACK_SCHEMA_VERSION:
        raise SemanticDecoyPackError(
            "semantic decoy pack manifest has unsupported schema_version"
        )
    root = path.parent
    validate_artifact_checksums(artifact, root=root)
    pair_rows_path = root / _metadata_path(artifact, "pair_rows_path")
    summary_path = root / _metadata_path(artifact, "summary_path")
    pairs = tuple(_decoy_pair_from_row(row) for row in _load_jsonl(pair_rows_path))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise SemanticDecoyPackError("semantic decoy summary must be a JSON object")
    return LoadedSemanticDecoyPack(
        artifact_manifest=artifact,
        pair_rows_path=pair_rows_path,
        summary_path=summary_path,
        pairs=pairs,
        summary=dict(summary),
    )


@dataclass(frozen=True)
class _PackPaths:
    root: Path
    pack_jsonl_path: Path
    execution_manifest_path: Path
    artifact_manifest_path: Path


def _resolve_pack_paths(value: Path | str) -> _PackPaths:
    raw = Path(value).resolve()
    if raw.is_file() and raw.name in {"pack.jsonl", "manifest.json", "artifact_manifest.json"}:
        root = raw.parent
    elif raw.is_dir():
        root = raw
    else:
        raise SemanticDecoyPackError(
            "--pack must be an execution pack directory, pack.jsonl, manifest.json, "
            "or artifact_manifest.json"
        )
    paths = _PackPaths(
        root=root,
        pack_jsonl_path=root / "pack.jsonl",
        execution_manifest_path=root / "manifest.json",
        artifact_manifest_path=root / "artifact_manifest.json",
    )
    for required in (
        paths.pack_jsonl_path,
        paths.execution_manifest_path,
        paths.artifact_manifest_path,
    ):
        if not required.is_file():
            raise SemanticDecoyPackError(f"execution pack file not found: {required}")
    return paths


def _read_verified_execution_pack(
    paths: _PackPaths,
) -> tuple[ArtifactManifest, Mapping[str, Any]]:
    try:
        execution_manifest = json.loads(
            paths.execution_manifest_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise SemanticDecoyPackError("execution pack manifest is invalid JSON") from exc
    if not isinstance(execution_manifest, Mapping):
        raise SemanticDecoyPackError("execution pack manifest must be a JSON object")
    if execution_manifest.get("schema_version") != EXECUTION_PACK_MANIFEST_SCHEMA_VERSION:
        raise SemanticDecoyPackError(
            "execution pack manifest schema_version is unsupported"
        )
    try:
        artifact = read_artifact_manifest(paths.artifact_manifest_path)
        validate_artifact_checksums(artifact, root=paths.root)
    except ArtifactManifestError as exc:
        raise SemanticDecoyPackError(str(exc)) from exc
    if artifact.artifact_kind != "dataset":
        raise SemanticDecoyPackError(
            "execution pack artifact manifest must have artifact_kind='dataset'"
        )
    pack_id = execution_manifest.get("pack_id")
    if isinstance(pack_id, str) and artifact.artifact_id != pack_id:
        raise SemanticDecoyPackError(
            f"execution pack artifact_id mismatch: {artifact.artifact_id!r} != {pack_id!r}"
        )
    return artifact, execution_manifest


def _load_pack_records(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SemanticDecoyPackError(
                    f"{path}:{line_no}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise SemanticDecoyPackError(f"{path}:{line_no}: row must be an object")
            if (
                row.get("schema_version")
                not in SUPPORTED_EXECUTION_PACK_RECORD_SCHEMA_VERSIONS
            ):
                raise SemanticDecoyPackError(
                    f"{path}:{line_no}: unsupported execution-pack record schema"
                )
            rows.append(row)
    if not rows:
        raise SemanticDecoyPackError("execution pack contains no records")
    return tuple(rows)


def _filter_records_for_splits(
    records: Sequence[Mapping[str, Any]], splits: Sequence[str]
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any]]:
    allowed = set(splits)
    selected = [record for record in records if str(record.get("split", "")) in allowed]
    split_counts = Counter(str(record.get("split", "")) for record in records)
    excluded_counts = Counter(
        str(record.get("split", ""))
        for record in records
        if str(record.get("split", "")) not in allowed
    )
    if not selected:
        raise SemanticDecoyPackError(
            f"execution pack has no records in selected splits: {', '.join(splits)}"
        )
    return tuple(selected), {
        "split_by": "source_problem_id",
        "selected_splits": list(splits),
        "input_split_counts": dict(sorted(split_counts.items())),
        "excluded_split_counts": dict(sorted(excluded_counts.items())),
        "train_rows_excluded": int(excluded_counts.get("train", 0)),
        "selected_record_count": len(selected),
    }


def _pair_rows(
    pairs: Sequence[DecoyPair], *, row_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        query = row_by_id.get(pair.query_record_id)
        decoy = row_by_id.get(pair.decoy_record_id)
        if query is None or decoy is None:
            skipped["missing_record"] += 1
            continue
        if query.get("split") == "train" or decoy.get("split") == "train":
            skipped["train_split_leak"] += 1
            continue
        if query.get("source_dataset") != decoy.get("source_dataset"):
            skipped["source_dataset_mismatch"] += 1
            continue
        if query.get("source_problem_id") != decoy.get("source_problem_id"):
            skipped["problem_mismatch"] += 1
            continue
        if pair.category == DEFAULT_SEMANTIC_DECOY_CATEGORY:
            if query.get("source_submission_id") == decoy.get("source_submission_id"):
                skipped["same_submission"] += 1
                continue
        elif pair.category == "same_code_different_input":
            if query.get("source_submission_id") != decoy.get("source_submission_id"):
                skipped["different_submission"] += 1
                continue
            if query.get("input_id") == decoy.get("input_id"):
                skipped["same_input"] += 1
                continue
        else:
            skipped["unsupported_category"] += 1
            continue
        if _output_fingerprint(query) == _output_fingerprint(decoy):
            skipped["outputs_identical"] += 1
            continue
        key = (pair.category, pair.query_record_id, pair.decoy_record_id)
        if key in seen:
            skipped["duplicate_pair"] += 1
            continue
        seen.add(key)
        input_relation = (
            "same_input"
            if query.get("input_id") == decoy.get("input_id")
            else "different_input"
        )
        rows.append(
            {
                "schema_version": SEMANTIC_DECOY_PAIR_SCHEMA_VERSION,
                "decoy_id": _pair_id(pair.category, pair.query_record_id, pair.decoy_record_id),
                "category": pair.category,
                "control_category": "semantic_same_problem",
                "source_dataset": str(query["source_dataset"]),
                "source_problem_id": str(query["source_problem_id"]),
                "query_record_id": pair.query_record_id,
                "decoy_record_id": pair.decoy_record_id,
                "query_record_schema_version": str(query.get("schema_version", "unknown")),
                "decoy_record_schema_version": str(decoy.get("schema_version", "unknown")),
                "query_source_dataset": str(query["source_dataset"]),
                "decoy_source_dataset": str(decoy["source_dataset"]),
                "query_source_submission_id": str(query["source_submission_id"]),
                "decoy_source_submission_id": str(decoy["source_submission_id"]),
                "query_input_id": str(query["input_id"]),
                "decoy_input_id": str(decoy["input_id"]),
                "input_relation": input_relation,
                "output_relation": "differing_output",
                "query_output_fingerprint": _output_fingerprint(query),
                "decoy_output_fingerprint": _output_fingerprint(decoy),
                "rationale": pair.rationale,
                "semantic_adversarial_reason": _semantic_reason(pair.category),
            }
        )
    return tuple(rows), {
        "checked_pair_count": len(pairs),
        "kept_pair_count": len(rows),
        "skipped_reasons": dict(sorted(skipped.items())),
        "train_split_leak_count": int(skipped.get("train_split_leak", 0)),
    }


def _claim_gate(
    *,
    pair_count: int,
    distinct_problem_count: int,
    min_pairs: int,
    min_problems: int,
) -> dict[str, Any]:
    failures: list[str] = []
    if pair_count < min_pairs:
        failures.append(f"pair_count={pair_count}<min_pairs={min_pairs}")
    if distinct_problem_count < min_problems:
        failures.append(
            f"distinct_problem_count={distinct_problem_count}<min_problems={min_problems}"
        )
    return {
        "name": "semantic_decoy_pair_count_gate.v1",
        "claim_allowed": not failures,
        "claim_reason": "pair_and_problem_count_gates_pass" if not failures else "; ".join(failures),
        "pair_count": pair_count,
        "distinct_problem_count": distinct_problem_count,
        "min_pairs": min_pairs,
        "min_distinct_problems": min_problems,
    }


def _decoy_pair_from_row(row: Mapping[str, Any]) -> DecoyPair:
    if row.get("schema_version") != SEMANTIC_DECOY_PAIR_SCHEMA_VERSION:
        raise SemanticDecoyPackError("semantic decoy pair row has unsupported schema")
    category = row.get("category")
    if category not in {
        DEFAULT_SEMANTIC_DECOY_CATEGORY,
        "same_code_different_input",
    }:
        raise SemanticDecoyPackError(f"unsupported semantic decoy category: {category!r}")
    return DecoyPair(
        category=str(category),
        query_record_id=_required_str(row, "query_record_id"),
        query_output_repr=_required_str(row, "query_output_fingerprint"),
        decoy_record_id=_required_str(row, "decoy_record_id"),
        decoy_output_repr=_required_str(row, "decoy_output_fingerprint"),
        rationale=_required_str(row, "rationale"),
    )


def _metadata_path(artifact: ArtifactManifest, key: str) -> str:
    value = artifact.metadata.get(key)
    if not isinstance(value, str) or not value:
        raise SemanticDecoyPackError(f"semantic decoy manifest metadata.{key} is missing")
    return value


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SemanticDecoyPackError(
                    f"{path}:{line_no}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise SemanticDecoyPackError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    return tuple(rows)


def _write_json(path_payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(path_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _reject_existing(
    paths: Sequence[Path], *, overwrite: bool, output_dir: Path
) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        rel = ", ".join(_relative_to_root(path, output_dir) for path in existing)
        raise SemanticDecoyPackError(
            f"output already exists; pass --overwrite to replace: {rel}"
        )


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _required_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise SemanticDecoyPackError(f"semantic decoy row {key} must be a string")
    return value


def _pair_id(category: str, query_record_id: str, decoy_record_id: str) -> str:
    digest = hashlib.sha256(
        f"{category}\0{query_record_id}\0{decoy_record_id}".encode("utf-8")
    ).hexdigest()
    return f"semantic-decoy-{digest[:24]}"


def _output_fingerprint(record: Mapping[str, Any]) -> str:
    raw = record.get("output_repr")
    if isinstance(raw, str) and raw:
        return raw
    checksum = record.get("output_repr_checksum")
    if isinstance(checksum, str) and checksum:
        return f"sha256:{checksum}"
    tokens = record.get("output_tokens")
    if isinstance(tokens, list):
        digest = hashlib.sha256(",".join(str(token) for token in tokens).encode("utf-8")).hexdigest()
        return f"tokens_sha256:{digest}"
    return ""


def _semantic_reason(category: str) -> str:
    if category == "same_code_different_input":
        return (
            "same problem and same submission, different input, differing output; "
            "tests input-conditioned semantic discrimination"
        )
    return (
        "same problem, different submission, differing output; "
        "surface/problem context is intentionally close"
    )


__all__ = [
    "DEFAULT_SEMANTIC_DECOY_CATEGORY",
    "SEMANTIC_DECOY_PACK_RUN_SCHEMA_VERSION",
    "SEMANTIC_DECOY_PACK_SCHEMA_VERSION",
    "SEMANTIC_DECOY_PAIR_SCHEMA_VERSION",
    "SEMANTIC_DECOY_SUMMARY_SCHEMA_VERSION",
    "LoadedSemanticDecoyPack",
    "SemanticDecoyPackError",
    "SemanticDecoyPackResult",
    "build_semantic_decoy_pack",
    "load_semantic_decoy_pack",
]
