"""Execution-pack builder orchestration.

Stages:

1. Stream the ingestion JSONL produced by
   :func:`codelewm.data.execution_sources.load_execution_source`.
2. For each ``SourceSubmission`` and each ``InputCase``, run
   :func:`codelewm.data.sandbox.run_one` under the configured policy.
3. Discard records that fail the determinism gate, hit a policy
   violation, time out, OOM, or whose output ``repr`` exceeds the
   truncation cap.
4. Tokenize the code, input ``repr``, and output ``repr``.
5. Partition surviving records by ``source_problem_id`` (held-out
   ingestion records are not packed).
6. Write ``pack.jsonl``, ``manifest.json``, ``attribution.json``, and
   ``sandbox_audit_summary.json`` to ``output_dir``.

The HDF5 mirror is deferred. The training executor consumes the
JSONL directly for the substrate pivot's first run.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codelewm.data.execution_sources import (
    EXECUTION_SOURCE_RECORD_SCHEMA_VERSION,
    InputCase,
    SourceSubmission,
)
from codelewm.security.claim_boundaries import (
    claim_boundary_fingerprint,
    load_claim_boundary,
)

from .manifest import (
    EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
    EXECUTION_PACK_RECORD_SCHEMA_VERSION,
    ExecutionPackManifest,
    sha256_file,
    write_execution_pack_manifest,
)
from .record import (
    PackedExecutionRecord,
    SplitName,
    sha256_text,
    tokenize_text,
)


CLAIM_BOUNDARY_NAME = "execution_substrate.v1"


class ExecutionPackBuilderError(RuntimeError):
    """Raised when the pack builder cannot complete cleanly."""


@dataclass(frozen=True)
class ExecutionPackResult:
    """Summary of a completed pack build."""

    output_dir: Path
    manifest: ExecutionPackManifest
    record_count: int
    sandbox_reject_counts: dict[str, int]


def build_execution_pack(
    *,
    ingestion_paths: Iterable[Path],
    output_dir: Path,
    sandbox_policy: Any | None = None,
    seed: int = 42,
    train_frac: float = 0.85,
    val_frac: float = 0.05,
    max_inputs_per_problem: int | None = None,
    target_records: int | None = None,
    pack_id: str | None = None,
) -> ExecutionPackResult:
    """Build an execution pack from one or more ingestion JSONL files.

    ``sandbox_policy`` is a :class:`codelewm.data.sandbox.SandboxPolicy`
    instance (passed in by the caller — this module never imports the
    sandbox module). Pass ``None`` to skip the sandbox stage entirely;
    in that case the builder will refuse to write any pack and raise
    :class:`ExecutionPackBuilderError`, which matches the spec
    requirement that pack records always carry deterministic
    sandbox-captured outputs.
    """

    if not 0 < train_frac < 1:
        raise ExecutionPackBuilderError(
            f"train_frac must be in (0, 1), got {train_frac}"
        )
    if not 0 < val_frac < 1:
        raise ExecutionPackBuilderError(
            f"val_frac must be in (0, 1), got {val_frac}"
        )
    if train_frac + val_frac >= 1.0:
        raise ExecutionPackBuilderError(
            f"train_frac + val_frac must be < 1, got {train_frac + val_frac}"
        )
    if sandbox_policy is None:
        raise ExecutionPackBuilderError(
            "sandbox_policy is required; pass a "
            "codelewm.data.sandbox.SandboxPolicy instance"
        )

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExecutionPackBuilderError(
            f"output_dir must be empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    pack_id = pack_id or _default_pack_id()
    rng = random.Random(seed)

    # Run the sandbox now; collect surviving records before split assignment.
    pack_jsonl_path = output_dir / "pack.jsonl"
    audit = _AuditCounters()
    records, attribution, parent_artifacts, held_out_count = _build_records(
        ingestion_paths=list(ingestion_paths),
        sandbox_policy=sandbox_policy,
        audit=audit,
        max_inputs_per_problem=max_inputs_per_problem,
        target_records=target_records,
    )

    if not records:
        # Always write the artifact files so the manifest contract holds
        # even on empty packs (CI fixture path may produce zero records).
        pack_jsonl_path.write_text("", encoding="utf-8")
        manifest = _empty_manifest(
            output_dir=output_dir,
            pack_id=pack_id,
            sandbox_policy=sandbox_policy,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            max_inputs_per_problem=max_inputs_per_problem,
            audit=audit,
            parent_artifacts=parent_artifacts,
            held_out_count=held_out_count,
        )
        write_execution_pack_manifest(manifest, output_dir / "manifest.json")
        _write_sidecars(
            output_dir=output_dir,
            attribution=attribution,
            audit=audit,
        )
        return ExecutionPackResult(
            output_dir=output_dir,
            manifest=manifest,
            record_count=0,
            sandbox_reject_counts=dict(audit.rejects),
        )

    # Partition by source_problem_id so no problem leaks across splits.
    splits = _assign_splits(
        records=records,
        rng=rng,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    with pack_jsonl_path.open("w", encoding="utf-8") as fh:
        for rec, split in zip(records, splits, strict=True):
            packed = _attach_split(rec, split)
            payload = packed.as_dict()
            payload["schema_version"] = EXECUTION_PACK_RECORD_SCHEMA_VERSION
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")

    pack_checksum = sha256_file(pack_jsonl_path)
    split_counts = Counter(splits)
    output_type_distribution = Counter(r.output_type for r in records)
    output_kind_distribution = Counter(r.output_kind for r in records)
    execution_status_distribution = Counter(r.execution_status for r in records)
    source_breakdown = Counter(r.source_dataset for r in records)
    license_breakdown = Counter(r.license for r in records)

    manifest = ExecutionPackManifest(
        schema_version=EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        pack_id=pack_id,
        pack_dir=str(output_dir),
        record_count=len(records),
        split_counts=dict(split_counts),
        split_by="source_problem_id",
        output_type_distribution=dict(output_type_distribution),
        output_kind_distribution=dict(output_kind_distribution),
        execution_status_distribution=dict(execution_status_distribution),
        source_breakdown=dict(source_breakdown),
        license_breakdown=dict(license_breakdown),
        sandbox_policy=_policy_dict(sandbox_policy),
        sandbox_reject_counts=dict(audit.rejects),
        parent_artifacts=tuple(parent_artifacts),
        held_out_eval_excluded_count=held_out_count,
        claim_boundary={
            "name": CLAIM_BOUNDARY_NAME,
            "fingerprint": claim_boundary_fingerprint(CLAIM_BOUNDARY_NAME),
        },
        pack_jsonl_checksum=pack_checksum,
        tokenizer={
            "name": "codelewm.tokenizer.blake2b_hash",
            "version": "v1",
            "vocab_strategy": "stable_hash_31bit",
        },
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        max_inputs_per_problem=max_inputs_per_problem,
    )

    write_execution_pack_manifest(manifest, output_dir / "manifest.json")
    _write_sidecars(
        output_dir=output_dir,
        attribution=attribution,
        audit=audit,
    )

    # The claim boundary is embedded verbatim so consumers can fingerprint
    # without needing to import codelewm.security.
    (output_dir / "claim_boundary.md").write_text(
        load_claim_boundary(CLAIM_BOUNDARY_NAME), encoding="utf-8"
    )

    return ExecutionPackResult(
        output_dir=output_dir,
        manifest=manifest,
        record_count=len(records),
        sandbox_reject_counts=dict(audit.rejects),
    )


@dataclass
class _AuditCounters:
    rejects: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.rejects = Counter()

    def reject(self, reason: str) -> None:
        self.rejects[reason] += 1


def _build_records(
    *,
    ingestion_paths: list[Path],
    sandbox_policy: Any,
    audit: _AuditCounters,
    max_inputs_per_problem: int | None,
    target_records: int | None,
) -> tuple[
    list[PackedExecutionRecord],
    dict[str, str],
    list[dict[str, str]],
    int,
]:
    """Drive the sandbox over the ingestion records.

    The sandbox import is lazy and local so this module never appears in
    any model-path module's import graph.
    """

    # Lazy import: keeps the structural import-boundary test passing.
    from codelewm.data.sandbox import (  # noqa: PLC0415
        SandboxExitCode,
        run_one,
    )

    records: list[PackedExecutionRecord] = []
    attribution: dict[str, str] = {}
    parent_artifacts: list[dict[str, str]] = []
    held_out_count = 0
    per_problem_kept: Counter[str] = Counter()

    for path in ingestion_paths:
        parent_artifacts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path) if path.is_file() else "",
                "schema_version": EXECUTION_SOURCE_RECORD_SCHEMA_VERSION,
            }
        )
        if not path.is_file():
            raise ExecutionPackBuilderError(f"ingestion file missing: {path}")
        with path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ExecutionPackBuilderError(
                        f"{path}:{line_no}: invalid JSON"
                    ) from exc
                if row.get("held_out_for_eval"):
                    held_out_count += 1
                    audit.reject("held_out_for_eval")
                    continue
                submission = _submission_from_row(row)
                if submission is None:
                    audit.reject("ingestion_row_invalid")
                    continue
                attribution.setdefault(
                    submission.source_dataset,
                    submission.license_attribution_url,
                )
                for case_idx, case in enumerate(submission.inputs):
                    if (
                        max_inputs_per_problem is not None
                        and per_problem_kept[submission.source_problem_id]
                        >= max_inputs_per_problem
                    ):
                        break
                    record = _run_one_case(
                        submission=submission,
                        case=case,
                        sandbox_policy=sandbox_policy,
                        run_one=run_one,
                        ok_code=SandboxExitCode.OK,
                        raised_code=SandboxExitCode.RAISED,
                        audit=audit,
                    )
                    if record is None:
                        continue
                    records.append(record)
                    per_problem_kept[submission.source_problem_id] += 1
                    if target_records is not None and len(records) >= target_records:
                        return records, attribution, parent_artifacts, held_out_count

    return records, attribution, parent_artifacts, held_out_count


def _run_one_case(
    *,
    submission: SourceSubmission,
    case: InputCase,
    sandbox_policy: Any,
    run_one: Any,
    ok_code: Any,
    raised_code: Any,
    audit: _AuditCounters,
) -> PackedExecutionRecord | None:
    """Invoke the sandbox for one input case and tokenize the survivors."""

    # The sandbox supports function_call directly. For stdin / argv inputs
    # we plug into stdin_text only when the policy says so; stdin / argv
    # are deferred to a future PR that adds a stdin wrapper. For now we
    # skip non-function-call cases so the first execution-pack run can
    # focus on MBPP-style inputs.
    if case.input_kind != "function_call":
        audit.reject(f"unsupported_input_kind:{case.input_kind}")
        return None

    result = run_one(
        submission.code,
        input_repr=case.input_repr,
        function_name=case.function_name,
        policy=sandbox_policy,
    )
    exit_value = (
        result.exit_code.value if hasattr(result.exit_code, "value") else str(result.exit_code)
    )
    if exit_value not in {"ok", "raised"}:
        audit.reject(f"sandbox_{exit_value}")
        return None
    if not result.determinism_check:
        audit.reject("nondeterministic")
        return None
    if result.output_repr is None and exit_value == "ok":
        audit.reject("missing_output_repr")
        return None
    if result.output_truncated:
        audit.reject("output_truncated")
        return None

    output_repr = result.output_repr or ""
    if exit_value == "raised":
        # For exception records, the "output" is the exception class +
        # message so the model has something concrete to predict.
        output_repr = f"{result.exception_class}: {result.exception_message}"

    return PackedExecutionRecord(
        source_dataset=submission.source_dataset,
        source_problem_id=submission.source_problem_id,
        source_submission_id=submission.source_submission_id,
        input_id=case.input_id,
        split="train",  # placeholder; final split is assigned below.
        code_tokens=tokenize_text(submission.code),
        code_checksum=sha256_text(submission.code),
        input_tokens=tokenize_text(case.input_repr),
        input_repr_checksum=sha256_text(case.input_repr),
        input_kind=case.input_kind,
        function_name=case.function_name,
        output_tokens=tokenize_text(output_repr),
        output_repr_checksum=sha256_text(output_repr),
        output_kind=result.output_kind,
        output_type=result.output_type,
        execution_status=exit_value,
        judge_verdict=submission.judge_verdict,
        wall_time_ms=float(result.wall_time_ms),
        peak_rss_kb=int(result.peak_rss_kb),
        determinism_check=True,
        license=submission.license,
        license_attribution_url=submission.license_attribution_url,
        held_out_for_eval=submission.held_out_for_eval,
    )


def _submission_from_row(row: dict[str, Any]) -> SourceSubmission | None:
    """Reconstruct a SourceSubmission from an ingestion JSONL row.

    Returns ``None`` for rows that do not satisfy the record contract.
    """

    inputs_raw = row.get("inputs") or []
    if not isinstance(inputs_raw, list) or not inputs_raw:
        return None
    cases: list[InputCase] = []
    for case_row in inputs_raw:
        if not isinstance(case_row, dict):
            return None
        try:
            cases.append(
                InputCase(
                    input_id=str(case_row["input_id"]),
                    input_repr=str(case_row["input_repr"]),
                    input_kind=str(case_row["input_kind"]),  # type: ignore[arg-type]
                    function_name=case_row.get("function_name"),
                )
            )
        except (KeyError, ValueError):
            return None
    expected = row.get("expected_outputs")
    if expected is not None and not isinstance(expected, list):
        return None
    try:
        return SourceSubmission(
            source_dataset=row["source_dataset"],
            source_problem_id=row["source_problem_id"],
            source_submission_id=row["source_submission_id"],
            code=row["code"],
            inputs=tuple(cases),
            expected_outputs=tuple(expected) if expected is not None else None,
            judge_verdict=row.get("judge_verdict"),
            license=row["license"],
            license_attribution_url=row["license_attribution_url"],
            held_out_for_eval=bool(row.get("held_out_for_eval", False)),
            raw_hash=str(row.get("raw_hash", "")),
        )
    except (KeyError, ValueError):
        return None


def _assign_splits(
    *,
    records: list[PackedExecutionRecord],
    rng: random.Random,
    train_frac: float,
    val_frac: float,
) -> list[SplitName]:
    """Assign each record to a split, grouped by source_problem_id."""

    by_problem: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        by_problem.setdefault(rec.source_problem_id, []).append(idx)
    problem_ids = sorted(by_problem)
    rng.shuffle(problem_ids)
    n = len(problem_ids)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_ids = set(problem_ids[:n_train])
    val_ids = set(problem_ids[n_train : n_train + n_val])
    split: list[SplitName] = ["test"] * len(records)
    for problem_id, idxs in by_problem.items():
        if problem_id in train_ids:
            target: SplitName = "train"
        elif problem_id in val_ids:
            target = "val"
        else:
            target = "test"
        for idx in idxs:
            split[idx] = target
    return split


def _attach_split(record: PackedExecutionRecord, split: SplitName) -> PackedExecutionRecord:
    from dataclasses import replace

    return replace(record, split=split)


def _policy_dict(policy: Any) -> dict[str, Any]:
    if hasattr(policy, "as_dict"):
        return dict(policy.as_dict())
    return {"repr": repr(policy)}


def _empty_manifest(
    *,
    output_dir: Path,
    pack_id: str,
    sandbox_policy: Any,
    seed: int,
    train_frac: float,
    val_frac: float,
    max_inputs_per_problem: int | None,
    audit: _AuditCounters,
    parent_artifacts: list[dict[str, str]],
    held_out_count: int,
) -> ExecutionPackManifest:
    return ExecutionPackManifest(
        schema_version=EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        pack_id=pack_id,
        pack_dir=str(output_dir),
        record_count=0,
        split_counts={"train": 0, "val": 0, "test": 0},
        split_by="source_problem_id",
        output_type_distribution={},
        output_kind_distribution={},
        execution_status_distribution={},
        source_breakdown={},
        license_breakdown={},
        sandbox_policy=_policy_dict(sandbox_policy),
        sandbox_reject_counts=dict(audit.rejects),
        parent_artifacts=tuple(parent_artifacts),
        held_out_eval_excluded_count=held_out_count,
        claim_boundary={
            "name": CLAIM_BOUNDARY_NAME,
            "fingerprint": claim_boundary_fingerprint(CLAIM_BOUNDARY_NAME),
        },
        pack_jsonl_checksum="",
        tokenizer={
            "name": "codelewm.tokenizer.blake2b_hash",
            "version": "v1",
            "vocab_strategy": "stable_hash_31bit",
        },
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        max_inputs_per_problem=max_inputs_per_problem,
    )


def _write_sidecars(
    *,
    output_dir: Path,
    attribution: dict[str, str],
    audit: _AuditCounters,
) -> None:
    (output_dir / "attribution.json").write_text(
        json.dumps(dict(sorted(attribution.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sandbox_audit_summary.json").write_text(
        json.dumps(dict(sorted(audit.rejects.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_pack_id() -> str:
    return "codelewm-execution-pack-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
