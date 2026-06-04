"""Build v0.8 pass/fail execution packs from completion-label artifacts."""

from __future__ import annotations

import ast
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codelewm.data.execution_rerank_sampler import (
    COMPLETION_LABEL_SCHEMA_VERSION,
    CompletionSamplingError,
)
from codelewm.data.execution_sources import (
    SourceSubmission,
    get_execution_source_adapter,
)
from codelewm.data.sandbox import (
    DEFAULT_SANDBOX_POLICY,
    SandboxExitCode,
    SandboxPolicy,
    SandboxPolicyError,
    SandboxRunnerError,
    run_one,
)
from codelewm.observability import build_artifact_manifest, write_artifact_manifest
from codelewm.security.claim_boundaries import (
    claim_boundary_fingerprint,
    load_claim_boundary,
)
from codelewm.security.secret_scan import scan_paths

from .builder import CLAIM_BOUNDARY_NAME, ExecutionPackBuilderError
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
    length_bucket,
    magnitude_bucket,
    sha256_text,
    tokenize_text,
)


PASSFAIL_PACK_REPORT_SCHEMA_VERSION = "codelewm.execution_passfail_pack_report.v1"
PASSFAIL_PACK_CONFIG_SCHEMA_VERSION = "codelewm.execution_passfail_pack_config.v1"


class PassFailPackBuilderError(ExecutionPackBuilderError):
    """Raised when the pass/fail pack builder cannot complete cleanly."""


@dataclass(frozen=True)
class PassFailPackResult:
    """Summary returned by :func:`build_passfail_pack`."""

    output_dir: Path
    manifest: ExecutionPackManifest
    record_count: int
    pass_label_counts: dict[str, int]
    pos_weight: float
    report_path: str
    secret_scan_report_path: str
    artifact_manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "manifest": self.manifest.as_dict(),
            "record_count": self.record_count,
            "pass_label_counts": dict(sorted(self.pass_label_counts.items())),
            "pos_weight": self.pos_weight,
            "report_path": self.report_path,
            "secret_scan_report_path": self.secret_scan_report_path,
            "artifact_manifest_path": self.artifact_manifest_path,
        }


def build_passfail_pack(
    *,
    completion_label_paths: Iterable[Path],
    source_path: Path,
    benchmark: str,
    output_dir: Path,
    sandbox_policy: SandboxPolicy = DEFAULT_SANDBOX_POLICY,
    seed: int = 42,
    train_frac: float = 0.85,
    val_frac: float = 0.05,
    max_completion_rows: int | None = None,
    max_records: int | None = None,
    pack_id: str | None = None,
    overwrite: bool = False,
    allow_secret_findings: bool = False,
    command: Sequence[str] = ("scripts/build-passfail-pack",),
) -> PassFailPackResult:
    """Convert completion-label rows into a supervised execution pack.

    Granularity is per ``(problem, completion, input)``. The adapter re-executes
    each completion against each persisted scoring input to recover
    ``output_repr`` for the world-model target, then computes ``passed`` for that
    exact input by comparing the recovered output hash with the expected-output
    hash stored in the completion-label row.
    """

    label_paths = tuple(Path(path) for path in completion_label_paths)
    if not label_paths:
        raise PassFailPackBuilderError("at least one completion-label path is required")
    for path in label_paths:
        if not path.is_file():
            raise PassFailPackBuilderError(f"completion-label file missing: {path}")
    if not source_path.is_file():
        raise PassFailPackBuilderError(f"source file missing: {source_path}")
    _validate_split_fracs(train_frac=train_frac, val_frac=val_frac)
    if max_completion_rows is not None:
        _positive_int(max_completion_rows, "max_completion_rows")
    if max_records is not None:
        _positive_int(max_records, "max_records")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise PassFailPackBuilderError(
            f"output_dir must be empty or --overwrite must be set: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_id = benchmark.strip().lower().replace("-", "_")
    submissions = _load_source_submissions(benchmark_id, source_path)
    rows = _load_completion_label_rows(
        label_paths,
        benchmark_id=benchmark_id,
        max_rows=max_completion_rows,
    )
    if not rows:
        raise PassFailPackBuilderError("completion-label inputs yielded zero rows")

    audit = Counter()
    records: list[PackedExecutionRecord] = []
    completion_label_mismatch_count = 0
    for row in rows:
        problem_id = str(row["problem_id"])
        submission = submissions.get(problem_id)
        if submission is None:
            audit["missing_source_problem"] += 1
            continue
        built, completion_pass_matches = _records_from_completion_row(
            row=row,
            submission=submission,
            sandbox_policy=sandbox_policy,
            audit=audit,
        )
        records.extend(built)
        if not completion_pass_matches:
            completion_label_mismatch_count += 1
        if max_records is not None and len(records) >= max_records:
            records = records[:max_records]
            break

    if not records:
        raise PassFailPackBuilderError(
            "no pass/fail records were produced; inspect sandbox reject counts"
        )

    splits = _assign_splits(
        records=records,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    records = [_attach_split(record, split) for record, split in zip(records, splits, strict=True)]

    pass_counts = Counter("true" if record.passed else "false" for record in records)
    if not pass_counts["true"] or not pass_counts["false"]:
        raise PassFailPackBuilderError(
            "pass/fail pack must contain both classes; "
            f"got {dict(sorted(pass_counts.items()))}"
        )
    pos_weight = pass_counts["false"] / pass_counts["true"]
    pack_id = pack_id or _default_passfail_pack_id()

    pack_jsonl_path = output_dir / "pack.jsonl"
    with pack_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.as_dict()
            payload["schema_version"] = EXECUTION_PACK_RECORD_SCHEMA_VERSION
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    attribution = {
        submission.source_dataset: submission.license_attribution_url
        for submission in submissions.values()
        if submission.license_attribution_url
    }
    split_counts = Counter(record.split for record in records)
    manifest = ExecutionPackManifest(
        schema_version=EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        pack_id=pack_id,
        pack_dir=str(output_dir),
        record_count=len(records),
        split_counts=dict(split_counts),
        split_by="source_problem_id",
        output_type_distribution=dict(Counter(record.output_type for record in records)),
        output_kind_distribution=dict(Counter(record.output_kind for record in records)),
        execution_status_distribution=dict(
            Counter(record.execution_status for record in records)
        ),
        source_breakdown=dict(Counter(record.source_dataset for record in records)),
        license_breakdown=dict(Counter(record.license for record in records)),
        sandbox_policy=sandbox_policy.as_dict(),
        sandbox_reject_counts=dict(audit),
        parent_artifacts=(
            {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "schema_version": "codelewm.execution_source_record.v1",
            },
            *(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
                }
                for path in label_paths
            ),
        ),
        held_out_eval_excluded_count=0,
        claim_boundary={
            "name": CLAIM_BOUNDARY_NAME,
            "fingerprint": claim_boundary_fingerprint(CLAIM_BOUNDARY_NAME),
        },
        pack_jsonl_checksum=sha256_file(pack_jsonl_path),
        tokenizer={
            "name": "codelewm.tokenizer.blake2b_hash",
            "version": "v1",
            "vocab_strategy": "stable_hash_31bit",
        },
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        max_inputs_per_problem=None,
    )

    config_path = output_dir / "config.json"
    report_path = output_dir / "reports" / "passfail_pack_report.json"
    secret_scan_report_path = output_dir / "reports" / "secret_scan_report.json"
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    output_dir.joinpath("reports").mkdir(parents=True, exist_ok=True)
    config_payload = {
        "schema_version": PASSFAIL_PACK_CONFIG_SCHEMA_VERSION,
        "benchmark": benchmark_id,
        "completion_label_paths": [str(path) for path in label_paths],
        "source_path": str(source_path),
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "max_completion_rows": max_completion_rows,
        "max_records": max_records,
        "sandbox_policy": sandbox_policy.as_dict(),
        "pass_label_granularity": "per_problem_completion_input",
    }
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_execution_pack_manifest(manifest, output_dir / "manifest.json")
    (output_dir / "attribution.json").write_text(
        json.dumps(dict(sorted(attribution.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sandbox_audit_summary.json").write_text(
        json.dumps(dict(sorted(audit.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "claim_boundary.md").write_text(
        load_claim_boundary(CLAIM_BOUNDARY_NAME), encoding="utf-8"
    )

    split_pass_counts = _split_pass_counts(records)
    report_payload = {
        "schema_version": PASSFAIL_PACK_REPORT_SCHEMA_VERSION,
        "record_schema_version": EXECUTION_PACK_RECORD_SCHEMA_VERSION,
        "record_count": len(records),
        "completion_label_row_count": len(rows),
        "completion_label_mismatch_count": completion_label_mismatch_count,
        "pass_label_granularity": "per_problem_completion_input",
        "pass_label_counts": dict(sorted(pass_counts.items())),
        "pass_label_rate": pass_counts["true"] / len(records),
        "pos_weight": pos_weight,
        "split_counts": dict(sorted(split_counts.items())),
        "split_pass_label_counts": split_pass_counts,
        "sandbox_reject_counts": dict(sorted(audit.items())),
        "class_balance_ok": bool(pass_counts["true"] and pass_counts["false"]),
        "claim_allowed": False,
        "claim_reason": "passfail_training_pack_only_model_not_trained",
    }
    report_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    scan_report = scan_paths(
        (
            pack_jsonl_path,
            config_path,
            output_dir / "manifest.json",
            output_dir / "attribution.json",
            output_dir / "sandbox_audit_summary.json",
            report_path,
        ),
        include_suffixes=(),
        recursive=False,
    )
    scan_payload = _relative_secret_scan_payload(scan_report.to_dict(), output_dir)
    secret_scan_report_path.write_text(
        json.dumps(scan_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if scan_payload["findings"] and not allow_secret_findings:
        raise PassFailPackBuilderError(
            "pass/fail pack contains secret-scan findings; refusing to publish"
        )

    artifact_manifest = build_artifact_manifest(
        artifact_kind="dataset",
        root=output_dir,
        files=(
            pack_jsonl_path,
            output_dir / "manifest.json",
            output_dir / "attribution.json",
            output_dir / "sandbox_audit_summary.json",
            output_dir / "claim_boundary.md",
            config_path,
            report_path,
            secret_scan_report_path,
        ),
        command=command,
        config=config_payload,
        metadata={
            "execution_pack_manifest_schema": manifest.schema_version,
            "record_schema_version": EXECUTION_PACK_RECORD_SCHEMA_VERSION,
            "record_count": len(records),
            "pass_label_counts": dict(sorted(pass_counts.items())),
            "pos_weight": pos_weight,
            "secret_scan_ok": bool(scan_payload["ok"]),
        },
        artifact_id=pack_id,
    )
    write_artifact_manifest(artifact_manifest, artifact_manifest_path)

    return PassFailPackResult(
        output_dir=output_dir,
        manifest=manifest,
        record_count=len(records),
        pass_label_counts=dict(sorted(pass_counts.items())),
        pos_weight=pos_weight,
        report_path=_relative_to_root(report_path, output_dir),
        secret_scan_report_path=_relative_to_root(secret_scan_report_path, output_dir),
        artifact_manifest_path=_relative_to_root(artifact_manifest_path, output_dir),
    )


def _load_source_submissions(
    benchmark_id: str, source_path: Path
) -> dict[str, SourceSubmission]:
    try:
        adapter = get_execution_source_adapter(benchmark_id)
        submissions = tuple(adapter.iter_submissions(source_path=source_path))
    except Exception as exc:  # noqa: BLE001 - convert adapter failures to builder errors.
        raise PassFailPackBuilderError(str(exc)) from exc
    by_problem = {submission.source_problem_id: submission for submission in submissions}
    if not by_problem:
        raise PassFailPackBuilderError(
            f"no {benchmark_id} submissions could be parsed from {source_path}"
        )
    return by_problem


def _load_completion_label_rows(
    paths: Sequence[Path], *, benchmark_id: str, max_rows: int | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise PassFailPackBuilderError(
                        f"{path}:{line_no}: invalid JSON"
                    ) from exc
                if row.get("schema_version") != COMPLETION_LABEL_SCHEMA_VERSION:
                    raise PassFailPackBuilderError(
                        f"{path}:{line_no}: unsupported completion-label schema"
                    )
                row_benchmark = str(row.get("benchmark_id", "")).replace("-", "_")
                if row_benchmark != benchmark_id:
                    raise PassFailPackBuilderError(
                        f"{path}:{line_no}: benchmark_id {row_benchmark!r} "
                        f"does not match {benchmark_id!r}"
                    )
                rows.append(row)
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
    return rows


def _records_from_completion_row(
    *,
    row: Mapping[str, Any],
    submission: SourceSubmission,
    sandbox_policy: SandboxPolicy,
    audit: Counter[str],
) -> tuple[list[PackedExecutionRecord], bool]:
    code = str(row.get("completion_text") or row.get("code") or "")
    if not code.strip():
        audit["missing_completion_text"] += 1
        return [], True
    completion_id = str(row.get("completion_id") or row.get("completion_sha256") or "")
    if not completion_id:
        audit["missing_completion_id"] += 1
        return [], True
    expected_hash_by_input = _expected_hashes_by_input(row)
    scoring_inputs = row.get("scoring_inputs")
    if not isinstance(scoring_inputs, list) or not scoring_inputs:
        audit["missing_scoring_inputs"] += 1
        return [], True

    records: list[PackedExecutionRecord] = []
    row_all_passed = True
    for input_row in scoring_inputs:
        if not isinstance(input_row, Mapping):
            audit["invalid_scoring_input"] += 1
            row_all_passed = False
            continue
        input_id = str(input_row.get("input_id") or "")
        expected_hash = expected_hash_by_input.get(input_id)
        if not input_id or expected_hash is None:
            audit["missing_expected_output_hash"] += 1
            row_all_passed = False
            continue
        record = _record_from_input(
            code=code,
            completion_id=completion_id,
            completion_level_passed=bool(row.get("passed")),
            input_row=input_row,
            expected_output_sha256=expected_hash,
            submission=submission,
            sandbox_policy=sandbox_policy,
            audit=audit,
        )
        if record is None:
            row_all_passed = False
            continue
        records.append(record)
        row_all_passed = row_all_passed and bool(record.passed)
    return records, row_all_passed == bool(row.get("passed"))


def _expected_hashes_by_input(row: Mapping[str, Any]) -> dict[str, str]:
    test_results = row.get("test_results")
    if not isinstance(test_results, list):
        return {}
    expected: dict[str, str] = {}
    for result in test_results:
        if not isinstance(result, Mapping):
            continue
        input_id = result.get("input_id")
        output_hash = result.get("expected_output_sha256")
        if isinstance(input_id, str) and isinstance(output_hash, str):
            expected[input_id] = output_hash
    return expected


def _record_from_input(
    *,
    code: str,
    completion_id: str,
    completion_level_passed: bool,
    input_row: Mapping[str, Any],
    expected_output_sha256: str,
    submission: SourceSubmission,
    sandbox_policy: SandboxPolicy,
    audit: Counter[str],
) -> PackedExecutionRecord | None:
    input_kind = str(input_row.get("input_kind") or "")
    input_repr = str(input_row.get("input_repr") or "")
    function_name = input_row.get("function_name")
    input_id = str(input_row.get("input_id") or "")
    try:
        if input_kind == "function_call":
            result = run_one(
                code,
                input_repr=input_repr,
                function_name=str(function_name) if function_name is not None else None,
                policy=sandbox_policy,
            )
        elif input_kind == "stdin":
            result = run_one(code, stdin_text=input_repr, policy=sandbox_policy)
        else:
            audit[f"unsupported_input_kind:{input_kind}"] += 1
            return None
    except (SandboxPolicyError, SandboxRunnerError, CompletionSamplingError) as exc:
        audit[f"sandbox_error:{type(exc).__name__}"] += 1
        return None

    exit_value = result.exit_code.value
    if result.exit_code not in {SandboxExitCode.OK, SandboxExitCode.RAISED}:
        audit[f"sandbox_{exit_value}"] += 1
        return None
    if not result.determinism_check:
        audit["nondeterministic"] += 1
        return None
    if result.output_truncated:
        audit["output_truncated"] += 1
        return None
    if result.exit_code is SandboxExitCode.OK and result.output_repr is None:
        audit["missing_output_repr"] += 1
        return None

    output_repr = result.output_repr or ""
    if result.exit_code is SandboxExitCode.RAISED:
        output_repr = f"{result.exception_class}: {result.exception_message}"
    observed_hash = sha256_text(result.output_repr or "")
    passed = result.ok and observed_hash == expected_output_sha256
    output_magnitude_bucket: str | None = None
    output_length_bucket: str | None = None
    if result.ok and result.output_repr is not None:
        try:
            output_value = ast.literal_eval(result.output_repr)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            output_value = None
        if output_value is not None:
            if result.output_type in {"int", "float"}:
                output_magnitude_bucket = magnitude_bucket(output_value)
            elif result.output_type in {"str", "list", "tuple", "dict", "set", "bytes"}:
                output_length_bucket = length_bucket(output_value)

    return PackedExecutionRecord(
        source_dataset=submission.source_dataset,
        source_problem_id=submission.source_problem_id,
        source_submission_id=completion_id,
        input_id=input_id,
        split="train",
        code_tokens=tokenize_text(code),
        code_checksum=sha256_text(code),
        input_tokens=tokenize_text(input_repr),
        input_repr_checksum=sha256_text(input_repr),
        input_kind=input_kind,
        function_name=str(function_name) if function_name is not None else None,
        output_tokens=tokenize_text(output_repr),
        output_repr_checksum=sha256_text(output_repr),
        output_kind=result.output_kind,
        output_type=result.output_type,
        execution_status=exit_value,
        judge_verdict="pass" if completion_level_passed else "fail",
        wall_time_ms=float(result.wall_time_ms),
        peak_rss_kb=int(result.peak_rss_kb),
        determinism_check=True,
        license=submission.license,
        license_attribution_url=submission.license_attribution_url,
        held_out_for_eval=submission.held_out_for_eval,
        output_magnitude_bucket=output_magnitude_bucket,
        output_length_bucket=output_length_bucket,
        passed=passed,
    )


def _assign_splits(
    *,
    records: Sequence[PackedExecutionRecord],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> list[SplitName]:
    by_problem: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        by_problem.setdefault(record.source_problem_id, []).append(idx)
    problem_ids = sorted(by_problem)
    random.Random(seed).shuffle(problem_ids)
    n = len(problem_ids)
    if n == 1:
        train_ids = set(problem_ids)
        val_ids: set[str] = set()
    elif n == 2:
        train_ids = {problem_ids[0]}
        val_ids = set()
    else:
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train_ids = set(problem_ids[:n_train])
        val_ids = set(problem_ids[n_train : n_train + n_val])
    splits: list[SplitName] = ["test"] * len(records)
    for problem_id, idxs in by_problem.items():
        if problem_id in train_ids:
            split: SplitName = "train"
        elif problem_id in val_ids:
            split = "val"
        else:
            split = "test"
        for idx in idxs:
            splits[idx] = split
    return splits


def _attach_split(record: PackedExecutionRecord, split: SplitName) -> PackedExecutionRecord:
    from dataclasses import replace

    return replace(record, split=split)


def _split_pass_counts(records: Sequence[PackedExecutionRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for record in records:
        label = "true" if record.passed else "false"
        counts.setdefault(record.split, Counter())[label] += 1
    return {split: dict(sorted(counter.items())) for split, counter in sorted(counts.items())}


def _relative_secret_scan_payload(
    payload: dict[str, Any], root: Path
) -> dict[str, Any]:
    findings = []
    for finding in payload.get("findings", []):
        item = dict(finding)
        path = item.get("path")
        if isinstance(path, str):
            try:
                item["path"] = str(Path(path).resolve().relative_to(root.resolve()))
            except ValueError:
                item["path"] = Path(path).name
        findings.append(item)
    payload = dict(payload)
    payload["findings"] = findings
    return payload


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _validate_split_fracs(*, train_frac: float, val_frac: float) -> None:
    if not 0 < train_frac < 1:
        raise PassFailPackBuilderError(
            f"train_frac must be in (0, 1), got {train_frac}"
        )
    if not 0 < val_frac < 1:
        raise PassFailPackBuilderError(f"val_frac must be in (0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise PassFailPackBuilderError(
            f"train_frac + val_frac must be < 1, got {train_frac + val_frac}"
        )


def _positive_int(value: int, name: str) -> None:
    if int(value) <= 0:
        raise PassFailPackBuilderError(f"{name} must be positive")


def _default_passfail_pack_id() -> str:
    return "codelewm-passfail-execution-pack-" + datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
