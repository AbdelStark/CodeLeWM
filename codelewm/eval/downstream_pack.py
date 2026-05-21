"""Manifest-backed downstream candidate-reranking benchmark packs."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codelewm.observability import build_artifact_manifest, write_artifact_manifest
from codelewm.security.secret_scan import scan_paths

from .downstream import (
    DOWNSTREAM_MIN_LABELED_EXAMPLES,
    DOWNSTREAM_REQUIRED_BASELINES,
    DOWNSTREAM_REQUIRED_METRICS,
    DownstreamCandidate,
    DownstreamRerankBenchmark,
    DownstreamTask,
)


DOWNSTREAM_BENCHMARK_CONFIG_SCHEMA_VERSION = "codelewm.downstream_rerank_benchmark_config.v1"
DOWNSTREAM_BENCHMARK_PACK_RUN_SCHEMA_VERSION = "codelewm.downstream_benchmark_pack_run.v1"
DOWNSTREAM_BENCHMARK_READINESS_SCHEMA_VERSION = "codelewm.downstream_benchmark_readiness.v1"
DOWNSTREAM_SOURCE_LICENSE_POLICY_SCHEMA_VERSION = "codelewm.downstream_source_license_policy.v1"
DOWNSTREAM_SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION = "codelewm.downstream_split_leakage_report.v1"
DOWNSTREAM_BENCHMARK_FILENAME = "benchmark.json"
_STABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SECRET_SCAN_SUFFIXES = (".json", ".txt", ".md", ".py", ".patch", ".diff")


class DownstreamBenchmarkPackError(ValueError):
    """Raised when a downstream benchmark pack cannot be built."""


@dataclass(frozen=True)
class DownstreamBenchmarkPackResult:
    """Summary returned after writing a downstream benchmark pack."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    benchmark_path: str
    readiness_report_path: str
    source_license_policy_path: str
    split_leakage_report_path: str
    secret_scan_report_path: str
    example_count: int
    labeled_example_count: int
    scaled_evaluation_ready: bool
    downstream_claim_allowed: bool
    schema_version: str = DOWNSTREAM_BENCHMARK_PACK_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "benchmark_path": self.benchmark_path,
            "readiness_report_path": self.readiness_report_path,
            "source_license_policy_path": self.source_license_policy_path,
            "split_leakage_report_path": self.split_leakage_report_path,
            "secret_scan_report_path": self.secret_scan_report_path,
            "example_count": self.example_count,
            "labeled_example_count": self.labeled_example_count,
            "scaled_evaluation_ready": self.scaled_evaluation_ready,
            "downstream_claim_allowed": self.downstream_claim_allowed,
        }


@dataclass(frozen=True)
class DownstreamCandidateConfig:
    """Input config for one benchmark candidate."""

    candidate_id: str
    llm_rank: int
    label: str
    patch_path: str | None = None
    after_state_path: str | None = None
    static_check: str = "not_run"
    test_check: str = "not_run"
    source: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DownstreamCandidateConfig":
        _reject_unknown(
            payload,
            {
                "candidate_id",
                "llm_rank",
                "label",
                "patch_path",
                "after_state_path",
                "static_check",
                "test_check",
                "source",
                "provenance",
            },
            "downstream candidate config",
        )
        return cls(
            candidate_id=_require_stable_id(payload, "candidate_id", "candidate"),
            llm_rank=_require_positive_int(payload, "llm_rank", "candidate"),
            label=_require_string(payload, "label", "candidate"),
            patch_path=_optional_string(payload, "patch_path", "candidate"),
            after_state_path=_optional_string(payload, "after_state_path", "candidate"),
            static_check=_optional_string(payload, "static_check", "candidate", default="not_run"),
            test_check=_optional_string(payload, "test_check", "candidate", default="not_run"),
            source=_optional_mapping(payload, "source", "candidate"),
            provenance=_optional_mapping(payload, "provenance", "candidate"),
        )

    def __post_init__(self) -> None:
        if bool(self.patch_path) == bool(self.after_state_path):
            raise DownstreamBenchmarkPackError(
                f"candidate {self.candidate_id} must set exactly one of patch_path or after_state_path"
            )
        DownstreamCandidate(
            candidate_id=self.candidate_id,
            llm_rank=self.llm_rank,
            label=self.label,  # type: ignore[arg-type]
            patch_path=self.patch_path,
            after_state_path=self.after_state_path,
            static_check=self.static_check,  # type: ignore[arg-type]
            test_check=self.test_check,  # type: ignore[arg-type]
            source=self.source,
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class DownstreamTaskConfig:
    """Input config for one benchmark task."""

    task_id: str
    task_type: str
    prompt: str
    before_path: str
    split: str
    candidates: tuple[DownstreamCandidateConfig, ...]
    repo_id: str | None = None
    source: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DownstreamTaskConfig":
        _reject_unknown(
            payload,
            {
                "task_id",
                "task_type",
                "prompt",
                "before_path",
                "split",
                "repo_id",
                "source",
                "provenance",
                "candidates",
            },
            "downstream task config",
        )
        candidates = tuple(
            DownstreamCandidateConfig.from_mapping(_require_mapping_item(item, "candidates"))
            for item in _require_sequence(payload, "candidates", "task")
        )
        return cls(
            task_id=_require_stable_id(payload, "task_id", "task"),
            task_type=_require_string(payload, "task_type", "task"),
            prompt=_require_string(payload, "prompt", "task"),
            before_path=_require_relative_path(payload, "before_path", "task"),
            split=_optional_string(payload, "split", "task", default="test"),
            repo_id=_optional_string(payload, "repo_id", "task"),
            source=_optional_mapping(payload, "source", "task"),
            provenance=_optional_mapping(payload, "provenance", "task"),
            candidates=candidates,
        )

    def __post_init__(self) -> None:
        if len(self.candidates) < 2:
            raise DownstreamBenchmarkPackError(f"task {self.task_id} must have at least two candidates")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise DownstreamBenchmarkPackError(f"task {self.task_id} has duplicate candidate_id values")


@dataclass(frozen=True)
class DownstreamBenchmarkPackConfig:
    """JSON config for materializing a downstream benchmark pack."""

    benchmark_id: str
    source_license_policy: Mapping[str, Any]
    tasks: tuple[DownstreamTaskConfig, ...]
    evaluation_only: bool = True
    min_labeled_examples: int = DOWNSTREAM_MIN_LABELED_EXAMPLES
    required_baselines: tuple[str, ...] = DOWNSTREAM_REQUIRED_BASELINES
    required_metrics: tuple[str, ...] = DOWNSTREAM_REQUIRED_METRICS
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DOWNSTREAM_BENCHMARK_CONFIG_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DownstreamBenchmarkPackConfig":
        _reject_unknown(
            payload,
            {
                "schema_version",
                "benchmark_id",
                "source_license_policy",
                "tasks",
                "evaluation_only",
                "min_labeled_examples",
                "required_baselines",
                "required_metrics",
                "provenance",
            },
            "downstream benchmark pack config",
        )
        schema_version = _require_string(payload, "schema_version", "benchmark pack config")
        if schema_version != DOWNSTREAM_BENCHMARK_CONFIG_SCHEMA_VERSION:
            raise DownstreamBenchmarkPackError("unsupported downstream benchmark pack config schema_version")
        return cls(
            schema_version=schema_version,
            benchmark_id=_require_stable_id(payload, "benchmark_id", "benchmark pack config"),
            source_license_policy=_require_source_license_policy(
                _require_mapping(payload, "source_license_policy", "benchmark pack config")
            ),
            tasks=tuple(
                DownstreamTaskConfig.from_mapping(_require_mapping_item(item, "tasks"))
                for item in _require_sequence(payload, "tasks", "benchmark pack config")
            ),
            evaluation_only=_optional_bool(payload, "evaluation_only", default=True),
            min_labeled_examples=_optional_positive_int(
                payload,
                "min_labeled_examples",
                default=DOWNSTREAM_MIN_LABELED_EXAMPLES,
            ),
            required_baselines=_optional_string_tuple(
                payload,
                "required_baselines",
                default=DOWNSTREAM_REQUIRED_BASELINES,
            ),
            required_metrics=_optional_string_tuple(
                payload,
                "required_metrics",
                default=DOWNSTREAM_REQUIRED_METRICS,
            ),
            provenance=_optional_mapping(payload, "provenance", "benchmark pack config"),
        )

    def __post_init__(self) -> None:
        if not self.tasks:
            raise DownstreamBenchmarkPackError("benchmark pack config must include at least one task")
        if self.min_labeled_examples < DOWNSTREAM_MIN_LABELED_EXAMPLES:
            raise DownstreamBenchmarkPackError("min_labeled_examples must be at least 100")
        missing_baselines = set(DOWNSTREAM_REQUIRED_BASELINES) - set(self.required_baselines)
        if missing_baselines:
            raise DownstreamBenchmarkPackError(
                "required_baselines is missing: " + ", ".join(sorted(missing_baselines))
            )
        missing_metrics = set(DOWNSTREAM_REQUIRED_METRICS) - set(self.required_metrics)
        if missing_metrics:
            raise DownstreamBenchmarkPackError(
                "required_metrics is missing: " + ", ".join(sorted(missing_metrics))
            )
        if not self.source_license_policy.get("publication_allowed"):
            raise DownstreamBenchmarkPackError("source license policy must allow publication")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "source_license_policy": dict(self.source_license_policy),
            "tasks": [_task_config_to_dict(task) for task in self.tasks],
            "evaluation_only": self.evaluation_only,
            "min_labeled_examples": self.min_labeled_examples,
            "required_baselines": list(self.required_baselines),
            "required_metrics": list(self.required_metrics),
            "provenance": dict(self.provenance),
        }


def load_downstream_benchmark_pack_config(path: Path | str) -> DownstreamBenchmarkPackConfig:
    """Load a downstream benchmark pack config from JSON."""

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DownstreamBenchmarkPackError(f"config is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise DownstreamBenchmarkPackError("config root must be a JSON object")
    return DownstreamBenchmarkPackConfig.from_mapping(payload)


def build_downstream_benchmark_pack(
    *,
    config_path: Path | str,
    out: Path | str,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "downstream-pack"),
) -> DownstreamBenchmarkPackResult:
    """Build a self-contained benchmark pack without executing candidate code."""

    config_file = Path(config_path)
    config = load_downstream_benchmark_pack_config(config_file)
    output_dir = Path(out).resolve()
    manifest_path = output_dir / "manifest.json"
    benchmark_path = output_dir / DOWNSTREAM_BENCHMARK_FILENAME
    reports_dir = output_dir / "reports"
    tasks_dir = output_dir / "tasks"
    config_copy_path = output_dir / "config.json"
    if not overwrite and output_dir.exists() and any(output_dir.iterdir()):
        raise DownstreamBenchmarkPackError(
            f"output already exists; pass overwrite=True to replace: {output_dir}"
        )
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    config_copy_path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    materialized_tasks: list[DownstreamTask] = []
    artifact_files: list[Path] = [config_copy_path]
    for task in config.tasks:
        materialized_task, task_files = _materialize_task(task, config_file.parent, tasks_dir)
        materialized_tasks.append(materialized_task)
        artifact_files.extend(task_files)

    benchmark = DownstreamRerankBenchmark(
        benchmark_id=config.benchmark_id,
        tasks=tuple(materialized_tasks),
        required_baselines=config.required_baselines,
        required_metrics=config.required_metrics,
        min_labeled_examples=config.min_labeled_examples,
        provenance={
            **dict(config.provenance),
            "evaluation_only": config.evaluation_only,
            "source_license_policy": "reports/source_license_policy.json",
            "split_leakage_report": "reports/split_leakage_report.json",
        },
    )
    benchmark_path.write_text(
        json.dumps(benchmark.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_files.append(benchmark_path)

    source_license_policy_path = reports_dir / "source_license_policy.json"
    source_license_policy_path.write_text(
        json.dumps(config.source_license_policy, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    split_leakage_report = _build_split_leakage_report(config)
    split_leakage_report_path = reports_dir / "split_leakage_report.json"
    split_leakage_report_path.write_text(
        json.dumps(split_leakage_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    readiness_report = _build_readiness_report(config, split_leakage_report)
    readiness_report_path = reports_dir / "benchmark_readiness.json"
    readiness_report_path.write_text(
        json.dumps(readiness_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_files.extend((source_license_policy_path, split_leakage_report_path, readiness_report_path))

    scan_report = scan_paths([output_dir], include_suffixes=_SECRET_SCAN_SUFFIXES)
    secret_scan_payload = _relative_secret_scan_payload(scan_report.to_dict(), output_dir)
    secret_scan_report_path = reports_dir / "secret_scan_report.json"
    secret_scan_report_path.write_text(
        json.dumps(secret_scan_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if secret_scan_payload["findings"]:
        raise DownstreamBenchmarkPackError("secret scan found publish-blocking findings in benchmark pack")
    artifact_files.append(secret_scan_report_path)

    artifact_manifest = build_artifact_manifest(
        artifact_kind="downstream_benchmark",
        root=output_dir,
        files=artifact_files,
        command=command,
        config=config.to_dict(),
        metadata={
            "schema_version": benchmark.schema_version,
            "benchmark_id": benchmark.benchmark_id,
            "example_count": readiness_report["example_count"],
            "labeled_example_count": readiness_report["labeled_example_count"],
            "scaled_evaluation_ready": readiness_report["scaled_evaluation_ready"],
            "downstream_claim_allowed": readiness_report["downstream_claim_allowed"],
            "source_license_policy": "reports/source_license_policy.json",
            "split_leakage_report": "reports/split_leakage_report.json",
            "secret_scan_report": "reports/secret_scan_report.json",
        },
    )
    write_artifact_manifest(artifact_manifest, manifest_path)
    return DownstreamBenchmarkPackResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        benchmark_path=DOWNSTREAM_BENCHMARK_FILENAME,
        readiness_report_path="reports/benchmark_readiness.json",
        source_license_policy_path="reports/source_license_policy.json",
        split_leakage_report_path="reports/split_leakage_report.json",
        secret_scan_report_path="reports/secret_scan_report.json",
        example_count=int(readiness_report["example_count"]),
        labeled_example_count=int(readiness_report["labeled_example_count"]),
        scaled_evaluation_ready=bool(readiness_report["scaled_evaluation_ready"]),
        downstream_claim_allowed=bool(readiness_report["downstream_claim_allowed"]),
    )


def read_downstream_rerank_benchmark(path: Path | str) -> DownstreamRerankBenchmark:
    """Read a materialized downstream benchmark payload."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DownstreamBenchmarkPackError("downstream benchmark payload must be a JSON object")
    tasks = []
    for task_payload in _require_sequence(payload, "tasks", "downstream benchmark"):
        task_mapping = _require_mapping_item(task_payload, "tasks")
        candidates = tuple(
            DownstreamCandidate(
                candidate_id=_require_string(candidate_payload, "candidate_id", "candidate"),
                llm_rank=_require_positive_int(candidate_payload, "llm_rank", "candidate"),
                label=_require_string(candidate_payload, "label", "candidate"),  # type: ignore[arg-type]
                patch_path=_optional_string(candidate_payload, "patch_path", "candidate"),
                after_state_path=_optional_string(candidate_payload, "after_state_path", "candidate"),
                static_check=_optional_string(  # type: ignore[arg-type]
                    candidate_payload,
                    "static_check",
                    "candidate",
                    default="not_run",
                ),
                test_check=_optional_string(  # type: ignore[arg-type]
                    candidate_payload,
                    "test_check",
                    "candidate",
                    default="not_run",
                ),
                source=_optional_mapping(candidate_payload, "source", "candidate"),
                provenance=_optional_mapping(candidate_payload, "provenance", "candidate"),
            )
            for candidate_payload in (
                _require_mapping_item(item, "candidates")
                for item in _require_sequence(task_mapping, "candidates", "task")
            )
        )
        tasks.append(
            DownstreamTask(
                task_id=_require_string(task_mapping, "task_id", "task"),
                task_type=_require_string(task_mapping, "task_type", "task"),
                prompt=_require_string(task_mapping, "prompt", "task"),
                before_path=_require_string(task_mapping, "before_path", "task"),
                candidates=candidates,
                provenance=_optional_mapping(task_mapping, "provenance", "task"),
            )
        )
    return DownstreamRerankBenchmark(
        benchmark_id=_require_string(payload, "benchmark_id", "downstream benchmark"),
        tasks=tuple(tasks),
        required_baselines=tuple(
            _require_string_item(item, "required_baselines")
            for item in _require_sequence(payload, "required_baselines", "downstream benchmark")
        ),
        required_metrics=tuple(
            _require_string_item(item, "required_metrics")
            for item in _require_sequence(payload, "required_metrics", "downstream benchmark")
        ),
        min_labeled_examples=_require_positive_int(payload, "min_labeled_examples", "downstream benchmark"),
        provenance=_optional_mapping(payload, "provenance", "downstream benchmark"),
        schema_version=_require_string(payload, "schema_version", "downstream benchmark"),
    )


def _materialize_task(
    task: DownstreamTaskConfig,
    config_root: Path,
    tasks_dir: Path,
) -> tuple[DownstreamTask, tuple[Path, ...]]:
    task_dir = tasks_dir / task.task_id
    candidates_dir = task_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    before_source = _resolve_config_path(config_root, task.before_path, field_name="before_path")
    before_dest = task_dir / "before.py"
    shutil.copyfile(before_source, before_dest)
    files = [before_dest]
    materialized_candidates = []
    for candidate in task.candidates:
        source_path_value = candidate.patch_path or candidate.after_state_path
        assert source_path_value is not None
        source_path = _resolve_config_path(config_root, source_path_value, field_name="candidate path")
        suffix = ".patch" if candidate.patch_path is not None else ".py"
        candidate_dest = candidates_dir / f"{candidate.candidate_id}{suffix}"
        shutil.copyfile(source_path, candidate_dest)
        files.append(candidate_dest)
        candidate_relative = _relative_to_root(candidate_dest, tasks_dir.parent)
        materialized_candidates.append(
            DownstreamCandidate(
                candidate_id=candidate.candidate_id,
                llm_rank=candidate.llm_rank,
                label=candidate.label,  # type: ignore[arg-type]
                patch_path=candidate_relative if candidate.patch_path is not None else None,
                after_state_path=candidate_relative if candidate.after_state_path is not None else None,
                static_check=candidate.static_check,  # type: ignore[arg-type]
                test_check=candidate.test_check,  # type: ignore[arg-type]
                source=candidate.source,
                provenance={
                    **dict(candidate.provenance),
                    "input_path": source_path_value,
                },
            )
        )
    materialized_task = DownstreamTask(
        task_id=task.task_id,
        task_type=task.task_type,
        prompt=task.prompt,
        before_path=_relative_to_root(before_dest, tasks_dir.parent),
        candidates=tuple(materialized_candidates),
        provenance={
            **dict(task.provenance),
            "split": task.split,
            "repo_id": task.repo_id,
            "source": dict(task.source),
            "input_before_path": task.before_path,
        },
    )
    return materialized_task, tuple(files)


def _build_split_leakage_report(config: DownstreamBenchmarkPackConfig) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    task_splits: dict[str, set[str]] = {}
    repo_splits: dict[str, set[str]] = {}
    for task in config.tasks:
        split_counts[task.split] = split_counts.get(task.split, 0) + 1
        task_splits.setdefault(task.task_id, set()).add(task.split)
        if task.repo_id:
            repo_splits.setdefault(task.repo_id, set()).add(task.split)
    leakage_findings: list[dict[str, Any]] = []
    if not config.evaluation_only:
        for task_id, splits in sorted(task_splits.items()):
            if len(splits) > 1:
                leakage_findings.append({"kind": "task_id", "id": task_id, "splits": sorted(splits)})
        for repo_id, splits in sorted(repo_splits.items()):
            if len(splits) > 1:
                leakage_findings.append({"kind": "repo_id", "id": repo_id, "splits": sorted(splits)})
    return {
        "schema_version": DOWNSTREAM_SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION,
        "ok": not leakage_findings,
        "evaluation_only": config.evaluation_only,
        "split_counts": split_counts,
        "checked_keys": ["task_id", "repo_id"],
        "leakage_findings": leakage_findings,
        "notes": [
            "evaluation_only packs do not define train/validation/test reuse",
        ]
        if config.evaluation_only
        else [],
    }


def _build_readiness_report(
    config: DownstreamBenchmarkPackConfig,
    split_leakage_report: Mapping[str, Any],
) -> dict[str, Any]:
    labeled_count = sum(
        1 for task in config.tasks if any(candidate.label in {"pass", "fail"} for candidate in task.candidates)
    )
    candidate_count = sum(len(task.candidates) for task in config.tasks)
    blocked_reasons: list[str] = []
    if labeled_count < config.min_labeled_examples:
        blocked_reasons.append(
            f"labeled_example_count_below_minimum:{labeled_count}<{config.min_labeled_examples}"
        )
    if not split_leakage_report.get("ok"):
        blocked_reasons.append("split_or_repository_leakage_detected")
    if not config.source_license_policy.get("publication_allowed"):
        blocked_reasons.append("source_license_policy_blocks_publication")
    scaled_evaluation_ready = not blocked_reasons
    if scaled_evaluation_ready:
        blocked_reasons.append("downstream_evaluation_not_run")
    return {
        "schema_version": DOWNSTREAM_BENCHMARK_READINESS_SCHEMA_VERSION,
        "benchmark_id": config.benchmark_id,
        "example_count": len(config.tasks),
        "candidate_count": candidate_count,
        "labeled_example_count": labeled_count,
        "min_labeled_examples": config.min_labeled_examples,
        "scaled_evaluation_ready": scaled_evaluation_ready,
        "downstream_claim_allowed": False,
        "blocked_reasons": blocked_reasons,
    }


def _relative_secret_scan_payload(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "ok": payload["ok"],
        "paths_scanned": [
            _relative_to_root(Path(path), root) for path in payload.get("paths_scanned", [])
        ],
        "findings": [
            {
                **dict(finding),
                "path": _relative_to_root(Path(str(finding["path"])), root),
            }
            for finding in payload.get("findings", [])
        ],
    }


def _require_source_license_policy(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = dict(payload)
    schema_version = policy.get("schema_version")
    if schema_version != DOWNSTREAM_SOURCE_LICENSE_POLICY_SCHEMA_VERSION:
        raise DownstreamBenchmarkPackError("unsupported source license policy schema_version")
    if not isinstance(policy.get("publication_allowed"), bool):
        raise DownstreamBenchmarkPackError("source license policy publication_allowed must be boolean")
    if not policy.get("source_kind"):
        raise DownstreamBenchmarkPackError("source license policy source_kind is required")
    if not policy.get("license"):
        raise DownstreamBenchmarkPackError("source license policy license is required")
    _ensure_json_native(policy, "source_license_policy")
    return policy


def _task_config_to_dict(task: DownstreamTaskConfig) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "prompt": task.prompt,
        "before_path": task.before_path,
        "split": task.split,
        "repo_id": task.repo_id,
        "source": dict(task.source),
        "provenance": dict(task.provenance),
        "candidates": [_candidate_config_to_dict(candidate) for candidate in task.candidates],
    }


def _candidate_config_to_dict(candidate: DownstreamCandidateConfig) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "llm_rank": candidate.llm_rank,
        "label": candidate.label,
        "patch_path": candidate.patch_path,
        "after_state_path": candidate.after_state_path,
        "static_check": candidate.static_check,
        "test_check": candidate.test_check,
        "source": dict(candidate.source),
        "provenance": dict(candidate.provenance),
    }


def _resolve_config_path(config_root: Path, value: str, *, field_name: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise DownstreamBenchmarkPackError(f"{field_name} must be relative to the config file")
    resolved = (config_root / path).resolve()
    if not resolved.is_file():
        raise DownstreamBenchmarkPackError(f"{field_name} does not exist: {value}")
    return resolved


def _require_relative_path(payload: Mapping[str, Any], key: str, section: str) -> str:
    value = _require_string(payload, key, section)
    if Path(value).is_absolute():
        raise DownstreamBenchmarkPackError(f"{section}.{key} must be relative")
    return value


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DownstreamBenchmarkPackError(f"path escapes benchmark artifact root: {path}") from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], section: str) -> None:
    extra = sorted(set(payload) - allowed)
    if extra:
        raise DownstreamBenchmarkPackError(f"{section} contains unknown key(s): {', '.join(extra)}")


def _require_mapping(payload: Mapping[str, Any], key: str, section: str) -> Mapping[str, Any]:
    if key not in payload or not isinstance(payload[key], Mapping):
        raise DownstreamBenchmarkPackError(f"{section}.{key} must be a JSON object")
    return payload[key]  # type: ignore[return-value]


def _optional_mapping(payload: Mapping[str, Any], key: str, section: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise DownstreamBenchmarkPackError(f"{section}.{key} must be a JSON object")
    _ensure_json_native(value, f"{section}.{key}")
    return dict(value)


def _require_sequence(payload: Mapping[str, Any], key: str, section: str) -> Sequence[Any]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DownstreamBenchmarkPackError(f"{section}.{key} must be a JSON array")
    return value


def _require_mapping_item(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DownstreamBenchmarkPackError(f"{section} entries must be JSON objects")
    return value


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise DownstreamBenchmarkPackError(f"{section}.{key} must be a non-empty string")
    return value


def _require_string_item(value: Any, section: str) -> str:
    if not isinstance(value, str) or not value:
        raise DownstreamBenchmarkPackError(f"{section} entries must be non-empty strings")
    return value


def _optional_string(
    payload: Mapping[str, Any],
    key: str,
    section: str,
    *,
    default: str | None = None,
) -> str | None:
    value = payload.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DownstreamBenchmarkPackError(f"{section}.{key} must be a non-empty string")
    return value


def _require_stable_id(payload: Mapping[str, Any], key: str, section: str) -> str:
    value = _require_string(payload, key, section)
    if not _STABLE_ID_RE.fullmatch(value):
        raise DownstreamBenchmarkPackError(f"{section}.{key} must be a stable artifact id")
    return value


def _require_positive_int(payload: Mapping[str, Any], key: str, section: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DownstreamBenchmarkPackError(f"{section}.{key} must be a positive integer")
    return value


def _optional_positive_int(payload: Mapping[str, Any], key: str, *, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DownstreamBenchmarkPackError(f"{key} must be a positive integer")
    return value


def _optional_string_tuple(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = payload.get(key, default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DownstreamBenchmarkPackError(f"{key} must be a JSON array")
    return tuple(_require_string_item(item, key) for item in value)


def _optional_bool(payload: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise DownstreamBenchmarkPackError(f"{key} must be a boolean")
    return value


def _ensure_json_native(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DownstreamBenchmarkPackError(f"{field_name} must be JSON-native: {exc}") from exc
