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
from .downstream_anti_saturation import (
    ANTI_SATURATION_PROFILE,
    MAX_CANDIDATES_PER_PROBLEM,
    MIN_CANDIDATES_PER_PROBLEM,
    DownstreamAntiSaturationError,
    build_anti_saturation_report,
    compute_model_independent_baselines,
    validate_hard_negative_class,
)
from .hard_negative_pool import (
    LABEL_CONSTRUCTION_REPORT_SCHEMA_VERSION,
    build_label_construction_report,
    generate_hard_negative_pool,
)
from .llm_candidate_ingest import (
    LLMCandidateIngestResult,
    build_llm_candidate_ingest_report,
    ingest_llm_candidate_pack,
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
    anti_saturation_report_path: str | None = None
    anti_saturation_eligible: bool | None = None
    label_construction_report_path: str | None = None
    llm_candidate_ingest_report_path: str | None = None
    ingested_llm_candidate_count: int = 0
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
            "anti_saturation_report_path": self.anti_saturation_report_path,
            "anti_saturation_eligible": self.anti_saturation_eligible,
            "label_construction_report_path": self.label_construction_report_path,
            "llm_candidate_ingest_report_path": self.llm_candidate_ingest_report_path,
            "ingested_llm_candidate_count": self.ingested_llm_candidate_count,
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
    hard_negative_class: str | None = None
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
                "hard_negative_class",
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
            hard_negative_class=_optional_string(payload, "hard_negative_class", "candidate"),
            source=_optional_mapping(payload, "source", "candidate"),
            provenance=_optional_mapping(payload, "provenance", "candidate"),
        )

    def __post_init__(self) -> None:
        if bool(self.patch_path) == bool(self.after_state_path):
            raise DownstreamBenchmarkPackError(
                f"candidate {self.candidate_id} must set exactly one of patch_path or after_state_path"
            )
        if self.hard_negative_class is not None:
            try:
                validate_hard_negative_class(self.hard_negative_class)
            except DownstreamAntiSaturationError as exc:
                raise DownstreamBenchmarkPackError(str(exc)) from exc
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
    generated_pool: Mapping[str, Any] | None = None
    llm_candidate_packs: tuple[str, ...] = ()
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
                "generated_pool",
                "llm_candidate_packs",
                "source",
                "provenance",
                "candidates",
            },
            "downstream task config",
        )
        candidates = tuple(
            DownstreamCandidateConfig.from_mapping(_require_mapping_item(item, "candidates"))
            for item in payload.get("candidates", ())
        )
        return cls(
            task_id=_require_stable_id(payload, "task_id", "task"),
            task_type=_require_string(payload, "task_type", "task"),
            prompt=_require_string(payload, "prompt", "task"),
            before_path=_require_relative_path(payload, "before_path", "task"),
            split=_optional_string(payload, "split", "task", default="test"),
            repo_id=_optional_string(payload, "repo_id", "task"),
            generated_pool=_parse_generated_pool(payload),
            llm_candidate_packs=_parse_llm_candidate_packs(payload),
            source=_optional_mapping(payload, "source", "task"),
            provenance=_optional_mapping(payload, "provenance", "task"),
            candidates=candidates,
        )

    def __post_init__(self) -> None:
        if (
            self.generated_pool is None
            and not self.llm_candidate_packs
            and len(self.candidates) < 2
        ):
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
    profile: str | None = None
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
                "profile",
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
            profile=_optional_string(payload, "profile", "benchmark pack config"),
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
        if self.profile is not None and self.profile != ANTI_SATURATION_PROFILE:
            raise DownstreamBenchmarkPackError(
                f"unsupported benchmark profile: {self.profile!r} (expected {ANTI_SATURATION_PROFILE!r})"
            )

    @property
    def is_anti_saturation(self) -> bool:
        return self.profile == ANTI_SATURATION_PROFILE

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if self.profile is not None:
            payload["profile"] = self.profile
        return payload


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
    generated_candidates_all: list[Any] = []
    ingest_results_all: list[LLMCandidateIngestResult] = []
    for task in config.tasks:
        materialized_task, task_files, generated, ingest_results = _materialize_task(
            task, config_file.parent, tasks_dir
        )
        materialized_tasks.append(materialized_task)
        artifact_files.extend(task_files)
        generated_candidates_all.extend(generated)
        ingest_results_all.extend(ingest_results)

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

    anti_saturation_report: dict[str, Any] | None = None
    anti_saturation_report_path_rel: str | None = None
    if config.is_anti_saturation:
        baseline_inputs = compute_model_independent_baselines(benchmark, root=output_dir)
        anti_saturation_report = build_anti_saturation_report(
            profile=config.profile or ANTI_SATURATION_PROFILE,
            source_license_ok=bool(config.source_license_policy.get("publication_allowed")),
            split_leakage_ok=bool(split_leakage_report["ok"]),
            # The secret-scan and manifest gates are enforced below: a produced
            # pack always has both passing because the build aborts on findings.
            manifest_ok=True,
            secret_scan_ok=True,
            **baseline_inputs,
        )
        anti_saturation_report_file = reports_dir / "anti_saturation_report.json"
        anti_saturation_report_file.write_text(
            json.dumps(anti_saturation_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        anti_saturation_report_path_rel = "reports/anti_saturation_report.json"

    label_construction_report_path_rel: str | None = None
    if generated_candidates_all:
        label_construction_report = build_label_construction_report(
            generated_candidates_all, sandbox_used=False
        )
        label_construction_report_file = reports_dir / "label_construction_report.json"
        label_construction_report_file.write_text(
            json.dumps(label_construction_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        artifact_files.append(label_construction_report_file)
        label_construction_report_path_rel = "reports/label_construction_report.json"

    llm_candidate_ingest_report_path_rel: str | None = None
    llm_candidate_ingest_report: dict[str, Any] | None = None
    llm_pack_parent_artifacts: tuple[str, ...] = ()
    if ingest_results_all:
        llm_candidate_ingest_report = build_llm_candidate_ingest_report(ingest_results_all)
        llm_candidate_ingest_report_file = reports_dir / "llm_candidate_ingest_report.json"
        llm_candidate_ingest_report_file.write_text(
            json.dumps(llm_candidate_ingest_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        artifact_files.append(llm_candidate_ingest_report_file)
        llm_candidate_ingest_report_path_rel = "reports/llm_candidate_ingest_report.json"
        llm_pack_parent_artifacts = tuple(llm_candidate_ingest_report["source_manifest_ids"])

    readiness_report = _build_readiness_report(
        config, split_leakage_report, anti_saturation_report=anti_saturation_report
    )
    readiness_report_path = reports_dir / "benchmark_readiness.json"
    readiness_report_path.write_text(
        json.dumps(readiness_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_files.extend((source_license_policy_path, split_leakage_report_path, readiness_report_path))
    if anti_saturation_report is not None:
        artifact_files.append(anti_saturation_report_file)

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
        parent_artifacts=llm_pack_parent_artifacts,
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
            **(
                {
                    "profile": config.profile,
                    "anti_saturation_report": anti_saturation_report_path_rel,
                    "anti_saturation_eligible": anti_saturation_report["eligible"],
                }
                if anti_saturation_report is not None
                else {}
            ),
            **(
                {"label_construction_report": label_construction_report_path_rel}
                if label_construction_report_path_rel is not None
                else {}
            ),
            **(
                {
                    "llm_candidate_ingest_report": llm_candidate_ingest_report_path_rel,
                    "llm_candidate_ingest_ok": llm_candidate_ingest_report["ok"],
                    "ingested_llm_candidate_count": llm_candidate_ingest_report[
                        "ingested_candidate_count"
                    ],
                }
                if llm_candidate_ingest_report is not None
                else {}
            ),
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
        anti_saturation_report_path=anti_saturation_report_path_rel,
        anti_saturation_eligible=(
            None if anti_saturation_report is None else bool(anti_saturation_report["eligible"])
        ),
        label_construction_report_path=label_construction_report_path_rel,
        llm_candidate_ingest_report_path=llm_candidate_ingest_report_path_rel,
        ingested_llm_candidate_count=(
            0
            if llm_candidate_ingest_report is None
            else int(llm_candidate_ingest_report["ingested_candidate_count"])
        ),
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
) -> tuple[DownstreamTask, tuple[Path, ...], tuple[Any, ...]]:
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
        candidate_source = dict(candidate.source)
        if candidate.hard_negative_class is not None:
            candidate_source["hard_negative_class"] = candidate.hard_negative_class
        materialized_candidates.append(
            DownstreamCandidate(
                candidate_id=candidate.candidate_id,
                llm_rank=candidate.llm_rank,
                label=candidate.label,  # type: ignore[arg-type]
                patch_path=candidate_relative if candidate.patch_path is not None else None,
                after_state_path=candidate_relative if candidate.after_state_path is not None else None,
                static_check=candidate.static_check,  # type: ignore[arg-type]
                test_check=candidate.test_check,  # type: ignore[arg-type]
                source=candidate_source,
                provenance={
                    **dict(candidate.provenance),
                    "input_path": source_path_value,
                },
            )
        )

    generated_candidates: tuple[Any, ...] = ()
    if task.generated_pool is not None:
        generated_candidates = _materialize_generated_pool(
            task,
            config_root=config_root,
            tasks_dir=tasks_dir,
            candidates_dir=candidates_dir,
            before_text=before_source.read_text(encoding="utf-8"),
            base_llm_rank=len(materialized_candidates),
            files=files,
            materialized_candidates=materialized_candidates,
        )

    ingest_results: tuple[LLMCandidateIngestResult, ...] = ()
    if task.llm_candidate_packs:
        ingest_results = _materialize_llm_candidate_packs(
            task,
            config_root=config_root,
            tasks_dir=tasks_dir,
            candidates_dir=candidates_dir,
            base_llm_rank=len(materialized_candidates),
            files=files,
            materialized_candidates=materialized_candidates,
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
            **(
                {"generated_pool": dict(task.generated_pool)}
                if task.generated_pool is not None
                else {}
            ),
            **(
                {"llm_candidate_packs": list(task.llm_candidate_packs)}
                if task.llm_candidate_packs
                else {}
            ),
        },
    )
    return materialized_task, tuple(files), generated_candidates, ingest_results


def _materialize_llm_candidate_packs(
    task: DownstreamTaskConfig,
    *,
    config_root: Path,
    tasks_dir: Path,
    candidates_dir: Path,
    base_llm_rank: int,
    files: list[Path],
    materialized_candidates: list[DownstreamCandidate],
) -> tuple[LLMCandidateIngestResult, ...]:
    results: list[LLMCandidateIngestResult] = []
    existing_ids = {candidate.candidate_id for candidate in materialized_candidates}
    next_rank = base_llm_rank
    for pack_path in task.llm_candidate_packs:
        manifest = _resolve_config_path(
            config_root, pack_path, field_name="llm_candidate_pack manifest"
        )
        result = ingest_llm_candidate_pack(manifest, base_llm_rank=next_rank)
        results.append(result)
        for ingested in result.candidates:
            if ingested.candidate_id in existing_ids:
                raise DownstreamBenchmarkPackError(
                    f"task {task.task_id} ingested candidate id collides: {ingested.candidate_id}"
                )
            existing_ids.add(ingested.candidate_id)
            next_rank += 1
            candidate_dest = candidates_dir / f"{ingested.candidate_id}.patch"
            candidate_dest.write_text(ingested.patch_text, encoding="utf-8")
            files.append(candidate_dest)
            candidate_relative = _relative_to_root(candidate_dest, tasks_dir.parent)
            materialized_candidates.append(
                DownstreamCandidate(
                    candidate_id=ingested.candidate_id,
                    llm_rank=next_rank,
                    label=ingested.label,  # type: ignore[arg-type]
                    patch_path=candidate_relative,
                    after_state_path=None,
                    static_check="not_run",
                    test_check="not_run",
                    source=dict(ingested.source),
                    provenance=dict(ingested.provenance),
                )
            )
    return tuple(results)


def _materialize_generated_pool(
    task: DownstreamTaskConfig,
    *,
    config_root: Path,
    tasks_dir: Path,
    candidates_dir: Path,
    before_text: str,
    base_llm_rank: int,
    files: list[Path],
    materialized_candidates: list[DownstreamCandidate],
) -> tuple[Any, ...]:
    assert task.generated_pool is not None
    spec = task.generated_pool
    reference_source = _resolve_config_path(
        config_root, str(spec["reference_after_path"]), field_name="generated_pool.reference_after_path"
    )
    reference_after_text = reference_source.read_text(encoding="utf-8")
    pool = generate_hard_negative_pool(
        before_text=before_text,
        reference_after_text=reference_after_text,
        seed=int(spec["seed"]),
        pool_size=int(spec["pool_size"]),
    )
    existing_ids = {candidate.candidate_id for candidate in materialized_candidates}
    license_status = task.source.get("license_policy_id", "inherited_from_task_policy")
    for offset, generated in enumerate(pool):
        if generated.candidate_id in existing_ids:
            raise DownstreamBenchmarkPackError(
                f"task {task.task_id} generated candidate id collides with an existing id: "
                f"{generated.candidate_id}"
            )
        existing_ids.add(generated.candidate_id)
        candidate_dest = candidates_dir / f"{generated.candidate_id}.py"
        candidate_dest.write_text(generated.after_text, encoding="utf-8")
        files.append(candidate_dest)
        candidate_relative = _relative_to_root(candidate_dest, tasks_dir.parent)
        materialized_candidates.append(
            DownstreamCandidate(
                candidate_id=generated.candidate_id,
                llm_rank=base_llm_rank + offset + 1,
                label=generated.label,  # type: ignore[arg-type]
                patch_path=None,
                after_state_path=candidate_relative,
                static_check=generated.static_check,  # type: ignore[arg-type]
                test_check="not_run",
                source={
                    "hard_negative_class": generated.hard_negative_class,
                    "candidate_kind": generated.hard_negative_class,
                    "generator": "hard_negative_pool",
                    "checksum": generated.checksum,
                    "label_source": str(generated.provenance.get("label_source", "unverified")),
                    "source_license_status": str(license_status),
                },
                provenance={
                    **dict(generated.provenance),
                    "generated": True,
                    "reference_after_path": str(spec["reference_after_path"]),
                    "seed": int(spec["seed"]),
                },
            )
        )
    return pool


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
    *,
    anti_saturation_report: Mapping[str, Any] | None = None,
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
    if anti_saturation_report is not None and not anti_saturation_report.get("eligible"):
        blocked_reasons.append("anti_saturation_slice_not_eligible")
    scaled_evaluation_ready = not blocked_reasons
    if scaled_evaluation_ready:
        blocked_reasons.append("downstream_evaluation_not_run")
    report: dict[str, Any] = {
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
    if config.profile is not None:
        report["profile"] = config.profile
    if anti_saturation_report is not None:
        report["anti_saturation_eligible"] = bool(anti_saturation_report.get("eligible"))
        report["anti_saturation_blocked_reasons"] = list(
            anti_saturation_report.get("blocked_reasons", [])
        )
    return report


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
    payload: dict[str, Any] = {
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
    if task.generated_pool is not None:
        payload["generated_pool"] = dict(task.generated_pool)
    if task.llm_candidate_packs:
        payload["llm_candidate_packs"] = list(task.llm_candidate_packs)
    return payload


def _parse_generated_pool(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = payload.get("generated_pool")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DownstreamBenchmarkPackError("task.generated_pool must be a JSON object")
    _reject_unknown(
        value, {"reference_after_path", "seed", "pool_size"}, "task generated_pool"
    )
    reference_after_path = _require_relative_path(value, "reference_after_path", "generated_pool")
    seed = value.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DownstreamBenchmarkPackError("generated_pool.seed must be a non-negative integer")
    pool_size = _optional_positive_int(
        value, "pool_size", default=MIN_CANDIDATES_PER_PROBLEM
    )
    if not (MIN_CANDIDATES_PER_PROBLEM <= pool_size <= MAX_CANDIDATES_PER_PROBLEM):
        raise DownstreamBenchmarkPackError(
            "generated_pool.pool_size must be in "
            f"{MIN_CANDIDATES_PER_PROBLEM}-{MAX_CANDIDATES_PER_PROBLEM}"
        )
    return {
        "reference_after_path": reference_after_path,
        "seed": seed,
        "pool_size": pool_size,
    }


def _parse_llm_candidate_packs(payload: Mapping[str, Any]) -> tuple[str, ...]:
    value = payload.get("llm_candidate_packs", ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DownstreamBenchmarkPackError("task.llm_candidate_packs must be a JSON array")
    packs: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise DownstreamBenchmarkPackError(
                "task.llm_candidate_packs entries must be non-empty relative paths"
            )
        if Path(item).is_absolute():
            raise DownstreamBenchmarkPackError(
                "task.llm_candidate_packs entries must be relative to the config file"
            )
        packs.append(item)
    return tuple(packs)


def _candidate_config_to_dict(candidate: DownstreamCandidateConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    if candidate.hard_negative_class is not None:
        payload["hard_negative_class"] = candidate.hard_negative_class
    return payload


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
