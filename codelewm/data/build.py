"""Dataset build pipeline and CLI-facing contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from codelewm.data.actions import ActionExtractionConfig, extract_edit_action
from codelewm.data.codestate import CodeStateConfig, extract_codestate_pair
from codelewm.data.filters import FilterPolicy, filter_raw_edit_records
from codelewm.data.masks import build_masked_codestate, stable_token_id
from codelewm.data.normalize import CodeStateNormalizationConfig
from codelewm.data.pack import (
    DATASET_SCHEMA_VERSION,
    ArtifactInfo,
    DatasetManifest,
    PackSpec,
    PackedTransition,
    TokenSequence,
    build_dataset_manifest,
    sha256_file,
    write_dataset_manifest,
)
from codelewm.data.sources import (
    AdapterKind,
    RawEditRecord,
    SourceRecordError,
    SourceSpec,
    SourceUnavailableError,
    load_source,
)
from codelewm.data.split_dedup import DedupPolicy, SplitPolicy, record_id, split_and_deduplicate
from codelewm.observability import ArtifactManifest, build_artifact_manifest, write_artifact_manifest
from codelewm.security import (
    SourceLicensePolicy,
    reject_code_execution_config,
)
from codelewm.security.license_policy import ArtifactPolicy
from codelewm.security.non_execution import NonExecutionPolicyError


DATASET_BUILD_CONFIG_SCHEMA_VERSION = "codelewm.dataset_build_config.v1"
DATASET_BUILD_REPORT_SCHEMA_VERSION = "codelewm.dataset_build_report.v1"
ROW_COUNTS_REPORT_SCHEMA_VERSION = "codelewm.dataset_row_counts.v1"
_SUPPORTED_SOURCES = {"fixture", "commitpackft"}
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class DatasetBuildConfigError(ValueError):
    """Raised when a dataset build config is malformed."""


class DatasetBuildError(RuntimeError):
    """Raised when a dataset artifact cannot be built from valid inputs."""


@dataclass(frozen=True)
class DatasetSourceConfig:
    """One raw edit source in a dataset build config."""

    source: AdapterKind
    path: str | None = None
    name: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetSourceConfig":
        _reject_unknown(payload, {"source", "path", "name", "options"}, "sources[]")
        source = _require_string(payload, "source", "sources[]")
        if source not in _SUPPORTED_SOURCES:
            allowed = ", ".join(sorted(_SUPPORTED_SOURCES))
            raise DatasetBuildConfigError(f"sources[].source must be one of: {allowed}")
        options = _optional_mapping(payload, "options", "sources[]", default={})
        _ensure_json_native(options, section="sources[].options")
        return cls(
            source=cast(AdapterKind, source),
            path=_optional_string(payload, "path", "sources[]"),
            name=_optional_string(payload, "name", "sources[]"),
            options=dict(options),
        )

    def to_source_spec(self, *, config_dir: Path) -> SourceSpec:
        path = None if self.path is None else _resolve_config_path(self.path, config_dir=config_dir)
        return SourceSpec(source=self.source, path=path, name=self.name, options=dict(self.options))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "options": dict(self.options),
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class DatasetLicenseConfig:
    """License gate settings for public dataset artifacts."""

    allowed_licenses: tuple[str, ...] = (
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "isc",
        "mit",
        "unlicense",
    )
    require_license_field: bool = True
    redistribution_allowed: bool = True
    artifact_policy: ArtifactPolicy = "full_text"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetLicenseConfig":
        _reject_unknown(
            payload,
            {
                "allowed_licenses",
                "require_license_field",
                "redistribution_allowed",
                "artifact_policy",
            },
            "license",
        )
        artifact_policy = _optional_string(payload, "artifact_policy", "license") or "full_text"
        if artifact_policy not in {"exclude", "metadata_only", "embeddings", "full_text"}:
            raise DatasetBuildConfigError(
                "license.artifact_policy must be one of: exclude, metadata_only, embeddings, full_text"
            )
        return cls(
            allowed_licenses=_optional_string_tuple(
                payload,
                "allowed_licenses",
                "license",
                default=cls.allowed_licenses,
            ),
            require_license_field=_optional_bool(
                payload,
                "require_license_field",
                "license",
                default=True,
            ),
            redistribution_allowed=_optional_bool(
                payload,
                "redistribution_allowed",
                "license",
                default=True,
            ),
            artifact_policy=cast(ArtifactPolicy, artifact_policy),
        )

    def to_policy(self) -> SourceLicensePolicy:
        return SourceLicensePolicy(
            allowed_licenses=self.allowed_licenses,
            require_license_field=self.require_license_field,
            redistribution_allowed=self.redistribution_allowed,
            derived_artifact_policy=self.artifact_policy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_licenses": list(self.allowed_licenses),
            "require_license_field": self.require_license_field,
            "redistribution_allowed": self.redistribution_allowed,
            "artifact_policy": self.artifact_policy,
        }


@dataclass(frozen=True)
class DatasetBuildConfig:
    """Validated config for `codelewm dataset build`."""

    schema_version: str
    name: str
    sources: tuple[DatasetSourceConfig, ...]
    seed: int = 0
    filter: FilterPolicy = FilterPolicy()
    split: SplitPolicy = SplitPolicy()
    dedup: DedupPolicy = DedupPolicy()
    codestate: CodeStateConfig = CodeStateConfig()
    normalize: CodeStateNormalizationConfig = CodeStateNormalizationConfig()
    action: ActionExtractionConfig = ActionExtractionConfig()
    license: DatasetLicenseConfig = DatasetLicenseConfig()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetBuildConfig":
        try:
            reject_code_execution_config(payload, context="dataset build config")
        except NonExecutionPolicyError as exc:
            raise DatasetBuildConfigError(str(exc)) from exc

        _reject_unknown(
            payload,
            {
                "schema_version",
                "name",
                "seed",
                "sources",
                "filter",
                "split",
                "dedup",
                "codestate",
                "normalize",
                "action",
                "license",
            },
            "dataset build config",
        )
        seed = _optional_int(payload, "seed", "dataset build config", default=0)
        split = _split_policy_from_mapping(
            _optional_mapping(payload, "split", "dataset build config", default={}),
            default_seed=f"codelewm.dataset_build.{seed}",
        )
        return cls(
            schema_version=_require_string(payload, "schema_version", "dataset build config"),
            name=_require_string(payload, "name", "dataset build config"),
            seed=seed,
            sources=_sources_from_sequence(
                _require_sequence(payload, "sources", "dataset build config")
            ),
            filter=_filter_policy_from_mapping(
                _optional_mapping(payload, "filter", "dataset build config", default={})
            ),
            split=split,
            dedup=_dedup_policy_from_mapping(
                _optional_mapping(payload, "dedup", "dataset build config", default={})
            ),
            codestate=_codestate_config_from_mapping(
                _optional_mapping(payload, "codestate", "dataset build config", default={})
            ),
            normalize=_normalize_config_from_mapping(
                _optional_mapping(payload, "normalize", "dataset build config", default={})
            ),
            action=_action_config_from_mapping(
                _optional_mapping(payload, "action", "dataset build config", default={})
            ),
            license=DatasetLicenseConfig.from_mapping(
                _optional_mapping(payload, "license", "dataset build config", default={})
            ),
        )

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_BUILD_CONFIG_SCHEMA_VERSION:
            raise DatasetBuildConfigError(
                f"schema_version must be {DATASET_BUILD_CONFIG_SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if not self.name.strip():
            raise DatasetBuildConfigError("name must not be empty")
        if self.seed < 0:
            raise DatasetBuildConfigError("seed must be non-negative")
        if not self.sources:
            raise DatasetBuildConfigError("sources must contain at least one source")
        _ensure_json_native(self.to_dict(), section="dataset build config")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "seed": self.seed,
            "sources": [source.to_dict() for source in self.sources],
            "filter": {
                "min_changed_lines": self.filter.min_changed_lines,
                "max_changed_lines": self.filter.max_changed_lines,
                "min_edit_ratio": self.filter.min_edit_ratio,
                "max_edit_ratio": self.filter.max_edit_ratio,
                "min_message_chars": self.filter.min_message_chars,
                "max_message_chars": self.filter.max_message_chars,
                "generated_path_markers": list(self.filter.generated_path_markers),
                "generated_filename_suffixes": list(self.filter.generated_filename_suffixes),
            },
            "split": {
                "train_ratio": self.split.train_ratio,
                "val_ratio": self.split.val_ratio,
                "test_ratio": self.split.test_ratio,
                "seed": self.split.seed,
                "split_overrides": dict(self.split.split_overrides),
            },
            "dedup": {
                "near_duplicate_hamming_threshold": self.dedup.near_duplicate_hamming_threshold,
            },
            "codestate": {
                "max_small_file_lines": self.codestate.max_small_file_lines,
                "region_context_lines": self.codestate.region_context_lines,
            },
            "normalize": {
                "token_budget": self.normalize.token_budget,
                "large_string_chars": self.normalize.large_string_chars,
                "large_number_digits": self.normalize.large_number_digits,
                "changed_context_lines": self.normalize.changed_context_lines,
                "include_docstrings": self.normalize.include_docstrings,
            },
            "action": {
                "max_text_chars": self.action.max_text_chars,
                "include_patch": self.action.include_patch,
                "max_abstract_ops": self.action.max_abstract_ops,
            },
            "license": self.license.to_dict(),
        }


@dataclass(frozen=True)
class DatasetBuildResult:
    """Files and manifests emitted by one dataset build."""

    output_dir: Path
    artifact_manifest: ArtifactManifest
    dataset_manifest: DatasetManifest
    row_counts: Mapping[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": DATASET_BUILD_REPORT_SCHEMA_VERSION,
            "ok": True,
            "artifact_manifest": str(self.output_dir / "manifest.json"),
            "dataset_manifest": str(self.output_dir / "dataset_manifest.json"),
            "artifact_id": self.artifact_manifest.artifact_id,
            "row_count": self.dataset_manifest.row_count,
            "split_counts": dict(self.dataset_manifest.split_counts),
            "source_counts": dict(self.dataset_manifest.source_counts),
            "license_gate_report": self.dataset_manifest.metadata.get("license_gate_report"),
        }


def load_dataset_build_config(path: Path | str) -> DatasetBuildConfig:
    """Load and validate a dataset build config from JSON or YAML."""

    payload = _load_config_mapping(Path(path))
    return DatasetBuildConfig.from_mapping(payload)


def build_dataset_from_config_path(
    *,
    config_path: Path | str,
    output_dir: Path | str,
    command: Sequence[str],
) -> DatasetBuildResult:
    """Build a dataset artifact from a config path."""

    path = Path(config_path)
    config = load_dataset_build_config(path)
    return build_dataset(
        config,
        output_dir=Path(output_dir),
        config_dir=path.parent,
        command=command,
    )


def build_dataset(
    config: DatasetBuildConfig,
    *,
    output_dir: Path,
    config_dir: Path,
    command: Sequence[str],
) -> DatasetBuildResult:
    """Build a schema-versioned dataset artifact directory."""

    _prepare_output_dir(output_dir)
    records = _load_configured_sources(config, config_dir=config_dir)
    source_counts_before = _source_counts(records)

    filtered = filter_raw_edit_records(
        records,
        policy=config.filter,
        license_policy=config.license.to_policy(),
    )
    if filtered.report.total_before != len(records):
        raise DatasetBuildError("filter accounting mismatch: input row count changed")

    split_dedup = split_and_deduplicate(
        filtered.kept,
        split_policy=config.split,
        dedup_policy=config.dedup,
    )
    if split_dedup.report.total_before != filtered.report.total_after:
        raise DatasetBuildError("split/dedup accounting mismatch: filtered row count changed")

    transitions = tuple(
        _assignment_to_transition(
            assignment.record,
            split=assignment.split,
            dedup_keys=assignment.dedup_keys.to_dict(),
            config=config,
        )
        for assignment in split_dedup.kept
    )
    if split_dedup.report.total_after != len(transitions):
        raise DatasetBuildError("transition accounting mismatch: split/dedup row count changed")
    if not transitions:
        raise DatasetBuildError(
            "dataset build produced zero kept transitions after filtering and deduplication"
        )

    config_path = output_dir / "config.json"
    transitions_path = output_dir / "transitions.jsonl"
    reports_dir = output_dir / "reports"
    filter_report_path = reports_dir / "filter_report.json"
    license_report_path = reports_dir / "license_gate_report.json"
    split_dedup_report_path = reports_dir / "split_dedup_report.json"
    row_counts_path = reports_dir / "row_counts.json"

    _write_json(config.to_dict(), config_path)
    _write_transitions_jsonl(transitions, transitions_path)
    _write_json(filtered.to_dict(), filter_report_path)
    _write_json(filtered.license_gate_report.to_dict(), license_report_path)
    _write_json(split_dedup.to_dict(), split_dedup_report_path)

    row_counts = _row_counts_report(
        source_counts_before=source_counts_before,
        filtered_report=filtered.report.to_dict(),
        split_dedup_report=split_dedup.report.to_dict(),
        kept_transitions=len(transitions),
    )
    _write_json(row_counts, row_counts_path)

    data_artifacts = (
        _artifact_info(transitions_path, root=output_dir, kind="transitions_jsonl", rows=len(transitions)),
        _artifact_info(filter_report_path, root=output_dir, kind="filter_report", rows=filtered.report.total_before),
        _artifact_info(
            license_report_path,
            root=output_dir,
            kind="license_gate_report",
            rows=filtered.license_gate_report.included_rows,
        ),
        _artifact_info(
            split_dedup_report_path,
            root=output_dir,
            kind="split_dedup_report",
            rows=split_dedup.report.total_before,
        ),
        _artifact_info(row_counts_path, root=output_dir, kind="row_counts_report", rows=len(transitions)),
        _artifact_info(config_path, root=output_dir, kind="config", rows=0),
    )
    metadata = {
        "build_config_schema_version": config.schema_version,
        "build_name": config.name,
        "source_counts_before_filter": source_counts_before,
        "filter_report": filtered.report.to_dict(),
        "split_dedup_report": split_dedup.report.to_dict(),
        "row_counts": row_counts,
    }
    dataset_manifest = build_dataset_manifest(
        transitions,
        artifacts=data_artifacts,
        spec=PackSpec(include_action_patch=config.action.include_patch),
        metadata=metadata,
        license_gate_report=filtered.license_gate_report,
    )
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    write_dataset_manifest(dataset_manifest, dataset_manifest_path)

    manifest_files = (
        config_path.resolve(),
        transitions_path.resolve(),
        dataset_manifest_path.resolve(),
        filter_report_path.resolve(),
        license_report_path.resolve(),
        split_dedup_report_path.resolve(),
        row_counts_path.resolve(),
    )
    artifact_manifest = build_artifact_manifest(
        artifact_kind="dataset",
        root=output_dir,
        files=manifest_files,
        command=command,
        config=config.to_dict(),
        metadata={
            "dataset_manifest": "dataset_manifest.json",
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "row_count": dataset_manifest.row_count,
            "split_counts": dict(dataset_manifest.split_counts),
            "source_counts": dict(dataset_manifest.source_counts),
            "license_gate_report": dataset_manifest.metadata.get("license_gate_report"),
        },
    )
    write_artifact_manifest(artifact_manifest, output_dir / "manifest.json")
    return DatasetBuildResult(
        output_dir=output_dir,
        artifact_manifest=artifact_manifest,
        dataset_manifest=dataset_manifest,
        row_counts=row_counts,
    )


def _load_configured_sources(config: DatasetBuildConfig, *, config_dir: Path) -> tuple[RawEditRecord, ...]:
    records: list[RawEditRecord] = []
    for source in config.sources:
        spec = source.to_source_spec(config_dir=config_dir)
        try:
            records.extend(load_source(spec))
        except SourceRecordError:
            raise
        except SourceUnavailableError:
            raise
    return tuple(records)


def _assignment_to_transition(
    record: RawEditRecord,
    *,
    split: Literal["train", "val", "test"],
    dedup_keys: Mapping[str, str],
    config: DatasetBuildConfig,
) -> PackedTransition:
    try:
        state_pair = extract_codestate_pair(record, config=config.codestate)
        before = build_masked_codestate(state_pair.before, config=config.normalize)
        after = build_masked_codestate(state_pair.after, config=config.normalize)
        action = extract_edit_action(record, config=config.action)
    except ValueError as exc:
        raise DatasetBuildError(f"failed to extract transition {record_id(record)}: {exc}") from exc

    filter_flags: list[str] = []
    filter_flags.extend(f"before_dropped:{item}" for item in before.normalized.dropped_sections)
    filter_flags.extend(f"after_dropped:{item}" for item in after.normalized.dropped_sections)
    if action.patch_is_leaky:
        filter_flags.append("action_patch_leaky")

    return PackedTransition(
        transition_id=_transition_id(record, dedup_keys),
        source=record.source,
        repo=record.repo,
        commit=record.commit,
        path=record.path_after,
        split=split,
        state_before=before.token_sequence,
        state_after=after.token_sequence,
        action_text=_token_sequence_from_text(action.text),
        action_abs=_token_sequence_from_tokens(action.abstract),
        action_patch=None if action.patch is None else _token_sequence_from_text(action.patch),
        edit_size=_edit_size(record),
        token_count_before=before.normalized.token_count,
        token_count_after=after.normalized.token_count,
        license=record.license,
        filter_flags=tuple(filter_flags),
        dedup_keys=tuple(f"{key}:{value}" for key, value in sorted(dedup_keys.items())),
    )


def _transition_id(record: RawEditRecord, dedup_keys: Mapping[str, str]) -> str:
    payload = {
        "source": record.source,
        "repo": record.repo,
        "commit": record.commit,
        "path": record.path_after,
        "dedup": dict(sorted(dedup_keys.items())),
    }
    return "transition-" + _json_sha256(payload)[:24]


def _token_sequence_from_text(text: str) -> TokenSequence:
    return _token_sequence_from_tokens(tuple(_TOKEN_PATTERN.findall(text)))


def _token_sequence_from_tokens(tokens: Sequence[str]) -> TokenSequence:
    return TokenSequence(
        input_ids=tuple(stable_token_id(token) for token in tokens),
        attention_mask=tuple(True for _ in tokens),
    )


def _edit_size(record: RawEditRecord) -> int:
    import difflib

    changed = 0
    matcher = difflib.SequenceMatcher(None, record.before.splitlines(), record.after.splitlines())
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag != "equal":
            changed += (before_end - before_start) + (after_end - after_start)
    return changed


def _write_transitions_jsonl(rows: Sequence[PackedTransition], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_parquet_row(), sort_keys=True, allow_nan=False) + "\n")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _artifact_info(path: Path, *, root: Path, kind: str, rows: int) -> ArtifactInfo:
    return ArtifactInfo(
        path=path.resolve().relative_to(root.resolve()).as_posix(),
        kind=kind,
        rows=rows,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
    )


def _row_counts_report(
    *,
    source_counts_before: Mapping[str, int],
    filtered_report: Mapping[str, Any],
    split_dedup_report: Mapping[str, Any],
    kept_transitions: int,
) -> dict[str, Any]:
    total_loaded = int(sum(source_counts_before.values()))
    filter_input = int(filtered_report["total_before"])
    filter_after = int(filtered_report["total_after"])
    split_input = int(split_dedup_report["total_before"])
    split_after = int(split_dedup_report["total_after"])
    return {
        "schema_version": ROW_COUNTS_REPORT_SCHEMA_VERSION,
        "total_loaded": total_loaded,
        "filter_total_before": filter_input,
        "filter_total_after": filter_after,
        "filter_total_dropped": int(filtered_report["total_dropped"]),
        "split_dedup_total_before": split_input,
        "split_dedup_total_after": split_after,
        "split_dedup_total_dropped": int(split_dedup_report["total_dropped"]),
        "kept_transitions": kept_transitions,
        "source_counts_before_filter": dict(source_counts_before),
        "accounting": {
            "loaded_equals_filter_input": total_loaded == filter_input,
            "filter_output_equals_split_input": filter_after == split_input,
            "split_output_equals_transitions": split_after == kept_transitions,
        },
    }


def _source_counts(records: Sequence[RawEditRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source] = counts.get(record.source, 0) + 1
    return counts


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and output_dir.is_file():
        raise DatasetBuildConfigError(f"output path is a file: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DatasetBuildConfigError(
            f"output directory is not empty: {output_dir}; choose an empty path"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_config_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise DatasetBuildConfigError(f"dataset build config does not exist: {path}")
    if not path.is_file():
        raise DatasetBuildConfigError(f"dataset build config is not a file: {path}")

    if path.suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise DatasetBuildConfigError(f"dataset build config is not valid JSON: {exc.msg}") from exc
    elif path.suffix in {".yaml", ".yml"}:
        payload = _load_yaml_mapping(path)
    else:
        raise DatasetBuildConfigError(f"unsupported dataset build config extension: {path.suffix}")
    if not isinstance(payload, Mapping):
        raise DatasetBuildConfigError(f"dataset build config root must be a mapping: {path}")
    return payload


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise DatasetBuildConfigError(
            "YAML dataset build configs require omegaconf; use JSON or install the train dependency group"
        ) from exc

    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # pragma: no cover - depends on optional runtime.
        raise DatasetBuildConfigError(f"failed to load dataset build config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DatasetBuildConfigError(f"dataset build config root must be a mapping: {path}")
    return payload


def _resolve_config_path(value: str, *, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_dir / path


def _sources_from_sequence(values: Sequence[Any]) -> tuple[DatasetSourceConfig, ...]:
    sources: list[DatasetSourceConfig] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise DatasetBuildConfigError("sources[] entries must be mappings")
        sources.append(DatasetSourceConfig.from_mapping(value))
    return tuple(sources)


def _filter_policy_from_mapping(payload: Mapping[str, Any]) -> FilterPolicy:
    _reject_unknown(
        payload,
        {
            "min_changed_lines",
            "max_changed_lines",
            "min_edit_ratio",
            "max_edit_ratio",
            "min_message_chars",
            "max_message_chars",
            "generated_path_markers",
            "generated_filename_suffixes",
        },
        "filter",
    )
    return FilterPolicy(
        min_changed_lines=_optional_int(payload, "min_changed_lines", "filter", default=1),
        max_changed_lines=_optional_int(payload, "max_changed_lines", "filter", default=150),
        min_edit_ratio=_optional_float(payload, "min_edit_ratio", "filter", default=0.02),
        max_edit_ratio=_optional_float(payload, "max_edit_ratio", "filter", default=0.60),
        min_message_chars=_optional_int(payload, "min_message_chars", "filter", default=8),
        max_message_chars=_optional_int(payload, "max_message_chars", "filter", default=512),
        generated_path_markers=_optional_string_tuple(
            payload,
            "generated_path_markers",
            "filter",
            default=FilterPolicy().generated_path_markers,
        ),
        generated_filename_suffixes=_optional_string_tuple(
            payload,
            "generated_filename_suffixes",
            "filter",
            default=FilterPolicy().generated_filename_suffixes,
        ),
    )


def _split_policy_from_mapping(payload: Mapping[str, Any], *, default_seed: str) -> SplitPolicy:
    _reject_unknown(
        payload,
        {"train_ratio", "val_ratio", "test_ratio", "seed", "split_overrides"},
        "split",
    )
    overrides = _optional_mapping(payload, "split_overrides", "split", default={})
    normalized_overrides: dict[str, Literal["train", "val", "test"]] = {}
    for key, value in overrides.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise DatasetBuildConfigError("split.split_overrides must map strings to split names")
        if value not in {"train", "val", "test"}:
            raise DatasetBuildConfigError("split.split_overrides values must be train, val, or test")
        normalized_overrides[key] = cast(Literal["train", "val", "test"], value)
    return SplitPolicy(
        train_ratio=_optional_float(payload, "train_ratio", "split", default=0.80),
        val_ratio=_optional_float(payload, "val_ratio", "split", default=0.10),
        test_ratio=_optional_float(payload, "test_ratio", "split", default=0.10),
        seed=_optional_string(payload, "seed", "split") or default_seed,
        split_overrides=normalized_overrides,
    )


def _dedup_policy_from_mapping(payload: Mapping[str, Any]) -> DedupPolicy:
    _reject_unknown(payload, {"near_duplicate_hamming_threshold"}, "dedup")
    return DedupPolicy(
        near_duplicate_hamming_threshold=_optional_int(
            payload,
            "near_duplicate_hamming_threshold",
            "dedup",
            default=3,
        )
    )


def _codestate_config_from_mapping(payload: Mapping[str, Any]) -> CodeStateConfig:
    _reject_unknown(payload, {"max_small_file_lines", "region_context_lines"}, "codestate")
    return CodeStateConfig(
        max_small_file_lines=_optional_int(payload, "max_small_file_lines", "codestate", default=40),
        region_context_lines=_optional_int(payload, "region_context_lines", "codestate", default=8),
    )


def _normalize_config_from_mapping(payload: Mapping[str, Any]) -> CodeStateNormalizationConfig:
    _reject_unknown(
        payload,
        {
            "token_budget",
            "large_string_chars",
            "large_number_digits",
            "changed_context_lines",
            "include_docstrings",
        },
        "normalize",
    )
    return CodeStateNormalizationConfig(
        token_budget=_optional_int(payload, "token_budget", "normalize", default=1024),
        large_string_chars=_optional_int(payload, "large_string_chars", "normalize", default=96),
        large_number_digits=_optional_int(payload, "large_number_digits", "normalize", default=12),
        changed_context_lines=_optional_int(payload, "changed_context_lines", "normalize", default=2),
        include_docstrings=_optional_bool(payload, "include_docstrings", "normalize", default=False),
    )


def _action_config_from_mapping(payload: Mapping[str, Any]) -> ActionExtractionConfig:
    _reject_unknown(payload, {"max_text_chars", "include_patch", "max_abstract_ops"}, "action")
    return ActionExtractionConfig(
        max_text_chars=_optional_int(payload, "max_text_chars", "action", default=256),
        include_patch=_optional_bool(payload, "include_patch", "action", default=False),
        max_abstract_ops=_optional_int(payload, "max_abstract_ops", "action", default=32),
    )


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise DatasetBuildConfigError(f"{section} contains unknown key(s): {', '.join(unknown)}")


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    if key not in payload:
        raise DatasetBuildConfigError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise DatasetBuildConfigError(f"{section}.{key} must be a string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str, section: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetBuildConfigError(f"{section}.{key} must be a string")
    return value


def _optional_int(payload: Mapping[str, Any], key: str, section: str, *, default: int) -> int:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetBuildConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_float(payload: Mapping[str, Any], key: str, section: str, *, default: float) -> float:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetBuildConfigError(f"{section}.{key} must be numeric")
    return float(value)


def _optional_bool(payload: Mapping[str, Any], key: str, section: str, *, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise DatasetBuildConfigError(f"{section}.{key} must be true or false")
    return value


def _require_sequence(payload: Mapping[str, Any], key: str, section: str) -> Sequence[Any]:
    if key not in payload:
        raise DatasetBuildConfigError(f"{section}.{key} is required")
    value = payload[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetBuildConfigError(f"{section}.{key} must be a JSON array")
    return value


def _optional_mapping(
    payload: Mapping[str, Any],
    key: str,
    section: str,
    *,
    default: Mapping[str, Any],
) -> Mapping[str, Any]:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, Mapping):
        raise DatasetBuildConfigError(f"{section}.{key} must be a mapping")
    return value


def _optional_string_tuple(
    payload: Mapping[str, Any],
    key: str,
    section: str,
    *,
    default: Sequence[str],
) -> tuple[str, ...]:
    if key not in payload:
        return tuple(default)
    value = payload[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetBuildConfigError(f"{section}.{key} must be a list of strings")
    output: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise DatasetBuildConfigError(f"{section}.{key} must contain only strings")
        output.append(item)
    return tuple(output)


def _ensure_json_native(payload: Any, *, section: str) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DatasetBuildConfigError(f"{section} must be JSON-native: {exc}") from exc


def _json_sha256(payload: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
