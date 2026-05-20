"""Pack built transition JSONL artifacts into training-ready dataset files."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codelewm.data.action_diagnostics import build_action_discriminative_shard_report
from codelewm.data.pack import (
    DATASET_SCHEMA_VERSION,
    ArtifactInfo,
    DatasetManifest,
    OptionalDependencyError,
    PackError,
    PackSpec,
    PackedTransition,
    build_dataset_manifest,
    read_dataset_manifest,
    read_packed_transitions_jsonl,
    sha256_file,
    validate_manifest_checksums,
    write_dataset_manifest,
    write_hdf5_pack,
    write_parquet_staging_shards,
)
from codelewm.data.split_dedup import SplitName
from codelewm.observability import ArtifactManifest, read_artifact_manifest
from codelewm.observability import build_artifact_manifest, validate_artifact_checksums, write_artifact_manifest


DATASET_PACK_CONFIG_SCHEMA_VERSION = "codelewm.dataset_pack_config.v1"
DATASET_PACK_REPORT_SCHEMA_VERSION = "codelewm.dataset_pack_report.v1"
_SPLITS: tuple[SplitName, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetPackResult:
    """Files and manifests emitted by one dataset pack."""

    output_dir: Path
    artifact_manifest: ArtifactManifest
    dataset_manifest: DatasetManifest
    pack_report: Mapping[str, Any]
    action_discriminative_report: Mapping[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": DATASET_PACK_REPORT_SCHEMA_VERSION,
            "ok": True,
            "artifact_manifest": str(self.output_dir / "manifest.json"),
            "dataset_manifest": str(self.output_dir / "dataset_manifest.json"),
            "artifact_id": self.artifact_manifest.artifact_id,
            "parent_artifacts": list(self.artifact_manifest.parent_artifacts),
            "row_count": self.dataset_manifest.row_count,
            "split_counts": dict(self.dataset_manifest.split_counts),
            "source_counts": dict(self.dataset_manifest.source_counts),
            "hdf5": {
                split: str(self.output_dir / "hdf5" / f"{split}.hdf5")
                for split in _SPLITS
            },
            "action_discriminative_shard_report": str(
                self.output_dir / "reports" / "action_discriminative_shard_report.json"
            ),
            "action_discriminative_claim_ready": bool(
                self.action_discriminative_report["claim_readiness"]["positive_action_use_claim_ready"]
            ),
        }


def pack_dataset_from_manifest(
    *,
    manifest_path: Path | str,
    output_dir: Path | str,
    command: Sequence[str],
    spec: PackSpec = PackSpec(),
    parquet_shard_size: int = 1000,
) -> DatasetPackResult:
    """Pack a build artifact manifest into split HDF5 and Parquet artifacts."""

    _require_pack_dependencies()
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir)

    parent_manifest = read_artifact_manifest(manifest_path)
    parent_root = manifest_path.parent
    validate_artifact_checksums(parent_manifest, root=parent_root)
    if parent_manifest.artifact_kind != "dataset":
        raise PackError("dataset pack input manifest must have artifact_kind='dataset'")

    transition_path = parent_root / _required_manifest_file(parent_manifest, "transitions.jsonl")
    input_dataset_manifest_path = parent_root / _input_dataset_manifest_path(parent_manifest)
    input_dataset_manifest = read_dataset_manifest(input_dataset_manifest_path)
    validate_manifest_checksums(input_dataset_manifest, root=parent_root)

    transitions = read_packed_transitions_jsonl(transition_path)
    if input_dataset_manifest.row_count != len(transitions):
        raise PackError(
            "input dataset manifest row_count does not match transition rows: "
            f"{input_dataset_manifest.row_count} != {len(transitions)}"
        )
    if not transitions:
        raise PackError("dataset pack input contains zero transitions")

    config_path = output_dir / "config.json"
    pack_config = _pack_config_payload(
        input_manifest=manifest_path,
        spec=spec,
        parquet_shard_size=parquet_shard_size,
    )
    _write_json(pack_config, config_path)

    by_split = _transitions_by_split(transitions)
    artifacts: list[ArtifactInfo] = []
    for split in _SPLITS:
        split_rows = by_split[split]
        hdf5_artifact = write_hdf5_pack(split_rows, output_dir / "hdf5" / f"{split}.hdf5", spec=spec)
        artifacts.append(_relative_artifact(hdf5_artifact, output_dir))
        parquet_artifacts = write_parquet_staging_shards(
            split_rows,
            output_dir / "parquet" / split,
            shard_size=parquet_shard_size,
        )
        artifacts.extend(_relative_artifact(artifact, output_dir) for artifact in parquet_artifacts)

    action_discriminative_report = build_action_discriminative_shard_report(transitions)
    action_discriminative_report_path = output_dir / "reports" / "action_discriminative_shard_report.json"
    _write_json(action_discriminative_report, action_discriminative_report_path)
    pack_report = _pack_report_payload(
        parent_manifest=parent_manifest,
        input_dataset_manifest=input_dataset_manifest,
        transitions=transitions,
        artifacts=artifacts,
    )
    pack_report_path = output_dir / "reports" / "pack_report.json"
    _write_json(pack_report, pack_report_path)
    artifacts.extend(
        (
            _artifact_info(config_path, root=output_dir, kind="config", rows=0),
            _artifact_info(pack_report_path, root=output_dir, kind="pack_report", rows=len(transitions)),
            _artifact_info(
                action_discriminative_report_path,
                root=output_dir,
                kind="action_discriminative_shard_report",
                rows=len(transitions),
            ),
        )
    )

    inherited_license_gate = input_dataset_manifest.metadata.get("license_gate_report")
    dataset_manifest = build_dataset_manifest(
        transitions,
        artifacts=tuple(artifacts),
        spec=spec,
        metadata={
            "pack_report": pack_report,
            "input_artifact_id": parent_manifest.artifact_id,
            "input_manifest": str(manifest_path),
            "input_dataset_manifest": _relative_to_parent(input_dataset_manifest_path, parent_root),
            "action_discriminative_shard_report": action_discriminative_report,
        },
        license_gate_report=inherited_license_gate if isinstance(inherited_license_gate, Mapping) else None,
    )
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    write_dataset_manifest(dataset_manifest, dataset_manifest_path)

    manifest_files = [
        (output_dir / artifact.path).resolve()
        for artifact in artifacts
    ]
    manifest_files.append(dataset_manifest_path.resolve())
    artifact_manifest = build_artifact_manifest(
        artifact_kind="dataset",
        root=output_dir,
        files=manifest_files,
        command=command,
        config=pack_config,
        parent_artifacts=(parent_manifest.artifact_id,),
        metadata={
            "dataset_manifest": "dataset_manifest.json",
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "input_artifact_id": parent_manifest.artifact_id,
            "row_count": dataset_manifest.row_count,
            "split_counts": dict(dataset_manifest.split_counts),
            "source_counts": dict(dataset_manifest.source_counts),
            "pack_report": "reports/pack_report.json",
            "action_discriminative_shard_report": "reports/action_discriminative_shard_report.json",
            "action_discriminative_claim_ready": bool(
                action_discriminative_report["claim_readiness"]["positive_action_use_claim_ready"]
            ),
            "action_discriminative_hard_negative_pools": action_discriminative_report["hard_negative_pools"],
        },
    )
    write_artifact_manifest(artifact_manifest, output_dir / "manifest.json")
    return DatasetPackResult(
        output_dir=output_dir,
        artifact_manifest=artifact_manifest,
        dataset_manifest=dataset_manifest,
        pack_report=pack_report,
        action_discriminative_report=action_discriminative_report,
    )


def _require_pack_dependencies() -> None:
    missing: list[str] = []
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError:
        missing.append("h5py")
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        missing.append("pyarrow")
    if missing:
        joined = ", ".join(missing)
        raise OptionalDependencyError(
            f"dataset packing requires {joined}; install the data dependency group with `uv sync --group data --group dev`"
        )


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and output_dir.is_file():
        raise PackError(f"output path is a file: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PackError(f"output directory is not empty: {output_dir}; choose an empty path")
    output_dir.mkdir(parents=True, exist_ok=True)


def _required_manifest_file(manifest: ArtifactManifest, path: str) -> str:
    for file in manifest.files:
        if file.path == path:
            return file.path
    raise PackError(f"input artifact manifest does not list required file: {path}")


def _input_dataset_manifest_path(manifest: ArtifactManifest) -> str:
    value = manifest.metadata.get("dataset_manifest")
    if not isinstance(value, str) or not value:
        raise PackError("input artifact manifest metadata.dataset_manifest is required")
    _required_manifest_file(manifest, value)
    return value


def _transitions_by_split(rows: Sequence[PackedTransition]) -> dict[SplitName, tuple[PackedTransition, ...]]:
    grouped: dict[SplitName, list[PackedTransition]] = {split: [] for split in _SPLITS}
    for row in rows:
        grouped[row.split].append(row)
    return {split: tuple(grouped[split]) for split in _SPLITS}


def _pack_config_payload(*, input_manifest: Path, spec: PackSpec, parquet_shard_size: int) -> dict[str, Any]:
    return {
        "schema_version": DATASET_PACK_CONFIG_SCHEMA_VERSION,
        "input_manifest": str(input_manifest),
        "pack_spec": {
            "schema_version": spec.schema_version,
            "state_length": spec.state_length,
            "action_text_length": spec.action_text_length,
            "action_abs_length": spec.action_abs_length,
            "action_patch_length": spec.action_patch_length,
            "include_action_patch": spec.include_action_patch,
        },
        "parquet_shard_size": parquet_shard_size,
    }


def _pack_report_payload(
    *,
    parent_manifest: ArtifactManifest,
    input_dataset_manifest: DatasetManifest,
    transitions: Sequence[PackedTransition],
    artifacts: Sequence[ArtifactInfo],
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_PACK_REPORT_SCHEMA_VERSION,
        "input_artifact_id": parent_manifest.artifact_id,
        "input_dataset_schema_version": input_dataset_manifest.schema_version,
        "input_row_count": input_dataset_manifest.row_count,
        "packed_row_count": len(transitions),
        "split_counts": dict(input_dataset_manifest.split_counts),
        "source_counts": dict(input_dataset_manifest.source_counts),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
    }


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


def _relative_artifact(artifact: ArtifactInfo, root: Path) -> ArtifactInfo:
    path = Path(artifact.path)
    try:
        artifact_path = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        artifact_path = artifact.path
    return ArtifactInfo(
        path=artifact_path,
        kind=artifact.kind,
        rows=artifact.rows,
        sha256=artifact.sha256,
        bytes=artifact.bytes,
    )


def _relative_to_parent(path: Path, parent: Path) -> str:
    try:
        return path.resolve().relative_to(parent.resolve()).as_posix()
    except ValueError:
        return str(path)
