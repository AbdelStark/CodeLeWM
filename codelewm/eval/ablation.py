"""Action-view ablation suite reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codelewm.observability import (
    ArtifactManifest,
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.training import (
    TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
    read_training_run_manifest,
)

from .action_policy import (
    build_action_view_report_policy,
    validate_action_view_report_policy,
)
from .retrieval import (
    ActionUseClaimGate,
    RetrievalMetrics,
    RetrievalReport,
    build_action_use_claim_gate,
    read_retrieval_report,
    validate_action_use_claim_gate,
)


ACTION_ABLATION_REPORT_SCHEMA_VERSION = "codelewm.eval.action_ablation_report.v1"
ACTION_ABLATION_RUN_SCHEMA_VERSION = "codelewm.eval.action_ablation_run.v1"

AblationFamily = Literal["action_view", "baseline", "retrieval_loss", "collapse"]
AblationStatus = Literal["completed", "blocked", "failed"]

_REQUIRED_BASELINES = ("random", "lexical", "no_action", "shuffled_action")


class ActionAblationError(ValueError):
    """Raised when an action-view ablation report is invalid."""


@dataclass(frozen=True)
class ActionAblationRow:
    """One completed, blocked, or failed ablation row."""

    name: str
    family: AblationFamily
    status: AblationStatus
    metrics: Mapping[str, float] | None = None
    action_view_policy: Mapping[str, Any] | None = None
    source_report: str | None = None
    artifact_manifest_id: str | None = None
    block_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ActionAblationError("ablation row name must not be empty")
        if self.family not in {"action_view", "baseline", "retrieval_loss", "collapse"}:
            raise ActionAblationError(f"unsupported ablation family: {self.family}")
        if self.status not in {"completed", "blocked", "failed"}:
            raise ActionAblationError(f"unsupported ablation status: {self.status}")
        if self.status == "completed" and self.block_reason is not None:
            raise ActionAblationError(
                "completed ablation rows must not include block_reason"
            )
        if self.status in {"blocked", "failed"} and not self.block_reason:
            raise ActionAblationError(
                "blocked or failed ablation rows must include block_reason"
            )
        if self.metrics is not None:
            _validate_metrics(self.metrics, f"rows.{self.name}.metrics")
        if self.action_view_policy is not None:
            validate_action_view_report_policy(dict(self.action_view_policy))
        _ensure_json_native(self.metadata, f"rows.{self.name}.metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "status": self.status,
            "metrics": None if self.metrics is None else dict(self.metrics),
            "action_view_policy": None
            if self.action_view_policy is None
            else dict(self.action_view_policy),
            "source_report": self.source_report,
            "artifact_manifest_id": self.artifact_manifest_id,
            "block_reason": self.block_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionAblationRow":
        return cls(
            name=_require_string(payload, "name", "ablation row"),
            family=_require_literal(
                payload,
                "family",
                "ablation row",
                {"action_view", "baseline", "retrieval_loss", "collapse"},
            ),
            status=_require_literal(
                payload,
                "status",
                "ablation row",
                {"completed", "blocked", "failed"},
            ),
            metrics=_optional_float_mapping(
                payload.get("metrics"), "ablation row metrics"
            ),
            action_view_policy=_optional_mapping(
                payload.get("action_view_policy"), "action_view_policy"
            ),
            source_report=_optional_string(payload, "source_report", "ablation row"),
            artifact_manifest_id=_optional_string(
                payload, "artifact_manifest_id", "ablation row"
            ),
            block_reason=_optional_string(payload, "block_reason", "ablation row"),
            metadata=dict(
                _optional_mapping(payload.get("metadata", {}), "metadata") or {}
            ),
        )


@dataclass(frozen=True)
class ActionAblationReport:
    """Consolidated report for action-view and objective ablations."""

    rows: tuple[ActionAblationRow, ...]
    source_artifacts: Mapping[str, str]
    required_baselines: tuple[str, ...] = _REQUIRED_BASELINES
    claim_gate: ActionUseClaimGate | None = None
    notes: tuple[str, ...] = ()
    schema_version: str = ACTION_ABLATION_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ACTION_ABLATION_REPORT_SCHEMA_VERSION:
            raise ActionAblationError(
                "schema_version must be "
                f"{ACTION_ABLATION_REPORT_SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if not self.rows:
            raise ActionAblationError("ablation report must include at least one row")
        names = [row.name for row in self.rows]
        if len(set(names)) != len(names):
            raise ActionAblationError("ablation row names must be unique")
        missing = [name for name in self.required_baselines if name not in names]
        if missing:
            raise ActionAblationError(
                f"ablation report missing required baseline rows: {', '.join(missing)}"
            )
        patch_rows = [row for row in self.rows if row.name == "patch_action_diagnostic"]
        if not patch_rows:
            raise ActionAblationError(
                "ablation report must include patch_action_diagnostic row"
            )
        patch_policy = patch_rows[0].action_view_policy
        if patch_policy is None:
            raise ActionAblationError(
                "patch_action_diagnostic row must include action_view_policy"
            )
        validate_action_view_report_policy(dict(patch_policy))
        if not dict(patch_policy).get("diagnostic_upper_bound"):
            raise ActionAblationError(
                "patch_action_diagnostic must be tagged diagnostic_upper_bound=true"
            )
        if self.claim_gate is not None:
            validate_action_use_claim_gate(self.claim_gate)
        _ensure_json_native(self.source_artifacts, "source_artifacts")
        _ensure_json_native(self.notes, "notes")

    @property
    def completed_count(self) -> int:
        return sum(row.status == "completed" for row in self.rows)

    @property
    def blocked_count(self) -> int:
        return sum(row.status == "blocked" for row in self.rows)

    @property
    def failed_count(self) -> int:
        return sum(row.status == "failed" for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rows": [row.to_dict() for row in self.rows],
            "required_baselines": list(self.required_baselines),
            "source_artifacts": dict(self.source_artifacts),
            "summary": {
                "row_count": len(self.rows),
                "completed": self.completed_count,
                "blocked": self.blocked_count,
                "failed": self.failed_count,
            },
            "claim_gate": None if self.claim_gate is None else self.claim_gate.to_dict(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionAblationReport":
        rows_value = payload.get("rows")
        if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes)):
            raise ActionAblationError("rows must be a JSON array")
        return cls(
            schema_version=_require_string(
                payload, "schema_version", "action ablation report"
            ),
            rows=tuple(
                ActionAblationRow.from_dict(_require_mapping(item, "rows[]"))
                for item in rows_value
            ),
            required_baselines=tuple(
                str(item)
                for item in payload.get("required_baselines", _REQUIRED_BASELINES)
            ),
            source_artifacts=dict(
                _require_mapping(
                    payload.get("source_artifacts", {}), "source_artifacts"
                )
            ),
            claim_gate=None
            if payload.get("claim_gate") is None
            else ActionUseClaimGate.from_dict(
                _require_mapping(payload["claim_gate"], "claim_gate")
            ),
            notes=tuple(str(note) for note in payload.get("notes", ())),
        )


@dataclass(frozen=True)
class ActionAblationRunResult:
    """CLI result for a materialized action-view ablation artifact."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    parent_artifacts: tuple[str, ...]
    rows: tuple[ActionAblationRow, ...]
    schema_version: str = ACTION_ABLATION_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "parent_artifacts": list(self.parent_artifacts),
            "row_count": len(self.rows),
            "completed": sum(row.status == "completed" for row in self.rows),
            "blocked": sum(row.status == "blocked" for row in self.rows),
            "failed": sum(row.status == "failed" for row in self.rows),
        }


def build_action_ablation_report(
    retrieval_report: RetrievalReport,
    *,
    retrieval_artifact_id: str,
    retrieval_report_path: str,
    training_artifact_id: str,
    training_manifest: Mapping[str, Any],
    train_config: Mapping[str, Any] | None = None,
) -> ActionAblationReport:
    """Build a consolidated ablation report from available artifact evidence."""

    rows: list[ActionAblationRow] = []
    policy = retrieval_report.metadata.get("action_view_policy")
    if not isinstance(policy, Mapping):
        policy = build_action_view_report_policy(
            "text", report_scope="headline"
        ).to_dict()
    validated_policy = validate_action_view_report_policy(dict(policy)).to_dict()
    rows.append(
        _completed_row(
            name="text_action",
            family="action_view",
            metrics=retrieval_report.metrics,
            action_view_policy=validated_policy,
            source_report=retrieval_report_path,
            artifact_manifest_id=retrieval_artifact_id,
        )
    )
    rows.append(
        _blocked_row(
            name="abstract_action",
            family="action_view",
            action_view_policy=build_action_view_report_policy(
                "abstract", report_scope="ablation"
            ).to_dict(),
            block_reason="no abstract-action checkpoint and retrieval report were supplied",
        )
    )
    rows.append(
        _blocked_row(
            name="patch_action_diagnostic",
            family="action_view",
            action_view_policy=build_action_view_report_policy(
                "patch", report_scope="diagnostic"
            ).to_dict(),
            block_reason="patch-action diagnostic upper-bound report was not supplied",
        )
    )

    for baseline in _REQUIRED_BASELINES:
        if baseline in retrieval_report.baselines:
            rows.append(
                _completed_row(
                    name=baseline,
                    family="baseline",
                    metrics=retrieval_report.baselines[baseline],
                    source_report=retrieval_report_path,
                    artifact_manifest_id=retrieval_artifact_id,
                )
            )
        else:
            rows.append(
                _blocked_row(
                    name=baseline,
                    family="baseline",
                    block_reason="required baseline missing from retrieval report",
                )
            )

    loss_config = _loss_config(train_config)
    retrieval_enabled = bool(loss_config.get("enable_retrieval_loss", False))
    rows.append(
        _completed_row(
            name="retrieval_loss_enabled"
            if retrieval_enabled
            else "retrieval_loss_disabled",
            family="retrieval_loss",
            metrics=retrieval_report.metrics,
            source_report=retrieval_report_path,
            artifact_manifest_id=retrieval_artifact_id,
            metadata={
                "enable_retrieval_loss": retrieval_enabled,
                "retrieval_weight": float(
                    loss_config.get("retrieval_weight", 0.0) or 0.0
                ),
            },
        )
    )
    rows.append(
        _blocked_row(
            name="retrieval_loss_disabled"
            if retrieval_enabled
            else "retrieval_loss_enabled",
            family="retrieval_loss",
            block_reason="paired retrieval-loss variant report was not supplied",
        )
    )

    sigreg_weight = float(loss_config.get("sigreg_weight", 0.09) or 0.09)
    rows.append(
        ActionAblationRow(
            name=f"collapse_sigreg_{sigreg_weight:g}",
            family="collapse",
            status="completed",
            metrics=_collapse_metrics(training_manifest),
            artifact_manifest_id=training_artifact_id,
            metadata={"sigreg_weight": sigreg_weight},
        )
    )
    for candidate in (0.05, 0.15):
        if not math.isclose(candidate, sigreg_weight):
            rows.append(
                _blocked_row(
                    name=f"collapse_sigreg_{candidate:g}",
                    family="collapse",
                    block_reason="paired SIGReg/collapse setting report was not supplied",
                    metadata={"sigreg_weight": candidate},
                )
            )

    claim_gate = build_action_use_claim_gate(
        retrieval_report.metrics,
        retrieval_report.baselines,
        additional_failure_reasons=_claim_gate_failure_reasons(rows),
    )
    return ActionAblationReport(
        rows=tuple(rows),
        source_artifacts={
            "retrieval": retrieval_artifact_id,
            "training": training_artifact_id,
        },
        claim_gate=claim_gate,
        notes=(
            "Rows marked blocked are explicit missing-run records, not dropped rows.",
            "Patch-action rows are diagnostic upper bounds only.",
        ),
    )


def run_action_ablation_suite(
    *,
    retrieval_artifact: Path | str,
    training_artifact: Path | str,
    out: Path | str,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "ablation"),
) -> ActionAblationRunResult:
    """Materialize an action-view ablation report artifact."""

    retrieval_manifest_path = Path(retrieval_artifact)
    training_manifest_path = Path(training_artifact)
    output_dir = Path(out).resolve()
    report_path = output_dir / "reports" / "action_view_ablation_report.json"
    artifact_manifest_path = output_dir / "manifest.json"
    if not overwrite and (report_path.exists() or artifact_manifest_path.exists()):
        raise ActionAblationError(
            f"output already exists; pass overwrite=True to replace: {output_dir}"
        )

    retrieval_manifest = _read_verified_manifest(retrieval_manifest_path)
    training_artifact_manifest = _read_verified_manifest(training_manifest_path)
    retrieval_report_path = _artifact_file_path(
        retrieval_manifest_path.parent,
        retrieval_manifest,
        "reports/retrieval_report.json",
    )
    train_config_path = _artifact_file_path(
        training_manifest_path.parent,
        training_artifact_manifest,
        "config.json",
    )
    retrieval_report = read_retrieval_report(retrieval_report_path)
    training_run_manifest = _training_manifest_payload(
        training_manifest_path.parent,
        training_artifact_manifest,
    )
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    report = build_action_ablation_report(
        retrieval_report,
        retrieval_artifact_id=retrieval_manifest.artifact_id,
        retrieval_report_path=_relative_to_root(
            retrieval_report_path, retrieval_manifest_path.parent
        ),
        training_artifact_id=training_artifact_manifest.artifact_id,
        training_manifest=training_run_manifest,
        train_config=train_config,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    parent_artifacts = (
        retrieval_manifest.artifact_id,
        training_artifact_manifest.artifact_id,
    )
    artifact_manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=output_dir,
        files=(report_path,),
        command=command,
        config={
            "retrieval_artifact": str(retrieval_manifest_path),
            "training_artifact": str(training_manifest_path),
        },
        parent_artifacts=parent_artifacts,
        metadata={
            "schema_version": ACTION_ABLATION_REPORT_SCHEMA_VERSION,
            "row_count": len(report.rows),
            "completed": report.completed_count,
            "blocked": report.blocked_count,
            "failed": report.failed_count,
            "claim_gate": None if report.claim_gate is None else report.claim_gate.to_dict(),
        },
    )
    write_artifact_manifest(artifact_manifest, artifact_manifest_path)
    return ActionAblationRunResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path=_relative_to_root(report_path, output_dir),
        parent_artifacts=parent_artifacts,
        rows=report.rows,
    )


def read_action_ablation_report(path: Path | str) -> ActionAblationReport:
    """Read and validate an action-view ablation report."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ActionAblationError("action ablation report must be a JSON object")
    return ActionAblationReport.from_dict(payload)


def _claim_gate_failure_reasons(rows: Sequence[ActionAblationRow]) -> tuple[str, ...]:
    reasons: list[str] = []
    for row in rows:
        if row.family == "action_view" and row.status != "completed":
            reasons.append(f"blocked_action_view_row:{row.name}")
        if row.family == "collapse" and row.status == "failed":
            reasons.append(f"collapse_diagnostics_failed:{row.name}")
    return tuple(reasons)


def _completed_row(
    *,
    name: str,
    family: AblationFamily,
    metrics: RetrievalMetrics,
    action_view_policy: Mapping[str, Any] | None = None,
    source_report: str | None = None,
    artifact_manifest_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ActionAblationRow:
    return ActionAblationRow(
        name=name,
        family=family,
        status="completed",
        metrics=_retrieval_metrics_summary(metrics),
        action_view_policy=action_view_policy,
        source_report=source_report,
        artifact_manifest_id=artifact_manifest_id,
        metadata={} if metadata is None else dict(metadata),
    )


def _blocked_row(
    *,
    name: str,
    family: AblationFamily,
    block_reason: str,
    action_view_policy: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ActionAblationRow:
    return ActionAblationRow(
        name=name,
        family=family,
        status="blocked",
        action_view_policy=action_view_policy,
        block_reason=block_reason,
        metadata={} if metadata is None else dict(metadata),
    )


def _read_verified_manifest(path: Path) -> ArtifactManifest:
    manifest = read_artifact_manifest(path)
    validate_artifact_checksums(manifest, root=path.parent)
    return manifest


def _artifact_file_path(root: Path, manifest: ArtifactManifest, suffix: str) -> Path:
    path = _optional_artifact_file_path(root, manifest, suffix)
    if path is None:
        raise ActionAblationError(
            f"artifact {manifest.artifact_id} does not include required file: {suffix}"
        )
    return path


def _optional_artifact_file_path(
    root: Path, manifest: ArtifactManifest, suffix: str
) -> Path | None:
    for file in manifest.files:
        if file.path == suffix or file.path.endswith(f"/{suffix}"):
            path = root / file.path
            if path.is_file():
                return path
    return None


def _training_manifest_payload(
    root: Path, manifest: ArtifactManifest
) -> Mapping[str, Any]:
    path = _optional_artifact_file_path(root, manifest, "training_manifest.json")
    if path is not None:
        return read_training_run_manifest(path).to_dict()
    metadata = dict(manifest.metadata)
    if metadata.get("schema_version") != TRAINING_RUN_MANIFEST_SCHEMA_VERSION:
        raise ActionAblationError(
            "training artifact must include training_manifest.json or "
            f"metadata.schema_version={TRAINING_RUN_MANIFEST_SCHEMA_VERSION}"
        )
    return metadata


def _loss_config(train_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if train_config is None:
        return {}
    loss = train_config.get("loss", {})
    return loss if isinstance(loss, Mapping) else {}


def _collapse_metrics(training_manifest: Mapping[str, Any]) -> dict[str, float]:
    final_metrics = _require_mapping(
        training_manifest.get("final_metrics", {}), "final_metrics"
    )
    keys = (
        "collapse/effective_rank",
        "collapse/effective_rank_ratio",
        "collapse/per_dim_variance_min",
        "collapse/per_dim_variance_median",
        "collapse/nearest_neighbor_entropy",
    )
    metrics: dict[str, float] = {}
    for key in keys:
        value = final_metrics.get(key)
        if value is None:
            raise ActionAblationError(
                f"training manifest is missing collapse metric: {key}"
            )
        metrics[key] = _finite_float(value, key)
    return metrics


def _retrieval_metrics_summary(metrics: RetrievalMetrics) -> dict[str, float]:
    return {
        "query_count": float(metrics.query_count),
        "recall_at_1": float(metrics.recall_at_1),
        "recall_at_5": float(metrics.recall_at_5),
        "recall_at_10": float(metrics.recall_at_10),
        "mrr": float(metrics.mrr),
        "median_rank": float(metrics.median_rank),
    }


def _validate_metrics(metrics: Mapping[str, Any], section: str) -> None:
    if not metrics:
        raise ActionAblationError(f"{section} must not be empty")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ActionAblationError(
                f"{section} metric names must be non-empty strings"
            )
        _finite_float(value, f"{section}.{key}")


def _optional_float_mapping(value: Any, section: str) -> Mapping[str, float] | None:
    if value is None:
        return None
    payload = _require_mapping(value, section)
    return {
        str(key): _finite_float(item, f"{section}.{key}")
        for key, item in payload.items()
    }


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionAblationError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ActionAblationError(f"{field} must be finite")
    return value


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    if key not in payload:
        raise ActionAblationError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise ActionAblationError(f"{section}.{key} must be a string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str, section: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ActionAblationError(f"{section}.{key} must be a string")
    return value


def _optional_mapping(value: Any, section: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _require_mapping(value, section)


def _require_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionAblationError(f"{section} must be a JSON object")
    return value


def _require_literal(
    payload: Mapping[str, Any], key: str, section: str, allowed: set[str]
) -> Any:
    value = _require_string(payload, key, section)
    if value not in allowed:
        raise ActionAblationError(
            f"{section}.{key} must be one of: {', '.join(sorted(allowed))}"
        )
    return value


def _ensure_json_native(payload: Any, section: str) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ActionAblationError(f"{section} must be JSON-native: {exc}") from exc


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
