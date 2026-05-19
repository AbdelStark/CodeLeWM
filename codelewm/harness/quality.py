"""Scorer and reranker quality report artifacts."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)

from .scorer import (
    ErrorReport,
    RerankResult,
    ScoreResult,
    load_scorer,
)


SCORER_QUALITY_CONFIG_SCHEMA_VERSION = "codelewm.harness.scorer_quality_config.v1"
SCORER_QUALITY_REPORT_SCHEMA_VERSION = "codelewm.harness.scorer_quality_report.v1"
SCORER_QUALITY_RUN_SCHEMA_VERSION = "codelewm.harness.scorer_quality_run.v1"

CandidateKind = Literal[
    "true_after",
    "hard_negative",
    "syntax_failure",
    "patch_failure",
    "other",
]
_CANDIDATE_KINDS = {
    "true_after",
    "hard_negative",
    "syntax_failure",
    "patch_failure",
    "other",
}


class ScorerQualityError(ValueError):
    """Raised when scorer quality inputs or reports are invalid."""


@dataclass(frozen=True)
class ScorerQualityExampleConfig:
    """One reranking quality example."""

    example_id: str
    before: str
    instruction: str
    candidates_dir: str
    true_candidate: str
    candidate_kinds: Mapping[str, CandidateKind]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScorerQualityExampleConfig":
        kinds = _require_mapping(payload.get("candidate_kinds", {}), "candidate_kinds")
        return cls(
            example_id=_require_string(payload, "id", "example"),
            before=_require_string(payload, "before", "example"),
            instruction=_require_string(payload, "instruction", "example"),
            candidates_dir=_require_string(payload, "candidates_dir", "example"),
            true_candidate=_require_string(payload, "true_candidate", "example"),
            candidate_kinds={
                str(name): _candidate_kind(value) for name, value in kinds.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "before": self.before,
            "instruction": self.instruction,
            "candidates_dir": self.candidates_dir,
            "true_candidate": self.true_candidate,
            "candidate_kinds": dict(self.candidate_kinds),
        }


@dataclass(frozen=True)
class ScorerQualityConfig:
    """Config for scorer/reranker quality evaluation."""

    examples: tuple[ScorerQualityExampleConfig, ...]
    schema_version: str = SCORER_QUALITY_CONFIG_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScorerQualityConfig":
        schema_version = _require_string(payload, "schema_version", "quality config")
        if schema_version != SCORER_QUALITY_CONFIG_SCHEMA_VERSION:
            raise ScorerQualityError(
                "schema_version must be "
                f"{SCORER_QUALITY_CONFIG_SCHEMA_VERSION!r}; got {schema_version!r}"
            )
        raw_examples = payload.get("examples")
        if not isinstance(raw_examples, Sequence) or isinstance(
            raw_examples, (str, bytes)
        ):
            raise ScorerQualityError("examples must be a JSON array")
        examples = tuple(
            ScorerQualityExampleConfig.from_dict(_require_mapping(item, "examples[]"))
            for item in raw_examples
        )
        if not examples:
            raise ScorerQualityError("examples must not be empty")
        ids = [example.example_id for example in examples]
        if len(set(ids)) != len(ids):
            raise ScorerQualityError("example ids must be unique")
        return cls(schema_version=schema_version, examples=examples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "examples": [example.to_dict() for example in self.examples],
        }


@dataclass(frozen=True)
class ScorerQualityRunResult:
    """CLI result for a scorer quality report artifact."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    parent_artifacts: tuple[str, ...] = ()
    schema_version: str = SCORER_QUALITY_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "parent_artifacts": list(self.parent_artifacts),
        }


def run_scorer_quality_evaluation(
    *,
    config: Path | str,
    checkpoint: Path | str,
    out: Path | str,
    device: str = "auto",
    index: Path | str | None = None,
    retrieval_prior_weight: float = 0.0,
    retrieval_prior_k: int = 10,
    parent_manifests: Sequence[Path | str] = (),
    allow_unsafe_checkpoint: bool = False,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "eval", "scorer-quality"),
) -> ScorerQualityRunResult:
    """Run scorer/reranker quality evaluation and materialize an artifact."""

    config_path = Path(config)
    output_dir = Path(out)
    report_path = output_dir / "reports" / "scorer_quality_report.json"
    copied_config_path = output_dir / "config.json"
    artifact_manifest_path = output_dir / "manifest.json"
    if not overwrite and (report_path.exists() or artifact_manifest_path.exists()):
        raise ScorerQualityError(
            f"output already exists; pass overwrite=True to replace: {output_dir}"
        )

    quality_config = read_scorer_quality_config(config_path)
    parent_artifacts = _read_parent_artifact_ids(parent_manifests)
    scorer = load_scorer(
        checkpoint,
        device=device,
        allow_unsafe=allow_unsafe_checkpoint,
        index=index,
        retrieval_prior_weight=retrieval_prior_weight,
        retrieval_prior_k=retrieval_prior_k,
    )
    examples = [
        _evaluate_example(example, config_dir=config_path.parent, scorer=scorer)
        for example in quality_config.examples
    ]
    report = _build_report(
        examples,
        checkpoint_path=Path(checkpoint),
        checkpoint_sha256=scorer.checkpoint_sha256,
        model_id=scorer.model_id,
        index=index,
        retrieval_prior_weight=retrieval_prior_weight,
        retrieval_prior_k=retrieval_prior_k,
        config_path=config_path,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    copied_config_path.write_text(
        json.dumps(quality_config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_manifest = build_artifact_manifest(
        artifact_kind="score_report",
        root=output_dir,
        files=(copied_config_path, report_path),
        command=command,
        config={
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "device": device,
            "index": None if index is None else str(index),
            "retrieval_prior_weight": retrieval_prior_weight,
            "retrieval_prior_k": retrieval_prior_k,
            "parent_manifests": [str(path) for path in parent_manifests],
            "allow_unsafe_checkpoint": allow_unsafe_checkpoint,
        },
        parent_artifacts=parent_artifacts,
        metadata={
            "schema_version": SCORER_QUALITY_REPORT_SCHEMA_VERSION,
            "example_count": report["summary"]["example_count"],
            "candidate_count": report["summary"]["candidate_count"],
            "valid_count": report["summary"]["valid_count"],
            "error_count": report["summary"]["error_count"],
        },
    )
    write_artifact_manifest(artifact_manifest, artifact_manifest_path)
    return ScorerQualityRunResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path=_relative_to_root(report_path, output_dir),
        parent_artifacts=parent_artifacts,
    )


def read_scorer_quality_config(path: Path | str) -> ScorerQualityConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ScorerQualityError("scorer quality config must be a JSON object")
    return ScorerQualityConfig.from_dict(payload)


def read_scorer_quality_report(path: Path | str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ScorerQualityError("scorer quality report must be a JSON object")
    if payload.get("schema_version") != SCORER_QUALITY_REPORT_SCHEMA_VERSION:
        raise ScorerQualityError(
            "schema_version must be "
            f"{SCORER_QUALITY_REPORT_SCHEMA_VERSION!r}; got {payload.get('schema_version')!r}"
        )
    _validate_summary(_require_mapping(payload.get("summary"), "summary"))
    return payload


def _evaluate_example(
    example: ScorerQualityExampleConfig,
    *,
    config_dir: Path,
    scorer,
) -> dict[str, Any]:
    before = _resolve_config_path(example.before, config_dir)
    candidates_dir = _resolve_config_path(example.candidates_dir, config_dir)
    if not candidates_dir.is_dir():
        raise ScorerQualityError(f"candidates_dir is not a directory: {candidates_dir}")
    true_candidate = (candidates_dir / example.true_candidate).resolve()
    if not true_candidate.is_file():
        raise ScorerQualityError(f"true_candidate does not exist: {true_candidate}")

    rerank = scorer.rerank_files(
        before=before,
        instruction=example.instruction,
        candidates=candidates_dir,
    )
    candidate_rows = _candidate_rows(
        rerank,
        candidates_dir=candidates_dir,
        true_candidate=true_candidate,
        candidate_kinds=example.candidate_kinds,
    )
    true_rows = [
        row
        for row in candidate_rows
        if row["kind"] == "true_after" and row["status"] == "scored"
    ]
    true_rank = None if not true_rows else true_rows[0]["rank"]
    valid_count = sum(row["status"] == "scored" for row in candidate_rows)
    error_counts = Counter(
        str(row["error_type"])
        for row in candidate_rows
        if row["status"] == "error" and row.get("error_type")
    )
    caveats = []
    if true_rank is None:
        caveats.append("true candidate was not scored")
    if error_counts:
        caveats.append("one or more candidates failed parse or patch validation")
    return {
        "id": example.example_id,
        "before": _display_path(before),
        "candidates_dir": _display_path(candidates_dir),
        "true_candidate": _display_path(true_candidate),
        "candidate_count": len(candidate_rows),
        "valid_count": valid_count,
        "error_count": len(candidate_rows) - valid_count,
        "error_counts": dict(sorted(error_counts.items())),
        "true_rank": true_rank,
        "candidate_rows": candidate_rows,
        "caveats": caveats,
    }


def _candidate_rows(
    rerank: RerankResult,
    *,
    candidates_dir: Path,
    true_candidate: Path,
    candidate_kinds: Mapping[str, CandidateKind],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rank = 0
    for item in rerank.results:
        if isinstance(item, ScoreResult):
            rank += 1
            candidate = Path(item.candidate).resolve()
            name = _candidate_name(candidate, candidates_dir)
            kind = (
                "true_after"
                if candidate == true_candidate
                else candidate_kinds.get(name, "other")
            )
            rows.append(
                {
                    "candidate": name,
                    "kind": kind,
                    "status": "scored",
                    "rank": rank,
                    "transition_energy": item.transition_energy,
                    "retrieval_prior": item.retrieval_prior,
                    "risk_penalty": item.risk_penalty,
                    "final_score": item.final_score,
                    "input_digest": item.input_digest,
                }
            )
        elif isinstance(item, ErrorReport):
            artifact = (
                ""
                if item.artifact is None
                else _candidate_name(Path(item.artifact), candidates_dir)
            )
            kind = candidate_kinds.get(artifact, "other")
            rows.append(
                {
                    "candidate": artifact,
                    "kind": kind,
                    "status": "error",
                    "rank": None,
                    "error_type": item.error_type,
                    "message": item.message,
                    "remediation": item.remediation,
                }
            )
        else:
            raise ScorerQualityError("rerank results contain an unsupported row type")
    return rows


def _read_parent_artifact_ids(
    parent_manifests: Sequence[Path | str],
) -> tuple[str, ...]:
    parent_artifacts: list[str] = []
    for manifest_path in parent_manifests:
        path = Path(manifest_path)
        manifest = read_artifact_manifest(path)
        validate_artifact_checksums(manifest, root=path.parent)
        parent_artifacts.append(manifest.artifact_id)
    if len(set(parent_artifacts)) != len(parent_artifacts):
        raise ScorerQualityError(
            "parent_manifests must not contain duplicate artifact ids"
        )
    return tuple(parent_artifacts)


def _build_report(
    examples: Sequence[Mapping[str, Any]],
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    model_id: str,
    index: Path | str | None,
    retrieval_prior_weight: float,
    retrieval_prior_k: int,
    config_path: Path,
) -> dict[str, Any]:
    candidate_rows = [row for example in examples for row in example["candidate_rows"]]
    scored_rows = [row for row in candidate_rows if row["status"] == "scored"]
    error_rows = [row for row in candidate_rows if row["status"] == "error"]
    true_ranks = [
        int(example["true_rank"])
        for example in examples
        if example.get("true_rank") is not None
    ]
    summary = {
        "example_count": len(examples),
        "candidate_count": len(candidate_rows),
        "valid_count": len(scored_rows),
        "error_count": len(error_rows),
        "recall_at_1": _recall_at_1(true_ranks, len(examples)),
        "mrr": _mrr(true_ranks, len(examples)),
        "mean_true_rank": _mean(true_ranks),
        "median_true_rank": _median(true_ranks),
        "failure_counts": dict(
            sorted(Counter(str(row["error_type"]) for row in error_rows).items())
        ),
    }
    report = {
        "schema_version": SCORER_QUALITY_REPORT_SCHEMA_VERSION,
        "source": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "model_id": model_id,
            "index": None if index is None else str(index),
        },
        "scoring_policy": {
            "score_direction": "lower_is_better",
            "retrieval_prior_weight": retrieval_prior_weight,
            "retrieval_prior_k": retrieval_prior_k,
            "risk_penalty": "reserved; current scorer reports null and treats failures as error rows",
            "execution_policy": "candidate code is parsed and diff-applied as text but never executed",
        },
        "summary": summary,
        "score_distributions": {
            "transition_energy": _distribution(
                row["transition_energy"] for row in scored_rows
            ),
            "retrieval_prior": _distribution(
                row["retrieval_prior"]
                for row in scored_rows
                if row.get("retrieval_prior") is not None
            ),
            "final_score": _distribution(row["final_score"] for row in scored_rows),
        },
        "calibration_slices": _calibration_slices(candidate_rows),
        "examples": list(examples),
        "caveats": _report_caveats(examples, retrieval_prior_weight),
    }
    _validate_summary(summary)
    _ensure_json_native(report, "scorer quality report")
    return report


def _calibration_slices(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(str(row["kind"]), []).append(row)
    slices: dict[str, dict[str, Any]] = {}
    for kind, kind_rows in sorted(by_kind.items()):
        scored = [row for row in kind_rows if row["status"] == "scored"]
        errors = [row for row in kind_rows if row["status"] == "error"]
        slices[kind] = {
            "candidate_count": len(kind_rows),
            "valid_count": len(scored),
            "error_count": len(errors),
            "mean_transition_energy": _mean(row["transition_energy"] for row in scored),
            "mean_retrieval_prior": _mean(
                row["retrieval_prior"]
                for row in scored
                if row.get("retrieval_prior") is not None
            ),
            "mean_final_score": _mean(row["final_score"] for row in scored),
            "failure_counts": dict(
                sorted(Counter(str(row["error_type"]) for row in errors).items())
            ),
        }
    return slices


def _report_caveats(
    examples: Sequence[Mapping[str, Any]], retrieval_prior_weight: float
) -> list[str]:
    caveats = [
        "Fixture quality reports validate scorer/reranker plumbing, not scaled model usefulness.",
        "Lower final_score is better.",
    ]
    if retrieval_prior_weight == 0.0:
        caveats.append(
            "Retrieval priors may be computed but do not affect final_score at weight 0."
        )
    if any(example.get("caveats") for example in examples):
        caveats.append(
            "At least one example includes invalid syntax or patch-application failure rows."
        )
    return caveats


def _distribution(values: Any) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if _is_finite_number(value))
    if not finite:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": len(finite),
        "min": finite[0],
        "p25": _quantile(finite, 0.25),
        "median": _quantile(finite, 0.5),
        "p75": _quantile(finite, 0.75),
        "max": finite[-1],
        "mean": statistics.fmean(finite),
        "std": 0.0 if len(finite) == 1 else statistics.pstdev(finite),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _recall_at_1(ranks: Sequence[int], total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(1 for rank in ranks if rank == 1) / total


def _mrr(ranks: Sequence[int], total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(1.0 / rank for rank in ranks) / total


def _mean(values: Any) -> float | None:
    finite = [float(value) for value in values if _is_finite_number(value)]
    if not finite:
        return None
    return statistics.fmean(finite)


def _median(values: Any) -> float | None:
    finite = sorted(float(value) for value in values if _is_finite_number(value))
    if not finite:
        return None
    return statistics.median(finite)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_summary(summary: Mapping[str, Any]) -> None:
    for key in ("example_count", "candidate_count", "valid_count", "error_count"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ScorerQualityError(f"summary.{key} must be a non-negative integer")
    for key in ("recall_at_1", "mrr"):
        value = summary.get(key)
        if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ScorerQualityError(f"summary.{key} must be in [0, 1]")


def _candidate_kind(value: Any) -> CandidateKind:
    if not isinstance(value, str) or value not in _CANDIDATE_KINDS:
        raise ScorerQualityError(
            "candidate kind must be one of: " + ", ".join(sorted(_CANDIDATE_KINDS))
        )
    return value  # type: ignore[return-value]


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    if key not in payload:
        raise ScorerQualityError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ScorerQualityError(f"{section}.{key} must be a non-empty string")
    return value


def _require_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScorerQualityError(f"{section} must be a JSON object")
    return value


def _resolve_config_path(path: str, config_dir: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    return candidate.resolve()


def _candidate_name(candidate: Path, candidates_dir: Path) -> str:
    try:
        return candidate.resolve().relative_to(candidates_dir.resolve()).as_posix()
    except ValueError:
        return candidate.name


def _display_path(path: Path) -> str:
    return path.as_posix()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _ensure_json_native(payload: Any, section: str) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ScorerQualityError(f"{section} must be JSON-native: {exc}") from exc
