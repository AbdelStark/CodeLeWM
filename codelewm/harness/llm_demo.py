"""End-to-end LLM plus CodeLeWM demo runner."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codelewm.observability import (
    ArtifactManifestError,
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.security.secret_scan import scan_text

from .openrouter_adapter import (
    LLM_CANDIDATE_PACK_SCHEMA_VERSION,
    OpenRouterAdapterError,
    OpenRouterCandidateRequest,
    generate_candidate_pack,
    write_candidate_pack_artifact,
)
from .scorer import ErrorReport, RerankResult, ScoreError, ScoreResult, load_scorer


LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION = "codelewm.harness.demo_report.v1"
LLM_WORLD_MODEL_DEMO_RUN_SCHEMA_VERSION = "codelewm.harness.demo_run.v1"
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")


class LLMWorldModelDemoError(ValueError):
    """Raised when the LLM + CodeLeWM demo cannot be run or materialized."""


@dataclass(frozen=True)
class LLMWorldModelDemoRunResult:
    """Summary returned after writing a manifest-backed demo report."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    candidate_pack_manifest_path: str
    parent_artifacts: tuple[str, ...]
    success: bool
    schema_version: str = LLM_WORLD_MODEL_DEMO_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "candidate_pack_manifest_path": self.candidate_pack_manifest_path,
            "parent_artifacts": list(self.parent_artifacts),
            "success": self.success,
        }


def run_llm_world_model_demo(
    *,
    before: Path | str,
    instruction: str,
    checkpoint: Path | str,
    out: Path | str,
    task_id: str = "codelewm-demo",
    context_path: str | None = None,
    env: Mapping[str, str] | None = None,
    device: str = "auto",
    index: Path | str | None = None,
    retrieval_prior_weight: float = 0.0,
    retrieval_prior_k: int = 10,
    parent_manifests: Sequence[Path | str] = (),
    allow_unsafe_checkpoint: bool = False,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "llm-demo"),
) -> LLMWorldModelDemoRunResult:
    """Run candidate generation, candidate-pack capture, reranking, and report writing."""

    before_path = Path(before)
    checkpoint_path = Path(checkpoint)
    output_dir = Path(out).resolve()
    report_path = output_dir / "reports" / "llm_world_model_demo_report.json"
    manifest_path = output_dir / "manifest.json"
    candidate_pack_dir = output_dir / "candidate_pack"
    if not overwrite and (report_path.exists() or manifest_path.exists() or candidate_pack_dir.exists()):
        raise LLMWorldModelDemoError(
            f"output already exists; pass overwrite=True to replace: {output_dir}"
        )

    try:
        before_text = before_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMWorldModelDemoError(f"before file could not be read: {exc}") from exc
    context_key = context_path or before_path.name
    try:
        request = OpenRouterCandidateRequest.from_env(
            task_id=task_id,
            instruction=instruction,
            context_bundle={context_key: before_text},
            env=env,
        )
        generated_pack = generate_candidate_pack(request, env=env)
    except OpenRouterAdapterError as exc:
        raise LLMWorldModelDemoError(f"candidate generation failed: {exc}") from exc
    parent_artifact_ids = _read_parent_artifact_ids(parent_manifests)
    try:
        candidate_pack_result = write_candidate_pack_artifact(
            generated_pack,
            candidate_pack_dir,
            parent_artifacts=parent_artifact_ids,
            command=(*command, "candidate-pack"),
            overwrite=overwrite,
        )
    except OpenRouterAdapterError as exc:
        raise LLMWorldModelDemoError(f"candidate-pack capture failed: {exc}") from exc
    candidate_pack_manifest_path = candidate_pack_dir / candidate_pack_result.artifact_manifest_path
    candidate_pack_manifest = read_artifact_manifest(candidate_pack_manifest_path)
    validate_artifact_checksums(candidate_pack_manifest, root=candidate_pack_dir)
    candidate_pack_payload = json.loads(
        (candidate_pack_dir / candidate_pack_result.candidate_pack_path).read_text(encoding="utf-8")
    )

    scorer = load_scorer(
        checkpoint_path,
        device=device,
        index=index,
        retrieval_prior_weight=retrieval_prior_weight,
        retrieval_prior_k=retrieval_prior_k,
        allow_unsafe=allow_unsafe_checkpoint,
    )
    rerank = scorer.rerank_files(
        before=before_path,
        instruction=instruction,
        candidates=candidate_pack_dir / "candidates",
    )
    no_action = scorer.score_texts(
        before=before_text,
        instruction=instruction,
        candidate=before_text,
        candidate_name="no_action",
    )

    report = _build_demo_report(
        task_id=task_id,
        before_path=before_path,
        before_text=before_text,
        context_key=context_key,
        instruction=instruction,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=scorer.checkpoint_sha256,
        model_id=scorer.model_id,
        candidate_pack_manifest_id=candidate_pack_manifest.artifact_id,
        candidate_pack_manifest_path=f"candidate_pack/{candidate_pack_result.artifact_manifest_path}",
        candidate_pack_payload=candidate_pack_payload,
        rerank=rerank,
        no_action=no_action,
        index=index,
        retrieval_prior_weight=retrieval_prior_weight,
        retrieval_prior_k=retrieval_prior_k,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_scan = scan_text(report_path.read_text(encoding="utf-8"), path="llm_world_model_demo_report.json")
    if report_scan:
        raise LLMWorldModelDemoError("demo report contains secret-scan findings after redaction")

    artifact_manifest = build_artifact_manifest(
        artifact_kind="demo_report",
        root=output_dir,
        files=(report_path,),
        command=command,
        config={
            "task_id": task_id,
            "before": str(before_path),
            "checkpoint": str(checkpoint_path),
            "context_path": context_key,
            "device": device,
            "index": None if index is None else str(index),
            "retrieval_prior_weight": retrieval_prior_weight,
            "retrieval_prior_k": retrieval_prior_k,
            "allow_unsafe_checkpoint": allow_unsafe_checkpoint,
        },
        parent_artifacts=(candidate_pack_manifest.artifact_id, *parent_artifact_ids),
        metadata={
            "schema_version": LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION,
            "success": report["success"],
            "candidate_count": report["candidate_summary"]["candidate_count"],
            "valid_candidate_count": report["candidate_summary"]["valid_candidate_count"],
            "claim_allowed": report["claim_gate"]["allowed"],
        },
    )
    write_artifact_manifest(artifact_manifest, manifest_path)
    return LLMWorldModelDemoRunResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path=_relative_to_root(report_path, output_dir),
        candidate_pack_manifest_path=f"candidate_pack/{candidate_pack_result.artifact_manifest_path}",
        parent_artifacts=(candidate_pack_manifest.artifact_id, *parent_artifact_ids),
        success=bool(report["success"]),
    )


def read_llm_world_model_demo_report(path: Path | str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LLMWorldModelDemoError("demo report must be a JSON object")
    if payload.get("schema_version") != LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION:
        raise LLMWorldModelDemoError("unsupported demo report schema_version")
    return payload


def _build_demo_report(
    *,
    task_id: str,
    before_path: Path,
    before_text: str,
    context_key: str,
    instruction: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    model_id: str,
    candidate_pack_manifest_id: str,
    candidate_pack_manifest_path: str,
    candidate_pack_payload: Mapping[str, Any],
    rerank: RerankResult,
    no_action: ScoreResult,
    index: Path | str | None,
    retrieval_prior_weight: float,
    retrieval_prior_k: int,
) -> dict[str, Any]:
    candidates = list(candidate_pack_payload.get("candidates", []))
    candidate_ids = [str(candidate.get("candidate_id")) for candidate in candidates]
    valid_candidate_ids = [
        str(candidate.get("candidate_id"))
        for candidate in candidates
        if not candidate.get("errors") and candidate.get("parser_status") == "parseable_python_after_state"
    ]
    result_payloads = [item.to_dict() for item in rerank.results]
    codelewm_order = [
        _candidate_id_from_path(item.candidate)
        for item in rerank.results
        if isinstance(item, ScoreResult)
    ]
    rerank_errors = [
        item.to_dict()
        for item in rerank.results
        if isinstance(item, ErrorReport)
    ]
    candidate_errors = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "errors": candidate.get("errors", []),
            "parser_status": candidate.get("parser_status"),
            "dry_run_patch_status": candidate.get("dry_run_patch_status"),
        }
        for candidate in candidates
        if candidate.get("errors")
    ]
    success_reasons = []
    failure_reasons = []
    if len(candidate_ids) >= 2:
        success_reasons.append("at_least_two_candidates")
    else:
        failure_reasons.append("fewer_than_two_candidates")
    if valid_candidate_ids:
        success_reasons.append("at_least_one_valid_candidate")
    else:
        failure_reasons.append("zero_valid_candidates")
    if codelewm_order:
        success_reasons.append("codelewm_rerank_completed")
    else:
        failure_reasons.append("codelewm_rerank_no_scores")

    return {
        "schema_version": LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION,
        "success": not failure_reasons,
        "success_reasons": success_reasons,
        "failure_reasons": failure_reasons,
        "task": {
            "task_id": task_id,
            "instruction_sha256": _sha256_text(instruction),
            "context_path": context_key,
            "before_path": str(before_path),
            "before_sha256": _sha256_text(before_text),
        },
        "artifacts": {
            "candidate_pack_manifest_id": candidate_pack_manifest_id,
            "candidate_pack_manifest_path": candidate_pack_manifest_path,
            "demo_manifest_path": "manifest.json",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "transition_index": None if index is None else str(index),
        },
        "generator": dict(candidate_pack_payload.get("generator", {})),
        "candidate_summary": {
            "candidate_count": len(candidate_ids),
            "valid_candidate_count": len(valid_candidate_ids),
            "error_candidate_count": len(candidate_errors),
            "candidate_ids": candidate_ids,
            "valid_candidate_ids": valid_candidate_ids,
        },
        "orders": {
            "llm": candidate_ids,
            "codelewm": codelewm_order,
            "random": _deterministic_random_order(candidate_ids, task_id=task_id),
            "lexical": _lexical_order(candidates, instruction),
            "no_action": ["no_action"],
        },
        "scores": {
            "codelewm_rerank": result_payloads,
            "no_action": no_action.to_dict(),
            "model_id": model_id,
            "retrieval_prior_weight": retrieval_prior_weight,
            "retrieval_prior_k": retrieval_prior_k,
        },
        "baselines": {
            "llm_order": {"status": "completed", "order": candidate_ids},
            "random": {
                "status": "completed",
                "order": _deterministic_random_order(candidate_ids, task_id=task_id),
            },
            "lexical": {"status": "completed", "order": _lexical_order(candidates, instruction)},
            "no_action": {"status": "completed", "score": no_action.to_dict()},
        },
        "candidate_errors": candidate_errors,
        "rerank_errors": rerank_errors,
        "static_checks": {
            "status": "not_configured",
            "outcomes": [],
        },
        "claim_gate": {
            "allowed": False,
            "reason": "demo_report_is_not_downstream_benchmark_evidence",
            "required_next_issue": "#192",
        },
        "warnings": list(rerank.warnings),
        "candidate_pack_schema": LLM_CANDIDATE_PACK_SCHEMA_VERSION,
    }


def _read_parent_artifact_ids(parent_manifests: Sequence[Path | str]) -> tuple[str, ...]:
    parent_artifacts: list[str] = []
    for path in parent_manifests:
        manifest_path = Path(path)
        try:
            manifest = read_artifact_manifest(manifest_path)
            validate_artifact_checksums(manifest, root=manifest_path.parent)
        except ArtifactManifestError as exc:
            raise LLMWorldModelDemoError(f"parent manifest validation failed: {exc}") from exc
        parent_artifacts.append(manifest.artifact_id)
    return tuple(parent_artifacts)


def _candidate_id_from_path(path: str) -> str:
    name = Path(path).name
    for suffix in (".patch", ".diff", ".py"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _deterministic_random_order(candidate_ids: Sequence[str], *, task_id: str) -> list[str]:
    return sorted(candidate_ids, key=lambda item: _sha256_text(f"{task_id}:{item}"))


def _lexical_order(candidates: Sequence[Mapping[str, Any]], instruction: str) -> list[str]:
    instruction_tokens = set(_tokens(instruction))

    def key(candidate: Mapping[str, Any]) -> tuple[int, str]:
        candidate_tokens = set(_tokens(str(candidate.get("patch_text", ""))))
        overlap = len(instruction_tokens & candidate_tokens)
        return (-overlap, str(candidate.get("candidate_id")))

    return [str(candidate.get("candidate_id")) for candidate in sorted(candidates, key=key)]


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
