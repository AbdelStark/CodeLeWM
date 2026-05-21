"""End-to-end LLM plus CodeLeWM demo runner."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codelewm.observability import (
    ArtifactManifestError,
    RUN_TIMELINE_SCHEMA_VERSION,
    RunTimelineRecorder,
    build_artifact_manifest,
    read_artifact_manifest,
    sha256_file,
    validate_artifact_checksums,
    write_run_timeline_report,
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
from .visual_view_model import (
    DEMO_VISUAL_VIEW_MODEL_SCHEMA_VERSION,
    build_demo_visual_view_model,
    write_demo_visual_view_model,
)


LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION = "codelewm.harness.demo_report.v1"
LLM_WORLD_MODEL_DEMO_RUN_SCHEMA_VERSION = "codelewm.harness.demo_run.v1"
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
_MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION = "codelewm.model_checkpoint_inspection.v1"
_LATENT_MATRIX_REPORT_SCHEMA_VERSION = "codelewm.eval.latent_matrix_report.v1"
_TENSORBOARD_EXPORT_SCHEMA_VERSION = "codelewm.training.tensorboard_export.v1"


class LLMWorldModelDemoError(ValueError):
    """Raised when the LLM + CodeLeWM demo cannot be run or materialized."""


@dataclass(frozen=True)
class LLMWorldModelDemoRunResult:
    """Summary returned after writing a manifest-backed demo report."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    html_path: str
    visual_view_model_path: str
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
            "html_path": self.html_path,
            "visual_view_model_path": self.visual_view_model_path,
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
    checkpoint_inspection_manifest: Path | str | None = None,
    checkpoint_inspection_report: Path | str | None = None,
    latent_matrix_manifest: Path | str | None = None,
    latent_matrix_report: Path | str | None = None,
    tensorboard_manifest: Path | str | None = None,
    tensorboard_export: Path | str | None = None,
    allow_unsafe_checkpoint: bool = False,
    require_learned_scorer: bool = False,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "llm-demo"),
) -> LLMWorldModelDemoRunResult:
    """Run candidate generation, candidate-pack capture, reranking, and report writing."""

    before_path = Path(before)
    checkpoint_path = Path(checkpoint)
    output_dir = Path(out).resolve()
    report_path = output_dir / "reports" / "llm_world_model_demo_report.json"
    visual_view_model_path = output_dir / "reports" / "visual_view_model.json"
    timeline_path = output_dir / "reports" / "run_timeline.json"
    html_path = output_dir / "demo.html"
    manifest_path = output_dir / "manifest.json"
    candidate_pack_dir = output_dir / "candidate_pack"
    if not overwrite and (
        report_path.exists()
        or visual_view_model_path.exists()
        or timeline_path.exists()
        or html_path.exists()
        or manifest_path.exists()
        or candidate_pack_dir.exists()
    ):
        raise LLMWorldModelDemoError(
            f"output already exists; pass overwrite=True to replace: {output_dir}"
        )

    timeline = RunTimelineRecorder(run_id=task_id, command=command)
    with timeline.step("read inputs", command_id="llm_demo.read_inputs"):
        try:
            before_text = before_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LLMWorldModelDemoError(f"before file could not be read: {exc}") from exc
        context_key = context_path or before_path.name
    with timeline.step("candidate generation", command_id="llm_demo.candidate_generation"):
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
    with timeline.step("parent manifest validation", command_id="llm_demo.parent_manifests"):
        parent_artifact_ids = _read_parent_artifact_ids(parent_manifests)
    with timeline.step("diagnostic artifact validation", command_id="llm_demo.diagnostics"):
        diagnostics = _build_demo_diagnostics(
            checkpoint_inspection_manifest=checkpoint_inspection_manifest,
            checkpoint_inspection_report=checkpoint_inspection_report,
            latent_matrix_manifest=latent_matrix_manifest,
            latent_matrix_report=latent_matrix_report,
            tensorboard_manifest=tensorboard_manifest,
            tensorboard_export=tensorboard_export,
            run_timeline_path="reports/run_timeline.json",
        )
        diagnostic_parent_artifact_ids = _diagnostic_parent_artifact_ids(diagnostics)
    with timeline.step("candidate pack capture", command_id="llm_demo.candidate_pack") as step:
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
        step.add_artifact(candidate_pack_result.artifact_manifest_id)
        candidate_pack_manifest_path = candidate_pack_dir / candidate_pack_result.artifact_manifest_path
        candidate_pack_manifest = read_artifact_manifest(candidate_pack_manifest_path)
        validate_artifact_checksums(candidate_pack_manifest, root=candidate_pack_dir)
        candidate_pack_payload = json.loads(
            (candidate_pack_dir / candidate_pack_result.candidate_pack_path).read_text(encoding="utf-8")
        )

    with timeline.step("world model scoring", command_id="llm_demo.world_model_scoring"):
        scorer = load_scorer(
            checkpoint_path,
            device=device,
            index=index,
            retrieval_prior_weight=retrieval_prior_weight,
            retrieval_prior_k=retrieval_prior_k,
            allow_unsafe=allow_unsafe_checkpoint,
            require_learned_backend=require_learned_scorer,
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

    with timeline.step("report render", command_id="llm_demo.report_render"):
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
            run_timeline_path="reports/run_timeline.json",
            visual_view_model_path="reports/visual_view_model.json",
            diagnostics=diagnostics,
        )
        view_model = build_demo_visual_view_model(
            demo_report=report,
            candidate_pack=candidate_pack_payload,
            out_dir=output_dir,
        )

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_demo_visual_view_model(view_model, visual_view_model_path)
        html_path.write_text(
            render_llm_world_model_demo_html(
                report, candidate_pack_payload, visual_view_model=view_model
            ),
            encoding="utf-8",
        )
    with timeline.step("artifact secret scan", command_id="llm_demo.secret_scan"):
        report_scan = scan_text(report_path.read_text(encoding="utf-8"), path="llm_world_model_demo_report.json")
        if report_scan:
            raise LLMWorldModelDemoError("demo report contains secret-scan findings after redaction")
        html_scan = scan_text(html_path.read_text(encoding="utf-8"), path="demo.html")
        if html_scan:
            raise LLMWorldModelDemoError("demo HTML contains secret-scan findings after redaction")
        view_model_scan = scan_text(
            visual_view_model_path.read_text(encoding="utf-8"),
            path="visual_view_model.json",
        )
        if view_model_scan:
            raise LLMWorldModelDemoError("demo visual view model contains secret-scan findings")
    write_run_timeline_report(
        timeline.to_report(
            artifact_ids=(
                candidate_pack_manifest.artifact_id,
                *parent_artifact_ids,
                *diagnostic_parent_artifact_ids,
            ),
            metadata={
                "report_path": "reports/llm_world_model_demo_report.json",
                "visual_view_model_path": "reports/visual_view_model.json",
                "html_path": "demo.html",
            },
        ),
        timeline_path,
    )
    timeline_scan = scan_text(timeline_path.read_text(encoding="utf-8"), path="run_timeline.json")
    if timeline_scan:
        raise LLMWorldModelDemoError("demo timeline contains secret-scan findings after redaction")

    artifact_manifest = build_artifact_manifest(
        artifact_kind="demo_report",
        root=output_dir,
        files=(report_path, html_path, visual_view_model_path, timeline_path),
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
            "require_learned_scorer": require_learned_scorer,
            "diagnostics": _diagnostic_config(diagnostics),
        },
        parent_artifacts=(
            candidate_pack_manifest.artifact_id,
            *parent_artifact_ids,
            *diagnostic_parent_artifact_ids,
        ),
        metadata={
            "schema_version": LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION,
            "visual_view_model_schema_version": DEMO_VISUAL_VIEW_MODEL_SCHEMA_VERSION,
            "visual_view_model_path": "reports/visual_view_model.json",
            "run_timeline_schema_version": RUN_TIMELINE_SCHEMA_VERSION,
            "run_timeline_path": "reports/run_timeline.json",
            "diagnostics": diagnostics,
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
        html_path=_relative_to_root(html_path, output_dir),
        visual_view_model_path=_relative_to_root(visual_view_model_path, output_dir),
        candidate_pack_manifest_path=f"candidate_pack/{candidate_pack_result.artifact_manifest_path}",
        parent_artifacts=(
            candidate_pack_manifest.artifact_id,
            *parent_artifact_ids,
            *diagnostic_parent_artifact_ids,
        ),
        success=bool(report["success"]),
    )


def read_llm_world_model_demo_report(path: Path | str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LLMWorldModelDemoError("demo report must be a JSON object")
    if payload.get("schema_version") != LLM_WORLD_MODEL_DEMO_REPORT_SCHEMA_VERSION:
        raise LLMWorldModelDemoError("unsupported demo report schema_version")
    return payload


def render_llm_world_model_demo_html(
    report: Mapping[str, Any],
    candidate_pack_payload: Mapping[str, Any],
    *,
    visual_view_model: Mapping[str, Any] | None = None,
) -> str:
    """Render a self-contained visual report for the LLM plus world-model demo."""

    view_model = (
        visual_view_model
        if visual_view_model is not None
        else build_demo_visual_view_model(
            demo_report=report,
            candidate_pack=candidate_pack_payload,
            out_dir=".",
        )
    )
    summary = _mapping(view_model.get("summary"))
    generator = _mapping(view_model.get("generator"))
    orders_payload = _mapping(view_model.get("orders"))
    diagnostics = _mapping(view_model.get("diagnostics"))
    candidates = [
        candidate for candidate in view_model.get("candidates", []) if isinstance(candidate, Mapping)
    ]
    codelewm_order = _string_list(orders_payload.get("codelewm", ()))
    llm_order = _string_list(orders_payload.get("llm", ()))
    lexical_order = _string_list(orders_payload.get("lexical", ()))
    random_order = _string_list(orders_payload.get("random", ()))
    dry_run = bool(summary.get("dry_run"))
    mode = str(summary.get("mode", "fixture dry-run" if dry_run else "live OpenRouter"))
    status_sentence = (
        "In this run the LLM side used deterministic fixture candidates because dry-run mode was enabled."
        if dry_run
        else "In this run the LLM side called OpenRouter and captured provider output as untrusted patches."
    )
    status_class = "warn" if dry_run else "ok"
    claim_gate = _mapping(view_model.get("claim_gate"))
    max_score = max(
        (
            float(candidate["score"])
            for candidate in candidates
            if isinstance(candidate.get("score"), (int, float))
        ),
        default=1.0,
    )

    rows = "\n".join(
        _candidate_row(candidate, max_score=max_score)
        for candidate in candidates
    )
    patch_cards = "\n".join(_patch_card(candidate) for candidate in candidates)
    diagnostic_cards = _diagnostics_markup(diagnostics)
    orders = _orders_markup(
        (
            ("LLM order", llm_order),
            ("CodeLeWM order", codelewm_order),
            ("Lexical", lexical_order),
            ("Random", random_order),
        )
    )
    css = _demo_html_css()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CodeLeWM LLM World-Model Demo</title>
<style>
{css}
</style>
</head>
<body>
<header class="hero">
  <div class="wrap hero-grid">
    <div>
      <div class="ident">CodeLeWM / LLM candidate rerank</div>
      <h1>Visual demo report</h1>
      <p class="deck">This page shows what the demo actually did: how candidates were produced, how CodeLeWM ranked them, and why the claim gate stays closed.</p>
      <div class="pill-row">
        <span class="pill {status_class}">{_h(mode)}</span>
        <span class="pill">success: {_h(str(summary.get("success", False)).lower())}</span>
        <span class="pill">claim: {_h(str(claim_gate.get("allowed", False)).lower())}</span>
      </div>
    </div>
    <aside class="stat-panel">
      {_stat("model", generator.get("model"))}
      {_stat("provider", generator.get("provider"))}
      {_stat("sdk version", generator.get("sdk_version") or "not loaded")}
      {_stat("provider mode", "fixture" if dry_run else "live")}
      {_stat("BYOK", "enabled" if generator.get("byok_enabled") else "disabled")}
      {_stat("candidates", summary.get("candidate_count"))}
      {_stat("valid", summary.get("valid_candidate_count"))}
      {_stat("scorer", summary.get("scorer"))}
      {_stat("score direction", summary.get("score_direction"))}
    </aside>
  </div>
</header>

<main>
  <section class="s" id="status" data-num="01">
    <div class="wrap">
      <div class="section-head"><span class="section-num">01</span><span class="section-kind">status</span></div>
      <h2>Did it <em>work</em>?</h2>
      <p class="s-deck">Yes, the artifact path worked. {_h(status_sentence)}</p>
      <div class="grid-3">
        {_mini("Generation", mode, "CODELEWM_LLM_DRY_RUN controls whether the adapter calls OpenRouter.")}
        {_mini("Reranking", "completed", "Candidates were scored and ordered by the CodeLeWM harness.")}
        {_mini("Publication gate", str(claim_gate.get("allowed", False)).lower(), str(claim_gate.get("reason", "unknown")))}
      </div>
    </div>
  </section>

  <section class="s" id="ranking" data-num="02">
    <div class="wrap">
      <div class="section-head"><span class="section-num">02</span><span class="section-kind">ranking</span></div>
      <h2>Candidate scores, <em>lower</em> is better.</h2>
      <p class="s-deck">The bars show transition energy from the current scorer backend. Candidate-minus-no-action deltas are better when negative and worse when positive.</p>
      <table class="rank-table">
        <thead><tr><th>rank</th><th>candidate</th><th>score</th><th>vs no-op</th><th>status</th><th>bar</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </section>

  <section class="s" id="orders" data-num="03">
    <div class="wrap">
      <div class="section-head"><span class="section-num">03</span><span class="section-kind">baselines</span></div>
      <h2>Orders side by <em>side</em>.</h2>
      <p class="s-deck">This is the tangible harness use case: compare the LLM's original order with CodeLeWM and cheap baselines.</p>
      <div class="order-grid">{orders}</div>
    </div>
  </section>

  <section class="s" id="patches" data-num="04">
    <div class="wrap">
      <div class="section-head"><span class="section-num">04</span><span class="section-kind">patches</span></div>
      <h2>Compact candidate <em>diffs</em>.</h2>
      <p class="s-deck">Candidate code is treated as untrusted text. The shared view model shows bounded diff summaries; full patches stay in local artifacts and are not executed.</p>
      <div class="patch-grid">{patch_cards}</div>
    </div>
  </section>

  <section class="s" id="diagnostics" data-num="05">
    <div class="wrap">
      <div class="section-head"><span class="section-num">05</span><span class="section-kind">diagnostics</span></div>
      <h2>Model and latent <em>links</em>.</h2>
      <p class="s-deck">The same normalized view model feeds JSON, terminal, HTML, and the future Textual TUI. Missing diagnostics stay explicit.</p>
      <div class="grid-3">{diagnostic_cards}</div>
    </div>
  </section>

  <section class="s" id="next" data-num="06">
    <div class="wrap">
      <div class="section-head"><span class="section-num">06</span><span class="section-kind">next run</span></div>
      <h2>Make the next run <em>live</em>.</h2>
      <p class="s-deck">Set the dry-run flags to zero when you want provider output instead of fixture candidates.</p>
      <pre class="code">CODELEWM_LLM_DRY_RUN=0
CODELEWM_OPENROUTER_BYOK=1
CODELEWM_OPENROUTER_BYOK_REGISTER=0
uv run scripts/llm-world-model-demo</pre>
      <p class="footnote">Keep token values only in .env. Set OPENROUTER_MANAGEMENT_KEY and CODELEWM_OPENROUTER_BYOK_REGISTER=1 only when CodeLeWM should create the BYOK credential.</p>
    </div>
  </section>
</main>
</body>
</html>
"""


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
    run_timeline_path: str,
    visual_view_model_path: str,
    diagnostics: Mapping[str, Any],
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
            "visual_view_model_path": visual_view_model_path,
            "run_timeline_path": run_timeline_path,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "transition_index": None if index is None else str(index),
        },
        "diagnostics": dict(diagnostics),
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
            "score_direction": "lower_is_better",
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


def _build_demo_diagnostics(
    *,
    checkpoint_inspection_manifest: Path | str | None,
    checkpoint_inspection_report: Path | str | None,
    latent_matrix_manifest: Path | str | None,
    latent_matrix_report: Path | str | None,
    tensorboard_manifest: Path | str | None,
    tensorboard_export: Path | str | None,
    run_timeline_path: str,
) -> dict[str, dict[str, Any]]:
    return {
        "checkpoint_inspection": _diagnostic_reference(
            name="checkpoint_inspection",
            report_path=checkpoint_inspection_report,
            manifest_path=checkpoint_inspection_manifest,
            expected_schema_version=_MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION,
            default_manifest_report_path="reports/model_checkpoint_inspection.json",
        ),
        "latent_matrix": _diagnostic_reference(
            name="latent_matrix",
            report_path=latent_matrix_report,
            manifest_path=latent_matrix_manifest,
            expected_schema_version=_LATENT_MATRIX_REPORT_SCHEMA_VERSION,
            default_manifest_report_path="reports/latent_matrix_report.json",
        ),
        "tensorboard": _diagnostic_reference(
            name="tensorboard",
            report_path=tensorboard_export,
            manifest_path=tensorboard_manifest,
            expected_schema_version=_TENSORBOARD_EXPORT_SCHEMA_VERSION,
            default_manifest_report_path="reports/tensorboard_export.json",
        ),
        "run_timeline": {
            "status": "available",
            "path": run_timeline_path,
            "schema_version": RUN_TIMELINE_SCHEMA_VERSION,
            "artifact_id": None,
            "artifact_manifest_path": None,
            "sha256": None,
            "bytes": None,
            "source": "current_demo_run",
        },
    }


def _diagnostic_reference(
    *,
    name: str,
    report_path: Path | str | None,
    manifest_path: Path | str | None,
    expected_schema_version: str,
    default_manifest_report_path: str,
) -> dict[str, Any]:
    if report_path is None and manifest_path is None:
        return {
            "status": "not_configured",
            "path": None,
            "schema_version": expected_schema_version,
            "artifact_id": None,
            "artifact_manifest_path": None,
            "sha256": None,
            "bytes": None,
        }

    manifest = None
    manifest_file = None
    resolved_manifest_path: Path | None = None
    if manifest_path is not None:
        resolved_manifest_path = Path(manifest_path)
        try:
            manifest = read_artifact_manifest(resolved_manifest_path)
            validate_artifact_checksums(manifest, root=resolved_manifest_path.parent)
        except (ArtifactManifestError, OSError, json.JSONDecodeError) as exc:
            raise LLMWorldModelDemoError(
                f"{name} diagnostic manifest validation failed: {exc}"
            ) from exc

    if report_path is None:
        if manifest is None or resolved_manifest_path is None:
            raise LLMWorldModelDemoError(f"{name} diagnostic report requires a manifest")
        metadata_path = manifest.metadata.get("report_path") if isinstance(manifest.metadata, Mapping) else None
        report_path = str(metadata_path or default_manifest_report_path)

    resolved_report_path = _resolve_diagnostic_report_path(
        report_path,
        manifest_path=resolved_manifest_path,
    )
    if not resolved_report_path.is_file():
        raise LLMWorldModelDemoError(f"{name} diagnostic report does not exist: {resolved_report_path}")

    try:
        report_text = resolved_report_path.read_text(encoding="utf-8")
        payload = json.loads(report_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMWorldModelDemoError(f"{name} diagnostic report is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise LLMWorldModelDemoError(f"{name} diagnostic report must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != expected_schema_version:
        raise LLMWorldModelDemoError(
            f"{name} diagnostic report schema_version must be {expected_schema_version!r}; "
            f"got {schema_version!r}"
        )
    findings = scan_text(report_text, path=resolved_report_path.name)
    if findings:
        raise LLMWorldModelDemoError(f"{name} diagnostic report contains secret-scan findings")

    if manifest is not None and resolved_manifest_path is not None:
        manifest_file = _find_manifest_file_for_path(
            manifest,
            resolved_report_path,
            root=resolved_manifest_path.parent,
        )
        if manifest_file is None:
            raise LLMWorldModelDemoError(
                f"{name} diagnostic report is not listed in manifest: {resolved_report_path}"
            )

    return {
        "status": "available",
        "path": str(resolved_report_path),
        "schema_version": str(schema_version),
        "artifact_id": None if manifest is None else manifest.artifact_id,
        "artifact_kind": None if manifest is None else manifest.artifact_kind,
        "artifact_manifest_path": None if resolved_manifest_path is None else str(resolved_manifest_path),
        "manifest_file_path": None if manifest_file is None else manifest_file.path,
        "sha256": manifest_file.sha256 if manifest_file is not None else sha256_file(resolved_report_path),
        "bytes": manifest_file.bytes if manifest_file is not None else resolved_report_path.stat().st_size,
    }


def _resolve_diagnostic_report_path(
    path: Path | str,
    *,
    manifest_path: Path | None,
) -> Path:
    report_path = Path(path)
    if report_path.is_absolute() or manifest_path is None:
        return report_path
    manifest_relative = manifest_path.parent / report_path
    if manifest_relative.exists():
        return manifest_relative
    return report_path


def _find_manifest_file_for_path(manifest: Any, path: Path, *, root: Path) -> Any:
    resolved = path.resolve()
    for file in manifest.files:
        candidate = (root / file.path).resolve()
        if candidate == resolved:
            return file
    return None


def _diagnostic_parent_artifact_ids(diagnostics: Mapping[str, Any]) -> tuple[str, ...]:
    parent_ids: list[str] = []
    for diagnostic in diagnostics.values():
        if not isinstance(diagnostic, Mapping):
            continue
        artifact_id = diagnostic.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id and artifact_id not in parent_ids:
            parent_ids.append(artifact_id)
    return tuple(parent_ids)


def _diagnostic_config(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "status": _mapping(value).get("status"),
            "path": _mapping(value).get("path"),
            "artifact_manifest_path": _mapping(value).get("artifact_manifest_path"),
        }
        for name, value in diagnostics.items()
    }


def _score_by_candidate_id(report: Mapping[str, Any]) -> dict[str, float | None]:
    scores = _mapping(report.get("scores"))
    rows = scores.get("codelewm_rerank", ())
    result: dict[str, float | None] = {}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return result
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate_id = _candidate_id_from_path(str(row.get("candidate", "")))
        score = row.get("final_score")
        result[candidate_id] = float(score) if isinstance(score, (int, float)) else None
    return result


def _candidate_row(
    candidate: Mapping[str, Any],
    *,
    max_score: float,
) -> str:
    candidate_id = str(candidate.get("candidate_id", "unknown"))
    score = candidate.get("score")
    score_value = float(score) if isinstance(score, (int, float)) else None
    rank = candidate.get("rank") or "n/a"
    width = (
        0
        if score_value is None or max_score <= 0
        else max(4, min(100, int((score_value / max_score) * 100)))
    )
    status = str(candidate.get("status", "unknown"))
    return (
        "<tr>"
        f"<td class=\"rank\">{_h(rank)}</td>"
        f"<td><code>{_h(candidate_id)}</code></td>"
        f"<td>{_h(_format_score(score_value))}</td>"
        f"<td>{_h(candidate.get('no_action_delta_display', 'n/a'))}</td>"
        f"<td>{_h(status)}</td>"
        f"<td><div class=\"bar\"><span style=\"width:{width}%\"></span></div></td>"
        "</tr>"
    )


def _patch_card(candidate: Mapping[str, Any]) -> str:
    candidate_id = str(candidate.get("candidate_id", "unknown"))
    patch_summary = _mapping(candidate.get("patch_summary"))
    preview_lines = _string_list(patch_summary.get("preview_lines", ()))
    patch_text = "\n".join(preview_lines) if preview_lines else "no diff preview available"
    status = str(candidate.get("status", "unknown"))
    meta = (
        f"{patch_summary.get('changed_file_count', 0)} files, "
        f"{patch_summary.get('hunk_count', 0)} hunks, "
        f"+{patch_summary.get('additions', 0)}/-{patch_summary.get('deletions', 0)}"
    )
    return (
        "<article class=\"patch-card\">"
        f"<div class=\"patch-head\"><code>{_h(candidate_id)}</code><span>{_h(status)}</span></div>"
        f"<div class=\"patch-meta\">{_h(meta)}</div>"
        f"<pre>{_h(patch_text)}</pre>"
        "</article>"
    )


def _diagnostics_markup(diagnostics: Mapping[str, Any]) -> str:
    cards = []
    for label, key in (
        ("Checkpoint inspection", "checkpoint_inspection"),
        ("Latent matrix", "latent_matrix"),
        ("Run timeline", "run_timeline"),
    ):
        slot = _mapping(diagnostics.get(key))
        cards.append(_mini(label, slot.get("status", "not_configured"), str(slot.get("path") or "not configured")))
    return "\n".join(cards)


def _orders_markup(orders: Sequence[tuple[str, Sequence[str]]]) -> str:
    cards = []
    for label, order in orders:
        items = "".join(f"<li><code>{_h(item)}</code></li>" for item in order)
        cards.append(
            "<article class=\"order-card\">"
            f"<h3>{_h(label)}</h3>"
            f"<ol>{items}</ol>"
            "</article>"
        )
    return "\n".join(cards)


def _mini(title: str, value: object, body: str) -> str:
    return (
        "<article class=\"mini\">"
        f"<span>{_h(title)}</span>"
        f"<strong>{_h(value)}</strong>"
        f"<p>{_h(body)}</p>"
        "</article>"
    )


def _stat(key: str, value: object) -> str:
    rendered = "null" if value is None else str(value)
    return f"<div class=\"stat\"><span>{_h(key)}</span><b>{_h(rendered)}</b></div>"


def _format_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.6f}"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _h(value: object) -> str:
    return html.escape(str(value), quote=True)


def _demo_html_css() -> str:
    return """
:root{
  --bg:#090b09;
  --bg-2:#0f1411;
  --line:#203023;
  --line-2:#2f4534;
  --ink:#dce7dd;
  --ink-dim:#8d9b8e;
  --paper:#f1eadc;
  --acid:#9aff5e;
  --acid-glow:rgba(154,255,94,.18);
  --amber:#ffb454;
  --rose:#ff6b8b;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  --serif: Georgia, Times New Roman, serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:400 13px/1.55 var(--mono)}
.wrap{max-width:1040px;margin:0 auto;padding:0 24px}
.hero{padding:72px 0 48px;border-bottom:1px solid var(--line)}
.hero-grid{display:grid;grid-template-columns:1.3fr .9fr;gap:36px;align-items:start}
.ident{font:700 11px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--acid)}
h1{margin:16px 0 14px;font:700 52px/1.02 var(--mono);letter-spacing:-.02em;color:var(--paper)}
.deck,.s-deck{font:400 17px/1.55 var(--serif);color:var(--ink-dim);max-width:760px}
.pill-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px}
.pill{border:1px solid var(--line-2);padding:8px 10px;color:var(--ink);background:var(--bg-2)}
.pill.ok{border-color:var(--acid);color:var(--acid);background:var(--acid-glow)}
.pill.warn{border-color:var(--amber);color:var(--amber)}
.stat-panel,.mini,.order-card,.patch-card{background:var(--bg-2);border:1px solid var(--line-2)}
.stat-panel{padding:14px}
.stat{display:grid;grid-template-columns:120px 1fr;gap:12px;padding:9px 0;border-bottom:1px solid var(--line)}
.stat:last-child{border-bottom:0}
.stat span{color:var(--ink-dim)}
.stat b{color:var(--paper);font-weight:500;word-break:break-word}
.s{position:relative;padding:62px 0;border-bottom:1px solid var(--line)}
.section-head{display:flex;gap:12px;align-items:center;margin-bottom:20px}
.section-num{font:700 11px/1 var(--mono);color:var(--bg);background:var(--acid);padding:7px 8px}
.section-kind{font:700 10px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--ink-dim)}
h2{margin:0 0 10px;font:500 31px/1.15 var(--mono);color:var(--paper)}
h2 em{font-style:normal;color:var(--acid)}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.mini{padding:16px}
.mini span{display:block;color:var(--ink-dim);text-transform:uppercase;letter-spacing:.14em;font-size:10px}
.mini strong{display:block;margin:8px 0;color:var(--paper);font-size:18px}
.mini p{margin:0;color:var(--ink-dim);font:400 14px/1.55 var(--serif)}
.rank-table{width:100%;border-collapse:collapse;background:var(--bg-2);border:1px solid var(--line-2)}
.rank-table th,.rank-table td{padding:12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
.rank-table th{color:var(--ink-dim);font-size:10px;text-transform:uppercase;letter-spacing:.14em}
.rank{color:var(--acid);font-weight:700}
code{color:var(--paper)}
.bar{height:11px;background:#111a13;border:1px solid var(--line)}
.bar span{display:block;height:100%;background:linear-gradient(90deg,var(--acid),var(--amber))}
.order-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.order-card{padding:14px}
.order-card h3{margin:0 0 10px;color:var(--paper);font-size:14px}
.order-card ol{margin:0;padding-left:24px;color:var(--ink-dim)}
.order-card li{margin:7px 0}
.patch-grid{display:grid;gap:14px}
.patch-head{display:flex;justify-content:space-between;gap:16px;padding:12px 14px;border-bottom:1px solid var(--line)}
.patch-head span{color:var(--ink-dim);font-size:11px}
.patch-meta{padding:9px 14px;border-bottom:1px solid var(--line);color:var(--ink-dim);font-size:11px}
pre{margin:0;white-space:pre-wrap;overflow:auto;color:var(--ink);background:#070907}
.patch-card pre{padding:14px;max-height:260px}
.code{padding:16px;border:1px solid var(--line-2);background:#070907;color:var(--paper)}
.footnote{color:var(--ink-dim);font:400 14px/1.55 var(--serif)}
@media (max-width:820px){
  .hero-grid,.grid-3,.order-grid{grid-template-columns:1fr}
  h1{font-size:38px}
}
"""


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
