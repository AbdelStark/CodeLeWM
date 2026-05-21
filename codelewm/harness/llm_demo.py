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
    html_path: str
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
    html_path = output_dir / "demo.html"
    manifest_path = output_dir / "manifest.json"
    candidate_pack_dir = output_dir / "candidate_pack"
    if not overwrite and (
        report_path.exists() or html_path.exists() or manifest_path.exists() or candidate_pack_dir.exists()
    ):
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
    html_path.write_text(
        render_llm_world_model_demo_html(report, candidate_pack_payload),
        encoding="utf-8",
    )
    report_scan = scan_text(report_path.read_text(encoding="utf-8"), path="llm_world_model_demo_report.json")
    if report_scan:
        raise LLMWorldModelDemoError("demo report contains secret-scan findings after redaction")
    html_scan = scan_text(html_path.read_text(encoding="utf-8"), path="demo.html")
    if html_scan:
        raise LLMWorldModelDemoError("demo HTML contains secret-scan findings after redaction")

    artifact_manifest = build_artifact_manifest(
        artifact_kind="demo_report",
        root=output_dir,
        files=(report_path, html_path),
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
        html_path=_relative_to_root(html_path, output_dir),
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


def render_llm_world_model_demo_html(
    report: Mapping[str, Any],
    candidate_pack_payload: Mapping[str, Any],
) -> str:
    """Render a self-contained visual report for the LLM plus world-model demo."""

    candidates = [
        candidate for candidate in candidate_pack_payload.get("candidates", []) if isinstance(candidate, Mapping)
    ]
    score_by_id = _score_by_candidate_id(report)
    codelewm_order = _string_list(report.get("orders", {}).get("codelewm", ()))
    llm_order = _string_list(report.get("orders", {}).get("llm", ()))
    lexical_order = _string_list(report.get("orders", {}).get("lexical", ()))
    random_order = _string_list(report.get("orders", {}).get("random", ()))
    generation_config = _mapping(candidate_pack_payload.get("generation_config"))
    provider_routing = _mapping(candidate_pack_payload.get("provider_routing"))
    generator = _mapping(candidate_pack_payload.get("generator"))
    byok = _mapping(provider_routing.get("byok"))
    metadata = _mapping(provider_routing.get("response_metadata"))
    dry_run = bool(generation_config.get("dry_run"))
    mode = "fixture dry-run" if dry_run else "live OpenRouter"
    status_sentence = (
        "In this run the LLM side used deterministic fixture candidates because dry-run mode was enabled."
        if dry_run
        else "In this run the LLM side called OpenRouter and captured provider output as untrusted patches."
    )
    status_class = "warn" if dry_run else "ok"
    claim_gate = _mapping(report.get("claim_gate"))
    max_score = max((score for score in score_by_id.values() if score is not None), default=1.0)

    rows = "\n".join(
        _candidate_row(candidate, score_by_id=score_by_id, max_score=max_score, codelewm_order=codelewm_order)
        for candidate in candidates
    )
    patch_cards = "\n".join(_patch_card(candidate) for candidate in candidates)
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
        <span class="pill">success: {_h(str(report.get("success", False)).lower())}</span>
        <span class="pill">claim: {_h(str(claim_gate.get("allowed", False)).lower())}</span>
      </div>
    </div>
    <aside class="stat-panel">
      {_stat("model", generator.get("model"))}
      {_stat("provider", generator.get("provider"))}
      {_stat("sdk version", generator.get("sdk_version") or "not loaded")}
      {_stat("provider mode", metadata.get("mode") or ("live" if not dry_run else "fixture"))}
      {_stat("BYOK", "enabled" if byok.get("enabled") else "disabled")}
      {_stat("candidates", _mapping(report.get("candidate_summary")).get("candidate_count"))}
      {_stat("valid", _mapping(report.get("candidate_summary")).get("valid_candidate_count"))}
      {_stat("scorer", _mapping(report.get("scores")).get("model_id"))}
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
      <p class="s-deck">The bars show transition energy from the current scorer backend. The rank number is CodeLeWM's order.</p>
      <table class="rank-table">
        <thead><tr><th>rank</th><th>candidate</th><th>score</th><th>status</th><th>bar</th></tr></thead>
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
      <h2>The actual candidate <em>patches</em>.</h2>
      <p class="s-deck">Candidate code is treated as untrusted text. The demo parses and dry-run-applies patches; it does not execute them.</p>
      <div class="patch-grid">{patch_cards}</div>
    </div>
  </section>

  <section class="s" id="next" data-num="05">
    <div class="wrap">
      <div class="section-head"><span class="section-num">05</span><span class="section-kind">next run</span></div>
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
    score_by_id: Mapping[str, float | None],
    max_score: float,
    codelewm_order: Sequence[str],
) -> str:
    candidate_id = str(candidate.get("candidate_id", "unknown"))
    score = score_by_id.get(candidate_id)
    rank = codelewm_order.index(candidate_id) + 1 if candidate_id in codelewm_order else "n/a"
    width = 0 if score is None or max_score <= 0 else max(4, min(100, int((score / max_score) * 100)))
    status = f"{candidate.get('parser_status', 'unknown')} / {candidate.get('dry_run_patch_status', 'unknown')}"
    return (
        "<tr>"
        f"<td class=\"rank\">{_h(rank)}</td>"
        f"<td><code>{_h(candidate_id)}</code></td>"
        f"<td>{_h(_format_score(score))}</td>"
        f"<td>{_h(status)}</td>"
        f"<td><div class=\"bar\"><span style=\"width:{width}%\"></span></div></td>"
        "</tr>"
    )


def _patch_card(candidate: Mapping[str, Any]) -> str:
    candidate_id = str(candidate.get("candidate_id", "unknown"))
    patch_text = str(candidate.get("patch_text", ""))
    status = f"{candidate.get('parser_status', 'unknown')} / {candidate.get('dry_run_patch_status', 'unknown')}"
    return (
        "<article class=\"patch-card\">"
        f"<div class=\"patch-head\"><code>{_h(candidate_id)}</code><span>{_h(status)}</span></div>"
        f"<pre>{_h(patch_text)}</pre>"
        "</article>"
    )


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
