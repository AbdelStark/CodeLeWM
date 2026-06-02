"""Execution-substrate LLM rerank tour for the v0.6 showcase."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codelewm.data.sandbox import SandboxExitCode, SandboxPolicy, run_one
from codelewm.eval.execution_rerank import (
    CompletionLabel,
    ScoredCompletion,
    rerank_completions,
)
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.observability.logging import redact_text, redact_value
from codelewm.security.secret_scan import scan_text

from .demo_scenarios import EXECUTION_RERANK_SCENARIO_ID
from .execution_rerank_view_model import (
    EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION,
    build_execution_rerank_view_model,
    ordered_diagnostic_slot_names,
    validate_execution_rerank_view_model_payload,
)
from .openrouter_adapter import (
    OpenRouterAdapterError,
    OpenRouterCandidateRequest,
    generate_candidate_pack,
    write_candidate_pack_artifact,
)
from .scorer import ErrorReport, ScoreError, _apply_unified_diff, load_scorer


EXECUTION_RERANK_TOUR_REPORT_SCHEMA_VERSION = "codelewm.harness.execution_rerank_tour.v1"
EXECUTION_RERANK_TOUR_RUN_SCHEMA_VERSION = "codelewm.harness.execution_rerank_tour_run.v1"
EXECUTION_RERANK_TOUR_DEFAULT_BENCHMARK = "execution-rerank-mbpp-tour"


class ExecutionRerankDemoError(ValueError):
    """Raised when the execution-rerank tour cannot be run."""


@dataclass(frozen=True)
class ExecutionHiddenTest:
    """One operator-reviewed hidden input/output pair."""

    input_id: str
    input_repr: str
    expected_output_repr: str

    def as_dict(self) -> dict[str, str]:
        return {
            "input_id": self.input_id,
            "input_repr": self.input_repr,
            "expected_output_repr": self.expected_output_repr,
        }


@dataclass(frozen=True)
class ExecutionRerankProblem:
    """One public-safe MBPP-style tour problem."""

    problem_id: str
    title: str
    function_name: str
    before_code: str
    instruction: str
    hidden_tests: tuple[ExecutionHiddenTest, ...]
    expected_terms: tuple[str, ...]
    source: str = "synthetic_mbpp_style"
    license: str = "mit"

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "title": self.title,
            "function_name": self.function_name,
            "instruction": self.instruction,
            "hidden_tests": [test.as_dict() for test in self.hidden_tests],
            "expected_terms": list(self.expected_terms),
            "source": self.source,
            "license": self.license,
        }


@dataclass(frozen=True)
class ExecutionRerankTourResult:
    """CLI-facing summary for a manifest-backed execution-rerank tour."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    view_model_path: str
    html_path: str
    asciicast_path: str
    html_export_path: str | None
    problem_count: int
    completion_count: int
    claim_allowed: bool
    schema_version: str = EXECUTION_RERANK_TOUR_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "view_model_path": self.view_model_path,
            "html_path": self.html_path,
            "asciicast_path": self.asciicast_path,
            "html_export_path": self.html_export_path,
            "problem_count": self.problem_count,
            "completion_count": self.completion_count,
            "claim_allowed": self.claim_allowed,
        }


def run_execution_rerank_tour(
    *,
    checkpoint: Path | str,
    out: Path | str,
    tour_count: int = 5,
    scenario_id: str = EXECUTION_RERANK_SCENARIO_ID,
    device: str = "cpu",
    env: Mapping[str, str] | None = None,
    html_export: Path | str | None = None,
    allow_unsafe_checkpoint: bool = False,
    require_learned_scorer: bool = True,
    overwrite: bool = False,
    command: Sequence[str] = ("scripts/llm-world-model-demo",),
) -> ExecutionRerankTourResult:
    """Run a multi-problem execution-rerank tour and write public-safe artifacts."""

    if scenario_id != EXECUTION_RERANK_SCENARIO_ID:
        raise ExecutionRerankDemoError(
            f"execution rerank tour requires scenario {EXECUTION_RERANK_SCENARIO_ID!r}"
        )
    if tour_count < 1:
        raise ExecutionRerankDemoError("tour_count must be >= 1")
    selected = execution_rerank_demo_problems()[:tour_count]
    if len(selected) < tour_count:
        raise ExecutionRerankDemoError(
            f"requested tour_count={tour_count}, but only {len(selected)} problems are registered"
        )

    output_dir = Path(out).resolve()
    report_path = output_dir / "reports" / "execution_rerank_tour_report.json"
    view_model_path = output_dir / "reports" / "execution_rerank_view_model.json"
    html_path = output_dir / "demo.html"
    asciicast_path = output_dir / "docs" / "demo" / "execution_rerank_tour.cast"
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ExecutionRerankDemoError(
            f"output already exists; pass overwrite=True to replace: {output_dir}"
        )
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    asciicast_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ if env is None else env
    checkpoint_path = Path(checkpoint)
    scorer = load_scorer(
        checkpoint_path,
        device=device,
        allow_unsafe=allow_unsafe_checkpoint,
        require_learned_backend=require_learned_scorer,
    )
    sandbox_policy = SandboxPolicy(timeout_ms=3_000, memory_mb=1024, cpu_seconds=2)

    problems_payload: list[dict[str, Any]] = []
    completion_records: list[dict[str, Any]] = []
    scored: list[ScoredCompletion] = []
    candidate_manifest_ids: list[str] = []
    for ordinal, problem in enumerate(selected, start=1):
        problem_dir = output_dir / "problems" / f"{ordinal:02d}-{problem.problem_id}"
        problem_dir.mkdir(parents=True, exist_ok=True)
        before_path = problem_dir / "app.py"
        before_path.write_text(problem.before_code, encoding="utf-8")
        request = OpenRouterCandidateRequest.from_env(
            task_id=f"{EXECUTION_RERANK_SCENARIO_ID}-{problem.problem_id}",
            instruction=problem.instruction,
            context_bundle={"app.py": problem.before_code},
            env=env,
        )
        try:
            generated = generate_candidate_pack(request, env=env)
            candidate_pack_result = write_candidate_pack_artifact(
                generated,
                problem_dir / "candidate_pack",
                command=(*command, "candidate-pack", problem.problem_id),
                overwrite=True,
            )
        except OpenRouterAdapterError as exc:
            raise ExecutionRerankDemoError(
                f"candidate generation failed for {problem.problem_id}: {exc}"
            ) from exc
        candidate_manifest_path = problem_dir / "candidate_pack" / candidate_pack_result.artifact_manifest_path
        candidate_manifest = read_artifact_manifest(candidate_manifest_path)
        validate_artifact_checksums(candidate_manifest, root=candidate_manifest_path.parent)
        candidate_manifest_ids.append(candidate_manifest.artifact_id)
        candidate_pack = json.loads(
            (problem_dir / "candidate_pack" / candidate_pack_result.candidate_pack_path).read_text(
                encoding="utf-8"
            )
        )

        no_action_score = _average_score(
            scorer,
            before=problem.before_code,
            input_reprs=tuple(test.input_repr for test in problem.hidden_tests),
            candidate=problem.before_code,
            candidate_name=f"{problem.problem_id}::no_action",
        )
        problem_candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidate_pack.get("candidates", ()), start=1):
            candidate_id = str(candidate.get("candidate_id") or f"candidate_{index:03d}")
            completion_id = f"{problem.problem_id}::{candidate_id}"
            candidate_code, apply_error = _candidate_after_code(
                problem.before_code,
                problem_dir / "candidate_pack",
                candidate,
            )
            test_results = _sandbox_test_results(
                candidate_code,
                problem=problem,
                policy=sandbox_policy,
            ) if apply_error is None else []
            passed = bool(test_results) and all(result["passed"] for result in test_results)
            if apply_error is None:
                try:
                    codelewm_raw_score = _average_score(
                        scorer,
                        before=problem.before_code,
                        input_reprs=tuple(test.input_repr for test in problem.hidden_tests),
                        candidate=candidate_code,
                        candidate_name=completion_id,
                    )
                    scoring_error = None
                except ScoreError as exc:
                    codelewm_raw_score = math.inf
                    scoring_error = exc.to_error_report(record_id=completion_id).to_dict()
            else:
                codelewm_raw_score = math.inf
                scoring_error = apply_error.to_dict()
            codelewm_score = -codelewm_raw_score if math.isfinite(codelewm_raw_score) else -1.0e12
            scores = {
                "codelewm": codelewm_score,
                "llm_order": -float(index),
                "random": _deterministic_unit_score(completion_id),
                "lexical": _lexical_completion_score(candidate_code, problem.expected_terms),
                "no_action": -no_action_score if math.isfinite(no_action_score) else -1.0e12,
                "shuffled_action": codelewm_score - _deterministic_unit_score(problem.problem_id),
            }
            record = {
                "problem_id": problem.problem_id,
                "completion_id": completion_id,
                "code": candidate_code,
                "llm_order_rank": index,
                "passed": passed,
                "scores": scores,
                "test_results": test_results,
                "predicted_output_latent": {
                    "score_direction": "higher_is_better_after_negating_energy",
                    "mean_execution_latent_score": codelewm_score,
                    "mean_raw_transition_score": (
                        None if not math.isfinite(codelewm_raw_score) else codelewm_raw_score
                    ),
                    "no_action_raw_transition_score": no_action_score,
                    "hidden_case_count": len(problem.hidden_tests),
                    "scoring_error": scoring_error,
                },
            }
            completion_records.append(record)
            problem_candidates.append(record)
            scored.append(
                ScoredCompletion(
                    label=CompletionLabel(
                        problem_id=problem.problem_id,
                        completion_id=completion_id,
                        code=candidate_code,
                        llm_order_rank=index,
                        passed=passed,
                    ),
                    scores=scores,
                )
            )

        problem_payload = {
            **problem.as_dict(),
            "before_sha256": _sha256_text(problem.before_code),
            "candidate_pack_manifest_id": candidate_manifest.artifact_id,
            "candidate_pack_manifest_path": _relative(problem_dir / "candidate_pack" / "manifest.json", output_dir),
            "candidate_count": len(problem_candidates),
            "passed_candidate_count": sum(1 for candidate in problem_candidates if candidate["passed"]),
            "candidates": problem_candidates,
            "codelewm_order": [
                item["completion_id"]
                for item in sorted(problem_candidates, key=lambda row: (-row["scores"]["codelewm"], row["completion_id"]))
            ],
            "llm_order": [item["completion_id"] for item in problem_candidates],
        }
        problems_payload.append(problem_payload)

    rerank_report = rerank_completions(
        completions=tuple(scored),
        benchmark=EXECUTION_RERANK_TOUR_DEFAULT_BENCHMARK,
        bootstrap_seed=17,
        bootstrap_samples=300,
    ).as_dict()
    rerank_report["claim_allowed"] = False
    rerank_report["claim_reason"] = (
        "demo_tour_is_not_scaled_downstream_benchmark_evidence; "
        "coding-usefulness claims require the 100-example gate"
    )
    diagnostics = {
        "retrieval_evidence": {
            "status": "not_recorded",
            "reason": (
                "the tour scores generated candidates and does not run "
                "execution-pack retrieval; the retrieval no-action margin "
                "evidence is published separately"
            ),
            "reference": "docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md",
        },
        "checkpoint": {
            "status": "available",
            "model_id": scorer.model_id,
            "sha256_short": (
                scorer.checkpoint_sha256[:12] if scorer.checkpoint_sha256 else None
            ),
            "device": device,
        },
        "sandbox": {
            "status": "available",
            "policy": sandbox_policy.as_dict(),
            "scope": "operator-reviewed built-in execution-rerank tour inputs only",
        },
    }
    artifact_lineage = {
        "parent_artifact_ids": tuple(candidate_manifest_ids),
        "command": tuple(command),
        "manifest_path": "manifest.json",
        "report_path": "reports/execution_rerank_tour_report.json",
        "view_model_path": "reports/execution_rerank_view_model.json",
        "html_path": "demo.html",
        "asciicast_path": "docs/demo/execution_rerank_tour.cast",
    }
    view_model = validate_execution_rerank_view_model_payload(
        build_execution_rerank_view_model(
            rerank_report=rerank_report,
            scenario_id=scenario_id,
            completion_records=tuple(completion_records),
            diagnostics=diagnostics,
            artifact_lineage=artifact_lineage,
        ).as_dict()
    )

    report = {
        "schema_version": EXECUTION_RERANK_TOUR_REPORT_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "benchmark": EXECUTION_RERANK_TOUR_DEFAULT_BENCHMARK,
        "created_at": _utc_now(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": scorer.checkpoint_sha256,
            "model_id": scorer.model_id,
            "device": device,
            "warnings": list(scorer._warnings()),
        },
        "generator": _generator_summary(completion_records, output_dir),
        "sandbox": {
            "policy": sandbox_policy.as_dict(),
            "execution_scope": "operator-reviewed built-in execution-rerank tour inputs only",
        },
        "rerank_report": rerank_report,
        "execution_rerank_view_model_path": "reports/execution_rerank_view_model.json",
        "problems": problems_payload,
        "candidate_pack_manifest_ids": candidate_manifest_ids,
        "claim_gate": {
            "allowed": False,
            "reason": rerank_report["claim_reason"],
            "blocked_claims": [
                "CodeLeWM improves generated code",
                "downstream coding usefulness",
                "semantic latent axes from the demo",
            ],
        },
    }
    # Redact home paths/secrets from the published report (e.g. the absolute
    # checkpoint path and resolved output root) before it is written, rendered,
    # or hashed into the manifest. The view model is redacted separately via
    # validate_execution_rerank_view_model_payload.
    report = redact_value(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    view_model_path.write_text(
        json.dumps(view_model, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    terminal_text = render_execution_rerank_tour_terminal(report)
    asciicast_path.write_text(_render_asciicast(terminal_text), encoding="utf-8")
    html_text = render_execution_rerank_tour_html(report, view_model)
    html_path.write_text(html_text, encoding="utf-8")
    html_export_path: Path | None = None
    if html_export is not None:
        html_export_path = Path(html_export).resolve()
        html_export_path.parent.mkdir(parents=True, exist_ok=True)
        html_export_path.write_text(html_text, encoding="utf-8")

    _assert_secret_scan_clean((report_path, view_model_path, html_path, asciicast_path))
    if html_export_path is not None:
        _assert_secret_scan_clean((html_export_path,))

    manifest = build_artifact_manifest(
        artifact_kind="demo_report",
        root=output_dir,
        files=(report_path, view_model_path, html_path, asciicast_path),
        command=command,
        config={
            "scenario_id": scenario_id,
            "tour_count": tour_count,
            "checkpoint": redact_text(str(checkpoint_path)),
            "device": device,
            "require_learned_scorer": require_learned_scorer,
        },
        parent_artifacts=tuple(candidate_manifest_ids),
        metadata={
            "schema_version": EXECUTION_RERANK_TOUR_REPORT_SCHEMA_VERSION,
            "view_model_schema_version": EXECUTION_RERANK_VIEW_MODEL_SCHEMA_VERSION,
            "problem_count": len(selected),
            "completion_count": len(completion_records),
            "claim_allowed": False,
            "html_export_path": None if html_export_path is None else str(html_export_path),
        },
    )
    write_artifact_manifest(manifest, manifest_path)
    validate_artifact_checksums(manifest, root=output_dir)
    return ExecutionRerankTourResult(
        artifact_manifest_id=manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path="reports/execution_rerank_tour_report.json",
        view_model_path="reports/execution_rerank_view_model.json",
        html_path="demo.html",
        asciicast_path="docs/demo/execution_rerank_tour.cast",
        html_export_path=None if html_export_path is None else str(html_export_path),
        problem_count=len(selected),
        completion_count=len(completion_records),
        claim_allowed=False,
    )


def execution_rerank_demo_problems() -> tuple[ExecutionRerankProblem, ...]:
    """Return the built-in public-safe MBPP-style tour set."""

    return (
        ExecutionRerankProblem(
            problem_id="mbpp-demo-square",
            title="Square an integer",
            function_name="compute_square",
            before_code="def compute_square(n):\n    pass\n",
            instruction="Implement compute_square so it returns n * n for any integer n.",
            hidden_tests=(
                ExecutionHiddenTest("square-pos", "[3]", "9"),
                ExecutionHiddenTest("square-neg", "[-4]", "16"),
            ),
            expected_terms=("return", "n", "*", "**"),
        ),
        ExecutionRerankProblem(
            problem_id="mbpp-demo-total",
            title="Sum a list",
            function_name="total",
            before_code="def total(xs):\n    pass\n",
            instruction="Implement total so it returns the sum of a list and returns 0 for an empty list.",
            hidden_tests=(
                ExecutionHiddenTest("total-values", "[[1, 2, 3]]", "6"),
                ExecutionHiddenTest("total-empty", "[[]]", "0"),
            ),
            expected_terms=("sum", "return", "xs", "for"),
        ),
        ExecutionRerankProblem(
            problem_id="mbpp-demo-even",
            title="Detect even integers",
            function_name="is_even",
            before_code="def is_even(n):\n    pass\n",
            instruction="Implement is_even so it returns True for even integers and False for odd integers.",
            hidden_tests=(
                ExecutionHiddenTest("even-true", "[4]", "True"),
                ExecutionHiddenTest("even-false", "[5]", "False"),
            ),
            expected_terms=("%", "2", "return", "=="),
        ),
        ExecutionRerankProblem(
            problem_id="mbpp-demo-reverse",
            title="Reverse a string",
            function_name="reverse_text",
            before_code="def reverse_text(text):\n    pass\n",
            instruction="Implement reverse_text so it returns the input string in reverse order.",
            hidden_tests=(
                ExecutionHiddenTest("reverse-word", "[\"abc\"]", "'cba'"),
                ExecutionHiddenTest("reverse-empty", "[\"\"]", "''"),
            ),
            expected_terms=("[::-1]", "reversed", "return", "text"),
        ),
        ExecutionRerankProblem(
            problem_id="mbpp-demo-clamp",
            title="Clamp a value",
            function_name="clamp",
            before_code="def clamp(value, low, high):\n    pass\n",
            instruction="Implement clamp so values below low return low, values above high return high, and in-range values are unchanged.",
            hidden_tests=(
                ExecutionHiddenTest("clamp-high", "[7, 0, 5]", "5"),
                ExecutionHiddenTest("clamp-low", "[-1, 0, 5]", "0"),
            ),
            expected_terms=("min", "max", "low", "high", "return"),
        ),
    )


def render_execution_rerank_tour_terminal(report: Mapping[str, Any]) -> str:
    lines = [
        "CodeLeWM execution-rerank tour",
        f"scenario={report.get('scenario_id')} benchmark={report.get('benchmark')}",
        f"model={_nested(report, 'checkpoint', 'model_id')} claim_allowed={_nested(report, 'claim_gate', 'allowed')}",
        "",
    ]
    for index, problem in enumerate(report.get("problems") or (), start=1):
        if isinstance(problem, Mapping):
            lines.append(_problem_terminal_section(index, problem))
    rerank = report.get("rerank_report") if isinstance(report.get("rerank_report"), Mapping) else {}
    lines.extend(
        [
            "Aggregate",
            f"  problem_count={rerank.get('problem_count', 0)} completions_per_problem={rerank.get('completions_per_problem', 0)}",
            f"  codelewm_lift_over_llm_order={float(rerank.get('codelewm_lift_over_llm_order', 0.0)):.2f} pts",
            f"  claim_allowed={rerank.get('claim_allowed', False)} reason={rerank.get('claim_reason', '')}",
            "",
        ]
    )
    return "\n".join(lines)


def render_execution_rerank_tour_html(
    report: Mapping[str, Any],
    view_model: Mapping[str, Any],
) -> str:
    """Render the self-contained execution-rerank web report.

    Both the web report and the Textual TUI consume the same
    schema-versioned ``execution_rerank_view_model`` so the two surfaces
    stay in lockstep. Per-problem candidate code is read from the report
    (untrusted text, HTML-escaped) and never executed here.
    """

    headline = _mapping(view_model.get("headline_panel"))
    no_action = _mapping(view_model.get("no_action_panel"))
    diagnostics = _mapping(view_model.get("diagnostics"))
    lineage = _mapping(view_model.get("artifact_lineage"))
    panels = [p for p in view_model.get("completion_panels", ()) if isinstance(p, Mapping)]
    baselines = [b for b in view_model.get("baselines", ()) if isinstance(b, Mapping)]
    claim_allowed = bool(view_model.get("claim_allowed"))
    claim_reason = str(view_model.get("claim_reason") or _nested(report, "claim_gate", "reason") or "")
    notes = [str(item) for item in view_model.get("notes", ())]

    model_id = str(_nested(report, "checkpoint", "model_id") or view_model.get("scenario_id") or "")
    problem_count = int(view_model.get("problem_count") or len(report.get("problems", ())))
    completions_per_problem = int(view_model.get("completions_per_problem") or 0)

    stat_panel = "".join(
        (
            _exec_stat("CodeLeWM pass@1", _fmt_pass_at_1(headline.get("codelewm_pass_at_1"))),
            _exec_stat("LLM-order pass@1", _fmt_pass_at_1(headline.get("llm_order_pass_at_1"))),
            _exec_stat("No-action pass@1", _fmt_pass_at_1(headline.get("no_action_pass_at_1"))),
            _exec_stat("Lift vs LLM order", _fmt_pts(view_model.get("pass_at_1_lift"))),
            _exec_stat(
                "Lift vs no-action", _fmt_pts(no_action.get("codelewm_lift_over_no_action"))
            ),
            _exec_stat("Problems x completions", f"{problem_count} x {completions_per_problem}"),
        )
    )
    claim_banner = _claim_banner_markup(claim_allowed, claim_reason)
    no_action_markup = _no_action_markup(no_action)
    ranking_rows = _ranking_rows_markup(panels)
    baseline_rows = _baseline_rows_markup(baselines)
    problem_cards = "\n".join(
        _problem_html_card(problem)
        for problem in report.get("problems", ())
        if isinstance(problem, Mapping)
    )
    diagnostics_markup = _diagnostics_markup(diagnostics)
    lineage_markup = _lineage_markup(lineage)
    notes_markup = (
        "".join(f"<li>{_h(note)}</li>" for note in notes)
        if notes
        else "<li class=\"muted\">no notes recorded</li>"
    )
    css = _execution_demo_html_css()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CodeLeWM execution-rerank tour</title>
<style>{css}</style>
</head>
<body>
<header class="hero">
  <div class="wrap hero-grid">
    <div>
      <div class="ident">CodeLeWM v0.6 execution substrate</div>
      <h1>Execution-trace <em>showcase</em>.</h1>
      <p class="deck">LLM candidates are generated, labeled through the dedicated sandbox path, and reranked by the world-model checkpoint. This report and the Textual TUI read one schema-versioned view model; the public claim gate stays closed.</p>
      <div class="pill-row">
        <span class="pill">model: {_h(model_id)}</span>
        <span class="pill">problems: {_h(str(problem_count))}</span>
        <span class="pill claim-pill">claim allowed: {_h(str(claim_allowed).lower())}</span>
      </div>
    </div>
    <aside class="stat-panel">{stat_panel}</aside>
  </div>
</header>
<main>
  <section class="s" id="claim" data-num="01">
    <div class="wrap">
      <div class="section-head"><span class="section-num">01</span><span class="section-kind">claim gate</span></div>
      <h2>Diagnostic evidence, claim gate <em>closed</em>.</h2>
      <p class="s-deck">This tour is workflow evidence only. It does not support a downstream coding-usefulness claim; that requires the scaled benchmark gate.</p>
      {claim_banner}
    </div>
  </section>

  <section class="s" id="no-action" data-num="02">
    <div class="wrap">
      <div class="section-head"><span class="section-num">02</span><span class="section-kind">no-action</span></div>
      <h2>CodeLeWM versus <em>no-action</em>.</h2>
      <p class="s-deck">The honest baseline question: does reranking beat doing nothing? Higher pass@1 is better.</p>
      {no_action_markup}
    </div>
  </section>

  <section class="s" id="ranking" data-num="03">
    <div class="wrap">
      <div class="section-head"><span class="section-num">03</span><span class="section-kind">trace ranking</span></div>
      <h2>Candidate <em>ranking</em> trace.</h2>
      <p class="s-deck">Each completion's CodeLeWM rank, hidden-test outcome, and the rank it would get under the LLM order and lexical baselines. The bar shows CodeLeWM score separation.</p>
      <div class="panel"><table class="rank-table"><thead><tr><th>CodeLeWM rank</th><th>completion</th><th>tests</th><th>CodeLeWM score</th><th>LLM rank</th><th>lexical rank</th><th>separation</th></tr></thead><tbody>{ranking_rows}</tbody></table></div>
    </div>
  </section>

  <section class="s" id="problems" data-num="04">
    <div class="wrap">
      <div class="section-head"><span class="section-num">04</span><span class="section-kind">per problem</span></div>
      <h2>Per-problem <em>detail</em>.</h2>
      <p class="s-deck">Candidate code is treated as untrusted text and HTML-escaped; it is never executed by this report. Hidden-test cases are summarized as pass/fail.</p>
      <div class="grid">{problem_cards}</div>
    </div>
  </section>

  <section class="s" id="baselines" data-num="05">
    <div class="wrap">
      <div class="section-head"><span class="section-num">05</span><span class="section-kind">baselines</span></div>
      <h2>Aggregate <em>pass@1</em>.</h2>
      <p class="s-deck">Pass@1 under every baseline ordering. Lift is reported in percentage points.</p>
      <div class="panel"><table><thead><tr><th>baseline</th><th>pass@1</th><th>passed</th></tr></thead><tbody>{baseline_rows}</tbody></table></div>
    </div>
  </section>

  <section class="s" id="diagnostics" data-num="06">
    <div class="wrap">
      <div class="section-head"><span class="section-num">06</span><span class="section-kind">diagnostics</span></div>
      <h2>Diagnostics stay <em>explicit</em>.</h2>
      <p class="s-deck">Missing diagnostics are shown, not hidden. Retrieval evidence is published separately and is marked not_recorded here.</p>
      <div class="grid-3">{diagnostics_markup}</div>
    </div>
  </section>

  <section class="s" id="lineage" data-num="07">
    <div class="wrap">
      <div class="section-head"><span class="section-num">07</span><span class="section-kind">lineage</span></div>
      <h2>Artifact <em>lineage</em>.</h2>
      <p class="s-deck">Provenance for this manifest-backed, secret-scanned artifact set. Both surfaces read the same view model.</p>
      {lineage_markup}
      <ul class="notes">{notes_markup}</ul>
    </div>
  </section>
</main>
<footer class="foot"><div class="wrap"><p class="footnote">Diagnostic showcase for the v0.6 execution substrate. Not a claim that CodeLeWM improves coding agents; the claim gate is closed and scaled downstream usefulness remains unsupported.</p></div></footer>
</body>
</html>
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the CodeLeWM execution-rerank tour")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tour", type=int, default=5)
    parser.add_argument("--scenario", default=EXECUTION_RERANK_SCENARIO_ID)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument("--html", type=Path, help="optional extra self-contained HTML export path")
    parser.add_argument("--allow-unsafe-checkpoint", action="store_true")
    parser.add_argument("--fixture-scorer", action="store_true", help="allow deterministic fixture scorer for tests")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_execution_rerank_tour(
            checkpoint=args.checkpoint,
            out=args.out,
            tour_count=args.tour,
            scenario_id=args.scenario,
            device=args.device,
            html_export=args.html,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            require_learned_scorer=not args.fixture_scorer,
            overwrite=args.overwrite,
            command=(
                "scripts/llm-world-model-demo",
                "--scenario",
                args.scenario,
                "--tour",
                str(args.tour),
            ),
        )
    except (ExecutionRerankDemoError, ScoreError) as exc:
        payload = {
            "schema_version": "codelewm.error.v1",
            "error_type": "scoring_error",
            "message": str(exc),
            "remediation": "inspect the execution-rerank tour inputs and retry",
            "record_id": None,
            "artifact": str(args.out),
            "caused_by": f"{exc.__class__.__name__}: {exc}",
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(payload["message"])
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        report = json.loads((args.out / result.report_path).read_text(encoding="utf-8"))
        print(render_execution_rerank_tour_terminal(report))
        print(f"html: {args.out / result.html_path}")
        if result.html_export_path is not None:
            print(f"html_export: {result.html_export_path}")
    return 0


def _candidate_after_code(
    before_text: str,
    candidate_pack_dir: Path,
    candidate: Mapping[str, Any],
) -> tuple[str, ErrorReport | None]:
    candidate_id = str(candidate.get("candidate_id") or "candidate")
    patch_path_raw = candidate.get("patch_path")
    if not isinstance(patch_path_raw, str) or not patch_path_raw:
        return before_text, ErrorReport(
            error_type="patch_apply_failed",
            message="candidate has no materialized patch_path",
            remediation="inspect candidate-pack capture",
            record_id=candidate_id,
        )
    patch_path = candidate_pack_dir / patch_path_raw
    try:
        patch_text = patch_path.read_text(encoding="utf-8")
        return _apply_unified_diff(before_text, patch_text, artifact=candidate_id), None
    except (OSError, ScoreError) as exc:
        if isinstance(exc, ScoreError):
            return before_text, exc.to_error_report(record_id=candidate_id)
        return before_text, ErrorReport(
            error_type="missing_file",
            message=f"candidate patch could not be read: {exc}",
            remediation="inspect candidate-pack artifact files",
            record_id=candidate_id,
            artifact=str(patch_path),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )


def _sandbox_test_results(
    candidate_code: str,
    *,
    problem: ExecutionRerankProblem,
    policy: SandboxPolicy,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for test in problem.hidden_tests:
        result = run_one(
            candidate_code,
            input_repr=test.input_repr,
            function_name=problem.function_name,
            policy=policy,
        )
        exit_value = result.exit_code.value if isinstance(result.exit_code, SandboxExitCode) else str(result.exit_code)
        passed = exit_value == "ok" and result.output_repr == test.expected_output_repr
        results.append(
            {
                "input_id": test.input_id,
                "input_repr_sha256": _sha256_text(test.input_repr),
                "expected_output_repr": test.expected_output_repr,
                "observed_output_repr": result.output_repr,
                "exit_code": exit_value,
                "output_type": result.output_type,
                "passed": passed,
                "policy_violations": list(result.policy_violations),
                "wall_time_ms": result.wall_time_ms,
            }
        )
    return results


def _average_score(
    scorer: Any,
    *,
    before: str,
    input_reprs: Sequence[str],
    candidate: str,
    candidate_name: str,
) -> float:
    scores = [
        scorer.score_texts(
            before=before,
            instruction=input_repr,
            candidate=candidate,
            candidate_name=f"{candidate_name}::{index}",
        ).final_score
        for index, input_repr in enumerate(input_reprs, start=1)
    ]
    if not scores:
        return math.inf
    return float(sum(scores) / len(scores))


def _lexical_completion_score(code: str, expected_terms: Sequence[str]) -> float:
    if not expected_terms:
        return 0.0
    lowered = code.lower()
    hits = sum(1 for term in expected_terms if term.lower() in lowered)
    return hits / len(expected_terms)


def _deterministic_unit_score(value: str) -> float:
    digest = _sha256_text(value)
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _generator_summary(records: Sequence[Mapping[str, Any]], root: Path) -> dict[str, Any]:
    return {
        "completion_count": len(records),
        "output_root": str(root),
        "mode": "fixture_or_live_per_environment",
    }


def _problem_terminal_section(index: int, problem: Mapping[str, Any]) -> str:
    candidates = [
        candidate for candidate in problem.get("candidates", ()) if isinstance(candidate, Mapping)
    ]
    lines = [
        f"Problem {index}: {problem.get('problem_id')} - {problem.get('title')}",
        f"  llm_order: {', '.join(str(x) for x in problem.get('llm_order', ())) }",
        f"  codelewm_order: {', '.join(str(x) for x in problem.get('codelewm_order', ())) }",
    ]
    for candidate in candidates:
        status = "pass" if candidate.get("passed") else "fail"
        lines.append(
            "  "
            f"{candidate.get('completion_id')}: {status} "
            f"score={float(_nested(candidate, 'scores', 'codelewm') or 0.0):.4f}"
        )
    return "\n".join(lines) + "\n"


def _problem_html_card(problem: Mapping[str, Any]) -> str:
    rows = []
    for candidate in problem.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        passed = bool(candidate.get("passed"))
        hidden_results = candidate.get("test_results") if isinstance(candidate.get("test_results"), list) else []
        hidden_summary = ", ".join(
            f"{str(result.get('input_id', 'case'))}:{'pass' if result.get('passed') else 'fail'}"
            for result in hidden_results
            if isinstance(result, Mapping)
        )
        rows.append(
            "<tr>"
            f"<td>{_h(str(candidate.get('completion_id', '')))}</td>"
            f"<td class=\"{'ok' if passed else 'bad'}\">{_h('pass' if passed else 'fail')}</td>"
            f"<td>{float(_nested(candidate, 'scores', 'codelewm') or 0.0):.4f}</td>"
            f"<td>{int(candidate.get('llm_order_rank', 0))}</td>"
            f"<td>{_h(hidden_summary)}</td>"
            "</tr>"
            "<tr>"
            f"<td colspan=\"5\"><pre class=\"code\">{_h(str(candidate.get('code', '')))}</pre></td>"
            "</tr>"
        )
    table = "".join(rows)
    codelewm_order = ", ".join(str(value) for value in problem.get("codelewm_order", ()))
    return (
        "<article class=\"panel problem-card\">"
        f"<h3>{_h(str(problem.get('title', '')))}</h3>"
        f"<p class=\"muted\">{_h(str(problem.get('instruction', '')))}</p>"
        f"<p class=\"rank\">CodeLeWM order: {_h(codelewm_order)}</p>"
        "<table><thead><tr><th>completion</th><th>tests</th><th>score</th><th>LLM rank</th><th>hidden cases</th></tr></thead>"
        f"<tbody>{table}</tbody></table>"
        f"<p class=\"hash\">before sha256: {_h(str(problem.get('before_sha256', '')))}</p>"
        "</article>"
    )


def _exec_stat(label: str, value: object) -> str:
    return (
        "<div class=\"stat\">"
        f"<span>{_h(label)}</span>"
        f"<b>{_h(str(value))}</b>"
        "</div>"
    )


def _claim_banner_markup(allowed: bool, reason: str) -> str:
    state = "open" if allowed else "closed"
    cls = "ok" if allowed else "bad"
    return (
        f"<div class=\"claim-banner {cls}\">"
        f"<strong>claim gate: {_h(state)}</strong>"
        f"<span>{_h(reason or 'not recorded')}</span>"
        "</div>"
    )


def _no_action_markup(no_action: Mapping[str, Any]) -> str:
    status = str(no_action.get("status", "not_recorded"))
    if status != "available":
        return (
            "<div class=\"panel na-panel\">"
            "<p class=\"muted\">No-action pass@1 was not recorded for this run; "
            "the CodeLeWM-versus-no-action margin is therefore explicit as "
            "<code>not_recorded</code>.</p>"
            "</div>"
        )
    interpretation = str(no_action.get("interpretation", "not_recorded"))
    delta = no_action.get("codelewm_minus_no_action")
    delta_cls = (
        "good"
        if interpretation == "better_than_no_action"
        else "bad"
        if interpretation == "worse_than_no_action"
        else "tie"
    )
    width = _bar_width_signed(delta)
    return (
        "<div class=\"panel na-panel\">"
        "<div class=\"na-stats\">"
        f"{_exec_stat('CodeLeWM pass@1', _fmt_pass_at_1(no_action.get('codelewm_pass_at_1')))}"
        f"{_exec_stat('No-action pass@1', _fmt_pass_at_1(no_action.get('no_action_pass_at_1')))}"
        f"{_exec_stat('CodeLeWM - no-action', _fmt_delta(delta))}"
        f"{_exec_stat('Lift vs no-action', _fmt_pts(no_action.get('codelewm_lift_over_no_action')))}"
        "</div>"
        "<div class=\"delta-track\">"
        f"<i class=\"delta-fill {delta_cls}\" style=\"width:{width}%\"></i>"
        "</div>"
        f"<p class=\"muted\">interpretation: {_h(interpretation)}</p>"
        "</div>"
    )


def _ranking_rows_markup(panels: Sequence[Mapping[str, Any]]) -> str:
    scores = [
        float(_nested(panel, "scores", "codelewm") or 0.0)
        for panel in panels
        if isinstance(panel, Mapping)
    ]
    lo = min(scores) if scores else 0.0
    hi = max(scores) if scores else 0.0
    rows = []
    for panel in panels:
        if not isinstance(panel, Mapping):
            continue
        passed = bool(panel.get("passed"))
        score = float(_nested(panel, "scores", "codelewm") or 0.0)
        width = _normalized_bar_width(score, lo, hi)
        ranks = _mapping(panel.get("rank_by_baseline"))
        rows.append(
            "<tr>"
            f"<td class=\"rank\">{_h(str(panel.get('codelewm_rank', 'n/a')))}</td>"
            f"<td><code>{_h(str(panel.get('completion_id', '')))}</code></td>"
            f"<td class=\"{'ok' if passed else 'bad'}\">{_h('pass' if passed else 'fail')}</td>"
            f"<td>{_h(_fmt_score(score))}</td>"
            f"<td>{_h(str(ranks.get('llm_order', 'n/a')))}</td>"
            f"<td>{_h(str(ranks.get('lexical', 'n/a')))}</td>"
            "<td><div class=\"bar\"><span style=\"width:"
            f"{width}%\"></span></div></td>"
            "</tr>"
        )
    return "".join(rows) or "<tr><td colspan=\"7\" class=\"muted\">no completions recorded</td></tr>"


def _baseline_rows_markup(baselines: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for row in baselines:
        if not isinstance(row, Mapping):
            continue
        highlight = " class=\"rank\"" if row.get("baseline") == "codelewm" else ""
        rows.append(
            "<tr>"
            f"<td{highlight}>{_h(str(row.get('baseline', '')))}</td>"
            f"<td>{_h(_fmt_score(row.get('pass_at_1')))}</td>"
            f"<td>{int(row.get('pass_count', 0))}/{int(row.get('problem_count', 0))}</td>"
            "</tr>"
        )
    return "".join(rows) or "<tr><td colspan=\"3\" class=\"muted\">no baselines recorded</td></tr>"


def _diagnostics_markup(diagnostics: Mapping[str, Any]) -> str:
    cards = []
    for name in ordered_diagnostic_slot_names(diagnostics):
        slot_map = _mapping(diagnostics.get(name))
        status = str(slot_map.get("status", "not_recorded"))
        detail = (
            slot_map.get("reason")
            or slot_map.get("model_id")
            or slot_map.get("scope")
            or slot_map.get("reference")
            or ""
        )
        status_cls = "ok" if status == "available" else "warn"
        cards.append(
            "<article class=\"mini\">"
            f"<span>{_h(str(name))}</span>"
            f"<strong class=\"{status_cls}\">{_h(status)}</strong>"
            f"<p>{_h(str(detail))}</p>"
            "</article>"
        )
    return "".join(cards) or "<article class=\"mini\"><span>diagnostics</span><strong>not recorded</strong></article>"


def _lineage_markup(lineage: Mapping[str, Any]) -> str:
    parents = [str(item) for item in lineage.get("parent_artifact_ids", ())]
    command = " ".join(str(item) for item in lineage.get("command", ()))
    paths = [
        ("manifest", lineage.get("manifest_path")),
        ("report", lineage.get("report_path")),
        ("view model", lineage.get("view_model_path")),
        ("html", lineage.get("html_path")),
        ("asciicast", lineage.get("asciicast_path")),
    ]
    path_rows = "".join(
        f"<tr><td>{_h(label)}</td><td><code>{_h(str(value))}</code></td></tr>"
        for label, value in paths
        if value
    )
    parent_items = (
        "".join(f"<li><code>{_h(pid)}</code></li>" for pid in parents)
        if parents
        else "<li class=\"muted\">no parent artifacts recorded</li>"
    )
    return (
        "<div class=\"panel\">"
        f"<p class=\"muted\">command: <code>{_h(command or 'n/a')}</code></p>"
        "<p class=\"muted\">parent candidate-pack artifacts:</p>"
        f"<ul class=\"notes\">{parent_items}</ul>"
        f"<table><tbody>{path_rows}</tbody></table>"
        "</div>"
    )


def _fmt_pass_at_1(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pts(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if number >= 0 else ""
    return f"{sign}{number:.2f} pts"


def _fmt_score(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_delta(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if number >= 0 else ""
    return f"{sign}{number:.3f}"


def _normalized_bar_width(value: float, lo: float, hi: float) -> int:
    if not math.isfinite(value) or not math.isfinite(lo) or not math.isfinite(hi):
        return 4
    if hi <= lo:
        return 100
    fraction = (value - lo) / (hi - lo)
    return max(4, min(100, int(fraction * 100)))


def _bar_width_signed(value: object) -> int:
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number):
        return 0
    # pass@1 deltas live in [-1, 1]; scale magnitude to a 0-100 bar.
    return max(0, min(100, int(number * 100)))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _execution_demo_html_css() -> str:
    return """
:root{--bg:#090b09;--bg-2:#0f1411;--line:#203023;--line-2:#2f4534;--ink:#dce7dd;--ink-dim:#8d9b8e;--paper:#f1eadc;--acid:#9aff5e;--amber:#ffb454;--rose:#ff6b8b;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--serif:Georgia,"Times New Roman",serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:400 13px/1.55 var(--mono)}
.wrap{max-width:1040px;margin:0 auto;padding:0 20px}
.hero{padding:64px 0 44px;border-bottom:1px solid var(--line)}
.hero-grid{display:grid;grid-template-columns:1.3fr .9fr;gap:36px;align-items:start}
.ident{font:700 11px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--acid)}
h1{margin:16px 0 14px;font:700 48px/1.04 var(--mono);letter-spacing:0;color:var(--paper)}
h1 em,h2 em{font-style:normal;color:var(--acid)}
.deck,.s-deck{font:400 17px/1.55 var(--serif);color:var(--ink-dim);max-width:760px}
.pill-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}
.pill{border:1px solid var(--line-2);padding:8px 10px;color:var(--ink);background:var(--bg-2)}
.claim-pill{color:var(--amber)}
.stat-panel{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--line-2);background:var(--bg-2)}
.stat{padding:14px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.stat span{display:block;color:var(--ink-dim);text-transform:uppercase;letter-spacing:.12em;font-size:10px}
.stat b{display:block;margin-top:8px;color:var(--paper);font-size:18px}
.s{padding:40px 0;border-bottom:1px solid var(--line)}
.section-head{display:flex;gap:12px;align-items:baseline;margin-bottom:8px}
.section-num{font:700 12px/1 var(--mono);color:var(--acid)}
.section-kind{font:700 11px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--ink-dim)}
h2{margin:0 0 12px;font:700 28px/1.12 var(--mono);color:var(--paper)}
.panel{margin-top:18px;background:var(--bg-2);border:1px solid var(--line-2);padding:16px}
table{width:100%;border-collapse:collapse;font:400 12px/1.5 var(--mono)}
th,td{text-align:left;border-bottom:1px solid var(--line);padding:8px;vertical-align:top}
th{color:var(--ink-dim);text-transform:uppercase;letter-spacing:.08em;font-size:10px}
code{color:var(--paper)}
.rank{color:var(--acid);font-weight:700}
.ok{color:var(--acid);font-weight:700}
.bad{color:var(--rose);font-weight:700}
.warn{color:var(--amber);font-weight:700}
.muted{color:var(--ink-dim)}
.bar{height:11px;background:#111a13;border:1px solid var(--line)}
.bar span{display:block;height:100%;background:linear-gradient(90deg,var(--acid),var(--amber))}
.claim-banner{margin-top:18px;display:grid;gap:6px;padding:16px;border:1px solid var(--line-2);background:var(--bg-2)}
.claim-banner.bad{border-color:var(--rose)}
.claim-banner strong{color:var(--paper);text-transform:uppercase;letter-spacing:.08em}
.na-stats{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--bg)}
.delta-track{height:14px;background:#111a13;border:1px solid var(--line);margin-top:12px}
.delta-fill{display:block;height:100%}
.delta-fill.good{background:var(--acid)}
.delta-fill.bad{background:var(--rose)}
.delta-fill.tie{background:var(--amber)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:18px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px}
.problem-card h3{margin:0 0 8px;color:var(--paper);font-size:15px}
.problem-card pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#070907;color:var(--ink);border:1px solid var(--line);padding:10px;font-size:12px;margin:0}
.problem-card .hash{color:var(--ink-dim);font-size:11px;word-break:break-all}
.mini{border:1px solid var(--line-2);background:var(--bg-2);padding:14px}
.mini span{display:block;color:var(--ink-dim);text-transform:uppercase;letter-spacing:.12em;font-size:10px}
.mini strong{display:block;margin:8px 0;font-size:15px}
.mini p{margin:0;color:var(--ink-dim);font:400 13px/1.5 var(--serif)}
.notes{margin:8px 0 0;padding-left:18px;color:var(--ink-dim)}
.foot{padding:32px 0}
.footnote{color:var(--ink-dim);font:400 14px/1.55 var(--serif)}
@media (max-width:820px){
  .hero-grid,.stat-panel,.na-stats,.grid-3{grid-template-columns:1fr}
  h1{font-size:36px}
  .stat{border-right:0}
}
"""


def _render_asciicast(terminal_text: str) -> str:
    header = {
        "version": 2,
        "width": 120,
        "height": 32,
        "timestamp": int(datetime.now(timezone.utc).timestamp()),
        "env": {"TERM": "xterm-256color"},
    }
    event = [0.0, "o", terminal_text]
    return json.dumps(header, sort_keys=True) + "\n" + json.dumps(event) + "\n"


def _assert_secret_scan_clean(paths: Sequence[Path]) -> None:
    for path in paths:
        findings = scan_text(path.read_text(encoding="utf-8"), path=str(path))
        if findings:
            raise ExecutionRerankDemoError(f"secret scan failed for {path}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _h(value: str) -> str:
    import html

    return html.escape(value, quote=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
