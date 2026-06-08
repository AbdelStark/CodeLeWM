"""Final v1.0 paper-demo artifact assembly.

The paper demo is a deterministic replay over checked-in v0.9 downstream
rerank artifacts. It does not import, execute, or re-label candidate code.
"""

from __future__ import annotations

import html
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codelewm.observability import (
    RUN_TIMELINE_SCHEMA_VERSION,
    build_artifact_manifest,
    detect_source_git_sha,
    read_artifact_manifest,
    sha256_file,
    validate_artifact_checksums,
    write_artifact_manifest,
    write_run_timeline_report,
)
from codelewm.observability.timeline import RunTimelineRecorder
from codelewm.security.secret_scan import scan_paths


PAPER_DEMO_REPORT_SCHEMA_VERSION = "codelewm.harness.paper_demo_report.v1"
PAPER_DEMO_CLAIM_GATE_SCHEMA_VERSION = "codelewm.harness.paper_demo_claim_gate.v1"
PAPER_DEMO_RUN_SCHEMA_VERSION = "codelewm.harness.paper_demo_run.v1"
PAPER_DEMO_DEFAULT_OUT = ".artifacts/paper-demo"
PAPER_DEMO_SCORE_SOURCE = "replay_existing_scores"

_SOURCE_ROOT = Path("docs") / "benchmark" / "v0_9"
_RESULT_REPORT_PATH = Path("docs") / "benchmark" / "EXECUTION_V0_9_RESULTS_2026-06-07.md"
_ARTIFACT_INDEX_PATH = Path("docs") / "benchmark" / "PUBLIC_ARTIFACT_INDEX_2026-06-07.md"
_BASELINE_ORDER = (
    "codelewm",
    "llm_order",
    "random",
    "lexical",
    "no_action",
    "shuffled_action",
)
_OPTIONAL_INDEX_BASELINES = ("retrieval_prior_only", "retrieval_ensemble")

_CHECKPOINTS: dict[int, dict[str, str]] = {
    42: {
        "run_artifact": (
            "abdelstark/codelewm-runs/"
            "codelewm-v0-9-short-execution-20260606-69f798a-seed-42"
        ),
        "training_artifact": "training_run-992f7757f2780da4",
        "checkpoint_sha256": (
            "c783fa0dbe5da6bd072ff0b2f2753bdbac9fe684b49bf82e70ab6a2f69d513da"
        ),
    },
    1729: {
        "run_artifact": (
            "abdelstark/codelewm-runs/"
            "codelewm-v0-9-short-execution-20260606-69f798a-seed-1729"
        ),
        "training_artifact": "training_run-91e9cf7c645379b3",
        "checkpoint_sha256": (
            "34ebb282b284580dd123c781ae77c93cc36bbffc4eeeee9f0bd4cdf8042001eb"
        ),
    },
}


class PaperDemoError(ValueError):
    """Raised when the paper demo cannot be assembled."""


@dataclass(frozen=True)
class PaperDemoResult:
    """CLI-facing summary for the paper-demo artifact package."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    report_path: str
    claim_gate_path: str
    table_path: str
    timeline_path: str
    html_path: str
    secret_scan_report_path: str
    parent_manifest_paths: tuple[str, ...]
    slice_count: int
    problem_count: int
    completion_count: int
    claim_allowed: bool
    schema_version: str = PAPER_DEMO_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "report_path": self.report_path,
            "claim_gate_path": self.claim_gate_path,
            "table_path": self.table_path,
            "timeline_path": self.timeline_path,
            "html_path": self.html_path,
            "secret_scan_report_path": self.secret_scan_report_path,
            "parent_manifest_paths": list(self.parent_manifest_paths),
            "slice_count": self.slice_count,
            "problem_count": self.problem_count,
            "completion_count": self.completion_count,
            "claim_allowed": self.claim_allowed,
        }


def run_paper_demo(
    *,
    source_root: Path | str = Path("."),
    out: Path | str = PAPER_DEMO_DEFAULT_OUT,
    overwrite: bool = False,
    command: Sequence[str] = ("codelewm", "paper-demo"),
) -> PaperDemoResult:
    """Assemble the deterministic v1.0 paper-demo artifact package."""

    source_root_path = Path(source_root).resolve()
    output_dir = Path(out).resolve()
    report_path = output_dir / "reports" / "paper_demo_report.json"
    claim_gate_path = output_dir / "reports" / "paper_demo_claim_gate.json"
    table_path = output_dir / "reports" / "paper_demo_table.md"
    timeline_path = output_dir / "reports" / "run_timeline.json"
    html_path = output_dir / "demo.html"
    secret_scan_path = output_dir / "reports" / "secret_scan_report.json"
    manifest_path = output_dir / "manifest.json"

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise PaperDemoError(
            f"output already exists; pass overwrite=True to replace: {output_dir}"
        )
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    recorder = RunTimelineRecorder(command=tuple(command))

    with recorder.step(
        "verify source manifests",
        metadata={"source_root": _repo_relative(source_root_path, source_root_path)},
    ):
        source_slices = tuple(_load_source_slice(source_root_path, seed, benchmark) for seed in (42, 1729) for benchmark in ("humaneval", "mbpp_plus"))

    with recorder.step("aggregate replay score rows"):
        report, claim_gate = _build_report(source_root_path, source_slices)

    with recorder.step("write paper demo reports"):
        _write_json(report_path, report)
        _write_json(claim_gate_path, claim_gate)
        table_path.write_text(render_paper_demo_table(report), encoding="utf-8")
        html_path.write_text(render_paper_demo_html(report), encoding="utf-8")

    timeline = recorder.to_report(
        status="completed",
        metadata={
            "schema_version": RUN_TIMELINE_SCHEMA_VERSION,
            "paper_demo_schema_version": PAPER_DEMO_REPORT_SCHEMA_VERSION,
            "score_source": PAPER_DEMO_SCORE_SOURCE,
        },
    )
    write_run_timeline_report(timeline, timeline_path)

    scan = scan_paths(
        (report_path, claim_gate_path, table_path, html_path, timeline_path),
        include_suffixes=(),
        recursive=False,
    )
    secret_scan_payload = _relative_secret_scan_payload(scan.to_dict(), output_dir)
    _write_json(secret_scan_path, secret_scan_payload)
    if secret_scan_payload["findings"]:
        raise PaperDemoError("paper-demo outputs failed secret scan")

    config_payload = {
        "source_root": _repo_relative(source_root_path, source_root_path),
        "score_source": PAPER_DEMO_SCORE_SOURCE,
        "source_slices": [
            {
                "seed": item["seed"],
                "benchmark": item["benchmark"],
                "manifest_path": item["manifest_path"],
            }
            for item in report["slices"]
        ],
    }
    parent_artifacts = tuple(item["source_artifact_id"] for item in report["slices"])
    manifest = build_artifact_manifest(
        artifact_kind="demo_report",
        root=output_dir,
        files=(
            report_path,
            claim_gate_path,
            table_path,
            timeline_path,
            html_path,
            secret_scan_path,
        ),
        command=command,
        config=config_payload,
        parent_artifacts=parent_artifacts,
        source_git_sha=detect_source_git_sha(source_root_path),
        metadata={
            "schema_version": PAPER_DEMO_RUN_SCHEMA_VERSION,
            "report_schema_version": PAPER_DEMO_REPORT_SCHEMA_VERSION,
            "claim_gate_schema_version": PAPER_DEMO_CLAIM_GATE_SCHEMA_VERSION,
            "timeline_schema_version": RUN_TIMELINE_SCHEMA_VERSION,
            "score_source": PAPER_DEMO_SCORE_SOURCE,
            "claim_allowed": bool(claim_gate["allowed"]),
            "slice_count": len(report["slices"]),
            "problem_count": int(report["aggregate_summary"]["problem_count_sum"]),
            "completion_count": int(report["aggregate_summary"]["completion_count_sum"]),
            "secret_scan_ok": bool(secret_scan_payload["ok"]),
            "parent_manifest_paths": [
                item["manifest_path"] for item in report["slices"]
            ],
        },
    )
    write_artifact_manifest(manifest, manifest_path)
    validate_artifact_checksums(manifest, root=output_dir)

    return PaperDemoResult(
        artifact_manifest_id=manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        report_path="reports/paper_demo_report.json",
        claim_gate_path="reports/paper_demo_claim_gate.json",
        table_path="reports/paper_demo_table.md",
        timeline_path="reports/run_timeline.json",
        html_path="demo.html",
        secret_scan_report_path="reports/secret_scan_report.json",
        parent_manifest_paths=tuple(item["manifest_path"] for item in report["slices"]),
        slice_count=len(report["slices"]),
        problem_count=int(report["aggregate_summary"]["problem_count_sum"]),
        completion_count=int(report["aggregate_summary"]["completion_count_sum"]),
        claim_allowed=bool(claim_gate["allowed"]),
    )


def render_paper_demo_terminal(report: Mapping[str, Any]) -> str:
    """Render a compact terminal summary for ``codelewm paper-demo``."""

    aggregate = _mapping(report.get("aggregate_summary"))
    gate = _mapping(report.get("claim_gate"))
    lines = [
        "CodeLeWM v1.0 paper demo",
        f"score_source={report.get('score_source')} claim_allowed={gate.get('allowed')}",
        (
            "slices="
            f"{aggregate.get('slice_count')} problems_sum={aggregate.get('problem_count_sum')} "
            f"completions_sum={aggregate.get('completion_count_sum')}"
        ),
        "",
        "Slice metrics",
    ]
    for item in report.get("slices", ()):
        if not isinstance(item, Mapping):
            continue
        metrics = _mapping(item.get("metrics"))
        codelewm = _mapping(_baseline(metrics, "codelewm"))
        no_action = _mapping(_baseline(metrics, "no_action"))
        llm_order = _mapping(_baseline(metrics, "llm_order"))
        lines.append(
            "  "
            f"seed={item.get('seed')} benchmark={item.get('benchmark')} "
            f"codelewm_pass_at_1={_fmt_fraction(codelewm.get('pass_at_1'))} "
            f"no_action_pass_at_1={_fmt_fraction(no_action.get('pass_at_1'))} "
            f"llm_order_pass_at_1={_fmt_fraction(llm_order.get('pass_at_1'))} "
            f"claim_allowed={item.get('claim_allowed')}"
        )
    lines.extend(
        [
            "",
            "Claim gate",
            f"  allowed={gate.get('allowed')} status={gate.get('status')}",
        ]
    )
    for reason in gate.get("reasons", ()):
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def render_paper_demo_table(report: Mapping[str, Any]) -> str:
    """Render a paper-ready Markdown metric table."""

    lines = [
        "# CodeLeWM v1.0 Paper Demo Table",
        "",
        "| Seed | Benchmark | CodeLeWM pass@1 | No-action pass@1 | LLM-order pass@1 | Lift vs no-action | Lift vs LLM-order | Claim gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report.get("slices", ()):
        if not isinstance(item, Mapping):
            continue
        metrics = _mapping(item.get("metrics"))
        codelewm = _mapping(_baseline(metrics, "codelewm"))
        no_action = _mapping(_baseline(metrics, "no_action"))
        llm_order = _mapping(_baseline(metrics, "llm_order"))
        deltas = _mapping(metrics.get("deltas"))
        lines.append(
            "| "
            f"{item.get('seed')} | "
            f"{item.get('benchmark_display')} | "
            f"{_pct(codelewm.get('pass_at_1'))} | "
            f"{_pct(no_action.get('pass_at_1'))} | "
            f"{_pct(llm_order.get('pass_at_1'))} | "
            f"{_fmt_points(_nested(deltas, 'codelewm_minus_no_action', 'pass_at_1_points'))} | "
            f"{_fmt_points(_nested(deltas, 'codelewm_minus_llm_order', 'pass_at_1_points'))} | "
            f"{'open' if item.get('claim_allowed') else 'closed'} |"
        )
    gate = _mapping(report.get("claim_gate"))
    lines.extend(
        [
            "",
            f"Aggregate claim gate: {'open' if gate.get('allowed') else 'closed'}.",
            "",
            "Approved wording:",
            "",
            str(gate.get("approved_public_wording") or ""),
            "",
        ]
    )
    return "\n".join(lines)


def render_paper_demo_html(report: Mapping[str, Any]) -> str:
    """Render the self-contained HTML paper-demo report."""

    aggregate = _mapping(report.get("aggregate_summary"))
    gate = _mapping(report.get("claim_gate"))
    slice_rows = []
    for item in report.get("slices", ()):
        if not isinstance(item, Mapping):
            continue
        metrics = _mapping(item.get("metrics"))
        codelewm = _mapping(_baseline(metrics, "codelewm"))
        no_action = _mapping(_baseline(metrics, "no_action"))
        llm_order = _mapping(_baseline(metrics, "llm_order"))
        deltas = _mapping(metrics.get("deltas"))
        slice_rows.append(
            "<tr>"
            f"<td>{_h(item.get('seed'))}</td>"
            f"<td>{_h(item.get('benchmark_display'))}</td>"
            f"<td>{_h(_pct(codelewm.get('pass_at_1')))}</td>"
            f"<td>{_h(_pct(no_action.get('pass_at_1')))}</td>"
            f"<td>{_h(_pct(llm_order.get('pass_at_1')))}</td>"
            f"<td>{_h(_fmt_points(_nested(deltas, 'codelewm_minus_no_action', 'pass_at_1_points')))}</td>"
            f"<td>{_h('open' if item.get('claim_allowed') else 'closed')}</td>"
            "</tr>"
        )
    ranking_sections = "\n".join(
        _slice_ranking_html(item)
        for item in report.get("slices", ())
        if isinstance(item, Mapping)
    )
    lineage_items = "\n".join(
        f"<li>{_h(parent)}</li>" for parent in _mapping(report.get("artifact_lineage")).get("source_rerank_report_artifact_ids", ())
    )
    reasons = "".join(f"<li>{_h(reason)}</li>" for reason in gate.get("reasons", ()))
    css = """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f7f8fa; color: #111827; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }
h1, h2, h3 { margin: 0 0 12px; }
p { line-height: 1.5; }
.banner { border: 1px solid #c9d2df; border-left: 6px solid #b42318; background: #fff; padding: 16px; border-radius: 6px; margin: 16px 0 22px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin: 18px 0; }
.stat { background: #fff; border: 1px solid #d7dde7; border-radius: 6px; padding: 12px; }
.label { color: #5b6472; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.value { font-size: 20px; font-weight: 700; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d7dde7; border-radius: 6px; overflow: hidden; }
th, td { padding: 10px 12px; border-bottom: 1px solid #e5e9f0; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-size: 12px; text-transform: uppercase; color: #374151; }
.slice { margin-top: 24px; }
.problem { background: #fff; border: 1px solid #d7dde7; border-radius: 6px; padding: 12px; margin: 10px 0; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
.muted { color: #5b6472; }
""".strip()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CodeLeWM v1.0 paper demo</title>
<style>{css}</style>
</head>
<body>
<main>
<h1>CodeLeWM v1.0 Paper Demo</h1>
<p class="muted">Deterministic replay over checked-in v0.9 WS-D score rows. Candidate code is treated as untrusted data and is not imported or executed.</p>
<section class="banner">
<h2>Claim gate: {_h('open' if gate.get('allowed') else 'closed')}</h2>
<p>{_h(gate.get('approved_public_wording'))}</p>
<ul>{reasons}</ul>
</section>
<section class="stats">
<div class="stat"><div class="label">score source</div><div class="value">{_h(report.get('score_source'))}</div></div>
<div class="stat"><div class="label">slices</div><div class="value">{_h(aggregate.get('slice_count'))}</div></div>
<div class="stat"><div class="label">problems sum</div><div class="value">{_h(aggregate.get('problem_count_sum'))}</div></div>
<div class="stat"><div class="label">completions sum</div><div class="value">{_h(aggregate.get('completion_count_sum'))}</div></div>
</section>
<h2>Baseline Comparisons</h2>
<table>
<thead><tr><th>Seed</th><th>Benchmark</th><th>CodeLeWM pass@1</th><th>No-action pass@1</th><th>LLM-order pass@1</th><th>Lift vs no-action</th><th>Claim</th></tr></thead>
<tbody>{''.join(slice_rows)}</tbody>
</table>
<h2>Candidate Rankings</h2>
{ranking_sections}
<h2>Artifact Lineage</h2>
<ul class="mono">{lineage_items}</ul>
</main>
</body>
</html>
"""


def _load_source_slice(source_root: Path, seed: int, benchmark: str) -> dict[str, Any]:
    source_dir = source_root / _SOURCE_ROOT / f"seed-{seed}" / "rerank" / benchmark
    manifest_path = source_dir / "manifest.json"
    report_path = source_dir / "reports" / "execution_rerank_report.json"
    score_rows_path = source_dir / "reports" / "completion_scores.jsonl"
    secret_scan_path = source_dir / "reports" / "secret_scan_report.json"
    for required in (manifest_path, report_path, score_rows_path, secret_scan_path):
        if not required.is_file():
            raise PaperDemoError(f"missing paper-demo source artifact: {required}")

    manifest = read_artifact_manifest(manifest_path)
    validate_artifact_checksums(manifest, root=source_dir)
    report = _read_json(report_path)
    secret_scan = _read_json(secret_scan_path)
    score_rows = tuple(_read_jsonl(score_rows_path))
    if not score_rows:
        raise PaperDemoError(f"paper-demo source score rows are empty: {score_rows_path}")
    if report.get("benchmark") != benchmark:
        raise PaperDemoError(
            f"source report benchmark mismatch for {report_path}: {report.get('benchmark')!r}"
        )
    return {
        "seed": seed,
        "benchmark": benchmark,
        "benchmark_display": "HumanEval WS-D" if benchmark == "humaneval" else "MBPP-Plus WS-D",
        "source_dir": source_dir,
        "manifest_path": _repo_relative(manifest_path, source_root),
        "report_path": _repo_relative(report_path, source_root),
        "score_rows_path": _repo_relative(score_rows_path, source_root),
        "secret_scan_path": _repo_relative(secret_scan_path, source_root),
        "manifest": manifest,
        "report": report,
        "score_rows": score_rows,
        "source_secret_scan_ok": bool(secret_scan.get("ok")),
    }


def _build_report(
    source_root: Path, source_slices: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    slices = [_summarize_slice(source_root, item) for item in source_slices]
    claim_gate = _aggregate_claim_gate(slices)
    problem_count_sum = sum(int(item["metrics"]["problem_count"]) for item in slices)
    completion_count_sum = sum(int(item["metrics"]["completion_count"]) for item in slices)
    report = {
        "schema_version": PAPER_DEMO_REPORT_SCHEMA_VERSION,
        "score_source": PAPER_DEMO_SCORE_SOURCE,
        "mode": "local_replay",
        "source_selection": {
            "description": "fixed public-safe paper-demo replay over checked-in v0.9 WS-D artifacts",
            "source_root": _repo_relative(source_root, source_root),
            "result_report_path": _RESULT_REPORT_PATH.as_posix(),
            "artifact_index_path": _ARTIFACT_INDEX_PATH.as_posix(),
            "required_slices": [
                {"seed": item["seed"], "benchmark": item["benchmark"]}
                for item in slices
            ],
        },
        "candidate_code_policy": {
            "candidate_code_is_untrusted": True,
            "imports_candidate_code": False,
            "executes_candidate_code": False,
            "test_runs_candidate_code": False,
            "pass_labels_source": "checked_in_v0_9_wsd_artifacts",
            "candidate_checksum_status": "not_recorded_in_checked_in_score_rows",
            "candidate_checksum_reason": (
                "clean-checkout replay uses tracked score rows; raw completion labels "
                "and candidate code are not required or copied into this artifact"
            ),
        },
        "learned_checkpoint_lineage": [
            {
                "seed": seed,
                **payload,
                "score_source": PAPER_DEMO_SCORE_SOURCE,
                "local_replay_trust_gate_status": "not_rechecked_in_replay_mode",
            }
            for seed, payload in sorted(_CHECKPOINTS.items())
        ],
        "required_baselines": _required_baseline_status(slices),
        "aggregate_summary": {
            "slice_count": len(slices),
            "seed_count": len({item["seed"] for item in slices}),
            "benchmarks": sorted({str(item["benchmark"]) for item in slices}),
            "problem_count_sum": problem_count_sum,
            "completion_count_sum": completion_count_sum,
            "claim_allowed": bool(claim_gate["allowed"]),
            "score_direction": "higher_is_better_after_negating_energy",
        },
        "artifact_lineage": {
            "manifest_parent_policy": (
                "local replay manifest parent_artifacts are the four checksum-verifiable "
                "tracked source rerank report artifacts; checkpoint run lineage is "
                "recorded in learned_checkpoint_lineage and is rechecked in publication mode"
            ),
            "source_rerank_report_artifact_ids": [
                item["source_artifact_id"] for item in slices
            ],
            "source_benchmark_artifact_ids": sorted(
                {
                    parent
                    for item in slices
                    for parent in item["source_parent_artifact_ids"]
                }
            ),
            "training_artifact_ids": [
                _CHECKPOINTS[42]["training_artifact"],
                _CHECKPOINTS[1729]["training_artifact"],
            ],
            "checkpoint_run_artifacts": [
                _CHECKPOINTS[42]["run_artifact"],
                _CHECKPOINTS[1729]["run_artifact"],
            ],
        },
        "claim_gate": claim_gate,
        "slices": slices,
    }
    return report, claim_gate


def _summarize_slice(source_root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    report = _mapping(source["report"])
    rows = tuple(_mapping(item) for item in source["score_rows"])
    baselines = [_mapping(item) for item in report.get("baselines", ())]
    baseline_names = {str(item.get("baseline")) for item in baselines}
    missing_required = [
        baseline for baseline in _BASELINE_ORDER if baseline not in baseline_names
    ]
    if missing_required:
        raise PaperDemoError(
            f"source report missing required baseline(s): {', '.join(missing_required)}"
        )
    score_error_count = sum(1 for row in rows if bool(row.get("errors")))
    problem_rankings = _problem_rankings(rows)
    codelewm = _baseline_by_name(baselines, "codelewm")
    no_action = _baseline_by_name(baselines, "no_action")
    llm_order = _baseline_by_name(baselines, "llm_order")
    lexical = _baseline_by_name(baselines, "lexical")
    random = _baseline_by_name(baselines, "random")
    shuffled = _baseline_by_name(baselines, "shuffled_action")
    manifest = source["manifest"]
    checkpoint = _CHECKPOINTS[int(source["seed"])]
    metrics = {
        "problem_count": int(report.get("problem_count")),
        "completion_count": int(_nested(report, "scoring_summary", "completion_count") or len(rows)),
        "completion_rows": len(rows),
        "valid_candidate_count": len(rows) - score_error_count,
        "candidate_parser_apply_error_count": score_error_count,
        "score_error_count": score_error_count,
        "pass_at_k": int(report.get("pass_at_k")),
        "confidence_level": _json_number(report.get("confidence_level")),
        "baselines": baselines,
        "deltas": {
            "codelewm_minus_no_action": _metric_delta(codelewm, no_action),
            "codelewm_minus_llm_order": _metric_delta(codelewm, llm_order),
            "codelewm_minus_lexical": _metric_delta(codelewm, lexical),
            "codelewm_minus_random": _metric_delta(codelewm, random),
            "codelewm_minus_shuffled_action": _metric_delta(codelewm, shuffled),
        },
        "bootstrap_lift_ci": report.get("bootstrap_lift_ci"),
        "bootstrap_lift_over_no_action_ci": report.get("bootstrap_lift_over_no_action_ci"),
    }
    return {
        "seed": source["seed"],
        "benchmark": source["benchmark"],
        "benchmark_display": source["benchmark_display"],
        "score_source": PAPER_DEMO_SCORE_SOURCE,
        "checkpoint_lineage": {
            **checkpoint,
            "score_direction": "higher_is_better_after_negating_energy",
            "source_report_checkpoint_sha256": _nested(report, "scoring_summary", "checkpoint_sha256"),
        },
        "manifest_path": source["manifest_path"],
        "report_path": source["report_path"],
        "score_rows_path": source["score_rows_path"],
        "source_artifact_id": manifest.artifact_id,
        "source_artifact_kind": manifest.artifact_kind,
        "source_git_sha": manifest.source_git_sha,
        "source_parent_artifact_ids": list(manifest.parent_artifacts),
        "source_checksums_verified": True,
        "source_secret_scan_ok": bool(source["source_secret_scan_ok"]),
        "source_file_sha256": {
            "report": sha256_file(source_root / source["report_path"]),
            "score_rows": sha256_file(source_root / source["score_rows_path"]),
        },
        "metrics": metrics,
        "required_baseline_status": _slice_baseline_status(baseline_names),
        "claim_allowed": bool(report.get("claim_allowed")),
        "claim_reason": str(report.get("claim_reason") or ""),
        "candidate_rankings": problem_rankings,
    }


def _problem_rankings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("problem_id"))].append(row)
    summaries: list[dict[str, Any]] = []
    for problem_id in sorted(grouped):
        candidates = grouped[problem_id]
        rankings_by_baseline = {
            baseline: [
                str(candidate.get("completion_id"))
                for candidate in _rank_by_baseline(candidates, baseline)
            ]
            for baseline in _BASELINE_ORDER
        }
        top_candidates = [
            _candidate_preview(candidate)
            for candidate in _rank_by_baseline(candidates, "codelewm")[:3]
        ]
        codelewm_top = _rank_by_baseline(candidates, "codelewm")[0]
        no_action_top = _rank_by_baseline(candidates, "no_action")[0]
        llm_top = _rank_by_baseline(candidates, "llm_order")[0]
        summaries.append(
            {
                "problem_id": problem_id,
                "split": str(candidates[0].get("split") or "unknown"),
                "completion_count": len(candidates),
                "rankings_by_baseline": rankings_by_baseline,
                "top_candidates": top_candidates,
                "top_candidate_outcomes": {
                    "codelewm": _top_outcome(codelewm_top),
                    "no_action": _top_outcome(no_action_top),
                    "llm_order": _top_outcome(llm_top),
                },
            }
        )
    return summaries


def _rank_by_baseline(
    rows: Sequence[Mapping[str, Any]], baseline: str
) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -_score(row, baseline),
            int(row.get("llm_order_rank") or 0),
            str(row.get("completion_id")),
        ),
    )


def _candidate_preview(row: Mapping[str, Any]) -> dict[str, Any]:
    completion_id = str(row.get("completion_id") or "")
    scores = _mapping(row.get("scores"))
    return {
        "completion_id": completion_id,
        "completion_id_sha256": _sha256_text(completion_id),
        "llm_order_rank": int(row.get("llm_order_rank") or 0),
        "passed": bool(row.get("passed")),
        "scores": {
            baseline: _json_number(scores.get(baseline))
            for baseline in _BASELINE_ORDER
            if baseline in scores
        },
        "score_deltas": {
            "codelewm_minus_no_action": _json_number(
                _score(row, "codelewm") - _score(row, "no_action")
            ),
            "codelewm_minus_llm_order": _json_number(
                _score(row, "codelewm") - _score(row, "llm_order")
            ),
        },
    }


def _top_outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "completion_id": str(row.get("completion_id") or ""),
        "passed": bool(row.get("passed")),
        "llm_order_rank": int(row.get("llm_order_rank") or 0),
    }


def _aggregate_claim_gate(slices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    for item in slices:
        metrics = _mapping(item.get("metrics"))
        codelewm = _mapping(_baseline(metrics, "codelewm"))
        no_action = _mapping(_baseline(metrics, "no_action"))
        llm_order = _mapping(_baseline(metrics, "llm_order"))
        seed = item.get("seed")
        benchmark = item.get("benchmark")
        if not bool(item.get("claim_allowed")):
            reasons.append(f"slice_claim_closed:seed_{seed}_{benchmark}:{item.get('claim_reason')}")
        if float(codelewm.get("pass_at_1", 0.0)) <= float(no_action.get("pass_at_1", 0.0)):
            reasons.append(f"no_action_not_strictly_beaten:seed_{seed}_{benchmark}")
        if float(codelewm.get("pass_at_1", 0.0)) <= float(llm_order.get("pass_at_1", 0.0)):
            reasons.append(f"llm_order_not_strictly_beaten:seed_{seed}_{benchmark}")
    allowed = not reasons
    if not reasons:
        reasons = ["all_seed_benchmark_slices_clear_required_gates"]
    return {
        "schema_version": PAPER_DEMO_CLAIM_GATE_SCHEMA_VERSION,
        "allowed": allowed,
        "status": "open" if allowed else "closed",
        "score_source": PAPER_DEMO_SCORE_SOURCE,
        "reasons": reasons,
        "blocked_positive_claims": [
            "aggregate downstream coding usefulness",
            "CodeLeWM improves generated code across both v0.9 downstream benchmarks",
            "positive model-quality claim beyond the HumanEval WS-D diagnostic slice",
        ],
        "approved_public_wording": (
            "On the v0.9 WS-D replay, CodeLeWM strongly reranks HumanEval "
            "slices but the aggregate downstream claim remains closed because "
            "MBPP-Plus is saturated against the no-action baseline."
        ),
    }


def _required_baseline_status(
    slices: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    observed = {
        str(baseline.get("baseline"))
        for item in slices
        for baseline in _mapping(item.get("metrics")).get("baselines", ())
        if isinstance(baseline, Mapping)
    }
    status = [
        {"baseline": baseline, "status": "recorded"}
        for baseline in _BASELINE_ORDER
        if baseline in observed
    ]
    for baseline in _OPTIONAL_INDEX_BASELINES:
        status.append(
            {
                "baseline": baseline,
                "status": "not_recorded",
                "reason": (
                    "local replay uses checked-in v0.9 score rows and no verified "
                    "transition index was part of the paper-demo source selection"
                ),
            }
        )
    return status


def _slice_baseline_status(observed: set[str]) -> list[dict[str, Any]]:
    status = [
        {"baseline": baseline, "status": "recorded"}
        for baseline in _BASELINE_ORDER
        if baseline in observed
    ]
    for baseline in _OPTIONAL_INDEX_BASELINES:
        status.append(
            {
                "baseline": baseline,
                "status": "not_recorded",
                "reason": "no verified transition index score row in v0.9 replay artifact",
            }
        )
    return status


def _metric_delta(
    codelewm: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "baseline": baseline.get("baseline"),
        "pass_at_1_points": _json_number(
            100.0
            * (
                float(codelewm.get("pass_at_1", 0.0))
                - float(baseline.get("pass_at_1", 0.0))
            )
        ),
        "mrr_points": _json_number(
            100.0
            * (
                float(codelewm.get("mrr", 0.0))
                - float(baseline.get("mrr", 0.0))
            )
        ),
        "pass_count_delta": int(codelewm.get("pass_count", 0))
        - int(baseline.get("pass_count", 0)),
    }


def _baseline_by_name(
    baselines: Sequence[Mapping[str, Any]], name: str
) -> Mapping[str, Any]:
    for baseline in baselines:
        if baseline.get("baseline") == name:
            return baseline
    raise PaperDemoError(f"source report missing baseline: {name}")


def _baseline(metrics: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for item in metrics.get("baselines", ()):
        if isinstance(item, Mapping) and item.get("baseline") == name:
            return item
    return None


def _slice_ranking_html(item: Mapping[str, Any]) -> str:
    problems = item.get("candidate_rankings", ())
    problem_markup = []
    for problem in problems[:8]:
        if not isinstance(problem, Mapping):
            continue
        top = _mapping(_nested(problem, "top_candidate_outcomes", "codelewm"))
        top_candidates = "".join(
            f"<li>{_h(candidate.get('completion_id'))} "
            f"pass={_h(candidate.get('passed'))} "
            f"score={_h(_fmt_number(_nested(candidate, 'scores', 'codelewm')))}</li>"
            for candidate in problem.get("top_candidates", ())
            if isinstance(candidate, Mapping)
        )
        problem_markup.append(
            "<div class=\"problem\">"
            f"<h3>{_h(problem.get('problem_id'))}</h3>"
            f"<p class=\"muted\">CodeLeWM top: {_h(top.get('completion_id'))} "
            f"pass={_h(top.get('passed'))}</p>"
            f"<ol class=\"mono\">{top_candidates}</ol>"
            "</div>"
        )
    return (
        f"<section class=\"slice\"><h3>Seed {_h(item.get('seed'))} "
        f"{_h(item.get('benchmark_display'))}</h3>"
        f"{''.join(problem_markup)}</section>"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PaperDemoError(f"expected JSON object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise PaperDemoError(f"expected JSON object at {path}:{line_number}")
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative_secret_scan_payload(
    payload: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    root_path = root.resolve()

    def rel(value: str) -> str:
        path = Path(value)
        try:
            return path.resolve().relative_to(root_path).as_posix()
        except (OSError, ValueError):
            return value

    return {
        "schema_version": payload["schema_version"],
        "ok": bool(payload["ok"]),
        "paths_scanned": [rel(str(path)) for path in payload.get("paths_scanned", ())],
        "findings": [
            {
                **dict(finding),
                "path": rel(str(_mapping(finding).get("path"))),
            }
            for finding in payload.get("findings", ())
            if isinstance(finding, Mapping)
        ],
    }


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _score(row: Mapping[str, Any], baseline: str) -> float:
    scores = _mapping(row.get("scores"))
    value = scores.get(baseline)
    if value is None:
        return float("-inf")
    return float(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(payload: Mapping[str, Any] | Any, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _json_number(value: Any) -> float | int:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PaperDemoError(f"non-finite number in paper-demo artifact: {value!r}")
    return int(parsed) if parsed.is_integer() else parsed


def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_fraction(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_points(value: Any) -> str:
    try:
        return f"{float(value):+.1f} pts"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _h(value: Any) -> str:
    return html.escape(str(value))


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
