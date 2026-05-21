"""Terminal renderer for the local LLM plus world-model demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .visual_view_model import build_demo_visual_view_model


_RESET = "\033[0m"
_COLORS = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
}


def render_demo_terminal_report(
    *,
    demo_run: Mapping[str, Any],
    manifest_verify: Mapping[str, Any],
    secret_scan: Mapping[str, Any],
    html_secret_scan: Mapping[str, Any],
    demo_report: Mapping[str, Any],
    candidate_pack: Mapping[str, Any],
    out_dir: str | Path,
    color: bool = False,
) -> str:
    """Render a concise terminal walkthrough from demo artifacts."""

    painter = _Painter(enabled=color)
    lines: list[str] = []
    view_model = build_demo_visual_view_model(
        demo_run=demo_run,
        manifest_verify=manifest_verify,
        secret_scan=secret_scan,
        html_secret_scan=html_secret_scan,
        demo_report=demo_report,
        candidate_pack=candidate_pack,
        out_dir=out_dir,
    )
    summary = _mapping(view_model.get("summary"))
    diagnostics = _mapping(view_model.get("diagnostics"))
    view_candidates = [
        candidate for candidate in view_model.get("candidates", ()) if isinstance(candidate, Mapping)
    ]
    generation_config = _mapping(candidate_pack.get("generation_config"))
    generator = _mapping(candidate_pack.get("generator"))
    provider_routing = _mapping(candidate_pack.get("provider_routing"))
    byok = _mapping(provider_routing.get("byok"))
    response_metadata = _mapping(provider_routing.get("response_metadata"))
    scores = _mapping(demo_report.get("scores"))
    claim_gate = _mapping(demo_report.get("claim_gate"))
    candidate_summary = _mapping(demo_report.get("candidate_summary"))
    candidates = [
        candidate
        for candidate in candidate_pack.get("candidates", ())
        if isinstance(candidate, Mapping)
    ]
    score_by_id = _score_by_candidate_id(scores.get("codelewm_rerank", ()))
    codelewm_order = _string_list(_mapping(demo_report.get("orders")).get("codelewm", ()))
    no_action = _mapping(scores.get("no_action"))
    no_action_score = _optional_float(no_action.get("final_score"))
    dry_run = bool(generation_config.get("dry_run"))
    mode = "fixture dry-run" if dry_run else "live OpenRouter"
    provider_options = _mapping(provider_routing.get("requested_provider_options"))

    lines.append(painter.paint("CodeLeWM LLM + World-Model Demo", "bold"))
    lines.append("=" * 42)
    lines.append(
        "mode: "
        + painter.status(mode, ok=not dry_run)
        + "  |  scorer: "
        + painter.status(str(scores.get("model_id", "unknown")), ok=_uses_learned_scorer(scores))
        + "  |  success: "
        + painter.status(str(bool(demo_report.get("success"))).lower(), ok=bool(demo_report.get("success")))
    )
    lines.append("")

    lines.extend(
        _stage(
            painter,
            "1/6",
            "Runtime and inputs",
            ok=True,
            details=(
                f"task: {_safe_text(summary.get('task_id'))}",
                f"context: {_safe_text(summary.get('context_path'))}",
                f"before: {_safe_text(summary.get('before_path'))}",
                f"checkpoint: {_safe_text(summary.get('checkpoint_short_sha'))}",
                f"output: {Path(out_dir).as_posix()}",
            ),
        )
    )
    lines.extend(
        _stage(
            painter,
            "2/6",
            "Candidate generation",
            ok=bool(candidates),
            details=(
                f"provider: {_safe_text(generator.get('provider'))}",
                f"model: {_safe_text(response_metadata.get('model') or generator.get('model'))}",
                f"sdk: {_safe_text(generator.get('sdk'))} {_safe_text(generator.get('sdk_version'))}",
                f"routing: {_routing_summary(provider_options, byok)}",
                f"candidates: {candidate_summary.get('valid_candidate_count', 0)}/{candidate_summary.get('candidate_count', 0)} valid",
            ),
        )
    )
    lines.extend(
        _stage(
            painter,
            "3/6",
            "Candidate pack captured",
            ok=_artifact_ok(demo_run),
            details=(
                f"artifact: {_safe_text(_mapping(demo_report.get('artifacts')).get('candidate_pack_manifest_id'))}",
                "patch content: stored in artifact, not printed in terminal",
            ),
        )
    )

    best_candidate = codelewm_order[0] if codelewm_order else "none"
    best_score = score_by_id.get(best_candidate)
    score_range = _score_range(score_by_id.values())
    delta = None if best_score is None or no_action_score is None else best_score - no_action_score
    lines.extend(
        _stage(
            painter,
            "4/6",
            "World-model inference",
            ok=_uses_learned_scorer(scores),
            details=(
                f"backend: {_safe_text(scores.get('model_id'))}",
                f"warnings: {_compact_warnings(demo_report.get('warnings', ()))}",
                f"score direction: {_score_direction_text(summary.get('score_direction'))}",
                f"best candidate: {best_candidate} ({_format_score(best_score)})",
                f"no-action: {_format_score(no_action_score)}  candidate - no-action: {_format_noop_delta(delta)}",
                f"score range: {score_range}",
            ),
        )
    )
    lines.extend(
        _stage(
            painter,
            "5/6",
            "Artifact gates",
            ok=bool(manifest_verify.get("ok")) and bool(secret_scan.get("ok")) and bool(html_secret_scan.get("ok")),
            details=(
                f"manifest verify: ok={str(bool(manifest_verify.get('ok'))).lower()} files={manifest_verify.get('files_checked', 0)}",
                f"artifact scan: ok={str(bool(secret_scan.get('ok'))).lower()} findings={len(secret_scan.get('findings', ())) if isinstance(secret_scan.get('findings'), Sequence) else 0}",
                f"html scan: ok={str(bool(html_secret_scan.get('ok'))).lower()} findings={len(html_secret_scan.get('findings', ())) if isinstance(html_secret_scan.get('findings'), Sequence) else 0}",
            ),
        )
    )
    lines.extend(
        _stage(
            painter,
            "6/6",
            "Claim gate",
            ok=not bool(claim_gate.get("allowed")),
            details=(
                f"allowed: {str(bool(claim_gate.get('allowed'))).lower()}",
                f"reason: {_safe_text(claim_gate.get('reason'))}",
                "interpretation: workflow evidence only, not a model-quality claim",
            ),
        )
    )

    lines.append("")
    lines.append(painter.paint("Candidate ranking", "bold"))
    lines.append("-" * 42)
    lines.extend(_candidate_table(view_candidates, painter))

    lines.append("")
    lines.append(painter.paint("Diagnostics", "bold"))
    lines.append("-" * 42)
    lines.extend(_diagnostics_table(diagnostics))

    lines.append("")
    lines.append(painter.paint("Artifacts", "bold"))
    lines.append("-" * 42)
    lines.append(f"html report: {Path(out_dir, str(demo_run.get('html_path', 'demo.html'))).as_posix()}")
    lines.append(f"json report: {Path(out_dir, str(demo_run.get('report_path', 'reports/llm_world_model_demo_report.json'))).as_posix()}")
    lines.append(
        "view model: "
        f"{Path(out_dir, str(demo_run.get('visual_view_model_path', 'reports/visual_view_model.json'))).as_posix()}"
    )
    lines.append(f"manifest: {Path(out_dir, str(demo_run.get('artifact_manifest_path', 'manifest.json'))).as_posix()}")
    lines.append("")
    lines.append(
        "raw mode: "
        + painter.paint("uv run scripts/llm-world-model-demo --json", "cyan")
        + " or "
        + painter.paint("CODELEWM_LLM_DEMO_OUTPUT=json", "cyan")
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="render a terminal report for a CodeLeWM demo run")
    parser.add_argument("--demo-run", type=Path, required=True)
    parser.add_argument("--manifest-verify", type=Path, required=True)
    parser.add_argument("--secret-scan", type=Path, required=True)
    parser.add_argument("--html-secret-scan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-pack", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    args = parser.parse_args(argv)

    color = args.color == "always" or (
        args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    )
    print(
        render_demo_terminal_report(
            demo_run=_read_json(args.demo_run),
            manifest_verify=_read_json(args.manifest_verify),
            secret_scan=_read_json(args.secret_scan),
            html_secret_scan=_read_json(args.html_secret_scan),
            demo_report=_read_json(args.report),
            candidate_pack=_read_json(args.candidate_pack),
            out_dir=args.out_dir,
            color=color,
        ),
        end="",
    )
    return 0


class _Painter:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, value: object, color: str) -> str:
        text = str(value)
        if not self.enabled:
            return text
        return _COLORS[color] + text + _RESET

    def status(self, value: object, *, ok: bool) -> str:
        return self.paint(value, "green" if ok else "yellow")


def _stage(
    painter: _Painter,
    index: str,
    title: str,
    *,
    ok: bool,
    details: Sequence[str],
) -> list[str]:
    marker = painter.paint("[ok]", "green") if ok else painter.paint("[warn]", "yellow")
    lines = [f"{marker} {index} {painter.paint(title, 'bold')}"]
    lines.extend(f"     {detail}" for detail in details)
    lines.append("")
    return lines


def _candidate_table(
    candidates: Sequence[Mapping[str, Any]],
    painter: _Painter,
) -> list[str]:
    if not candidates:
        return ["no candidates captured"]
    lines = [
        f"{'rank':<4} {'candidate':<14} {'score':>12} {'vs no-op':>12} "
        f"{'status':<28} {'diff':<17} patch"
    ]
    ordered = sorted(
        candidates,
        key=lambda item: item.get("rank") if isinstance(item.get("rank"), int) else 9999,
    )
    for candidate in ordered:
        rank_value = candidate.get("rank")
        rank = painter.paint(rank_value or "n/a", "green" if rank_value == 1 else "dim")
        candidate_id = str(candidate.get("candidate_id", "unknown"))
        patch_summary = _mapping(candidate.get("patch_summary"))
        diff_summary = (
            f"{patch_summary.get('changed_file_count', 0)}f/"
            f"{patch_summary.get('hunk_count', 0)}h "
            f"+{patch_summary.get('additions', 0)}"
            f"-{patch_summary.get('deletions', 0)}"
        )
        patch = _short_sha(candidate.get("patch_sha256"))
        lines.append(
            f"{str(rank):<4} {candidate_id:<14} {_safe_text(candidate.get('score_display')):>12} "
            f"{_safe_text(candidate.get('no_action_delta_display')):>12} "
            f"{_truncate(str(candidate.get('status', 'unknown')), 28):<28} "
            f"{diff_summary:<17} {patch}"
        )
    return lines


def _diagnostics_table(diagnostics: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for label, key in (
        ("checkpoint inspection", "checkpoint_inspection"),
        ("latent matrix", "latent_matrix"),
        ("run timeline", "run_timeline"),
        ("tensorboard", "tensorboard"),
    ):
        slot = _mapping(diagnostics.get(key))
        artifact = slot.get("artifact_id")
        checksum = slot.get("sha256")
        suffix = ""
        if artifact:
            suffix += f" artifact={_safe_text(artifact)}"
        if checksum:
            suffix += f" sha256={_safe_text(str(checksum)[:12])}"
        lines.append(
            f"{label}: {_safe_text(slot.get('status'))}"
            f"{' -> ' + _safe_text(slot.get('path')) if slot.get('path') else ''}"
            f"{suffix}"
        )
    return lines


def _score_by_candidate_id(rows: Any) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return scores
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate = Path(str(row.get("candidate", ""))).stem
        value = _optional_float(row.get("final_score"))
        if candidate and value is not None:
            scores[candidate] = value
    return scores


def _uses_learned_scorer(scores: Mapping[str, Any]) -> bool:
    return str(scores.get("model_id", "")).startswith("codelewm.torch_transition_scorer")


def _artifact_ok(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("success", False)) and bool(str(payload.get("artifact_manifest_id", "")))


def _routing_summary(provider_options: Mapping[str, Any], byok: Mapping[str, Any]) -> str:
    only = provider_options.get("only")
    only_text = ",".join(str(item) for item in only) if isinstance(only, Sequence) and not isinstance(only, str) else "any"
    fallback = provider_options.get("allow_fallbacks")
    byok_state = "byok:on" if byok.get("enabled") else "byok:off"
    return f"only={only_text} fallback={fallback} {byok_state}"


def _compact_warnings(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return "none"
    warnings = [str(item) for item in value if str(item)]
    if not warnings:
        return "none"
    return "; ".join(warnings[:3])


def _score_range(values: Iterable[float]) -> str:
    finite = [value for value in values if isinstance(value, float)]
    if not finite:
        return "n/a"
    return f"{min(finite):.6f} .. {max(finite):.6f}"


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.6f}"


def _format_noop_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0:
        return f"{_format_delta(value)} (better than no-op)"
    if value > 0:
        return f"{_format_delta(value)} (worse than no-op)"
    return f"{_format_delta(value)} (tied with no-op)"


def _score_direction_text(value: object) -> str:
    if value == "lower_is_better":
        return "lower transition energy is better"
    return _safe_text(value)


def _optional_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def _safe_text(value: object) -> str:
    if value is None:
        return "n/a"
    return _truncate(str(value), 88)


def _short_sha(value: object) -> str:
    text = "" if value is None else str(value)
    return text[:12] if text else "n/a"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
