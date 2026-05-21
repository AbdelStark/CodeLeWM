"""Shared visual view model for LLM plus world-model demo reports."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from codelewm.observability.logging import redact_text, redact_value


DEMO_VISUAL_VIEW_MODEL_SCHEMA_VERSION = "codelewm.harness.visual_view_model.v1"
SCORE_DIRECTION_LOWER_IS_BETTER = "lower_is_better"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DIFF_FILE_RE = re.compile(r"^\+\+\+\s+b/(.+)$")


class DemoVisualViewModelError(ValueError):
    """Raised when a demo visual view model is malformed."""


def build_demo_visual_view_model(
    *,
    demo_report: Mapping[str, Any],
    candidate_pack: Mapping[str, Any],
    out_dir: str | Path,
    demo_run: Mapping[str, Any] | None = None,
    manifest_verify: Mapping[str, Any] | None = None,
    secret_scan: Mapping[str, Any] | None = None,
    html_secret_scan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize demo state for JSON, terminal, HTML, and future TUI renderers."""

    artifacts = _mapping(demo_report.get("artifacts"))
    scores = _mapping(demo_report.get("scores"))
    claim_gate = _mapping(demo_report.get("claim_gate"))
    candidate_summary = _mapping(demo_report.get("candidate_summary"))
    task = _mapping(demo_report.get("task"))
    generator = _mapping(candidate_pack.get("generator"))
    generation_config = _mapping(candidate_pack.get("generation_config"))
    provider_routing = _mapping(candidate_pack.get("provider_routing"))
    response_metadata = _mapping(provider_routing.get("response_metadata"))
    byok = _mapping(provider_routing.get("byok"))
    provider_options = _mapping(provider_routing.get("requested_provider_options"))
    codelewm_order = _string_list(_mapping(demo_report.get("orders")).get("codelewm", ()))
    score_by_id = _score_by_candidate_id(scores.get("codelewm_rerank", ()))
    no_action = _mapping(scores.get("no_action"))
    no_action_score = _optional_float(no_action.get("final_score"))
    best_candidate = codelewm_order[0] if codelewm_order else None
    best_score = score_by_id.get(best_candidate or "")
    dry_run = bool(generation_config.get("dry_run"))
    mode = "fixture dry-run" if dry_run else "live OpenRouter"
    output_root = Path(out_dir)
    candidates = [
        _candidate_view(
            candidate,
            rank=(codelewm_order.index(str(candidate.get("candidate_id"))) + 1)
            if str(candidate.get("candidate_id")) in codelewm_order
            else None,
            score=score_by_id.get(str(candidate.get("candidate_id"))),
            no_action_score=no_action_score,
        )
        for candidate in candidate_pack.get("candidates", ())
        if isinstance(candidate, Mapping)
    ]
    payload = {
        "schema_version": DEMO_VISUAL_VIEW_MODEL_SCHEMA_VERSION,
        "summary": {
            "mode": mode,
            "dry_run": dry_run,
            "success": bool(demo_report.get("success")),
            "task_id": _safe_string(task.get("task_id")),
            "context_path": _safe_string(task.get("context_path")),
            "before_path": _safe_path(task.get("before_path")),
            "checkpoint_sha256": _safe_string(artifacts.get("checkpoint_sha256")),
            "checkpoint_short_sha": _short_sha(artifacts.get("checkpoint_sha256")),
            "scorer": _safe_string(scores.get("model_id")),
            "score_direction": _safe_string(
                scores.get("score_direction") or SCORE_DIRECTION_LOWER_IS_BETTER
            ),
            "best_candidate": best_candidate,
            "best_score": best_score,
            "no_action_score": no_action_score,
            "best_candidate_minus_no_action": _delta(best_score, no_action_score),
            "best_no_action_delta_interpretation": _delta_interpretation(
                _delta(best_score, no_action_score)
            ),
            "score_range": _score_range(score_by_id.values()),
            "candidate_count": _optional_int(candidate_summary.get("candidate_count")) or len(candidates),
            "valid_candidate_count": _optional_int(candidate_summary.get("valid_candidate_count")),
            "claim_allowed": bool(claim_gate.get("allowed")),
            "claim_reason": _safe_string(claim_gate.get("reason")),
        },
        "generator": {
            "provider": _safe_string(generator.get("provider")),
            "model": _safe_string(response_metadata.get("model") or generator.get("model")),
            "sdk": _safe_string(generator.get("sdk")),
            "sdk_version": _safe_string(generator.get("sdk_version")),
            "routing": _routing_summary(provider_options, byok),
            "byok_enabled": bool(byok.get("enabled")),
        },
        "orders": {
            "llm": _string_list(_mapping(demo_report.get("orders")).get("llm", ())),
            "codelewm": codelewm_order,
            "lexical": _string_list(_mapping(demo_report.get("orders")).get("lexical", ())),
            "random": _string_list(_mapping(demo_report.get("orders")).get("random", ())),
            "no_action": _string_list(_mapping(demo_report.get("orders")).get("no_action", ())),
        },
        "candidates": candidates,
        "diagnostics": {
            "checkpoint_inspection": _diagnostic_slot(
                artifacts.get("checkpoint_inspection_path")
                or artifacts.get("model_checkpoint_inspection_path")
            ),
            "latent_matrix": _diagnostic_slot(artifacts.get("latent_matrix_report_path")),
            "run_timeline": _diagnostic_slot(artifacts.get("run_timeline_path")),
            "tensorboard": _diagnostic_slot(artifacts.get("tensorboard_export_path")),
        },
        "artifact_gates": {
            "manifest_verify": _gate_view(manifest_verify),
            "artifact_secret_scan": _gate_view(secret_scan),
            "html_secret_scan": _gate_view(html_secret_scan),
        },
        "artifacts": {
            "root": output_root.as_posix(),
            "html_path": _join_artifact_path(output_root, _path_from_demo_run(demo_run, "html_path", "demo.html")),
            "report_path": _join_artifact_path(
                output_root,
                _path_from_demo_run(demo_run, "report_path", "reports/llm_world_model_demo_report.json"),
            ),
            "manifest_path": _join_artifact_path(
                output_root,
                _path_from_demo_run(demo_run, "artifact_manifest_path", "manifest.json"),
            ),
            "visual_view_model_path": _join_artifact_path(
                output_root, "reports/visual_view_model.json"
            ),
            "candidate_pack_manifest_path": _join_artifact_path(
                output_root,
                _path_from_demo_run(
                    demo_run,
                    "candidate_pack_manifest_path",
                    artifacts.get("candidate_pack_manifest_path")
                    or "candidate_pack/manifest.json",
                ),
            ),
            "run_timeline_path": _join_artifact_path(output_root, artifacts.get("run_timeline_path")),
        },
        "warnings": [redact_text(str(item)) for item in demo_report.get("warnings", ())],
        "claim_gate": dict(redact_value(claim_gate)),
    }
    return validate_demo_visual_view_model_payload(payload)


def write_demo_visual_view_model(payload: Mapping[str, Any], path: Path | str) -> None:
    """Write a validated visual view-model JSON artifact."""

    normalized = validate_demo_visual_view_model_payload(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_demo_visual_view_model(path: Path | str) -> Mapping[str, Any]:
    """Read and validate a visual view-model JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DemoVisualViewModelError("visual view model must be a JSON object")
    return validate_demo_visual_view_model_payload(payload)


def validate_demo_visual_view_model_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a visual view-model payload and return its JSON-native dict."""

    if payload.get("schema_version") != DEMO_VISUAL_VIEW_MODEL_SCHEMA_VERSION:
        raise DemoVisualViewModelError(
            "unsupported visual view model schema_version; "
            f"expected {DEMO_VISUAL_VIEW_MODEL_SCHEMA_VERSION!r}"
        )
    required = {
        "schema_version",
        "summary",
        "generator",
        "orders",
        "candidates",
        "diagnostics",
        "artifact_gates",
        "artifacts",
        "warnings",
        "claim_gate",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise DemoVisualViewModelError(
            f"visual view model missing required key(s): {', '.join(missing)}"
        )
    normalized = dict(redact_value(payload))
    _reject_ansi(normalized)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DemoVisualViewModelError("visual view model must be JSON-native") from exc
    return normalized


def _candidate_view(
    candidate: Mapping[str, Any],
    *,
    rank: int | None,
    score: float | None,
    no_action_score: float | None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id", "unknown"))
    patch_text = str(candidate.get("patch_text", ""))
    patch_summary = _patch_summary(patch_text)
    delta = _delta(score, no_action_score)
    return {
        "candidate_id": candidate_id,
        "rank": rank,
        "score": score,
        "score_display": _format_score(score),
        "candidate_minus_no_action": delta,
        "no_action_delta_display": _format_delta(delta),
        "no_action_delta_interpretation": _delta_interpretation(delta),
        "parser_status": _safe_string(candidate.get("parser_status")),
        "dry_run_patch_status": _safe_string(candidate.get("dry_run_patch_status")),
        "status": (
            f"{_safe_string(candidate.get('parser_status'))}/"
            f"{_safe_string(candidate.get('dry_run_patch_status'))}"
        ),
        "patch_sha256": _safe_string(
            candidate.get("normalized_patch_sha256") or candidate.get("content_sha256")
        ),
        "patch_short_sha": _short_sha(
            candidate.get("normalized_patch_sha256") or candidate.get("content_sha256")
        ),
        "is_valid": not bool(candidate.get("errors"))
        and candidate.get("parser_status") == "parseable_python_after_state",
        "errors": [redact_text(str(item)) for item in candidate.get("errors", ())],
        "patch_summary": patch_summary,
    }


def _patch_summary(patch_text: str) -> dict[str, Any]:
    files: list[str] = []
    additions = 0
    deletions = 0
    hunk_count = 0
    preview: list[str] = []
    for line in patch_text.splitlines():
        file_match = _DIFF_FILE_RE.match(line)
        if file_match and file_match.group(1) not in files:
            files.append(redact_text(file_match.group(1)))
        if line.startswith("@@"):
            hunk_count += 1
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
        if len(preview) < 14 and (
            line.startswith("@@")
            or line.startswith("+++")
            or line.startswith("---")
            or line.startswith("+")
            or line.startswith("-")
        ):
            preview.append(redact_text(line[:160]))
    return {
        "changed_files": files,
        "changed_file_count": len(files),
        "hunk_count": hunk_count,
        "additions": additions,
        "deletions": deletions,
        "preview_lines": preview,
        "preview_truncated": len(patch_text.splitlines()) > len(preview),
    }


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


def _gate_view(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"status": "not_configured", "ok": None, "findings": None, "files_checked": None}
    findings = payload.get("findings")
    return {
        "status": "completed",
        "ok": bool(payload.get("ok")),
        "findings": len(findings) if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes)) else 0,
        "files_checked": _optional_int(payload.get("files_checked")),
    }


def _diagnostic_slot(path: Any) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {"status": "not_configured", "path": None}
    return {"status": "available", "path": redact_text(str(path))}


def _routing_summary(provider_options: Mapping[str, Any], byok: Mapping[str, Any]) -> str:
    only = provider_options.get("only")
    only_text = (
        ",".join(str(item) for item in only)
        if isinstance(only, Sequence) and not isinstance(only, str)
        else "any"
    )
    fallback = provider_options.get("allow_fallbacks")
    byok_state = "byok:on" if byok.get("enabled") else "byok:off"
    return f"only={only_text} fallback={fallback} {byok_state}"


def _delta(score: float | None, no_action_score: float | None) -> float | None:
    if score is None or no_action_score is None:
        return None
    return float(score - no_action_score)


def _delta_interpretation(value: float | None) -> str:
    if value is None:
        return "not_available"
    if value < 0:
        return "better_than_no_action"
    if value > 0:
        return "worse_than_no_action"
    return "tied_with_no_action"


def _score_range(values: Sequence[float] | Any) -> dict[str, float | None]:
    finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not finite:
        return {"min": None, "max": None}
    return {"min": min(finite), "max": max(finite)}


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.6f}"


def _short_sha(value: object) -> str:
    text = "" if value is None else str(value)
    return text[:12] if text else "n/a"


def _join_artifact_path(root: Path, relative: object) -> str | None:
    if relative is None:
        return None
    text = str(relative)
    if not text:
        return None
    return redact_text(Path(root, text).as_posix())


def _path_from_demo_run(demo_run: Mapping[str, Any] | None, key: str, default: object) -> object:
    if isinstance(demo_run, Mapping):
        return demo_run.get(key, default)
    return default


def _safe_path(value: object) -> str:
    if value is None:
        return "n/a"
    return redact_text(str(value))


def _safe_string(value: object) -> str:
    if value is None:
        return "n/a"
    return redact_text(str(value))


def _optional_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [redact_text(str(item)) for item in value]
    return []


def _reject_ansi(value: Any) -> None:
    if isinstance(value, str):
        if _ANSI_RE.search(value):
            raise DemoVisualViewModelError("visual view model must not contain ANSI escape codes")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_ansi(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_ansi(item)
