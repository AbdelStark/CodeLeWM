"""OpenRouter candidate-generation adapter for the LLM harness."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from codelewm.observability.logging import redact_text, redact_value
from codelewm.security.secret_scan import scan_text


OPENROUTER_CANDIDATE_REQUEST_SCHEMA_VERSION = "codelewm.openrouter_candidate_request.v1"
LLM_CANDIDATE_PACK_SCHEMA_VERSION = "codelewm.llm_candidate_pack.v1"
OPENROUTER_ADAPTER_VERSION = "codelewm.openrouter_adapter.v0.1"
OPENROUTER_SDK_PACKAGE = "openrouter"
OPENROUTER_SDK_VERSION_PIN = "0.9.1"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-4.5-sonnet"
DEFAULT_PROMPT_TEMPLATE_ID = "codelewm.openrouter.patch_candidates.v1"
DEFAULT_OUTPUT_POLICY = "unified_diff"
ALLOWED_PROVIDER_OPTION_KEYS = frozenset(
    {"order", "sort", "allow_fallbacks", "require_parameters", "zdr"}
)


class OpenRouterAdapterError(ValueError):
    """Raised when OpenRouter candidate generation cannot be configured or run."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "openrouter_adapter_error",
        remediation: str = "fix the OpenRouter adapter request and retry",
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.remediation = remediation

    def to_error_report(self) -> dict[str, Any]:
        return {
            "schema_version": "codelewm.error.v1",
            "error_type": self.error_type,
            "message": redact_text(str(self)),
            "remediation": self.remediation,
            "record_id": None,
            "artifact": None,
            "caused_by": None,
        }


@dataclass(frozen=True)
class OpenRouterCandidateRequest:
    """Schema-versioned request for OpenRouter candidate generation."""

    task_id: str
    instruction: str
    context_bundle: Mapping[str, str] = field(default_factory=dict)
    model: str = DEFAULT_OPENROUTER_MODEL
    max_candidates: int = 4
    timeout_seconds: int = 120
    temperature: float = 0.2
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = True
    retry_limit: int = 2
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID
    output_policy: str = DEFAULT_OUTPUT_POLICY
    http_referer: str | None = None
    app_title: str | None = "CodeLeWM"
    schema_version: str = OPENROUTER_CANDIDATE_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPENROUTER_CANDIDATE_REQUEST_SCHEMA_VERSION:
            raise OpenRouterAdapterError(
                f"schema_version must be {OPENROUTER_CANDIDATE_REQUEST_SCHEMA_VERSION!r}",
                error_type="schema_error",
            )
        if not self.task_id:
            raise OpenRouterAdapterError("task_id must not be empty", error_type="schema_error")
        if not self.instruction:
            raise OpenRouterAdapterError("instruction must not be empty", error_type="schema_error")
        if not self.model:
            raise OpenRouterAdapterError("model must not be empty", error_type="schema_error")
        if self.max_candidates < 1:
            raise OpenRouterAdapterError("max_candidates must be >= 1", error_type="schema_error")
        if self.timeout_seconds < 1:
            raise OpenRouterAdapterError("timeout_seconds must be >= 1", error_type="schema_error")
        if not 0.0 <= self.temperature <= 2.0:
            raise OpenRouterAdapterError(
                "temperature must be between 0.0 and 2.0",
                error_type="schema_error",
            )
        if self.retry_limit < 0:
            raise OpenRouterAdapterError("retry_limit must be >= 0", error_type="schema_error")
        if self.output_policy != DEFAULT_OUTPUT_POLICY:
            raise OpenRouterAdapterError(
                f"output_policy must be {DEFAULT_OUTPUT_POLICY!r}",
                error_type="schema_error",
            )
        _validate_context_bundle(self.context_bundle)
        _validate_provider_options(self.provider_options)

    @classmethod
    def from_env(
        cls,
        *,
        task_id: str,
        instruction: str,
        context_bundle: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "OpenRouterCandidateRequest":
        """Build a request from documented CodeLeWM/OpenRouter env vars."""

        env = os.environ if env is None else env
        provider = env.get("CODELEWM_LLM_PROVIDER", "openrouter")
        if provider != "openrouter":
            raise OpenRouterAdapterError(
                "CODELEWM_LLM_PROVIDER must be 'openrouter'",
                error_type="config_error",
                remediation="set CODELEWM_LLM_PROVIDER=openrouter",
            )
        return cls(
            task_id=task_id,
            instruction=instruction,
            context_bundle=context_bundle or {},
            model=env.get("CODELEWM_LLM_MODEL", DEFAULT_OPENROUTER_MODEL),
            max_candidates=_parse_int_env(env, "CODELEWM_LLM_MAX_CANDIDATES", default=4),
            timeout_seconds=_parse_int_env(env, "CODELEWM_LLM_TIMEOUT_SECONDS", default=120),
            temperature=_parse_float_env(env, "CODELEWM_LLM_TEMPERATURE", default=0.2),
            provider_options=_parse_provider_options_env(env),
            dry_run=_parse_bool_env(env, "CODELEWM_LLM_DRY_RUN", default=True),
            retry_limit=_parse_int_env(env, "CODELEWM_LLM_RETRY_LIMIT", default=2),
            http_referer=_empty_to_none(env.get("OPENROUTER_HTTP_REFERER")),
            app_title=_empty_to_none(env.get("OPENROUTER_APP_TITLE")) or "CodeLeWM",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "instruction": redact_text(self.instruction),
            "context_bundle": _redacted_context_bundle(self.context_bundle),
            "model": self.model,
            "max_candidates": self.max_candidates,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "provider_options": redact_value(dict(self.provider_options)),
            "dry_run": self.dry_run,
            "retry_limit": self.retry_limit,
            "prompt_template_id": self.prompt_template_id,
            "output_policy": self.output_policy,
            "http_referer": self.http_referer,
            "app_title": self.app_title,
        }


@dataclass(frozen=True)
class LLMCandidate:
    """One generated candidate patch."""

    candidate_id: str
    patch_text: str
    parser_status: str
    dry_run_patch_status: str
    generation_error: str | None = None
    provider_finish_reason: str | None = None
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        scan = scan_text(self.patch_text, path=f"{self.candidate_id}.patch")
        return {
            "candidate_id": self.candidate_id,
            "patch_text": redact_text(self.patch_text),
            "normalized_patch_sha256": _sha256_text(_normalize_patch(self.patch_text)),
            "parser_status": self.parser_status,
            "dry_run_patch_status": self.dry_run_patch_status,
            "generation_error": None
            if self.generation_error is None
            else redact_text(self.generation_error),
            "provider_finish_reason": self.provider_finish_reason,
            "token_count": self.token_count,
            "content_sha256": _sha256_text(self.patch_text),
            "redaction": {
                "secret_scan_ok": not scan,
                "secret_findings_count": len(scan),
            },
        }


@dataclass(frozen=True)
class LLMCandidatePack:
    """Schema-versioned generated candidate pack."""

    request: OpenRouterCandidateRequest
    prompt_text: str
    candidates: tuple[LLMCandidate, ...]
    errors: tuple[Mapping[str, Any], ...] = ()
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)
    sdk_version: str | None = None
    created_at: str = "1970-01-01T00:00:00Z"
    artifact_manifest: str | None = None
    schema_version: str = LLM_CANDIDATE_PACK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        if self.schema_version != LLM_CANDIDATE_PACK_SCHEMA_VERSION:
            raise OpenRouterAdapterError("unsupported candidate-pack schema", error_type="schema_error")
        prompt_scan = scan_text(self.prompt_text, path="prompt.txt")
        return {
            "schema_version": self.schema_version,
            "task_id": self.request.task_id,
            "prompt": {
                "template_id": self.request.prompt_template_id,
                "rendered_sha256": _sha256_text(self.prompt_text),
                "redacted_prompt_path": None,
                "prompt_preview": redact_text(self.prompt_text[:1000]),
                "secret_scan": {
                    "ok": not prompt_scan,
                    "findings_count": len(prompt_scan),
                },
            },
            "context_hash": _sha256_json(self.request.context_bundle),
            "generator": {
                "provider": "openrouter",
                "model": self.request.model,
                "sdk": OPENROUTER_SDK_PACKAGE,
                "sdk_version": self.sdk_version,
                "adapter_version": OPENROUTER_ADAPTER_VERSION,
            },
            "provider_routing": {
                "requested_model": self.request.model,
                "requested_provider_options": redact_value(dict(self.request.provider_options)),
                "response_metadata": redact_value(dict(self.provider_metadata)),
            },
            "generation_config": {
                "max_candidates": self.request.max_candidates,
                "timeout_seconds": self.request.timeout_seconds,
                "temperature": self.request.temperature,
                "retry_limit": self.request.retry_limit,
                "dry_run": self.request.dry_run,
                "output_policy": self.request.output_policy,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "errors": [redact_value(dict(error)) for error in self.errors],
            "created_at": self.created_at,
            "artifact_manifest": self.artifact_manifest,
        }


def generate_candidate_pack(
    request: OpenRouterCandidateRequest,
    *,
    env: Mapping[str, str] | None = None,
) -> LLMCandidatePack:
    """Generate a candidate pack using dry-run fixtures or OpenRouter live mode."""

    env = os.environ if env is None else env
    prompt = render_candidate_prompt(request)
    if request.dry_run:
        return _generate_dry_run_candidate_pack(request, prompt)

    api_key = env.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterAdapterError(
            "OPENROUTER_API_KEY is required for live OpenRouter generation",
            error_type="missing_openrouter_api_key",
            remediation="set OPENROUTER_API_KEY or use CODELEWM_LLM_DRY_RUN=1",
        )
    if env.get("OPENROUTER_DEBUG", "").lower() in {"1", "true", "yes", "on"}:
        raise OpenRouterAdapterError(
            "OPENROUTER_DEBUG must be disabled for publishable live runs",
            error_type="unsafe_debug_logging",
            remediation="unset OPENROUTER_DEBUG before running CodeLeWM live generation",
        )

    sdk_version = _openrouter_sdk_version()
    try:
        from openrouter import OpenRouter  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OpenRouterAdapterError(
            "OpenRouter SDK is not installed",
            error_type="optional_dependency_missing",
            remediation="install the LLM extra/group that pins openrouter==0.9.1",
        ) from exc

    errors: list[Mapping[str, Any]] = []
    response_text = ""
    response_metadata: Mapping[str, Any] = {}
    for attempt in range(request.retry_limit + 1):
        try:
            with OpenRouter(
                api_key=api_key,
                http_referer=request.http_referer,
                x_open_router_title=request.app_title,
                timeout_ms=request.timeout_seconds * 1000,
            ) as open_router:
                response = open_router.chat.send(
                    messages=[{"role": "user", "content": prompt}],
                    model=request.model,
                    provider=dict(request.provider_options),
                    temperature=request.temperature,
                    stream=False,
                    timeout_ms=request.timeout_seconds * 1000,
                )
            response_text = _extract_response_text(response)
            response_metadata = _extract_response_metadata(response)
            break
        except Exception as exc:  # pragma: no cover - live provider failure path
            errors.append(
                {
                    "error_type": "provider_request_error",
                    "attempt": attempt,
                    "message": redact_text(str(exc)),
                }
            )
            if attempt >= request.retry_limit:
                raise OpenRouterAdapterError(
                    "OpenRouter provider request failed",
                    error_type="provider_request_error",
                    remediation="check model slug, provider options, quota, and network",
                ) from exc

    candidates = _candidates_from_response_text(response_text, request.max_candidates)
    return LLMCandidatePack(
        request=request,
        prompt_text=prompt,
        candidates=tuple(candidates),
        errors=tuple(errors),
        provider_metadata=response_metadata,
        sdk_version=sdk_version,
        created_at=_utc_now_iso(),
    )


def render_candidate_prompt(request: OpenRouterCandidateRequest) -> str:
    """Render the deterministic prompt sent to OpenRouter."""

    context_lines = []
    for path, content in sorted(request.context_bundle.items()):
        context_lines.append(f"### File: {path}\n```python\n{content}\n```")
    context_block = "\n\n".join(context_lines) if context_lines else "No repository context provided."
    return (
        "You are generating candidate unified diffs for CodeLeWM.\n"
        f"Task id: {request.task_id}\n"
        f"Instruction: {request.instruction}\n"
        f"Return exactly {request.max_candidates} candidates when possible.\n"
        "Each candidate must start with '### Candidate <id>' and contain only a unified diff.\n"
        "Do not include prose outside candidate blocks.\n\n"
        f"{context_block}\n"
    )


def _generate_dry_run_candidate_pack(
    request: OpenRouterCandidateRequest, prompt: str
) -> LLMCandidatePack:
    candidates = tuple(
        LLMCandidate(
            candidate_id=f"candidate_{index:03d}",
            patch_text=_dry_run_patch(request, index=index),
            parser_status="not_parsed",
            dry_run_patch_status="not_applied_issue_188",
            provider_finish_reason="dry_run_fixture",
            token_count=None,
        )
        for index in range(1, request.max_candidates + 1)
    )
    return LLMCandidatePack(
        request=request,
        prompt_text=prompt,
        candidates=candidates,
        provider_metadata={"mode": "dry_run_fixture"},
        sdk_version=None,
        created_at="1970-01-01T00:00:00Z",
    )


def _dry_run_patch(request: OpenRouterCandidateRequest, *, index: int) -> str:
    first_path = next(iter(sorted(request.context_bundle)), "candidate.py")
    marker = _sha256_text(f"{request.task_id}:{request.instruction}:{index}")[:12]
    return (
        f"### Candidate candidate_{index:03d}\n"
        f"--- a/{first_path}\n"
        f"+++ b/{first_path}\n"
        "@@\n"
        f"+# CodeLeWM dry-run candidate {index}: {marker}\n"
    )


def _candidates_from_response_text(text: str, max_candidates: int) -> list[LLMCandidate]:
    blocks = _split_candidate_blocks(text)
    candidates: list[LLMCandidate] = []
    for index, block in enumerate(blocks[:max_candidates], start=1):
        candidates.append(
            LLMCandidate(
                candidate_id=f"candidate_{index:03d}",
                patch_text=block.strip(),
                parser_status="not_parsed",
                dry_run_patch_status="not_applied_issue_188",
                provider_finish_reason=None,
                token_count=None,
            )
        )
    if not candidates and text.strip():
        candidates.append(
            LLMCandidate(
                candidate_id="candidate_001",
                patch_text=text.strip(),
                parser_status="not_parsed",
                dry_run_patch_status="not_applied_issue_188",
            )
        )
    return candidates


def _split_candidate_blocks(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if "### Candidate" in stripped:
        parts = stripped.split("### Candidate")
        return ["### Candidate" + part for part in parts[1:] if part.strip()]
    if "\n--- " in stripped:
        parts = stripped.split("\n--- ")
        return [parts[0]] + ["--- " + part for part in parts[1:] if part.strip()]
    return [stripped]


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            return _message_content(choices[0])
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return _message_content(choices[0])
    content = getattr(response, "content", None)
    return content if isinstance(content, str) else str(response)


def _message_content(choice: Any) -> str:
    if isinstance(choice, Mapping):
        message = choice.get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(choice.get("content"), str):
            return choice["content"]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    content = getattr(choice, "content", None)
    return content if isinstance(content, str) else str(choice)


def _extract_response_metadata(response: Any) -> Mapping[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("id", "model", "provider", "created"):
        value = response.get(key) if isinstance(response, Mapping) else getattr(response, key, None)
        if value is not None:
            metadata[key] = value
    return metadata


def _validate_context_bundle(context_bundle: Mapping[str, str]) -> None:
    if not isinstance(context_bundle, Mapping):
        raise OpenRouterAdapterError("context_bundle must be an object", error_type="schema_error")
    for path, content in context_bundle.items():
        if not isinstance(path, str) or not path:
            raise OpenRouterAdapterError(
                "context_bundle keys must be non-empty strings",
                error_type="schema_error",
            )
        if not isinstance(content, str):
            raise OpenRouterAdapterError(
                "context_bundle values must be strings",
                error_type="schema_error",
            )
        lowered = path.lower()
        if lowered.endswith(".env") or ".env" in lowered or "token" in lowered:
            raise OpenRouterAdapterError(
                "context_bundle must not include token-bearing or .env paths",
                error_type="security_error",
                remediation="remove secrets from the LLM context bundle",
            )


def _validate_provider_options(provider_options: Mapping[str, Any]) -> None:
    if not isinstance(provider_options, Mapping):
        raise OpenRouterAdapterError("provider_options must be an object", error_type="schema_error")
    unknown = sorted(set(provider_options) - ALLOWED_PROVIDER_OPTION_KEYS)
    if unknown:
        raise OpenRouterAdapterError(
            f"unsupported provider option(s): {', '.join(unknown)}",
            error_type="schema_error",
        )
    _json_native(dict(provider_options), "provider_options")


def _redacted_context_bundle(context_bundle: Mapping[str, str]) -> dict[str, Any]:
    return {
        path: {
            "sha256": _sha256_text(content),
            "chars": len(content),
            "preview": redact_text(content[:400]),
        }
        for path, content in sorted(context_bundle.items())
    }


def _parse_provider_options_env(env: Mapping[str, str]) -> Mapping[str, Any]:
    raw = env.get("CODELEWM_LLM_PROVIDER_OPTIONS_JSON", "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OpenRouterAdapterError(
            "CODELEWM_LLM_PROVIDER_OPTIONS_JSON must be valid JSON",
            error_type="config_error",
        ) from exc
    if not isinstance(parsed, Mapping):
        raise OpenRouterAdapterError(
            "CODELEWM_LLM_PROVIDER_OPTIONS_JSON must decode to an object",
            error_type="config_error",
        )
    _validate_provider_options(parsed)
    return dict(parsed)


def _parse_int_env(env: Mapping[str, str], key: str, *, default: int) -> int:
    value = env.get(key)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise OpenRouterAdapterError(f"{key} must be an integer", error_type="config_error") from exc


def _parse_float_env(env: Mapping[str, str], key: str, *, default: float) -> float:
    value = env.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise OpenRouterAdapterError(f"{key} must be a float", error_type="config_error") from exc


def _parse_bool_env(env: Mapping[str, str], key: str, *, default: bool) -> bool:
    value = env.get(key)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _empty_to_none(value: str | None) -> str | None:
    return None if value in (None, "") else value


def _openrouter_sdk_version() -> str:
    try:
        return importlib.metadata.version(OPENROUTER_SDK_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_patch(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _json_native(value: Any, name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise OpenRouterAdapterError(f"{name} must be JSON-native: {exc}", error_type="schema_error") from exc
