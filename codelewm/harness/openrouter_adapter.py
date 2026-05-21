"""OpenRouter candidate-generation adapter for the LLM harness."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from codelewm.observability import build_artifact_manifest, write_artifact_manifest
from codelewm.observability.logging import redact_text, redact_value
from codelewm.security.secret_scan import scan_text
from codelewm.security import parse_python_source_text

from .scorer import ScoreError, _apply_unified_diff


OPENROUTER_CANDIDATE_REQUEST_SCHEMA_VERSION = "codelewm.openrouter_candidate_request.v1"
LLM_CANDIDATE_PACK_SCHEMA_VERSION = "codelewm.llm_candidate_pack.v1"
LLM_CANDIDATE_PACK_ARTIFACT_SCHEMA_VERSION = "codelewm.llm_candidate_pack_artifact.v1"
OPENROUTER_BYOK_REGISTER_SCHEMA_VERSION = "codelewm.openrouter_byok_register.v1"
OPENROUTER_ADAPTER_VERSION = "codelewm.openrouter_adapter.v0.1"
OPENROUTER_SDK_PACKAGE = "openrouter"
OPENROUTER_SDK_VERSION_PIN = "0.9.1"
OPENROUTER_BYOK_API_URL = "https://openrouter.ai/api/v1/byok"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-4.5-sonnet"
DEFAULT_OPENROUTER_BYOK_PROVIDER = "anthropic"
DEFAULT_OPENROUTER_BYOK_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_OPENROUTER_MANAGEMENT_KEY_ENV = "OPENROUTER_MANAGEMENT_KEY"
DEFAULT_PROMPT_TEMPLATE_ID = "codelewm.openrouter.patch_candidates.v1"
DEFAULT_OUTPUT_POLICY = "unified_diff"
MAX_CAPTURE_PATCH_CHARS = 200_000
MAX_SERIALIZED_TEXT_CHARS = 50_000
ALLOWED_PROVIDER_OPTION_KEYS = frozenset(
    {"order", "only", "sort", "allow_fallbacks", "require_parameters", "zdr"}
)
OPENROUTER_SECRET_RE = re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]+\b")


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
class CandidatePackArtifactResult:
    """Summary returned after writing a manifest-backed candidate-pack artifact."""

    artifact_manifest_id: str
    artifact_manifest_path: str
    candidate_pack_path: str
    prompt_path: str
    parent_artifacts: tuple[str, ...] = ()
    schema_version: str = LLM_CANDIDATE_PACK_ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "candidate_pack_path": self.candidate_pack_path,
            "prompt_path": self.prompt_path,
            "parent_artifacts": list(self.parent_artifacts),
        }


@dataclass(frozen=True)
class OpenRouterBYOKRegisterResult:
    """Summary returned after creating or dry-running an OpenRouter BYOK key."""

    provider: str
    credential_name: str | None
    key_env: str
    management_key_env: str
    allowed_models: tuple[str, ...]
    is_fallback: bool
    workspace_id_set: bool
    dry_run: bool
    registered: bool
    response_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = OPENROUTER_BYOK_REGISTER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "credential_name": self.credential_name,
            "key_env": self.key_env,
            "management_key_env": self.management_key_env,
            "allowed_models": list(self.allowed_models),
            "is_fallback": self.is_fallback,
            "workspace_id_set": self.workspace_id_set,
            "dry_run": self.dry_run,
            "registered": self.registered,
            "response_metadata": redact_value(dict(self.response_metadata)),
        }


@dataclass(frozen=True)
class OpenRouterBYOKConfig:
    """Redacted request-time BYOK routing and optional registration settings."""

    enabled: bool = False
    provider: str = DEFAULT_OPENROUTER_BYOK_PROVIDER
    key_env: str = DEFAULT_OPENROUTER_BYOK_KEY_ENV
    management_key_env: str = DEFAULT_OPENROUTER_MANAGEMENT_KEY_ENV
    require: bool = False
    register: bool = False
    registration_dry_run: bool = False
    credential_name: str | None = None
    allowed_models: tuple[str, ...] = ()
    workspace_id_set: bool = False
    is_fallback: bool = False

    def __post_init__(self) -> None:
        if self.enabled and not self.provider:
            raise OpenRouterAdapterError(
                "CODELEWM_OPENROUTER_BYOK_PROVIDER must not be empty",
                error_type="config_error",
            )
        if self.enabled and not self.key_env:
            raise OpenRouterAdapterError(
                "CODELEWM_OPENROUTER_BYOK_KEY_ENV must not be empty",
                error_type="config_error",
            )
        if self.enabled and self.register and not self.management_key_env:
            raise OpenRouterAdapterError(
                "CODELEWM_OPENROUTER_BYOK_MANAGEMENT_KEY_ENV must not be empty",
                error_type="config_error",
            )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str],
        *,
        model: str,
    ) -> "OpenRouterBYOKConfig":
        enabled = _parse_bool_env(env, "CODELEWM_OPENROUTER_BYOK", default=False)
        provider = env.get(
            "CODELEWM_OPENROUTER_BYOK_PROVIDER",
            DEFAULT_OPENROUTER_BYOK_PROVIDER,
        ).strip()
        key_env = env.get(
            "CODELEWM_OPENROUTER_BYOK_KEY_ENV",
            DEFAULT_OPENROUTER_BYOK_KEY_ENV,
        ).strip()
        management_key_env = env.get(
            "CODELEWM_OPENROUTER_BYOK_MANAGEMENT_KEY_ENV",
            DEFAULT_OPENROUTER_MANAGEMENT_KEY_ENV,
        ).strip()
        allowed_models = _parse_csv_env(env.get("CODELEWM_OPENROUTER_BYOK_ALLOWED_MODELS"))
        if enabled and not allowed_models:
            allowed_models = (model,)
        return cls(
            enabled=enabled,
            provider=provider,
            key_env=key_env,
            management_key_env=management_key_env,
            require=_parse_bool_env(env, "CODELEWM_OPENROUTER_BYOK_REQUIRE", default=enabled),
            register=_parse_bool_env(env, "CODELEWM_OPENROUTER_BYOK_REGISTER", default=False),
            registration_dry_run=_parse_bool_env(
                env,
                "CODELEWM_OPENROUTER_BYOK_DRY_RUN",
                default=False,
            ),
            credential_name=_empty_to_none(env.get("CODELEWM_OPENROUTER_BYOK_NAME")),
            allowed_models=allowed_models,
            workspace_id_set=bool(_empty_to_none(env.get("CODELEWM_OPENROUTER_BYOK_WORKSPACE_ID"))),
            is_fallback=_parse_bool_env(env, "CODELEWM_OPENROUTER_BYOK_IS_FALLBACK", default=False),
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "key_env": self.key_env,
            "management_key_env": self.management_key_env,
            "require": self.require,
            "register": self.register,
            "registration_dry_run": self.registration_dry_run,
            "credential_name": self.credential_name,
            "allowed_models": list(self.allowed_models),
            "workspace_id_set": self.workspace_id_set,
            "is_fallback": self.is_fallback,
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
    byok: OpenRouterBYOKConfig = field(default_factory=OpenRouterBYOKConfig)
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
        if not self.prompt_template_id:
            raise OpenRouterAdapterError(
                "prompt_template_id must not be empty",
                error_type="schema_error",
            )
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
        model = env.get("CODELEWM_LLM_MODEL", DEFAULT_OPENROUTER_MODEL)
        byok = OpenRouterBYOKConfig.from_env(env, model=model)
        provider_options = _with_byok_provider_options(_parse_provider_options_env(env), byok)
        return cls(
            task_id=task_id,
            instruction=instruction,
            context_bundle=context_bundle or {},
            model=model,
            max_candidates=_parse_int_env(env, "CODELEWM_LLM_MAX_CANDIDATES", default=4),
            timeout_seconds=_parse_int_env(env, "CODELEWM_LLM_TIMEOUT_SECONDS", default=120),
            temperature=_parse_float_env(env, "CODELEWM_LLM_TEMPERATURE", default=0.2),
            provider_options=provider_options,
            dry_run=_parse_bool_env(env, "CODELEWM_LLM_DRY_RUN", default=True),
            retry_limit=_parse_int_env(env, "CODELEWM_LLM_RETRY_LIMIT", default=2),
            prompt_template_id=env.get(
                "CODELEWM_LLM_PROMPT_TEMPLATE_ID",
                DEFAULT_PROMPT_TEMPLATE_ID,
            ),
            http_referer=_empty_to_none(env.get("OPENROUTER_HTTP_REFERER")),
            app_title=_empty_to_none(env.get("OPENROUTER_APP_TITLE")) or "CodeLeWM",
            byok=byok,
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
            "byok": self.byok.to_dict(),
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
    patch_path: str | None = None
    applied_before_path: str | None = None
    after_state_sha256: str | None = None
    patch_bytes: int | None = None
    errors: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        scan = scan_text(self.patch_text, path=f"{self.candidate_id}.patch")
        has_errors = bool(self.errors) or self.generation_error is not None
        return {
            "candidate_id": self.candidate_id,
            "patch_text": _redacted_text_preview(self.patch_text),
            "patch_path": self.patch_path,
            "normalized_patch_sha256": _sha256_text(
                _normalize_patch(_unified_diff_payload(self.patch_text))
            ),
            "parser_status": self.parser_status,
            "dry_run_patch_status": self.dry_run_patch_status,
            "generation_error": None
            if self.generation_error is None
            else redact_text(self.generation_error),
            "provider_finish_reason": self.provider_finish_reason,
            "token_count": self.token_count,
            "applied_before_path": self.applied_before_path,
            "after_state_sha256": self.after_state_sha256,
            "patch_bytes": self.patch_bytes
            if self.patch_bytes is not None
            else len(self.patch_text.encode("utf-8")),
            "content_sha256": _sha256_text(self.patch_text),
            "redaction": {
                "secret_scan_ok": not scan,
                "secret_findings_count": len(scan),
            },
            "rankability": {
                "rankable": True,
                "fallback_order": "after_valid_candidates" if has_errors else "normal",
            },
            "errors": [redact_value(dict(error)) for error in self.errors],
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
                "byok": self.request.byok.to_dict(),
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


def llm_candidate_pack_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for LLM candidate-pack artifacts."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": LLM_CANDIDATE_PACK_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": True,
        "required": [
            "schema_version",
            "task_id",
            "prompt",
            "context_hash",
            "generator",
            "provider_routing",
            "generation_config",
            "candidates",
            "errors",
            "created_at",
            "artifact_manifest",
        ],
        "properties": {
            "schema_version": {"const": LLM_CANDIDATE_PACK_SCHEMA_VERSION},
            "task_id": {"type": "string", "minLength": 1},
            "prompt": {"type": "object"},
            "context_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "generator": {
                "type": "object",
                "required": ["provider", "model", "sdk", "sdk_version", "adapter_version"],
            },
            "provider_routing": {"type": "object"},
            "generation_config": {"type": "object"},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "candidate_id",
                        "patch_text",
                        "patch_path",
                        "normalized_patch_sha256",
                        "parser_status",
                        "dry_run_patch_status",
                        "errors",
                    ],
                },
            },
            "errors": {"type": "array"},
            "created_at": {"type": "string"},
            "artifact_manifest": {"type": ["string", "null"]},
        },
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

    byok_registration: OpenRouterBYOKRegisterResult | None = None
    if request.byok.enabled and request.byok.register:
        byok_registration = register_openrouter_byok_credential(
            env=env,
            provider=request.byok.provider,
            key_env=request.byok.key_env,
            management_key_env=request.byok.management_key_env,
            name=request.byok.credential_name,
            allowed_models=request.byok.allowed_models,
            is_fallback=request.byok.is_fallback,
            dry_run=request.byok.registration_dry_run,
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
            if byok_registration is not None:
                response_metadata = {
                    **dict(response_metadata),
                    "byok_registration": byok_registration.to_dict(),
                }
            break
        except Exception as exc:  # pragma: no cover - live provider failure path
            detail = _redact_provider_error_detail(str(exc))
            errors.append(
                {
                    "error_type": "provider_request_error",
                    "attempt": attempt,
                    "message": detail,
                }
            )
            if attempt >= request.retry_limit:
                message, remediation = _provider_request_error_details(detail)
                raise OpenRouterAdapterError(
                    message,
                    error_type="provider_request_error",
                    remediation=remediation,
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


def register_openrouter_byok_credential(
    *,
    env: Mapping[str, str] | None = None,
    provider: str | None = None,
    key_env: str | None = None,
    management_key_env: str | None = None,
    name: str | None = None,
    allowed_models: Sequence[str] | None = None,
    workspace_id: str | None = None,
    is_fallback: bool | None = None,
    dry_run: bool | None = None,
    timeout_seconds: int = 30,
) -> OpenRouterBYOKRegisterResult:
    """Create an OpenRouter BYOK provider credential without returning raw keys."""

    env = os.environ if env is None else env
    provider = (
        provider or env.get("CODELEWM_OPENROUTER_BYOK_PROVIDER") or DEFAULT_OPENROUTER_BYOK_PROVIDER
    ).strip()
    key_env = (
        key_env or env.get("CODELEWM_OPENROUTER_BYOK_KEY_ENV") or DEFAULT_OPENROUTER_BYOK_KEY_ENV
    ).strip()
    management_key_env = (
        management_key_env
        or env.get("CODELEWM_OPENROUTER_BYOK_MANAGEMENT_KEY_ENV")
        or DEFAULT_OPENROUTER_MANAGEMENT_KEY_ENV
    ).strip()
    name = _empty_to_none(name) or _empty_to_none(env.get("CODELEWM_OPENROUTER_BYOK_NAME"))
    workspace_id = _empty_to_none(workspace_id) or _empty_to_none(env.get("CODELEWM_OPENROUTER_BYOK_WORKSPACE_ID"))
    if allowed_models is None:
        allowed_models = _parse_csv_env(env.get("CODELEWM_OPENROUTER_BYOK_ALLOWED_MODELS"))
    allowed_models_tuple = tuple(str(model).strip() for model in allowed_models if str(model).strip())
    if is_fallback is None:
        is_fallback = _parse_bool_env(env, "CODELEWM_OPENROUTER_BYOK_IS_FALLBACK", default=False)
    if dry_run is None:
        dry_run = _parse_bool_env(env, "CODELEWM_OPENROUTER_BYOK_DRY_RUN", default=False)

    if provider != DEFAULT_OPENROUTER_BYOK_PROVIDER:
        raise OpenRouterAdapterError(
            "only Anthropic BYOK registration is supported in this adapter",
            error_type="config_error",
            remediation="set CODELEWM_OPENROUTER_BYOK_PROVIDER=anthropic",
        )
    if not key_env:
        raise OpenRouterAdapterError("BYOK key env name must not be empty", error_type="config_error")
    if not management_key_env:
        raise OpenRouterAdapterError(
            "OpenRouter management key env name must not be empty",
            error_type="config_error",
        )

    result = OpenRouterBYOKRegisterResult(
        provider=provider,
        credential_name=name,
        key_env=key_env,
        management_key_env=management_key_env,
        allowed_models=allowed_models_tuple,
        is_fallback=bool(is_fallback),
        workspace_id_set=workspace_id is not None,
        dry_run=bool(dry_run),
        registered=False,
    )
    if dry_run:
        return result

    management_key = env.get(management_key_env)
    provider_key = env.get(key_env)
    if not management_key:
        raise OpenRouterAdapterError(
            f"{management_key_env} is required to create an OpenRouter BYOK credential",
            error_type="missing_openrouter_management_key",
            remediation=(
                f"set {management_key_env}, or set CODELEWM_OPENROUTER_BYOK_REGISTER=0 "
                "if the BYOK credential already exists"
            ),
        )
    if not provider_key:
        raise OpenRouterAdapterError(
            f"{key_env} is required to create an Anthropic BYOK credential",
            error_type="missing_provider_api_key",
            remediation=f"set {key_env} or run with --dry-run",
        )

    body: dict[str, Any] = {
        "provider": provider,
        "key": provider_key,
        "name": name or "CodeLeWM Anthropic BYOK",
        "is_fallback": bool(is_fallback),
    }
    if allowed_models_tuple:
        body["allowed_models"] = list(allowed_models_tuple)
    if workspace_id is not None:
        body["workspace_id"] = workspace_id
    request = urllib.request.Request(
        OPENROUTER_BYOK_API_URL,
        data=json.dumps(body, allow_nan=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {management_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _read_http_error_detail(exc)
        raise OpenRouterAdapterError(
            f"OpenRouter BYOK registration failed with HTTP {exc.code}: {detail}",
            error_type="byok_registration_error",
            remediation="verify the OpenRouter management key, Anthropic key, provider, and workspace settings",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenRouterAdapterError(
            "OpenRouter BYOK registration failed",
            error_type="byok_registration_error",
            remediation="check network access and OpenRouter BYOK API availability",
        ) from exc

    data = payload.get("data") if isinstance(payload, Mapping) else None
    if isinstance(data, Mapping):
        response_metadata = dict(data)
    elif isinstance(payload, Mapping):
        response_metadata = dict(payload)
    else:
        response_metadata = {"response_type": type(payload).__name__}
    return OpenRouterBYOKRegisterResult(
        provider=provider,
        credential_name=name or "CodeLeWM Anthropic BYOK",
        key_env=key_env,
        management_key_env=management_key_env,
        allowed_models=allowed_models_tuple,
        is_fallback=bool(is_fallback),
        workspace_id_set=workspace_id is not None,
        dry_run=False,
        registered=True,
        response_metadata=redact_value(response_metadata),
    )


def capture_candidate_pack(
    pack: LLMCandidatePack,
    *,
    max_patch_chars: int = MAX_CAPTURE_PATCH_CHARS,
) -> LLMCandidatePack:
    """Parse and dry-run candidate patches as text without executing candidate code."""

    if max_patch_chars < 1:
        raise OpenRouterAdapterError("max_patch_chars must be >= 1", error_type="schema_error")
    captured = tuple(
        _capture_candidate(candidate, pack.request.context_bundle, max_patch_chars=max_patch_chars)
        for candidate in pack.candidates
    )
    return replace(pack, candidates=captured)


def write_candidate_pack_artifact(
    pack: LLMCandidatePack,
    out: Path | str,
    *,
    parent_artifacts: Sequence[str] = (),
    command: Sequence[str] = ("codelewm", "harness", "candidate-pack"),
    overwrite: bool = False,
    allow_secret_findings: bool = False,
    max_patch_chars: int = MAX_CAPTURE_PATCH_CHARS,
) -> CandidatePackArtifactResult:
    """Write a manifest-backed candidate-pack artifact after secret and patch checks."""

    secret_findings = _raw_candidate_pack_secret_findings(pack)
    if secret_findings and not allow_secret_findings:
        raise OpenRouterAdapterError(
            "candidate pack contains secret-scan findings; refusing to write publishable artifact",
            error_type="secret_scan_failed",
            remediation="remove secrets from prompt/context/candidate output before publication",
        )

    output_dir = Path(out).resolve()
    candidate_pack_path = output_dir / "candidate_pack.json"
    prompt_path = output_dir / "prompt" / "redacted_prompt.txt"
    candidates_dir = output_dir / "candidates"
    manifest_path = output_dir / "manifest.json"
    if not overwrite and any(path.exists() for path in (candidate_pack_path, prompt_path, manifest_path, candidates_dir)):
        raise OpenRouterAdapterError(
            f"output already exists; pass overwrite=True to replace: {output_dir}",
            error_type="config_error",
            remediation="choose a clean output directory or enable overwrite",
        )

    captured = capture_candidate_pack(pack, max_patch_chars=max_patch_chars)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(_redacted_text_preview(captured.prompt_text) + "\n", encoding="utf-8")

    candidates_with_paths: list[LLMCandidate] = []
    candidate_files: list[Path] = []
    for candidate in captured.candidates:
        patch_path = candidates_dir / f"{candidate.candidate_id}.patch"
        patch_text = _unified_diff_payload(candidate.patch_text)
        if len(patch_text) > max_patch_chars or scan_text(patch_text, path=str(patch_path)):
            patch_path = candidates_dir / f"{candidate.candidate_id}.patch.redacted.txt"
            patch_text = _redacted_text_preview(candidate.patch_text)
        patch_path.write_text(patch_text.rstrip("\n") + "\n", encoding="utf-8")
        candidate_files.append(patch_path)
        candidates_with_paths.append(
            replace(candidate, patch_path=_relative_to_root(patch_path, output_dir))
        )

    materialized = replace(
        captured,
        candidates=tuple(candidates_with_paths),
        artifact_manifest="manifest.json",
    )
    candidate_pack_path.write_text(
        json.dumps(materialized.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report_scan = scan_text(candidate_pack_path.read_text(encoding="utf-8"), path="candidate_pack.json")
    if report_scan and not allow_secret_findings:
        raise OpenRouterAdapterError(
            "candidate pack report contains secret-scan findings after redaction",
            error_type="secret_scan_failed",
            remediation="inspect redaction patterns before publication",
        )

    artifact_manifest = build_artifact_manifest(
        artifact_kind="candidate_pack",
        root=output_dir,
        files=(candidate_pack_path, prompt_path, *candidate_files),
        command=command,
        config={
            "task_id": materialized.request.task_id,
            "model": materialized.request.model,
            "dry_run": materialized.request.dry_run,
            "max_candidates": materialized.request.max_candidates,
            "max_patch_chars": max_patch_chars,
            "allow_secret_findings": allow_secret_findings,
        },
        parent_artifacts=tuple(parent_artifacts),
        metadata={
            "schema_version": LLM_CANDIDATE_PACK_SCHEMA_VERSION,
            "candidate_count": len(materialized.candidates),
            "valid_candidate_count": sum(
                1 for candidate in materialized.candidates if not candidate.errors
            ),
            "error_candidate_count": sum(
                1 for candidate in materialized.candidates if candidate.errors
            ),
            "secret_findings_count": len(secret_findings),
        },
    )
    write_artifact_manifest(artifact_manifest, manifest_path)
    return CandidatePackArtifactResult(
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path="manifest.json",
        candidate_pack_path="candidate_pack.json",
        prompt_path=_relative_to_root(prompt_path, output_dir),
        parent_artifacts=tuple(parent_artifacts),
    )


def render_candidate_prompt(request: OpenRouterCandidateRequest) -> str:
    """Render the deterministic prompt sent to OpenRouter."""

    context_lines = []
    for path, content in sorted(request.context_bundle.items()):
        content_block = content if content.endswith("\n") else f"{content}\n"
        context_lines.append(f"### File: {path}\n```python\n{content_block}```")
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
    first_content = request.context_bundle.get(first_path, "")
    if (
        "normalize_label" in first_content
        and "return value.strip().lower().replace" in first_content
    ):
        return _bugfix_edge_case_dry_run_patch(first_path, index=index)
    marker = _sha256_text(f"{request.task_id}:{request.instruction}:{index}")[:12]
    return (
        f"### Candidate candidate_{index:03d}\n"
        f"--- a/{first_path}\n"
        f"+++ b/{first_path}\n"
        "@@ -1,0 +1,1 @@\n"
        f"+# CodeLeWM dry-run candidate {index}: {marker}\n"
    )


def _bugfix_edge_case_dry_run_patch(path: str, *, index: int) -> str:
    replacements = (
        (
            "    normalized = \"-\".join(value.strip().lower().split())\n"
            "    if not normalized:\n"
            "        return \"untitled\"\n"
            "    return normalized\n"
        ),
        (
            "    parts = value.strip().lower().split()\n"
            "    if not parts:\n"
            "        return \"untitled\"\n"
            "    return \"-\".join(parts)\n"
        ),
        (
            "    text = \"-\".join(value.strip().lower().split())\n"
            "    return text or \"untitled\"\n"
        ),
        (
            "    words = value.strip().lower().split()\n"
            "    return \"-\".join(words) if words else \"untitled\"\n"
        ),
    )
    replacement = replacements[(index - 1) % len(replacements)]
    added_lines = "".join(f"+{line}\n" for line in replacement.splitlines())
    return (
        f"### Candidate candidate_{index:03d}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1,2 +1,"
        f"{1 + len(replacement.splitlines())} @@\n"
        " def normalize_label(value: str) -> str:\n"
        "-    return value.strip().lower().replace(\" \", \"-\")\n"
        f"{added_lines}"
    )


def _capture_candidate(
    candidate: LLMCandidate,
    context_bundle: Mapping[str, str],
    *,
    max_patch_chars: int,
) -> LLMCandidate:
    errors: list[Mapping[str, Any]] = list(candidate.errors)
    patch_bytes = len(candidate.patch_text.encode("utf-8"))
    secret_findings = scan_text(candidate.patch_text, path=f"{candidate.candidate_id}.patch")
    if secret_findings:
        errors.append(
            _candidate_error(
                candidate.candidate_id,
                "secret_scan_failed",
                "candidate patch contains secret-scan findings",
                "remove secrets from the candidate patch before publication",
            )
        )

    if len(candidate.patch_text) > max_patch_chars:
        errors.append(
            _candidate_error(
                candidate.candidate_id,
                "patch_too_large",
                "candidate patch exceeds the configured capture limit",
                "reduce candidate output size or raise max_patch_chars for local diagnostics",
            )
        )
        return replace(
            candidate,
            parser_status="not_parsed",
            dry_run_patch_status="blocked_patch_too_large",
            patch_bytes=patch_bytes,
            errors=tuple(errors),
        )

    before_path: str | None = None
    after_state_sha256: str | None = None
    parser_status = "not_parsed"
    dry_run_patch_status = "not_applied"
    try:
        patch_payload = _unified_diff_payload(candidate.patch_text)
        before_path = _patch_context_path(patch_payload, context_bundle)
        after_text = _apply_unified_diff(
            context_bundle[before_path],
            patch_payload,
            artifact=candidate.candidate_id,
        )
        dry_run_patch_status = "applied"
        after_state_sha256 = _sha256_text(after_text)
        try:
            parse_python_source_text(after_text, filename=before_path)
            parser_status = "parseable_python_after_state"
        except SyntaxError as exc:
            parser_status = "invalid_syntax"
            errors.append(
                _candidate_error(
                    candidate.candidate_id,
                    "invalid_syntax",
                    "candidate patch applies but produces invalid Python syntax",
                    "generate a syntactically valid Python after-state",
                    artifact=before_path,
                    caused_by=f"{exc.__class__.__name__}: {exc.msg}",
                )
            )
    except ScoreError as exc:
        dry_run_patch_status = exc.error_type
        errors.append(
            _candidate_error(
                candidate.candidate_id,
                exc.error_type,
                str(exc),
                exc.remediation,
                artifact=exc.artifact,
                caused_by=exc.caused_by,
            )
        )
    except OpenRouterAdapterError as exc:
        dry_run_patch_status = exc.error_type
        errors.append(
            _candidate_error(
                candidate.candidate_id,
                exc.error_type,
                str(exc),
                exc.remediation,
            )
        )

    return replace(
        candidate,
        parser_status=parser_status,
        dry_run_patch_status=dry_run_patch_status,
        applied_before_path=before_path,
        after_state_sha256=after_state_sha256,
        patch_bytes=patch_bytes,
        errors=tuple(errors),
    )


def _patch_context_path(patch_text: str, context_bundle: Mapping[str, str]) -> str:
    source_path, target_path = _patch_header_paths(_unified_diff_payload(patch_text))
    for candidate_path in (target_path, source_path):
        if candidate_path in context_bundle:
            return candidate_path
    available = ", ".join(sorted(context_bundle)) or "<empty>"
    raise OpenRouterAdapterError(
        "candidate patch does not target a file in the supplied context bundle",
        error_type="source_unavailable",
        remediation=f"target one of the context bundle paths: {available}",
    )


def _patch_header_paths(patch_text: str) -> tuple[str, str]:
    old_paths: list[str] = []
    new_paths: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            old_paths.append(_normalize_patch_header_path(line[4:].strip()))
        elif line.startswith("+++ "):
            new_paths.append(_normalize_patch_header_path(line[4:].strip()))

    if not old_paths or not new_paths:
        raise OpenRouterAdapterError(
            "candidate patch must include unified-diff source and target headers",
            error_type="patch_parse_failed",
            remediation="generate a unified diff with --- and +++ headers",
        )
    if len(old_paths) != 1 or len(new_paths) != 1:
        raise OpenRouterAdapterError(
            "candidate pack capture supports exactly one changed file per candidate",
            error_type="multi_file_patch_unsupported",
            remediation="split multi-file changes into one candidate per file for the v0 harness",
        )
    return old_paths[0], new_paths[0]


def _normalize_patch_header_path(raw: str) -> str:
    token = raw.split("\t", 1)[0].split(" ", 1)[0].strip()
    if token == "/dev/null":
        raise OpenRouterAdapterError(
            "candidate patch creates or deletes files, which v0 capture does not support",
            error_type="patch_parse_failed",
            remediation="generate a patch against an existing context file",
        )
    if token.startswith("a/") or token.startswith("b/"):
        token = token[2:]
    path = PurePosixPath(token)
    lowered = token.lower()
    if path.is_absolute() or ".." in path.parts or not token:
        raise OpenRouterAdapterError(
            "candidate patch path must be a safe relative repository path",
            error_type="security_error",
            remediation="remove absolute paths and path traversal from the patch",
        )
    if ".env" in lowered or "token" in lowered:
        raise OpenRouterAdapterError(
            "candidate patch path must not target token-bearing or .env files",
            error_type="security_error",
            remediation="remove secret-bearing files from candidate generation",
        )
    return path.as_posix()


def _candidate_error(
    candidate_id: str,
    error_type: str,
    message: str,
    remediation: str,
    *,
    artifact: str | None = None,
    caused_by: str | None = None,
) -> Mapping[str, Any]:
    return {
        "schema_version": "codelewm.error.v1",
        "candidate_id": candidate_id,
        "error_type": error_type,
        "message": redact_text(message),
        "remediation": remediation,
        "artifact": artifact,
        "caused_by": caused_by,
    }


def _provider_request_error_details(detail: str) -> tuple[str, str]:
    message = "OpenRouter provider request failed"
    if detail:
        message = f"{message}: {detail}"
    lowered = detail.lower()
    if "zero data retention" in lowered or "data policy" in lowered or "zdr" in lowered:
        return (
            message,
            (
                "remove provider.zdr from CODELEWM_LLM_PROVIDER_OPTIONS_JSON or configure "
                "OpenRouter privacy/ZDR settings for a matching endpoint"
            ),
        )
    return message, "check model slug, provider options, quota, and network"


def _redact_provider_error_detail(detail: str) -> str:
    return OPENROUTER_SECRET_RE.sub("[REDACTED_OPENROUTER_KEY]", redact_text(detail)).strip()


def _unified_diff_payload(text: str) -> str:
    """Extract the unified diff body from common markdown-fenced LLM output."""

    stripped = text.strip()
    lines = stripped.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith("--- ")), None)
    if start is None:
        return stripped
    payload: list[str] = []
    for line in lines[start:]:
        if line.strip().startswith("```"):
            break
        payload.append(line)
    return "\n".join(payload).strip("\n") + ("\n" if payload else "")


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


def _with_byok_provider_options(
    provider_options: Mapping[str, Any],
    byok: OpenRouterBYOKConfig,
) -> Mapping[str, Any]:
    merged = dict(provider_options)
    if not byok.enabled:
        return merged
    provider = byok.provider
    merged["order"] = _dedupe_strings((provider, *_string_sequence(merged.get("order"))))
    if byok.require:
        merged["only"] = _dedupe_strings((provider, *_string_sequence(merged.get("only"))))
        merged["allow_fallbacks"] = False
    _validate_provider_options(merged)
    return merged


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _parse_csv_env(value: str | None) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


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


def _read_http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except OSError:
        return "response body unavailable"
    if not raw:
        return "empty response body"
    return redact_text(raw[:1000])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _raw_candidate_pack_secret_findings(pack: LLMCandidatePack) -> tuple[Any, ...]:
    findings = list(scan_text(pack.prompt_text, path="prompt.txt"))
    for candidate in pack.candidates:
        findings.extend(scan_text(candidate.patch_text, path=f"{candidate.candidate_id}.patch"))
    return tuple(findings)


def _redacted_text_preview(text: str, *, limit: int = MAX_SERIALIZED_TEXT_CHARS) -> str:
    redacted = redact_text(text)
    if len(redacted) <= limit:
        return redacted
    digest = _sha256_text(redacted)
    return redacted[:limit] + f"\n...[truncated sha256={digest} chars={len(redacted)}]"


def _relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


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
