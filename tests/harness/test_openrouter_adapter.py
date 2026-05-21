from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib

from codelewm.harness import (
    DEFAULT_OPENROUTER_MODEL,
    LLM_CANDIDATE_PACK_SCHEMA_VERSION,
    OPENROUTER_CANDIDATE_REQUEST_SCHEMA_VERSION,
    OPENROUTER_SDK_VERSION_PIN,
    OpenRouterAdapterError,
    OpenRouterCandidateRequest,
    generate_candidate_pack,
)


ROOT = Path(__file__).resolve().parents[2]


class OpenRouterAdapterTest(unittest.TestCase):
    def test_dry_run_candidate_pack_is_deterministic_without_api_key(self) -> None:
        request = OpenRouterCandidateRequest(
            task_id="task-1",
            instruction="make add return the sum explicitly",
            context_bundle={"src/math.py": "def add(a, b):\n    return a + b\n"},
            max_candidates=2,
            provider_options={"sort": "price", "zdr": True},
            dry_run=True,
        )

        first = generate_candidate_pack(request, env={}).to_dict()
        second = generate_candidate_pack(request, env={"ANTHROPIC_API_KEY": "sk-ant-secret"}).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], LLM_CANDIDATE_PACK_SCHEMA_VERSION)
        self.assertEqual(first["task_id"], "task-1")
        self.assertEqual(first["generator"]["provider"], "openrouter")
        self.assertEqual(first["generator"]["model"], DEFAULT_OPENROUTER_MODEL)
        self.assertIsNone(first["generator"]["sdk_version"])
        self.assertEqual(len(first["candidates"]), 2)
        self.assertEqual(first["candidates"][0]["candidate_id"], "candidate_001")
        self.assertIn("CodeLeWM dry-run candidate", first["candidates"][0]["patch_text"])
        self.assertNotIn("sk-ant-secret", json.dumps(first, sort_keys=True))

    def test_request_from_env_reads_only_documented_openrouter_settings(self) -> None:
        request = OpenRouterCandidateRequest.from_env(
            task_id="task-2",
            instruction="update parser",
            context_bundle={"parser.py": "value = 1\n"},
            env={
                "CODELEWM_LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "openrouter_xxx",
                "ANTHROPIC_API_KEY": "sk-ant-secret",
                "CODELEWM_LLM_MODEL": "anthropic/claude-4.5-sonnet",
                "CODELEWM_LLM_MAX_CANDIDATES": "3",
                "CODELEWM_LLM_TIMEOUT_SECONDS": "45",
                "CODELEWM_LLM_TEMPERATURE": "0.4",
                "CODELEWM_LLM_DRY_RUN": "1",
                "CODELEWM_LLM_RETRY_LIMIT": "1",
                "CODELEWM_LLM_PROVIDER_OPTIONS_JSON": '{"sort":"price","zdr":true}',
                "OPENROUTER_HTTP_REFERER": "https://example.test",
                "OPENROUTER_APP_TITLE": "CodeLeWM",
            },
        )

        payload = request.to_dict()
        self.assertEqual(payload["schema_version"], OPENROUTER_CANDIDATE_REQUEST_SCHEMA_VERSION)
        self.assertEqual(request.model, "anthropic/claude-4.5-sonnet")
        self.assertEqual(request.max_candidates, 3)
        self.assertEqual(request.timeout_seconds, 45)
        self.assertEqual(request.temperature, 0.4)
        self.assertEqual(request.retry_limit, 1)
        self.assertEqual(request.provider_options, {"sort": "price", "zdr": True})
        self.assertNotIn("ANTHROPIC_API_KEY", json.dumps(payload, sort_keys=True))
        self.assertNotIn("sk-ant-secret", json.dumps(payload, sort_keys=True))

    def test_live_mode_requires_openrouter_api_key_and_ignores_direct_provider_key(self) -> None:
        request = OpenRouterCandidateRequest(
            task_id="task-3",
            instruction="change behavior",
            context_bundle={"app.py": "value = 1\n"},
            dry_run=False,
        )

        with self.assertRaises(OpenRouterAdapterError) as raised:
            generate_candidate_pack(request, env={"ANTHROPIC_API_KEY": "sk-ant-secret"})

        report = raised.exception.to_error_report()
        self.assertEqual(report["error_type"], "missing_openrouter_api_key")
        self.assertNotIn("sk-ant-secret", json.dumps(report, sort_keys=True))
        self.assertNotIn("ANTHROPIC_API_KEY", json.dumps(report, sort_keys=True))

    def test_live_mode_passes_generation_controls_to_openrouter_sdk(self) -> None:
        calls: dict[str, object] = {}

        class FakeChat:
            def send(self, **kwargs: object) -> dict[str, object]:
                calls["send"] = kwargs
                return {
                    "id": "chatcmpl-test",
                    "model": kwargs["model"],
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "### Candidate live\n"
                                    "--- a/app.py\n"
                                    "+++ b/app.py\n"
                                    "@@\n"
                                    "+value = 2\n"
                                )
                            }
                        }
                    ],
                }

        class FakeOpenRouter:
            def __init__(self, **kwargs: object) -> None:
                calls["client"] = kwargs
                self.chat = FakeChat()

            def __enter__(self) -> "FakeOpenRouter":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        fake_module = types.ModuleType("openrouter")
        fake_module.OpenRouter = FakeOpenRouter  # type: ignore[attr-defined]
        request = OpenRouterCandidateRequest(
            task_id="task-live",
            instruction="change behavior",
            context_bundle={"app.py": "value = 1\n"},
            model="anthropic/claude-4.5-sonnet",
            max_candidates=1,
            timeout_seconds=17,
            temperature=0.7,
            provider_options={"sort": "price", "allow_fallbacks": False},
            dry_run=False,
            retry_limit=0,
        )

        with mock.patch.dict(sys.modules, {"openrouter": fake_module}):
            pack = generate_candidate_pack(
                request,
                env={
                    "OPENROUTER_API_KEY": "openrouter_live_secret_value",
                    "ANTHROPIC_API_KEY": "sk-ant-secret",
                },
            ).to_dict()

        self.assertEqual(len(pack["candidates"]), 1)
        self.assertEqual(pack["candidates"][0]["candidate_id"], "candidate_001")
        self.assertEqual(calls["client"]["api_key"], "openrouter_live_secret_value")
        self.assertEqual(calls["client"]["timeout_ms"], 17000)
        self.assertEqual(calls["send"]["model"], "anthropic/claude-4.5-sonnet")
        self.assertEqual(calls["send"]["provider"], {"sort": "price", "allow_fallbacks": False})
        self.assertEqual(calls["send"]["temperature"], 0.7)
        self.assertEqual(calls["send"]["timeout_ms"], 17000)
        self.assertFalse(calls["send"]["stream"])
        serialized = json.dumps(pack, sort_keys=True)
        self.assertNotIn("openrouter_live_secret_value", serialized)
        self.assertNotIn("sk-ant-secret", serialized)

    def test_provider_options_reject_unknown_keys(self) -> None:
        with self.assertRaises(OpenRouterAdapterError) as raised:
            OpenRouterCandidateRequest(
                task_id="task-4",
                instruction="change behavior",
                provider_options={"unsafe_key": True},
            )

        self.assertEqual(raised.exception.error_type, "schema_error")
        self.assertIn("unsupported provider option", str(raised.exception))

    def test_context_bundle_rejects_env_and_token_paths(self) -> None:
        with self.assertRaises(OpenRouterAdapterError) as raised:
            OpenRouterCandidateRequest(
                task_id="task-5",
                instruction="change behavior",
                context_bundle={".env": "OPENROUTER_API_KEY=openrouter_xxx\n"},
            )

        self.assertEqual(raised.exception.error_type, "security_error")

    def test_openrouter_optional_dependency_is_pinned_in_pyproject(self) -> None:
        payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        optional = payload["project"]["optional-dependencies"]["llm"]
        group = payload["dependency-groups"]["llm"]
        self.assertIn(f"openrouter=={OPENROUTER_SDK_VERSION_PIN}", optional)
        self.assertIn(f"openrouter=={OPENROUTER_SDK_VERSION_PIN}", group)


if __name__ == "__main__":
    unittest.main()
