from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib

from codelewm.harness import (
    DEFAULT_OPENROUTER_MANAGEMENT_KEY_ENV,
    DEFAULT_OPENROUTER_MODEL,
    LLM_CANDIDATE_PACK_SCHEMA_VERSION,
    OPENROUTER_BYOK_REGISTER_SCHEMA_VERSION,
    OPENROUTER_CANDIDATE_REQUEST_SCHEMA_VERSION,
    OPENROUTER_SDK_VERSION_PIN,
    OpenRouterAdapterError,
    OpenRouterCandidateRequest,
    capture_candidate_pack,
    generate_candidate_pack,
    get_demo_scenario,
    register_openrouter_byok_credential,
    render_candidate_prompt,
    write_candidate_pack_artifact,
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
        second = generate_candidate_pack(request, env={"ANTHROPIC_API_KEY": "anthropic-secret-value"}).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], LLM_CANDIDATE_PACK_SCHEMA_VERSION)
        self.assertEqual(first["task_id"], "task-1")
        self.assertEqual(first["generator"]["provider"], "openrouter")
        self.assertEqual(first["generator"]["model"], DEFAULT_OPENROUTER_MODEL)
        self.assertIsNone(first["generator"]["sdk_version"])
        self.assertEqual(len(first["candidates"]), 2)
        self.assertEqual(first["candidates"][0]["candidate_id"], "candidate_001")
        self.assertIn("CodeLeWM dry-run candidate", first["candidates"][0]["patch_text"])
        self.assertNotIn("anthropic-secret-value", json.dumps(first, sort_keys=True))

    def test_dry_run_default_demo_scenario_generates_behavior_patch(self) -> None:
        scenario = get_demo_scenario()
        request = OpenRouterCandidateRequest(
            task_id=scenario.task_id,
            instruction=scenario.instruction,
            context_bundle={scenario.primary_file.path: scenario.primary_file.content},
            prompt_template_id=scenario.prompt_template_id,
            max_candidates=2,
            dry_run=True,
        )

        pack = generate_candidate_pack(request, env={})
        captured = capture_candidate_pack(pack)
        payload = captured.to_dict()

        self.assertEqual(len(payload["candidates"]), 2)
        self.assertEqual(payload["prompt"]["template_id"], scenario.prompt_template_id)
        self.assertIn("untitled", payload["candidates"][0]["patch_text"])
        self.assertIn("split", payload["candidates"][0]["patch_text"])
        self.assertNotIn("CodeLeWM dry-run candidate", payload["candidates"][0]["patch_text"])
        self.assertEqual(payload["candidates"][0]["parser_status"], "parseable_python_after_state")
        self.assertEqual(payload["candidates"][0]["dry_run_patch_status"], "applied")

    def test_request_from_env_reads_only_documented_openrouter_settings(self) -> None:
        request = OpenRouterCandidateRequest.from_env(
            task_id="task-2",
            instruction="update parser",
            context_bundle={"parser.py": "value = 1\n"},
            env={
                "CODELEWM_LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "openrouter_xxx",
                "ANTHROPIC_API_KEY": "anthropic-secret-value",
                "CODELEWM_LLM_MODEL": "anthropic/claude-4.5-sonnet",
                "CODELEWM_LLM_MAX_CANDIDATES": "3",
                "CODELEWM_LLM_TIMEOUT_SECONDS": "45",
                "CODELEWM_LLM_TEMPERATURE": "0.4",
                "CODELEWM_LLM_DRY_RUN": "1",
                "CODELEWM_LLM_RETRY_LIMIT": "1",
                "CODELEWM_LLM_PROMPT_TEMPLATE_ID": "codelewm.test.template.v1",
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
        self.assertEqual(request.prompt_template_id, "codelewm.test.template.v1")
        self.assertEqual(request.provider_options, {"sort": "price", "zdr": True})
        self.assertNotIn("ANTHROPIC_API_KEY", json.dumps(payload, sort_keys=True))
        self.assertNotIn("anthropic-secret-value", json.dumps(payload, sort_keys=True))

    def test_prompt_rendering_does_not_add_phantom_context_blank_line(self) -> None:
        request = OpenRouterCandidateRequest(
            task_id="task-prompt",
            instruction="add a comment",
            context_bundle={"app.py": "value = 1\n"},
        )

        prompt = render_candidate_prompt(request)

        self.assertIn("```python\nvalue = 1\n```", prompt)
        self.assertNotIn("value = 1\n\n```", prompt)

    def test_byok_env_adds_anthropic_only_routing_without_serializing_keys(self) -> None:
        request = OpenRouterCandidateRequest.from_env(
            task_id="task-byok",
            instruction="update parser",
            context_bundle={"parser.py": "value = 1\n"},
            env={
                "CODELEWM_LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "openrouter_xxx",
                "ANTHROPIC_API_KEY": "anthropic-secret-value",
                "CODELEWM_LLM_MODEL": "anthropic/claude-4.5-sonnet",
                "CODELEWM_LLM_DRY_RUN": "1",
                "CODELEWM_OPENROUTER_BYOK": "1",
                "CODELEWM_OPENROUTER_BYOK_PROVIDER": "anthropic",
                "CODELEWM_OPENROUTER_BYOK_KEY_ENV": "ANTHROPIC_API_KEY",
                "CODELEWM_OPENROUTER_BYOK_MANAGEMENT_KEY_ENV": "OPENROUTER_MANAGEMENT_KEY",
                "CODELEWM_OPENROUTER_BYOK_REQUIRE": "1",
                "CODELEWM_OPENROUTER_BYOK_REGISTER": "1",
                "CODELEWM_OPENROUTER_BYOK_DRY_RUN": "1",
            },
        )

        payload = request.to_dict()

        self.assertEqual(request.provider_options["order"], ["anthropic"])
        self.assertEqual(request.provider_options["only"], ["anthropic"])
        self.assertFalse(request.provider_options["allow_fallbacks"])
        self.assertTrue(payload["byok"]["enabled"])
        self.assertTrue(payload["byok"]["register"])
        self.assertTrue(payload["byok"]["registration_dry_run"])
        self.assertEqual(payload["byok"]["management_key_env"], "OPENROUTER_MANAGEMENT_KEY")
        self.assertEqual(payload["byok"]["allowed_models"], ["anthropic/claude-4.5-sonnet"])
        self.assertNotIn("anthropic-secret-value", json.dumps(payload, sort_keys=True))

    def test_byok_registration_dry_run_does_not_require_secrets(self) -> None:
        result = register_openrouter_byok_credential(
            env={},
            provider="anthropic",
            key_env="ANTHROPIC_API_KEY",
            management_key_env=DEFAULT_OPENROUTER_MANAGEMENT_KEY_ENV,
            name="CodeLeWM Anthropic BYOK",
            allowed_models=("anthropic/claude-4.5-sonnet",),
            dry_run=True,
        ).to_dict()

        self.assertEqual(result["schema_version"], OPENROUTER_BYOK_REGISTER_SCHEMA_VERSION)
        self.assertEqual(result["provider"], "anthropic")
        self.assertEqual(result["key_env"], "ANTHROPIC_API_KEY")
        self.assertEqual(result["management_key_env"], DEFAULT_OPENROUTER_MANAGEMENT_KEY_ENV)
        self.assertEqual(result["allowed_models"], ["anthropic/claude-4.5-sonnet"])
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["registered"])

    def test_byok_registration_posts_redacted_payload_summary(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "data": {
                            "id": "11111111-2222-3333-4444-555555555555",
                            "provider": "anthropic",
                            "label": "sk-...test",
                            "name": "CodeLeWM Anthropic BYOK",
                        }
                    }
                ).encode("utf-8")

        captured: dict[str, object] = {}

        def fake_urlopen(request: object, timeout: int) -> FakeResponse:
            captured["timeout"] = timeout
            captured["request"] = request
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = register_openrouter_byok_credential(
                env={
                    "OPENROUTER_API_KEY": "openrouter_live_secret_value",
                    "OPENROUTER_MANAGEMENT_KEY": "openrouter_management_secret_value",
                    "ANTHROPIC_API_KEY": "anthropic-secret-value",
                },
                provider="anthropic",
                key_env="ANTHROPIC_API_KEY",
                management_key_env="OPENROUTER_MANAGEMENT_KEY",
                name="CodeLeWM Anthropic BYOK",
                allowed_models=("anthropic/claude-4.5-sonnet",),
            ).to_dict()

        self.assertTrue(result["registered"])
        self.assertFalse(result["dry_run"])
        self.assertEqual(captured["timeout"], 30)
        request = captured["request"]
        body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        self.assertEqual(body["provider"], "anthropic")
        self.assertEqual(body["key"], "anthropic-secret-value")
        self.assertEqual(body["allowed_models"], ["anthropic/claude-4.5-sonnet"])
        self.assertEqual(request.headers["Authorization"], "Bearer openrouter_management_secret_value")
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("openrouter_live_secret_value", serialized)
        self.assertNotIn("openrouter_management_secret_value", serialized)
        self.assertNotIn("anthropic-secret-value", serialized)

    def test_byok_registration_requires_management_key_not_chat_key(self) -> None:
        with self.assertRaises(OpenRouterAdapterError) as raised:
            register_openrouter_byok_credential(
                env={
                    "OPENROUTER_API_KEY": "openrouter_live_secret_value",
                    "ANTHROPIC_API_KEY": "anthropic-secret-value",
                },
                provider="anthropic",
                key_env="ANTHROPIC_API_KEY",
                dry_run=False,
            )

        report = raised.exception.to_error_report()
        self.assertEqual(report["error_type"], "missing_openrouter_management_key")
        self.assertIn("CODELEWM_OPENROUTER_BYOK_REGISTER=0", report["remediation"])
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("openrouter_live_secret_value", serialized)
        self.assertNotIn("anthropic-secret-value", serialized)

    def test_live_mode_requires_openrouter_api_key_and_ignores_direct_provider_key(self) -> None:
        request = OpenRouterCandidateRequest(
            task_id="task-3",
            instruction="change behavior",
            context_bundle={"app.py": "value = 1\n"},
            dry_run=False,
        )

        with self.assertRaises(OpenRouterAdapterError) as raised:
            generate_candidate_pack(request, env={"ANTHROPIC_API_KEY": "anthropic-secret-value"})

        report = raised.exception.to_error_report()
        self.assertEqual(report["error_type"], "missing_openrouter_api_key")
        self.assertNotIn("anthropic-secret-value", json.dumps(report, sort_keys=True))
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
                    "ANTHROPIC_API_KEY": "anthropic-secret-value",
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
        self.assertNotIn("anthropic-secret-value", serialized)

    def test_live_provider_error_includes_redacted_zdr_cause(self) -> None:
        class FakeChat:
            def send(self, **kwargs: object) -> object:
                raise RuntimeError(
                    "No endpoints found matching your data policy "
                    "(Zero data retention). key sk-or-v1-secret"
                )

        class FakeOpenRouter:
            def __init__(self, **kwargs: object) -> None:
                self.chat = FakeChat()

            def __enter__(self) -> "FakeOpenRouter":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        fake_module = types.ModuleType("openrouter")
        fake_module.OpenRouter = FakeOpenRouter  # type: ignore[attr-defined]
        request = OpenRouterCandidateRequest(
            task_id="task-live-zdr",
            instruction="change behavior",
            context_bundle={"app.py": "value = 1\n"},
            provider_options={"zdr": True},
            dry_run=False,
            retry_limit=0,
        )

        with mock.patch.dict(sys.modules, {"openrouter": fake_module}):
            with self.assertRaises(OpenRouterAdapterError) as raised:
                generate_candidate_pack(
                    request,
                    env={"OPENROUTER_API_KEY": "openrouter_live_secret_value"},
                )

        report = raised.exception.to_error_report()
        serialized = json.dumps(report, sort_keys=True)
        self.assertEqual(report["error_type"], "provider_request_error")
        self.assertIn("Zero data retention", report["message"])
        self.assertIn("provider.zdr", report["remediation"])
        self.assertNotIn("sk-or-v1-secret", serialized)
        self.assertNotIn("openrouter_live_secret_value", serialized)

    def test_capture_candidate_accepts_markdown_fenced_unified_diff(self) -> None:
        class FakeChat:
            def send(self, **kwargs: object) -> dict[str, object]:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "### Candidate 1\n"
                                    "```diff\n"
                                    "--- app.py\n"
                                    "+++ app.py\n"
                                    "@@ -1,1 +1,2 @@\n"
                                    "+# Initialize value\n"
                                    " value = 1\n"
                                    "```\n"
                                )
                            }
                        }
                    ],
                }

        class FakeOpenRouter:
            def __init__(self, **kwargs: object) -> None:
                self.chat = FakeChat()

            def __enter__(self) -> "FakeOpenRouter":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        fake_module = types.ModuleType("openrouter")
        fake_module.OpenRouter = FakeOpenRouter  # type: ignore[attr-defined]
        request = OpenRouterCandidateRequest(
            task_id="task-live-fenced",
            instruction="add a comment",
            context_bundle={"app.py": "value = 1\n"},
            max_candidates=1,
            dry_run=False,
            retry_limit=0,
        )

        with mock.patch.dict(sys.modules, {"openrouter": fake_module}):
            pack = generate_candidate_pack(
                request,
                env={"OPENROUTER_API_KEY": "openrouter_live_secret_value"},
            )
        captured = capture_candidate_pack(pack).to_dict()

        self.assertEqual(captured["candidates"][0]["dry_run_patch_status"], "applied")
        self.assertEqual(
            captured["candidates"][0]["parser_status"],
            "parseable_python_after_state",
        )
        self.assertEqual(captured["candidates"][0]["errors"], [])
        self.assertIn("```diff", captured["candidates"][0]["patch_text"])

        with tempfile.TemporaryDirectory() as tmp:
            result = write_candidate_pack_artifact(pack, Path(tmp), overwrite=True)
            patch_file = Path(tmp) / "candidates" / "candidate_001.patch"
            patch_text = patch_file.read_text(encoding="utf-8")
            candidate_pack = json.loads((Path(tmp) / result.candidate_pack_path).read_text())

        self.assertTrue(patch_text.startswith("--- app.py\n+++ app.py\n"))
        self.assertNotIn("```diff", patch_text)
        self.assertIn("```diff", candidate_pack["candidates"][0]["patch_text"])

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
