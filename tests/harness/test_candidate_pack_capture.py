from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.harness import (
    LLM_CANDIDATE_PACK_SCHEMA_VERSION,
    MAX_CAPTURE_PATCH_CHARS,
    LLMCandidate,
    LLMCandidatePack,
    OpenRouterAdapterError,
    OpenRouterCandidateRequest,
    capture_candidate_pack,
    generate_candidate_pack,
    write_candidate_pack_artifact,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


class CandidatePackCaptureTest(unittest.TestCase):
    def test_candidate_pack_artifact_is_manifested_and_checksum_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = _pack(
                _candidate(
                    "candidate_001",
                    "### Candidate valid\n"
                    "--- a/app.py\n"
                    "+++ b/app.py\n"
                    "@@ -1,1 +1,1 @@\n"
                    "-value = 1\n"
                    "+value = 2\n",
                )
            )

            result = write_candidate_pack_artifact(
                pack,
                root / "pack",
                command=("codelewm", "harness", "candidate-pack"),
            )
            manifest = read_artifact_manifest(root / "pack" / "manifest.json")
            checked = validate_artifact_checksums(manifest, root=root / "pack")
            payload = json.loads((root / "pack" / result.candidate_pack_path).read_text())

        self.assertEqual(payload["schema_version"], LLM_CANDIDATE_PACK_SCHEMA_VERSION)
        self.assertEqual(manifest.artifact_kind, "candidate_pack")
        self.assertEqual(result.artifact_manifest_id, manifest.artifact_id)
        self.assertEqual(
            {path.relative_to((root / "pack").resolve()).as_posix() for path in checked},
            {
                "candidate_pack.json",
                "prompt/redacted_prompt.txt",
                "candidates/candidate_001.patch",
            },
        )
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["parser_status"], "parseable_python_after_state")
        self.assertEqual(candidate["dry_run_patch_status"], "applied")
        self.assertEqual(candidate["applied_before_path"], "app.py")
        self.assertEqual(len(candidate["after_state_sha256"]), 64)
        self.assertEqual(candidate["patch_path"], "candidates/candidate_001.patch")
        self.assertTrue(candidate["rankability"]["rankable"])
        self.assertEqual(candidate["rankability"]["fallback_order"], "normal")

        recaptured = capture_candidate_pack(pack).to_dict()
        self.assertEqual(
            recaptured["candidates"][0]["normalized_patch_sha256"],
            candidate["normalized_patch_sha256"],
        )

    def test_malicious_candidate_text_is_parsed_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "executed.txt"
            pack = _pack(
                _candidate(
                    "candidate_001",
                    "### Candidate malicious\n"
                    "--- a/app.py\n"
                    "+++ b/app.py\n"
                    "@@ -1,1 +1,3 @@\n"
                    "-value = 1\n"
                    "+import os\n"
                    f"+os.system('touch {marker}')\n"
                    "+value = 2\n",
                )
            )

            captured = capture_candidate_pack(pack).to_dict()

        self.assertFalse(marker.exists())
        candidate = captured["candidates"][0]
        self.assertEqual(candidate["parser_status"], "parseable_python_after_state")
        self.assertEqual(candidate["dry_run_patch_status"], "applied")
        self.assertEqual(candidate["errors"], [])

    def test_invalid_syntax_becomes_candidate_error_and_remains_rankable(self) -> None:
        pack = _pack(
            _candidate(
                "candidate_001",
                "### Candidate valid\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1,1 +1,1 @@\n"
                "-value = 1\n"
                "+value = 2\n",
            ),
            _candidate(
                "candidate_002",
                "### Candidate invalid\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1,1 +1,1 @@\n"
                "-value = 1\n"
                "+value =\n",
            ),
        )

        captured = capture_candidate_pack(pack).to_dict()

        self.assertEqual(captured["candidates"][0]["parser_status"], "parseable_python_after_state")
        invalid = captured["candidates"][1]
        self.assertEqual(invalid["parser_status"], "invalid_syntax")
        self.assertEqual(invalid["dry_run_patch_status"], "applied")
        self.assertEqual(invalid["errors"][0]["error_type"], "invalid_syntax")
        self.assertTrue(invalid["rankability"]["rankable"])
        self.assertEqual(invalid["rankability"]["fallback_order"], "after_valid_candidates")

    def test_oversized_candidate_is_truncated_and_not_patch_applied(self) -> None:
        patch_text = "### Candidate huge\n" + ("+x\n" * (MAX_CAPTURE_PATCH_CHARS // 2))
        pack = _pack(_candidate("candidate_001", patch_text))

        captured = capture_candidate_pack(pack, max_patch_chars=64).to_dict()

        candidate = captured["candidates"][0]
        self.assertEqual(candidate["parser_status"], "not_parsed")
        self.assertEqual(candidate["dry_run_patch_status"], "blocked_patch_too_large")
        self.assertEqual(candidate["errors"][0]["error_type"], "patch_too_large")
        self.assertLess(len(candidate["patch_text"]), len(patch_text))
        self.assertIn("REDACTED_LONG_TEXT", candidate["patch_text"])

    def test_secret_scan_blocks_publishable_candidate_pack_artifact(self) -> None:
        secret_line = "+api_" + "key=" + "abcdefghijklmnop\n"
        pack = _pack(
            _candidate(
                "candidate_001",
                "### Candidate secret\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1,1 +1,1 @@\n"
                "-value = 1\n"
                + secret_line,
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OpenRouterAdapterError) as raised:
                write_candidate_pack_artifact(pack, Path(tmp) / "pack")

        self.assertEqual(raised.exception.error_type, "secret_scan_failed")

    def test_dry_run_openrouter_fixture_now_produces_captureable_patches(self) -> None:
        request = OpenRouterCandidateRequest(
            task_id="task-dry",
            instruction="add a marker",
            context_bundle={"app.py": "value = 1\n"},
            max_candidates=1,
            dry_run=True,
        )

        captured = capture_candidate_pack(generate_candidate_pack(request, env={})).to_dict()

        candidate = captured["candidates"][0]
        self.assertEqual(candidate["parser_status"], "parseable_python_after_state")
        self.assertEqual(candidate["dry_run_patch_status"], "applied")


def _pack(*candidates: LLMCandidate) -> LLMCandidatePack:
    request = OpenRouterCandidateRequest(
        task_id="task-1",
        instruction="update app",
        context_bundle={"app.py": "value = 1\n"},
        dry_run=True,
    )
    return LLMCandidatePack(
        request=request,
        prompt_text="Generate candidate patches",
        candidates=tuple(candidates),
        provider_metadata={"mode": "fixture"},
        sdk_version=None,
    )


def _candidate(candidate_id: str, patch_text: str) -> LLMCandidate:
    return LLMCandidate(
        candidate_id=candidate_id,
        patch_text=patch_text,
        parser_status="not_parsed",
        dry_run_patch_status="not_applied",
    )


if __name__ == "__main__":
    unittest.main()
