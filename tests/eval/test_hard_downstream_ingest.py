from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    LLM_CANDIDATE_PACK_SCHEMA_VERSION,
    LLMCandidateIngestError,
    build_downstream_benchmark_pack,
    ingest_llm_candidate_pack,
    read_downstream_rerank_benchmark,
    read_llm_candidate_pack,
)
from codelewm.harness import (
    OpenRouterCandidateRequest,
    generate_candidate_pack,
    write_candidate_pack_artifact,
)
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    write_artifact_manifest,
)


def _dry_run_pack(pack_dir: Path, *, task_id: str = "accumulator", max_candidates: int = 3) -> Path:
    request = OpenRouterCandidateRequest(
        task_id=task_id,
        instruction="rewrite the accumulator update explicitly",
        context_bundle={"app.py": "value = 1\n"},
        max_candidates=max_candidates,
        dry_run=True,
    )
    result = write_candidate_pack_artifact(
        generate_candidate_pack(request, env={}),
        pack_dir,
        command=("codelewm", "harness", "candidate-pack"),
    )
    return pack_dir / result.artifact_manifest_path


def _raw_pack(pack_dir: Path, candidates: list[dict]) -> Path:
    """Hand-author a candidate-pack artifact for blocker tests."""
    (pack_dir / "candidates").mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    candidate_json: list[dict] = []
    for candidate in candidates:
        entry = dict(candidate)
        patch_content = entry.pop("patch_content", None)
        patch_path = entry.get("patch_path")
        if patch_content is not None and isinstance(patch_path, str):
            patch_file = pack_dir / patch_path
            patch_file.parent.mkdir(parents=True, exist_ok=True)
            patch_file.write_text(patch_content, encoding="utf-8")
            files.append(patch_file)
        candidate_json.append(entry)
    payload = {
        "schema_version": LLM_CANDIDATE_PACK_SCHEMA_VERSION,
        "task_id": "accumulator",
        "generator": {
            "provider": "openrouter",
            "model": "anthropic/claude-4.5-sonnet",
            "sdk": "openrouter",
            "sdk_version": "0.9.1",
            "adapter_version": "codelewm.openrouter_adapter.v0.1",
        },
        "candidates": candidate_json,
    }
    pack_file = pack_dir / "candidate_pack.json"
    pack_file.write_text(json.dumps(payload), encoding="utf-8")
    files.append(pack_file)
    manifest = build_artifact_manifest(
        artifact_kind="candidate_pack",
        root=pack_dir,
        files=files,
        command=("test", "raw-pack"),
        config=payload,
    )
    write_artifact_manifest(manifest, pack_dir / "manifest.json")
    return pack_dir / "manifest.json"


_CLEAN_PATCH = (
    "--- a/app.py\n+++ b/app.py\n@@ -1,1 +1,1 @@\n-value = 1\n+value = 2\n"
)


def _clean_candidate(candidate_id: str = "candidate_001") -> dict:
    return {
        "candidate_id": candidate_id,
        "patch_path": f"candidates/{candidate_id}.patch",
        "patch_content": _CLEAN_PATCH,
        "content_sha256": "deadbeef",
        "normalized_patch_sha256": "cafef00d",
        "parser_status": "parseable_python_after_state",
        "dry_run_patch_status": "applied",
        "redaction": {"secret_scan_ok": True, "secret_findings_count": 0},
    }


class LLMCandidateIngestReadTest(unittest.TestCase):
    def test_read_verifies_manifest_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _dry_run_pack(Path(tmp) / "pack")
            payload, artifact_id, pack_dir = read_llm_candidate_pack(manifest)
        self.assertEqual(payload["schema_version"], LLM_CANDIDATE_PACK_SCHEMA_VERSION)
        self.assertTrue(artifact_id)
        self.assertEqual(pack_dir.name, "pack")

    def test_missing_manifest_raises(self) -> None:
        with self.assertRaises(LLMCandidateIngestError):
            read_llm_candidate_pack(Path("/nonexistent/manifest.json"))


class LLMCandidateIngestTest(unittest.TestCase):
    def test_successful_ingestion_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _dry_run_pack(Path(tmp) / "pack", max_candidates=2)
            result = ingest_llm_candidate_pack(manifest, base_llm_rank=0)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.blockers, ())
        self.assertTrue(result.source_manifest_id)
        candidate = result.candidates[0]
        self.assertEqual(candidate.label, "unknown")
        self.assertEqual(candidate.source["hard_negative_class"], "llm_generated")
        self.assertEqual(candidate.source["provider"], "openrouter")
        self.assertTrue(candidate.source["model_slug"])
        self.assertTrue(candidate.source["checksum"])
        self.assertEqual(candidate.provenance["source_manifest_id"], result.source_manifest_id)
        self.assertEqual(candidate.provenance["parser_status"], "parseable_python_after_state")
        self.assertTrue(candidate.patch_text.startswith("---"))

    def test_missing_manifest_becomes_typed_blocker(self) -> None:
        result = ingest_llm_candidate_pack(Path("/nope/manifest.json"))
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.blockers[0]["reason"], "pack_unreadable")

    def test_secret_positive_flag_in_pack_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _clean_candidate()
            candidate["redaction"] = {"secret_scan_ok": False, "secret_findings_count": 1}
            manifest = _raw_pack(Path(tmp) / "pack", [candidate])
            result = ingest_llm_candidate_pack(manifest)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.blockers[0]["reason"], "secret_positive_in_pack")

    def test_secret_on_rescan_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _clean_candidate()
            # JSON claims clean, but the on-disk patch carries a secret.
            candidate["patch_content"] = (
                "--- a/app.py\n+++ b/app.py\n@@ -1,1 +1,1 @@\n-x = 1\n"
                + "+api_" + "key=" + "abcdefghijklmnop\n"
            )
            manifest = _raw_pack(Path(tmp) / "pack", [candidate])
            result = ingest_llm_candidate_pack(manifest)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.blockers[0]["reason"], "secret_positive_on_rescan")

    def test_unscanned_candidate_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _clean_candidate()
            candidate.pop("redaction")  # no secret-scan evidence
            manifest = _raw_pack(Path(tmp) / "pack", [candidate])
            result = ingest_llm_candidate_pack(manifest)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.blockers[0]["reason"], "unscanned_candidate")

    def test_unsafe_patch_path_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _clean_candidate()
            candidate["patch_content"] = (
                "--- a/../../etc/passwd\n+++ b/../../etc/passwd\n@@ -1,1 +1,1 @@\n-x\n+y\n"
            )
            manifest = _raw_pack(Path(tmp) / "pack", [candidate])
            result = ingest_llm_candidate_pack(manifest)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.blockers[0]["reason"], "unsafe_patch_path")


class LLMCandidatePackBuildIntegrationTest(unittest.TestCase):
    def test_pack_build_ingests_llm_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _dry_run_pack(tmp_path / "pack", max_candidates=3)
            (tmp_path / "before.py").write_text(
                "def accumulate(values):\n    total = 0\n    for v in values:\n"
                "        total += v\n    return total\n",
                encoding="utf-8",
            )
            config = {
                "schema_version": "codelewm.downstream_rerank_benchmark_config.v1",
                "benchmark_id": "codelewm-llm-ingest-fixture",
                "profile": "anti_saturation_semantic_v1",
                "min_labeled_examples": 100,
                "required_baselines": [
                    "llm_order", "random", "lexical", "no_action",
                    "codelewm", "retrieval_prior", "score_ensemble",
                ],
                "required_metrics": [
                    "pass_at_1", "pass_at_k", "mrr", "valid_patch_rate", "check_pass_rate"
                ],
                "source_license_policy": {
                    "schema_version": "codelewm.downstream_source_license_policy.v1",
                    "publication_allowed": True,
                    "source_kind": "project_fixture",
                    "license": "repository license",
                },
                "tasks": [
                    {
                        "task_id": "accumulator",
                        "task_type": "refactor",
                        "prompt": "rewrite the accumulator update explicitly",
                        "before_path": "before.py",
                        "split": "test",
                        "repo_id": "r1",
                        "candidates": [],
                        "llm_candidate_packs": ["pack/manifest.json"],
                    }
                ],
            }
            (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            out = tmp_path / "out"
            result = build_downstream_benchmark_pack(config_path=tmp_path / "config.json", out=out)

            ingest_report = json.loads(
                (out / result.llm_candidate_ingest_report_path).read_text(encoding="utf-8")
            )
            benchmark = read_downstream_rerank_benchmark(out / result.benchmark_path)
            manifest = read_artifact_manifest(out / result.artifact_manifest_path)

        self.assertEqual(result.ingested_llm_candidate_count, 3)
        self.assertTrue(ingest_report["ok"])
        self.assertEqual(ingest_report["pack_count"], 1)
        self.assertEqual(len(ingest_report["source_manifest_ids"]), 1)

        candidates = benchmark.tasks[0].candidates
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            self.assertEqual(candidate.source["hard_negative_class"], "llm_generated")
            self.assertEqual(candidate.source["origin"], "llm")
            self.assertTrue(candidate.source["model_slug"])
            self.assertTrue(candidate.provenance["source_manifest_id"])
            self.assertIsNotNone(candidate.patch_path)  # never executed; applied text-only later
            self.assertEqual(candidate.label, "unknown")

        # Lineage: the pack artifact id is recorded as a manifest parent.
        self.assertEqual(len(manifest.parent_artifacts), 1)
        self.assertEqual(manifest.parent_artifacts[0], ingest_report["source_manifest_ids"][0])
        self.assertTrue(manifest.metadata["llm_candidate_ingest_ok"])


if __name__ == "__main__":
    unittest.main()
