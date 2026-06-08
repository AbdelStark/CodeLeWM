from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    HARD_DOWNSTREAM_ARTIFACT_INDEX_SCHEMA_VERSION,
    HARD_DOWNSTREAM_CLAIM_AUDIT_SCHEMA_VERSION,
    assemble_hard_downstream_artifact_set,
    build_downstream_benchmark_pack,
    read_hard_downstream_claim_audit,
    run_downstream_rerank_evaluation,
)
from codelewm.eval.hard_downstream_publish import DIAGNOSTIC_FALLBACK_WORDING
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums
from codelewm.observability.manifest import sha256_file


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "config" / "benchmark" / "hard_downstream_anti_saturation_fixture.json"


def _build_pack_and_rerank(root: Path) -> tuple[Path, Path]:
    benchmark_dir = root / "benchmark"
    rerank_dir = root / "rerank"
    checkpoint = root / "checkpoint.bin"
    checkpoint.write_bytes(b"fixture checkpoint")
    build_downstream_benchmark_pack(config_path=FIXTURE_CONFIG, out=benchmark_dir)
    run_downstream_rerank_evaluation(
        benchmark_manifest=benchmark_dir / "manifest.json",
        checkpoint=checkpoint,
        out=rerank_dir,
        allow_unsafe_checkpoint=True,
        bootstrap_samples=0,
        hard_mode=True,
    )
    return benchmark_dir / "manifest.json", rerank_dir / "manifest.json"


class HardDownstreamPublishTest(unittest.TestCase):
    def test_assembles_verified_secret_scanned_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_manifest, rerank_manifest = _build_pack_and_rerank(root)
            out = root / "publication"
            result = assemble_hard_downstream_artifact_set(
                pack_manifest=pack_manifest,
                rerank_manifest=rerank_manifest,
                out=out,
            )
            manifest = read_artifact_manifest(out / result.artifact_manifest_path)
            checked = validate_artifact_checksums(manifest, root=out)
            claim_audit = read_hard_downstream_claim_audit(out / result.claim_audit_path)
            index = json.loads((out / result.artifact_index_path).read_text(encoding="utf-8"))
            names = {path.name for path in checked}

        # The gate is closed on the one-task fixture; no broad claim is asserted.
        self.assertFalse(result.claim_allowed)
        self.assertFalse(result.anti_saturation_eligible)
        self.assertFalse(result.broad_coding_improvement_claim_allowed)

        # The required published artifact set is present.
        for required in (
            "benchmark.json",
            "downstream_rerank_report.json",
            "anti_saturation_report.json",
            "source_license_policy.json",
            "split_leakage_report.json",
            "claim_audit.json",
            "artifact_index.json",
            "publication_secret_scan_report.json",
        ):
            self.assertIn(required, names, required)

        # Claim audit follows the gate: diagnostic wording, saturated slice listed.
        self.assertEqual(claim_audit["schema_version"], HARD_DOWNSTREAM_CLAIM_AUDIT_SCHEMA_VERSION)
        self.assertEqual(claim_audit["public_wording"], DIAGNOSTIC_FALLBACK_WORDING)
        self.assertFalse(claim_audit["broad_coding_improvement_claim_allowed"])
        self.assertEqual(claim_audit["saturated_slices"][0]["eligible"], False)
        self.assertIn(
            "problem_count_below_minimum:1<100",
            claim_audit["headline_slice"]["blocked_reasons"],
        )
        self.assertIn("p_pass", claim_audit["missing_baselines"])

        # The artifact index checksums are accurate.
        self.assertEqual(index["schema_version"], HARD_DOWNSTREAM_ARTIFACT_INDEX_SCHEMA_VERSION)
        self.assertEqual(len(index["parent_artifacts"]), 2)
        self.assertTrue(index["files"])

    def test_index_checksums_match_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_manifest, rerank_manifest = _build_pack_and_rerank(root)
            out = root / "publication"
            result = assemble_hard_downstream_artifact_set(
                pack_manifest=pack_manifest, rerank_manifest=rerank_manifest, out=out
            )
            index = json.loads((out / result.artifact_index_path).read_text(encoding="utf-8"))
            for entry in index["files"]:
                if entry["path"] == "artifact_index.json":
                    continue  # self-referential: hashed before it lists itself
                self.assertEqual(sha256_file(out / entry["path"]), entry["sha256"], entry["path"])

    def test_cli_publishes_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_manifest, rerank_manifest = _build_pack_and_rerank(root)
            out = root / "publication"
            completed = subprocess.run(
                [
                    sys.executable, "-m", "codelewm.harness.cli", "eval", "hard-downstream-publish",
                    "--pack-manifest", str(pack_manifest),
                    "--rerank-manifest", str(rerank_manifest),
                    "--out", str(out),
                    "--json",
                ],
                cwd=ROOT, check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            claim_audit = read_hard_downstream_claim_audit(out / payload["claim_audit_path"])
        self.assertFalse(payload["claim_allowed"])
        self.assertEqual(claim_audit["public_wording"], DIAGNOSTIC_FALLBACK_WORDING)


if __name__ == "__main__":
    unittest.main()
