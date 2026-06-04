"""Tests for the execution-pack publish gate and card rendering."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from codelewm.data.execution_pack import (
    EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
    PrePublishGateError,
    build_execution_pack,
    context_from_manifest,
    render_dataset_card,
    run_pre_publish_gate,
)
from codelewm.data.execution_sources import load_execution_source
from codelewm.data.sandbox import SandboxPolicy


FIXTURES = (
    Path(__file__).resolve().parents[1] / "execution_sources" / "fixtures"
)
SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "hf-publish-execution-pack"
)


def _build_small_pack(tmpdir: Path) -> Path:
    ingest = tmpdir / "mbpp.jsonl"
    load_execution_source(
        source="mbpp",
        source_path=FIXTURES / "mbpp_tiny.jsonl",
        output_path=ingest,
    )
    out = tmpdir / "pack"
    build_execution_pack(
        ingestion_paths=[ingest],
        output_dir=out,
        sandbox_policy=SandboxPolicy(
            timeout_ms=3000,
            memory_mb=1024,
            cpu_seconds=2,
            determinism_check=True,
        ),
        seed=11,
    )
    return out


class PrePublishGateTest(unittest.TestCase):
    def test_passes_on_freshly_built_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _build_small_pack(Path(tmpdir))
            report = run_pre_publish_gate(pack)
            self.assertTrue(
                report.allowed, msg=f"findings: {report.findings}"
            )
            self.assertGreater(report.record_count, 0)
            self.assertTrue(report.checks["manifest_schema_version_supported"])
            self.assertTrue(report.checks["pack_jsonl_checksum_matches"])
            self.assertTrue(report.checks["claim_boundary_embedded"])
            self.assertTrue(report.checks["claim_boundary_fingerprint_matches"])
            self.assertTrue(report.checks["all_licenses_permissive"])
            self.assertTrue(report.checks["every_source_has_attribution"])
            self.assertTrue(report.checks["record_count_nonzero"])

    def test_missing_manifest_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(PrePublishGateError):
                run_pre_publish_gate(Path(tmpdir))

    def test_pack_jsonl_checksum_mismatch_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _build_small_pack(Path(tmpdir))
            # Corrupt pack.jsonl after the fact.
            (pack / "pack.jsonl").write_text("tampered\n", encoding="utf-8")
            report = run_pre_publish_gate(pack)
            self.assertFalse(report.checks["pack_jsonl_checksum_matches"])
            self.assertFalse(report.allowed)

    def test_missing_claim_boundary_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _build_small_pack(Path(tmpdir))
            (pack / "claim_boundary.md").unlink()
            report = run_pre_publish_gate(pack)
            self.assertFalse(report.checks["claim_boundary_embedded"])
            self.assertFalse(report.allowed)

    def test_non_permissive_license_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _build_small_pack(Path(tmpdir))
            manifest_path = pack / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["license_breakdown"]["GPL-3.0"] = 1
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = run_pre_publish_gate(pack)
            self.assertFalse(report.checks["all_licenses_permissive"])
            self.assertFalse(report.allowed)


class DatasetCardRenderTest(unittest.TestCase):
    def test_card_includes_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _build_small_pack(Path(tmpdir))
            context = context_from_manifest(
                manifest_path=pack / "manifest.json",
                attribution_path=pack / "attribution.json",
                repo_id="abdelstark/codelewm-execution-pack",
                revision="v0.6.0",
            )
            card = render_dataset_card(context=context)
            for section in (
                "# abdelstark/codelewm-execution-pack",
                "## Summary",
                "## Provenance",
                "## License Summary",
                "## Attribution",
                "## Sandbox Policy",
                "## Determinism And Reject Counts",
                "## Split Policy",
                "## Output Distribution",
                "## Claim Boundary",
                "## How To Verify",
            ):
                self.assertIn(section, card, msg=f"card missing section {section!r}")
            self.assertIn(EXECUTION_PACK_MANIFEST_SCHEMA_VERSION, card)
            self.assertIn("execution_substrate.v1", card)


class HFPublishScriptDryRunTest(unittest.TestCase):
    def test_uv_script_metadata_declares_runtime_dependencies(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"huggingface_hub>=1.0"', text)
        self.assertIn('"numpy>=1.24"', text)

    def test_dry_run_emits_json_plan_and_renders_readme(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _build_small_pack(Path(tmpdir))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--pack-dir",
                    str(pack),
                    "--repo-id",
                    "test/codelewm-execution-pack",
                    "--revision",
                    "v0.6.0-test",
                    "--dry-run",
                    "--json",
                ],
                env={
                    "CODELEWM_HF_PUBLISH_DRY_RUN": "1",
                    "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode, 0,
                msg=f"stderr={completed.stderr!r} stdout={completed.stdout!r}",
            )
            plan = json.loads(completed.stdout)
            self.assertEqual(plan["repo_id"], "test/codelewm-execution-pack")
            self.assertTrue(plan["dry_run"])
            self.assertTrue(plan["gate"]["allowed"])
            self.assertTrue((pack / "README.md").is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
