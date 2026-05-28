"""Execution-pack builder tests.

These tests run the real sandbox (subprocess per record), so they cost
~50-200 ms per record. Fixtures are kept small.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from codelewm.data.execution_pack import (
    EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
    EXECUTION_PACK_RECORD_SCHEMA_VERSION,
    ExecutionPackBuilderError,
    build_execution_pack,
)
from codelewm.data.execution_sources import load_execution_source
from codelewm.data.sandbox import SandboxPolicy


FIXTURES = Path(__file__).resolve().parents[1] / "execution_sources" / "fixtures"


def _ingest_fixture(name: str, source: str, tmpdir: Path) -> Path:
    """Run the ingestion adapter, return the JSONL path."""

    out = tmpdir / f"{source}.jsonl"
    load_execution_source(
        source=source,  # type: ignore[arg-type]
        source_path=FIXTURES / name,
        output_path=out,
    )
    return out


def _fast_policy() -> SandboxPolicy:
    return SandboxPolicy(
        timeout_ms=3000,
        memory_mb=1024,
        cpu_seconds=2,
        determinism_check=True,
    )


class ExecutionPackHappyPathTest(unittest.TestCase):
    def test_packs_mbpp_fixture_with_three_problems(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ingest = _ingest_fixture("mbpp_tiny.jsonl", "mbpp", tmp)
            out = tmp / "pack"
            result = build_execution_pack(
                ingestion_paths=[ingest],
                output_dir=out,
                sandbox_policy=_fast_policy(),
                seed=7,
                train_frac=0.5,
                val_frac=0.25,
            )
            self.assertGreater(result.record_count, 0)
            self.assertEqual(
                result.manifest.schema_version,
                EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
            )
            self.assertEqual(result.manifest.split_by, "source_problem_id")
            # Manifest sums to record_count.
            self.assertEqual(
                sum(result.manifest.split_counts.values()),
                result.record_count,
            )
            # Pack JSONL has one line per record and per-line schema marker.
            lines = (out / "pack.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), result.record_count)
            first = json.loads(lines[0])
            self.assertEqual(
                first["schema_version"], EXECUTION_PACK_RECORD_SCHEMA_VERSION
            )
            for key in (
                "code_tokens",
                "input_tokens",
                "output_tokens",
                "code_checksum",
                "input_repr_checksum",
                "output_repr_checksum",
                "split",
                "license",
            ):
                self.assertIn(key, first, msg=f"missing field: {key}")
            # Sidecars exist.
            self.assertTrue((out / "attribution.json").is_file())
            self.assertTrue((out / "sandbox_audit_summary.json").is_file())
            self.assertTrue((out / "claim_boundary.md").is_file())
            # Claim boundary fingerprint matches the registered file.
            from codelewm.security import claim_boundary_fingerprint

            self.assertEqual(
                result.manifest.claim_boundary["fingerprint"],
                claim_boundary_fingerprint("execution_substrate.v1"),
            )

    def test_split_by_problem_id_no_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ingest = _ingest_fixture("mbpp_tiny.jsonl", "mbpp", tmp)
            out = tmp / "pack"
            build_execution_pack(
                ingestion_paths=[ingest],
                output_dir=out,
                sandbox_policy=_fast_policy(),
                seed=42,
                train_frac=0.5,
                val_frac=0.25,
            )
            problem_to_split: dict[str, set[str]] = {}
            for line in (out / "pack.jsonl").read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                problem_to_split.setdefault(payload["source_problem_id"], set()).add(
                    payload["split"]
                )
            for pid, splits in problem_to_split.items():
                self.assertEqual(
                    len(splits), 1, msg=f"problem {pid} appears in {splits}"
                )

    def test_pack_checksum_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ingest = _ingest_fixture("mbpp_tiny.jsonl", "mbpp", tmp)
            out = tmp / "pack"
            result = build_execution_pack(
                ingestion_paths=[ingest],
                output_dir=out,
                sandbox_policy=_fast_policy(),
            )
            self.assertTrue(result.manifest.pack_jsonl_checksum)
            self.assertEqual(len(result.manifest.pack_jsonl_checksum), 64)


class ExecutionPackHeldOutTest(unittest.TestCase):
    def test_mbpp_plus_records_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ingest = _ingest_fixture("mbpp_plus_tiny.jsonl", "mbpp_plus", tmp)
            out = tmp / "pack"
            result = build_execution_pack(
                ingestion_paths=[ingest],
                output_dir=out,
                sandbox_policy=_fast_policy(),
            )
            self.assertEqual(result.record_count, 0)
            self.assertGreaterEqual(result.manifest.held_out_eval_excluded_count, 1)


class ExecutionPackErrorTest(unittest.TestCase):
    def test_non_empty_output_dir_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ingest = _ingest_fixture("mbpp_tiny.jsonl", "mbpp", tmp)
            out = tmp / "pack"
            out.mkdir()
            (out / "junk.txt").write_text("not empty", encoding="utf-8")
            with self.assertRaises(ExecutionPackBuilderError):
                build_execution_pack(
                    ingestion_paths=[ingest],
                    output_dir=out,
                    sandbox_policy=_fast_policy(),
                )

    def test_missing_sandbox_policy_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ingest = _ingest_fixture("mbpp_tiny.jsonl", "mbpp", tmp)
            out = tmp / "pack"
            with self.assertRaises(ExecutionPackBuilderError):
                build_execution_pack(
                    ingestion_paths=[ingest],
                    output_dir=out,
                    sandbox_policy=None,
                )


class ExecutionPackCLITest(unittest.TestCase):
    def test_cli_round_trip(self) -> None:
        from codelewm.harness.cli import build_parser

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ingest = _ingest_fixture("mbpp_tiny.jsonl", "mbpp", tmp)
            out = tmp / "pack"
            parser = build_parser()
            namespace = parser.parse_args(
                [
                    "dataset",
                    "execution-pack",
                    "--ingestion",
                    str(ingest),
                    "--output",
                    str(out),
                    "--memory-mb",
                    "1024",
                    "--timeout-ms",
                    "3000",
                    "--cpu-seconds",
                    "2",
                    "--json",
                ]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = namespace.func(namespace)
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertGreater(payload["record_count"], 0)
            self.assertEqual(
                payload["manifest"]["schema_version"],
                EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
