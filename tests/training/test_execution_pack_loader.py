"""Tests for the execution-pack data loader and smoke runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codelewm.data.execution_pack import build_execution_pack
from codelewm.data.execution_sources import load_execution_source
from codelewm.data.sandbox import SandboxPolicy
from codelewm.training import (
    EXECUTION_PACK_BATCH_SCHEMA_VERSION,
    OUTPUT_TYPE_VOCAB,
    ExecutionPackLoaderConfig,
    LoaderDiagnostics,
    collect_diagnostics,
    iter_batches,
    iter_records,
)


FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "execution_sources"
    / "fixtures"
)
SMOKE_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "smoke-execution-train"
)


def _build_pack(tmpdir: Path) -> Path:
    ingest = tmpdir / "mbpp.jsonl"
    load_execution_source(
        source="mbpp",
        source_path=FIXTURES / "mbpp_tiny.jsonl",
        output_path=ingest,
    )
    pack = tmpdir / "pack"
    build_execution_pack(
        ingestion_paths=[ingest],
        output_dir=pack,
        sandbox_policy=SandboxPolicy(
            timeout_ms=3000,
            memory_mb=1024,
            cpu_seconds=2,
            determinism_check=True,
        ),
    )
    return pack / "pack.jsonl"


class ExecutionPackLoaderTest(unittest.TestCase):
    def test_iter_batches_produces_correctly_shaped_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_jsonl = _build_pack(Path(tmpdir))
            config = ExecutionPackLoaderConfig(
                pack_jsonl=pack_jsonl,
                code_sequence_length=64,
                action_sequence_length=32,
                output_sequence_length=16,
                batch_size=4,
            )
            batches = list(iter_batches(config))
            self.assertGreater(len(batches), 0)
            for batch in batches:
                self.assertEqual(
                    batch.schema_version, EXECUTION_PACK_BATCH_SCHEMA_VERSION
                )
                self.assertLessEqual(batch.batch_size, 4)
                self.assertEqual(
                    batch.code_tokens.shape,
                    (batch.batch_size, 64),
                )
                self.assertEqual(
                    batch.input_tokens.shape,
                    (batch.batch_size, 32),
                )
                self.assertEqual(
                    batch.output_tokens.shape,
                    (batch.batch_size, 16),
                )
                self.assertEqual(batch.code_tokens.dtype, np.int32)
                self.assertEqual(batch.code_attention_mask.dtype, bool)
                self.assertIsNone(batch.passed)

    def test_diagnostics_record_count_matches_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_jsonl = _build_pack(Path(tmpdir))
            config = ExecutionPackLoaderConfig(
                pack_jsonl=pack_jsonl,
                code_sequence_length=128,
                action_sequence_length=64,
                output_sequence_length=64,
                batch_size=2,
            )
            n_records = sum(1 for _ in iter_records(config))
            diagnostics: LoaderDiagnostics = collect_diagnostics(config)
            self.assertEqual(diagnostics.record_count, n_records)
            self.assertGreater(diagnostics.record_count, 0)
            self.assertEqual(
                sum(diagnostics.output_type_histogram.values()),
                diagnostics.record_count,
            )

    def test_split_filter_returns_only_matching_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_jsonl = _build_pack(Path(tmpdir))
            # Read all records to discover which splits exist.
            seen_splits: set[str] = set()
            with pack_jsonl.open(encoding="utf-8") as fh:
                for line in fh:
                    seen_splits.add(json.loads(line)["split"])
            for split in seen_splits:
                with self.subTest(split=split):
                    config = ExecutionPackLoaderConfig(
                        pack_jsonl=pack_jsonl,
                        batch_size=2,
                        split=split,
                    )
                    diag = collect_diagnostics(config)
                    self.assertEqual(
                        set(diag.split_histogram.keys()), {split}
                    )

    def test_truncation_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_jsonl = _build_pack(Path(tmpdir))
            # Tiny sequence length forces truncation.
            config = ExecutionPackLoaderConfig(
                pack_jsonl=pack_jsonl,
                code_sequence_length=2,
                action_sequence_length=1,
                output_sequence_length=1,
                batch_size=2,
            )
            diag = collect_diagnostics(config)
            total_truncations = (
                diag.truncated_code + diag.truncated_input + diag.truncated_output
            )
            self.assertGreater(total_truncations, 0)


class OutputTypeVocabTest(unittest.TestCase):
    def test_vocab_is_stable(self) -> None:
        # Stability is load-bearing for the probe target (#266) — a
        # change to this order would invalidate downstream probe indices.
        self.assertEqual(
            OUTPUT_TYPE_VOCAB,
            (
                "none",
                "bool",
                "int",
                "float",
                "str",
                "bytes",
                "tuple",
                "list",
                "dict",
                "set",
                "exception",
                "other",
            ),
        )


class SmokeRunnerCLITest(unittest.TestCase):
    def test_smoke_runs_to_completion(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(SMOKE_SCRIPT),
                "--ingestion",
                str(FIXTURES / "mbpp_tiny.jsonl"),
                "--batch-size",
                "2",
                "--code-sequence-length",
                "128",
                "--action-sequence-length",
                "32",
                "--output-sequence-length",
                "32",
                "--timeout-ms",
                "3000",
                "--memory-mb",
                "1024",
                "--cpu-seconds",
                "2",
                "--json",
            ],
            env={
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
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
        payload = json.loads(completed.stdout)
        self.assertEqual(
            payload["schema_version"], "codelewm.execution_smoke_report.v1"
        )
        self.assertTrue(payload["passed"])
        self.assertGreater(payload["record_count"], 0)
        self.assertIn(
            "loader_diagnostics", payload, msg=f"missing field: {payload}"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
