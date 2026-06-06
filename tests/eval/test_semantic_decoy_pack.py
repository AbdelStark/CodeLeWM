"""Tests for manifest-backed semantic decoy pack artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.data.execution_pack.manifest import (
    EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
    EXECUTION_PACK_RECORD_SCHEMA_VERSION,
)
from codelewm.eval import (
    SEMANTIC_DECOY_PACK_RUN_SCHEMA_VERSION,
    SEMANTIC_DECOY_PACK_SCHEMA_VERSION,
    SEMANTIC_DECOY_PAIR_SCHEMA_VERSION,
    SEMANTIC_DECOY_SUMMARY_SCHEMA_VERSION,
    build_semantic_decoy_pack,
    generate_same_problem_different_submission_pairs,
    load_semantic_decoy_pack,
)
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


class SameProblemSemanticGeneratorTest(unittest.TestCase):
    def test_strengthened_mode_allows_different_inputs_same_problem(self) -> None:
        records = (
            _record("p1", "s1", "i1", "1", split="val"),
            _record("p1", "s2", "i2", "2", split="val"),
        )
        strict_pairs, _ = generate_same_problem_different_submission_pairs(
            records,
            same_input_only=True,
        )
        strengthened_pairs, _ = generate_same_problem_different_submission_pairs(
            records,
            same_input_only=False,
        )
        self.assertEqual(strict_pairs, [])
        self.assertEqual(len(strengthened_pairs), 2)
        self.assertTrue(
            all("different_input_id" in pair.rationale for pair in strengthened_pairs)
        )


class SemanticDecoyPackTest(unittest.TestCase):
    def test_builds_manifested_pack_with_pair_count_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_root = _write_execution_pack(root / "pack")
            out = root / "semantic_decoys"

            result = build_semantic_decoy_pack(
                pack=pack_root,
                out=out,
                min_pairs_for_claim=6,
                min_distinct_problems_for_claim=3,
                max_pairs_per_query=2,
                command=("codelewm", "eval", "semantic-decoy-pack"),
            )

            self.assertEqual(
                result.schema_version, SEMANTIC_DECOY_PACK_RUN_SCHEMA_VERSION
            )
            self.assertGreaterEqual(result.pair_count, 6)
            self.assertEqual(result.distinct_problem_count, 3)
            self.assertTrue(result.claim_allowed)

            manifest = read_artifact_manifest(out / result.artifact_manifest_path)
            validate_artifact_checksums(manifest, root=out)
            self.assertEqual(manifest.artifact_kind, "downstream_benchmark")
            self.assertEqual(
                manifest.metadata["schema_version"], SEMANTIC_DECOY_PACK_SCHEMA_VERSION
            )
            self.assertEqual(
                manifest.metadata["pair_schema_version"],
                SEMANTIC_DECOY_PAIR_SCHEMA_VERSION,
            )
            self.assertEqual(
                manifest.metadata["summary_schema_version"],
                SEMANTIC_DECOY_SUMMARY_SCHEMA_VERSION,
            )

            summary = json.loads((out / result.summary_path).read_text(encoding="utf-8"))
            self.assertTrue(summary["claim_gate"]["claim_allowed"])
            self.assertEqual(
                summary["filtering_summary"]["split_policy"]["train_rows_excluded"],
                1,
            )
            self.assertEqual(
                summary["source_license_policy"]["license_breakdown"],
                {"mit": 9},
            )
            pair_rows = [
                json.loads(line)
                for line in (out / result.pair_rows_path)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                {row["schema_version"] for row in pair_rows},
                {SEMANTIC_DECOY_PAIR_SCHEMA_VERSION},
            )
            self.assertEqual(
                {row["control_category"] for row in pair_rows},
                {"semantic_same_problem"},
            )
            self.assertEqual(
                {
                    "same_code_different_input",
                    "same_problem_different_submission",
                },
                {row["category"] for row in pair_rows},
            )
            self.assertEqual(
                {row["source_dataset"] for row in pair_rows},
                {"fixture"},
            )
            self.assertEqual(
                {row["query_source_dataset"] for row in pair_rows},
                {"fixture"},
            )
            self.assertEqual(
                {row["decoy_source_dataset"] for row in pair_rows},
                {"fixture"},
            )
            self.assertEqual(
                {row["query_record_schema_version"] for row in pair_rows},
                {EXECUTION_PACK_RECORD_SCHEMA_VERSION},
            )
            self.assertEqual(
                summary["benchmark_counts"],
                {"fixture": 8},
            )
            self.assertEqual(
                summary["record_schema_versions"],
                {EXECUTION_PACK_RECORD_SCHEMA_VERSION: 8},
            )

            loaded = load_semantic_decoy_pack(out / result.artifact_manifest_path)
            self.assertEqual(len(loaded.pairs), result.pair_count)
            self.assertEqual(
                {
                    "same_code_different_input",
                    "same_problem_different_submission",
                },
                {report.category for report in loaded.generation_reports()},
            )

    def test_low_count_pack_keeps_claim_gate_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_root = _write_execution_pack(root / "pack")
            result = build_semantic_decoy_pack(
                pack=pack_root,
                out=root / "semantic_decoys",
                min_pairs_for_claim=100,
                min_distinct_problems_for_claim=30,
            )
            self.assertFalse(result.claim_allowed)
            summary = json.loads(
                (root / "semantic_decoys" / result.summary_path).read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(summary["claim_gate"]["claim_allowed"])
            self.assertIn("pair_count", summary["claim_gate"]["claim_reason"])

    def test_cli_builds_semantic_decoy_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_root = _write_execution_pack(root / "pack")
            out = root / "semantic_decoys"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "eval",
                    "semantic-decoy-pack",
                    "--pack",
                    str(pack_root),
                    "--out",
                    str(out),
                    "--min-pairs-for-claim",
                    "6",
                    "--min-distinct-problems-for-claim",
                    "3",
                    "--max-pairs-per-query",
                    "2",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(
                payload["schema_version"], SEMANTIC_DECOY_PACK_RUN_SCHEMA_VERSION
            )
            self.assertTrue(payload["claim_allowed"])
            self.assertTrue((out / payload["artifact_manifest_path"]).is_file())


def _record(
    problem: str,
    submission: str,
    input_id: str,
    output: str,
    *,
    split: str,
) -> dict[str, object]:
    return {
        "schema_version": EXECUTION_PACK_RECORD_SCHEMA_VERSION,
        "record_id": f"{problem}::{submission}::{input_id}",
        "source_dataset": "fixture",
        "source_problem_id": problem,
        "source_submission_id": submission,
        "input_id": input_id,
        "split": split,
        "code_tokens": [1, 2],
        "code_checksum": "c" * 64,
        "input_tokens": [3],
        "input_repr_checksum": "i" * 64,
        "input_kind": "function_call",
        "function_name": "f",
        "output_tokens": [ord(output[0])],
        "output_repr": output,
        "output_repr_checksum": "o" * 64,
        "output_kind": "value",
        "output_type": "str",
        "execution_status": "ok",
        "judge_verdict": "accepted",
        "wall_time_ms": 1.0,
        "peak_rss_kb": 100,
        "determinism_check": True,
        "license": "mit",
        "license_attribution_url": "https://example.invalid/license",
        "held_out_for_eval": False,
    }


def _write_execution_pack(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = (
        _record("p1", "s1", "i1", "1", split="val"),
        _record("p1", "s2", "i2", "2", split="val"),
        _record("p1", "s3", "i1", "3", split="val"),
        _record("p2", "s1", "i1", "a", split="val"),
        _record("p2", "s1", "i2", "c", split="val"),
        _record("p2", "s2", "i2", "b", split="val"),
        _record("p3", "s1", "i1", "x", split="test"),
        _record("p3", "s2", "i1", "y", split="test"),
        _record("p4", "s1", "i1", "z", split="train"),
    )
    pack_jsonl = root / "pack.jsonl"
    pack_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest_payload = {
        "schema_version": EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
        "pack_id": "semantic-decoy-pack-fixture",
        "record_count": len(rows),
        "split_counts": {"test": 2, "train": 1, "val": 6},
        "source_breakdown": {"fixture": len(rows)},
        "license_breakdown": {"mit": len(rows)},
    }
    manifest_json = root / "manifest.json"
    manifest_json.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact = build_artifact_manifest(
        artifact_kind="dataset",
        root=root,
        files=(pack_jsonl, manifest_json),
        command=("test", "semantic-decoy-pack-fixture"),
        config={"fixture": True},
        artifact_id="semantic-decoy-pack-fixture",
        metadata={"schema_version": EXECUTION_PACK_MANIFEST_SCHEMA_VERSION},
    )
    write_artifact_manifest(artifact, root / "artifact_manifest.json")
    return root


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
