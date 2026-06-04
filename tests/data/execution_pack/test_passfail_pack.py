"""Tests for the v0.8 pass/fail execution-pack adapter."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from codelewm.data.execution_pack import EXECUTION_PACK_RECORD_SCHEMA_VERSION
from codelewm.data.execution_pack.build_passfail_pack import (
    PASSFAIL_PACK_REPORT_SCHEMA_VERSION,
    build_passfail_pack,
)
from codelewm.data.execution_rerank_sampler import build_mutation_rerank_pack
from codelewm.data.sandbox import SandboxPolicy
from codelewm.observability import (
    read_artifact_manifest,
    validate_artifact_checksums,
)
from codelewm.training import ExecutionPackLoaderConfig, iter_batches


FIXTURES = Path(__file__).resolve().parents[1] / "execution_sources" / "fixtures"


def _fast_policy() -> SandboxPolicy:
    return SandboxPolicy(
        timeout_ms=3000,
        memory_mb=1024,
        cpu_seconds=2,
        determinism_check=True,
    )


class PassFailExecutionPackTest(unittest.TestCase):
    def test_builds_labeled_pack_from_mutation_completion_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            labels_dir = tmp / "labels"
            source = FIXTURES / "mbpp_tiny.jsonl"
            labels = build_mutation_rerank_pack(
                benchmark="mbpp",
                source_path=source,
                out=labels_dir,
                mutants_per_problem=8,
                pool_size=2,
                max_problems=2,
                max_cases_per_problem=2,
                sandbox_policy=_fast_policy(),
            )
            pack_dir = tmp / "passfail"
            result = build_passfail_pack(
                completion_label_paths=(labels_dir / labels.labels_path,),
                source_path=source,
                benchmark="mbpp",
                output_dir=pack_dir,
                sandbox_policy=_fast_policy(),
                train_frac=0.5,
                val_frac=0.25,
            )

            self.assertGreater(result.record_count, 0)
            self.assertIn("true", result.pass_label_counts)
            self.assertIn("false", result.pass_label_counts)
            self.assertGreater(result.pos_weight, 0.0)

            rows = [
                json.loads(line)
                for line in (pack_dir / "pack.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(rows), result.record_count)
            self.assertEqual(
                {row["schema_version"] for row in rows},
                {EXECUTION_PACK_RECORD_SCHEMA_VERSION},
            )
            self.assertEqual({row["passed"] for row in rows}, {False, True})

            problem_to_split: dict[str, set[str]] = {}
            for row in rows:
                problem_to_split.setdefault(row["source_problem_id"], set()).add(
                    row["split"]
                )
            self.assertTrue(problem_to_split)
            for problem_id, splits in problem_to_split.items():
                self.assertEqual(
                    len(splits),
                    1,
                    msg=f"problem {problem_id} appears in splits {splits}",
                )

            report = json.loads(
                (pack_dir / result.report_path).read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["schema_version"], PASSFAIL_PACK_REPORT_SCHEMA_VERSION
            )
            self.assertTrue(report["class_balance_ok"])
            self.assertEqual(
                report["pass_label_granularity"],
                "per_problem_completion_input",
            )

            secret_scan = json.loads(
                (pack_dir / result.secret_scan_report_path).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(secret_scan["ok"], msg=secret_scan)

            artifact = read_artifact_manifest(pack_dir / "artifact_manifest.json")
            self.assertEqual(artifact.artifact_kind, "dataset")
            self.assertEqual(artifact.artifact_id, result.manifest.pack_id)
            validate_artifact_checksums(artifact, root=pack_dir)

            batches = list(
                iter_batches(
                    ExecutionPackLoaderConfig(
                        pack_jsonl=pack_dir / "pack.jsonl",
                        batch_size=2,
                        code_sequence_length=64,
                        action_sequence_length=32,
                        output_sequence_length=16,
                    )
                )
            )
            self.assertTrue(batches)
            self.assertTrue(all(batch.passed is not None for batch in batches))

    def test_relative_output_dir_writes_valid_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            labels_dir = tmp / "labels"
            source = FIXTURES / "mbpp_tiny.jsonl"
            labels = build_mutation_rerank_pack(
                benchmark="mbpp",
                source_path=source,
                out=labels_dir,
                mutants_per_problem=8,
                pool_size=2,
                max_problems=2,
                max_cases_per_problem=2,
                sandbox_policy=_fast_policy(),
            )

            cwd = Path.cwd()
            os.chdir(tmp)
            try:
                result = build_passfail_pack(
                    completion_label_paths=(labels_dir / labels.labels_path,),
                    source_path=source,
                    benchmark="mbpp",
                    output_dir=Path("relative-passfail"),
                    sandbox_policy=_fast_policy(),
                    train_frac=0.5,
                    val_frac=0.25,
                )
            finally:
                os.chdir(cwd)

            pack_dir = tmp / "relative-passfail"
            self.assertEqual(result.output_dir, pack_dir.resolve())
            artifact_path = pack_dir / result.artifact_manifest_path
            self.assertTrue(artifact_path.is_file())
            artifact = read_artifact_manifest(artifact_path)
            self.assertEqual(artifact.artifact_kind, "dataset")
            self.assertEqual(artifact.artifact_id, result.manifest.pack_id)
            validate_artifact_checksums(artifact, root=pack_dir)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
