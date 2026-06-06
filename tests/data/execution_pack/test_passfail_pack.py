"""Tests for the v0.8 pass/fail execution-pack adapter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.data.execution_pack import EXECUTION_PACK_RECORD_SCHEMA_VERSION
from codelewm.data.execution_pack.build_passfail_pack import (
    PASSFAIL_PACK_REPORT_SCHEMA_VERSION,
    PassFailPackBuilderError,
    PassFailPackSource,
    build_passfail_pack,
)
from codelewm.data.execution_rerank_sampler import build_mutation_rerank_pack
from codelewm.data.execution_rerank_sampler import COMPLETION_LABEL_SCHEMA_VERSION
from codelewm.data.execution_sources import get_execution_source_adapter
from codelewm.data.sandbox import SandboxPolicy
from codelewm.data.execution_pack.record import sha256_text
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_humaneval_source(path: Path, *, count: int = 3) -> None:
    rows: list[dict[str, object]] = []
    for idx in range(count):
        offset = idx + 1
        entry_point = f"he_add_{offset}"
        rows.append(
            {
                "task_id": f"HumanEval/{idx}",
                "prompt": f"def {entry_point}(n):\n    \"\"\"fixture\"\"\"\n    ",
                "canonical_solution": f"    return n + {offset}\n",
                "test": (
                    "def check(candidate):\n"
                    f"    assert candidate(1) == {1 + offset}\n"
                    f"    assert candidate(3) == {3 + offset}\n"
                ),
                "entry_point": entry_point,
            }
        )
    _write_jsonl(path, rows)


def _write_mbpp_plus_source(path: Path, *, count: int = 3) -> None:
    rows: list[dict[str, object]] = []
    for idx in range(count):
        offset = idx + 10
        entry_point = f"mbpp_add_{offset}"
        inputs = [[1], [2], [3]]
        rows.append(
            {
                "task_id": f"Mbpp/{100 + idx}",
                "prompt": "fixture",
                "canonical_solution": f"def {entry_point}(n):\n    return n + {offset}\n",
                "entry_point": entry_point,
                "base_input": inputs[:2],
                "plus_input": inputs[2:],
                "expected_output": [item[0] + offset for item in inputs],
            }
        )
    _write_jsonl(path, rows)


def _write_completion_labels_from_source(
    *,
    benchmark: str,
    source_path: Path,
    labels_path: Path,
) -> Path:
    adapter = get_execution_source_adapter(benchmark)
    rows: list[dict[str, object]] = []
    for submission in adapter.iter_submissions(source_path=source_path):
        scoring_inputs = [
            {
                "input_id": input_case.input_id,
                "input_repr": input_case.input_repr,
                "input_kind": input_case.input_kind,
                "function_name": input_case.function_name,
            }
            for input_case in submission.inputs
        ]
        test_results = [
            {
                "input_id": input_case.input_id,
                "expected_output_sha256": sha256_text(expected_output),
            }
            for input_case, expected_output in zip(
                submission.inputs, submission.expected_outputs or (), strict=True
            )
        ]
        for passed in (True, False):
            code = submission.code if passed else _make_failing_addition(submission.code)
            rows.append(
                {
                    "schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
                    "benchmark_id": benchmark,
                    "problem_id": submission.source_problem_id,
                    "completion_id": (
                        f"{submission.source_problem_id}::"
                        f"{'pass' if passed else 'fail'}"
                    ),
                    "completion_text": code,
                    "code": code,
                    "completion_sha256": sha256_text(code),
                    "llm_order_rank": 0 if passed else 1,
                    "llm": "fixture",
                    "passed": passed,
                    "label": "pass" if passed else "fail",
                    "valid_candidate": True,
                    "scoring_inputs": scoring_inputs,
                    "test_results": test_results,
                }
            )
    _write_jsonl(labels_path, rows)
    return labels_path


def _make_failing_addition(code: str) -> str:
    if " + " not in code:
        raise AssertionError(f"fixture code does not contain addition: {code}")
    return code.replace(" + ", " - ", 1)


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

    def test_builds_cross_benchmark_pack_with_split_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            humaneval_source = tmp / "humaneval.jsonl"
            mbpp_plus_source = tmp / "mbpp_plus.jsonl"
            _write_humaneval_source(humaneval_source)
            _write_mbpp_plus_source(mbpp_plus_source)
            humaneval_labels = _write_completion_labels_from_source(
                benchmark="humaneval",
                source_path=humaneval_source,
                labels_path=tmp / "labels" / "humaneval_completion_labels.jsonl",
            )
            mbpp_plus_labels = _write_completion_labels_from_source(
                benchmark="mbpp_plus",
                source_path=mbpp_plus_source,
                labels_path=tmp / "labels" / "mbpp_plus_completion_labels.jsonl",
            )

            pack_dir = tmp / "v0_9_pack"
            result = build_passfail_pack(
                sources=(
                    PassFailPackSource(
                        benchmark="humaneval",
                        source_path=humaneval_source,
                        completion_label_paths=(humaneval_labels,),
                    ),
                    PassFailPackSource(
                        benchmark="mbpp_plus",
                        source_path=mbpp_plus_source,
                        completion_label_paths=(mbpp_plus_labels,),
                    ),
                ),
                output_dir=pack_dir,
                sandbox_policy=_fast_policy(),
                train_frac=0.5,
                val_frac=0.25,
                require_split_coverage=True,
                required_probe_targets=("output_magnitude_bucket",),
            )

            self.assertGreater(result.record_count, 0)
            rows = [
                json.loads(line)
                for line in (pack_dir / "pack.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                {row["source_dataset"] for row in rows},
                {"humaneval", "mbpp_plus"},
            )
            self.assertEqual(
                {row["benchmark_id"] for row in rows},
                {"humaneval", "mbpp_plus"},
            )
            self.assertEqual({row["passed"] for row in rows}, {False, True})
            self.assertTrue(all(row["output_magnitude_bucket"] for row in rows))

            problem_to_split: dict[tuple[str, str], set[str]] = {}
            for row in rows:
                key = (str(row["source_dataset"]), str(row["source_problem_id"]))
                problem_to_split.setdefault(key, set()).add(str(row["split"]))
            for key, splits in problem_to_split.items():
                self.assertEqual(len(splits), 1, msg=f"{key} leaked: {splits}")

            for split in ("val", "test"):
                split_rows = [row for row in rows if row["split"] == split]
                self.assertTrue(any(row["passed"] is True for row in split_rows))
                self.assertTrue(any(row["passed"] is False for row in split_rows))
                self.assertTrue(
                    any(row["output_magnitude_bucket"] for row in split_rows)
                )

            report = json.loads(
                (pack_dir / result.report_path).read_text(encoding="utf-8")
            )
            self.assertEqual(report["benchmark_counts"].keys(), {"humaneval", "mbpp_plus"})
            self.assertEqual(
                report["completion_label_row_counts_by_benchmark"].keys(),
                {"humaneval", "mbpp_plus"},
            )
            self.assertTrue(
                report["readiness_gates"]["held_out_split_label_coverage"]["passed"]
            )
            self.assertEqual(
                report["readiness_gates"]["held_out_split_label_coverage"][
                    "required_probe_targets"
                ],
                ["output_magnitude_bucket"],
            )
            self.assertTrue(report["output_magnitude_bucket_counts"])

            artifact = read_artifact_manifest(pack_dir / "artifact_manifest.json")
            self.assertEqual(artifact.metadata["benchmarks"], ["humaneval", "mbpp_plus"])
            validate_artifact_checksums(artifact, root=pack_dir)

            batches = list(
                iter_batches(
                    ExecutionPackLoaderConfig(
                        pack_jsonl=pack_dir / "pack.jsonl",
                        batch_size=4,
                        code_sequence_length=64,
                        action_sequence_length=32,
                        output_sequence_length=16,
                    )
                )
            )
            self.assertTrue(batches)
            self.assertTrue(all(batch.passed is not None for batch in batches))

    def test_cli_builds_cross_benchmark_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            humaneval_source = tmp / "humaneval.jsonl"
            mbpp_plus_source = tmp / "mbpp_plus.jsonl"
            _write_humaneval_source(humaneval_source)
            _write_mbpp_plus_source(mbpp_plus_source)
            humaneval_labels = _write_completion_labels_from_source(
                benchmark="humaneval",
                source_path=humaneval_source,
                labels_path=tmp / "labels" / "humaneval_completion_labels.jsonl",
            )
            mbpp_plus_labels = _write_completion_labels_from_source(
                benchmark="mbpp_plus",
                source_path=mbpp_plus_source,
                labels_path=tmp / "labels" / "mbpp_plus_completion_labels.jsonl",
            )
            pack_dir = tmp / "v0_9_pack_cli"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path("scripts/build-passfail-pack")),
                    "--benchmark-source",
                    f"humaneval={humaneval_source}",
                    "--benchmark-source",
                    f"mbpp_plus={mbpp_plus_source}",
                    "--benchmark-completion-labels",
                    f"humaneval={humaneval_labels}",
                    "--benchmark-completion-labels",
                    f"mbpp_plus={mbpp_plus_labels}",
                    "--out",
                    str(pack_dir),
                    "--train-frac",
                    "0.5",
                    "--val-frac",
                    "0.25",
                    "--require-split-coverage",
                    "--required-probe-target",
                    "output_magnitude_bucket",
                    "--timeout-ms",
                    "3000",
                    "--memory-mb",
                    "1024",
                    "--cpu-seconds",
                    "2",
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[3],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertGreater(payload["record_count"], 0)
            report = json.loads(
                (pack_dir / "reports" / "passfail_pack_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                report["readiness_gates"]["held_out_split_label_coverage"][
                    "passed"
                ]
            )
            self.assertEqual(
                set(report["benchmark_counts"]),
                {"humaneval", "mbpp_plus"},
            )

    def test_split_coverage_failure_is_typed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            humaneval_source = tmp / "humaneval.jsonl"
            _write_humaneval_source(humaneval_source)
            labels = _write_completion_labels_from_source(
                benchmark="humaneval",
                source_path=humaneval_source,
                labels_path=tmp / "labels" / "humaneval_completion_labels.jsonl",
            )

            with self.assertRaisesRegex(
                PassFailPackBuilderError,
                "split_coverage_blocker: .*output_length_bucket",
            ):
                build_passfail_pack(
                    sources=(
                        PassFailPackSource(
                            benchmark="humaneval",
                            source_path=humaneval_source,
                            completion_label_paths=(labels,),
                        ),
                    ),
                    output_dir=tmp / "blocked",
                    sandbox_policy=_fast_policy(),
                    train_frac=0.5,
                    val_frac=0.25,
                    require_split_coverage=True,
                    required_probe_targets=("output_length_bucket",),
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
