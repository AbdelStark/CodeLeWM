"""Tests for the execution-substrate rerank evaluation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.data.execution_rerank_sampler import sample_execution_rerank_completions
from codelewm.eval import (
    COMPLETION_LABEL_SCHEMA_VERSION,
    COMPLETION_SCORE_SCHEMA_VERSION,
    EXECUTION_RERANK_EVAL_RUN_SCHEMA_VERSION,
    EXECUTION_RERANK_REPORT_SCHEMA_VERSION,
    CompletionLabel,
    ExecutionRerankEvalError,
    ExecutionRerankError,
    ScoredCompletion,
    load_completion_labels,
    run_execution_rerank_evaluation,
    rerank_completions,
)
from codelewm.harness.scorer import ScoreError
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "data" / "execution_sources" / "fixtures"


def _make(
    problem: str,
    completion: str,
    rank: int,
    passed: bool,
    scores: dict[str, float],
) -> ScoredCompletion:
    return ScoredCompletion(
        label=CompletionLabel(
            problem_id=problem,
            completion_id=completion,
            code="def f(): pass\n",
            llm_order_rank=rank,
            passed=passed,
        ),
        scores=scores,
    )


class RerankProtocolTest(unittest.TestCase):
    def test_codelewm_pass_at_1_when_score_picks_passing_completion(self) -> None:
        completions = [
            _make("p1", "c1", 1, False, {"codelewm": 0.1, "lexical": 0.5}),
            _make("p1", "c2", 2, True, {"codelewm": 0.9, "lexical": 0.3}),
        ]
        report = rerank_completions(
            completions=completions, benchmark="fixture", min_lift_for_claim=0.0
        )
        self.assertEqual(
            report.schema_version, EXECUTION_RERANK_REPORT_SCHEMA_VERSION
        )
        codelewm = next(b for b in report.baselines if b.baseline == "codelewm")
        llm = next(b for b in report.baselines if b.baseline == "llm_order")
        self.assertEqual(codelewm.pass_at_1, 1.0)
        self.assertEqual(llm.pass_at_1, 0.0)
        self.assertEqual(report.codelewm_lift_over_llm_order, 100.0)

    def test_uniform_completion_counts_required(self) -> None:
        completions = [
            _make("p1", "c1", 1, True, {"codelewm": 0.9}),
            _make("p2", "c1", 1, False, {"codelewm": 0.1}),
            _make("p2", "c2", 2, True, {"codelewm": 0.9}),
        ]
        with self.assertRaises(ExecutionRerankError):
            rerank_completions(completions=completions, benchmark="fixture")

    def test_codelewm_score_required(self) -> None:
        completions = [
            _make("p1", "c1", 1, True, {"lexical": 0.9}),
            _make("p1", "c2", 2, False, {"lexical": 0.1}),
        ]
        with self.assertRaises(ExecutionRerankError):
            rerank_completions(completions=completions, benchmark="fixture")

    def test_empty_completions_rejected(self) -> None:
        with self.assertRaises(ExecutionRerankError):
            rerank_completions(completions=[], benchmark="fixture")

    def test_bootstrap_sample_count_required(self) -> None:
        completions = [
            _make("p1", "c1", 1, True, {"codelewm": 0.9}),
            _make("p1", "c2", 2, False, {"codelewm": 0.1}),
        ]
        with self.assertRaises(ExecutionRerankError):
            rerank_completions(
                completions=completions,
                benchmark="fixture",
                bootstrap_samples=0,
            )

    def test_lift_ci_excludes_zero_when_codelewm_dominates(self) -> None:
        # 5 problems where CodeLeWM picks the passing one and LLM order
        # picks the failing one every time.
        completions = []
        for i in range(5):
            completions.append(
                _make(
                    f"p{i}",
                    f"p{i}-c1",
                    1,
                    False,
                    {"codelewm": 0.1, "lexical": 0.5},
                )
            )
            completions.append(
                _make(
                    f"p{i}",
                    f"p{i}-c2",
                    2,
                    True,
                    {"codelewm": 0.9, "lexical": 0.3},
                )
            )
        report = rerank_completions(
            completions=completions,
            benchmark="fixture",
            bootstrap_samples=500,
            min_lift_for_claim=3.0,
        )
        self.assertGreater(report.codelewm_lift_over_llm_order, 50.0)
        self.assertGreater(report.bootstrap_lift_ci[0], 0.0)
        self.assertTrue(report.claim_allowed)

    def test_claim_blocked_when_lift_is_small(self) -> None:
        completions = []
        for i in range(5):
            # Both orderings pass on each problem -> zero lift.
            completions.append(
                _make(
                    f"p{i}", f"p{i}-c1", 1, True, {"codelewm": 0.9}
                )
            )
            completions.append(
                _make(
                    f"p{i}", f"p{i}-c2", 2, True, {"codelewm": 0.1}
                )
            )
        report = rerank_completions(
            completions=completions,
            benchmark="fixture",
            bootstrap_samples=200,
            min_lift_for_claim=3.0,
        )
        self.assertEqual(report.codelewm_lift_over_llm_order, 0.0)
        self.assertFalse(report.claim_allowed)


class CompletionLabelsLoaderTest(unittest.TestCase):
    def test_load_filters_by_benchmark_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "benchmark_id": "humaneval",
                            "problem_id": "HumanEval/0",
                            "completion_id": "HumanEval/0::0",
                            "code": "def f():\n    return 1\n",
                            "llm_order_rank": 1,
                            "passed": True,
                        },
                        {
                            "benchmark_id": "mbpp_plus",
                            "problem_id": "Mbpp/1",
                            "completion_id": "Mbpp/1::0",
                            "code": "def g():\n    return 2\n",
                            "llm_order_rank": 1,
                            "passed": False,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            humaneval = load_completion_labels(path, benchmark_id="humaneval")
            mbpp = load_completion_labels(path, benchmark_id="mbpp_plus")
            self.assertEqual(len(humaneval), 1)
            self.assertEqual(humaneval[0].problem_id, "HumanEval/0")
            self.assertEqual(len(mbpp), 1)
            self.assertEqual(mbpp[0].problem_id, "Mbpp/1")

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ExecutionRerankError):
            load_completion_labels(
                Path("/nonexistent.jsonl"), benchmark_id="humaneval"
            )

    def test_loads_completion_label_v1_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
                        "benchmark_id": "mbpp-plus",
                        "problem_id": "Mbpp/1",
                        "completion_id": "Mbpp/1::seed-42::rank-1",
                        "completion_text": "def square(n):\n    return n * n\n",
                        "llm_order_rank": 1,
                        "label": "pass",
                        "scoring_inputs": [
                            {
                                "input_id": "Mbpp/1/case-0",
                                "input_repr": "square(3)",
                                "input_kind": "function_call",
                                "function_name": "square",
                            }
                        ],
                        "test_results": [
                            {
                                "input_id": "Mbpp/1/case-0",
                                "passed": True,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            labels = load_completion_labels(path, benchmark_id="mbpp_plus")
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0].code, "def square(n):\n    return n * n\n")
            self.assertTrue(labels[0].passed)
            self.assertEqual(labels[0].scoring_inputs[0].input_repr, "square(3)")


class CompletionRerankEvaluationTest(unittest.TestCase):
    def test_evaluation_writes_manifested_report_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels_out = root / "labels"
            sampling = sample_execution_rerank_completions(
                benchmark="humaneval",
                source_path=FIXTURES / "humaneval_tiny.jsonl",
                out=labels_out,
                samples_per_problem=2,
                llm_seeds=(17,),
            )
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")

            result = run_execution_rerank_evaluation(
                completion_manifest=labels_out / sampling.artifact_manifest_path,
                checkpoint=checkpoint,
                out=root / "rerank",
                benchmark="humaneval",
                allow_unsafe_checkpoint=True,
                bootstrap_samples=32,
                command=("codelewm", "eval", "rerank-humaneval"),
            )

            self.assertEqual(
                result.schema_version, EXECUTION_RERANK_EVAL_RUN_SCHEMA_VERSION
            )
            self.assertEqual(result.benchmark, "humaneval")
            self.assertEqual(result.problem_count, 1)
            self.assertEqual(result.completion_count, 2)
            self.assertFalse(result.claim_allowed)

            output_root = root / "rerank"
            manifest = read_artifact_manifest(output_root / result.artifact_manifest_path)
            validate_artifact_checksums(manifest, root=output_root)
            self.assertEqual(manifest.artifact_kind, "eval_report")
            self.assertEqual(manifest.parent_artifacts, (sampling.artifact_manifest_id,))

            report = json.loads((output_root / result.report_path).read_text())
            self.assertEqual(
                report["schema_version"], EXECUTION_RERANK_REPORT_SCHEMA_VERSION
            )
            self.assertEqual(report["benchmark"], "humaneval")
            self.assertEqual(report["pass_at_k"], 5)
            self.assertIn("codelewm_lift_over_no_action", report)
            self.assertIn("bootstrap_lift_over_no_action_ci", report)
            self.assertFalse(report["claim_allowed"])
            self.assertIn("no_action", report["claim_reason"])
            baselines = {row["baseline"]: row for row in report["baselines"]}
            self.assertEqual(
                {
                    "codelewm",
                    "no_action",
                    "shuffled_action",
                    "llm_order",
                    "random",
                    "lexical",
                },
                set(baselines),
            )
            for summary in baselines.values():
                self.assertIn("pass_at_1", summary)
                self.assertIn("pass_at_k", summary)
                self.assertIn("mrr", summary)

            score_rows = [
                json.loads(line)
                for line in (output_root / result.score_rows_path)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(score_rows), 2)
            self.assertEqual(
                {row["schema_version"] for row in score_rows},
                {COMPLETION_SCORE_SCHEMA_VERSION},
            )
            self.assertEqual({row["benchmark_id"] for row in score_rows}, {"humaneval"})
            self.assertEqual({row["split"] for row in score_rows}, {"test"})
            self.assertTrue(
                all(
                    {
                        "codelewm",
                        "no_action",
                        "shuffled_action",
                        "llm_order",
                        "random",
                        "lexical",
                    }.issubset(row["scores"])
                    for row in score_rows
                )
            )

    def test_evaluation_requires_scoring_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels_out = root / "labels"
            sampling = sample_execution_rerank_completions(
                benchmark="humaneval",
                source_path=FIXTURES / "humaneval_tiny.jsonl",
                out=labels_out,
                samples_per_problem=2,
                llm_seeds=(17,),
            )
            labels_path = labels_out / sampling.labels_path
            stripped_rows = []
            for line in labels_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                row.pop("scoring_inputs", None)
                stripped_rows.append(row)
            labels_path.write_text(
                "\n".join(json.dumps(row) for row in stripped_rows) + "\n",
                encoding="utf-8",
            )
            original_manifest = read_artifact_manifest(
                labels_out / sampling.artifact_manifest_path
            )
            refreshed_manifest = build_artifact_manifest(
                artifact_kind=original_manifest.artifact_kind,
                root=labels_out,
                files=tuple(labels_out / file.path for file in original_manifest.files),
                command=original_manifest.command,
                config={"fixture": "stripped_scoring_inputs"},
                metadata=original_manifest.metadata,
            )
            write_artifact_manifest(
                refreshed_manifest,
                labels_out / sampling.artifact_manifest_path,
            )
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")

            with self.assertRaises(ExecutionRerankEvalError):
                run_execution_rerank_evaluation(
                    completion_manifest=labels_out / sampling.artifact_manifest_path,
                    checkpoint=checkpoint,
                    out=root / "rerank",
                    allow_unsafe_checkpoint=True,
                )

    def test_evaluation_verifies_checkpoint_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels_out = root / "labels"
            sampling = sample_execution_rerank_completions(
                benchmark="humaneval",
                source_path=FIXTURES / "humaneval_tiny.jsonl",
                out=labels_out,
                samples_per_problem=2,
                llm_seeds=(17,),
            )
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")

            with self.assertRaises(ScoreError):
                run_execution_rerank_evaluation(
                    completion_manifest=labels_out / sampling.artifact_manifest_path,
                    checkpoint=checkpoint,
                    out=root / "rerank",
                    allow_unsafe_checkpoint=False,
                )

    def test_cli_runs_humaneval_completion_rerank_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels_out = root / "labels"
            sampling = sample_execution_rerank_completions(
                benchmark="humaneval",
                source_path=FIXTURES / "humaneval_tiny.jsonl",
                out=labels_out,
                samples_per_problem=2,
                llm_seeds=(17,),
            )
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "eval",
                    "rerank-humaneval",
                    "--completion-manifest",
                    str(labels_out / sampling.artifact_manifest_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(root / "rerank"),
                    "--allow-unsafe-checkpoint",
                    "--bootstrap-samples",
                    "32",
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
                payload["schema_version"], EXECUTION_RERANK_EVAL_RUN_SCHEMA_VERSION
            )
            self.assertEqual(payload["benchmark"], "humaneval")
            self.assertFalse(payload["claim_allowed"])
            self.assertTrue((root / "rerank" / payload["report_path"]).is_file())

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("einops") is not None,
        "torch scoring runtime is unavailable",
    )
    def test_cli_runs_humaneval_with_pass_head_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels_out = root / "labels"
            sampling = sample_execution_rerank_completions(
                benchmark="humaneval",
                source_path=FIXTURES / "humaneval_tiny.jsonl",
                out=labels_out,
                samples_per_problem=2,
                llm_seeds=(17,),
            )
            checkpoint = _write_pass_head_execution_checkpoint(root)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "eval",
                    "rerank-humaneval",
                    "--completion-manifest",
                    str(labels_out / sampling.artifact_manifest_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(root / "rerank"),
                    "--device",
                    "cpu",
                    "--require-learned-scorer",
                    "--bootstrap-samples",
                    "32",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            report = json.loads(
                (root / "rerank" / payload["report_path"]).read_text(
                    encoding="utf-8"
                )
            )
            score_rows = [
                json.loads(line)
                for line in (root / "rerank" / payload["score_rows_path"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(payload["schema_version"], EXECUTION_RERANK_EVAL_RUN_SCHEMA_VERSION)
        self.assertEqual(
            report["scoring_summary"]["model_id"],
            "codelewm.execution_torch_transition_scorer.v1",
        )
        self.assertEqual({row["benchmark_id"] for row in score_rows}, {"humaneval"})
        self.assertEqual({row["split"] for row in score_rows}, {"test"})
        self.assertTrue(all("codelewm" in row["scores"] for row in score_rows))


def _write_pass_head_execution_checkpoint(root: Path) -> Path:
    import torch

    from codelewm.harness.scorer import EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION
    from codelewm.model import (
        TorchCodeTransitionModelConfig,
        build_torch_transition_model,
        compute_config_hash,
    )
    from codelewm.model.checkpoint import (
        build_checkpoint_metadata,
        write_checkpoint_manifest,
    )
    from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE

    compatibility = {
        "wm": {
            "action_view": "text",
            "embed_dim": 256,
            "state_sequence_length": 1024,
            "action_sequence_length": 256,
            "action_fusion": "conditional_transformer",
            "enable_pass_head": True,
        },
        "objective": {
            "inverse_action_reconstruction_weight": 0.0,
            "p_pass_bce_weight": 0.5,
            "p_pass_bce_pos_weight": 1.0,
        },
        "loader": {"output_sequence_length": 256},
    }
    model = build_torch_transition_model(
        TorchCodeTransitionModelConfig(
            vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
            dropout=0.0,
            enable_pass_head=True,
        )
    )
    assert model.pass_head is not None
    with torch.no_grad():
        for param in model.pass_head.parameters():
            param.zero_()
        model.pass_head[-1].bias.fill_(2.5)
    checkpoint = root / "pass_head_last.pt"
    torch.save(
        {
            "schema_version": EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION,
            "step": 12,
            "model_state_dict": model.state_dict(),
            "compatibility_config": compatibility,
            "compatibility_config_hash": compute_config_hash(compatibility),
            "metrics": {"fixture": 1.0},
        },
        checkpoint,
    )
    write_checkpoint_manifest(
        metadata=build_checkpoint_metadata(
            compatibility,
            record_schema_version="codelewm.execution_pack_record.v2",
            action_view="text",
            model_class="TorchCodeTransitionModel",
        ),
        checkpoint_path=checkpoint,
        manifest_path=checkpoint.with_name(checkpoint.name + ".manifest.json"),
    )
    return checkpoint


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
