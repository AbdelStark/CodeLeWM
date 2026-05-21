from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from codelewm.data import DATASET_SCHEMA_VERSION, DatasetManifest
from codelewm.eval import RETRIEVAL_REPORT_SCHEMA_VERSION, build_retrieval_report
from codelewm.harness import (
    ERROR_REPORT_SCHEMA_VERSION,
    RERANK_RESULT_SCHEMA_VERSION,
    SCORE_RESULT_SCHEMA_VERSION,
    RerankResult,
    ScoreResult,
    error_report_json_schema,
    rerank_result_json_schema,
    score_result_json_schema,
)
from codelewm.model.inspection import MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION
from codelewm.observability import ManifestFile
from codelewm.training import TRAINING_RUN_MANIFEST_SCHEMA_VERSION, TrainingRunManifest


ROOT = Path(__file__).resolve().parents[2]


class PublicCliContractTest(unittest.TestCase):
    def test_root_help_snapshot_exposes_stable_command_surface(self) -> None:
        help_text = _run_help("--help")

        self.assertIn("usage: codelewm", help_text)
        self.assertIn("CodeLeWM command-line interface", help_text)
        self.assertIn("score", help_text)
        self.assertIn("rerank", help_text)
        self.assertIn("train", help_text)
        self.assertIn("model", help_text)
        self.assertIn("eval", help_text)
        self.assertIn("index", help_text)
        self.assertIn("dataset", help_text)
        self.assertIn("secret-scan", help_text)
        self.assertIn("llm-demo-tui", help_text)

    def test_score_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("score", "--help")

        for flag in (
            "--before",
            "--instruction",
            "--candidate",
            "--checkpoint",
            "--device",
            "--index",
            "--retrieval-prior-weight",
            "--retrieval-prior-k",
            "--json",
            "--log-jsonl",
            "--allow-unsafe-checkpoint",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_rerank_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("rerank", "--help")

        for flag in (
            "--before",
            "--instruction",
            "--candidates",
            "--checkpoint",
            "--device",
            "--index",
            "--retrieval-prior-weight",
            "--retrieval-prior-k",
            "--json",
            "--log-jsonl",
            "--allow-unsafe-checkpoint",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_dataset_build_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("dataset", "build", "--help")

        for flag in ("--config", "--out", "--json"):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_llm_demo_tui_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("llm-demo-tui", "--help")

        for flag in ("--view-model", "--demo-dir", "--snapshot-json", "--json"):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_llm_demo_help_snapshot_exposes_diagnostic_link_flags(self) -> None:
        help_text = _run_help("llm-demo", "--help")

        for flag in (
            "--checkpoint-inspection-manifest",
            "--checkpoint-inspection-report",
            "--latent-matrix-manifest",
            "--latent-matrix-report",
            "--tensorboard-manifest",
            "--tensorboard-export",
            "--tui",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_train_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("train", "--help")

        for flag in (
            "--config",
            "--out",
            "--device",
            "--executor",
            "--resume-from",
            "--tensorboard",
            "--tensorboard-dir",
            "--overwrite",
            "--json",
            "--log-jsonl",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_model_inspect_checkpoint_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("model", "inspect-checkpoint", "--help")

        for flag in (
            "--checkpoint",
            "--checkpoint-manifest",
            "--out",
            "--parent-manifest",
            "--histogram-bins",
            "--max-histogram-tensors",
            "--max-histogram-values",
            "--allow-unsafe-checkpoint",
            "--overwrite",
            "--json",
            "--log-jsonl",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_eval_retrieval_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("eval", "retrieval", "--help")

        for flag in (
            "--checkpoint",
            "--data",
            "--out",
            "--device",
            "--max-candidates",
            "--hard-negatives",
            "--seed",
            "--report-scope",
            "--overwrite",
            "--json",
            "--log-jsonl",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_eval_latent_probe_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("eval", "latent-probe", "--help")

        for flag in (
            "--checkpoint",
            "--data",
            "--out",
            "--device",
            "--max-examples-per-split",
            "--bootstrap-samples",
            "--seed",
            "--overwrite",
            "--json",
            "--log-jsonl",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_eval_latent_matrix_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("eval", "latent-matrix", "--help")

        for flag in (
            "--checkpoint",
            "--data",
            "--out",
            "--device",
            "--max-examples-per-split",
            "--matrix-dimension-limit",
            "--top-dimensions",
            "--max-pairwise-rows",
            "--latent-probe-report",
            "--seed",
            "--overwrite",
            "--json",
            "--log-jsonl",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_eval_surprise_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("eval", "surprise", "--help")

        for flag in (
            "--checkpoint",
            "--data",
            "--out",
            "--device",
            "--max-examples",
            "--random-decoys",
            "--same-file-decoys",
            "--mutation-decoys",
            "--action-cluster-decoys",
            "--seed",
            "--overwrite",
            "--json",
            "--log-jsonl",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_eval_ablation_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("eval", "ablation", "--help")

        for flag in (
            "--retrieval-artifact",
            "--training-artifact",
            "--out",
            "--overwrite",
            "--json",
            "--log-jsonl",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_eval_scorer_quality_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("eval", "scorer-quality", "--help")

        for flag in (
            "--config",
            "--checkpoint",
            "--out",
            "--device",
            "--index",
            "--retrieval-prior-weight",
            "--retrieval-prior-k",
            "--parent-manifest",
            "--overwrite",
            "--json",
            "--log-jsonl",
            "--allow-unsafe-checkpoint",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_index_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("index", "--help")

        for flag in (
            "--checkpoint",
            "--data",
            "--out",
            "--device",
            "--distance",
            "--name",
            "--overwrite",
            "--json",
            "--log-jsonl",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_dataset_pack_help_snapshot_exposes_required_flags(self) -> None:
        help_text = _run_help("dataset", "pack", "--help")

        for flag in ("--manifest", "--out", "--json"):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_invalid_cli_combinations_exit_with_argparse_error(self) -> None:
        cases = (
            ("score", "--instruction", "change"),
            (
                "score",
                "--before",
                "before.py",
                "--instruction",
                "change",
                "--candidate",
                "after.py",
                "--checkpoint",
                "ckpt",
                "--device",
                "tpu",
            ),
            (
                "rerank",
                "--before",
                "before.py",
                "--instruction",
                "change",
                "--checkpoint",
                "ckpt",
            ),
            ("train", "--config", "missing.json", "--device", "tpu"),
            (
                "eval",
                "retrieval",
                "--checkpoint",
                "ckpt",
                "--data",
                "pack",
                "--out",
                "out",
                "--device",
                "tpu",
            ),
            (
                "eval",
                "surprise",
                "--checkpoint",
                "ckpt",
                "--data",
                "pack",
                "--out",
                "out",
                "--device",
                "tpu",
            ),
            (
                "eval",
                "latent-probe",
                "--checkpoint",
                "ckpt",
                "--data",
                "pack",
                "--out",
                "out",
                "--device",
                "tpu",
            ),
            (
                "eval",
                "latent-matrix",
                "--checkpoint",
                "ckpt",
                "--data",
                "pack",
                "--out",
                "out",
                "--device",
                "tpu",
            ),
            (
                "eval",
                "scorer-quality",
                "--config",
                "quality.json",
                "--checkpoint",
                "ckpt",
                "--out",
                "out",
                "--device",
                "tpu",
            ),
            (
                "index",
                "--checkpoint",
                "ckpt",
                "--data",
                "pack",
                "--out",
                "out",
                "--device",
                "tpu",
            ),
            ("model", "inspect-checkpoint", "--checkpoint", "ckpt"),
        )

        for argv in cases:
            with self.subTest(argv=argv):
                completed = _run_cli(*argv)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("usage: codelewm", completed.stderr)

    def test_json_contracts_include_schema_versions_for_public_outputs(self) -> None:
        payloads = {
            "dataset": _dataset_manifest_payload(),
            "train": _training_manifest_payload(),
            "eval": build_retrieval_report((1,), candidate_counts=(1,)).to_dict(),
            "model": {"schema_version": MODEL_CHECKPOINT_INSPECTION_SCHEMA_VERSION},
            "score": _score_result_payload(),
            "rerank": _rerank_result_payload(),
        }

        self.assertEqual(payloads["dataset"]["schema_version"], DATASET_SCHEMA_VERSION)
        self.assertEqual(
            payloads["train"]["schema_version"], TRAINING_RUN_MANIFEST_SCHEMA_VERSION
        )
        self.assertEqual(
            payloads["eval"]["schema_version"], RETRIEVAL_REPORT_SCHEMA_VERSION
        )
        self.assertEqual(
            payloads["score"]["schema_version"], SCORE_RESULT_SCHEMA_VERSION
        )
        self.assertEqual(
            payloads["rerank"]["schema_version"], RERANK_RESULT_SCHEMA_VERSION
        )
        json.dumps(payloads, sort_keys=True, allow_nan=False)

    def test_harness_json_schemas_pin_schema_version_fields(self) -> None:
        schemas = {
            "score": score_result_json_schema(),
            "error": error_report_json_schema(),
            "rerank": rerank_result_json_schema(),
        }

        self.assertEqual(
            schemas["score"]["properties"]["schema_version"]["const"],
            SCORE_RESULT_SCHEMA_VERSION,
        )
        self.assertEqual(
            schemas["error"]["properties"]["schema_version"]["const"],
            ERROR_REPORT_SCHEMA_VERSION,
        )
        self.assertEqual(
            schemas["rerank"]["properties"]["schema_version"]["const"],
            RERANK_RESULT_SCHEMA_VERSION,
        )
        self.assertIn(
            "evaluation_gate_error",
            schemas["error"]["properties"]["error_type"]["enum"],
        )
        for name, schema in schemas.items():
            with self.subTest(schema=name):
                self.assertIn("schema_version", schema["required"])
                json.dumps(schema, sort_keys=True, allow_nan=False)


def _run_help(*args: str) -> str:
    completed = _run_cli(*args)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return completed.stdout


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "codelewm.harness.cli", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _dataset_manifest_payload() -> dict[str, object]:
    return DatasetManifest(
        schema_version=DATASET_SCHEMA_VERSION,
        row_count=0,
        features={"action_patch": False},
        artifacts=(),
        split_counts={"train": 0, "val": 0, "test": 0},
        source_counts={},
    ).to_dict()


def _training_manifest_payload() -> dict[str, object]:
    manifest_file = ManifestFile(
        path="checkpoints/checkpoint.state", sha256="0" * 64, bytes=0
    )
    return TrainingRunManifest(
        run_id="run-1",
        config_sha256="1" * 64,
        artifact_manifest_id="training-run-1",
        artifact_manifest_path="manifest.json",
        parent_artifacts=("dataset-1",),
        dataset_manifest_path="data/manifest.json",
        config_path="config.json",
        metrics_path="metrics.jsonl",
        metrics_report_path="reports/metrics.json",
        checkpoint_files=(manifest_file,),
        report_files=(),
        final_metrics={"loss/total": 0.1},
        step_count=1,
        seed=7,
    ).to_dict()


def _score_result_payload() -> dict[str, object]:
    return ScoreResult(
        candidate="candidate.py",
        transition_energy=0.5,
        final_score=0.5,
        model_id="fixture",
        checkpoint_sha256="2" * 64,
        input_digest="3" * 64,
    ).to_dict()


def _rerank_result_payload() -> dict[str, object]:
    return RerankResult(
        results=(ScoreResult.from_dict(_score_result_payload()),)
    ).to_dict()


if __name__ == "__main__":
    unittest.main()
