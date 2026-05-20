from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from codelewm.eval import (
    ACTION_ABLATION_REPORT_SCHEMA_VERSION,
    build_action_ablation_report,
    build_action_view_report_policy,
    build_baseline_metrics,
    build_retrieval_report,
    read_action_ablation_report,
    run_action_ablation_suite,
    write_retrieval_report,
)
from codelewm.model import compute_config_hash
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)
from codelewm.training import TRAINING_RUN_MANIFEST_SCHEMA_VERSION, load_train_config


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "d" * 40
CREATED_AT = "2026-05-19T00:00:00Z"


@contextmanager
def _chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ActionAblationReportTest(unittest.TestCase):
    def test_report_marks_patch_diagnostic_and_blocks_missing_variants(self) -> None:
        retrieval_report = _retrieval_report()
        training_manifest = _training_manifest_payload()
        train_config = load_train_config(
            ROOT / "config/train/codelewm_tiny.yaml"
        ).to_dict()

        report = build_action_ablation_report(
            retrieval_report,
            retrieval_artifact_id="retrieval-artifact",
            retrieval_report_path="reports/retrieval_report.json",
            training_artifact_id="training-artifact",
            training_manifest=training_manifest,
            train_config=train_config,
        )
        rows = {row.name: row for row in report.rows}

        self.assertEqual(report.schema_version, ACTION_ABLATION_REPORT_SCHEMA_VERSION)
        self.assertEqual(rows["text_action"].status, "completed")
        self.assertEqual(rows["random"].status, "completed")
        self.assertEqual(rows["no_action"].status, "completed")
        self.assertEqual(rows["shuffled_action"].status, "completed")
        self.assertEqual(rows["abstract_action"].status, "blocked")
        self.assertEqual(rows["retrieval_loss_disabled"].status, "completed")
        self.assertEqual(rows["retrieval_loss_enabled"].status, "blocked")
        self.assertEqual(rows["collapse_sigreg_0.09"].status, "completed")
        self.assertEqual(rows["patch_action_diagnostic"].status, "blocked")
        self.assertTrue(
            rows["patch_action_diagnostic"].action_view_policy["diagnostic_upper_bound"]
        )
        self.assertEqual(
            rows["patch_action_diagnostic"].action_view_policy["report_scope"],
            "diagnostic",
        )
        self.assertEqual(
            rows["patch_action_diagnostic"].action_view_policy["action_view"], "patch"
        )
        self.assertIsNotNone(report.claim_gate)
        assert report.claim_gate is not None
        self.assertFalse(report.claim_gate.claim_allowed)
        self.assertIn(
            "blocked_action_view_row:abstract_action",
            report.claim_gate.failure_reasons,
        )
        self.assertIn(
            "blocked_action_view_row:patch_action_diagnostic",
            report.claim_gate.failure_reasons,
        )

        round_tripped = read_action_ablation_report(_write_report_tmp(report.to_dict()))
        self.assertEqual(round_tripped.to_dict(), report.to_dict())


class ActionAblationRunnerTest(unittest.TestCase):
    def test_runner_writes_manifested_report_with_parent_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retrieval_manifest = _write_retrieval_artifact(root)
            training_manifest = _write_training_artifact(root)
            out = root / "ablation"

            result = run_action_ablation_suite(
                retrieval_artifact=retrieval_manifest,
                training_artifact=training_manifest,
                out=out,
                command=("codelewm", "eval", "ablation"),
            )
            artifact_manifest = read_artifact_manifest(out / "manifest.json")
            validate_artifact_checksums(artifact_manifest, root=out)
            report = read_action_ablation_report(out / result.report_path)

        self.assertEqual(
            result.parent_artifacts, ("retrieval-artifact", "training-artifact")
        )
        self.assertEqual(
            artifact_manifest.parent_artifacts,
            ("retrieval-artifact", "training-artifact"),
        )
        self.assertEqual(report.completed_count, 7)
        self.assertGreaterEqual(report.blocked_count, 4)
        self.assertFalse(report.claim_gate.claim_allowed)

    def test_runner_accepts_relative_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retrieval_manifest = _write_retrieval_artifact(root)
            training_manifest = _write_training_artifact(root)

            with _chdir(root):
                result = run_action_ablation_suite(
                    retrieval_artifact=retrieval_manifest.relative_to(root),
                    training_artifact=training_manifest.relative_to(root),
                    out="ablation-relative",
                    command=("codelewm", "eval", "ablation"),
                )
                artifact_manifest = read_artifact_manifest(
                    root / "ablation-relative" / "manifest.json"
                )
                checked_files = validate_artifact_checksums(
                    artifact_manifest,
                    root=root / "ablation-relative",
                )

        self.assertEqual(result.report_path, "reports/action_view_ablation_report.json")
        self.assertEqual(
            {path.name for path in checked_files},
            {"action_view_ablation_report.json"},
        )

    def test_cli_writes_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retrieval_manifest = _write_retrieval_artifact(root)
            training_manifest = _write_training_artifact(root)
            out = root / "cli-ablation"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "eval",
                    "ablation",
                    "--retrieval-artifact",
                    str(retrieval_manifest),
                    "--training-artifact",
                    str(training_manifest),
                    "--out",
                    str(out),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            payload["schema_version"], "codelewm.eval.action_ablation_run.v1"
        )
        self.assertEqual(
            payload["parent_artifacts"], ["retrieval-artifact", "training-artifact"]
        )
        self.assertGreaterEqual(payload["blocked"], 4)


def _retrieval_report():
    return build_retrieval_report(
        (1, 2),
        candidate_counts=(3, 3),
        baselines=build_baseline_metrics(
            {
                "random": (2, 3),
                "lexical": (1, 3),
                "no_action": (2, 2),
                "shuffled_action": (3, 2),
            },
            candidate_counts={
                "random": (3, 3),
                "lexical": (3, 3),
                "no_action": (3, 3),
                "shuffled_action": (3, 3),
            },
        ),
        metadata={
            "action_view_policy": build_action_view_report_policy(
                "text", report_scope="headline"
            ).to_dict()
        },
    )


def _training_manifest_payload() -> dict:
    return {
        "schema_version": TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": "fixture-train",
        "config_sha256": "1" * 64,
        "artifact_manifest_id": "training-artifact",
        "artifact_manifest_path": "manifest.json",
        "parent_artifacts": ["dataset-artifact"],
        "dataset_manifest_path": "pack/manifest.json",
        "config_path": "config.json",
        "metrics_path": "metrics.jsonl",
        "metrics_report_path": "reports/metrics_report.json",
        "checkpoint_files": [
            {"path": "checkpoints/checkpoint.pt", "sha256": "2" * 64, "bytes": 1}
        ],
        "report_files": [],
        "final_metrics": {
            "loss/total": 0.5,
            "collapse/effective_rank": 4.0,
            "collapse/effective_rank_ratio": 0.25,
            "collapse/per_dim_variance_min": 0.01,
            "collapse/per_dim_variance_median": 0.05,
            "collapse/nearest_neighbor_entropy": 1.0,
        },
        "step_count": 1,
        "seed": 1337,
        "metadata": {},
    }


def _write_retrieval_artifact(root: Path) -> Path:
    artifact_root = root / "retrieval"
    report_path = artifact_root / "reports" / "retrieval_report.json"
    write_retrieval_report(_retrieval_report(), report_path)
    manifest = build_artifact_manifest(
        artifact_kind="eval_report",
        root=artifact_root,
        files=(report_path,),
        command=("codelewm", "eval", "retrieval"),
        config={"fixture": "retrieval"},
        source_git_sha=SOURCE_SHA,
        created_at=CREATED_AT,
        artifact_id="retrieval-artifact",
        parent_artifacts=("training-artifact", "dataset-artifact"),
    )
    write_artifact_manifest(manifest, artifact_root / "manifest.json")
    return artifact_root / "manifest.json"


def _write_training_artifact(root: Path) -> Path:
    artifact_root = root / "train"
    artifact_root.mkdir(parents=True)
    config = load_train_config(ROOT / "config/train/codelewm_tiny.yaml").to_dict()
    (artifact_root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = _training_manifest_payload()
    payload["config_sha256"] = compute_config_hash(config)
    (artifact_root / "training_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = build_artifact_manifest(
        artifact_kind="training_run",
        root=artifact_root,
        files=(artifact_root / "config.json", artifact_root / "training_manifest.json"),
        command=("codelewm", "train"),
        config=config,
        source_git_sha=SOURCE_SHA,
        created_at=CREATED_AT,
        artifact_id="training-artifact",
        parent_artifacts=("dataset-artifact",),
    )
    write_artifact_manifest(manifest, artifact_root / "manifest.json")
    return artifact_root / "manifest.json"


def _write_report_tmp(payload: dict) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".json", delete=False
    )
    with handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
