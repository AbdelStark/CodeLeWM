from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from codelewm.harness import (
    SCORER_QUALITY_REPORT_SCHEMA_VERSION,
    SCORER_QUALITY_RUN_SCHEMA_VERSION,
    TransitionIndexEntry,
    build_transition_index,
    read_scorer_quality_report,
    run_scorer_quality_evaluation,
    write_transition_index,
)
from codelewm.harness.scorer import _hashed_vector
from codelewm.observability import (
    build_artifact_manifest,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_artifact_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ScorerQualityTest(unittest.TestCase):
    def test_runner_writes_manifested_quality_report_with_failures_and_caveats(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_quality_fixture(root)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            parent_file = root / "parent.txt"
            parent_file.write_text("parent artifact", encoding="utf-8")
            parent_manifest = build_artifact_manifest(
                artifact_kind="training_run",
                root=root,
                files=(parent_file,),
                command=("fixture",),
                config={"fixture": True},
            )
            parent_manifest_path = root / "parent-manifest.json"
            write_artifact_manifest(parent_manifest, parent_manifest_path)
            out = root / "quality"

            result = run_scorer_quality_evaluation(
                config=config,
                checkpoint=checkpoint,
                out=out,
                parent_manifests=(parent_manifest_path,),
                allow_unsafe_checkpoint=True,
                command=("codelewm", "eval", "scorer-quality"),
            )
            manifest = read_artifact_manifest(out / "manifest.json")
            checked_files = validate_artifact_checksums(manifest, root=out)
            report = read_scorer_quality_report(out / result.report_path)
            marker_executed = (root / "executed.txt").exists()

        self.assertEqual(result.schema_version, SCORER_QUALITY_RUN_SCHEMA_VERSION)
        self.assertEqual(report["schema_version"], SCORER_QUALITY_REPORT_SCHEMA_VERSION)
        self.assertEqual(manifest.artifact_kind, "score_report")
        self.assertEqual(result.parent_artifacts, (parent_manifest.artifact_id,))
        self.assertEqual(manifest.parent_artifacts, (parent_manifest.artifact_id,))
        self.assertEqual(
            {path.name for path in checked_files},
            {"config.json", "scorer_quality_report.json"},
        )
        self.assertEqual(report["summary"]["example_count"], 1)
        self.assertEqual(report["summary"]["candidate_count"], 4)
        self.assertEqual(report["summary"]["error_count"], 2)
        self.assertEqual(report["summary"]["failure_counts"]["invalid_syntax"], 1)
        self.assertEqual(report["summary"]["failure_counts"]["patch_apply_failed"], 1)
        self.assertIn("syntax_failure", report["calibration_slices"])
        self.assertIn("patch_failure", report["calibration_slices"])
        self.assertEqual(
            report["scoring_policy"]["risk_penalty"].split(";")[0], "reserved"
        )
        self.assertEqual(
            report["scoring_policy"]["execution_policy"],
            "candidate code is parsed and diff-applied as text but never executed",
        )
        self.assertFalse(marker_executed)

    def test_runner_records_retrieval_prior_distribution_when_index_is_supplied(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_quality_fixture(root)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            index_root = root / "index"
            true_text = (root / "candidates" / "true_after.py").read_text(
                encoding="utf-8"
            )
            index = build_transition_index(
                name="quality-prior-fixture",
                entries=(TransitionIndexEntry(transition_id="t-0", split="train"),),
                vectors=np.asarray(
                    [_hashed_vector(true_text, dim=8)], dtype=np.float32
                ),
            )
            write_transition_index(index, index_root)

            result = run_scorer_quality_evaluation(
                config=config,
                checkpoint=checkpoint,
                out=root / "quality",
                index=index_root,
                retrieval_prior_weight=2.0,
                retrieval_prior_k=1,
                allow_unsafe_checkpoint=True,
                command=("codelewm", "eval", "scorer-quality"),
            )
            report = read_scorer_quality_report(root / "quality" / result.report_path)

        self.assertEqual(report["scoring_policy"]["retrieval_prior_weight"], 2.0)
        self.assertGreater(report["score_distributions"]["retrieval_prior"]["count"], 0)
        scored = [
            row
            for row in report["examples"][0]["candidate_rows"]
            if row["status"] == "scored"
        ]
        self.assertTrue(all(row["retrieval_prior"] is not None for row in scored))

    def test_runner_accepts_relative_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_quality_fixture(root)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")

            with _chdir(root):
                result = run_scorer_quality_evaluation(
                    config=config.relative_to(root),
                    checkpoint=checkpoint.relative_to(root),
                    out="quality-relative",
                    allow_unsafe_checkpoint=True,
                    command=("codelewm", "eval", "scorer-quality"),
                )
                manifest = read_artifact_manifest(root / "quality-relative" / "manifest.json")
                checked_files = validate_artifact_checksums(
                    manifest,
                    root=root / "quality-relative",
                )

        self.assertEqual(result.report_path, "reports/scorer_quality_report.json")
        self.assertEqual(
            {path.name for path in checked_files},
            {"config.json", "scorer_quality_report.json"},
        )

    def test_cli_writes_json_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_quality_fixture(root)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            out = root / "cli-quality"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "eval",
                    "scorer-quality",
                    "--config",
                    str(config),
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(out),
                    "--allow-unsafe-checkpoint",
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(completed.stdout)
            report_exists = (out / "reports" / "scorer_quality_report.json").is_file()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["schema_version"], SCORER_QUALITY_RUN_SCHEMA_VERSION)
        self.assertTrue(report_exists)


def _write_quality_fixture(root: Path) -> Path:
    before = root / "before.py"
    candidates = root / "candidates"
    marker = root / "executed.txt"
    candidates.mkdir()
    before.write_text("def value():\n    return 1\n", encoding="utf-8")
    (candidates / "true_after.py").write_text(
        "def value():\n    return 2\n", encoding="utf-8"
    )
    (candidates / "hard_negative.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "def other():\n"
        "    return 100\n",
        encoding="utf-8",
    )
    (candidates / "syntax_failure.py").write_text(
        "def broken(:\n    return 1\n", encoding="utf-8"
    )
    (candidates / "patch_failure.patch").write_text(
        "--- before.py\n+++ after.py\n@@ -1 +1 @@\n-missing = 1\n+value = 3\n",
        encoding="utf-8",
    )
    config = {
        "schema_version": "codelewm.harness.scorer_quality_config.v1",
        "examples": [
            {
                "id": "fixture-quality",
                "before": str(before),
                "instruction": "change the returned value",
                "candidates_dir": str(candidates),
                "true_candidate": "true_after.py",
                "candidate_kinds": {
                    "true_after.py": "true_after",
                    "hard_negative.py": "hard_negative",
                    "syntax_failure.py": "syntax_failure",
                    "patch_failure.patch": "patch_failure",
                },
            }
        ],
    }
    config_path = root / "quality.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config_path


if __name__ == "__main__":
    unittest.main()
