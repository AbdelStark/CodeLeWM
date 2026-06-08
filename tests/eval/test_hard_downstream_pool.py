from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from codelewm.eval import (
    LABEL_CONSTRUCTION_REPORT_SCHEMA_VERSION,
    DownstreamBenchmarkPackError,
    HardNegativePoolError,
    build_downstream_benchmark_pack,
    build_label_construction_report,
    generate_hard_negative_pool,
    read_downstream_rerank_benchmark,
)
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "config" / "benchmark" / "hard_downstream_fixtures"
GENERATED_FIXTURE = ROOT / "config" / "benchmark" / "hard_downstream_generated_pool_fixture.json"

BEFORE = "def accumulate(values):\n    total = 0\n    for value in values:\n        total += value\n    return total\n"
REFERENCE = "def accumulate(values):\n    total = 0\n    for value in values:\n        total = total + value\n    return total\n"


class HardNegativePoolGeneratorTest(unittest.TestCase):
    def test_pool_class_accounting_and_labels(self) -> None:
        pool = generate_hard_negative_pool(
            before_text=BEFORE, reference_after_text=REFERENCE, seed=17, pool_size=6
        )
        self.assertEqual(len(pool), 6)
        by_class = {c.hard_negative_class: c for c in pool}
        self.assertEqual(by_class["passing_reference"].label, "pass")
        self.assertEqual(by_class["no_action_bait"].label, "fail")
        self.assertEqual(by_class["near_no_action_bait"].label, "fail")
        # no-action bait reproduces the unchanged before-state.
        self.assertEqual(by_class["no_action_bait"].after_text, BEFORE)
        # at least two distinct *failing* hard-negative classes (dual coverage).
        failing_classes = {c.hard_negative_class for c in pool if c.label == "fail"}
        self.assertGreaterEqual(len(failing_classes), 2)

    def test_each_candidate_has_checksum_and_static_check(self) -> None:
        pool = generate_hard_negative_pool(
            before_text=BEFORE, reference_after_text=REFERENCE, seed=17, pool_size=6
        )
        checksums = [c.checksum for c in pool]
        self.assertTrue(all(checksums))
        self.assertEqual(len(set(checksums)), len(checksums))  # unique per candidate
        self.assertTrue(all(c.static_check in {"pass", "fail"} for c in pool))
        self.assertTrue(all(c.candidate_id.startswith("hn_") for c in pool))

    def test_generation_is_deterministic(self) -> None:
        a = generate_hard_negative_pool(
            before_text=BEFORE, reference_after_text=REFERENCE, seed=17, pool_size=8
        )
        b = generate_hard_negative_pool(
            before_text=BEFORE, reference_after_text=REFERENCE, seed=17, pool_size=8
        )
        self.assertEqual([c.checksum for c in a], [c.checksum for c in b])

    def test_pool_size_out_of_range_raises(self) -> None:
        with self.assertRaises(HardNegativePoolError):
            generate_hard_negative_pool(
                before_text=BEFORE, reference_after_text=REFERENCE, pool_size=3
            )
        with self.assertRaises(HardNegativePoolError):
            generate_hard_negative_pool(
                before_text=BEFORE, reference_after_text=REFERENCE, pool_size=99
            )

    def test_unparseable_reference_raises(self) -> None:
        with self.assertRaises(HardNegativePoolError):
            generate_hard_negative_pool(
                before_text=BEFORE, reference_after_text="def broken(:\n", pool_size=6
            )

    def test_label_construction_report_counts(self) -> None:
        pool = generate_hard_negative_pool(
            before_text=BEFORE, reference_after_text=REFERENCE, seed=17, pool_size=6
        )
        report = build_label_construction_report(pool, sandbox_used=False)
        self.assertEqual(report["schema_version"], LABEL_CONSTRUCTION_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["candidate_count"], 6)
        self.assertFalse(report["sandbox_used"])
        self.assertEqual(report["label_counts"]["pass"], 1)
        self.assertGreaterEqual(report["label_counts"]["fail"], 2)

    def test_generator_module_does_not_import_sandbox(self) -> None:
        source = (ROOT / "codelewm" / "eval" / "hard_negative_pool.py").read_text(encoding="utf-8")
        self.assertNotIn("codelewm.data.sandbox", source)


class HardNegativePackBuildTest(unittest.TestCase):
    def test_pack_build_generates_pool_and_label_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pack"
            result = build_downstream_benchmark_pack(config_path=GENERATED_FIXTURE, out=out)
            manifest = read_artifact_manifest(out / result.artifact_manifest_path)
            checked = validate_artifact_checksums(manifest, root=out)
            benchmark = read_downstream_rerank_benchmark(out / result.benchmark_path)
            label_report = json.loads(
                (out / result.label_construction_report_path).read_text(encoding="utf-8")
            )

        self.assertEqual(
            result.label_construction_report_path, "reports/label_construction_report.json"
        )
        self.assertEqual(
            manifest.metadata["label_construction_report"], "reports/label_construction_report.json"
        )
        self.assertIn("label_construction_report.json", {p.name for p in checked})
        self.assertEqual(label_report["candidate_count"], 6)

        candidates = benchmark.tasks[0].candidates
        self.assertEqual(len(candidates), 6)
        for candidate in candidates:
            self.assertIn("hard_negative_class", candidate.source)
            self.assertTrue(candidate.source["checksum"])
            self.assertEqual(candidate.source["generator"], "hard_negative_pool")
            self.assertIn("source_license_status", candidate.source)
        labels = {c.label for c in candidates}
        self.assertIn("pass", labels)

    def test_source_license_blocker_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            payload = json.loads(GENERATED_FIXTURE.read_text(encoding="utf-8"))
            payload["source_license_policy"]["publication_allowed"] = False
            bad.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DownstreamBenchmarkPackError):
                build_downstream_benchmark_pack(config_path=bad, out=Path(tmp) / "pack")

    def test_split_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Copy the referenced fixtures next to the temp config so relative
            # before/reference paths resolve.
            shutil.copytree(FIXTURE_DIR, Path(tmp) / "hard_downstream_fixtures")
            cfg = Path(tmp) / "leak.json"
            payload = json.loads(GENERATED_FIXTURE.read_text(encoding="utf-8"))
            payload["evaluation_only"] = False
            base_task = payload["tasks"][0]
            # Distinct task_ids (so materialization does not collide) but the
            # same repo_id across train and test -> repository split leakage.
            train_task = json.loads(json.dumps(base_task))
            train_task["task_id"] = "accumulator-train"
            train_task["split"] = "train"
            train_task["repo_id"] = "shared-repo"
            test_task = json.loads(json.dumps(base_task))
            test_task["task_id"] = "accumulator-test"
            test_task["split"] = "test"
            test_task["repo_id"] = "shared-repo"
            payload["tasks"] = [train_task, test_task]
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            out = Path(tmp) / "pack"
            result = build_downstream_benchmark_pack(config_path=cfg, out=out)
            split_report = json.loads(
                (out / result.split_leakage_report_path).read_text(encoding="utf-8")
            )
            readiness = json.loads(
                (out / result.readiness_report_path).read_text(encoding="utf-8")
            )
        self.assertFalse(split_report["ok"])
        self.assertTrue(split_report["leakage_findings"])
        self.assertIn("split_or_repository_leakage_detected", readiness["blocked_reasons"])
        self.assertFalse(result.scaled_evaluation_ready)


if __name__ == "__main__":
    unittest.main()
