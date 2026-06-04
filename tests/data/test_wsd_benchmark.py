from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.data.execution_rerank_sampler import (
    COMPLETION_LABEL_SCHEMA_VERSION,
    build_mutation_rerank_pack,
)
from codelewm.eval import load_completion_labels
from codelewm.observability import read_artifact_manifest, validate_artifact_checksums

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "data" / "execution_sources" / "fixtures"


class WsdBenchmarkBuilderTest(unittest.TestCase):
    def test_builds_unsaturated_mixed_pool_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "wsd"
            pool_size = 4
            result = build_mutation_rerank_pack(
                benchmark="humaneval",
                source_path=FIXTURES / "humaneval_tiny.jsonl",
                out=out,
                mutants_per_problem=10,
                pool_size=pool_size,
                seed=17,
                overwrite=True,
            )
            self.assertGreaterEqual(result.problem_count, 1)
            # fixed-size pools: every problem has exactly pool_size candidates
            self.assertEqual(result.completion_count, result.problem_count * pool_size)

            labels_path = out / f"{result.benchmark_id}_completion_labels.jsonl"
            rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
            # exactly one passing candidate (the reference) per problem
            by_problem: dict[str, list[dict]] = {}
            for r in rows:
                by_problem.setdefault(r["problem_id"], []).append(r)
            for pid, pool in by_problem.items():
                self.assertEqual(len(pool), pool_size, pid)
                self.assertEqual(sum(1 for r in pool if r["passed"]), 1, pid)
            # completion_label.v1 contract
            self.assertTrue(
                all(r["schema_version"] == COMPLETION_LABEL_SCHEMA_VERSION for r in rows)
            )
            for field in ("problem_id", "code", "passed", "valid_candidate",
                          "llm_order_rank", "test_results", "scoring_inputs"):
                self.assertTrue(all(field in r for r in rows), field)
            # the whole point of WS-D: a real pass/fail mix exists
            self.assertEqual({r["label"] for r in rows}, {"pass", "fail"})

            # every kept problem has a pass/fail mix (rerank headroom)
            report = json.loads((out / result.report_path).read_text())
            self.assertEqual(report["generator"], "wsd_mutation")
            self.assertEqual(report["mixed_problem_rate"], 1.0)
            # far from saturated (a frontier LLM pack sits ~0.95 pass rate)
            self.assertLess(report["test_pass_rate"], 0.9)

            # the existing rerank consumer can load it
            loaded = load_completion_labels(labels_path, benchmark_id=result.benchmark_id)
            self.assertTrue(any(label.passed for label in loaded))
            self.assertTrue(any(not label.passed for label in loaded))

            # manifest verifies
            manifest = read_artifact_manifest(out / result.artifact_manifest_path)
            validate_artifact_checksums(manifest, root=out)

    def test_llm_order_is_not_reference_first(self) -> None:
        # The reference (candidate 0) must not always be llm_order rank 0, else
        # the "llm_order" baseline would trivially win and the gate would be a lie.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "wsd"
            result = build_mutation_rerank_pack(
                benchmark="humaneval",
                source_path=FIXTURES / "humaneval_tiny.jsonl",
                out=out,
                mutants_per_problem=10,
                pool_size=4,
                seed=17,
                overwrite=True,
            )
            labels_path = out / f"{result.benchmark_id}_completion_labels.jsonl"
            rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
            # the reference (the one passing candidate) must not always be
            # llm_order rank 0, else the "llm_order" baseline trivially wins.
            ref_ranks = [r["llm_order_rank"] for r in rows if r["wsd_mutation"] == "reference"]
            self.assertTrue(ref_ranks)
            self.assertFalse(all(rank == 0 for rank in ref_ranks))


if __name__ == "__main__":
    unittest.main()
