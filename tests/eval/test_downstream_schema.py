from __future__ import annotations

import unittest

from codelewm.eval import (
    DOWNSTREAM_MIN_LABELED_EXAMPLES,
    DOWNSTREAM_REQUIRED_BASELINES,
    DOWNSTREAM_REQUIRED_METRICS,
    DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION,
    DOWNSTREAM_RERANK_CLAIM_GATE_SCHEMA_VERSION,
    DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION,
    DownstreamBenchmarkError,
    DownstreamCandidate,
    DownstreamRerankBenchmark,
    DownstreamTask,
    build_downstream_rerank_claim_gate,
    downstream_rerank_benchmark_json_schema,
    downstream_rerank_report_template,
    validate_downstream_rerank_claim_gate,
)


class DownstreamBenchmarkSchemaTest(unittest.TestCase):
    def test_benchmark_schema_records_required_task_candidate_and_gate_fields(self) -> None:
        benchmark = DownstreamRerankBenchmark(
            benchmark_id="fixture",
            tasks=(
                DownstreamTask(
                    task_id="task-1",
                    task_type="bugfix",
                    prompt="fix the value",
                    before_path="tasks/task-1/before.py",
                    candidates=(
                        DownstreamCandidate(
                            candidate_id="candidate_001",
                            llm_rank=1,
                            label="pass",
                            patch_path="tasks/task-1/candidates/candidate_001.patch",
                            static_check="pass",
                            test_check="not_run",
                            source={"model": "anthropic/claude-4.5-sonnet"},
                            provenance={"candidate_pack": "candidate-pack-id"},
                        ),
                        DownstreamCandidate(
                            candidate_id="candidate_002",
                            llm_rank=2,
                            label="fail",
                            patch_path="tasks/task-1/candidates/candidate_002.patch",
                            static_check="pass",
                            test_check="fail",
                        ),
                    ),
                    provenance={"repo": "fixture"},
                ),
            ),
        )
        payload = benchmark.to_dict()
        schema = downstream_rerank_benchmark_json_schema()

        self.assertEqual(payload["schema_version"], DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION)
        self.assertEqual(payload["min_labeled_examples"], DOWNSTREAM_MIN_LABELED_EXAMPLES)
        self.assertEqual(tuple(payload["required_baselines"]), DOWNSTREAM_REQUIRED_BASELINES)
        self.assertEqual(tuple(payload["required_metrics"]), DOWNSTREAM_REQUIRED_METRICS)
        self.assertEqual(schema["properties"]["schema_version"]["const"], DOWNSTREAM_RERANK_BENCHMARK_SCHEMA_VERSION)
        self.assertIn("tasks", schema["required"])

    def test_benchmark_rejects_missing_required_baselines_and_too_small_gate(self) -> None:
        task = DownstreamTask(
            task_id="task-1",
            task_type="bugfix",
            prompt="fix the value",
            before_path="before.py",
            candidates=(
                DownstreamCandidate(candidate_id="a", llm_rank=1, label="pass", patch_path="a.patch"),
                DownstreamCandidate(candidate_id="b", llm_rank=2, label="fail", patch_path="b.patch"),
            ),
        )

        with self.assertRaisesRegex(DownstreamBenchmarkError, "required_baselines"):
            DownstreamRerankBenchmark(
                benchmark_id="bad-baselines",
                tasks=(task,),
                required_baselines=("llm_order",),
            )
        with self.assertRaisesRegex(DownstreamBenchmarkError, "at least 100"):
            DownstreamRerankBenchmark(
                benchmark_id="bad-size",
                tasks=(task,),
                min_labeled_examples=99,
            )

    def test_claim_gate_blocks_small_or_non_improving_reports(self) -> None:
        blocked = build_downstream_rerank_claim_gate(
            example_count=1,
            metrics={
                "codelewm": {"pass_at_1": 0.4, "mrr": 0.5},
                "llm_order": {"pass_at_1": 0.5, "mrr": 0.5},
                "no_action": {"pass_at_1": 0.3, "mrr": 0.4},
            },
        )

        self.assertEqual(blocked["schema_version"], DOWNSTREAM_RERANK_CLAIM_GATE_SCHEMA_VERSION)
        self.assertFalse(blocked["allowed"])
        self.assertIn("example_count_below_minimum:1<100", blocked["failure_reasons"])
        self.assertTrue(any(reason.startswith("not_strictly_above:llm_order") for reason in blocked["failure_reasons"]))
        validate_downstream_rerank_claim_gate(blocked)

    def test_claim_gate_allows_only_strict_scaled_baseline_improvement(self) -> None:
        allowed = build_downstream_rerank_claim_gate(
            example_count=100,
            metrics={
                "codelewm": {"pass_at_1": 0.61, "mrr": 0.72},
                "llm_order": {"pass_at_1": 0.60, "mrr": 0.70},
                "no_action": {"pass_at_1": 0.40, "mrr": 0.50},
            },
        )

        self.assertTrue(allowed["allowed"])
        self.assertEqual(allowed["failure_reasons"], [])

    def test_report_template_names_falsification_criteria(self) -> None:
        template = downstream_rerank_report_template()

        self.assertEqual(template["schema_version"], DOWNSTREAM_RERANK_REPORT_SCHEMA_VERSION)
        self.assertFalse(template["claim_gate"]["allowed"])
        self.assertIn("would_falsify", template["falsification"])
        self.assertIn("pass_at_1", template["metrics"]["codelewm"])
        self.assertIn("llm_order", template["metrics"])


if __name__ == "__main__":
    unittest.main()
