from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs" / "benchmark" / "V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md"
PAPER_REPORT = (
    ROOT
    / "docs"
    / "benchmark"
    / "v1_0"
    / "paper_demo"
    / "reports"
    / "paper_demo_report.json"
)
README = ROOT / "README.md"
NEXT_GOAL = ROOT / "docs" / "roadmap" / "NEXT_GOAL_PROMPT.md"
FULL_COMPLETION = ROOT / "docs" / "roadmap" / "FULL_COMPLETION.md"
IMPLEMENTATION = ROOT / "docs" / "roadmap" / "IMPLEMENTATION.md"


def _pass_at_1(slice_report: dict[str, object], baseline: str) -> float:
    metrics = slice_report["metrics"]
    assert isinstance(metrics, dict)
    baselines = metrics["baselines"]
    assert isinstance(baselines, list)
    for row in baselines:
        assert isinstance(row, dict)
        if row["baseline"] == baseline:
            return float(row["pass_at_1"])
    raise AssertionError(f"missing baseline: {baseline}")


class V1FinalClaimAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = AUDIT.read_text(encoding="utf-8")
        self.report = json.loads(PAPER_REPORT.read_text(encoding="utf-8"))

    def test_audit_links_all_required_source_artifacts(self) -> None:
        self.assertTrue(AUDIT.is_file(), f"missing: {AUDIT}")
        for marker in (
            "docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md",
            "docs/benchmark/V0_6_RERANK_FULL_2026-06-01.md",
            "docs/benchmark/EXECUTION_V0_8_RESULTS_2026-06-05.md",
            "docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md",
            "docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md",
            "docs/benchmark/v1_0/paper_demo/reports/paper_demo_report.json",
            "docs/benchmark/v0_9/seed-42/p_pass_calibration/downstream_completion/reports/p_pass_calibration_report.json",
            "docs/benchmark/v0_9/seed-1729/p_pass_calibration/downstream_completion/reports/p_pass_calibration_report.json",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_final_demo_table_matches_committed_json(self) -> None:
        for slice_report in self.report["slices"]:
            seed = int(slice_report["seed"])
            benchmark = str(slice_report["benchmark_display"])
            codelewm = _pass_at_1(slice_report, "codelewm")
            no_action = _pass_at_1(slice_report, "no_action")
            llm_order = _pass_at_1(slice_report, "llm_order")
            lexical = _pass_at_1(slice_report, "lexical")
            metrics = slice_report["metrics"]
            assert isinstance(metrics, dict)
            deltas = metrics["deltas"]
            assert isinstance(deltas, dict)
            no_action_delta = deltas["codelewm_minus_no_action"]
            assert isinstance(no_action_delta, dict)
            lift = float(no_action_delta["pass_at_1_points"])
            ci = metrics["bootstrap_lift_over_no_action_ci"]
            assert isinstance(ci, list)
            gate = "open" if slice_report["claim_allowed"] else "closed"
            expected_row = (
                f"| v1.0 demo replay | {seed} | {benchmark} | {codelewm:.4f} | "
                f"{no_action:.4f} | {llm_order:.4f} | {lexical:.4f} | "
                f"{lift:+.2f} pts | [{float(ci[0]):.2f}, {float(ci[1]):.2f}] | {gate} |"
            )
            with self.subTest(seed=seed, benchmark=benchmark):
                self.assertIn(expected_row, self.text)

    def test_audit_blocks_unsupported_claims_and_preserves_narrow_positive(self) -> None:
        for marker in (
            "Claim verdict: CLOSED",
            "HumanEval WS-D narrow positive slice | ALLOWED",
            "MBPP-Plus/general downstream improvement | BLOCKED",
            "Broad coding improvement | BLOCKED",
            "Action-conditioned retrieval/model-quality claim | BLOCKED",
            "Semantic latent representation axes | BLOCKED",
            "`p_pass` score key in downstream rows | NOT RECORDED",
            "demo_report-e6fc06c328eed245",
            "claim_allowed=false",
            "0.263",
            "0.441",
            "+10.64 pts",
            "+8.51 pts",
            "1.0000 | 1.0000 | 0.1765 | 1.0000 | +0.00 pts",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_readme_and_roadmaps_point_at_audit_and_final_package(self) -> None:
        readme = README.read_text(encoding="utf-8")
        next_goal = NEXT_GOAL.read_text(encoding="utf-8")
        full_completion = FULL_COMPLETION.read_text(encoding="utf-8")
        normalized_full_completion = " ".join(full_completion.split())
        implementation = IMPLEMENTATION.read_text(encoding="utf-8")

        self.assertIn("docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md", readme)
        self.assertIn("#406 consolidated benchmark tables", readme)
        self.assertIn("#406 - complete", next_goal)
        self.assertIn("#407 - complete", next_goal)
        self.assertIn("#408 - complete", next_goal)
        self.assertIn("No active final release child issue remains.", next_goal)
        self.assertIn("docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md", next_goal)
        self.assertIn("docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md", next_goal)
        self.assertIn("Issues #402 through #408 are now complete", normalized_full_completion)
        self.assertIn("docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md", normalized_full_completion)
        self.assertIn(
            "| #406 | v1.0 results: consolidate benchmark tables and final claim audit | evaluation/results/docs | p1 | m | follow-up | Closed |",
            implementation,
        )


if __name__ == "__main__":
    unittest.main()
