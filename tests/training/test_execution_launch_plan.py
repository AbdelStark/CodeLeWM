"""Tests for the v0.6 execution-substrate launch-plan generator."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from codelewm.training import (
    EXECUTION_LAUNCH_PLAN_SCHEMA_VERSION,
    ExecutionLaunchPlanError,
    build_launch_plans,
    load_v0_6_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "train"
    / "scaled"
    / "codelewm_execution_v0_6_a10g.yaml"
)
LAUNCHER = REPO_ROOT / "scripts" / "hf-launch-execution-run"


class V0_6ConfigTest(unittest.TestCase):
    def test_config_loads(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        self.assertEqual(config["schema_version"], "codelewm.execution_train_config.v1")
        self.assertEqual(config["substrate"], "execution_trace_v1")
        self.assertEqual(set(config["seeds"]), {42, 1729})

    def test_required_keys_present(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        for top in (
            "data",
            "loader",
            "trainer",
            "optimizer",
            "wm",
            "objective",
            "seeds",
            "hf_jobs",
            "claim_gates",
            "claim_boundary",
        ):
            with self.subTest(key=top):
                self.assertIn(top, config)
        for k in (
            "prediction_mse_weight",
            "sigreg_weight",
            "action_swap_contrastive_weight",
            "inverse_action_reconstruction_weight",
        ):
            with self.subTest(weight=k):
                self.assertIn(k, config["objective"])


class LaunchPlanBuilderTest(unittest.TestCase):
    def test_one_plan_per_seed(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        plans = build_launch_plans(
            config=config, config_path=CONFIG_PATH, git_sha="abc1234", date="20260601"
        )
        self.assertEqual(len(plans), 2)
        seeds = {p.seed for p in plans}
        self.assertEqual(seeds, {42, 1729})
        for plan in plans:
            self.assertEqual(
                plan.schema_version, EXECUTION_LAUNCH_PLAN_SCHEMA_VERSION
            )
            self.assertIn("abc1234", plan.run_name)
            self.assertIn("20260601", plan.run_name)
            self.assertIn(f"seed-{plan.seed}", plan.run_name)
            self.assertIn(
                f"--env",
                plan.command,
            )
            command_str = " ".join(plan.command)
            self.assertIn("hf jobs run", command_str)
            self.assertIn(plan.pack_repo_id, command_str)
            self.assertIn(plan.pack_revision, command_str)
            self.assertIn(str(plan.seed), command_str)

    def test_claim_gates_round_trip(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        plans = build_launch_plans(
            config=config, config_path=CONFIG_PATH, git_sha="x", date="y"
        )
        for plan in plans:
            self.assertEqual(
                plan.claim_gates["retrieval_min_recall_at_1_lift_over_no_action"],
                0.05,
            )
            self.assertEqual(
                plan.claim_gates["collapse_effective_rank_ratio_min"], 0.20
            )


class V0_6ConfigErrorsTest(unittest.TestCase):
    def test_missing_top_level_key_rejected(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        broken = dict(config)
        del broken["objective"]
        with self.assertRaises(ExecutionLaunchPlanError):
            build_launch_plans(
                config=broken, config_path=CONFIG_PATH, git_sha="x", date="y"
            )

    def test_seed_must_be_int(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        broken = dict(config)
        broken["seeds"] = ["not-an-int"]
        with self.assertRaises(ExecutionLaunchPlanError):
            build_launch_plans(
                config=broken, config_path=CONFIG_PATH, git_sha="x", date="y"
            )


class LauncherCLITest(unittest.TestCase):
    def test_dry_run_emits_json_plans(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(LAUNCHER),
                "--config",
                str(CONFIG_PATH),
                "--git-sha",
                "abc1234",
                "--date",
                "20260601",
                "--json",
            ],
            env={
                "PYTHONPATH": str(REPO_ROOT),
                "PATH": "/usr/bin:/bin:/usr/local/bin",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode, 0,
            msg=f"stderr={completed.stderr!r}",
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(len(payload), 2)
        seeds = sorted(p["seed"] for p in payload)
        self.assertEqual(seeds, [42, 1729])

    def test_seed_filter_subsets_plans(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(LAUNCHER),
                "--config",
                str(CONFIG_PATH),
                "--git-sha",
                "abc1234",
                "--date",
                "20260601",
                "--seed",
                "42",
                "--json",
            ],
            env={
                "PYTHONPATH": str(REPO_ROOT),
                "PATH": "/usr/bin:/bin:/usr/local/bin",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        payload = json.loads(completed.stdout)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["seed"], 42)


class V0_6RunbookTest(unittest.TestCase):
    def test_runbook_doc_exists_and_links_config(self) -> None:
        doc = (
            REPO_ROOT
            / "docs"
            / "operations"
            / "V0_6_EXECUTION_RUN_RUNBOOK.md"
        )
        self.assertTrue(doc.is_file())
        text = doc.read_text(encoding="utf-8")
        self.assertIn("codelewm_execution_v0_6_a10g.yaml", text)
        self.assertIn("hf-launch-execution-run", text)
        # Claim-gate language is required, verbatim, in the runbook header
        # so the operator is reminded before pressing go.
        self.assertIn("Recall@1", text)
        self.assertIn("effective rank ratio", text)
        self.assertIn("pass@1 lift", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
