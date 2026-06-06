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
V0_8_CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "train"
    / "scaled"
    / "codelewm_execution_v0_8_a10g.yaml"
)
V0_8_SHORT_CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "train"
    / "scaled"
    / "codelewm_execution_v0_8_short_a10g.yaml"
)
V0_9_SHORT_CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "train"
    / "scaled"
    / "codelewm_execution_v0_9_short_a10g.yaml"
)
LAUNCHER = REPO_ROOT / "scripts" / "hf-launch-execution-run"
DIGEST = "sha256:" + "a" * 64


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

    def test_runtime_image_from_config(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        plans = build_launch_plans(
            config=config, config_path=CONFIG_PATH, git_sha="x", date="y"
        )
        for plan in plans:
            self.assertEqual(
                plan.runtime_image,
                "ghcr.io/abdelstark/codelewm-runtime:v0.6",
            )
            command_str = " ".join(plan.command)
            self.assertIn(plan.runtime_image, command_str)

    def test_passfail_objective_fields_round_trip_when_present(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        config["objective"] = dict(config["objective"])
        config["objective"]["p_pass_bce_weight"] = 0.4
        config["objective"]["p_pass_bce_pos_weight"] = 2.0

        plans = build_launch_plans(
            config=config, config_path=CONFIG_PATH, git_sha="x", date="y"
        )

        for plan in plans:
            self.assertEqual(plan.objective["p_pass_bce_weight"], 0.4)
            self.assertEqual(plan.objective["p_pass_bce_pos_weight"], 2.0)

    def test_output_value_objective_field_round_trips_when_present(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        config["objective"] = dict(config["objective"])
        config["objective"]["output_value_ce_weight"] = 0.2

        plans = build_launch_plans(
            config=config, config_path=CONFIG_PATH, git_sha="x", date="y"
        )

        for plan in plans:
            self.assertEqual(plan.objective["output_value_ce_weight"], 0.2)

    def test_checked_in_v0_8_launch_plan_wires_correctness_recipe(self) -> None:
        config = load_v0_6_config(V0_8_CONFIG_PATH)
        plans = build_launch_plans(
            config=config,
            config_path=V0_8_CONFIG_PATH,
            git_sha="deadbee",
            date="20260604",
        )

        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertEqual(plan.pack_revision, "v0.8.0-rc1")
            self.assertEqual(
                plan.runtime_image,
                "ghcr.io/abdelstark/codelewm-runtime:v0.8",
            )
            self.assertEqual(plan.objective["p_pass_bce_weight"], 0.5)
            self.assertAlmostEqual(
                plan.objective["p_pass_bce_pos_weight"],
                0.9145473041709054,
            )
            self.assertEqual(plan.objective["output_value_ce_weight"], 0.2)
            self.assertEqual(plan.objective["retrieval_weight"], 0.05)
            self.assertIn("20260604", plan.run_name)
            self.assertIn("deadbee", plan.run_name)
            command_str = " ".join(plan.command)
            self.assertIn("CODELEWM_EXECUTION_PACK_REVISION=v0.8.0-rc1", command_str)
            self.assertIn(str(V0_8_CONFIG_PATH), command_str)
            self.assertIn(plan.runtime_image, command_str)

    def test_checked_in_v0_8_short_launch_plan_has_distinct_uploads(self) -> None:
        config = load_v0_6_config(V0_8_SHORT_CONFIG_PATH)
        plans = build_launch_plans(
            config=config,
            config_path=V0_8_SHORT_CONFIG_PATH,
            git_sha="deadbee",
            date="20260604",
        )

        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertIn("short", plan.run_name)
            self.assertIn("short", plan.checkpoint_revision)
            self.assertEqual(plan.trainer["max_steps"], 12000)
            self.assertEqual(
                plan.runtime_image,
                "ghcr.io/abdelstark/codelewm-runtime:v0.8",
            )

    def test_checked_in_v0_9_launch_plan_requires_digest_pin_for_live_use(self) -> None:
        config = load_v0_6_config(V0_9_SHORT_CONFIG_PATH)
        plans = build_launch_plans(
            config=config,
            config_path=V0_9_SHORT_CONFIG_PATH,
            git_sha="feed123",
            date="20260606",
            runtime_image_digest=DIGEST,
        )

        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertEqual(plan.pack_revision, "v0.9.0-rc1")
            self.assertEqual(plan.runtime_image, "ghcr.io/abdelstark/codelewm-runtime:v0.9")
            self.assertEqual(plan.runtime_image_digest, DIGEST)
            self.assertEqual(
                plan.runtime_image_reference,
                f"ghcr.io/abdelstark/codelewm-runtime:v0.9@{DIGEST}",
            )
            self.assertIn("feed123", plan.run_name)
            self.assertIn("20260606", plan.run_name)
            self.assertIn("codelewm_execution_v0_9_short_a10g.yaml", plan.config_path)
            command_str = " ".join(plan.command)
            self.assertIn(plan.runtime_image_reference, command_str)
            self.assertIn("CODELEWM_EXECUTION_PACK_REVISION=v0.9.0-rc1", command_str)
            self.assertIn("CODELEWM_UPLOAD_PATH_IN_REPO=" + plan.run_name, command_str)

    def test_runtime_image_digest_format_is_validated(self) -> None:
        config = load_v0_6_config(V0_9_SHORT_CONFIG_PATH)
        with self.assertRaisesRegex(ExecutionLaunchPlanError, "runtime image digest"):
            build_launch_plans(
                config=config,
                config_path=V0_9_SHORT_CONFIG_PATH,
                git_sha="x",
                date="y",
                runtime_image_digest="latest",
            )

    def test_command_invokes_entrypoint_then_codelewm(self) -> None:
        """HF Jobs strips ENTRYPOINT when COMMAND is supplied.

        The launcher's command vector must therefore invoke the
        entrypoint script explicitly, then run ``codelewm`` directly
        (skipping ``uv run`` whose cache dir fails on HF Jobs' non-
        root runtime). The runner is the same in either case; this
        keeps the pack-download pre-step intact.
        """

        config = load_v0_6_config(CONFIG_PATH)
        plans = build_launch_plans(
            config=config, config_path=CONFIG_PATH, git_sha="x", date="y"
        )
        for plan in plans:
            cmd = list(plan.command)
            image_idx = cmd.index(plan.runtime_image)
            after_image = cmd[image_idx + 1 :]
            self.assertEqual(
                after_image[0], "/usr/local/bin/codelewm-runtime-entrypoint"
            )
            self.assertEqual(after_image[1], "codelewm")
            self.assertEqual(after_image[2], "train")
            self.assertNotIn("uv", after_image)

    def test_command_includes_hf_token_secret(self) -> None:
        """The entrypoint's HF download needs HF_TOKEN, so the command
        must pass it as a secret on every invocation."""

        config = load_v0_6_config(CONFIG_PATH)
        plans = build_launch_plans(
            config=config, config_path=CONFIG_PATH, git_sha="x", date="y"
        )
        for plan in plans:
            cmd = list(plan.command)
            self.assertIn("--secrets", cmd)
            secret_idx = cmd.index("--secrets")
            self.assertEqual(cmd[secret_idx + 1], "HF_TOKEN")

    def test_command_wires_artifact_upload(self) -> None:
        """HF Jobs containers don't persist /tmp past completion.

        The launcher must pass the env vars the entrypoint reads to
        upload the run output dir to the configured artifact repo,
        plus the matching ``--out`` flag so the runner writes to a
        path the entrypoint knows to read.
        """

        config = load_v0_6_config(CONFIG_PATH)
        plans = build_launch_plans(
            config=config, config_path=CONFIG_PATH, git_sha="x", date="y"
        )
        for plan in plans:
            cmd = list(plan.command)
            expected_run_dir = f"/tmp/runs/{plan.run_name}"
            # Env vars: run-output-dir, upload-repo-id, upload path.
            command_env_pairs = [
                cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--env"
            ]
            self.assertIn(
                f"CODELEWM_RUN_OUTPUT_DIR={expected_run_dir}",
                command_env_pairs,
            )
            self.assertIn(
                f"CODELEWM_UPLOAD_REPO_ID={plan.artifact_repo_id}",
                command_env_pairs,
            )
            self.assertIn(
                f"CODELEWM_UPLOAD_PATH_IN_REPO={plan.run_name}",
                command_env_pairs,
            )
            # --out flag matches.
            self.assertIn("--out", cmd)
            out_idx = cmd.index("--out")
            self.assertEqual(cmd[out_idx + 1], expected_run_dir)
            # --json so the training_manifest is captured in stdout.
            self.assertIn("--json", cmd)

    def test_runtime_image_falls_back_to_default_when_missing(self) -> None:
        from codelewm.training.execution_launch_plan import DEFAULT_RUNTIME_IMAGE

        config = load_v0_6_config(CONFIG_PATH)
        config["hf_jobs"] = dict(config["hf_jobs"])
        config["hf_jobs"].pop("runtime_image", None)
        plans = build_launch_plans(
            config=config, config_path=CONFIG_PATH, git_sha="x", date="y"
        )
        for plan in plans:
            self.assertEqual(plan.runtime_image, DEFAULT_RUNTIME_IMAGE)


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

    def test_unknown_objective_key_rejected(self) -> None:
        config = load_v0_6_config(CONFIG_PATH)
        broken = dict(config)
        broken["objective"] = dict(config["objective"])
        broken["objective"]["mystery_loss"] = 1.0

        with self.assertRaisesRegex(ExecutionLaunchPlanError, "unknown key"):
            build_launch_plans(
                config=broken, config_path=CONFIG_PATH, git_sha="x", date="y"
            )


class LauncherCLITest(unittest.TestCase):
    def test_uv_script_metadata_declares_runtime_dependencies(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('"numpy>=1.24"', text)

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

    def test_digest_pin_is_emitted_by_cli_plan(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(LAUNCHER),
                "--config",
                str(V0_9_SHORT_CONFIG_PATH),
                "--git-sha",
                "feed123",
                "--date",
                "20260606",
                "--runtime-image-digest",
                DIGEST,
                "--require-runtime-image-digest",
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
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(len(payload), 2)
        for plan in payload:
            self.assertEqual(plan["runtime_image_digest"], DIGEST)
            self.assertIn("@sha256:", plan["runtime_image_reference"])
            self.assertIn(plan["runtime_image_reference"], " ".join(plan["command"]))

    def test_required_digest_cli_fails_closed(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(LAUNCHER),
                "--config",
                str(V0_9_SHORT_CONFIG_PATH),
                "--require-runtime-image-digest",
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
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--runtime-image-digest is required", completed.stderr)


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
