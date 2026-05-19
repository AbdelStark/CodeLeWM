from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PR_WORKFLOW = ROOT / ".github" / "workflows" / "pr.yml"
TESTING_STRATEGY = ROOT / "docs" / "spec" / "07-testing-strategy.md"


class PullRequestWorkflowContractTest(unittest.TestCase):
    def test_workflow_file_exists(self) -> None:
        self.assertTrue(PR_WORKFLOW.is_file(), f"missing workflow: {PR_WORKFLOW}")

    def test_workflow_triggers_on_pull_request_and_main_push(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("pull_request", text)
        self.assertRegex(text, r"push:\s*\n\s+branches:\s*\[\s*main\s*\]")

    def test_workflow_runs_pytest_invocation(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("uv run python -m pytest", text)

    def test_workflow_runs_test_directory_documented_in_strategy(self) -> None:
        strategy = TESTING_STRATEGY.read_text(encoding="utf-8")
        workflow = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("uv run python -m pytest", strategy)
        self.assertIn("tests/", workflow)

    def test_workflow_compiles_python_sources(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("uv run python -m compileall", text)
        self.assertIn("codelewm tests", text)

    def test_workflow_excludes_intentional_invalid_parser_fixtures_from_compileall(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("invalid_(before|after)", text)

    def test_workflow_runs_manifest_and_secret_release_gates(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("artifact-gates:", text)
        self.assertIn("manifest verify", text)
        self.assertIn("secret-scan", text)
        self.assertIn(".ci/manifest/manifest.json", text)

    def test_workflow_runs_dataset_fixture_build_and_pack_gate(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("dataset-fixture:", text)
        self.assertIn("uv sync --frozen --group dev --group data", text)
        self.assertIn("codelewm dataset build", text)
        self.assertIn("codelewm dataset pack", text)
        self.assertIn("--parent-manifest .ci/dataset-build/manifest.json", text)

    def test_workflow_pins_supported_python_versions(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        matrix_versions = re.findall(r'"3\.(\d+)"', text)
        major_minors = sorted({int(version) for version in matrix_versions if int(version) >= 10})

        self.assertIn(10, major_minors)
        self.assertIn(13, major_minors)

    def test_workflow_uses_pinned_action_majors(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("actions/checkout@v4", text)
        self.assertIn("actions/setup-python@v5", text)
        self.assertIn("astral-sh/setup-uv@v8.1.0", text)

    def test_workflow_installs_with_uv(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("uv sync --frozen --group dev", text)
        self.assertIn("uv lock --check", text)
        self.assertNotIn("python -m pip install", text)

    def test_workflow_runs_docs_sanity_job(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("docs-check:", text)
        self.assertIn("docs/spec/02-public-api.md", text)

    def test_workflow_minimum_permissions(self) -> None:
        text = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertRegex(text, r"permissions:\s*\n\s+contents:\s+read")


class TestingStrategyConsistencyTest(unittest.TestCase):
    def test_testing_strategy_documents_ci_policy(self) -> None:
        text = TESTING_STRATEGY.read_text(encoding="utf-8")

        self.assertIn("CI Policy", text)
        self.assertIn("unit tests", text)
        self.assertIn("manifest", text)
        self.assertIn("security scan", text)

    def test_local_pytest_command_is_documented(self) -> None:
        text = TESTING_STRATEGY.read_text(encoding="utf-8")

        self.assertIn("uv run python -m pytest", text)


if __name__ == "__main__":
    unittest.main()
