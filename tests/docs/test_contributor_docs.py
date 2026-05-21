from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
SECURITY = ROOT / "SECURITY.md"
CHANGELOG = ROOT / "CHANGELOG.md"
PR_TEMPLATE = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
RELEASE_SPEC = ROOT / "docs" / "spec" / "09-release-and-versioning.md"


class ContributorDocsExistenceTest(unittest.TestCase):
    def test_contributing_doc_exists(self) -> None:
        self.assertTrue(CONTRIBUTING.is_file(), f"missing: {CONTRIBUTING}")

    def test_security_doc_exists(self) -> None:
        self.assertTrue(SECURITY.is_file(), f"missing: {SECURITY}")

    def test_changelog_exists(self) -> None:
        self.assertTrue(CHANGELOG.is_file(), f"missing: {CHANGELOG}")

    def test_pull_request_template_exists(self) -> None:
        self.assertTrue(PR_TEMPLATE.is_file(), f"missing: {PR_TEMPLATE}")


class ContributingContentTest(unittest.TestCase):
    def test_contributing_doc_references_validation_and_deprecation(self) -> None:
        text = CONTRIBUTING.read_text(encoding="utf-8")

        self.assertIn("python -m pytest", text)
        self.assertIn("Deprecation", text)
        self.assertIn("one minor release before removal", text)

    def test_contributing_doc_directs_security_reports_to_security_md(self) -> None:
        text = CONTRIBUTING.read_text(encoding="utf-8")

        self.assertIn("SECURITY.md", text)


class SecurityPolicyContentTest(unittest.TestCase):
    def test_security_doc_lists_private_reporting_channel(self) -> None:
        text = SECURITY.read_text(encoding="utf-8")

        self.assertIn("Security Advisor", text)
        self.assertIn("do not open a public GitHub issue", text)

    def test_security_doc_states_triage_timeline(self) -> None:
        text = SECURITY.read_text(encoding="utf-8")

        self.assertIn("acknowledge", text)
        self.assertIn("remediation", text)

    def test_security_doc_lists_in_scope_categories(self) -> None:
        text = SECURITY.read_text(encoding="utf-8")

        self.assertIn("secret-pattern leakage", text)
        self.assertIn("require_trusted_checkpoint", text)
        self.assertIn("non-execution", text)


class ChangelogContentTest(unittest.TestCase):
    def test_changelog_uses_keep_a_changelog_sections(self) -> None:
        text = CHANGELOG.read_text(encoding="utf-8")

        for section in ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"):
            with self.subTest(section=section):
                self.assertRegex(text, rf"###\s+{section}")

    def test_changelog_documents_schema_reference_table(self) -> None:
        text = CHANGELOG.read_text(encoding="utf-8")

        self.assertIn("Schema Reference", text)
        for schema in (
            "codelewm.dataset.v1",
            "codelewm.artifact_manifest.v1",
            "codelewm.manifest_verify.v1",
            "codelewm.checkpoint.v1",
            "codelewm.train_config.v1",
            "codelewm.training_run.v1",
            "codelewm.training_metrics.v1",
            "codelewm.cpu_smoke_checkpoint.v1",
            "codelewm.cpu_smoke_report.v1",
            "codelewm.index_build.v1",
            "codelewm.eval.retrieval_metrics.v1",
            "codelewm.eval.retrieval_report.v1",
            "codelewm.eval.candidate_pool.v1",
            "codelewm.eval.action_contrast_pool_report.v1",
            "codelewm.eval.latent_probe_run.v1",
            "codelewm.eval.latent_probe_report.v1",
            "codelewm.eval.hard_negative_sample.v1",
            "codelewm.eval.hard_negative_sampler_report.v1",
            "codelewm.eval.surprise_report.v1",
            "codelewm.eval.action_view_policy.v1",
            "codelewm.harness.scorer_quality_config.v1",
            "codelewm.harness.scorer_quality_report.v1",
            "codelewm.harness.scorer_quality_run.v1",
            "codelewm.openrouter_byok_register.v1",
            "codelewm.downstream_rerank_benchmark_config.v1",
            "codelewm.downstream_benchmark_pack_run.v1",
            "codelewm.downstream_benchmark_readiness.v1",
            "codelewm.downstream_rerank_eval_run.v1",
            "codelewm.eval.collapse_report.v1",
            "codelewm.eval.kill_report.v1",
            "codelewm.public_license_gate.v1",
            "codelewm.secret_scan.v1",
            "codelewm.score.v1",
            "codelewm.rerank.v1",
            "codelewm.error.v1",
            "codelewm.transition_index.v1",
            "codelewm.log_event.v1",
            "codelewm.release_provenance.v1",
            "codelewm.transition.v1",
        ):
            with self.subTest(schema=schema):
                self.assertIn(schema, text)


class PullRequestTemplateContentTest(unittest.TestCase):
    def test_pull_request_template_requires_linked_issue(self) -> None:
        text = PR_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("Closes #", text)
        self.assertRegex(text, r"##\s+Linked Issue")

    def test_pull_request_template_requires_spec_or_rfc_reference(self) -> None:
        text = PR_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("docs/spec/", text)
        self.assertIn("docs/rfcs/", text)
        self.assertRegex(text, r"##\s+Spec\s*/\s*RFC Reference")

    def test_pull_request_template_requires_validation_commands(self) -> None:
        text = PR_TEMPLATE.read_text(encoding="utf-8")

        self.assertRegex(text, r"##\s+Validation")
        self.assertIn("python -m pytest", text)

    def test_pull_request_template_covers_artifact_and_deprecation_impact(self) -> None:
        text = PR_TEMPLATE.read_text(encoding="utf-8")

        self.assertRegex(text, r"##\s+Artifact Impact")
        self.assertRegex(text, r"##\s+Deprecations")
        self.assertRegex(text, r"##\s+Public Surface Impact")


class ReleaseAndVersioningCrossReferenceTest(unittest.TestCase):
    def test_release_spec_documents_deprecation_policy(self) -> None:
        text = RELEASE_SPEC.read_text(encoding="utf-8")

        self.assertIn("Deprecation", text)

    def test_contributing_points_at_release_spec_section(self) -> None:
        text = CONTRIBUTING.read_text(encoding="utf-8")

        self.assertTrue(re.search(r"docs/spec/06-security\.md", text))


if __name__ == "__main__":
    unittest.main()
