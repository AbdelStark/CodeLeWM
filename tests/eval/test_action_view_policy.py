from __future__ import annotations

import unittest

from codelewm.eval import (
    ACTION_VIEW_POLICY_SCHEMA_VERSION,
    ActionViewPolicyError,
    ActionViewReportPolicy,
    build_action_view_report_policy,
    validate_action_view_report_policy,
)


class ActionViewPolicyTest(unittest.TestCase):
    def test_headline_report_accepts_text_action_view(self) -> None:
        policy = build_action_view_report_policy("text", report_scope="headline")

        self.assertEqual(policy.action_view, "text")
        self.assertEqual(policy.report_scope, "headline")
        self.assertFalse(policy.diagnostic_upper_bound)
        self.assertEqual(policy.warnings, ())

    def test_headline_report_rejects_patch_action_view(self) -> None:
        with self.assertRaisesRegex(ActionViewPolicyError, "headline"):
            build_action_view_report_policy("patch", report_scope="headline")

    def test_headline_report_rejects_abstract_ablation_action_view(self) -> None:
        with self.assertRaisesRegex(ActionViewPolicyError, "headline"):
            build_action_view_report_policy("abstract", report_scope="headline")

    def test_patch_report_must_be_diagnostic_upper_bound(self) -> None:
        policy = build_action_view_report_policy("patch", report_scope="diagnostic")

        self.assertEqual(policy.action_view, "patch")
        self.assertEqual(policy.report_scope, "diagnostic")
        self.assertTrue(policy.diagnostic_upper_bound)
        self.assertIn("diagnostic upper bound", policy.warnings[0])

    def test_report_validation_round_trips_json_native_payload(self) -> None:
        policy = build_action_view_report_policy("patch", report_scope="diagnostic")

        loaded = validate_action_view_report_policy(policy.to_dict())

        self.assertEqual(loaded, policy)
        self.assertEqual(loaded.schema_version, ACTION_VIEW_POLICY_SCHEMA_VERSION)

    def test_report_validation_rejects_mislabeled_patch_report(self) -> None:
        policy = ActionViewReportPolicy(
            action_view="patch",
            report_scope="diagnostic",
            diagnostic_upper_bound=False,
        )

        with self.assertRaisesRegex(ActionViewPolicyError, "diagnostic_upper_bound"):
            validate_action_view_report_policy(policy)

    def test_report_validation_rejects_upper_bound_tag_on_non_patch_report(self) -> None:
        policy = ActionViewReportPolicy(
            action_view="text",
            report_scope="ablation",
            diagnostic_upper_bound=True,
        )

        with self.assertRaisesRegex(ActionViewPolicyError, "only patch"):
            validate_action_view_report_policy(policy)

    def test_report_validation_rejects_unknown_schema(self) -> None:
        payload = build_action_view_report_policy("text", report_scope="headline").to_dict()
        payload["schema_version"] = "codelewm.eval.action_view_policy.v0"

        with self.assertRaisesRegex(ActionViewPolicyError, "unsupported"):
            validate_action_view_report_policy(payload)


if __name__ == "__main__":
    unittest.main()
