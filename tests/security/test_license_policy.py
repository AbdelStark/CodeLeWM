from __future__ import annotations

import unittest

from codelewm.security import SourceLicensePolicy, decide_license, normalize_license


class LicensePolicyTest(unittest.TestCase):
    def test_normalizes_common_license_names(self) -> None:
        self.assertEqual(normalize_license("MIT License"), "mit")
        self.assertEqual(normalize_license("Apache License 2.0"), "apache-2.0")
        self.assertEqual(normalize_license("CC0"), "cc0-1.0")
        self.assertIsNone(normalize_license("unknown"))

    def test_allows_permissive_public_licenses(self) -> None:
        decision = decide_license(source="commitpackft", license="BSD-3-Clause")

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "allowed")
        self.assertEqual(decision.artifact_policy, "full_text")

    def test_denies_missing_license_by_default(self) -> None:
        decision = decide_license(source="commitpackft", license=None)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "missing_license")
        self.assertEqual(decision.artifact_policy, "exclude")

    def test_denies_non_allowlisted_license(self) -> None:
        decision = decide_license(source="commitpackft", license="agpl-3.0")

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "license_not_allowed")
        self.assertEqual(decision.license, "agpl-3.0")

    def test_source_policy_can_allow_missing_license_for_metadata_only_artifacts(self) -> None:
        policy = SourceLicensePolicy(
            allowed_licenses=("mit",),
            require_license_field=False,
            derived_artifact_policy="metadata_only",
        )

        decision = decide_license(source="local_repo", license=None, policy=policy)

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "source_policy_allows_missing_license")
        self.assertEqual(decision.artifact_policy, "metadata_only")

    def test_redistribution_denial_overrides_allowlist(self) -> None:
        policy = SourceLicensePolicy(redistribution_allowed=False)

        decision = decide_license(source="local_repo", license="mit", policy=policy)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "redistribution_not_allowed")
        self.assertEqual(decision.artifact_policy, "exclude")

    def test_license_decision_serializes_to_json_native_dict(self) -> None:
        decision = decide_license(source="commitpackft", license="mit")

        self.assertEqual(
            decision.to_dict(),
            {
                "allowed": True,
                "reason": "allowed",
                "source": "commitpackft",
                "license": "mit",
                "artifact_policy": "full_text",
            },
        )


if __name__ == "__main__":
    unittest.main()
