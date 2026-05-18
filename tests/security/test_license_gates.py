from __future__ import annotations

import unittest

from codelewm.data import (
    ArtifactInfo,
    PackedTransition,
    RawEditRecord,
    TokenSequence,
    build_dataset_manifest,
    filter_raw_edit_records,
)
from codelewm.security import (
    PUBLIC_LICENSE_GATE_SCHEMA_VERSION,
    LicenseGateError,
    build_public_license_gate_report,
    decide_license,
    enforce_public_license_gate,
    validate_public_license_gate_report,
)


class PublicLicenseGateTest(unittest.TestCase):
    def test_filter_result_excludes_unknown_license_and_reports_public_gate(self) -> None:
        result = filter_raw_edit_records(
            [
                _raw_record(commit="allowed", license="mit"),
                _raw_record(commit="missing", license="unknown"),
            ]
        )

        gate = result.license_gate_report

        self.assertEqual(len(result.kept), 1)
        self.assertEqual(len(result.dropped), 1)
        self.assertEqual(result.report.drop_reasons, {"license_denied": 1})
        self.assertEqual(gate.schema_version, PUBLIC_LICENSE_GATE_SCHEMA_VERSION)
        self.assertEqual(gate.included_rows, 1)
        self.assertEqual(gate.excluded_rows, 1)
        self.assertEqual(gate.excluded_reasons, {"missing_license": 1})
        self.assertEqual(gate.excluded_licenses, {"missing": 1})
        enforce_public_license_gate(gate)

    def test_public_gate_rejects_included_non_allowlisted_license(self) -> None:
        gate = build_public_license_gate_report(
            included=(decide_license(source="commitpackft", license="agpl-3.0"),),
        )

        self.assertFalse(gate.release_allowed)
        self.assertEqual(gate.blocked_rows, 1)
        with self.assertRaisesRegex(LicenseGateError, "license gate failed"):
            enforce_public_license_gate(gate)

    def test_public_gate_payload_round_trips_json_native_report(self) -> None:
        gate = build_public_license_gate_report(
            included=(decide_license(source="commitpackft", license="Apache License 2.0"),),
            excluded=(decide_license(source="local_repo", license=None),),
        )

        loaded = validate_public_license_gate_report(gate.to_dict())

        self.assertEqual(loaded.to_dict(), gate.to_dict())
        self.assertEqual(loaded.included_licenses, {"apache-2.0": 1})
        self.assertEqual(loaded.excluded_sources, {"local_repo": 1})

    def test_dataset_manifest_embeds_license_summary_and_gate_report(self) -> None:
        gate = build_public_license_gate_report(
            included=(
                decide_license(source="commitpackft", license="mit"),
                decide_license(source="synthetic", license="bsd-3-clause"),
            ),
            excluded=(decide_license(source="commitpackft", license="unknown"),),
        )

        manifest = build_dataset_manifest(
            [
                _transition(transition_id="t0", source="commitpackft", license="mit"),
                _transition(transition_id="t1", source="synthetic", license="bsd-3-clause"),
            ],
            artifacts=(
                ArtifactInfo(
                    path="dataset.h5",
                    kind="hdf5",
                    rows=2,
                    sha256="0" * 64,
                    bytes=10,
                ),
            ),
            license_gate_report=gate,
        )

        self.assertEqual(manifest.metadata["license_summary"]["included_licenses"], {"mit": 1, "bsd-3-clause": 1})
        self.assertEqual(manifest.metadata["license_gate_report"]["excluded_reasons"], {"missing_license": 1})


def _raw_record(*, commit: str, license: str | None) -> RawEditRecord:
    return RawEditRecord(
        source="commitpackft",
        repo="example/repo",
        commit=commit,
        path_before="pkg/mod.py",
        path_after="pkg/mod.py",
        before="def value():\n    return 1\n",
        after="def value():\n    return 2\n",
        message="change return value",
        license=license,
    )


def _transition(**overrides: object) -> PackedTransition:
    values: dict[str, object] = {
        "transition_id": "t0",
        "source": "commitpackft",
        "repo": "example/repo",
        "commit": "abc123",
        "path": "pkg/mod.py",
        "split": "train",
        "state_before": TokenSequence(input_ids=(1, 2)),
        "state_after": TokenSequence(input_ids=(1, 3)),
        "action_text": TokenSequence(input_ids=(10,)),
        "action_abs": TokenSequence(input_ids=(20,)),
        "license": "mit",
    }
    values.update(overrides)
    return PackedTransition(**values)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
