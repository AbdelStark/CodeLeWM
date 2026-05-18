from __future__ import annotations

import unittest

from codelewm.data import (
    FilterPolicy,
    RawEditRecord,
    evaluate_raw_edit_record,
    filter_raw_edit_records,
)


def _record(**overrides: object) -> RawEditRecord:
    values: dict[str, object] = {
        "source": "commitpackft",
        "repo": "example/repo",
        "commit": "abc123",
        "path_before": "pkg/mod.py",
        "path_after": "pkg/mod.py",
        "before": "def f():\n    return 1\n",
        "after": "def f():\n    return 2\n",
        "message": "change return value",
        "license": "mit",
        "metadata": {},
    }
    values.update(overrides)
    return RawEditRecord(**values)  # type: ignore[arg-type]


class RawEditFilterTest(unittest.TestCase):
    def test_keeps_parse_valid_python_edit_with_allowed_license(self) -> None:
        result = filter_raw_edit_records([_record()])

        self.assertEqual(len(result.kept), 1)
        self.assertEqual(result.report.total_before, 1)
        self.assertEqual(result.report.total_after, 1)
        self.assertEqual(result.report.total_dropped, 0)
        self.assertEqual(result.report.drop_reasons, {})

    def test_parse_failure_is_reported_as_drop_reason(self) -> None:
        dropped = evaluate_raw_edit_record(_record(after="def f(:\n    return 2\n"))

        self.assertIsNotNone(dropped)
        self.assertEqual(dropped.reason.code, "parse_error")
        self.assertEqual(dropped.reason.details["field"], "after")

    def test_edit_size_bounds_are_reported(self) -> None:
        result = filter_raw_edit_records([_record()], policy=FilterPolicy(max_changed_lines=1))

        self.assertEqual(result.report.drop_reasons, {"edit_size": 1})
        self.assertEqual(result.dropped[0].reason.details["changed_lines"], 2)

    def test_message_length_bounds_are_reported(self) -> None:
        result = filter_raw_edit_records([_record(message="short")])

        self.assertEqual(result.report.drop_reasons, {"message_length": 1})
        self.assertEqual(result.dropped[0].reason.code, "message_length")

    def test_generated_file_path_is_reported(self) -> None:
        result = filter_raw_edit_records([_record(path_after="pkg/generated/model.py")])

        self.assertEqual(result.report.drop_reasons, {"generated_file": 1})
        self.assertEqual(result.dropped[0].reason.details["path"], "pkg/generated/model.py")

    def test_license_denial_is_reported(self) -> None:
        result = filter_raw_edit_records([_record(license="agpl-3.0")])

        self.assertEqual(result.report.drop_reasons, {"license_denied": 1})
        self.assertEqual(result.dropped[0].license_decision.reason, "license_not_allowed")

    def test_report_contains_machine_readable_drop_records(self) -> None:
        result = filter_raw_edit_records(
            [
                _record(commit="keep"),
                _record(commit="bad-path", path_after="README.md"),
                _record(commit="bad-license", license=None),
            ]
        )
        payload = result.to_dict()

        self.assertEqual(payload["kept"], 1)
        self.assertEqual(result.report.to_dict()["total_before"], 3)
        self.assertEqual(result.report.to_dict()["total_after"], 1)
        self.assertEqual(result.report.drop_reasons, {"non_python_path": 1, "license_denied": 1})
        self.assertEqual(result.dropped[0].to_dict()["reason"]["code"], "non_python_path")
        self.assertIn("commitpackft:example/repo:bad-path:README.md", result.dropped[0].record_id)


if __name__ == "__main__":
    unittest.main()
