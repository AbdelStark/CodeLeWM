from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRACKER = ROOT / "docs" / "roadmap" / "IMPLEMENTATION.md"
SPEC_DIR = ROOT / "docs" / "spec"
RFC_DIR = ROOT / "docs" / "rfcs"

_TABLE_ROW_RE = re.compile(
    r"^\|\s+#(?P<number>\d+)\s+\|"
    r"\s*(?P<title>[^|]+?)\s+\|"
    r"\s*(?P<area>[^|]+?)\s+\|"
    r"\s*(?P<priority>[^|]+?)\s+\|"
    r"\s*(?P<effort>[^|]+?)\s+\|"
    r"\s*(?P<rfc>RFC-\d{4}|follow-up)\s+\|"
    r"\s*(?P<status>[^|]+?)\s+\|"
    r"\s*$"
)
_TRACKING_ROW_RE = re.compile(r"^- #(?P<number>\d+) \[Tracking\] .+$")
_LAST_UPDATED_RE = re.compile(r"^- Last updated: (?P<date>\d{4}-\d{2}-\d{2})$", re.MULTILINE)


class ImplementationTrackerStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = TRACKER.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()

    def test_tracker_file_exists(self) -> None:
        self.assertTrue(TRACKER.is_file(), f"missing: {TRACKER}")

    def test_tracker_has_last_updated_iso_date(self) -> None:
        match = _LAST_UPDATED_RE.search(self.text)

        self.assertIsNotNone(match, "Last updated line must be 'YYYY-MM-DD'")

    def test_tracker_has_required_sections(self) -> None:
        for heading in (
            "# Implementation Tracker",
            "## How This Tracker Is Maintained",
            "## Milestone: v0.1",
            "## Milestone: v1.0",
            "## Milestone: v1.1",
            "## Milestone: v1.2",
            "## Tracking Issues",
            "## Cross-Cutting Dependencies",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, self.text)

    def test_tracker_milestone_tables_have_canonical_header(self) -> None:
        header = "| # | Title | Area | Priority | Effort | RFC | Status |"

        self.assertEqual(self.text.count(header), 4, "four milestone tables expected")


class ImplementationTrackerContentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = TRACKER.read_text(encoding="utf-8")
        self.rows = [_TABLE_ROW_RE.match(line) for line in self.text.splitlines()]
        self.rows = [match for match in self.rows if match is not None]

    def test_every_issue_row_parses_with_expected_columns(self) -> None:
        self.assertGreater(len(self.rows), 0, "no issue rows found")
        seen_numbers: set[int] = set()
        for match in self.rows:
            number = int(match.group("number"))
            with self.subTest(number=number):
                self.assertNotIn(number, seen_numbers, f"#{number} listed twice")
                seen_numbers.add(number)
                self.assertIn(match.group("priority"), {"p0", "p1", "p2"})
                self.assertIn(match.group("effort"), {"s", "m", "l"})
                self.assertIn(match.group("status"), {"Open", "Closed"})

    def test_every_referenced_rfc_file_exists(self) -> None:
        rfc_numbers = sorted(
            {match.group("rfc") for match in self.rows if match.group("rfc").startswith("RFC-")}
        )

        self.assertGreater(len(rfc_numbers), 0)
        for rfc_number in rfc_numbers:
            with self.subTest(rfc=rfc_number):
                matches = list(RFC_DIR.glob(f"{rfc_number}-*.md"))
                self.assertEqual(len(matches), 1, f"missing RFC file for {rfc_number}")

    def test_tracking_issues_are_listed_for_every_rfc(self) -> None:
        text = TRACKER.read_text(encoding="utf-8")
        tracking_numbers = {
            int(match.group("number"))
            for match in (_TRACKING_ROW_RE.match(line) for line in text.splitlines())
            if match is not None
        }

        self.assertEqual(tracking_numbers, set(range(2, 14)))

    def test_referenced_spec_files_exist(self) -> None:
        text = TRACKER.read_text(encoding="utf-8")

        for spec_link in re.findall(r"docs/spec/[A-Za-z0-9_./-]+\.md", text):
            with self.subTest(spec=spec_link):
                self.assertTrue((ROOT / spec_link).is_file(), f"missing: {spec_link}")


class ImplementationTrackerMaintenancePointersTest(unittest.TestCase):
    def test_tracker_points_at_contract_test_path(self) -> None:
        text = TRACKER.read_text(encoding="utf-8")

        self.assertIn("tests/docs/test_implementation_tracker.py", text)

    def test_tracker_documents_maintenance_routine(self) -> None:
        text = TRACKER.read_text(encoding="utf-8")

        self.assertIn("How This Tracker Is Maintained", text)
        self.assertIn("CONTRIBUTING.md", text)


if __name__ == "__main__":
    unittest.main()
