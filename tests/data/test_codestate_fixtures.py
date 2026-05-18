from __future__ import annotations

import json
import unittest
from pathlib import Path

from codelewm.data import (
    CodeStateExtractionError,
    RawEditRecord,
    extract_codestate_pair,
    normalize_codestate,
)


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "codestate"


def _record(name: str) -> RawEditRecord:
    before = (FIXTURE_ROOT / f"{name}_before.py").read_text()
    after = (FIXTURE_ROOT / f"{name}_after.py").read_text()
    return RawEditRecord(
        source="local_repo",
        repo="example/repo",
        commit=name,
        path_before=f"pkg/{name}.py",
        path_after=f"pkg/{name}.py",
        before=before,
        after=after,
        message=f"fixture {name}",
        license="mit",
    )


class CodeStateFixtureCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected = json.loads((FIXTURE_ROOT / "expected_snapshots.json").read_text())

    def test_fixture_snapshots_match_expected_codestates(self) -> None:
        for name, expected in self.expected.items():
            with self.subTest(name=name):
                state = extract_codestate_pair(_record(name)).after
                normalized = normalize_codestate(state)
                normalized_primary = normalized.text.split("<PRIMARY>\n", 1)[1]

                self.assertEqual(state.kind, expected["kind"])
                self.assertEqual(state.symbol, expected["symbol"])
                self.assertEqual(state.enclosing_class, expected["enclosing_class"])
                self.assertEqual(list(state.changed_hunk_mask), expected["changed_hunk_mask"])
                self.assertEqual(list(state.sibling_signatures), expected["sibling_signatures"])
                self.assertEqual(list(state.callee_signatures), expected["callee_signatures"])
                self.assertEqual(state.primary, expected["primary"])
                self.assertEqual(normalized_primary, expected["normalized_primary"])

    def test_decorator_lines_are_preserved_in_primary_snapshot(self) -> None:
        state = extract_codestate_pair(_record("decorated_async")).after

        self.assertTrue(state.primary.startswith("@trace\nasync def fetch"))

    def test_invalid_syntax_fixture_reports_after_field(self) -> None:
        with self.assertRaisesRegex(CodeStateExtractionError, "after"):
            extract_codestate_pair(_record("invalid"))


if __name__ == "__main__":
    unittest.main()
