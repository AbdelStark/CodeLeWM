from __future__ import annotations

import unittest

from codelewm.data import (
    CodeStateConfig,
    CodeStateExtractionError,
    RawEditRecord,
    changed_line_numbers,
    extract_codestate_pair,
    module_name_from_path,
)


def _record(before: str, after: str, *, path: str = "pkg/mod.py") -> RawEditRecord:
    return RawEditRecord(
        source="local_repo",
        repo="example/repo",
        commit="abc123",
        path_before=path,
        path_after=path,
        before=before,
        after=after,
        message="update code",
        license="mit",
    )


class CodeStateExtractTest(unittest.TestCase):
    def test_changed_function_is_selected_by_line_overlap(self) -> None:
        before = """\
import math

def helper(x):
    return math.sqrt(x)

def add(a, b):
    total = a + b
    return total
"""
        after = before.replace("total = a + b", "total = a - b")

        pair = extract_codestate_pair(_record(before, after))

        self.assertEqual(pair.after.kind, "function")
        self.assertEqual(pair.after.symbol, "add")
        self.assertEqual(pair.after.module, "pkg.mod")
        self.assertEqual(pair.after.imports, "import math")
        self.assertIn("total = a - b", pair.after.primary)
        self.assertIn("def helper(x)", pair.after.sibling_signatures)
        self.assertEqual(pair.after.changed_hunk_mask, (False, True, False))

    def test_changed_method_records_enclosing_class_and_siblings(self) -> None:
        before = """\
class Counter:
    def reset(self):
        self.value = 0

    def bump(self, value):
        self.value += value
        return int(self.value)
"""
        after = before.replace("self.value += value", "self.value = self.value + value")

        pair = extract_codestate_pair(_record(before, after))

        self.assertEqual(pair.after.kind, "method")
        self.assertEqual(pair.after.symbol, "Counter.bump")
        self.assertEqual(pair.after.enclosing_class, "Counter")
        self.assertIn("def reset(self)", pair.after.sibling_signatures)
        self.assertIn("self.value = self.value + value", pair.after.primary)
        self.assertIn("int", pair.after.callee_signatures)

    def test_changed_class_is_selected_when_class_body_changes_without_method(self) -> None:
        before = """\
class Settings:
    mode = "dev"
    retries = 1
"""
        after = before.replace('mode = "dev"', 'mode = "prod"')

        pair = extract_codestate_pair(_record(before, after))

        self.assertEqual(pair.after.kind, "class")
        self.assertEqual(pair.after.symbol, "Settings")
        self.assertIn('mode = "prod"', pair.after.primary)

    def test_small_file_fallback_records_reason_for_top_level_change(self) -> None:
        before = "VALUE = 1\n"
        after = "VALUE = 2\n"

        pair = extract_codestate_pair(_record(before, after))

        self.assertEqual(pair.after.kind, "small_file")
        self.assertIsNone(pair.after.symbol)
        self.assertEqual(pair.after.fallback_reason, "small_file_no_symbol_overlap")
        self.assertEqual(pair.after.changed_hunk_mask, (True,))

    def test_region_fallback_records_reason_for_large_top_level_change(self) -> None:
        before_lines = [f"VALUE_{index} = {index}" for index in range(60)]
        after_lines = list(before_lines)
        after_lines[30] = "VALUE_30 = 300"
        before = "\n".join(before_lines) + "\n"
        after = "\n".join(after_lines) + "\n"

        pair = extract_codestate_pair(
            _record(before, after),
            config=CodeStateConfig(max_small_file_lines=10, region_context_lines=2),
        )

        self.assertEqual(pair.after.kind, "region")
        self.assertEqual(pair.after.fallback_reason, "changed_region_no_symbol_overlap")
        self.assertIn("VALUE_30 = 300", pair.after.primary)
        self.assertEqual(pair.after.changed_hunk_mask, (False, False, True, False, False))

    def test_syntax_failure_is_reported_with_field_name(self) -> None:
        before = "def ok():\n    return 1\n"
        after = "def broken(:\n    return 2\n"

        with self.assertRaisesRegex(CodeStateExtractionError, "after"):
            extract_codestate_pair(_record(before, after))

    def test_changed_line_numbers_track_before_and_after_sides(self) -> None:
        before_changed, after_changed = changed_line_numbers("a\nb\nc\n", "a\nB\nc\nd\n")

        self.assertEqual(before_changed, {2})
        self.assertEqual(after_changed, {2, 4})

    def test_module_name_from_path_handles_init_modules(self) -> None:
        self.assertEqual(module_name_from_path("pkg/sub/__init__.py"), "pkg.sub")


if __name__ == "__main__":
    unittest.main()
