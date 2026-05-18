from __future__ import annotations

import unittest

from codelewm.data import (
    CodeState,
    CodeStateNormalizationConfig,
    CodeStateNormalizationError,
    normalize_codestate,
)


def _state(**overrides: object) -> CodeState:
    values: dict[str, object] = {
        "language": "python",
        "path": "pkg/mod.py",
        "module": "pkg.mod",
        "symbol": "compute",
        "kind": "function",
        "imports": "import os\nimport sys",
        "enclosing_class": None,
        "sibling_signatures": ("def helper(value)",),
        "callee_signatures": ("len", "str"),
        "primary": """\
def compute(customer_id):
    \"\"\"This docstring is removed from the main model view.\"\"\"
    token = "abcdefghijklmnopqrstuvwxyz"
    huge = 12345678901234567890
    return customer_id
""",
        "token_count": 15,
        "changed_hunk_mask": (False, False, True, True, False),
        "fallback_reason": None,
    }
    values.update(overrides)
    return CodeState(**values)  # type: ignore[arg-type]


class CodeStateNormalizeTest(unittest.TestCase):
    def test_normalized_pack_preserves_identifiers_and_replaces_large_literals(self) -> None:
        normalized = normalize_codestate(
            _state(),
            config=CodeStateNormalizationConfig(large_string_chars=8, large_number_digits=6),
        )

        self.assertIn("<LANG python>", normalized.text)
        self.assertIn("<PATH pkg/mod.py>", normalized.text)
        self.assertIn("<SYMBOL compute>", normalized.text)
        self.assertIn("def compute(customer_id):", normalized.text)
        self.assertIn("customer_id", normalized.text)
        self.assertIn("'<STR_LITERAL>'", normalized.text)
        self.assertIn("__CODELEWM_NUM_LITERAL__", normalized.text)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", normalized.text)
        self.assertNotIn("docstring is removed", normalized.text)
        self.assertEqual(normalized.dropped_sections, ())

    def test_token_budget_drops_context_before_structured_primary_truncation(self) -> None:
        primary_lines = ["def compute(value):"]
        primary_lines.extend(f"    scratch_{index} = {index}" for index in range(20))
        primary_lines.append("    changed = value + 1")
        primary_lines.extend(f"    tail_{index} = {index}" for index in range(20))
        primary_lines.append("    return changed")
        changed_mask = tuple(index == 21 for index in range(len(primary_lines)))
        state = _state(
            imports="import os\nimport sys\nimport pathlib",
            sibling_signatures=tuple(f"def sibling_{index}()" for index in range(20)),
            callee_signatures=tuple(f"callee_{index}" for index in range(20)),
            primary="\n".join(primary_lines) + "\n",
            changed_hunk_mask=changed_mask,
        )

        normalized = normalize_codestate(
            state,
            config=CodeStateNormalizationConfig(token_budget=80, changed_context_lines=1),
        )

        self.assertLessEqual(normalized.token_count, 80)
        self.assertEqual(
            normalized.dropped_sections,
            ("sibling_signatures", "callee_signatures", "imports", "primary_context"),
        )
        self.assertIn("def compute(value):", normalized.text)
        self.assertIn("changed = value + 1", normalized.text)
        self.assertIn("return changed", normalized.text)
        self.assertNotIn("scratch_0", normalized.text)
        self.assertNotIn("tail_19", normalized.text)

    def test_over_budget_after_structured_truncation_raises(self) -> None:
        state = _state(
            primary="def compute(value):\n    changed = value + 1\n    return changed\n",
            changed_hunk_mask=(False, True, False),
        )

        with self.assertRaisesRegex(CodeStateNormalizationError, "budget"):
            normalize_codestate(state, config=CodeStateNormalizationConfig(token_budget=8))

    def test_unparse_failure_falls_back_to_whitespace_normalization_for_regions(self) -> None:
        state = _state(
            symbol=None,
            kind="region",
            primary="    value    =    1\n    other = 2\n",
            changed_hunk_mask=(True, False),
            fallback_reason="changed_region_no_symbol_overlap",
        )

        normalized = normalize_codestate(state)

        self.assertIn("value = 1", normalized.text)
        self.assertIn("other = 2", normalized.text)


if __name__ == "__main__":
    unittest.main()
