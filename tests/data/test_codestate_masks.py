from __future__ import annotations

import unittest

from codelewm.data import (
    SEGMENT_IDS,
    CodeState,
    CodeStateNormalizationConfig,
    build_masked_codestate,
    stable_token_id,
    tokenize_normalized_codestate,
)


def _state(**overrides: object) -> CodeState:
    values: dict[str, object] = {
        "language": "python",
        "path": "pkg/mod.py",
        "module": "pkg.mod",
        "symbol": "compute",
        "kind": "function",
        "imports": "import math",
        "enclosing_class": None,
        "sibling_signatures": ("def helper(value)",),
        "callee_signatures": ("math.sqrt",),
        "primary": """\
def compute(value):
    base = value + 1
    result = math.sqrt(base)
    return result
""",
        "token_count": 12,
        "changed_hunk_mask": (False, False, True, False),
        "fallback_reason": None,
    }
    values.update(overrides)
    return CodeState(**values)  # type: ignore[arg-type]


class CodeStateMaskTest(unittest.TestCase):
    def test_segment_ids_cover_pack_sections(self) -> None:
        masked = build_masked_codestate(_state())

        self.assertEqual(len(masked.tokens), len(masked.token_sequence.segment_ids))
        self.assertEqual(len(masked.tokens), len(masked.token_sequence.changed_hunk_mask))
        self.assertIn(SEGMENT_IDS["path"], masked.token_sequence.segment_ids)
        self.assertIn(SEGMENT_IDS["imports"], masked.token_sequence.segment_ids)
        self.assertIn(SEGMENT_IDS["siblings"], masked.token_sequence.segment_ids)
        self.assertIn(SEGMENT_IDS["callees"], masked.token_sequence.segment_ids)
        self.assertIn(SEGMENT_IDS["primary"], masked.token_sequence.segment_ids)

    def test_changed_hunk_mask_is_aligned_to_primary_tokens_only(self) -> None:
        masked = build_masked_codestate(_state())
        changed_tokens = [
            token
            for token, changed in zip(masked.tokens, masked.token_sequence.changed_hunk_mask, strict=True)
            if changed
        ]

        self.assertIn("result", changed_tokens)
        self.assertIn("sqrt", changed_tokens)
        self.assertNotIn("compute", changed_tokens)
        self.assertTrue(
            all(
                segment == SEGMENT_IDS["primary"]
                for segment, changed in zip(
                    masked.token_sequence.segment_ids,
                    masked.token_sequence.changed_hunk_mask,
                    strict=True,
                )
                if changed
            )
        )

    def test_stable_token_ids_are_deterministic_and_positive(self) -> None:
        self.assertEqual(stable_token_id("compute"), stable_token_id("compute"))
        self.assertNotEqual(stable_token_id("compute"), stable_token_id("helper"))
        self.assertGreater(stable_token_id("compute"), 0)

    def test_truncated_primary_keeps_mask_alignment_for_retained_changed_lines(self) -> None:
        primary_lines = ["def compute(value):"]
        primary_lines.extend(f"    scratch_{index} = {index}" for index in range(12))
        primary_lines.append("    changed = value + 1")
        primary_lines.extend(f"    tail_{index} = {index}" for index in range(12))
        primary_lines.append("    return changed")
        state = _state(
            imports="import math\nimport pathlib",
            sibling_signatures=tuple(f"def sibling_{index}()" for index in range(12)),
            callee_signatures=tuple(f"callee_{index}" for index in range(12)),
            primary="\n".join(primary_lines) + "\n",
            changed_hunk_mask=tuple(index == 13 for index in range(len(primary_lines))),
        )

        masked = build_masked_codestate(
            state,
            config=CodeStateNormalizationConfig(token_budget=80, changed_context_lines=1),
        )
        tokens, segments, changed = tokenize_normalized_codestate(state, masked.normalized)

        self.assertEqual(tokens, masked.tokens)
        self.assertEqual(segments, masked.token_sequence.segment_ids)
        self.assertEqual(changed, masked.token_sequence.changed_hunk_mask)
        self.assertIn("changed", [token for token, is_changed in zip(tokens, changed, strict=True) if is_changed])
        self.assertLessEqual(masked.normalized.token_count, 80)


if __name__ == "__main__":
    unittest.main()
