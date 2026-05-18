from __future__ import annotations

import unittest

from codelewm.data import (
    ActionExtractionConfig,
    ActionExtractionError,
    RawEditRecord,
    extract_abstract_actions,
    extract_edit_action,
)


def _record(before: str, after: str, *, message: str = "Update code", metadata: dict[str, object] | None = None) -> RawEditRecord:
    return RawEditRecord(
        source="local_repo",
        repo="example/repo",
        commit="abc123",
        path_before="pkg/mod.py",
        path_after="pkg/mod.py",
        before=before,
        after=after,
        message=message,
        license="mit",
        metadata={} if metadata is None else metadata,
    )


class ActionExtractionTest(unittest.TestCase):
    def test_text_action_uses_message_and_truncates_to_budget(self) -> None:
        action = extract_edit_action(
            _record("x = 1\n", "x = 2\n", message="  " + "a" * 300),
            config=ActionExtractionConfig(max_text_chars=16),
        )

        self.assertEqual(action.text, "a" * 16)

    def test_text_action_can_fall_back_to_synthetic_template(self) -> None:
        action = extract_edit_action(
            _record(
                "x = 1\n",
                "x = 2\n",
                message="",
                metadata={"synthetic_transform_id": "rename-value-arg-to-result"},
            )
        )

        self.assertIn("rename-value-arg-to-result", action.text)
        self.assertEqual(action.metadata["synthetic_transform_id"], "rename-value-arg-to-result")

    def test_empty_text_action_without_template_is_error(self) -> None:
        with self.assertRaisesRegex(ActionExtractionError, "empty"):
            extract_edit_action(_record("x = 1\n", "x = 2\n", message=""))

    def test_update_action_omits_inserted_after_code_text(self) -> None:
        actions = extract_abstract_actions("def f():\n    return 1\n", "def f():\n    return 2\n")

        self.assertTrue(any(action.startswith("OP_UPDATE") for action in actions))
        self.assertFalse(any("return 2" in action for action in actions))

    def test_insert_action_detects_exception_handler(self) -> None:
        before = """\
def f():
    try:
        risky()
    finally:
        cleanup()
"""
        after = """\
def f():
    try:
        risky()
    except ValueError:
        handle()
    finally:
        cleanup()
"""

        actions = extract_abstract_actions(before, after)

        self.assertTrue(any(action.startswith("OP_INSERT NODE_ExceptHandler") for action in actions))
        self.assertFalse(any("handle()" in action for action in actions))

    def test_delete_action_detects_removed_assignment(self) -> None:
        actions = extract_abstract_actions(
            "def f():\n    unused = 1\n    return 2\n",
            "def f():\n    return 2\n",
        )

        self.assertTrue(any(action.startswith("OP_DELETE NODE_Assign") for action in actions))

    def test_rename_action_detects_argument_and_name_renames(self) -> None:
        actions = extract_abstract_actions(
            "def f(value):\n    return value\n",
            "def f(result):\n    return result\n",
        )

        self.assertTrue(any(action.startswith("OP_RENAME") for action in actions))
        self.assertFalse(any("result" in action for action in actions))

    def test_patch_action_is_marked_leaky_when_enabled(self) -> None:
        action = extract_edit_action(
            _record("def f():\n    return 1\n", "def f():\n    return 2\n"),
            config=ActionExtractionConfig(include_patch=True),
        )

        self.assertTrue(action.patch_is_leaky)
        self.assertEqual(action.metadata["action_patch_leaky"], True)
        self.assertIn("-    return 1", action.patch or "")
        self.assertIn("+    return 2", action.patch or "")


if __name__ == "__main__":
    unittest.main()
