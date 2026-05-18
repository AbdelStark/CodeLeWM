from __future__ import annotations

import ast
import unittest

from codelewm.data import (
    DEFAULT_SYNTHETIC_TRANSFORMS,
    RawEditRecord,
    SyntheticSourceFile,
    SyntheticTransform,
    SyntheticTransformError,
    generate_synthetic_records,
)


SOURCE = """\
def compute(value):
    items = set([1, 2, 3])
    total = value + len(items)
    print(total)
"""


class SyntheticTransformTest(unittest.TestCase):
    def test_generates_parse_valid_synthetic_records_with_transform_metadata(self) -> None:
        source = SyntheticSourceFile(
            repo="example/repo",
            path="pkg/example.py",
            contents=SOURCE,
            license="mit",
            source_id="fixture-1",
            split="train",
            metadata={"fixture": True},
        )

        records = generate_synthetic_records(source)

        self.assertEqual(
            [record.metadata["synthetic_transform_id"] for record in records],
            [
                "rename-value-arg-to-result",
                "add-explicit-return-none",
                "modernize-set-literal",
            ],
        )
        self.assertTrue(all(isinstance(record, RawEditRecord) for record in records))
        for record in records:
            self.assertEqual(record.source, "synthetic")
            self.assertEqual(record.repo, "example/repo")
            self.assertEqual(record.path_before, "pkg/example.py")
            self.assertEqual(record.path_after, "pkg/example.py")
            self.assertEqual(record.license, "mit")
            self.assertEqual(record.metadata["source_split"], "train")
            self.assertEqual(record.metadata["source_id"], "fixture-1")
            self.assertEqual(record.metadata["fixture"], True)
            self.assertEqual(record.metadata["synthetic_transform_version"], "1")
            self.assertEqual(len(str(record.metadata["source_digest"])), 64)
            ast.parse(record.before)
            ast.parse(record.after)
            self.assertNotEqual(record.before, record.after)

    def test_transforms_are_deterministic(self) -> None:
        source = SyntheticSourceFile(repo="example/repo", path="pkg/example.py", contents=SOURCE)

        first = generate_synthetic_records(source)
        second = generate_synthetic_records(source)

        self.assertEqual(first, second)
        self.assertEqual([record.commit for record in first], [record.commit for record in second])

    def test_rename_value_argument_transform_updates_parameter_and_uses(self) -> None:
        record = generate_synthetic_records(
            SyntheticSourceFile(repo="example/repo", path="pkg/example.py", contents=SOURCE),
            transforms=(DEFAULT_SYNTHETIC_TRANSFORMS[0],),
        )[0]

        self.assertIn("def compute(result):", record.after)
        self.assertIn("total = result + len(items)", record.after)
        self.assertEqual(record.message, "Rename the local value parameter to result.")

    def test_explicit_return_transform_adds_return_none(self) -> None:
        record = generate_synthetic_records(
            SyntheticSourceFile(repo="example/repo", path="pkg/example.py", contents=SOURCE),
            transforms=(DEFAULT_SYNTHETIC_TRANSFORMS[1],),
        )[0]

        self.assertTrue(record.after.rstrip().endswith("return None"))

    def test_set_literal_transform_modernizes_constant_set_constructor(self) -> None:
        record = generate_synthetic_records(
            SyntheticSourceFile(repo="example/repo", path="pkg/example.py", contents=SOURCE),
            transforms=(DEFAULT_SYNTHETIC_TRANSFORMS[2],),
        )[0]

        self.assertIn("items = {1, 2, 3}", record.after)

    def test_non_python_source_path_is_rejected(self) -> None:
        source = SyntheticSourceFile(repo="example/repo", path="README.md", contents=SOURCE)

        with self.assertRaisesRegex(SyntheticTransformError, "Python file"):
            generate_synthetic_records(source)

    def test_invalid_before_source_is_rejected(self) -> None:
        source = SyntheticSourceFile(
            repo="example/repo",
            path="pkg/example.py",
            contents="def broken(:\n    pass\n",
        )

        with self.assertRaisesRegex(SyntheticTransformError, "before"):
            generate_synthetic_records(source)

    def test_invalid_transform_output_is_rejected(self) -> None:
        bad_transform = SyntheticTransform(
            transform_id="bad",
            version="1",
            instruction="Emit invalid Python.",
            apply=lambda source: "def broken(:\n",
        )

        source = SyntheticSourceFile(repo="example/repo", path="pkg/example.py", contents=SOURCE)

        with self.assertRaisesRegex(SyntheticTransformError, "after:bad"):
            generate_synthetic_records(source, transforms=(bad_transform,))


if __name__ == "__main__":
    unittest.main()
