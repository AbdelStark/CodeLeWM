"""Tests for the execution-substrate probe target label extractors."""

from __future__ import annotations

import unittest

from codelewm.eval import (
    EXECUTION_PROBE_TARGETS,
    ExecutionProbeTargetError,
    extract_labels,
    label_record,
)


class OutputTypeProbeTest(unittest.TestCase):
    def test_known_kinds_are_returned_verbatim(self) -> None:
        for kind in ("int", "float", "str", "list", "exception", "none"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    label_record({"output_type": kind}, "output_type"), kind
                )

    def test_missing_field_returns_none(self) -> None:
        self.assertIsNone(label_record({}, "output_type"))


class WillRaiseProbeTest(unittest.TestCase):
    def test_exception_kind_is_true(self) -> None:
        self.assertEqual(
            label_record({"output_kind": "exception"}, "will_raise"), True
        )

    def test_value_kind_is_false(self) -> None:
        self.assertEqual(
            label_record({"output_kind": "value"}, "will_raise"), False
        )

    def test_stdout_kind_is_false(self) -> None:
        self.assertEqual(
            label_record({"output_kind": "stdout"}, "will_raise"), False
        )

    def test_unknown_kind_is_none(self) -> None:
        self.assertIsNone(label_record({"output_kind": "timeout"}, "will_raise"))


class OutputMagnitudeProbeTest(unittest.TestCase):
    def _make(self, repr_value: str, output_type: str = "int") -> dict[str, str]:
        return {"output_repr": repr_value, "output_type": output_type}

    def test_buckets_cover_signed_range(self) -> None:
        cases = (
            ("-1", "negative"),
            ("0", "zero"),
            ("5", "small"),
            ("100", "medium"),
            ("100000", "large"),
            ("3.14", "small"),
        )
        for repr_value, expected in cases:
            with self.subTest(repr_value=repr_value):
                record = self._make(
                    repr_value,
                    output_type="float" if "." in repr_value else "int",
                )
                self.assertEqual(
                    label_record(record, "output_magnitude_bucket"), expected
                )

    def test_non_numeric_returns_none(self) -> None:
        self.assertIsNone(
            label_record(
                {"output_repr": "'hi'", "output_type": "str"},
                "output_magnitude_bucket",
            )
        )

    def test_bool_excluded(self) -> None:
        # bool subclasses int but is semantically not a magnitude.
        self.assertIsNone(
            label_record(
                {"output_repr": "True", "output_type": "bool"},
                "output_magnitude_bucket",
            )
        )


class OutputLengthProbeTest(unittest.TestCase):
    def test_buckets(self) -> None:
        cases = (
            ("[]", "list", "empty"),
            ("[1, 2]", "list", "short"),
            ("[" + ", ".join(str(i) for i in range(20)) + "]", "list", "medium"),
            ("'hi'", "str", "short"),
            ("''", "str", "empty"),
            ("{'a': 1}", "dict", "short"),
        )
        for repr_value, output_type, expected in cases:
            with self.subTest(repr_value=repr_value):
                self.assertEqual(
                    label_record(
                        {"output_repr": repr_value, "output_type": output_type},
                        "output_length_bucket",
                    ),
                    expected,
                )

    def test_non_collection_returns_none(self) -> None:
        self.assertIsNone(
            label_record(
                {"output_repr": "3", "output_type": "int"},
                "output_length_bucket",
            )
        )


class CoarseKindProbeTest(unittest.TestCase):
    def test_arithmetic_string_collection(self) -> None:
        cases = (
            ("int", "arithmetic"),
            ("float", "arithmetic"),
            ("bool", "arithmetic"),
            ("str", "string"),
            ("bytes", "string"),
            ("list", "collection"),
            ("tuple", "collection"),
            ("dict", "collection"),
            ("set", "collection"),
        )
        for output_type, expected in cases:
            with self.subTest(output_type=output_type):
                self.assertEqual(
                    label_record(
                        {"output_type": output_type, "output_kind": "value"},
                        "arithmetic_vs_string_vs_collection",
                    ),
                    expected,
                )

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(
            label_record(
                {"output_type": "exception", "output_kind": "exception"},
                "arithmetic_vs_string_vs_collection",
            )
        )


class JudgeVerdictProbeTest(unittest.TestCase):
    def test_normalizes_canonical_strings(self) -> None:
        cases = (
            ("accepted", "accepted"),
            ("Accepted", "accepted"),
            ("wrong_answer", "wrong_answer"),
            ("runtime_error", "runtime_error"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(
                    label_record({"judge_verdict": raw}, "judge_verdict"),
                    expected,
                )

    def test_unknown_verdict_is_none(self) -> None:
        self.assertIsNone(
            label_record({"judge_verdict": "unknown"}, "judge_verdict")
        )

    def test_missing_field_is_none(self) -> None:
        self.assertIsNone(label_record({}, "judge_verdict"))


class ExtractLabelsTest(unittest.TestCase):
    def test_class_distribution_counts_applicable(self) -> None:
        records = [
            {"output_type": "int", "output_kind": "value"},
            {"output_type": "str", "output_kind": "value"},
            {"output_type": "int", "output_kind": "value"},
            {"output_kind": "exception"},
        ]
        result = extract_labels(records, target="output_type")
        self.assertEqual(result.target, "output_type")
        self.assertEqual(result.applicable_count, 3)
        self.assertEqual(result.class_distribution, {"int": 2, "str": 1})
        # The fourth record is missing output_type so its label is None.
        self.assertIsNone(result.labels[3])


class TargetRegistryTest(unittest.TestCase):
    def test_unknown_target_raises(self) -> None:
        with self.assertRaises(ExecutionProbeTargetError):
            label_record({"output_type": "int"}, "does_not_exist")

    def test_known_targets_resolve_for_minimal_record(self) -> None:
        record = {
            "output_type": "int",
            "output_kind": "value",
            "output_repr": "5",
            "judge_verdict": "accepted",
        }
        for target in EXECUTION_PROBE_TARGETS:
            with self.subTest(target=target):
                value = label_record(record, target)
                if target == "output_length_bucket":
                    # ints are not in the length-bucket domain.
                    self.assertIsNone(value)
                else:
                    self.assertIsNotNone(value)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
