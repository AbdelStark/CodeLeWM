from __future__ import annotations

import unittest

from codelewm.data.wsd_mutations import (
    Mutant,
    enumerate_single_point_mutants,
    generate_mutants,
)


_REFERENCE = """def f(nums):
    total = 0
    for x in nums:
        if x > 0:
            total += x
    return total
"""


def _run(source: str, arg):
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - test-only evaluation of generated code
    return namespace["f"](arg)


class WsdMutationsTest(unittest.TestCase):
    def test_enumerates_distinct_behavior_changing_mutants(self) -> None:
        mutants = enumerate_single_point_mutants(_REFERENCE)
        self.assertGreater(len(mutants), 3)
        # all distinct
        self.assertEqual(len({m.source for m in mutants}), len(mutants))
        # none equal the original
        for m in mutants:
            self.assertNotEqual(m.source.strip(), _REFERENCE.strip())
        # at least one mutant actually changes behavior on a discriminating input
        ref = _run(_REFERENCE, [1, -2, 3])
        changed = 0
        for m in mutants:
            try:
                if _run(m.source, [1, -2, 3]) != ref:
                    changed += 1
            except Exception:
                changed += 1
        self.assertGreater(changed, 0)

    def test_operators_present(self) -> None:
        ops = {m.operator for m in enumerate_single_point_mutants(_REFERENCE)}
        # the reference has a comparison, an augmented assignment, and a constant
        self.assertIn("compare", ops)
        self.assertIn("augassign", ops)
        self.assertIn("const", ops)

    def test_generate_is_deterministic(self) -> None:
        a = generate_mutants(_REFERENCE, count=4, seed=17)
        b = generate_mutants(_REFERENCE, count=4, seed=17)
        self.assertEqual([m.source for m in a], [m.source for m in b])
        self.assertLessEqual(len(a), 4)
        for m in a:
            self.assertIsInstance(m, Mutant)

    def test_generate_different_seeds_can_differ(self) -> None:
        pool = enumerate_single_point_mutants(_REFERENCE)
        # only meaningful when there are more sites than the sample count
        if len(pool) > 4:
            a = generate_mutants(_REFERENCE, count=4, seed=1)
            b = generate_mutants(_REFERENCE, count=4, seed=2)
            self.assertNotEqual([m.source for m in a], [m.source for m in b])

    def test_count_exceeding_pool_returns_pool(self) -> None:
        pool = enumerate_single_point_mutants(_REFERENCE)
        got = generate_mutants(_REFERENCE, count=10_000, seed=0)
        self.assertEqual(len(got), len(pool))

    def test_zero_count_returns_empty(self) -> None:
        self.assertEqual(generate_mutants(_REFERENCE, count=0, seed=0), [])

    def test_syntax_error_source_returns_empty(self) -> None:
        self.assertEqual(enumerate_single_point_mutants("def f(:\n  pass"), [])
        self.assertEqual(generate_mutants("def f(:\n  pass", count=3, seed=0), [])

    def test_source_without_sites_returns_empty(self) -> None:
        # no comparisons / binops / mutable constants
        self.assertEqual(enumerate_single_point_mutants("def f(x):\n    return x\n"), [])


if __name__ == "__main__":
    unittest.main()
