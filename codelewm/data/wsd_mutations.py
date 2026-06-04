"""WS-D mutation operators for unsaturated reranking candidate pools.

The v0.6/v0.7 downstream rerank benchmarks *saturate*: a frontier LLM passes
~95% of its own candidates, so fewer than 4% of problems carry a pass/fail
mix and reranking has no headroom (it can only change pass@1 on a problem
whose candidate pool contains both a passing and a failing completion).

WS-D builds candidate pools a different way: take a *known-correct* reference
solution and derive plausible near-miss variants with single-point,
behavior-changing AST mutations. A pool then mixes the passing reference with
lexically-similar failing distractors, so

- most problems have rerank headroom (pass/fail mix by construction), and
- the lexical / token-overlap baseline cannot discriminate (the distractors
  differ from the reference by one operator or constant) — only a model that
  understands *execution* can pick the passing candidate.

Mutations are deterministic given a seed, so the benchmark is reproducible
with no LLM and no provider budget. This module only *generates* candidate
source; labeling (sandbox execution against hidden tests) is done by the
existing completion-label pipeline.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass

__all__ = ["Mutant", "enumerate_single_point_mutants", "generate_mutants"]


# Each operator class maps to the list of replacements we try at that site.
# Replacements include both boundary tweaks (Lt->LtE, off-by-one) and strong
# inversions (Lt->Gt) so the failing-distractor yield is high once labeled.
_COMPARE_OPTIONS: dict[type[ast.cmpop], list[type[ast.cmpop]]] = {
    ast.Lt: [ast.LtE, ast.Gt, ast.GtE],
    ast.LtE: [ast.Lt, ast.Gt, ast.GtE],
    ast.Gt: [ast.GtE, ast.Lt, ast.LtE],
    ast.GtE: [ast.Gt, ast.Lt, ast.LtE],
    ast.Eq: [ast.NotEq],
    ast.NotEq: [ast.Eq],
    ast.Is: [ast.IsNot],
    ast.IsNot: [ast.Is],
}
_BINOP_OPTIONS: dict[type[ast.operator], list[type[ast.operator]]] = {
    ast.Add: [ast.Sub],
    ast.Sub: [ast.Add],
    ast.Mult: [ast.FloorDiv, ast.Add],
    ast.Div: [ast.Mult],
    ast.FloorDiv: [ast.Mult],
    ast.Mod: [ast.FloorDiv],
}
_BOOLOP_OPTIONS: dict[type[ast.boolop], list[type[ast.boolop]]] = {
    ast.And: [ast.Or],
    ast.Or: [ast.And],
}
_AUGOP_OPTIONS: dict[type[ast.operator], list[type[ast.operator]]] = {
    ast.Add: [ast.Sub],
    ast.Sub: [ast.Add],
    ast.Mult: [ast.FloorDiv],
}
_CONST_DELTAS = (1, -1)


@dataclass(frozen=True)
class Mutant:
    """A single-point mutation of a reference solution."""

    source: str
    operator: str
    description: str


@dataclass(frozen=True)
class _Plan:
    """One concrete mutation: apply ``payload`` at the ``index``-th eligible site."""

    index: int
    operator: str
    description: str
    payload: object


class _Mutator(ast.NodeTransformer):
    """Single shared traversal used for both enumeration and application.

    In enumeration mode (``target is None``) it records a :class:`_Plan` for
    every eligible (node, replacement) option without mutating. In application
    mode it mutates exactly the site whose running index equals ``target`` and
    leaves everything else untouched. Sharing one traversal guarantees the
    enumeration index and the application index always refer to the same site.
    """

    def __init__(self, *, target: int | None) -> None:
        self.target = target
        self.cursor = 0
        self.plans: list[_Plan] = []

    def _options(self, kind, opts, description_fn):
        """Yield/consume one site index per replacement option."""
        out = []
        for payload in opts:
            idx = self.cursor
            self.cursor += 1
            if self.target is None:
                self.plans.append(
                    _Plan(idx, kind, description_fn(payload), payload)
                )
            out.append((idx, payload))
        return out

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        new_ops = list(node.ops)
        for op_index, op in enumerate(node.ops):
            opts = _COMPARE_OPTIONS.get(type(op))
            if not opts:
                continue
            for idx, repl in self._options(
                "compare", opts,
                lambda r, o=op: f"{type(o).__name__}->{r.__name__}",
            ):
                if idx == self.target:
                    new_ops[op_index] = repl()
        node.ops = new_ops
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        opts = _BINOP_OPTIONS.get(type(node.op))
        if opts:
            for idx, repl in self._options(
                "binop", opts,
                lambda r, o=node.op: f"{type(o).__name__}->{r.__name__}",
            ):
                if idx == self.target:
                    node.op = repl()
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        opts = _BOOLOP_OPTIONS.get(type(node.op))
        if opts:
            for idx, repl in self._options(
                "boolop", opts,
                lambda r, o=node.op: f"{type(o).__name__}->{r.__name__}",
            ):
                if idx == self.target:
                    node.op = repl()
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        opts = _AUGOP_OPTIONS.get(type(node.op))
        if opts:
            for idx, repl in self._options(
                "augassign", opts,
                lambda r, o=node.op: f"aug {type(o).__name__}->{r.__name__}",
            ):
                if idx == self.target:
                    node.op = repl()
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            for idx, _ in self._options("not", [None], lambda _: "remove `not`"):
                if idx == self.target:
                    return node.operand
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            for idx, delta in self._options(
                "const", list(_CONST_DELTAS),
                lambda d, v=node.value: f"const {v!r}{d:+d}",
            ):
                if idx == self.target:
                    return ast.copy_location(
                        ast.Constant(value=node.value + delta), node
                    )
        return node


def enumerate_single_point_mutants(source: str) -> list[Mutant]:
    """Return every distinct single-point mutant of ``source`` (deterministic order)."""

    try:
        ast.parse(source)
    except SyntaxError:
        return []
    enum = _Mutator(target=None)
    enum.visit(ast.parse(source))
    normalized_original = _normalize(source)
    seen: set[str] = set()
    mutants: list[Mutant] = []
    for plan in enum.plans:
        try:
            mutated = ast.parse(source)
            _Mutator(target=plan.index).visit(mutated)
            ast.fix_missing_locations(mutated)
            mutated_source = ast.unparse(mutated)
        except (SyntaxError, ValueError, RecursionError):
            continue
        if _normalize(mutated_source) == normalized_original:
            continue
        if mutated_source in seen:
            continue
        seen.add(mutated_source)
        mutants.append(
            Mutant(source=mutated_source, operator=plan.operator, description=plan.description)
        )
    return mutants


def generate_mutants(source: str, *, count: int, seed: int) -> list[Mutant]:
    """Deterministically sample up to ``count`` distinct mutants of ``source``.

    Sampling is seeded, so the same (source, count, seed) always yields the
    same mutants. Returns fewer than ``count`` only when the reference has too
    few applicable sites.
    """

    if count <= 0:
        return []
    pool = enumerate_single_point_mutants(source)
    if len(pool) <= count:
        return pool
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pool)), count))
    return [pool[i] for i in indices]


def _normalize(source: str) -> str:
    """Round-trip through the AST so cosmetic differences don't count as mutations."""
    try:
        return ast.unparse(ast.parse(source))
    except SyntaxError:
        return source
