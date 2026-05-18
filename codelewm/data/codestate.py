"""AST-based CodeState extraction for Python edit records."""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from codelewm.data.sources import RawEditRecord


CodeStateKind = Literal["function", "method", "class", "region", "small_file"]


class CodeStateExtractionError(ValueError):
    """Raised when source text cannot be parsed into a CodeState."""


@dataclass(frozen=True)
class CodeStateConfig:
    """Configuration for AST-based CodeState extraction."""

    max_small_file_lines: int = 40
    region_context_lines: int = 8


@dataclass(frozen=True)
class CodeState:
    """Deterministic context capsule for one Python code state."""

    language: Literal["python"]
    path: str
    module: str
    symbol: str | None
    kind: CodeStateKind
    imports: str
    enclosing_class: str | None
    sibling_signatures: tuple[str, ...]
    callee_signatures: tuple[str, ...]
    primary: str
    token_count: int
    changed_hunk_mask: tuple[bool, ...]
    fallback_reason: str | None = None


@dataclass(frozen=True)
class CodeStatePair:
    """Before/after CodeState pair extracted from one raw edit record."""

    before: CodeState
    after: CodeState


@dataclass(frozen=True)
class _Candidate:
    node: ast.AST
    kind: CodeStateKind
    symbol: str
    enclosing_class: str | None
    start_line: int
    end_line: int
    parent: ast.AST

    @property
    def span(self) -> int:
        return self.end_line - self.start_line + 1


def extract_codestate_pair(
    record: RawEditRecord,
    *,
    config: CodeStateConfig = CodeStateConfig(),
) -> CodeStatePair:
    before_changed, after_changed = changed_line_numbers(record.before, record.after)
    before = extract_codestate(
        record.before,
        path=record.path_before,
        changed_lines=before_changed,
        config=config,
        field_name="before",
    )
    after = extract_codestate(
        record.after,
        path=record.path_after,
        changed_lines=after_changed,
        config=config,
        field_name="after",
    )
    return CodeStatePair(before=before, after=after)


def extract_codestate(
    source: str,
    *,
    path: str,
    changed_lines: set[int],
    config: CodeStateConfig = CodeStateConfig(),
    field_name: str = "source",
) -> CodeState:
    tree = _parse_python(source, field_name=field_name)
    lines = source.splitlines()
    candidates = _candidate_nodes(tree)
    candidate = _best_candidate(candidates, changed_lines)
    imports = _imports(source, tree)
    module = module_name_from_path(path)

    if candidate is not None:
        primary = _source_lines(lines, candidate.start_line, candidate.end_line)
        return CodeState(
            language="python",
            path=path,
            module=module,
            symbol=candidate.symbol,
            kind=candidate.kind,
            imports=imports,
            enclosing_class=candidate.enclosing_class,
            sibling_signatures=_sibling_signatures(candidate),
            callee_signatures=_callee_signatures(candidate.node),
            primary=primary,
            token_count=_token_count(primary),
            changed_hunk_mask=_changed_mask(candidate.start_line, candidate.end_line, changed_lines),
        )

    if len(lines) <= config.max_small_file_lines:
        primary = source.rstrip() + ("\n" if source else "")
        return CodeState(
            language="python",
            path=path,
            module=module,
            symbol=None,
            kind="small_file",
            imports=imports,
            enclosing_class=None,
            sibling_signatures=_module_signatures(tree),
            callee_signatures=_callee_signatures(tree),
            primary=primary,
            token_count=_token_count(primary),
            changed_hunk_mask=_changed_mask(1, max(len(lines), 1), changed_lines),
            fallback_reason="small_file_no_symbol_overlap" if changed_lines else "small_file_no_changed_lines",
        )

    start, end = _region_bounds(lines, changed_lines, config.region_context_lines)
    primary = _source_lines(lines, start, end)
    return CodeState(
        language="python",
        path=path,
        module=module,
        symbol=None,
        kind="region",
        imports=imports,
        enclosing_class=None,
        sibling_signatures=_module_signatures(tree),
        callee_signatures=_callee_signatures(tree),
        primary=primary,
        token_count=_token_count(primary),
        changed_hunk_mask=_changed_mask(start, end, changed_lines),
        fallback_reason="changed_region_no_symbol_overlap" if changed_lines else "region_no_changed_lines",
    )


def changed_line_numbers(before: str, after: str) -> tuple[set[int], set[int]]:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    before_changed: set[int] = set()
    after_changed: set[int] = set()
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_changed.update(range(before_start + 1, before_end + 1))
        after_changed.update(range(after_start + 1, after_end + 1))
    return before_changed, after_changed


def module_name_from_path(path: str) -> str:
    pure = PurePosixPath(path.replace("\\", "/"))
    if pure.suffix == ".py":
        pure = pure.with_suffix("")
    return ".".join(part for part in pure.parts if part and part != "__init__")


def _parse_python(source: str, *, field_name: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise CodeStateExtractionError(
            f"{field_name} source is not parse-valid Python: {exc.msg}"
        ) from exc


def _candidate_nodes(tree: ast.Module) -> list[_Candidate]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    candidates: list[_Candidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if start_line is None or end_line is None:
            continue
        parent = parents.get(node, tree)
        enclosing_class = _enclosing_class(node, parents)
        if isinstance(node, ast.ClassDef):
            symbol = node.name
            kind: CodeStateKind = "class"
        elif enclosing_class is not None:
            symbol = f"{enclosing_class}.{node.name}"
            kind = "method"
        else:
            symbol = node.name
            kind = "function"
        candidates.append(
            _Candidate(
                node=node,
                kind=kind,
                symbol=symbol,
                enclosing_class=enclosing_class,
                start_line=start_line,
                end_line=end_line,
                parent=parent,
            )
        )
    return candidates


def _best_candidate(candidates: list[_Candidate], changed_lines: set[int]) -> _Candidate | None:
    if not changed_lines:
        return None
    ranked: list[tuple[int, int, int, _Candidate]] = []
    for candidate in candidates:
        overlap = len(changed_lines & set(range(candidate.start_line, candidate.end_line + 1)))
        if overlap == 0:
            continue
        ranked.append((-overlap, candidate.span, candidate.start_line, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[:3])
    return ranked[0][3]


def _enclosing_class(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.ClassDef):
            return parent.name
        parent = parents.get(parent)
    return None


def _imports(source: str, tree: ast.Module) -> str:
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(source, node) or ast.unparse(node))
    return "\n".join(imports)


def _sibling_signatures(candidate: _Candidate) -> tuple[str, ...]:
    body = getattr(candidate.parent, "body", ())
    signatures = []
    for node in body:
        if node is candidate.node:
            continue
        signature = _signature(node)
        if signature is not None:
            signatures.append(signature)
    return tuple(signatures)


def _module_signatures(tree: ast.Module) -> tuple[str, ...]:
    signatures = [_signature(node) for node in tree.body]
    return tuple(signature for signature in signatures if signature is not None)


def _signature(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({ast.unparse(node.args)})"
    if isinstance(node, ast.ClassDef):
        if node.bases:
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            return f"class {node.name}({bases})"
        return f"class {node.name}"
    return None


def _callee_signatures(node: ast.AST) -> tuple[str, ...]:
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            calls.add(_call_name(child.func))
    return tuple(sorted(call for call in calls if call))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _source_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1 : end]).rstrip() + "\n"


def _changed_mask(start: int, end: int, changed_lines: set[int]) -> tuple[bool, ...]:
    return tuple(line_number in changed_lines for line_number in range(start, end + 1))


def _region_bounds(lines: list[str], changed_lines: set[int], context_lines: int) -> tuple[int, int]:
    if not changed_lines:
        return 1, min(len(lines), max(1, context_lines * 2))
    start = max(1, min(changed_lines) - context_lines)
    end = min(len(lines), max(changed_lines) + context_lines)
    return start, max(start, end)


def _token_count(source: str) -> int:
    return len(source.split())
