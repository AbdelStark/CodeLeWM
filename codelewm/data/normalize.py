"""CodeState normalization and structured truncation."""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass

from codelewm.data.codestate import CodeState
from codelewm.security import parse_python_source_text


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class CodeStateNormalizationError(ValueError):
    """Raised when a CodeState cannot fit the token budget without unsafe clipping."""


@dataclass(frozen=True)
class CodeStateNormalizationConfig:
    """Normalization and structured budget policy."""

    token_budget: int = 1024
    large_string_chars: int = 96
    large_number_digits: int = 12
    changed_context_lines: int = 2
    include_docstrings: bool = False

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.changed_context_lines < 0:
            raise ValueError("changed_context_lines must be non-negative")


@dataclass(frozen=True)
class NormalizedCodeState:
    """Packed normalized CodeState text plus truncation evidence."""

    text: str
    token_count: int
    dropped_sections: tuple[str, ...]
    primary_line_count: int
    primary_changed_hunk_mask: tuple[bool, ...]


def normalize_codestate(
    state: CodeState,
    *,
    config: CodeStateNormalizationConfig = CodeStateNormalizationConfig(),
) -> NormalizedCodeState:
    """Normalize a CodeState and enforce the configured token budget."""

    primary = _normalize_primary(state.primary, config=config)
    primary_changed_hunk_mask = _align_primary_mask(state.changed_hunk_mask, primary)
    sibling_signatures = tuple(_normalize_line(item) for item in state.sibling_signatures)
    callee_signatures = tuple(_normalize_line(item) for item in state.callee_signatures)
    imports = _normalize_block(state.imports)
    dropped: list[str] = []

    text = _pack_text(state, primary, imports, sibling_signatures, callee_signatures)
    if _count_tokens(text) <= config.token_budget:
        return _normalized(
            text,
            dropped_sections=dropped,
            primary=primary,
            primary_changed_hunk_mask=primary_changed_hunk_mask,
        )

    sibling_signatures = ()
    dropped.append("sibling_signatures")
    text = _pack_text(state, primary, imports, sibling_signatures, callee_signatures)
    if _count_tokens(text) <= config.token_budget:
        return _normalized(
            text,
            dropped_sections=dropped,
            primary=primary,
            primary_changed_hunk_mask=primary_changed_hunk_mask,
        )

    callee_signatures = ()
    dropped.append("callee_signatures")
    text = _pack_text(state, primary, imports, sibling_signatures, callee_signatures)
    if _count_tokens(text) <= config.token_budget:
        return _normalized(
            text,
            dropped_sections=dropped,
            primary=primary,
            primary_changed_hunk_mask=primary_changed_hunk_mask,
        )

    imports = ""
    dropped.append("imports")
    text = _pack_text(state, primary, imports, sibling_signatures, callee_signatures)
    if _count_tokens(text) <= config.token_budget:
        return _normalized(
            text,
            dropped_sections=dropped,
            primary=primary,
            primary_changed_hunk_mask=primary_changed_hunk_mask,
        )

    primary, primary_changed_hunk_mask = _truncate_primary(
        primary,
        primary_changed_hunk_mask,
        config=config,
    )
    dropped.append("primary_context")
    text = _pack_text(state, primary, imports, sibling_signatures, callee_signatures)
    token_count = _count_tokens(text)
    if token_count > config.token_budget:
        raise CodeStateNormalizationError(
            f"normalized CodeState is {token_count} tokens after structured truncation; "
            f"budget is {config.token_budget}"
        )
    return _normalized(
        text,
        dropped_sections=dropped,
        primary=primary,
        primary_changed_hunk_mask=primary_changed_hunk_mask,
    )


def _normalized(
    text: str,
    *,
    dropped_sections: list[str],
    primary: str,
    primary_changed_hunk_mask: tuple[bool, ...],
) -> NormalizedCodeState:
    normalized = text.rstrip() + "\n"
    return NormalizedCodeState(
        text=normalized,
        token_count=_count_tokens(normalized),
        dropped_sections=tuple(dropped_sections),
        primary_line_count=len(primary.splitlines()),
        primary_changed_hunk_mask=primary_changed_hunk_mask,
    )


def _pack_text(
    state: CodeState,
    primary: str,
    imports: str,
    sibling_signatures: tuple[str, ...],
    callee_signatures: tuple[str, ...],
) -> str:
    symbol = state.symbol or ""
    enclosing = state.enclosing_class or ""
    return "\n".join(
        (
            "<LANG python>",
            f"<PATH {state.path}>",
            f"<MODULE {state.module}>",
            f"<SYMBOL {symbol}>",
            f"<KIND {state.kind}>",
            f"<ENCLOSING_CLASS {enclosing}>",
            "<IMPORTS>",
            imports,
            "<SIBLING_SIGNATURES>",
            "\n".join(sibling_signatures),
            "<CALLEE_SIGNATURES>",
            "\n".join(callee_signatures),
            "<PRIMARY>",
            primary.rstrip(),
        )
    )


def _normalize_primary(primary: str, *, config: CodeStateNormalizationConfig) -> str:
    source = textwrap.dedent(primary).strip()
    if not source:
        return ""
    try:
        tree = parse_python_source_text(source, filename="primary")
    except SyntaxError:
        return _normalize_block(source)
    tree = _LiteralNormalizer(config).visit(tree)
    if not config.include_docstrings:
        tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip() + "\n"


def _normalize_block(block: str) -> str:
    return "\n".join(_normalize_line(line) for line in textwrap.dedent(block).splitlines() if line.strip())


def _normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def _truncate_primary(
    primary: str,
    changed_hunk_mask: tuple[bool, ...],
    *,
    config: CodeStateNormalizationConfig,
) -> tuple[str, tuple[bool, ...]]:
    lines = primary.splitlines()
    if not lines:
        return "", ()
    required = _required_primary_line_indexes(lines, changed_hunk_mask, config.changed_context_lines)
    ordered = sorted(required)
    truncated = "\n".join(lines[index] for index in ordered).rstrip() + "\n"
    truncated_mask = tuple(changed_hunk_mask[index] if index < len(changed_hunk_mask) else False for index in ordered)
    return truncated, truncated_mask


def _required_primary_line_indexes(
    lines: list[str],
    changed_hunk_mask: tuple[bool, ...],
    context_lines: int,
) -> set[int]:
    required: set[int] = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0 or stripped.startswith("@") or stripped.startswith(("def ", "async def ", "class ")):
            required.add(index)
        if stripped.startswith(("return", "raise", "except", "try:", "finally:", "with ")):
            required.add(index)
    for index, changed in enumerate(changed_hunk_mask[: len(lines)]):
        if not changed:
            continue
        start = max(0, index - context_lines)
        end = min(len(lines), index + context_lines + 1)
        required.update(range(start, end))
    return required


def _count_tokens(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


def _align_primary_mask(changed_hunk_mask: tuple[bool, ...], primary: str) -> tuple[bool, ...]:
    line_count = len(primary.splitlines())
    return tuple(changed_hunk_mask[index] if index < len(changed_hunk_mask) else False for index in range(line_count))


class _LiteralNormalizer(ast.NodeTransformer):
    def __init__(self, config: CodeStateNormalizationConfig) -> None:
        self.config = config

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str) and len(node.value) > self.config.large_string_chars:
            return ast.copy_location(ast.Constant(value="<STR_LITERAL>"), node)
        if isinstance(node.value, (int, float, complex)):
            rendered = str(node.value)
            digits = sum(character.isdigit() for character in rendered)
            if digits > self.config.large_number_digits:
                return ast.copy_location(ast.Name(id="__CODELEWM_NUM_LITERAL__", ctx=ast.Load()), node)
        return node


class _DocstringStripper(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = _strip_docstring(node.body)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = _strip_docstring(node.body)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = _strip_docstring(node.body)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = _strip_docstring(node.body)
        return self.generic_visit(node)


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if not body:
        return body
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return body[1:]
    return body
