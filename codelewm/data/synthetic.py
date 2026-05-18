"""Deterministic synthetic Python edit transforms."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from codelewm.data.sources import RawEditRecord
from codelewm.data.split_dedup import SplitName


class SyntheticTransformError(ValueError):
    """Raised when a synthetic source or transform is invalid."""


@dataclass(frozen=True)
class SyntheticSourceFile:
    """Input file for deterministic synthetic transition generation."""

    repo: str
    path: str
    contents: str
    license: str | None = None
    source_id: str | None = None
    split: SplitName | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticTransform:
    """One deterministic text-to-text source transform."""

    transform_id: str
    version: str
    instruction: str
    apply: Callable[[str], str | None]


DEFAULT_SYNTHETIC_TRANSFORMS: tuple[SyntheticTransform, ...] = (
    SyntheticTransform(
        transform_id="rename-value-arg-to-result",
        version="1",
        instruction="Rename the local value parameter to result.",
        apply=lambda source: _rename_value_argument(source),
    ),
    SyntheticTransform(
        transform_id="add-explicit-return-none",
        version="1",
        instruction="Add an explicit return None to a function with implicit None return.",
        apply=lambda source: _add_explicit_return_none(source),
    ),
    SyntheticTransform(
        transform_id="modernize-set-literal",
        version="1",
        instruction="Replace set([...]) with a set literal.",
        apply=lambda source: _modernize_set_literal(source),
    ),
)


def generate_synthetic_records(
    source: SyntheticSourceFile,
    *,
    transforms: Iterable[SyntheticTransform] = DEFAULT_SYNTHETIC_TRANSFORMS,
) -> tuple[RawEditRecord, ...]:
    """Generate deterministic parse-valid synthetic edit rows for one file."""

    _validate_source(source)
    source_digest = _source_digest(source.contents)
    records: list[RawEditRecord] = []

    for transform in transforms:
        after = transform.apply(source.contents)
        if after is None or _normalize_source(after) == _normalize_source(source.contents):
            continue
        _parse_python(after, field_name=f"after:{transform.transform_id}")

        metadata = dict(source.metadata)
        metadata.update(
            {
                "synthetic_transform_id": transform.transform_id,
                "synthetic_transform_version": transform.version,
                "source_digest": source_digest,
                "source_id": source.source_id or f"{source.repo}:{source.path}",
            }
        )
        if source.split is not None:
            metadata["source_split"] = source.split

        records.append(
            RawEditRecord(
                source="synthetic",
                repo=source.repo,
                commit=f"synthetic:{source_digest[:12]}:{transform.transform_id}",
                path_before=source.path,
                path_after=source.path,
                before=source.contents,
                after=after,
                message=transform.instruction,
                license=source.license,
                metadata=metadata,
            )
        )

    return tuple(records)


def _validate_source(source: SyntheticSourceFile) -> None:
    if not source.path.endswith(".py"):
        raise SyntheticTransformError(f"synthetic source must be a Python file: {source.path}")
    _parse_python(source.contents, field_name="before")


def _parse_python(source: str, *, field_name: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise SyntheticTransformError(
            f"{field_name} source is not parse-valid Python: {exc.msg}"
        ) from exc


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _normalize_source(source: str) -> str:
    return "\n".join(line.rstrip() for line in source.splitlines()).strip()


def _unparse(tree: ast.AST) -> str:
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip() + "\n"


def _rename_value_argument(source: str) -> str | None:
    tree = _parse_python(source, field_name="before")
    transformer = _RenameValueArgument()
    transformed = transformer.visit(tree)
    if not transformer.changed:
        return None
    return _unparse(transformed)


def _add_explicit_return_none(source: str) -> str | None:
    tree = _parse_python(source, field_name="before")
    transformer = _AddExplicitReturnNone()
    transformed = transformer.visit(tree)
    if not transformer.changed:
        return None
    return _unparse(transformed)


def _modernize_set_literal(source: str) -> str | None:
    tree = _parse_python(source, field_name="before")
    transformer = _ModernizeSetLiteral()
    transformed = transformer.visit(tree)
    if not transformer.changed:
        return None
    return _unparse(transformed)


class _RenameValueArgument(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if self.changed or _has_nested_scope(node):
            return node
        if not _can_rename_value_argument(node):
            return self.generic_visit(node)
        self.changed = True
        return _LocalRename("value", "result").visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if self.changed or _has_nested_scope(node):
            return node
        if not _can_rename_value_argument(node):
            return self.generic_visit(node)
        self.changed = True
        return _LocalRename("value", "result").visit(node)


class _LocalRename(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg == self.old:
            node.arg = self.new
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == self.old:
            node.id = self.new
        return node


class _AddExplicitReturnNone(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if self.changed or _has_nested_scope(node):
            return node
        node = self.generic_visit(node)
        if _has_return(node) or _ends_with_return(node):
            return node
        node.body.append(ast.Return(value=ast.Constant(value=None)))
        self.changed = True
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if self.changed or _has_nested_scope(node):
            return node
        node = self.generic_visit(node)
        if _has_return(node) or _ends_with_return(node):
            return node
        node.body.append(ast.Return(value=ast.Constant(value=None)))
        self.changed = True
        return node


class _ModernizeSetLiteral(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if self.changed:
            return node
        if not isinstance(node.func, ast.Name) or node.func.id != "set" or node.keywords:
            return node
        if len(node.args) != 1 or not isinstance(node.args[0], (ast.List, ast.Tuple)):
            return node
        if not node.args[0].elts or not all(isinstance(item, ast.Constant) for item in node.args[0].elts):
            return node
        self.changed = True
        return ast.copy_location(ast.Set(elts=node.args[0].elts), node)


def _can_rename_value_argument(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if _has_nested_scope(node):
        return False
    names = _scope_names(node)
    arguments = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
    return "value" in arguments and "result" not in names


def _scope_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.arg):
            names.add(child.arg)
        elif isinstance(child, ast.Name):
            names.add(child.id)
    return names


def _has_nested_scope(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return True
    return False


def _has_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(isinstance(child, ast.Return) for child in ast.walk(node) if child is not node)


def _ends_with_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return bool(node.body and isinstance(node.body[-1], ast.Return))
