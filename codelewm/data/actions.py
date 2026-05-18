"""Text, abstract, and diagnostic patch action extraction."""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field

from codelewm.data.codestate import changed_line_numbers
from codelewm.data.sources import RawEditRecord
from codelewm.security import parse_python_source_text


class ActionExtractionError(ValueError):
    """Raised when a required action view cannot be extracted."""


@dataclass(frozen=True)
class EditAction:
    """Action views for one code edit transition."""

    text: str
    abstract: tuple[str, ...]
    patch: str | None = None
    patch_is_leaky: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionExtractionConfig:
    """Configuration for action view extraction."""

    max_text_chars: int = 256
    include_patch: bool = False
    max_abstract_ops: int = 32


@dataclass(frozen=True)
class _NodeInfo:
    node_type: str
    fingerprint: str
    shape: str
    scope: str
    depth: int
    rename_value: str | None


def extract_edit_action(
    record: RawEditRecord,
    *,
    config: ActionExtractionConfig = ActionExtractionConfig(),
) -> EditAction:
    text = _text_action(record, max_chars=config.max_text_chars)
    abstract = extract_abstract_actions(
        record.before,
        record.after,
        max_ops=config.max_abstract_ops,
    )
    patch = _patch_action(record) if config.include_patch else None
    metadata: dict[str, object] = {}
    if patch is not None:
        metadata["action_patch_leaky"] = True
    if record.metadata.get("synthetic_transform_id"):
        metadata["synthetic_transform_id"] = record.metadata["synthetic_transform_id"]
    return EditAction(
        text=text,
        abstract=abstract,
        patch=patch,
        patch_is_leaky=patch is not None,
        metadata=metadata,
    )


def extract_abstract_actions(before: str, after: str, *, max_ops: int = 32) -> tuple[str, ...]:
    before_tree = _parse(before, field_name="before")
    after_tree = _parse(after, field_name="after")
    before_nodes = _node_map(before_tree)
    after_nodes = _node_map(after_tree)
    before_changed, after_changed = changed_line_numbers(before, after)
    size = _size_bucket(len(before_changed) + len(after_changed))

    actions: list[str] = []
    for path in sorted(set(before_nodes) | set(after_nodes)):
        old = before_nodes.get(path)
        new = after_nodes.get(path)
        if old is None and new is not None:
            actions.append(_action_token("OP_INSERT", new, None, new, size))
        elif old is not None and new is None:
            actions.append(_action_token("OP_DELETE", old, old, None, size))
        elif old is not None and new is not None:
            if old.rename_value is not None and new.rename_value is not None and old.rename_value != new.rename_value:
                actions.append(_action_token("OP_RENAME", new, old, new, size))
            elif old.node_type != new.node_type or old.fingerprint != new.fingerprint:
                actions.append(_action_token("OP_UPDATE", new, old, new, size))
        if len(actions) >= max_ops:
            break
    return tuple(actions)


def _text_action(record: RawEditRecord, *, max_chars: int) -> str:
    text = " ".join(record.message.split())
    if not text:
        synthetic_id = record.metadata.get("synthetic_transform_id")
        if synthetic_id:
            text = f"Apply synthetic transform {synthetic_id}"
    if not text:
        raise ActionExtractionError("text action is empty")
    return text[:max_chars]


def _patch_action(record: RawEditRecord) -> str:
    return "".join(
        difflib.unified_diff(
            record.before.splitlines(keepends=True),
            record.after.splitlines(keepends=True),
            fromfile=record.path_before,
            tofile=record.path_after,
            lineterm="",
        )
    )


def _parse(source: str, *, field_name: str) -> ast.Module:
    try:
        return parse_python_source_text(source, filename=field_name)
    except SyntaxError as exc:
        raise ActionExtractionError(f"{field_name} source is not parse-valid Python: {exc.msg}") from exc


def _node_map(tree: ast.AST) -> dict[tuple[str, ...], _NodeInfo]:
    nodes: dict[tuple[str, ...], _NodeInfo] = {}

    def visit(node: ast.AST, path: tuple[str, ...], scope: str) -> None:
        next_scope = _scope_for(node, scope)
        nodes[path] = _NodeInfo(
            node_type=type(node).__name__,
            fingerprint=_fingerprint(node),
            shape=_shape(node),
            scope=next_scope,
            depth=len(path),
            rename_value=_rename_value(node),
        )
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit(value, (*path, field_name, type(value).__name__), next_scope)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        visit(item, (*path, field_name, str(index), type(item).__name__), next_scope)

    visit(tree, ("Module",), "module")
    return nodes


def _scope_for(node: ast.AST, current: str) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "function"
    if isinstance(node, ast.ClassDef):
        return "class"
    return current


def _fingerprint(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _shape(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return "CallWithKeyword" if node.keywords else "Call"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return "ConstantStr"
        if isinstance(node.value, (int, float, complex)):
            return "ConstantNum"
        return "Constant"
    return type(node).__name__


def _rename_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.arg):
        return node.arg
    return None


def _action_token(
    op: str,
    node: _NodeInfo,
    old: _NodeInfo | None,
    new: _NodeInfo | None,
    size: str,
) -> str:
    old_shape = "None" if old is None else old.shape
    new_shape = "None" if new is None else new.shape
    return " ".join(
        (
            op,
            f"NODE_{node.node_type}",
            f"PATH_DEPTH_{_depth_bucket(node.depth)}",
            f"OLD_{old_shape}",
            f"NEW_{new_shape}",
            f"SCOPE_{node.scope}",
            f"SIZE_{size}",
        )
    )


def _depth_bucket(depth: int) -> str:
    if depth <= 2:
        return "1"
    if depth <= 4:
        return "2"
    if depth <= 8:
        return "4"
    return "8PLUS"


def _size_bucket(changed_lines: int) -> str:
    if changed_lines <= 2:
        return "SMALL"
    if changed_lines <= 10:
        return "MEDIUM"
    return "LARGE"
