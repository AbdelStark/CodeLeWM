"""Segment IDs and changed-hunk masks for normalized CodeState packs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from codelewm.data.codestate import CodeState
from codelewm.data.normalize import CodeStateNormalizationConfig, NormalizedCodeState, normalize_codestate
from codelewm.data.pack import TokenSequence


SEGMENT_IDS = {
    "path": 1,
    "imports": 2,
    "class": 3,
    "siblings": 4,
    "callees": 5,
    "primary": 6,
}
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class MaskedCodeState:
    """Normalized CodeState with aligned tokens, segment IDs, and hunk masks."""

    normalized: NormalizedCodeState
    tokens: tuple[str, ...]
    token_sequence: TokenSequence


def build_masked_codestate(
    state: CodeState,
    *,
    config: CodeStateNormalizationConfig = CodeStateNormalizationConfig(),
) -> MaskedCodeState:
    normalized = normalize_codestate(state, config=config)
    tokens, segment_ids, changed_mask = tokenize_normalized_codestate(state, normalized)
    token_sequence = TokenSequence(
        input_ids=tuple(stable_token_id(token) for token in tokens),
        attention_mask=tuple(True for _ in tokens),
        segment_ids=segment_ids,
        changed_hunk_mask=changed_mask,
    )
    return MaskedCodeState(
        normalized=normalized,
        tokens=tokens,
        token_sequence=token_sequence,
    )


def tokenize_normalized_codestate(
    state: CodeState,
    normalized: NormalizedCodeState,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[bool, ...]]:
    tokens: list[str] = []
    segment_ids: list[int] = []
    changed_mask: list[bool] = []
    section = "path"
    primary_line_index = 0

    for line in normalized.text.splitlines():
        if line == "<IMPORTS>":
            section = "imports"
        elif line == "<SIBLING_SIGNATURES>":
            section = "siblings"
        elif line == "<CALLEE_SIGNATURES>":
            section = "callees"
        elif line == "<PRIMARY>":
            section = "primary"
            primary_line_index = 0
        elif line.startswith("<KIND") or line.startswith("<ENCLOSING_CLASS"):
            section = "class"
        elif line.startswith(("<PATH", "<MODULE", "<SYMBOL", "<LANG")):
            section = "path"

        line_tokens = _TOKEN_PATTERN.findall(line)
        line_changed = section == "primary" and _primary_line_changed(normalized, primary_line_index)
        tokens.extend(line_tokens)
        segment_ids.extend([SEGMENT_IDS[section]] * len(line_tokens))
        changed_mask.extend([line_changed] * len(line_tokens))

        if section == "primary" and line != "<PRIMARY>":
            primary_line_index += 1

    return tuple(tokens), tuple(segment_ids), tuple(changed_mask)


def stable_token_id(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _primary_line_changed(normalized: NormalizedCodeState, primary_line_index: int) -> bool:
    if primary_line_index >= len(normalized.primary_changed_hunk_mask):
        return False
    return normalized.primary_changed_hunk_mask[primary_line_index]
