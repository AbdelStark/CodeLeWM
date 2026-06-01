"""Surprise-eval decoy generators specific to the execution substrate.

The existing :mod:`codelewm.eval.surprise` module covers the commit-edit
substrate's decoy categories (random, same-file, mutation,
action_cluster). The v0.6 substrate (#259, RFC-0014) needs two
additional decoy categories that test specifically whether the latent
encodes program semantics rather than surface code similarity:

- ``same_problem_different_submission`` — for a query
  ``(code_A, input_X, output_A_X)``, the decoy is the output of a
  different submission for the **same** problem on the **same**
  input. If outputs match (both submissions are correct on this
  input), the pair is filtered. We want pairs where outputs differ
  (one is wrong, or edge cases diverge).
- ``same_code_different_input`` — for a query
  ``(code, input_X, output_X)``, the decoy is the output of the
  **same** code on a **different** input. Query and decoy share code
  and input tokens; only the input conditioning should distinguish
  them — the strongest possible test of input-sensitivity.

The module operates on plain ``dict`` records (pack.jsonl lines). It
returns deterministic pair lists that the downstream
:func:`codelewm.eval.surprise` AUC machinery can consume.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal


EXECUTION_SURPRISE_DECOY_SCHEMA_VERSION = "codelewm.eval.execution_surprise_decoy.v1"

EXECUTION_SURPRISE_DECOY_CATEGORIES: tuple[str, ...] = (
    "same_problem_different_submission",
    "same_code_different_input",
)

ExecutionSurpriseCategory = Literal[
    "same_problem_different_submission",
    "same_code_different_input",
]


class ExecutionSurpriseDecoyError(ValueError):
    """Raised when a decoy generator cannot satisfy its contract."""


@dataclass(frozen=True)
class DecoyPair:
    """One query-decoy pair.

    ``query_record_id`` and ``decoy_record_id`` are the ``record_id``
    fields of the corresponding packed records. ``category`` identifies
    which decoy generator produced the pair.
    """

    category: str
    query_record_id: str
    query_output_repr: str
    decoy_record_id: str
    decoy_output_repr: str
    rationale: str


@dataclass(frozen=True)
class DecoyGenerationReport:
    """Summary of one generator's output.

    ``pair_count`` reflects pairs that satisfied the generator's
    contract (e.g. outputs differ for the same-problem decoy). Rejects
    are tallied by reason so the caller can audit.
    """

    schema_version: str
    category: str
    pair_count: int
    eligible_query_count: int
    skipped_reasons: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "category": self.category,
            "pair_count": self.pair_count,
            "eligible_query_count": self.eligible_query_count,
            "skipped_reasons": dict(sorted(self.skipped_reasons.items())),
        }


def generate_same_problem_different_submission_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    max_pairs_per_query: int = 1,
    same_input_only: bool = True,
) -> tuple[list[DecoyPair], DecoyGenerationReport]:
    """Build pairs where the decoy is a different submission's output for the
    same problem.

    By default, the decoy must also share the same input. Pass
    ``same_input_only=False`` for the strengthened semantic pack, where
    same-problem/different-submission pairs may use different inputs as long as
    outputs differ and the pair records its exact input relation.
    """

    records_list = list(records)
    by_problem_input: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    by_problem: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records_list:
        key = (
            str(rec.get("source_problem_id", "")),
            str(rec.get("input_id", "")),
        )
        if not key[0] or not key[1]:
            continue
        by_problem_input.setdefault(key, []).append(rec)
        by_problem.setdefault(key[0], []).append(rec)

    rng = random.Random(seed)
    pairs: list[DecoyPair] = []
    skipped: dict[str, int] = {}
    eligible = 0
    for query in records_list:
        problem_id = str(query.get("source_problem_id", ""))
        input_id = str(query.get("input_id", ""))
        if not problem_id or not input_id:
            _bump(skipped, "missing_ids")
            continue
        pool = (
            by_problem_input.get((problem_id, input_id), ())
            if same_input_only
            else by_problem.get(problem_id, ())
        )
        candidates = [
            r for r in pool if r.get("source_submission_id") != query.get("source_submission_id")
        ]
        if not candidates:
            _bump(skipped, "no_other_submission")
            continue
        eligible += 1
        # Filter to candidates whose output differs from the query.
        diff_candidates = [
            r
            for r in candidates
            if _output_fingerprint(r) != _output_fingerprint(query)
        ]
        if not diff_candidates:
            _bump(skipped, "outputs_identical")
            continue
        # Deterministic pick: shuffle by seed-derived key.
        diff_candidates.sort(
            key=lambda r: _hash_key(seed, str(r.get("record_id", "")))
        )
        rng.shuffle(diff_candidates)  # re-shuffle for second-order diversity
        for decoy in diff_candidates[: max_pairs_per_query]:
            input_relation = (
                "same_input_id"
                if decoy.get("input_id") == query.get("input_id")
                else "different_input_id"
            )
            pairs.append(
                DecoyPair(
                    category="same_problem_different_submission",
                    query_record_id=str(query.get("record_id", "")),
                    query_output_repr=_output_fingerprint(query),
                    decoy_record_id=str(decoy.get("record_id", "")),
                    decoy_output_repr=_output_fingerprint(decoy),
                    rationale=(
                        "same_problem_id,"
                        f"{input_relation},"
                        "different_submission_id,differing_output"
                    ),
                )
            )
    report = DecoyGenerationReport(
        schema_version=EXECUTION_SURPRISE_DECOY_SCHEMA_VERSION,
        category="same_problem_different_submission",
        pair_count=len(pairs),
        eligible_query_count=eligible,
        skipped_reasons=skipped,
    )
    return pairs, report


def generate_same_code_different_input_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    max_pairs_per_query: int = 1,
) -> tuple[list[DecoyPair], DecoyGenerationReport]:
    """Build pairs where the decoy is the same submission's output on a
    different input.

    Pairs are filtered to drop cases where the same code produced the
    same output on two different inputs (deterministic but
    input-insensitive functions), since those provide no signal.
    """

    records_list = list(records)
    by_submission: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records_list:
        key = str(rec.get("source_submission_id", ""))
        if not key:
            continue
        by_submission.setdefault(key, []).append(rec)

    rng = random.Random(seed + 1)
    pairs: list[DecoyPair] = []
    skipped: dict[str, int] = {}
    eligible = 0
    for query in records_list:
        sub_id = str(query.get("source_submission_id", ""))
        input_id = str(query.get("input_id", ""))
        if not sub_id or not input_id:
            _bump(skipped, "missing_ids")
            continue
        candidates = [
            r
            for r in by_submission.get(sub_id, ())
            if r.get("input_id") != query.get("input_id")
        ]
        if not candidates:
            _bump(skipped, "no_other_input")
            continue
        eligible += 1
        diff_candidates = [
            r
            for r in candidates
            if _output_fingerprint(r) != _output_fingerprint(query)
        ]
        if not diff_candidates:
            _bump(skipped, "outputs_identical")
            continue
        diff_candidates.sort(
            key=lambda r: _hash_key(seed, str(r.get("record_id", "")))
        )
        rng.shuffle(diff_candidates)
        for decoy in diff_candidates[: max_pairs_per_query]:
            pairs.append(
                DecoyPair(
                    category="same_code_different_input",
                    query_record_id=str(query.get("record_id", "")),
                    query_output_repr=_output_fingerprint(query),
                    decoy_record_id=str(decoy.get("record_id", "")),
                    decoy_output_repr=_output_fingerprint(decoy),
                    rationale="same_submission_id,different_input_id,differing_output",
                )
            )
    report = DecoyGenerationReport(
        schema_version=EXECUTION_SURPRISE_DECOY_SCHEMA_VERSION,
        category="same_code_different_input",
        pair_count=len(pairs),
        eligible_query_count=eligible,
        skipped_reasons=skipped,
    )
    return pairs, report


def _hash_key(seed: int, value: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{value}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _output_fingerprint(record: Mapping[str, Any]) -> str:
    raw = record.get("output_repr")
    if isinstance(raw, str) and raw:
        return raw
    checksum = record.get("output_repr_checksum")
    if isinstance(checksum, str) and checksum:
        return f"sha256:{checksum}"
    tokens = record.get("output_tokens")
    if isinstance(tokens, list):
        digest = hashlib.sha256(
            _json_like_tokens(tokens).encode("utf-8")
        ).hexdigest()
        return f"tokens_sha256:{digest}"
    return ""


def _json_like_tokens(tokens: list[Any]) -> str:
    return ",".join(str(int(token)) for token in tokens)
