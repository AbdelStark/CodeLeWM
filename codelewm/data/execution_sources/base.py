"""Base interfaces for execution-substrate source adapters."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from .record import EXECUTION_SOURCE_DATASETS, SourceSubmission


EXECUTION_SOURCE_RECORD_SCHEMA_VERSION = "codelewm.execution_source_record.v1"
ExecutionSourceKind = Literal["codenet", "mbpp", "mbpp_plus", "apps", "humaneval"]


class ExecutionSourceError(RuntimeError):
    """Raised when a source adapter cannot parse the upstream artifact."""


@runtime_checkable
class ExecutionSourceAdapter(Protocol):
    """Protocol that every execution-source adapter satisfies.

    Adapters are stateless and reentrant. The :attr:`held_out_for_eval`
    flag prevents the pack builder from putting MBPP-Plus and HumanEval
    records in train/val splits. Adapters declare their canonical
    license and attribution URL up front; per-record overrides remain
    optional via the upstream metadata.
    """

    dataset: ExecutionSourceKind
    license: str
    license_attribution_url: str
    held_out_for_eval: bool

    def iter_submissions(self, *, source_path: Path) -> Iterator[SourceSubmission]:
        """Yield normalized submissions from ``source_path``.

        ``source_path`` may be a single file or a directory; the adapter
        documents what it accepts. Records that do not satisfy the
        adapter's expectations raise :class:`ExecutionSourceError`.
        """


_ADAPTERS: dict[str, ExecutionSourceAdapter] = {}


def register_execution_source_adapter(
    name: str, adapter: ExecutionSourceAdapter
) -> None:
    """Register an adapter under ``name`` so the CLI can look it up."""

    if name not in EXECUTION_SOURCE_DATASETS:
        raise ValueError(f"unsupported source name: {name!r}")
    _ADAPTERS[name] = adapter


def get_execution_source_adapter(name: str) -> ExecutionSourceAdapter:
    """Return the registered adapter for ``name``."""

    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise ExecutionSourceError(
            f"no execution source adapter is registered for {name!r}; "
            f"known adapters are {sorted(_ADAPTERS)}"
        ) from exc


def load_execution_source(
    *,
    source: ExecutionSourceKind,
    source_path: Path,
    output_path: Path,
    limit: int | None = None,
    deduplicate: bool = False,
) -> dict[str, object]:
    """Write a JSONL ingestion artifact from ``source_path``.

    Returns a summary dict with the count, output path, dataset, license,
    and a list of unique ``source_problem_id``s. The output file is JSONL
    where every line is a :class:`SourceSubmission` ``as_dict``.

    When ``deduplicate`` is True (RFC-0015 WS-B3), submissions whose content
    ``raw_hash`` was already written are skipped, so byte-identical
    (code, input, output) submissions are not ingested more than once. The
    default is False so existing build outputs are unchanged; the v0.7 pack
    build opts in.
    """

    adapter = get_execution_source_adapter(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_problems: set[str] = set()
    seen_submissions: set[str] = set()
    seen_hashes: set[str] = set()
    count = 0
    duplicate_count = 0
    held_out = adapter.held_out_for_eval
    with output_path.open("w", encoding="utf-8") as fh:
        for submission in adapter.iter_submissions(source_path=source_path):
            if limit is not None and count >= limit:
                break
            if deduplicate and submission.raw_hash in seen_hashes:
                duplicate_count += 1
                continue
            payload = submission.as_dict()
            payload["held_out_for_eval"] = held_out
            payload["schema_version"] = EXECUTION_SOURCE_RECORD_SCHEMA_VERSION
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")
            seen_problems.add(submission.source_problem_id)
            seen_submissions.add(submission.source_submission_id)
            seen_hashes.add(submission.raw_hash)
            count += 1
    return {
        "source": source,
        "license": adapter.license,
        "license_attribution_url": adapter.license_attribution_url,
        "held_out_for_eval": held_out,
        "submission_count": count,
        "unique_problem_count": len(seen_problems),
        "unique_submission_count": len(seen_submissions),
        "deduplicate": deduplicate,
        "duplicate_skipped_count": duplicate_count,
        "output_path": str(output_path),
        "schema_version": EXECUTION_SOURCE_RECORD_SCHEMA_VERSION,
    }
