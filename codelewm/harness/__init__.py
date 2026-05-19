"""Harness entry points for CodeLeWM."""

from __future__ import annotations

from collections.abc import Sequence

from .transition_index import (
    TRANSITION_INDEX_SCHEMA_VERSION,
    TransitionIndex,
    TransitionIndexEntry,
    TransitionIndexError,
    TransitionIndexSearchHit,
    build_transition_index,
    read_transition_index,
    transition_index_header_json_schema,
    write_transition_index,
)
from .index_runner import (
    INDEX_BUILD_RESULT_SCHEMA_VERSION,
    IndexBuildResult,
    build_transition_index_artifact,
)
from .quality import (
    SCORER_QUALITY_CONFIG_SCHEMA_VERSION,
    SCORER_QUALITY_REPORT_SCHEMA_VERSION,
    SCORER_QUALITY_RUN_SCHEMA_VERSION,
    ScorerQualityConfig,
    ScorerQualityError,
    ScorerQualityExampleConfig,
    ScorerQualityRunResult,
    read_scorer_quality_config,
    read_scorer_quality_report,
    run_scorer_quality_evaluation,
)
from .scorer import (
    ERROR_REPORT_SCHEMA_VERSION,
    RERANK_RESULT_SCHEMA_VERSION,
    SCORE_RESULT_SCHEMA_VERSION,
    CodeLeWMScorer,
    ErrorReport,
    HashingTransitionScoringBackend,
    RerankResult,
    ScoreError,
    ScoreResult,
    TransitionScoringBackend,
    error_report_json_schema,
    error_report_to_json,
    load_scorer,
    rerank_result_json_schema,
    rerank_result_to_json,
    score_input_digest,
    score_result_json_schema,
    score_result_to_json,
    validate_error_report_payload,
    validate_rerank_result_payload,
    validate_score_result_payload,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entry point without importing it during package initialization."""

    from .cli import main as _main

    return _main(argv)


__all__ = [
    "ERROR_REPORT_SCHEMA_VERSION",
    "INDEX_BUILD_RESULT_SCHEMA_VERSION",
    "RERANK_RESULT_SCHEMA_VERSION",
    "SCORER_QUALITY_CONFIG_SCHEMA_VERSION",
    "SCORER_QUALITY_REPORT_SCHEMA_VERSION",
    "SCORER_QUALITY_RUN_SCHEMA_VERSION",
    "SCORE_RESULT_SCHEMA_VERSION",
    "TRANSITION_INDEX_SCHEMA_VERSION",
    "CodeLeWMScorer",
    "ErrorReport",
    "HashingTransitionScoringBackend",
    "IndexBuildResult",
    "RerankResult",
    "ScoreError",
    "ScoreResult",
    "ScorerQualityConfig",
    "ScorerQualityError",
    "ScorerQualityExampleConfig",
    "ScorerQualityRunResult",
    "TransitionIndex",
    "TransitionIndexEntry",
    "TransitionIndexError",
    "TransitionIndexSearchHit",
    "TransitionScoringBackend",
    "build_transition_index",
    "build_transition_index_artifact",
    "error_report_json_schema",
    "error_report_to_json",
    "load_scorer",
    "main",
    "read_transition_index",
    "read_scorer_quality_config",
    "read_scorer_quality_report",
    "rerank_result_json_schema",
    "rerank_result_to_json",
    "score_input_digest",
    "score_result_json_schema",
    "score_result_to_json",
    "run_scorer_quality_evaluation",
    "transition_index_header_json_schema",
    "validate_error_report_payload",
    "validate_rerank_result_payload",
    "validate_score_result_payload",
    "write_transition_index",
]
