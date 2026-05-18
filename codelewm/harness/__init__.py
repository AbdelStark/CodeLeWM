"""Harness entry points for CodeLeWM."""

from __future__ import annotations

from .cli import main
from .scorer import (
    SCORE_RESULT_SCHEMA_VERSION,
    CodeLeWMScorer,
    HashingTransitionScoringBackend,
    ScoreError,
    ScoreResult,
    TransitionScoringBackend,
    load_scorer,
    score_input_digest,
    score_result_to_json,
)

__all__ = [
    "SCORE_RESULT_SCHEMA_VERSION",
    "CodeLeWMScorer",
    "HashingTransitionScoringBackend",
    "ScoreError",
    "ScoreResult",
    "TransitionScoringBackend",
    "load_scorer",
    "main",
    "score_input_digest",
    "score_result_to_json",
]
