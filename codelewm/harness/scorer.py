"""Local score API for candidate after-states."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from codelewm.model.checkpoint import sha256_file
from codelewm.model.transition import transition_energy


SCORE_RESULT_SCHEMA_VERSION = "codelewm.score.v1"
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")


class ScoreError(ValueError):
    """Raised when a score request cannot be evaluated."""


class TransitionScoringBackend(Protocol):
    """Backend protocol for model-backed or fixture scoring implementations."""

    model_id: str
    warnings: tuple[str, ...]

    def transition_energy(self, before: str, instruction: str, candidate: str) -> float:
        """Return transition energy for a before/instruction/candidate triple."""


@dataclass(frozen=True)
class ScoreResult:
    """Schema-versioned score output for one candidate after-state."""

    candidate: str
    transition_energy: float
    final_score: float
    model_id: str
    checkpoint_sha256: str
    input_digest: str
    retrieval_prior: float | None = None
    risk_penalty: float | None = None
    warnings: tuple[str, ...] = ()
    schema_version: str = SCORE_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCORE_RESULT_SCHEMA_VERSION:
            raise ScoreError("unsupported score result schema")
        if not math.isfinite(self.transition_energy):
            raise ScoreError("transition_energy must be finite")
        if not math.isfinite(self.final_score):
            raise ScoreError("final_score must be finite")
        if not self.model_id:
            raise ScoreError("model_id must not be empty")
        if len(self.checkpoint_sha256) != 64:
            raise ScoreError("checkpoint_sha256 must be a SHA-256 digest")
        if len(self.input_digest) != 64:
            raise ScoreError("input_digest must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate": self.candidate,
            "transition_energy": self.transition_energy,
            "retrieval_prior": self.retrieval_prior,
            "risk_penalty": self.risk_penalty,
            "final_score": self.final_score,
            "model_id": self.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "input_digest": self.input_digest,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoreResult":
        return cls(
            schema_version=str(payload["schema_version"]),
            candidate=str(payload["candidate"]),
            transition_energy=float(payload["transition_energy"]),
            retrieval_prior=None
            if payload.get("retrieval_prior") is None
            else float(payload["retrieval_prior"]),
            risk_penalty=None if payload.get("risk_penalty") is None else float(payload["risk_penalty"]),
            final_score=float(payload["final_score"]),
            model_id=str(payload["model_id"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            input_digest=str(payload["input_digest"]),
            warnings=tuple(str(warning) for warning in payload.get("warnings", ())),
        )


@dataclass(frozen=True)
class HashingTransitionScoringBackend:
    """Deterministic lightweight scoring backend used until ML runtime wiring lands."""

    latent_dim: int = 64
    model_id: str = "codelewm.hashing_transition_scorer.v1"
    warnings: tuple[str, ...] = (
        "deterministic lightweight scorer backend; model runtime is not loaded",
    )

    def transition_energy(self, before: str, instruction: str, candidate: str) -> float:
        before_vec = _hashed_vector(before, dim=self.latent_dim)
        instruction_vec = _hashed_vector(instruction, dim=self.latent_dim)
        candidate_vec = _hashed_vector(candidate, dim=self.latent_dim)
        predicted = [
            before_value + instruction_value
            for before_value, instruction_value in zip(before_vec, instruction_vec)
        ]
        return float(transition_energy(predicted, candidate_vec, reduction="sum"))


@dataclass(frozen=True)
class CodeLeWMScorer:
    """Python API wrapper for scoring candidate after-state files."""

    checkpoint: Path
    checkpoint_sha256: str
    backend: TransitionScoringBackend
    device: str = "auto"

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def score_files(self, *, before: Path, instruction: str, candidate: Path) -> ScoreResult:
        before_text = _read_text_file(before, "before")
        candidate_text = _read_text_file(candidate, "candidate")
        return self.score_texts(
            before=before_text,
            instruction=instruction,
            candidate=candidate_text,
            candidate_name=str(candidate),
        )

    def score_texts(
        self,
        *,
        before: str,
        instruction: str,
        candidate: str,
        candidate_name: str = "<candidate>",
    ) -> ScoreResult:
        if not instruction.strip():
            raise ScoreError("instruction must not be empty")
        energy = self.backend.transition_energy(before, instruction, candidate)
        return ScoreResult(
            candidate=candidate_name,
            transition_energy=energy,
            retrieval_prior=None,
            risk_penalty=None,
            final_score=energy,
            model_id=self.model_id,
            checkpoint_sha256=self.checkpoint_sha256,
            input_digest=score_input_digest(before, instruction, candidate),
            warnings=self.backend.warnings,
        )


def load_scorer(
    checkpoint: Path | str,
    *,
    device: str = "auto",
    backend: TransitionScoringBackend | None = None,
) -> CodeLeWMScorer:
    """Load a local scorer wrapper after verifying the checkpoint path exists."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise ScoreError(f"checkpoint file does not exist: {checkpoint_path}")
    return CodeLeWMScorer(
        checkpoint=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        backend=HashingTransitionScoringBackend() if backend is None else backend,
        device=device,
    )


def score_input_digest(before: str, instruction: str, candidate: str) -> str:
    """Return a digest over score inputs without storing raw source text."""

    payload = {
        "before_sha256": _sha256_text(before),
        "instruction_sha256": _sha256_text(instruction),
        "candidate_sha256": _sha256_text(candidate),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score_result_to_json(result: ScoreResult) -> str:
    """Serialize a score result as stable JSON."""

    return json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))


def _read_text_file(path: Path, label: str) -> str:
    if not path.is_file():
        raise ScoreError(f"{label} file does not exist: {path}")
    return path.read_text()


def _hashed_vector(text: str, *, dim: int) -> list[float]:
    if dim <= 0:
        raise ScoreError("latent dimension must be positive")
    vector = [0.0] * dim
    tokens = _TOKEN_RE.findall(text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
