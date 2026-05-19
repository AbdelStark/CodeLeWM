"""Local score API for candidate after-states."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from codelewm.model.checkpoint import sha256_file
from codelewm.model.transition import transition_energy
from codelewm.security import (
    CheckpointTrustError,
    parse_python_source_text,
    require_trusted_checkpoint,
)


SCORE_RESULT_SCHEMA_VERSION = "codelewm.score.v1"
RERANK_RESULT_SCHEMA_VERSION = "codelewm.rerank.v1"
ERROR_REPORT_SCHEMA_VERSION = "codelewm.error.v1"
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
HarnessErrorType = Literal[
    "missing_file",
    "invalid_syntax",
    "patch_apply_failed",
    "config_error",
    "source_unavailable",
    "dataset_build_error",
    "empty_dataset",
    "manifest_error",
    "checkpoint_error",
    "scoring_error",
]


class ScoreError(ValueError):
    """Raised when a score request cannot be evaluated."""

    def __init__(
        self,
        message: str,
        *,
        error_type: HarnessErrorType = "scoring_error",
        remediation: str = "inspect the score request and retry",
        artifact: str | None = None,
        caused_by: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.remediation = remediation
        self.artifact = artifact
        self.caused_by = caused_by

    def to_error_report(self, *, record_id: str | None = None) -> "ErrorReport":
        return ErrorReport(
            error_type=self.error_type,
            message=str(self),
            remediation=self.remediation,
            record_id=record_id,
            artifact=self.artifact,
            caused_by=self.caused_by,
        )


@dataclass(frozen=True)
class ErrorReport:
    """Schema-versioned CLI error report."""

    error_type: HarnessErrorType
    message: str
    remediation: str
    record_id: str | None = None
    artifact: str | None = None
    caused_by: str | None = None
    schema_version: str = ERROR_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ERROR_REPORT_SCHEMA_VERSION:
            raise ScoreError("unsupported error report schema")
        if self.error_type not in {
            "missing_file",
            "invalid_syntax",
            "patch_apply_failed",
            "config_error",
            "source_unavailable",
            "dataset_build_error",
            "empty_dataset",
            "manifest_error",
            "checkpoint_error",
            "scoring_error",
        }:
            raise ScoreError(f"unsupported error_type: {self.error_type}")
        if not self.message:
            raise ScoreError("error report message must not be empty")
        if not self.remediation:
            raise ScoreError("error report remediation must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "error_type": self.error_type,
            "message": self.message,
            "remediation": self.remediation,
            "record_id": self.record_id,
            "artifact": self.artifact,
            "caused_by": self.caused_by,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ErrorReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            error_type=payload["error_type"],
            message=str(payload["message"]),
            remediation=str(payload["remediation"]),
            record_id=None if payload.get("record_id") is None else str(payload["record_id"]),
            artifact=None if payload.get("artifact") is None else str(payload["artifact"]),
            caused_by=None if payload.get("caused_by") is None else str(payload["caused_by"]),
        )


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
class RerankResult:
    """Schema-versioned rerank output for candidate after-states or patches."""

    results: tuple[ScoreResult | ErrorReport, ...]
    warnings: tuple[str, ...] = ()
    schema_version: str = RERANK_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RERANK_RESULT_SCHEMA_VERSION:
            raise ScoreError("unsupported rerank result schema")
        for result in self.results:
            if not isinstance(result, (ScoreResult, ErrorReport)):
                raise ScoreError("rerank results must be score results or error reports")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "results": [result.to_dict() for result in self.results],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RerankResult":
        results: list[ScoreResult | ErrorReport] = []
        for item in payload["results"]:
            if item.get("schema_version") == SCORE_RESULT_SCHEMA_VERSION:
                results.append(ScoreResult.from_dict(item))
            elif item.get("schema_version") == ERROR_REPORT_SCHEMA_VERSION:
                results.append(ErrorReport.from_dict(item))
            else:
                raise ScoreError("unsupported rerank result item schema")
        return cls(
            schema_version=str(payload["schema_version"]),
            results=tuple(results),
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
        before_text = _read_python_file(before, "before")
        candidate_text = _read_python_file(candidate, "candidate")
        return self.score_texts(
            before=before_text,
            instruction=instruction,
            candidate=candidate_text,
            candidate_name=str(candidate),
        )

    def rerank_files(self, *, before: Path, instruction: str, candidates: Path) -> RerankResult:
        before_text = _read_python_file(before, "before")
        if not instruction.strip():
            raise ScoreError("instruction must not be empty")

        score_results: list[ScoreResult] = []
        error_reports: list[ErrorReport] = []
        for candidate_path in _candidate_paths(candidates):
            try:
                candidate_text = _candidate_path_to_text(
                    candidate_path,
                    before_text=before_text,
                )
                _validate_candidate_text(candidate_text, candidate_path)
                score_results.append(
                    self.score_texts(
                        before=before_text,
                        instruction=instruction,
                        candidate=candidate_text,
                        candidate_name=str(candidate_path),
                    )
                )
            except ScoreError as exc:
                error_reports.append(exc.to_error_report(record_id=str(candidate_path)))

        ordered_scores = tuple(sorted(score_results, key=lambda result: (result.final_score, result.candidate)))
        return RerankResult(
            results=(*ordered_scores, *error_reports),
            warnings=self.backend.warnings,
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
    allow_unsafe: bool = False,
    checkpoint_manifest: Path | str | None = None,
) -> CodeLeWMScorer:
    """Load a local scorer wrapper after verifying the checkpoint path exists.

    When ``allow_unsafe`` is False (the default), the checkpoint must be
    accompanied by a trusted manifest at ``<checkpoint>.manifest.json`` (or
    the explicit ``checkpoint_manifest`` path). The manifest is validated
    via :func:`codelewm.security.require_trusted_checkpoint`. Pass
    ``allow_unsafe=True`` to load checkpoints without a manifest; this is
    only safe for fixture checkpoints in a trusted local environment.
    """

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise ScoreError(
            f"checkpoint file does not exist: {checkpoint_path}",
            error_type="missing_file",
            remediation="provide an existing checkpoint file",
            artifact=str(checkpoint_path),
        )
    if not allow_unsafe:
        try:
            require_trusted_checkpoint(checkpoint_path, manifest_path=checkpoint_manifest)
        except CheckpointTrustError as exc:
            raise ScoreError(
                f"checkpoint trust check failed: {exc}",
                error_type="checkpoint_error",
                remediation=(
                    "provide a trusted checkpoint manifest at "
                    "<checkpoint>.manifest.json or rerun with allow_unsafe=True"
                ),
                artifact=str(checkpoint_path),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ) from exc
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


def error_report_to_json(report: ErrorReport) -> str:
    """Serialize an error report as stable JSON."""

    return json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))


def rerank_result_to_json(result: RerankResult) -> str:
    """Serialize a rerank result as stable JSON."""

    return json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))


def validate_score_result_payload(payload: dict[str, Any]) -> ScoreResult:
    """Return a validated score result payload."""

    return ScoreResult.from_dict(payload)


def validate_error_report_payload(payload: dict[str, Any]) -> ErrorReport:
    """Return a validated error report payload."""

    return ErrorReport.from_dict(payload)


def validate_rerank_result_payload(payload: dict[str, Any]) -> RerankResult:
    """Return a validated rerank result payload."""

    return RerankResult.from_dict(payload)


def score_result_json_schema() -> dict[str, Any]:
    """Return the JSON schema for `ScoreResult` payloads."""

    return {
        "type": "object",
        "required": [
            "schema_version",
            "candidate",
            "transition_energy",
            "retrieval_prior",
            "risk_penalty",
            "final_score",
            "model_id",
            "checkpoint_sha256",
            "input_digest",
            "warnings",
        ],
        "properties": {
            "schema_version": {"const": SCORE_RESULT_SCHEMA_VERSION},
            "candidate": {"type": "string"},
            "transition_energy": {"type": "number"},
            "retrieval_prior": {"type": ["number", "null"]},
            "risk_penalty": {"type": ["number", "null"]},
            "final_score": {"type": "number"},
            "model_id": {"type": "string"},
            "checkpoint_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "input_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def error_report_json_schema() -> dict[str, Any]:
    """Return the JSON schema for CLI error reports."""

    return {
        "type": "object",
        "required": [
            "schema_version",
            "error_type",
            "message",
            "remediation",
            "record_id",
            "artifact",
            "caused_by",
        ],
        "properties": {
            "schema_version": {"const": ERROR_REPORT_SCHEMA_VERSION},
            "error_type": {
                "enum": [
                    "missing_file",
                    "invalid_syntax",
                    "patch_apply_failed",
                    "config_error",
                    "source_unavailable",
                    "dataset_build_error",
                    "empty_dataset",
                    "manifest_error",
                    "checkpoint_error",
                    "scoring_error",
                ]
            },
            "message": {"type": "string"},
            "remediation": {"type": "string"},
            "record_id": {"type": ["string", "null"]},
            "artifact": {"type": ["string", "null"]},
            "caused_by": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    }


def rerank_result_json_schema() -> dict[str, Any]:
    """Return the reserved v0.1 JSON schema for rerank outputs."""

    return {
        "type": "object",
        "required": ["schema_version", "results", "warnings"],
        "properties": {
            "schema_version": {"const": RERANK_RESULT_SCHEMA_VERSION},
            "results": {
                "type": "array",
                "items": {
                    "oneOf": [
                        score_result_json_schema(),
                        error_report_json_schema(),
                    ]
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def _read_python_file(path: Path, label: str) -> str:
    text = _read_text_file(path, label)
    try:
        parse_python_source_text(text, filename=str(path))
    except SyntaxError as exc:
        raise ScoreError(
            f"{label} file is not valid Python",
            error_type="invalid_syntax",
            remediation="provide a parseable Python file",
            artifact=str(path),
            caused_by=f"{exc.__class__.__name__}: {exc.msg}",
        ) from exc
    return text


def _validate_candidate_text(text: str, path: Path) -> None:
    try:
        parse_python_source_text(text, filename=str(path))
    except SyntaxError as exc:
        raise ScoreError(
            "candidate file is not valid Python",
            error_type="invalid_syntax",
            remediation="provide a parseable Python file or patch",
            artifact=str(path),
            caused_by=f"{exc.__class__.__name__}: {exc.msg}",
        ) from exc


def _read_text_file(path: Path, label: str) -> str:
    if not path.is_file():
        raise ScoreError(
            f"{label} file does not exist: {path}",
            error_type="missing_file",
            remediation=f"provide an existing {label} file",
            artifact=str(path),
        )
    return path.read_text()


def _candidate_paths(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.exists():
        raise ScoreError(
            f"candidate path does not exist: {path}",
            error_type="missing_file",
            remediation="provide an existing candidate file or directory",
            artifact=str(path),
        )
    if not path.is_dir():
        raise ScoreError(
            f"candidate path is neither file nor directory: {path}",
            remediation="provide a candidate file or directory",
            artifact=str(path),
        )
    candidates = tuple(sorted(candidate for candidate in path.iterdir() if candidate.is_file()))
    if not candidates:
        raise ScoreError(
            f"candidate directory is empty: {path}",
            remediation="provide at least one candidate file or patch",
            artifact=str(path),
        )
    return candidates


def _candidate_path_to_text(path: Path, *, before_text: str) -> str:
    text = _read_text_file(path, "candidate")
    if path.suffix.lower() in {".diff", ".patch"}:
        return _apply_unified_diff(before_text, text, artifact=str(path))
    return text


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def _apply_unified_diff(source: str, patch: str, *, artifact: str) -> str:
    source_lines = source.splitlines(keepends=True)
    patch_lines = patch.splitlines(keepends=True)
    output: list[str] = []
    source_index = 0
    patch_index = 0
    saw_hunk = False

    while patch_index < len(patch_lines):
        line = patch_lines[patch_index]
        if not line.startswith("@@"):
            patch_index += 1
            continue

        match = _HUNK_HEADER_RE.match(line.rstrip("\n"))
        if match is None:
            raise _patch_error("patch has an invalid unified-diff hunk header", artifact)
        saw_hunk = True
        old_start = int(match.group("old_start"))
        old_count = int(match.group("old_count") or "1")
        source_target = old_start - 1
        if source_target < source_index or source_target > len(source_lines):
            raise _patch_error("patch hunk is outside the before file", artifact)
        output.extend(source_lines[source_index:source_target])
        source_index = source_target
        patch_index += 1

        consumed_old = 0
        while patch_index < len(patch_lines) and not patch_lines[patch_index].startswith("@@"):
            hunk_line = patch_lines[patch_index]
            if hunk_line.startswith("\\"):
                patch_index += 1
                continue
            if not hunk_line:
                raise _patch_error("patch contains an empty hunk line", artifact)
            marker = hunk_line[0]
            content = hunk_line[1:]
            if marker == " ":
                _consume_source_line(source_lines, source_index, content, artifact)
                output.append(source_lines[source_index])
                source_index += 1
                consumed_old += 1
            elif marker == "-":
                _consume_source_line(source_lines, source_index, content, artifact)
                source_index += 1
                consumed_old += 1
            elif marker == "+":
                output.append(content)
            else:
                raise _patch_error("patch contains a non-unified-diff hunk line", artifact)
            patch_index += 1
        if consumed_old != old_count:
            raise _patch_error("patch hunk old-line count does not match its header", artifact)

    if not saw_hunk:
        raise _patch_error("patch contains no unified-diff hunks", artifact)
    output.extend(source_lines[source_index:])
    return "".join(output)


def _consume_source_line(source_lines: list[str], source_index: int, expected: str, artifact: str) -> None:
    if source_index >= len(source_lines):
        raise _patch_error("patch hunk extends past the before file", artifact)
    if source_lines[source_index] != expected:
        raise _patch_error("patch context does not match the before file", artifact)


def _patch_error(message: str, artifact: str) -> ScoreError:
    return ScoreError(
        message,
        error_type="patch_apply_failed",
        remediation="provide a single-file unified diff that applies cleanly to the before file",
        artifact=artifact,
    )


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
