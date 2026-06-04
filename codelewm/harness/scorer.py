"""Local score API for candidate after-states."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from codelewm.data.codestate import (
    CodeStateExtractionError,
    changed_line_numbers,
    extract_codestate,
)
from codelewm.data.execution_pack.record import tokenize_text as tokenize_execution_text
from codelewm.data.masks import build_masked_codestate, stable_token_id
from codelewm.model.checkpoint import sha256_file
from codelewm.model.transition import (
    ABSTRACT_ACTION_SEQUENCE_LENGTH,
    STATE_SEQUENCE_LENGTH,
    TEXT_ACTION_SEQUENCE_LENGTH,
    ActionBatch,
    ActionView,
    CodeStateBatch,
    transition_energy,
)
from codelewm.security import (
    CheckpointTrustError,
    parse_python_source_text,
    require_trusted_checkpoint,
)
from codelewm.training import DEFAULT_TRAINING_VOCAB_SIZE, TORCH_CHECKPOINT_SCHEMA_VERSION

from .transition_index import (
    TransitionIndex,
    TransitionIndexError,
    read_transition_index,
)


SCORE_RESULT_SCHEMA_VERSION = "codelewm.score.v1"
RERANK_RESULT_SCHEMA_VERSION = "codelewm.rerank.v1"
ERROR_REPORT_SCHEMA_VERSION = "codelewm.error.v1"
EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION = "codelewm.execution_train_checkpoint.v1"
SUPPORTED_EXECUTION_PACK_RECORD_SCHEMA_VERSIONS = (
    "codelewm.execution_pack_record.v1",
    "codelewm.execution_pack_record.v2",
)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
_DATA_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
HarnessErrorType = Literal[
    "missing_file",
    "invalid_syntax",
    "patch_apply_failed",
    "config_error",
    "source_unavailable",
    "optional_dependency_missing",
    "dataset_build_error",
    "empty_dataset",
    "manifest_error",
    "checkpoint_error",
    "evaluation_gate_error",
    "scoring_error",
    "input_missing",
    "invalid_arguments",
    "sandbox_runner_error",
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
            "optional_dependency_missing",
            "dataset_build_error",
            "empty_dataset",
            "manifest_error",
            "checkpoint_error",
            "evaluation_gate_error",
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
        if self.retrieval_prior is not None and not math.isfinite(self.retrieval_prior):
            raise ScoreError("retrieval_prior must be finite")
        if self.risk_penalty is not None and not math.isfinite(self.risk_penalty):
            raise ScoreError("risk_penalty must be finite")
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
class TorchCheckpointTransitionScoringBackend:
    """Learned torch checkpoint scorer for before/instruction/candidate triples."""

    model: Any
    runtime: Any
    device: Any
    action_view: ActionView
    action_sequence_length: int
    vocab_size: int = DEFAULT_TRAINING_VOCAB_SIZE
    checkpoint_step: int | None = None
    model_id: str = "codelewm.torch_transition_scorer.v1"

    @property
    def warnings(self) -> tuple[str, ...]:
        step = "unknown" if self.checkpoint_step is None else str(self.checkpoint_step)
        return (
            "learned torch transition model runtime loaded from checkpoint",
            f"checkpoint_step={step}",
            f"action_view={self.action_view}",
        )

    @classmethod
    def load(cls, checkpoint: Path | str, *, device: str = "auto") -> "TorchCheckpointTransitionScoringBackend":
        checkpoint_path = Path(checkpoint)
        runtime = _require_torch_runtime_for_scoring()
        selected_device = _resolve_torch_device(device, runtime)
        try:
            payload = runtime.load(checkpoint_path, map_location=selected_device, weights_only=True)
        except TypeError:  # pragma: no cover - older torch compatibility.
            payload = runtime.load(checkpoint_path, map_location=selected_device)
        except Exception as exc:  # pragma: no cover - torch error classes vary by version.
            raise ScoreError(
                f"torch checkpoint could not be loaded: {exc}",
                error_type="checkpoint_error",
                remediation="provide a valid package-native CodeLeWM torch checkpoint",
                artifact=str(checkpoint_path),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        if not isinstance(payload, Mapping):
            raise ScoreError(
                "checkpoint payload must be a mapping",
                error_type="checkpoint_error",
                remediation="provide a package-native CodeLeWM torch checkpoint",
                artifact=str(checkpoint_path),
            )
        if payload.get("schema_version") != TORCH_CHECKPOINT_SCHEMA_VERSION:
            raise ScoreError(
                f"checkpoint schema_version is unsupported: {payload.get('schema_version')!r}",
                error_type="checkpoint_error",
                remediation="provide a codelewm.torch_checkpoint.v1 checkpoint",
                artifact=str(checkpoint_path),
            )
        compatibility = payload.get("compatibility_config")
        if not isinstance(compatibility, Mapping):
            raise ScoreError(
                "checkpoint compatibility_config must be a mapping",
                error_type="checkpoint_error",
                remediation="provide a checkpoint written by codelewm train --executor torch",
                artifact=str(checkpoint_path),
            )
        model = _build_torch_model_from_compatibility(
            compatibility,
            artifact=str(checkpoint_path),
            state_dict=payload.get("model_state_dict"),
        )
        try:
            model.load_state_dict(payload["model_state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ScoreError(
                f"checkpoint model state could not be loaded: {exc}",
                error_type="checkpoint_error",
                remediation="inspect the checkpoint compatibility config and model state",
                artifact=str(checkpoint_path),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        model.to(selected_device)
        model.eval()
        return cls(
            model=model,
            runtime=runtime,
            device=selected_device,
            action_view=model.config.action_view,
            action_sequence_length=model.config.action_sequence_length,
            vocab_size=model.config.vocab_size,
            checkpoint_step=_optional_int(payload.get("step")),
        )

    def transition_energy(self, before: str, instruction: str, candidate: str) -> float:
        before_changed, after_changed = changed_line_numbers(before, candidate)
        before_batch = self._state_batch_from_source(
            before,
            path="before.py",
            changed_lines=before_changed,
            field_name="before",
        )
        after_batch = self._state_batch_from_source(
            candidate,
            path="candidate.py",
            changed_lines=after_changed,
            field_name="candidate",
        )
        action_batch = self._action_batch_from_text(instruction)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The PyTorch API of nested tensors is in prototype stage.*",
                    category=UserWarning,
                )
                with self.runtime.no_grad():
                    z_before = self.model.encode_state(before_batch)
                    action_emb = self.model.encode_action(action_batch)
                    z_after = self.model.encode_state(after_batch)
                    z_pred_after = self.model.predict_after(z_before, action_emb)
                    energy = self.model.transition_energy(z_pred_after, z_after, reduction="sum")
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            raise ScoreError(
                f"torch transition scoring failed: {exc}",
                error_type="scoring_error",
                remediation="retry with --device cpu or inspect the checkpoint/runtime compatibility",
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        value = float(energy.detach().cpu().item())
        if not math.isfinite(value):
            raise ScoreError("torch transition energy must be finite")
        return value

    def _state_batch_from_source(
        self,
        source: str,
        *,
        path: str,
        changed_lines: set[int],
        field_name: str,
    ) -> CodeStateBatch:
        try:
            state = extract_codestate(
                source,
                path=path,
                changed_lines=changed_lines,
                field_name=field_name,
            )
            sequence = build_masked_codestate(state).token_sequence
        except CodeStateExtractionError as exc:
            raise ScoreError(
                f"{field_name} source is not parse-valid Python: {exc}",
                error_type="invalid_syntax",
                remediation="provide parse-valid Python source text",
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        except ValueError as exc:
            raise ScoreError(
                f"{field_name} source could not be tokenized for the torch scorer: {exc}",
                error_type="scoring_error",
                remediation="inspect the source length, syntax, and CodeState extraction policy",
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        input_ids = _pad_int_array(
            _bucket_token_ids(sequence.input_ids, vocab_size=self.vocab_size),
            STATE_SEQUENCE_LENGTH,
            name=f"{field_name}.input_ids",
        )
        attention_mask = _pad_bool_array(
            sequence.attention_mask
            if sequence.attention_mask is not None
            else tuple(True for _ in sequence.input_ids),
            STATE_SEQUENCE_LENGTH,
            name=f"{field_name}.attention_mask",
        )
        segment_ids = _pad_int_array(
            sequence.segment_ids
            if sequence.segment_ids is not None
            else tuple(0 for _ in sequence.input_ids),
            STATE_SEQUENCE_LENGTH,
            name=f"{field_name}.segment_ids",
        )
        changed_hunk_mask = _pad_bool_array(
            sequence.changed_hunk_mask
            if sequence.changed_hunk_mask is not None
            else tuple(False for _ in sequence.input_ids),
            STATE_SEQUENCE_LENGTH,
            name=f"{field_name}.changed_hunk_mask",
        )
        return CodeStateBatch(
            input_ids=self.runtime.as_tensor(input_ids, device=self.device).long(),
            attention_mask=self.runtime.as_tensor(attention_mask, device=self.device).bool(),
            segment_ids=self.runtime.as_tensor(segment_ids, device=self.device).long(),
            changed_hunk_mask=self.runtime.as_tensor(changed_hunk_mask, device=self.device).bool(),
        )

    def _action_batch_from_text(self, instruction: str) -> ActionBatch:
        tokens = _torch_tokenize_text(instruction)
        if not tokens:
            raise ScoreError("instruction must not be empty")
        input_ids = _pad_int_array(
            _bucket_token_ids(tuple(stable_token_id(token) for token in tokens), vocab_size=self.vocab_size),
            self.action_sequence_length,
            name="action.input_ids",
        )
        attention_mask = _pad_bool_array(
            tuple(True for _ in tokens),
            self.action_sequence_length,
            name="action.attention_mask",
        )
        return ActionBatch(
            input_ids=self.runtime.as_tensor(input_ids, device=self.device).long(),
            attention_mask=self.runtime.as_tensor(attention_mask, device=self.device).bool(),
            action_view=self.action_view,
        )


@dataclass(frozen=True)
class ExecutionTorchTransitionScoringBackend:
    """Learned v0.6 execution-substrate scorer for candidate code and one input.

    The public score/rerank API is still shaped as
    ``before/instruction/candidate`` for compatibility. For execution
    checkpoints the ``candidate`` text is the candidate program and
    ``instruction`` is the serialized input repr. The score is a diagnostic
    latent score, not a correctness label.
    """

    model: Any
    runtime: Any
    device: Any
    action_view: ActionView
    state_sequence_length: int
    action_sequence_length: int
    vocab_size: int = DEFAULT_TRAINING_VOCAB_SIZE
    checkpoint_step: int | None = None
    model_id: str = "codelewm.execution_torch_transition_scorer.v1"

    @property
    def warnings(self) -> tuple[str, ...]:
        step = "unknown" if self.checkpoint_step is None else str(self.checkpoint_step)
        energy_formula = (
            "neg_pass_logit"
            if getattr(self.model, "pass_head", None) is not None
            else "predicted_output_latent_norm_plus_abs_code_similarity"
        )
        return (
            "learned execution-substrate torch runtime loaded from checkpoint",
            f"checkpoint_step={step}",
            f"action_view={self.action_view}",
            f"transition_energy={energy_formula}",
            "execution score is diagnostic; correctness requires sandbox labels or a downstream benchmark",
        )

    @classmethod
    def load(
        cls,
        checkpoint: Path | str,
        *,
        device: str = "auto",
    ) -> "ExecutionTorchTransitionScoringBackend":
        checkpoint_path = Path(checkpoint)
        runtime = _require_torch_runtime_for_scoring()
        selected_device = _resolve_torch_device(device, runtime)
        try:
            payload = runtime.load(checkpoint_path, map_location=selected_device, weights_only=True)
        except TypeError:  # pragma: no cover - older torch compatibility.
            payload = runtime.load(checkpoint_path, map_location=selected_device)
        except Exception as exc:  # pragma: no cover - torch error classes vary by version.
            raise ScoreError(
                f"execution torch checkpoint could not be loaded: {exc}",
                error_type="checkpoint_error",
                remediation="provide a valid package-native CodeLeWM execution torch checkpoint",
                artifact=str(checkpoint_path),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        if not isinstance(payload, Mapping):
            raise ScoreError(
                "execution checkpoint payload must be a mapping",
                error_type="checkpoint_error",
                remediation="provide a package-native CodeLeWM execution torch checkpoint",
                artifact=str(checkpoint_path),
            )
        if payload.get("schema_version") != EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION:
            raise ScoreError(
                f"execution checkpoint schema_version is unsupported: {payload.get('schema_version')!r}",
                error_type="checkpoint_error",
                remediation="provide a codelewm.execution_train_checkpoint.v1 checkpoint",
                artifact=str(checkpoint_path),
            )
        compatibility = payload.get("compatibility_config")
        if not isinstance(compatibility, Mapping):
            raise ScoreError(
                "execution checkpoint compatibility_config must be a mapping",
                error_type="checkpoint_error",
                remediation="provide a checkpoint written by codelewm train with an execution config",
                artifact=str(checkpoint_path),
            )
        model = _build_torch_model_from_compatibility(
            compatibility,
            artifact=str(checkpoint_path),
            state_dict=payload.get("model_state_dict"),
        )
        try:
            model.load_state_dict(payload["model_state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ScoreError(
                f"execution checkpoint model state could not be loaded: {exc}",
                error_type="checkpoint_error",
                remediation="inspect the execution checkpoint compatibility config and model state",
                artifact=str(checkpoint_path),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        model.to(selected_device)
        model.eval()
        return cls(
            model=model,
            runtime=runtime,
            device=selected_device,
            action_view=model.config.action_view,
            state_sequence_length=int(model.config.state_sequence_length),
            action_sequence_length=int(model.config.action_sequence_length),
            vocab_size=int(model.config.vocab_size or DEFAULT_TRAINING_VOCAB_SIZE),
            checkpoint_step=_optional_int(payload.get("step")),
        )

    def transition_energy(self, before: str, instruction: str, candidate: str) -> float:
        del before
        if not instruction.strip():
            raise ScoreError(
                "execution-substrate scoring requires instruction to contain the input repr",
                error_type="config_error",
                remediation="pass the execution input repr as --instruction",
            )
        code_state = self._state_batch_from_text(candidate, field_name="candidate")
        action_batch = self._action_batch_from_text(instruction)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The PyTorch API of nested tensors is in prototype stage.*",
                    category=UserWarning,
                )
                with self.runtime.no_grad():
                    z_code = self.model.encode_state(code_state)
                    action_emb = self.model.encode_action(action_batch)
                    z_pred_after = self.model.predict_after(z_code, action_emb)
                    if getattr(self.model, "pass_head", None) is not None:
                        energy = -self.model.pass_logit(
                            z_code, action_emb, z_pred_after
                        ).squeeze(-1)
                    else:
                        pred_norm = z_pred_after.norm(p=2, dim=-1)
                        similarity = self.runtime.nn.functional.cosine_similarity(
                            z_pred_after,
                            z_code,
                            dim=-1,
                            eps=1e-12,
                        ).abs()
                        energy = pred_norm + similarity
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            raise ScoreError(
                f"execution torch transition scoring failed: {exc}",
                error_type="scoring_error",
                remediation="retry with --device cpu or inspect the checkpoint/runtime compatibility",
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        value = float(energy.detach().cpu().item())
        if not math.isfinite(value):
            raise ScoreError("execution transition score must be finite")
        return value

    def _state_batch_from_text(self, text: str, *, field_name: str) -> CodeStateBatch:
        token_ids = _bucket_token_ids(tokenize_execution_text(text), vocab_size=self.vocab_size)
        input_ids = _pad_int_array(
            token_ids,
            self.state_sequence_length,
            name=f"{field_name}.input_ids",
        )
        attention_mask = _pad_bool_array(
            tuple(True for _ in token_ids),
            self.state_sequence_length,
            name=f"{field_name}.attention_mask",
        )
        segment_ids = _pad_int_array(
            (),
            self.state_sequence_length,
            name=f"{field_name}.segment_ids",
        )
        changed_hunk_mask = _pad_bool_array(
            (),
            self.state_sequence_length,
            name=f"{field_name}.changed_hunk_mask",
        )
        return CodeStateBatch(
            input_ids=self.runtime.as_tensor(input_ids, device=self.device).long(),
            attention_mask=self.runtime.as_tensor(attention_mask, device=self.device).bool(),
            segment_ids=self.runtime.as_tensor(segment_ids, device=self.device).long(),
            changed_hunk_mask=self.runtime.as_tensor(changed_hunk_mask, device=self.device).bool(),
        )

    def _action_batch_from_text(self, input_repr: str) -> ActionBatch:
        token_ids = _bucket_token_ids(tokenize_execution_text(input_repr), vocab_size=self.vocab_size)
        if not token_ids:
            raise ScoreError(
                "execution input repr must tokenize to at least one token",
                error_type="config_error",
                remediation="pass a non-empty input repr as --instruction",
            )
        input_ids = _pad_int_array(
            token_ids,
            self.action_sequence_length,
            name="execution_input.input_ids",
        )
        attention_mask = _pad_bool_array(
            tuple(True for _ in token_ids),
            self.action_sequence_length,
            name="execution_input.attention_mask",
        )
        return ActionBatch(
            input_ids=self.runtime.as_tensor(input_ids, device=self.device).long(),
            attention_mask=self.runtime.as_tensor(attention_mask, device=self.device).bool(),
            action_view=self.action_view,
        )


@dataclass(frozen=True)
class CodeLeWMScorer:
    """Python API wrapper for scoring candidate after-state files."""

    checkpoint: Path
    checkpoint_sha256: str
    backend: TransitionScoringBackend
    device: str = "auto"
    transition_index: TransitionIndex | None = None
    retrieval_prior_weight: float = 0.0
    retrieval_prior_k: int = 10

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
            warnings=self._warnings(),
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
        retrieval_prior = self._retrieval_prior(candidate)
        final_score = energy
        if retrieval_prior is not None:
            final_score = energy + self.retrieval_prior_weight * retrieval_prior
        return ScoreResult(
            candidate=candidate_name,
            transition_energy=energy,
            retrieval_prior=retrieval_prior,
            risk_penalty=None,
            final_score=final_score,
            model_id=self.model_id,
            checkpoint_sha256=self.checkpoint_sha256,
            input_digest=score_input_digest(before, instruction, candidate),
            warnings=self._warnings(),
        )

    def _retrieval_prior(self, candidate: str) -> float | None:
        if self.transition_index is None:
            return None
        if self.transition_index.count == 0:
            raise ScoreError(
                "transition index is empty",
                error_type="manifest_error",
                remediation="rebuild the transition index from a non-empty train split",
            )
        query = np.asarray(_hashed_vector(candidate, dim=self.transition_index.dim), dtype=np.float32)
        hits = self.transition_index.search(query, k=min(self.retrieval_prior_k, self.transition_index.count))
        if not hits:
            raise ScoreError(
                "transition index returned no hits",
                error_type="scoring_error",
                remediation="inspect the transition index and retry",
            )
        prior = float(sum(hit.distance for hit in hits) / len(hits))
        if not math.isfinite(prior):
            raise ScoreError("retrieval_prior must be finite")
        return prior

    def _warnings(self) -> tuple[str, ...]:
        warnings = list(self.backend.warnings)
        if self.transition_index is not None:
            warnings.append(
                "retrieval prior computed from local transition index "
                f"(k={self.retrieval_prior_k}, weight={self.retrieval_prior_weight:g})"
            )
            if self.retrieval_prior_weight == 0.0:
                warnings.append("retrieval prior weight is 0; final_score is unchanged")
        return tuple(warnings)


def load_scorer(
    checkpoint: Path | str,
    *,
    device: str = "auto",
    backend: TransitionScoringBackend | None = None,
    allow_unsafe: bool = False,
    require_learned_backend: bool = False,
    checkpoint_manifest: Path | str | None = None,
    index: Path | str | None = None,
    retrieval_prior_weight: float = 0.0,
    retrieval_prior_k: int = 10,
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
    trusted_manifest = None
    if not allow_unsafe:
        try:
            trusted_manifest = require_trusted_checkpoint(checkpoint_path, manifest_path=checkpoint_manifest)
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
    retrieval_weight = _non_negative_float(
        retrieval_prior_weight,
        "retrieval_prior_weight",
    )
    retrieval_k = _positive_int(retrieval_prior_k, "retrieval_prior_k")
    if index is None and retrieval_weight != 0.0:
        raise ScoreError(
            "retrieval_prior_weight requires an index",
            error_type="config_error",
            remediation="pass --index or set retrieval_prior_weight to 0",
        )
    transition_index = _load_transition_index(index)
    selected_backend = (
        _default_scoring_backend(
            checkpoint_path,
            device=device,
            trusted_manifest=trusted_manifest,
            allow_unsafe=allow_unsafe,
            require_learned_backend=require_learned_backend,
        )
        if backend is None
        else backend
    )
    return CodeLeWMScorer(
        checkpoint=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        backend=selected_backend,
        device=device,
        transition_index=transition_index,
        retrieval_prior_weight=retrieval_weight,
        retrieval_prior_k=retrieval_k,
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


def _default_scoring_backend(
    checkpoint: Path,
    *,
    device: str,
    trusted_manifest: Any | None,
    allow_unsafe: bool,
    require_learned_backend: bool,
) -> TransitionScoringBackend:
    model_class = None if trusted_manifest is None else trusted_manifest.metadata.model_class
    record_schema = None if trusted_manifest is None else trusted_manifest.metadata.record_schema_version
    is_execution_manifest = (
        model_class == "TorchCodeTransitionModel"
        and record_schema in SUPPORTED_EXECUTION_PACK_RECORD_SCHEMA_VERSIONS
    )
    is_torch_manifest = (
        model_class == "TorchCodeTransitionModel"
        and record_schema not in SUPPORTED_EXECUTION_PACK_RECORD_SCHEMA_VERSIONS
    )
    unsafe_schema = (
        _peek_torch_checkpoint_schema(checkpoint, device=device)
        if trusted_manifest is None and allow_unsafe and require_learned_backend
        else None
    )
    if is_execution_manifest or unsafe_schema == EXECUTION_TRAIN_CHECKPOINT_SCHEMA_VERSION:
        try:
            return ExecutionTorchTransitionScoringBackend.load(checkpoint, device=device)
        except ScoreError:
            if is_execution_manifest or require_learned_backend:
                raise
    should_load_torch = is_torch_manifest or (allow_unsafe and require_learned_backend)
    if should_load_torch:
        try:
            return TorchCheckpointTransitionScoringBackend.load(checkpoint, device=device)
        except ScoreError:
            if is_torch_manifest or require_learned_backend:
                raise
    if require_learned_backend:
        detail = "missing checkpoint manifest" if trusted_manifest is None else f"model_class={model_class!r}"
        raise ScoreError(
            f"learned scorer backend was required, but the checkpoint is not a torch transition checkpoint ({detail})",
            error_type="checkpoint_error",
            remediation="provide a trusted TorchCodeTransitionModel checkpoint manifest or disable the learned-backend requirement",
            artifact=str(checkpoint),
        )
    return HashingTransitionScoringBackend()


def _peek_torch_checkpoint_schema(checkpoint: Path, *, device: str) -> str | None:
    try:
        runtime = _require_torch_runtime_for_scoring()
        selected_device = _resolve_torch_device(device, runtime)
        try:
            payload = runtime.load(checkpoint, map_location=selected_device, weights_only=True)
        except TypeError:  # pragma: no cover - older torch compatibility.
            payload = runtime.load(checkpoint, map_location=selected_device)
    except ScoreError:
        return None
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    schema = payload.get("schema_version")
    return schema if isinstance(schema, str) else None


def _require_torch_runtime_for_scoring() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise ScoreError(
            "learned scoring requires torch",
            error_type="optional_dependency_missing",
            remediation="install the training runtime with `uv sync --group train --group dev`",
        )
    if importlib.util.find_spec("einops") is None:
        raise ScoreError(
            "learned scoring requires einops",
            error_type="optional_dependency_missing",
            remediation="install the training runtime with `uv sync --group train --group dev`",
        )
    import torch

    return torch


def _resolve_torch_device(device: str, runtime: Any) -> Any:
    if device not in {"cpu", "cuda", "mps", "auto"}:
        raise ScoreError(
            "device must be cpu, cuda, mps, or auto",
            error_type="config_error",
            remediation="choose one of cpu, cuda, mps, or auto",
        )
    if device == "auto":
        if runtime.cuda.is_available():
            return runtime.device("cuda")
        return runtime.device("cpu")
    if device == "cuda" and not runtime.cuda.is_available():
        raise ScoreError(
            "CUDA device requested but torch.cuda is unavailable",
            error_type="config_error",
            remediation="choose --device cpu, --device mps, or run on a CUDA host",
        )
    if device == "mps" and not (
        hasattr(runtime.backends, "mps") and runtime.backends.mps.is_available()
    ):
        raise ScoreError(
            "MPS device requested but torch.backends.mps is unavailable",
            error_type="config_error",
            remediation="choose --device cpu or run on an Apple Silicon host with MPS support",
        )
    return runtime.device(device)


def _build_torch_model_from_compatibility(
    compatibility: Mapping[str, Any],
    *,
    artifact: str,
    state_dict: Any = None,
) -> Any:
    from codelewm.model import (
        TorchCodeTransitionModelConfig,
        build_torch_transition_model,
        resolve_ema_target_encoder_config,
        resolve_state_encoder_arch,
    )

    wm = compatibility.get("wm")
    if not isinstance(wm, Mapping):
        raise ScoreError(
            "checkpoint compatibility_config.wm must be a mapping",
            error_type="checkpoint_error",
            remediation="provide a checkpoint written by codelewm train --executor torch",
            artifact=artifact,
        )
    action_view = str(wm.get("action_view", "text"))
    if action_view not in {"text", "abstract"}:
        raise ScoreError(
            "patch action is diagnostic only and cannot be a learned scoring model",
            error_type="checkpoint_error",
            remediation="provide a text or abstract action checkpoint",
            artifact=artifact,
        )
    encoder_type, encoder_layers, encoder_heads = resolve_state_encoder_arch(
        wm, state_dict
    )
    has_pass_head_weights = (
        isinstance(state_dict, Mapping)
        and any(str(key).startswith("pass_head.") for key in state_dict)
    )
    try:
        enable_ema_target_encoder, ema_target_decay = resolve_ema_target_encoder_config(
            wm, state_dict
        )
        config = TorchCodeTransitionModelConfig(
            action_view=action_view,  # type: ignore[arg-type]
            latent_dim=int(wm.get("embed_dim", 256)),
            state_sequence_length=int(wm.get("state_sequence_length", STATE_SEQUENCE_LENGTH)),
            action_sequence_length=int(
                wm.get(
                    "action_sequence_length",
                    TEXT_ACTION_SEQUENCE_LENGTH
                    if action_view == "text"
                    else ABSTRACT_ACTION_SEQUENCE_LENGTH,
                )
            ),
            vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
            dropout=0.0,
            action_fusion=str(wm.get("action_fusion", "conditional_transformer")),
            enable_inverse_action_head=bool(
                wm.get("enable_inverse_action_head")
                or (
                    isinstance(compatibility.get("loss"), Mapping)
                    and compatibility["loss"].get("enable_inverse_action_reconstruction")
                )
                or (
                    isinstance(compatibility.get("objective"), Mapping)
                    and float(
                        compatibility["objective"].get(
                            "inverse_action_reconstruction_weight",
                            0.0,
                        )
                    )
                    > 0.0
                )
            ),
            enable_pass_head=bool(wm.get("enable_pass_head") or has_pass_head_weights),
            enable_ema_target_encoder=enable_ema_target_encoder,
            ema_target_decay=ema_target_decay,
            state_encoder_type=encoder_type,
            state_encoder_layers=encoder_layers,
            state_encoder_heads=encoder_heads,
        )
        return build_torch_transition_model(config)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ScoreError(
            f"torch transition model could not be built from checkpoint compatibility config: {exc}",
            error_type="checkpoint_error",
            remediation="inspect the checkpoint compatibility_config",
            artifact=artifact,
            caused_by=f"{exc.__class__.__name__}: {exc}",
        ) from exc


def _torch_tokenize_text(text: str) -> tuple[str, ...]:
    return tuple(_DATA_TOKEN_RE.findall(text.strip()))


def _bucket_token_ids(values: Sequence[int], *, vocab_size: int) -> tuple[int, ...]:
    if vocab_size <= 1:
        raise ScoreError("vocab_size must be greater than 1", error_type="checkpoint_error")
    return tuple(((int(value) - 1) % (vocab_size - 1)) + 1 for value in values if int(value) > 0)


def _pad_int_array(values: Sequence[int], length: int, *, name: str) -> np.ndarray:
    if len(values) > length:
        raise ScoreError(
            f"{name} length {len(values)} exceeds fixed width {length}",
            error_type="scoring_error",
            remediation="shorten the source or instruction so it fits the CodeLeWM scorer contract",
        )
    output = np.zeros((1, length), dtype=np.int64)
    output[0, : len(values)] = tuple(int(value) for value in values)
    return output


def _pad_bool_array(values: Sequence[bool], length: int, *, name: str) -> np.ndarray:
    if len(values) > length:
        raise ScoreError(
            f"{name} length {len(values)} exceeds fixed width {length}",
            error_type="scoring_error",
            remediation="shorten the source or instruction so it fits the CodeLeWM scorer contract",
        )
    output = np.zeros((1, length), dtype=bool)
    output[0, : len(values)] = tuple(bool(value) for value in values)
    return output


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_transition_index(index: Path | str | None) -> TransitionIndex | None:
    if index is None:
        return None
    index_path = Path(index)
    try:
        return read_transition_index(index_path)
    except (TransitionIndexError, OSError, json.JSONDecodeError) as exc:
        raise ScoreError(
            f"transition index could not be loaded: {exc}",
            error_type="manifest_error",
            remediation="provide a valid transition index directory built by `codelewm index`",
            artifact=str(index_path),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        ) from exc


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
                    "optional_dependency_missing",
                    "dataset_build_error",
                    "empty_dataset",
                    "manifest_error",
                    "checkpoint_error",
                    "evaluation_gate_error",
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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ScoreError(
            f"{name} must be a positive integer",
            error_type="config_error",
            remediation="provide a positive integer",
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ScoreError(
            f"{name} must be a positive integer",
            error_type="config_error",
            remediation="provide a positive integer",
        ) from exc
    if result <= 0 or result != value:
        raise ScoreError(
            f"{name} must be a positive integer",
            error_type="config_error",
            remediation="provide a positive integer",
        )
    return result


def _non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ScoreError(
            f"{name} must be a non-negative finite number",
            error_type="config_error",
            remediation="provide a non-negative finite number",
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoreError(
            f"{name} must be a non-negative finite number",
            error_type="config_error",
            remediation="provide a non-negative finite number",
        ) from exc
    if result < 0.0 or not math.isfinite(result):
        raise ScoreError(
            f"{name} must be a non-negative finite number",
            error_type="config_error",
            remediation="provide a non-negative finite number",
        )
    return result


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
