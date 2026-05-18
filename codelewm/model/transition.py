"""Typed transition-model contracts for CodeLeWM.

This module intentionally avoids requiring the ML runtime at import time. When
Torch is installed, `CodeTransitionModel` subclasses `torch.nn.Module`; otherwise
it remains an importable interface class so packaging and CLI smoke tests can run
in lightweight environments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

try:  # pragma: no cover - exercised when the optional ML stack is installed.
    from torch import nn as _torch_nn
except ModuleNotFoundError:  # pragma: no cover - local smoke env has no torch.
    _BaseModule = object
else:  # pragma: no cover
    _BaseModule = _torch_nn.Module


TensorLike = Any
ShapeDim = int | str
ActionView = Literal["text", "abstract", "patch"]
Reduction = Literal["none", "mean", "sum"]

STATE_SEQUENCE_LENGTH = 1024
TEXT_ACTION_SEQUENCE_LENGTH = 256
ABSTRACT_ACTION_SEQUENCE_LENGTH = 192
LATENT_DIM = 256


@dataclass(frozen=True)
class TensorContract:
    """Named tensor contract for public CodeLeWM interfaces."""

    name: str
    shape: tuple[ShapeDim, ...]
    dtype: str
    device: str | None = None


STATE_INPUT_IDS = TensorContract("state.input_ids", ("batch", STATE_SEQUENCE_LENGTH), "int64")
STATE_ATTENTION_MASK = TensorContract(
    "state.attention_mask", ("batch", STATE_SEQUENCE_LENGTH), "bool"
)
STATE_SEGMENT_IDS = TensorContract("state.segment_ids", ("batch", STATE_SEQUENCE_LENGTH), "int64")
ACTION_TEXT_INPUT_IDS = TensorContract(
    "action_text.input_ids", ("batch", TEXT_ACTION_SEQUENCE_LENGTH), "int64"
)
ACTION_ABSTRACT_INPUT_IDS = TensorContract(
    "action_abs.input_ids", ("batch", ABSTRACT_ACTION_SEQUENCE_LENGTH), "int64"
)
LATENT_EMBEDDING = TensorContract("latent", ("batch", LATENT_DIM), "float")


@dataclass(frozen=True)
class CodeStateBatch:
    """Tokenized CodeState batch consumed by state encoders."""

    input_ids: TensorLike
    attention_mask: TensorLike
    segment_ids: TensorLike
    changed_hunk_mask: TensorLike | None = None

    contracts: ClassVar[tuple[TensorContract, ...]] = (
        STATE_INPUT_IDS,
        STATE_ATTENTION_MASK,
        STATE_SEGMENT_IDS,
    )


@dataclass(frozen=True)
class ActionBatch:
    """Tokenized edit-action batch consumed by action encoders."""

    input_ids: TensorLike
    attention_mask: TensorLike
    action_view: ActionView = "text"

    @property
    def expected_sequence_length(self) -> int:
        return expected_action_sequence_length(self.action_view)


@dataclass(frozen=True)
class TransitionBatch:
    """One-step transition batch for latent after-state prediction."""

    state_before: CodeStateBatch
    action: ActionBatch
    state_after: CodeStateBatch


class CodeTransitionModel(_BaseModule):
    """Interface for action-conditioned latent transition models."""

    encoder: Any
    action_encoder: Any
    predictor: Any
    projector: Any
    pred_proj: Any

    def encode_state(self, batch: CodeStateBatch) -> TensorLike:
        raise NotImplementedError

    def encode_action(self, batch: ActionBatch) -> TensorLike:
        raise NotImplementedError

    def predict_after(self, z_before: TensorLike, action_emb: TensorLike) -> TensorLike:
        raise NotImplementedError

    def transition_energy(
        self,
        z_pred_after: TensorLike,
        z_after: TensorLike,
        *,
        reduction: Reduction = "none",
    ) -> TensorLike:
        return transition_energy(z_pred_after, z_after, reduction=reduction)


def expected_action_sequence_length(action_view: ActionView) -> int:
    if action_view == "text":
        return TEXT_ACTION_SEQUENCE_LENGTH
    if action_view == "abstract":
        return ABSTRACT_ACTION_SEQUENCE_LENGTH
    if action_view == "patch":
        return 512
    raise ValueError(f"Unsupported action view: {action_view}")


def infer_shape(value: TensorLike) -> tuple[int, ...]:
    """Infer a simple tensor-like shape for validation and error messages."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return (0,)
        return (len(value), *infer_shape(value[0]))

    return ()


def transition_energy(
    z_pred_after: TensorLike,
    z_after: TensorLike,
    *,
    reduction: Reduction = "none",
) -> TensorLike:
    """Return squared last-dimension distance between predicted and target latents."""

    pred_shape = infer_shape(z_pred_after)
    after_shape = infer_shape(z_after)
    if pred_shape != after_shape:
        raise ValueError(
            "z_pred_after and z_after must have the same shape; "
            f"got {pred_shape} and {after_shape}"
        )
    if not pred_shape:
        raise ValueError("transition_energy expects at least one latent dimension")

    try:
        diff = z_pred_after - z_after
        if hasattr(diff, "pow"):
            energy = diff.pow(2).sum(dim=-1)
        else:
            energy = np.square(diff).sum(axis=-1)
    except TypeError:
        energy = _transition_energy_from_sequences(z_pred_after, z_after)

    return _reduce_energy(energy, reduction)


def _transition_energy_from_sequences(z_pred_after: TensorLike, z_after: TensorLike) -> TensorLike:
    pred = np.asarray(z_pred_after, dtype=float)
    after = np.asarray(z_after, dtype=float)
    return np.square(pred - after).sum(axis=-1)


def _reduce_energy(energy: TensorLike, reduction: Reduction) -> TensorLike:
    if reduction == "none":
        if isinstance(energy, np.ndarray):
            return energy.item() if energy.ndim == 0 else energy.tolist()
        return energy
    if reduction == "mean":
        return energy.mean() if hasattr(energy, "mean") else float(np.asarray(energy).mean())
    if reduction == "sum":
        return energy.sum() if hasattr(energy, "sum") else float(np.asarray(energy).sum())
    raise ValueError(f"Unsupported reduction: {reduction}")
