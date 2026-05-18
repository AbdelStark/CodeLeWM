"""Base CodeLeWM objective: next-latent MSE plus SIGReg."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when torch is installed.
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight local env.
    torch = None


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for the base v0.1 transition objective."""

    sigreg_weight: float = 0.09
    enable_retrieval_loss: bool = False
    retrieval_weight: float = 0.0
    retrieval_temperature: float = 0.1
    retrieval_weight_cap: float = 0.10
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    sigreg_seed: int | None = None

    def __post_init__(self) -> None:
        if self.sigreg_weight < 0.0:
            raise ValueError("sigreg_weight must be non-negative")
        if self.retrieval_weight < 0.0:
            raise ValueError("retrieval_weight must be non-negative")
        if self.retrieval_weight_cap <= 0.0:
            raise ValueError("retrieval_weight_cap must be positive")
        if self.retrieval_weight > self.retrieval_weight_cap:
            raise ValueError("retrieval_weight exceeds retrieval_weight_cap")
        if self.retrieval_temperature <= 0.0:
            raise ValueError("retrieval_temperature must be positive")
        if self.enable_retrieval_loss and self.retrieval_weight <= 0.0:
            raise ValueError("enable_retrieval_loss requires nonzero retrieval_weight")
        if not self.enable_retrieval_loss and self.retrieval_weight != 0.0:
            raise ValueError("retrieval_weight requires enable_retrieval_loss=true")
        if self.sigreg_knots < 2:
            raise ValueError("sigreg_knots must be at least 2")
        if self.sigreg_num_proj <= 0:
            raise ValueError("sigreg_num_proj must be positive")


@dataclass(frozen=True)
class ObjectiveTerms:
    """Objective tensors and scalar logging view."""

    total: Any
    prediction_mse: Any
    sigreg: Any
    sigreg_weighted: Any
    retrieval: Any | None = None
    retrieval_weighted: Any | None = None

    def scalars(self) -> dict[str, float]:
        values = {
            "loss/total": _as_float(self.total),
            "loss/prediction_mse": _as_float(self.prediction_mse),
            "loss/sigreg": _as_float(self.sigreg),
            "loss/sigreg_weighted": _as_float(self.sigreg_weighted),
        }
        if self.retrieval is not None:
            values["loss/retrieval"] = _as_float(self.retrieval)
        if self.retrieval_weighted is not None:
            values["loss/retrieval_weighted"] = _as_float(self.retrieval_weighted)
        return values


def compute_transition_objective(
    z_before: Any,
    z_after: Any,
    z_pred_after: Any,
    *,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> ObjectiveTerms:
    """Return MSE, SIGReg, and total loss for one-step latent prediction."""

    prediction_mse = compute_prediction_mse(z_pred_after, z_after)
    sigreg_input = stack_objective_embeddings(z_before, z_after, z_pred_after)
    sigreg = compute_sigreg_loss(
        sigreg_input,
        knots=config.sigreg_knots,
        num_proj=config.sigreg_num_proj,
        seed=config.sigreg_seed,
    )
    sigreg_weighted = sigreg * config.sigreg_weight
    total = prediction_mse + sigreg_weighted
    retrieval = None
    retrieval_weighted = None
    if config.enable_retrieval_loss:
        retrieval = compute_in_batch_retrieval_loss(
            z_pred_after,
            z_after,
            temperature=config.retrieval_temperature,
        )
        retrieval_weighted = retrieval * config.retrieval_weight
        total = total + retrieval_weighted
    _check_finite(total, "loss/total")
    _check_finite(prediction_mse, "loss/prediction_mse")
    _check_finite(sigreg, "loss/sigreg")
    if retrieval is not None:
        _check_finite(retrieval, "loss/retrieval")
    return ObjectiveTerms(
        total=total,
        prediction_mse=prediction_mse,
        sigreg=sigreg,
        sigreg_weighted=sigreg_weighted,
        retrieval=retrieval,
        retrieval_weighted=retrieval_weighted,
    )


def compute_prediction_mse(z_pred_after: Any, z_after: Any) -> Any:
    """Compute mean squared prediction error for after-state latents."""

    _require_same_shape(z_pred_after, z_after, "z_pred_after", "z_after")
    _check_finite(z_pred_after, "z_pred_after")
    _check_finite(z_after, "z_after")
    if _is_torch_tensor(z_pred_after):
        return (z_pred_after - z_after).pow(2).mean()
    diff = np.asarray(z_pred_after, dtype=float) - np.asarray(z_after, dtype=float)
    return float(np.square(diff).mean())


def compute_retrieval_score_matrix(z_pred_after: Any, z_after: Any, *, temperature: float = 0.1) -> Any:
    """Return in-batch cosine score matrix where diagonal entries are positives."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    _require_same_shape(z_pred_after, z_after, "z_pred_after", "z_after")
    _check_finite(z_pred_after, "z_pred_after")
    _check_finite(z_after, "z_after")
    if _is_torch_tensor(z_pred_after):
        pred = z_pred_after / z_pred_after.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        target = z_after / z_after.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        return (pred @ target.T) / temperature
    pred = np.asarray(z_pred_after, dtype=float)
    target = np.asarray(z_after, dtype=float)
    pred = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-12)
    target = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-12)
    return (pred @ target.T) / temperature


def compute_in_batch_retrieval_loss(z_pred_after: Any, z_after: Any, *, temperature: float = 0.1) -> Any:
    """Compute cross-entropy retrieval loss with diagonal targets."""

    scores = compute_retrieval_score_matrix(z_pred_after, z_after, temperature=temperature)
    if _is_torch_tensor(scores):
        targets = torch.arange(scores.size(0), device=scores.device)
        return torch.nn.functional.cross_entropy(scores, targets)
    stabilized = scores - scores.max(axis=1, keepdims=True)
    log_probs = stabilized - np.log(np.exp(stabilized).sum(axis=1, keepdims=True))
    return float(-np.diag(log_probs).mean())


def stack_objective_embeddings(z_before: Any, z_after: Any, z_pred_after: Any) -> Any:
    """Stack before, after, and predicted embeddings for SIGReg."""

    _require_same_shape(z_before, z_after, "z_before", "z_after")
    _require_same_shape(z_before, z_pred_after, "z_before", "z_pred_after")
    _check_finite(z_before, "z_before")
    _check_finite(z_after, "z_after")
    _check_finite(z_pred_after, "z_pred_after")
    if _is_torch_tensor(z_before):
        return torch.stack((z_before, z_after, z_pred_after), dim=0)
    return np.stack(
        (
            np.asarray(z_before, dtype=float),
            np.asarray(z_after, dtype=float),
            np.asarray(z_pred_after, dtype=float),
        ),
        axis=0,
    )


def compute_sigreg_loss(
    embeddings: Any,
    *,
    knots: int = 17,
    num_proj: int = 1024,
    seed: int | None = None,
) -> Any:
    """Compute Sketch Isotropic Gaussian Regularizer for embeddings."""

    if knots < 2:
        raise ValueError("knots must be at least 2")
    if num_proj <= 0:
        raise ValueError("num_proj must be positive")
    _check_finite(embeddings, "sigreg embeddings")
    if _is_torch_tensor(embeddings):
        return _sigreg_torch(embeddings, knots=knots, num_proj=num_proj, seed=seed)
    return _sigreg_numpy(np.asarray(embeddings, dtype=float), knots=knots, num_proj=num_proj, seed=seed)


def _sigreg_numpy(embeddings: np.ndarray, *, knots: int, num_proj: int, seed: int | None) -> float:
    if embeddings.ndim != 3:
        raise ValueError(f"sigreg embeddings must have shape [time, batch, dim]; got {embeddings.shape}")
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(embeddings.shape[-1], num_proj))
    projections = projections / np.maximum(np.linalg.norm(projections, axis=0, keepdims=True), 1e-12)
    t = np.linspace(0.0, 3.0, knots, dtype=float)
    dt = 3.0 / (knots - 1)
    weights = np.full((knots,), 2.0 * dt, dtype=float)
    weights[[0, -1]] = dt
    phi = np.exp(-(t**2) / 2.0)
    weights = weights * phi
    x_t = (embeddings @ projections)[..., None] * t
    err = np.square(np.cos(x_t).mean(axis=1) - phi) + np.square(np.sin(x_t).mean(axis=1))
    statistic = (err @ weights) * embeddings.shape[1]
    return float(statistic.mean())


def _sigreg_torch(embeddings: Any, *, knots: int, num_proj: int, seed: int | None) -> Any:
    if embeddings.ndim != 3:
        raise ValueError(
            f"sigreg embeddings must have shape [time, batch, dim]; got {tuple(embeddings.shape)}"
        )
    generator = None
    if seed is not None:
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(seed)
    projections = torch.randn(
        embeddings.size(-1),
        num_proj,
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    projections = projections / projections.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
    t = torch.linspace(0.0, 3.0, knots, dtype=embeddings.dtype, device=embeddings.device)
    dt = 3.0 / (knots - 1)
    weights = torch.full((knots,), 2.0 * dt, dtype=embeddings.dtype, device=embeddings.device)
    weights[0] = dt
    weights[-1] = dt
    phi = torch.exp(-(t.square()) / 2.0)
    weights = weights * phi
    x_t = (embeddings @ projections).unsqueeze(-1) * t
    err = (x_t.cos().mean(dim=1) - phi).square() + x_t.sin().mean(dim=1).square()
    statistic = (err @ weights) * embeddings.size(1)
    return statistic.mean()


def _require_same_shape(left: Any, right: Any, left_name: str, right_name: str) -> None:
    left_shape = _shape(left)
    right_shape = _shape(right)
    if left_shape != right_shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape; got {left_shape} and {right_shape}")


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    return tuple(np.asarray(value).shape)


def _check_finite(value: Any, name: str) -> None:
    if _is_torch_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or inf")
        return
    if not np.isfinite(np.asarray(value, dtype=float)).all():
        raise ValueError(f"{name} contains NaN or inf")


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and torch.is_tensor(value)


def _as_float(value: Any) -> float:
    if _is_torch_tensor(value):
        value = value.detach().cpu().item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("objective scalar contains NaN or inf")
    return result
