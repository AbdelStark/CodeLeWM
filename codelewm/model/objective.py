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
    prediction_mse_weight: float = 1.0
    enable_retrieval_loss: bool = False
    retrieval_weight: float = 0.0
    retrieval_temperature: float = 0.1
    retrieval_weight_cap: float = 0.10
    enable_action_use_margin: bool = False
    action_use_margin_weight: float = 0.0
    action_use_margin: float = 0.0
    enable_action_swap_contrastive: bool = False
    action_swap_contrastive_weight: float = 0.0
    action_swap_contrastive_margin: float = 0.0
    enable_inverse_action_reconstruction: bool = False
    inverse_action_reconstruction_weight: float = 0.0
    enable_p_pass_bce: bool = False
    p_pass_bce_weight: float = 0.0
    p_pass_bce_pos_weight: float = 1.0
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    sigreg_seed: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigreg_weight) or self.sigreg_weight < 0.0:
            raise ValueError("sigreg_weight must be finite and non-negative")
        if not math.isfinite(self.prediction_mse_weight) or self.prediction_mse_weight < 0.0:
            raise ValueError("prediction_mse_weight must be finite and non-negative")
        if not math.isfinite(self.retrieval_weight) or self.retrieval_weight < 0.0:
            raise ValueError("retrieval_weight must be finite and non-negative")
        if not math.isfinite(self.retrieval_weight_cap) or self.retrieval_weight_cap <= 0.0:
            raise ValueError("retrieval_weight_cap must be finite and positive")
        if self.retrieval_weight > self.retrieval_weight_cap:
            raise ValueError("retrieval_weight exceeds retrieval_weight_cap")
        if not math.isfinite(self.retrieval_temperature) or self.retrieval_temperature <= 0.0:
            raise ValueError("retrieval_temperature must be finite and positive")
        if self.enable_retrieval_loss and self.retrieval_weight <= 0.0:
            raise ValueError("enable_retrieval_loss requires nonzero retrieval_weight")
        if not self.enable_retrieval_loss and self.retrieval_weight != 0.0:
            raise ValueError("retrieval_weight requires enable_retrieval_loss=true")
        if not math.isfinite(self.action_use_margin_weight) or self.action_use_margin_weight < 0.0:
            raise ValueError("action_use_margin_weight must be finite and non-negative")
        if self.action_use_margin_weight > 1.0:
            raise ValueError("action_use_margin_weight must be at most 1.0")
        if not math.isfinite(self.action_use_margin) or self.action_use_margin < 0.0:
            raise ValueError("action_use_margin must be finite and non-negative")
        if self.enable_action_use_margin and self.action_use_margin_weight <= 0.0:
            raise ValueError("enable_action_use_margin requires nonzero action_use_margin_weight")
        if self.enable_action_use_margin and self.action_use_margin <= 0.0:
            raise ValueError("enable_action_use_margin requires nonzero action_use_margin")
        if not self.enable_action_use_margin and self.action_use_margin_weight != 0.0:
            raise ValueError("action_use_margin_weight requires enable_action_use_margin=true")
        if not self.enable_action_use_margin and self.action_use_margin != 0.0:
            raise ValueError("action_use_margin requires enable_action_use_margin=true")
        if not math.isfinite(self.action_swap_contrastive_weight) or self.action_swap_contrastive_weight < 0.0:
            raise ValueError("action_swap_contrastive_weight must be finite and non-negative")
        if self.action_swap_contrastive_weight > 1.0:
            raise ValueError("action_swap_contrastive_weight must be at most 1.0")
        if not math.isfinite(self.action_swap_contrastive_margin) or self.action_swap_contrastive_margin < 0.0:
            raise ValueError("action_swap_contrastive_margin must be finite and non-negative")
        if self.enable_action_swap_contrastive and self.action_swap_contrastive_weight <= 0.0:
            raise ValueError("enable_action_swap_contrastive requires nonzero action_swap_contrastive_weight")
        if self.enable_action_swap_contrastive and self.action_swap_contrastive_margin <= 0.0:
            raise ValueError("enable_action_swap_contrastive requires nonzero action_swap_contrastive_margin")
        if not self.enable_action_swap_contrastive and self.action_swap_contrastive_weight != 0.0:
            raise ValueError("action_swap_contrastive_weight requires enable_action_swap_contrastive=true")
        if not self.enable_action_swap_contrastive and self.action_swap_contrastive_margin != 0.0:
            raise ValueError("action_swap_contrastive_margin requires enable_action_swap_contrastive=true")
        if (
            not math.isfinite(self.inverse_action_reconstruction_weight)
            or self.inverse_action_reconstruction_weight < 0.0
        ):
            raise ValueError("inverse_action_reconstruction_weight must be finite and non-negative")
        if self.inverse_action_reconstruction_weight > 1.0:
            raise ValueError("inverse_action_reconstruction_weight must be at most 1.0")
        if self.enable_inverse_action_reconstruction and self.inverse_action_reconstruction_weight <= 0.0:
            raise ValueError(
                "enable_inverse_action_reconstruction requires nonzero inverse_action_reconstruction_weight"
            )
        if (
            not self.enable_inverse_action_reconstruction
            and self.inverse_action_reconstruction_weight != 0.0
        ):
            raise ValueError(
                "inverse_action_reconstruction_weight requires enable_inverse_action_reconstruction=true"
            )
        if not math.isfinite(self.p_pass_bce_weight) or self.p_pass_bce_weight < 0.0:
            raise ValueError("p_pass_bce_weight must be finite and non-negative")
        if not math.isfinite(self.p_pass_bce_pos_weight) or self.p_pass_bce_pos_weight <= 0.0:
            raise ValueError("p_pass_bce_pos_weight must be finite and positive")
        if self.enable_p_pass_bce and self.p_pass_bce_weight <= 0.0:
            raise ValueError("enable_p_pass_bce requires nonzero p_pass_bce_weight")
        if not self.enable_p_pass_bce and self.p_pass_bce_weight != 0.0:
            raise ValueError("p_pass_bce_weight requires enable_p_pass_bce=true")
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
    action_use_margin: Any | None = None
    action_use_margin_weighted: Any | None = None
    action_swap_contrastive: Any | None = None
    action_swap_contrastive_weighted: Any | None = None
    inverse_action_reconstruction: Any | None = None
    inverse_action_reconstruction_weighted: Any | None = None
    p_pass_bce: Any | None = None
    p_pass_bce_weighted: Any | None = None

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
        if self.action_use_margin is not None:
            values["loss/action_use_margin"] = _as_float(self.action_use_margin)
        if self.action_use_margin_weighted is not None:
            values["loss/action_use_margin_weighted"] = _as_float(self.action_use_margin_weighted)
        if self.action_swap_contrastive is not None:
            values["loss/action_swap_contrastive"] = _as_float(self.action_swap_contrastive)
        if self.action_swap_contrastive_weighted is not None:
            values["loss/action_swap_contrastive_weighted"] = _as_float(
                self.action_swap_contrastive_weighted
            )
        if self.inverse_action_reconstruction is not None:
            values["loss/inverse_action_reconstruction"] = _as_float(
                self.inverse_action_reconstruction
            )
        if self.inverse_action_reconstruction_weighted is not None:
            values["loss/inverse_action_reconstruction_weighted"] = _as_float(
                self.inverse_action_reconstruction_weighted
            )
        if self.p_pass_bce is not None:
            values["loss/p_pass_bce"] = _as_float(self.p_pass_bce)
        if self.p_pass_bce_weighted is not None:
            values["loss/p_pass_bce_weighted"] = _as_float(
                self.p_pass_bce_weighted
            )
        return values


def compute_transition_objective(
    z_before: Any,
    z_after: Any,
    z_pred_after: Any,
    *,
    config: ObjectiveConfig = ObjectiveConfig(),
    z_pred_after_swapped: Any | None = None,
    action_emb: Any | None = None,
    action_reconstruction: Any | None = None,
    p_pass_logit: Any | None = None,
    pass_labels: Any | None = None,
    z_after_for_sigreg: Any | None = None,
) -> ObjectiveTerms:
    """Return MSE, SIGReg, and total loss for one-step latent prediction."""

    prediction_mse = compute_prediction_mse(z_pred_after, z_after)
    sigreg_target = z_after if z_after_for_sigreg is None else z_after_for_sigreg
    sigreg_input = stack_objective_embeddings(z_before, sigreg_target, z_pred_after)
    sigreg = compute_sigreg_loss(
        sigreg_input,
        knots=config.sigreg_knots,
        num_proj=config.sigreg_num_proj,
        seed=config.sigreg_seed,
    )
    sigreg_weighted = sigreg * config.sigreg_weight
    total = config.prediction_mse_weight * prediction_mse + sigreg_weighted
    retrieval = None
    retrieval_weighted = None
    action_use_margin = None
    action_use_margin_weighted = None
    action_swap_contrastive = None
    action_swap_contrastive_weighted = None
    inverse_action_reconstruction = None
    inverse_action_reconstruction_weighted = None
    p_pass_bce = None
    p_pass_bce_weighted = None
    if config.enable_retrieval_loss:
        retrieval = compute_in_batch_retrieval_loss(
            z_pred_after,
            z_after,
            temperature=config.retrieval_temperature,
        )
        retrieval_weighted = retrieval * config.retrieval_weight
        total = total + retrieval_weighted
    if config.enable_action_use_margin:
        action_use_margin = compute_action_use_margin_loss(
            z_before,
            z_after,
            z_pred_after,
            margin=config.action_use_margin,
        )
        action_use_margin_weighted = action_use_margin * config.action_use_margin_weight
        total = total + action_use_margin_weighted
    if config.enable_action_swap_contrastive:
        if z_pred_after_swapped is None:
            raise ValueError("enable_action_swap_contrastive requires z_pred_after_swapped")
        action_swap_contrastive = compute_action_swap_contrastive_loss(
            z_after,
            z_pred_after,
            z_pred_after_swapped,
            margin=config.action_swap_contrastive_margin,
        )
        action_swap_contrastive_weighted = (
            action_swap_contrastive * config.action_swap_contrastive_weight
        )
        total = total + action_swap_contrastive_weighted
    if config.enable_inverse_action_reconstruction:
        if action_emb is None or action_reconstruction is None:
            raise ValueError(
                "enable_inverse_action_reconstruction requires action_emb and action_reconstruction"
            )
        inverse_action_reconstruction = compute_inverse_action_reconstruction_loss(
            action_reconstruction,
            action_emb,
        )
        inverse_action_reconstruction_weighted = (
            inverse_action_reconstruction * config.inverse_action_reconstruction_weight
        )
        total = total + inverse_action_reconstruction_weighted
    if config.enable_p_pass_bce:
        if p_pass_logit is None or pass_labels is None:
            raise ValueError("enable_p_pass_bce requires p_pass_logit and pass_labels")
        p_pass_bce = compute_p_pass_bce_loss(
            p_pass_logit,
            pass_labels,
            pos_weight=config.p_pass_bce_pos_weight,
        )
        p_pass_bce_weighted = p_pass_bce * config.p_pass_bce_weight
        total = total + p_pass_bce_weighted
    _check_finite(total, "loss/total")
    _check_finite(prediction_mse, "loss/prediction_mse")
    _check_finite(sigreg, "loss/sigreg")
    if retrieval is not None:
        _check_finite(retrieval, "loss/retrieval")
    if action_use_margin is not None:
        _check_finite(action_use_margin, "loss/action_use_margin")
    if action_swap_contrastive is not None:
        _check_finite(action_swap_contrastive, "loss/action_swap_contrastive")
    if inverse_action_reconstruction is not None:
        _check_finite(inverse_action_reconstruction, "loss/inverse_action_reconstruction")
    if p_pass_bce is not None:
        _check_finite(p_pass_bce, "loss/p_pass_bce")
    return ObjectiveTerms(
        total=total,
        prediction_mse=prediction_mse,
        sigreg=sigreg,
        sigreg_weighted=sigreg_weighted,
        retrieval=retrieval,
        retrieval_weighted=retrieval_weighted,
        action_use_margin=action_use_margin,
        action_use_margin_weighted=action_use_margin_weighted,
        action_swap_contrastive=action_swap_contrastive,
        action_swap_contrastive_weighted=action_swap_contrastive_weighted,
        inverse_action_reconstruction=inverse_action_reconstruction,
        inverse_action_reconstruction_weighted=inverse_action_reconstruction_weighted,
        p_pass_bce=p_pass_bce,
        p_pass_bce_weighted=p_pass_bce_weighted,
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


def compute_action_use_margin_loss(
    z_before: Any,
    z_after: Any,
    z_pred_after: Any,
    *,
    margin: float,
) -> Any:
    """Penalize predictions that do not beat the no-action identity baseline."""

    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    _require_same_shape(z_before, z_after, "z_before", "z_after")
    _require_same_shape(z_pred_after, z_after, "z_pred_after", "z_after")
    _check_finite(z_before, "z_before")
    _check_finite(z_after, "z_after")
    _check_finite(z_pred_after, "z_pred_after")
    if _is_torch_tensor(z_pred_after):
        target = z_after.detach()
        no_action_delta = z_before.detach() - target
        pred_delta = z_pred_after - target
        reduce_dims = tuple(range(1, pred_delta.ndim))
        if reduce_dims:
            no_action_dist = no_action_delta.pow(2).mean(dim=reduce_dims)
            pred_dist = pred_delta.pow(2).mean(dim=reduce_dims)
        else:
            no_action_dist = no_action_delta.pow(2)
            pred_dist = pred_delta.pow(2)
        return torch.relu(pred_dist - no_action_dist + margin).mean()
    before = np.asarray(z_before, dtype=float)
    target = np.asarray(z_after, dtype=float)
    pred = np.asarray(z_pred_after, dtype=float)
    reduce_axes = tuple(range(1, pred.ndim))
    no_action_delta = before - target
    pred_delta = pred - target
    if reduce_axes:
        no_action_dist = np.square(no_action_delta).mean(axis=reduce_axes)
        pred_dist = np.square(pred_delta).mean(axis=reduce_axes)
    else:
        no_action_dist = np.square(no_action_delta)
        pred_dist = np.square(pred_delta)
    return float(np.maximum(pred_dist - no_action_dist + margin, 0.0).mean())


def compute_action_swap_contrastive_loss(
    z_after: Any,
    z_pred_after: Any,
    z_pred_after_swapped: Any,
    *,
    margin: float,
) -> Any:
    """Penalize swapped-action predictions that stay too close to the true after-state."""

    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    _require_same_shape(z_after, z_pred_after, "z_after", "z_pred_after")
    _require_same_shape(z_after, z_pred_after_swapped, "z_after", "z_pred_after_swapped")
    _check_finite(z_after, "z_after")
    _check_finite(z_pred_after, "z_pred_after")
    _check_finite(z_pred_after_swapped, "z_pred_after_swapped")
    if _is_torch_tensor(z_after):
        target = z_after.detach()
        positive_delta = z_pred_after - target
        swapped_delta = z_pred_after_swapped - target
        reduce_dims = tuple(range(1, positive_delta.ndim))
        if reduce_dims:
            positive_dist = positive_delta.pow(2).mean(dim=reduce_dims)
            swapped_dist = swapped_delta.pow(2).mean(dim=reduce_dims)
        else:
            positive_dist = positive_delta.pow(2)
            swapped_dist = swapped_delta.pow(2)
        return torch.relu(positive_dist - swapped_dist + margin).mean()
    target = np.asarray(z_after, dtype=float)
    positive = np.asarray(z_pred_after, dtype=float)
    swapped = np.asarray(z_pred_after_swapped, dtype=float)
    reduce_axes = tuple(range(1, positive.ndim))
    positive_delta = positive - target
    swapped_delta = swapped - target
    if reduce_axes:
        positive_dist = np.square(positive_delta).mean(axis=reduce_axes)
        swapped_dist = np.square(swapped_delta).mean(axis=reduce_axes)
    else:
        positive_dist = np.square(positive_delta)
        swapped_dist = np.square(swapped_delta)
    return float(np.maximum(positive_dist - swapped_dist + margin, 0.0).mean())


def compute_inverse_action_reconstruction_loss(action_reconstruction: Any, action_emb: Any) -> Any:
    """Recover the encoded action from before/after latents as an auxiliary signal."""

    _require_same_shape(action_reconstruction, action_emb, "action_reconstruction", "action_emb")
    _check_finite(action_reconstruction, "action_reconstruction")
    _check_finite(action_emb, "action_emb")
    if _is_torch_tensor(action_reconstruction):
        return (action_reconstruction - action_emb.detach()).pow(2).mean()
    reconstruction = np.asarray(action_reconstruction, dtype=float)
    target = np.asarray(action_emb, dtype=float)
    return float(np.square(reconstruction - target).mean())


def compute_p_pass_bce_loss(
    p_pass_logit: Any,
    pass_labels: Any,
    *,
    pos_weight: float = 1.0,
) -> Any:
    """Binary cross entropy over execution pass/fail labels."""

    if not math.isfinite(pos_weight) or pos_weight <= 0.0:
        raise ValueError("pos_weight must be finite and positive")
    if _is_torch_tensor(p_pass_logit):
        labels = pass_labels
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, device=p_pass_logit.device)
        labels = labels.to(device=p_pass_logit.device, dtype=p_pass_logit.dtype)
        logits = p_pass_logit.squeeze(-1)
        _require_same_shape(logits, labels, "p_pass_logit", "pass_labels")
        _check_finite(logits, "p_pass_logit")
        _check_finite(labels, "pass_labels")
        invalid = ((labels != 0.0) & (labels != 1.0)).any()
        if bool(invalid.detach().cpu().item()):
            raise ValueError("pass_labels must contain only 0/1 or bool values")
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=torch.as_tensor(
                pos_weight, dtype=logits.dtype, device=logits.device
            ),
        )
    logits = np.asarray(p_pass_logit, dtype=float)
    if logits.ndim > 0 and logits.shape[-1] == 1:
        logits = np.squeeze(logits, axis=-1)
    labels = np.asarray(pass_labels, dtype=float)
    _require_same_shape(logits, labels, "p_pass_logit", "pass_labels")
    _check_finite(logits, "p_pass_logit")
    _check_finite(labels, "pass_labels")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("pass_labels must contain only 0/1 or bool values")
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    loss = -(
        pos_weight * labels * np.log(probs)
        + (1.0 - labels) * np.log(1.0 - probs)
    )
    return float(loss.mean())


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
