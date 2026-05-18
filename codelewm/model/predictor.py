"""Predictor modules for pooled CodeState latent transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codelewm.model.actions import ModelRuntimeUnavailableError
from codelewm.model.transition import LATENT_DIM

try:  # pragma: no cover - exercised when torch is installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - lightweight local env.
    torch = None
    nn = None


@dataclass(frozen=True)
class CodeLatentPredictorConfig:
    """Configuration for one-step pooled code latent prediction."""

    history_size: int = 1
    num_preds: int = 1
    latent_dim: int = LATENT_DIM
    action_dim: int = LATENT_DIM
    hidden_dim: int = LATENT_DIM
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 1024
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")
        if self.num_preds != 1:
            raise ValueError("num_preds=1 is the only supported v0.1 predictor contract")
        if self.latent_dim <= 0 or self.action_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("predictor dimensions must be positive")
        if self.depth <= 0 or self.heads <= 0 or self.mlp_dim <= 0 or self.dim_head <= 0:
            raise ValueError("predictor architecture values must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.emb_dropout < 1.0:
            raise ValueError("emb_dropout must be in [0, 1)")


class CodeLatentPredictor(nn.Module if nn is not None else object):
    """One-step adapter from pooled code/action latents to predicted after-state latents."""

    def __init__(
        self,
        config: CodeLatentPredictorConfig = CodeLatentPredictorConfig(),
        *,
        predictor: Any | None = None,
        pred_proj: Any | None = None,
    ) -> None:
        if nn is None or torch is None:
            raise ModelRuntimeUnavailableError("CodeLatentPredictor requires torch")
        super().__init__()
        self.config = config
        self.predictor = predictor if predictor is not None else self._build_predictor(config)
        self.pred_proj = pred_proj if pred_proj is not None else nn.Identity()

    def forward(self, z_before: Any, action_emb: Any) -> Any:
        return self.predict_after(z_before, action_emb)

    def predict_after(self, z_before: Any, action_emb: Any) -> Any:
        """Predict a single after-state latent from code-state and action latents."""

        z_history = self._as_history(z_before, "z_before", self.config.latent_dim)
        action_history = self._as_history(action_emb, "action_emb", self.config.action_dim)
        if z_history.shape[:2] != action_history.shape[:2]:
            raise ValueError(
                "z_before and action_emb must share batch/history shape; "
                f"got {tuple(z_history.shape)} and {tuple(action_history.shape)}"
            )

        sequence_pred = self.predictor(z_history, action_history)
        if sequence_pred.ndim != 3:
            raise ValueError(f"predictor must return rank-3 sequence output; got rank {sequence_pred.ndim}")
        if sequence_pred.shape[:2] != z_history.shape[:2]:
            raise ValueError(
                "predictor output must preserve batch/history shape; "
                f"got {tuple(sequence_pred.shape)} for input {tuple(z_history.shape)}"
            )
        if sequence_pred.size(-1) != self.config.latent_dim:
            raise ValueError(
                f"predictor output latent dim must be {self.config.latent_dim}; "
                f"got {sequence_pred.size(-1)}"
            )

        return self.pred_proj(sequence_pred[:, -1, :])

    def _as_history(self, tensor: Any, name: str, expected_dim: int) -> Any:
        if not hasattr(tensor, "ndim"):
            raise TypeError(f"{name} must be a tensor-like object with ndim")

        if tensor.ndim == 2:
            if self.config.history_size != 1:
                raise ValueError(
                    f"{name} rank-2 pooled input requires history_size=1; "
                    f"got {self.config.history_size}"
                )
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [batch, dim] or [batch, history, dim]")

        if tensor.size(1) != self.config.history_size:
            raise ValueError(
                f"{name} history dimension must be {self.config.history_size}; got {tensor.size(1)}"
            )
        if tensor.size(-1) != expected_dim:
            raise ValueError(f"{name} latent dimension must be {expected_dim}; got {tensor.size(-1)}")
        return tensor

    @staticmethod
    def _build_predictor(config: CodeLatentPredictorConfig) -> Any:
        from codelewm.model.modules import ARPredictor

        return ARPredictor(
            num_frames=config.history_size,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            dim_head=config.dim_head,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
        )
