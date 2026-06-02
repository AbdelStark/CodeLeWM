"""Torch-backed CodeLeWM transition model assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codelewm.model.actions import (
    AbstractActionEncoder,
    AbstractActionEncoderConfig,
    ModelRuntimeUnavailableError,
    TextActionEncoder,
    TextActionEncoderConfig,
)
from codelewm.model.predictor import CodeLatentPredictor, CodeLatentPredictorConfig
from codelewm.model.state import CodeStateEncoder, CodeStateEncoderConfig
from codelewm.model.transition import (
    LATENT_DIM,
    ABSTRACT_ACTION_SEQUENCE_LENGTH,
    STATE_SEQUENCE_LENGTH,
    TEXT_ACTION_SEQUENCE_LENGTH,
    ActionBatch,
    ActionView,
    CodeStateBatch,
    CodeTransitionModel,
    TransitionBatch,
)

try:  # pragma: no cover - exercised when torch is installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - lightweight local env.
    torch = None
    nn = None


@dataclass(frozen=True)
class TorchCodeTransitionModelConfig:
    """Configuration for the package-native torch transition model."""

    action_view: ActionView = "text"
    latent_dim: int = LATENT_DIM
    state_sequence_length: int = STATE_SEQUENCE_LENGTH
    action_sequence_length: int = TEXT_ACTION_SEQUENCE_LENGTH
    vocab_size: int = 32768
    dropout: float = 0.1
    action_fusion: str = "conditional_transformer"
    enable_inverse_action_head: bool = False
    # RFC-0015 WS-C1: state encoder backbone. "pool" is the v0.6 default;
    # "transformer" adds a contextual encoder before pooling.
    state_encoder_type: str = "pool"
    state_encoder_layers: int = 4
    state_encoder_heads: int = 8

    def __post_init__(self) -> None:
        if self.action_view not in ("text", "abstract"):
            raise ValueError("action_view must be 'text' or 'abstract'; patch is diagnostic only")
        if self.latent_dim != LATENT_DIM:
            raise ValueError(f"latent_dim must be {LATENT_DIM}")
        if self.state_sequence_length != STATE_SEQUENCE_LENGTH:
            raise ValueError(f"state_sequence_length must be {STATE_SEQUENCE_LENGTH}")
        expected_action_length = (
            TEXT_ACTION_SEQUENCE_LENGTH
            if self.action_view == "text"
            else ABSTRACT_ACTION_SEQUENCE_LENGTH
        )
        if self.action_sequence_length != expected_action_length:
            raise ValueError(
                "action_sequence_length must be "
                f"{expected_action_length} for action_view={self.action_view!r}"
            )
        if self.action_fusion not in {"conditional_transformer", "gated_residual"}:
            raise ValueError("action_fusion must be conditional_transformer or gated_residual")


class TorchCodeTransitionModel(CodeTransitionModel):
    """Concrete torch module for one-step CodeLeWM transition prediction."""

    def __init__(
        self,
        *,
        state_encoder: Any,
        action_encoder: Any,
        predictor: Any,
        config: TorchCodeTransitionModelConfig,
    ) -> None:
        if nn is None or torch is None:
            raise ModelRuntimeUnavailableError("TorchCodeTransitionModel requires torch")
        super().__init__()
        self.config = config
        self.encoder = state_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.projector = nn.Identity()
        self.pred_proj = nn.Identity()
        self.inverse_action_head = (
            nn.Sequential(
                nn.Linear(config.latent_dim * 2, config.latent_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.latent_dim, config.latent_dim),
            )
            if config.enable_inverse_action_head
            else None
        )

    def encode_state(self, batch: CodeStateBatch) -> Any:
        return self.encoder(
            batch.input_ids,
            batch.attention_mask,
            batch.segment_ids,
            batch.changed_hunk_mask,
        )

    def encode_action(self, batch: ActionBatch) -> Any:
        if batch.action_view != self.config.action_view:
            raise ValueError(
                f"action batch view {batch.action_view!r} does not match model action_view "
                f"{self.config.action_view!r}"
            )
        return self.action_encoder(batch.input_ids, batch.attention_mask)

    def predict_after(self, z_before: Any, action_emb: Any) -> Any:
        return self.predictor.predict_after(z_before, action_emb)

    def reconstruct_action(self, z_before: Any, z_after: Any) -> Any:
        if self.inverse_action_head is None:
            raise ValueError("inverse action head is disabled")
        return self.inverse_action_head(torch.cat((z_before, z_after), dim=-1))

    def forward(self, batch: TransitionBatch) -> dict[str, Any]:
        z_before = self.encode_state(batch.state_before)
        action_emb = self.encode_action(batch.action)
        z_after = self.encode_state(batch.state_after)
        z_pred_after = self.predict_after(z_before, action_emb)
        output = {
            "z_before": z_before,
            "action_emb": action_emb,
            "z_after": z_after,
            "z_pred_after": z_pred_after,
        }
        if self.inverse_action_head is not None:
            output["action_reconstruction"] = self.reconstruct_action(z_before, z_after)
        return output


def build_torch_transition_model(
    config: TorchCodeTransitionModelConfig = TorchCodeTransitionModelConfig(),
) -> TorchCodeTransitionModel:
    """Build the default package-native torch transition model."""

    state_encoder = CodeStateEncoder(
        CodeStateEncoderConfig(
            vocab_size=config.vocab_size,
            max_length=config.state_sequence_length,
            latent_dim=config.latent_dim,
            embed_dim=config.latent_dim,
            dropout=config.dropout,
            encoder_type=config.state_encoder_type,
            num_layers=config.state_encoder_layers,
            num_heads=config.state_encoder_heads,
        )
    )
    if config.action_view == "text":
        action_encoder = TextActionEncoder(
            TextActionEncoderConfig(
                vocab_size=config.vocab_size,
                max_length=config.action_sequence_length,
                latent_dim=config.latent_dim,
                embed_dim=config.latent_dim,
                dropout=config.dropout,
            )
        )
    elif config.action_view == "abstract":
        action_encoder = AbstractActionEncoder(
            AbstractActionEncoderConfig(
                vocab_size=config.vocab_size,
                max_length=config.action_sequence_length,
                latent_dim=config.latent_dim,
                embed_dim=config.latent_dim,
                dropout=config.dropout,
            )
        )
    else:  # pragma: no cover - guarded by config validation.
        raise ValueError("patch action is diagnostic only and cannot train the headline model")

    predictor = CodeLatentPredictor(
        CodeLatentPredictorConfig(
            history_size=1,
            num_preds=1,
            latent_dim=config.latent_dim,
            action_dim=config.latent_dim,
            hidden_dim=config.latent_dim,
            action_fusion=config.action_fusion,
        )
    )
    return TorchCodeTransitionModel(
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        config=config,
    )
