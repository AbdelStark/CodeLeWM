"""Torch-backed CodeLeWM transition model assembly."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
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
    enable_pass_head: bool = False
    enable_output_value_head: bool = False
    output_type_class_count: int = 12
    output_magnitude_bucket_class_count: int = 5
    output_length_bucket_class_count: int = 5
    enable_ema_target_encoder: bool = False
    ema_target_decay: float = 0.99
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
        if self.output_type_class_count <= 0:
            raise ValueError("output_type_class_count must be positive")
        if self.output_magnitude_bucket_class_count <= 0:
            raise ValueError("output_magnitude_bucket_class_count must be positive")
        if self.output_length_bucket_class_count <= 0:
            raise ValueError("output_length_bucket_class_count must be positive")
        if not math.isfinite(self.ema_target_decay) or not 0.0 <= self.ema_target_decay < 1.0:
            raise ValueError("ema_target_decay must be finite and in [0.0, 1.0)")


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
        self.target_encoder = (
            self._build_target_encoder(state_encoder)
            if config.enable_ema_target_encoder
            else None
        )
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
        self.pass_head = (
            nn.Sequential(
                nn.Linear(config.latent_dim * 3, config.latent_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.latent_dim, 1),
            )
            if config.enable_pass_head
            else None
        )
        self.output_value_head = (
            nn.ModuleDict(
                {
                    "shared": nn.Sequential(
                        nn.Linear(config.latent_dim, config.latent_dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                    ),
                    "output_type": nn.Linear(
                        config.latent_dim, config.output_type_class_count
                    ),
                    "output_magnitude_bucket": nn.Linear(
                        config.latent_dim,
                        config.output_magnitude_bucket_class_count,
                    ),
                    "output_length_bucket": nn.Linear(
                        config.latent_dim,
                        config.output_length_bucket_class_count,
                    ),
                }
            )
            if config.enable_output_value_head
            else None
        )

    def train(self, mode: bool = True) -> Any:
        result = super().train(mode)
        if self.target_encoder is not None and hasattr(self.target_encoder, "eval"):
            self.target_encoder.eval()
        return result

    def _build_target_encoder(self, state_encoder: Any) -> Any:
        target_encoder = copy.deepcopy(state_encoder)
        if hasattr(target_encoder, "requires_grad_"):
            target_encoder.requires_grad_(False)
        else:
            for parameter in getattr(target_encoder, "parameters", lambda: ())():
                parameter.requires_grad_(False)
        if hasattr(target_encoder, "eval"):
            target_encoder.eval()
        return target_encoder

    def encode_state(self, batch: CodeStateBatch) -> Any:
        return self.encoder(
            batch.input_ids,
            batch.attention_mask,
            batch.segment_ids,
            batch.changed_hunk_mask,
        )

    def encode_target_state(self, batch: CodeStateBatch) -> Any:
        if self.target_encoder is None:
            return self.encode_state(batch)
        with torch.no_grad():
            return self.target_encoder(
                batch.input_ids,
                batch.attention_mask,
                batch.segment_ids,
                batch.changed_hunk_mask,
            ).detach()

    def update_ema_target_encoder(self, decay: float | None = None) -> None:
        if self.target_encoder is None:
            raise ValueError("EMA target encoder is disabled")
        decay_value = self.config.ema_target_decay if decay is None else float(decay)
        if not math.isfinite(decay_value) or not 0.0 <= decay_value < 1.0:
            raise ValueError("EMA target decay must be finite and in [0.0, 1.0)")

        online_params = _named_parameters(self.encoder)
        target_params = _named_parameters(self.target_encoder)
        if set(online_params) != set(target_params):
            raise RuntimeError("EMA target encoder parameters do not match online encoder")
        with torch.no_grad():
            for name, online_param in online_params.items():
                target_param = target_params[name]
                target_param.mul_(decay_value).add_(
                    online_param.detach(),
                    alpha=1.0 - decay_value,
                )

            online_buffers = _named_buffers(self.encoder)
            target_buffers = _named_buffers(self.target_encoder)
            if set(online_buffers) != set(target_buffers):
                raise RuntimeError("EMA target encoder buffers do not match online encoder")
            for name, online_buffer in online_buffers.items():
                target_buffers[name].copy_(online_buffer.detach())

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

    def pass_logit(self, z_before: Any, action_emb: Any, z_pred_after: Any) -> Any:
        if self.pass_head is None:
            raise ValueError("pass head is disabled")
        if z_before.shape[-1] != self.config.latent_dim:
            raise ValueError("z_before latent dimension does not match config")
        if action_emb.shape[-1] != self.config.latent_dim:
            raise ValueError("action_emb latent dimension does not match config")
        if z_pred_after.shape[-1] != self.config.latent_dim:
            raise ValueError("z_pred_after latent dimension does not match config")
        return self.pass_head(torch.cat((z_before, action_emb, z_pred_after), dim=-1))

    def output_value_logits(self, z_pred_after: Any) -> dict[str, Any]:
        if self.output_value_head is None:
            raise ValueError("output value head is disabled")
        if z_pred_after.shape[-1] != self.config.latent_dim:
            raise ValueError("z_pred_after latent dimension does not match config")
        hidden = self.output_value_head["shared"](z_pred_after)
        return {
            "output_type": self.output_value_head["output_type"](hidden),
            "output_magnitude_bucket": self.output_value_head[
                "output_magnitude_bucket"
            ](hidden),
            "output_length_bucket": self.output_value_head[
                "output_length_bucket"
            ](hidden),
        }

    def forward(self, batch: TransitionBatch) -> dict[str, Any]:
        z_before = self.encode_state(batch.state_before)
        action_emb = self.encode_action(batch.action)
        z_after_online = self.encode_state(batch.state_after)
        z_after = (
            self.encode_target_state(batch.state_after)
            if self.target_encoder is not None
            else z_after_online
        )
        z_pred_after = self.predict_after(z_before, action_emb)
        output = {
            "z_before": z_before,
            "action_emb": action_emb,
            "z_after": z_after,
            "z_pred_after": z_pred_after,
        }
        if self.target_encoder is not None:
            output["z_after_online"] = z_after_online
        if self.inverse_action_head is not None:
            output["action_reconstruction"] = self.reconstruct_action(
                z_before, z_after_online
            )
        if self.pass_head is not None:
            output["pass_logit"] = self.pass_logit(z_before, action_emb, z_pred_after)
        if self.output_value_head is not None:
            output["output_value_logits"] = self.output_value_logits(z_pred_after)
        return output


def resolve_state_encoder_arch(
    wm_config: Any, state_dict: Any
) -> tuple[str, int, int]:
    """Resolve ``(encoder_type, num_layers, num_heads)`` for loading a checkpoint.

    Prefers the architecture persisted in the checkpoint's
    ``compatibility_config.wm``. Falls back to inferring from the
    ``model_state_dict`` for older checkpoints that did not persist the
    state-encoder architecture: the transformer state encoder was added in
    v0.7 (RFC-0015 WS-C1), so a checkpoint whose weights carry
    ``encoder.encoder.layers.*`` is a transformer encoder regardless of what
    its compatibility block records. ``num_heads`` is not recoverable from
    weight shapes, so it defaults to the v0.7 head count when not persisted.
    """

    wm = wm_config if isinstance(wm_config, Mapping) else {}
    sd = state_dict if isinstance(state_dict, Mapping) else {}
    layer_indices = [
        int(key.split(".")[3])
        for key in sd
        if key.startswith("encoder.encoder.layers.")
        and key.split(".")[3].isdigit()
    ]
    has_transformer_weights = bool(layer_indices)

    encoder_type = wm.get("state_encoder_type")
    if encoder_type is None:
        encoder_type = "transformer" if has_transformer_weights else "pool"

    num_layers = wm.get("state_encoder_layers")
    if num_layers is None:
        num_layers = (max(layer_indices) + 1) if layer_indices else 4

    num_heads = wm.get("state_encoder_heads", 8)
    return str(encoder_type), int(num_layers), int(num_heads)


def resolve_ema_target_encoder_config(
    wm_config: Any,
    state_dict: Any,
) -> tuple[bool, float]:
    """Resolve EMA target settings for checkpoint loading.

    New checkpoints persist the opt-in flag in ``compatibility_config.wm``.
    The state-dict fallback keeps older consumers able to load a checkpoint
    whose compatibility payload is incomplete but whose weights include the
    target encoder module.
    """

    wm = wm_config if isinstance(wm_config, Mapping) else {}
    sd = state_dict if isinstance(state_dict, Mapping) else {}
    has_target_weights = any(str(key).startswith("target_encoder.") for key in sd)
    enabled = bool(wm.get("enable_ema_target_encoder") or has_target_weights)
    decay = wm.get("ema_target_decay", 0.99)
    try:
        decay_value = float(decay)
    except (TypeError, ValueError) as exc:
        raise ValueError("ema_target_decay must be numeric") from exc
    return enabled, decay_value


def resolve_output_value_head_config(
    wm_config: Any,
    state_dict: Any,
) -> bool:
    """Return whether a checkpoint needs the optional output-value head."""

    wm = wm_config if isinstance(wm_config, Mapping) else {}
    sd = state_dict if isinstance(state_dict, Mapping) else {}
    return bool(
        wm.get("enable_output_value_head")
        or any(str(key).startswith("output_value_head.") for key in sd)
    )


def _named_parameters(module: Any) -> dict[str, Any]:
    if not hasattr(module, "named_parameters"):
        return {}
    return dict(module.named_parameters())


def _named_buffers(module: Any) -> dict[str, Any]:
    if not hasattr(module, "named_buffers"):
        return {}
    return dict(module.named_buffers())


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
