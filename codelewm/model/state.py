"""State tokenizer-facing encoder modules for CodeLeWM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from codelewm.model.actions import ModelRuntimeUnavailableError
from codelewm.model.transition import LATENT_DIM, STATE_SEQUENCE_LENGTH

try:  # pragma: no cover - exercised when torch is installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - lightweight local env.
    torch = None
    nn = None


@dataclass(frozen=True)
class CodeStateEncoderConfig:
    """Configuration for the v0.1 packed CodeState encoder."""

    vocab_size: int = 32768
    max_length: int = STATE_SEQUENCE_LENGTH
    latent_dim: int = LATENT_DIM
    embed_dim: int = LATENT_DIM
    segment_vocab_size: int = 16
    dropout: float = 0.1
    # RFC-0015 WS-C1: "pool" is the v0.6 bag-of-embeddings mean-pool (default,
    # backward compatible); "transformer" adds a contextual encoder before
    # pooling to close the capacity asymmetry with the action encoder/predictor.
    encoder_type: str = "pool"
    num_layers: int = 4
    num_heads: int = 8

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.latent_dim <= 0 or self.embed_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        if self.segment_vocab_size <= 1:
            raise ValueError("segment_vocab_size must be greater than 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.encoder_type not in {"pool", "transformer"}:
            raise ValueError("encoder_type must be 'pool' or 'transformer'")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0 or self.embed_dim % self.num_heads != 0:
            raise ValueError("num_heads must be positive and divide embed_dim")


class CodeStateEncoder(nn.Module if nn is not None else object):
    """Embedding-pool encoder for packed Python CodeState tensors."""

    def __init__(self, config: CodeStateEncoderConfig = CodeStateEncoderConfig()) -> None:
        if nn is None or torch is None:
            raise ModelRuntimeUnavailableError("CodeStateEncoder requires torch")
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.segment_embedding = nn.Embedding(config.segment_vocab_size, config.embed_dim, padding_idx=0)
        self.changed_embedding = nn.Embedding(2, config.embed_dim)
        self.position = nn.Parameter(torch.zeros(1, config.max_length, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.encoder = None
        if config.encoder_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.embed_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # enable_nested_tensor=False avoids the nested-tensor attention fast
            # path, which is unimplemented on Apple MPS and only warns here.
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=config.num_layers, enable_nested_tensor=False
            )
        self.proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim * 2),
            nn.Linear(config.embed_dim * 2, config.latent_dim),
        )
        self._reset_parameters()

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        segment_ids: Any,
        changed_hunk_mask: Any | None = None,
    ) -> Any:
        if input_ids.shape[-1] != self.config.max_length:
            raise ValueError(
                f"state input_ids must have sequence length {self.config.max_length}; "
                f"got {input_ids.shape[-1]}"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("state attention_mask must match input_ids shape")
        if segment_ids.shape != input_ids.shape:
            raise ValueError("state segment_ids must match input_ids shape")
        if changed_hunk_mask is not None and changed_hunk_mask.shape != input_ids.shape:
            raise ValueError("state changed_hunk_mask must match input_ids shape")

        segments = segment_ids.clamp(min=0, max=self.config.segment_vocab_size - 1)
        changed = (
            torch.zeros_like(input_ids, dtype=torch.long)
            if changed_hunk_mask is None
            else changed_hunk_mask.to(dtype=torch.long).clamp(min=0, max=1)
        )
        hidden = (
            self.token_embedding(input_ids)
            + self.segment_embedding(segments)
            + self.changed_embedding(changed)
            + self.position[:, : input_ids.shape[-1], :]
        )
        hidden = self.norm(self.dropout(hidden))
        if self.encoder is not None:
            # src_key_padding_mask marks positions to ignore (True = pad).
            pad_mask = attention_mask == 0
            hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position, mean=0.0, std=1.0 / math.sqrt(self.config.embed_dim))
