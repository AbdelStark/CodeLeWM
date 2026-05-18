"""Action tokenizer and encoder modules."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from codelewm.model.transition import LATENT_DIM, TEXT_ACTION_SEQUENCE_LENGTH, ActionBatch

try:  # pragma: no cover - exercised when torch is installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - lightweight local env.
    torch = None
    nn = None


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class ModelRuntimeUnavailableError(RuntimeError):
    """Raised when a concrete encoder is requested without the ML runtime."""


@dataclass(frozen=True)
class TextActionEncoderConfig:
    """Configuration for the v0.1 text action encoder."""

    vocab_size: int = 32768
    max_length: int = TEXT_ACTION_SEQUENCE_LENGTH
    latent_dim: int = LATENT_DIM
    embed_dim: int = LATENT_DIM
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.latent_dim <= 0 or self.embed_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")


class TextActionTokenizer:
    """Deterministic lightweight tokenizer for text action fixtures."""

    def __init__(self, *, vocab_size: int = 32768, max_length: int = TEXT_ACTION_SEQUENCE_LENGTH) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

    def encode(self, text: str) -> ActionBatch:
        tokens = _TOKEN_PATTERN.findall(text.strip())
        if not tokens:
            raise ValueError("text action must not be empty")
        token_ids = [self.token_id(token) for token in tokens[: self.max_length]]
        attention_mask = [True] * len(token_ids)
        padding = self.max_length - len(token_ids)
        if padding > 0:
            token_ids.extend([0] * padding)
            attention_mask.extend([False] * padding)
        return ActionBatch(
            input_ids=np.asarray([token_ids], dtype=np.int64),
            attention_mask=np.asarray([attention_mask], dtype=bool),
            action_view="text",
        )

    def batch_encode(self, texts: list[str] | tuple[str, ...]) -> ActionBatch:
        encoded = [self.encode(text) for text in texts]
        return ActionBatch(
            input_ids=np.concatenate([batch.input_ids for batch in encoded], axis=0),
            attention_mask=np.concatenate([batch.attention_mask for batch in encoded], axis=0),
            action_view="text",
        )

    def token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.casefold().encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big") % (self.vocab_size - 1) + 1


class TextActionEncoder(nn.Module if nn is not None else object):
    """Transformer encoder for headline text actions."""

    def __init__(self, config: TextActionEncoderConfig = TextActionEncoderConfig()) -> None:
        if nn is None or torch is None:
            raise ModelRuntimeUnavailableError("TextActionEncoder requires torch")
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, config.max_length, config.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.latent_dim)
        self._reset_parameters()

    def forward(self, input_ids: Any, attention_mask: Any) -> Any:
        if input_ids.shape[-1] != self.config.max_length:
            raise ValueError(
                f"text action input_ids must have sequence length {self.config.max_length}; "
                f"got {input_ids.shape[-1]}"
            )
        hidden = self.embedding(input_ids) + self.position[:, : input_ids.shape[-1], :]
        key_padding_mask = ~attention_mask.bool()
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        mask = attention_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position, mean=0.0, std=1.0 / math.sqrt(self.config.embed_dim))
