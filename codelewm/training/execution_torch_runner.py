"""Torch training runner for the execution-substrate pack.

This is the bridge between :mod:`codelewm.training.execution_pack_loader`
(NumPy batches over ``pack.jsonl``) and the existing JEPA torch model
(:class:`codelewm.model.TorchCodeTransitionModel`). The architecture,
objective registry, and encoder weights are reused verbatim — only the
data layer is new.

Mapping from execution substrate to the existing model's contract:

- ``code_tokens``       → ``CodeStateBatch.input_ids``                (z_before)
- ``input_tokens``      → ``ActionBatch.input_ids`` (action_view='text')
- ``output_tokens``     → ``CodeStateBatch.input_ids`` padded to ``STATE_SEQUENCE_LENGTH`` (z_target)

``output_tokens`` are padded up to the state sequence length so the same
state encoder produces ``z_output`` with shared weights. That keeps the
"same architecture, different substrate" story honest end-to-end.

The runner is CPU/MPS-friendly so the smoke runs on a laptop in a few
minutes. The same code path is what the v0.6 HF Jobs run (#265) will
invoke on A10G; this module supplies the data and forward; the
existing torch executor wraps the loop, optimizer, and checkpointing.
For the smoke we keep the loop minimal and self-contained.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.data.execution_pack import (
    EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
)
from codelewm.model.transition import (
    ActionBatch,
    CodeStateBatch,
    LATENT_DIM,
    STATE_SEQUENCE_LENGTH,
    TEXT_ACTION_SEQUENCE_LENGTH,
)
from codelewm.model.objective import (
    ObjectiveConfig,
    compute_transition_objective,
)
from codelewm.model.torch_transition import (
    TorchCodeTransitionModel,
    TorchCodeTransitionModelConfig,
    build_torch_transition_model,
)

from .execution_pack_loader import (
    EXECUTION_PACK_BATCH_SCHEMA_VERSION,
    ExecutionPackBatch,
    ExecutionPackLoaderConfig,
    iter_batches,
    iter_records,
)


EXECUTION_TRAIN_REPORT_SCHEMA_VERSION = "codelewm.execution_train_report.v1"
EXECUTION_TRAIN_STEP_SCHEMA_VERSION = "codelewm.execution_train_step.v1"
_DEFAULT_VOCAB_SIZE = 32768


class ExecutionTorchRunnerError(RuntimeError):
    """Raised when the execution torch runner cannot complete cleanly."""


@dataclass(frozen=True)
class ExecutionTorchTrainConfig:
    """Configuration for the smoke / v0.6 execution torch training run.

    Defaults are tuned for the laptop smoke; the v0.6 A10G config swaps
    the device, batch size, and step budget.
    """

    pack_jsonl: Path
    output_dir: Path
    code_sequence_length: int = STATE_SEQUENCE_LENGTH
    action_sequence_length: int = TEXT_ACTION_SEQUENCE_LENGTH
    output_sequence_length: int = 256
    batch_size: int = 4
    max_steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 20
    seed: int = 42
    device: str = "cpu"  # "cpu" | "mps" | "cuda"
    vocab_size: int = _DEFAULT_VOCAB_SIZE
    pad_token_id: int = 0
    sigreg_weight: float = 0.09
    enable_action_swap_contrastive: bool = True
    action_swap_contrastive_weight: float = 0.1
    action_swap_contrastive_margin: float = 0.05
    log_every: int = 10
    eval_every: int = 50
    enable_inverse_action_reconstruction: bool = False
    inverse_action_reconstruction_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.code_sequence_length != STATE_SEQUENCE_LENGTH:
            raise ExecutionTorchRunnerError(
                f"code_sequence_length must be {STATE_SEQUENCE_LENGTH} (model contract)"
            )
        if self.action_sequence_length != TEXT_ACTION_SEQUENCE_LENGTH:
            raise ExecutionTorchRunnerError(
                f"action_sequence_length must be {TEXT_ACTION_SEQUENCE_LENGTH} (model contract)"
            )
        if self.output_sequence_length > STATE_SEQUENCE_LENGTH:
            raise ExecutionTorchRunnerError(
                "output_sequence_length must be <= state sequence length so the "
                "shared encoder can consume padded output tokens"
            )
        if self.batch_size < 1:
            raise ExecutionTorchRunnerError("batch_size must be positive")
        if self.max_steps < 1:
            raise ExecutionTorchRunnerError("max_steps must be positive")


@dataclass(frozen=True)
class ExecutionTorchStep:
    """One training-step metrics row."""

    step: int
    loss_total: float
    loss_prediction_mse: float
    loss_sigreg: float
    loss_action_swap_contrastive: float | None
    no_action_mse: float
    margin_no_action_minus_pred: float
    wall_time_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_TRAIN_STEP_SCHEMA_VERSION,
            "step": self.step,
            "loss_total": self.loss_total,
            "loss_prediction_mse": self.loss_prediction_mse,
            "loss_sigreg": self.loss_sigreg,
            "loss_action_swap_contrastive": self.loss_action_swap_contrastive,
            "no_action_mse": self.no_action_mse,
            "margin_no_action_minus_pred": self.margin_no_action_minus_pred,
            "wall_time_s": self.wall_time_s,
        }


@dataclass(frozen=True)
class ExecutionTorchReport:
    """Aggregate report. JSON-serializable."""

    schema_version: str
    config: dict[str, Any]
    pack_record_count: int
    steps: tuple[ExecutionTorchStep, ...]
    initial_metrics: dict[str, float]
    final_metrics: dict[str, float]
    deltas: dict[str, float]
    z_diagnostics: dict[str, Any]
    device: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": dict(self.config),
            "pack_record_count": self.pack_record_count,
            "steps": [s.as_dict() for s in self.steps],
            "initial_metrics": dict(self.initial_metrics),
            "final_metrics": dict(self.final_metrics),
            "deltas": dict(self.deltas),
            "z_diagnostics": dict(self.z_diagnostics),
            "device": self.device,
            "seed": self.seed,
        }


def train_execution_smoke(config: ExecutionTorchTrainConfig) -> ExecutionTorchReport:
    """Train the existing JEPA model on the execution pack for ``max_steps``.

    Returns a structured report with per-step metrics and aggregate
    diagnostics. The smoke verifies the data path and the loss
    behavior; it is not the full v0.6 training run.
    """

    # Lazy import torch so the module is importable without it for tests
    # that only need the configs.
    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415, F401
        from torch.optim import AdamW  # noqa: PLC0415
    except ImportError as exc:
        raise ExecutionTorchRunnerError(
            "torch is required to run the execution smoke; install codelewm[train]"
        ) from exc

    if not config.pack_jsonl.is_file():
        raise ExecutionTorchRunnerError(
            f"pack.jsonl not found at {config.pack_jsonl}"
        )

    loader_config = ExecutionPackLoaderConfig(
        pack_jsonl=config.pack_jsonl,
        code_sequence_length=config.code_sequence_length,
        action_sequence_length=config.action_sequence_length,
        output_sequence_length=config.output_sequence_length,
        batch_size=config.batch_size,
        shuffle=True,
        shuffle_seed=config.seed,
    )

    pack_record_count = sum(1 for _ in iter_records(loader_config))
    if pack_record_count == 0:
        raise ExecutionTorchRunnerError(
            "pack.jsonl has no records; run codelewm dataset execution-pack first"
        )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(config.device)

    model = build_torch_transition_model(
        TorchCodeTransitionModelConfig(
            action_view="text",
            latent_dim=LATENT_DIM,
            state_sequence_length=STATE_SEQUENCE_LENGTH,
            action_sequence_length=TEXT_ACTION_SEQUENCE_LENGTH,
            vocab_size=config.vocab_size,
        )
    ).to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    objective_config = ObjectiveConfig(
        sigreg_weight=config.sigreg_weight,
        enable_action_swap_contrastive=config.enable_action_swap_contrastive,
        action_swap_contrastive_weight=(
            config.action_swap_contrastive_weight
            if config.enable_action_swap_contrastive
            else 0.0
        ),
        action_swap_contrastive_margin=(
            config.action_swap_contrastive_margin
            if config.enable_action_swap_contrastive
            else 0.0
        ),
        enable_inverse_action_reconstruction=config.enable_inverse_action_reconstruction,
        inverse_action_reconstruction_weight=(
            config.inverse_action_reconstruction_weight
            if config.enable_inverse_action_reconstruction
            else 0.0
        ),
    )

    steps: list[ExecutionTorchStep] = []
    initial: dict[str, float] | None = None

    batch_iter = _infinite_batch_iter(loader_config)

    for step in range(1, config.max_steps + 1):
        lr_now = _warmup_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        batch = next(batch_iter)
        terms, no_action_mse_val, margin_val, swap_val = _train_one_step(
            model=model,
            batch=batch,
            objective_config=objective_config,
            pad_token_id=config.pad_token_id,
            output_sequence_length=config.output_sequence_length,
            vocab_size=config.vocab_size,
            device=device,
            torch_=torch,
        )

        optimizer.zero_grad(set_to_none=True)
        terms.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scalar = terms.scalars()
        record = ExecutionTorchStep(
            step=step,
            loss_total=scalar["loss/total"],
            loss_prediction_mse=scalar["loss/prediction_mse"],
            loss_sigreg=scalar["loss/sigreg"],
            loss_action_swap_contrastive=swap_val,
            no_action_mse=no_action_mse_val,
            margin_no_action_minus_pred=margin_val,
            wall_time_s=time.perf_counter(),
        )
        steps.append(record)
        if initial is None:
            initial = {
                "loss_total": record.loss_total,
                "loss_prediction_mse": record.loss_prediction_mse,
                "loss_sigreg": record.loss_sigreg,
                "no_action_mse": record.no_action_mse,
                "margin_no_action_minus_pred": record.margin_no_action_minus_pred,
            }

    if not steps:  # pragma: no cover - max_steps>=1 enforced above
        raise ExecutionTorchRunnerError("no training steps executed")

    # Smoothed final by averaging the last 10% of steps (or last 5).
    window = max(5, len(steps) // 10)
    tail = steps[-window:]
    final = {
        "loss_total": _mean(s.loss_total for s in tail),
        "loss_prediction_mse": _mean(s.loss_prediction_mse for s in tail),
        "loss_sigreg": _mean(s.loss_sigreg for s in tail),
        "no_action_mse": _mean(s.no_action_mse for s in tail),
        "margin_no_action_minus_pred": _mean(
            s.margin_no_action_minus_pred for s in tail
        ),
    }
    initial = initial or final
    deltas = {
        k: final[k] - initial[k] for k in initial
    }

    # Latent diagnostics on a deterministic held-out batch.
    z_diagnostics = _latent_diagnostics(
        model=model,
        loader_config=loader_config,
        pad_token_id=config.pad_token_id,
        output_sequence_length=config.output_sequence_length,
        vocab_size=config.vocab_size,
        device=device,
        torch_=torch,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report = ExecutionTorchReport(
        schema_version=EXECUTION_TRAIN_REPORT_SCHEMA_VERSION,
        config={
            "pack_jsonl": str(config.pack_jsonl),
            "output_dir": str(config.output_dir),
            "code_sequence_length": config.code_sequence_length,
            "action_sequence_length": config.action_sequence_length,
            "output_sequence_length": config.output_sequence_length,
            "batch_size": config.batch_size,
            "max_steps": config.max_steps,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "seed": config.seed,
            "device": config.device,
            "vocab_size": config.vocab_size,
            "sigreg_weight": config.sigreg_weight,
            "enable_action_swap_contrastive": config.enable_action_swap_contrastive,
            "action_swap_contrastive_weight": config.action_swap_contrastive_weight,
            "action_swap_contrastive_margin": config.action_swap_contrastive_margin,
            "pack_batch_schema_version": EXECUTION_PACK_BATCH_SCHEMA_VERSION,
            "pack_manifest_schema_version": EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
        },
        pack_record_count=pack_record_count,
        steps=tuple(steps),
        initial_metrics=initial,
        final_metrics=final,
        deltas=deltas,
        z_diagnostics=z_diagnostics,
        device=str(device),
        seed=config.seed,
    )

    report_path = config.output_dir / "execution_train_report.json"
    report_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return report


def _infinite_batch_iter(
    loader_config: ExecutionPackLoaderConfig,
) -> Iterator[ExecutionPackBatch]:
    """Yield batches forever by restarting the loader at each epoch boundary."""

    epoch = 0
    while True:
        for batch in iter_batches(loader_config):
            yield batch
        epoch += 1


def _warmup_lr(step: int, config: ExecutionTorchTrainConfig) -> float:
    if step <= config.warmup_steps and config.warmup_steps > 0:
        return config.lr * step / config.warmup_steps
    return config.lr


def _train_one_step(
    *,
    model: TorchCodeTransitionModel,
    batch: ExecutionPackBatch,
    objective_config: ObjectiveConfig,
    pad_token_id: int,
    output_sequence_length: int,
    vocab_size: int,
    device: Any,
    torch_: Any,
) -> tuple[Any, float, float, float | None]:
    """Run one forward+loss pass; return objective terms and no-action margin."""

    code_state = _to_code_state_batch(
        batch.code_tokens,
        batch.code_attention_mask,
        vocab_size=vocab_size,
        device=device,
        torch_=torch_,
    )
    output_state = _output_state_batch(
        batch.output_tokens,
        batch.output_attention_mask,
        pad_token_id=pad_token_id,
        output_sequence_length=output_sequence_length,
        vocab_size=vocab_size,
        device=device,
        torch_=torch_,
    )
    action_batch = _to_action_batch(
        batch.input_tokens,
        batch.input_attention_mask,
        vocab_size=vocab_size,
        device=device,
        torch_=torch_,
    )
    action_batch_swapped = _to_action_batch(
        _roll_axis0(batch.input_tokens),
        _roll_axis0(batch.input_attention_mask),
        vocab_size=vocab_size,
        device=device,
        torch_=torch_,
    )

    z_before = model.encode_state(code_state)
    action_emb = model.encode_action(action_batch)
    action_emb_swapped = model.encode_action(action_batch_swapped)
    z_after = model.encode_state(output_state)
    z_pred_after = model.predict_after(z_before, action_emb)
    z_pred_after_swapped = model.predict_after(z_before, action_emb_swapped)

    # The inverse-action head reconstructs the action embedding from
    # (z_before, z_after); the objective compares it against the true
    # ``action_emb``. The head only exists when the model was built
    # with ``enable_inverse_action_head=True`` (see
    # :func:`codelewm.training.execution_runner.train_execution_run`).
    action_reconstruction = None
    if objective_config.enable_inverse_action_reconstruction:
        action_reconstruction = model.reconstruct_action(z_before, z_after)

    terms = compute_transition_objective(
        z_before,
        z_after,
        z_pred_after,
        config=objective_config,
        z_pred_after_swapped=(
            z_pred_after_swapped
            if objective_config.enable_action_swap_contrastive
            else None
        ),
        action_emb=action_emb if objective_config.enable_inverse_action_reconstruction else None,
        action_reconstruction=action_reconstruction,
    )

    # No-action baseline: how well would the identity "z_pred = z_before"
    # do? If our predicted output beats this on MSE, the action is actually
    # contributing.
    with torch_.no_grad():
        no_action_mse = ((z_before - z_after) ** 2).mean().item()
        pred_mse = ((z_pred_after - z_after) ** 2).mean().item()
        margin = no_action_mse - pred_mse

    swap_value: float | None = None
    if terms.action_swap_contrastive is not None:
        swap_value = float(terms.action_swap_contrastive.detach().item())

    return terms, float(no_action_mse), float(margin), swap_value


def _map_to_vocab(values: np.ndarray, vocab_size: int) -> np.ndarray:
    """Fold 31-bit stable-hash token IDs into ``[0, vocab_size)``.

    Mirrors the CommitPackFT pack reader so PAD (id 0) stays 0 and every
    other id lands in ``[1, vocab_size - 1]``.
    """

    arr = values.astype(np.int64, copy=False)
    return np.where(
        arr > 0,
        ((arr - 1) % (vocab_size - 1)) + 1,
        0,
    ).astype(np.int64)


def _to_code_state_batch(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    vocab_size: int,
    device: Any,
    torch_: Any,
) -> CodeStateBatch:
    folded = _map_to_vocab(tokens, vocab_size)
    input_ids = torch_.from_numpy(folded).to(device)
    attention_mask = torch_.from_numpy(mask).to(device)
    segment_ids = torch_.zeros_like(input_ids)
    return CodeStateBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        changed_hunk_mask=None,
    )


def _output_state_batch(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    pad_token_id: int,
    output_sequence_length: int,
    vocab_size: int,
    device: Any,
    torch_: Any,
) -> CodeStateBatch:
    """Pad output tokens up to STATE_SEQUENCE_LENGTH for the shared encoder."""

    folded = _map_to_vocab(tokens, vocab_size)
    batch_size = folded.shape[0]
    padded_tokens = np.full(
        (batch_size, STATE_SEQUENCE_LENGTH), pad_token_id, dtype=np.int64
    )
    padded_mask = np.zeros((batch_size, STATE_SEQUENCE_LENGTH), dtype=bool)
    n = min(output_sequence_length, STATE_SEQUENCE_LENGTH)
    padded_tokens[:, :n] = folded[:, :n]
    padded_mask[:, :n] = mask[:, :n]
    input_ids = torch_.from_numpy(padded_tokens).to(device)
    attention_mask = torch_.from_numpy(padded_mask).to(device)
    segment_ids = torch_.zeros_like(input_ids)
    return CodeStateBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        changed_hunk_mask=None,
    )


def _to_action_batch(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    vocab_size: int,
    device: Any,
    torch_: Any,
) -> ActionBatch:
    folded = _map_to_vocab(tokens, vocab_size)
    input_ids = torch_.from_numpy(folded).to(device)
    attention_mask = torch_.from_numpy(mask).to(device)
    return ActionBatch(
        input_ids=input_ids, attention_mask=attention_mask, action_view="text"
    )


def _roll_axis0(arr: np.ndarray) -> np.ndarray:
    """Shift batch by one along axis 0 to build the action-swap negative.

    If batch size is 1, returns the array unchanged; the contrastive
    loss will then degenerate (zero margin) for that step but won't
    crash. Real packs use batch_size >= 4.
    """

    if arr.shape[0] <= 1:
        return arr
    return np.concatenate([arr[1:], arr[:1]], axis=0)


def _mean(values: Iterator[float] | list[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _latent_diagnostics(
    *,
    model: TorchCodeTransitionModel,
    loader_config: ExecutionPackLoaderConfig,
    pad_token_id: int,
    output_sequence_length: int,
    vocab_size: int,
    device: Any,
    torch_: Any,
) -> dict[str, Any]:
    """Compute effective rank and norm stats on a deterministic eval pass."""

    model.eval()
    all_z_pred: list[np.ndarray] = []
    all_z_target: list[np.ndarray] = []
    with torch_.no_grad():
        for batch in iter_batches(loader_config):
            code_state = _to_code_state_batch(
                batch.code_tokens,
                batch.code_attention_mask,
                vocab_size=vocab_size,
                device=device,
                torch_=torch_,
            )
            output_state = _output_state_batch(
                batch.output_tokens,
                batch.output_attention_mask,
                pad_token_id=pad_token_id,
                output_sequence_length=output_sequence_length,
                vocab_size=vocab_size,
                device=device,
                torch_=torch_,
            )
            action_batch = _to_action_batch(
                batch.input_tokens,
                batch.input_attention_mask,
                vocab_size=vocab_size,
                device=device,
                torch_=torch_,
            )
            z_before = model.encode_state(code_state)
            action_emb = model.encode_action(action_batch)
            z_pred = model.predict_after(z_before, action_emb)
            z_target = model.encode_state(output_state)
            all_z_pred.append(z_pred.detach().cpu().numpy())
            all_z_target.append(z_target.detach().cpu().numpy())
    model.train()

    if not all_z_pred:  # pragma: no cover
        return {"sample_count": 0}

    z_pred = np.concatenate(all_z_pred, axis=0)
    z_target = np.concatenate(all_z_target, axis=0)
    eff_rank, ratio = _effective_rank(z_pred)
    target_eff_rank, target_ratio = _effective_rank(z_target)
    pred_norm = float(np.linalg.norm(z_pred, axis=-1).mean())
    target_norm = float(np.linalg.norm(z_target, axis=-1).mean())
    cosine = _mean_pairwise_cosine(z_pred)
    return {
        "sample_count": int(z_pred.shape[0]),
        "z_pred_effective_rank": eff_rank,
        "z_pred_effective_rank_ratio": ratio,
        "z_target_effective_rank": target_eff_rank,
        "z_target_effective_rank_ratio": target_ratio,
        "z_pred_mean_norm": pred_norm,
        "z_target_mean_norm": target_norm,
        "z_pred_mean_pairwise_cosine": cosine,
        "latent_dim": int(z_pred.shape[-1]),
    }


def _effective_rank(emb: np.ndarray) -> tuple[float, float]:
    if emb.size == 0:
        return 0.0, 0.0
    centered = emb - emb.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return 0.0, 0.0
    try:
        s = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0, 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = float(math.exp(entropy))
    return eff_rank, eff_rank / float(emb.shape[-1])


def _mean_pairwise_cosine(emb: np.ndarray) -> float:
    if emb.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12
    unit = emb / norms
    # np.errstate guards the spurious divide/overflow/invalid RuntimeWarnings
    # that numpy's Apple Accelerate matmul backend emits even on small, finite
    # matrices (same quirk handled in rerank_calibrator). This is a
    # diagnostics-only metric, so it never reaches the training gradient.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = unit @ unit.T
        n = sims.shape[0]
        iu = np.triu_indices(n, k=1)
        mean_cosine = float(sims[iu].mean())
    if not math.isfinite(mean_cosine):
        return 0.0
    return mean_cosine
