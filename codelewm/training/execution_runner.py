"""Production training runner for the v0.6 execution-substrate run.

This is the end-to-end entry point the v0.6 HF Jobs invocation calls:

    codelewm train --config <execution-yaml> --seed <int>

The runner is the bridge between three things:

1. The :class:`codelewm.training.execution_train_config.ExecutionTrainConfig`
   contract (operator-facing YAML).
2. The torch training step :func:`codelewm.training.execution_torch_runner._train_one_step`
   that the local smoke runner (#288) proved works on this substrate.
3. The artifact / manifest contract the rest of the codebase (eval
   harness, publish scripts, manifest verify) already expects — same
   shape as the legacy HDF5 runner (:mod:`codelewm.training.runner`).

Pack resolution is local-first: a ``CODELEWM_EXECUTION_PACK_LOCAL_DIR``
env var or an explicit ``pack_local_dir`` kwarg short-circuits the
Hugging Face download. The container entrypoint inside the v0.6 runtime
image populates the local dir before invoking the CLI, so the runner
never has to authenticate with the Hub itself. When the env var is
absent the runner falls back to
:func:`huggingface_hub.snapshot_download` so a developer can also run
the full pipeline directly with `HF_TOKEN` exported.

Architecture note: no EMA target encoder. The objective config is built
from the YAML's ``objective`` block verbatim — SIGReg alone is doing
the anti-collapse work, matching the substrate-pivot's headline claim
(#288 / RFC-0014). See ``docs/operations/V0_6_EXECUTION_RUN_RUNBOOK.md``.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.data import OptionalDependencyError, SourceUnavailableError
from codelewm.data.execution_pack import (
    EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
    EXECUTION_PACK_RECORD_SCHEMA_VERSION,
)
from codelewm.model import (
    CHECKPOINT_SCHEMA_VERSION,
    LATENT_DIM,
    STATE_SEQUENCE_LENGTH,
    TEXT_ACTION_SEQUENCE_LENGTH,
    ObjectiveConfig,
    TorchCodeTransitionModelConfig,
    build_checkpoint_metadata,
    build_torch_transition_model,
    compute_config_hash,
    write_checkpoint_manifest,
)
from codelewm.observability import (
    build_artifact_manifest,
    build_manifest_file,
    write_artifact_manifest,
)

from .execution_pack_loader import (
    EXECUTION_PACK_BATCH_SCHEMA_VERSION,
    ExecutionPackLoaderConfig,
    iter_batches,
    iter_records,
)
from .execution_torch_runner import (
    EXECUTION_TRAIN_REPORT_SCHEMA_VERSION,
    EXECUTION_TRAIN_STEP_SCHEMA_VERSION,
    ExecutionTorchRunnerError,
    _infinite_batch_iter,
    _latent_diagnostics,
    _mean,
    _train_one_step,
)
from .execution_train_config import (
    ExecutionTrainConfig,
    ExecutionTrainConfigError,
)
from .runner import (
    TRAINING_METRICS_SCHEMA_VERSION,
    TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
    TrainingRunError,
    TrainingRunManifest,
)


EXECUTION_TRAIN_RUN_REPORT_SCHEMA_VERSION = "codelewm.execution_train_run_report.v1"
EXECUTION_TRAIN_COLLAPSE_DIAGNOSTICS_SCHEMA_VERSION = (
    "codelewm.execution_train_collapse_diagnostics.v1"
)

_PACK_LOCAL_DIR_ENV = "CODELEWM_EXECUTION_PACK_LOCAL_DIR"
_DEFAULT_VOCAB_SIZE = 32768


@dataclass(frozen=True)
class ExecutionTrainRunResult:
    """Surface returned by :func:`train_execution_run`.

    The training-run manifest is the public contract; the other paths
    are convenience so callers don't have to re-derive them from the
    manifest entries.
    """

    training_manifest: TrainingRunManifest
    training_manifest_path: Path
    artifact_manifest_path: Path
    checkpoint_paths: tuple[Path, ...]
    metrics_path: Path
    report_path: Path
    pack_dir: Path
    collapse_diagnostics_path: Path | None


def train_execution_run(
    config: ExecutionTrainConfig,
    *,
    seed: int,
    output_dir: Path | str,
    root: Path | str = ".",
    command: Sequence[str] = ("codelewm", "train"),
    source_git_sha: str | None = None,
    created_at: str | None = None,
    overwrite: bool = False,
    pack_local_dir: Path | str | None = None,
    pack_jsonl_override: Path | str | None = None,
    device: str | None = None,
    tensorboard: bool | None = None,
    tensorboard_dir: Path | str | None = None,
) -> ExecutionTrainRunResult:
    """Run the v0.6 execution-substrate training to completion.

    Writes the manifest-backed artifact set (artifact manifest, training
    run manifest, checkpoint + manifest sidecar, metrics JSONL, training
    report, collapse-diagnostics JSONL) under ``output_dir``.

    Pack resolution order:

    1. ``pack_jsonl_override`` (kwarg) — explicit JSONL path; used by
       tests so they don't need a manifest sidecar.
    2. ``pack_local_dir`` (kwarg or ``CODELEWM_EXECUTION_PACK_LOCAL_DIR``
       env) — directory holding the pack JSONL + manifest. The container
       entrypoint populates this before invoking the CLI.
    3. Hugging Face snapshot download of ``config.data.pack_repo_id`` at
       ``config.data.pack_revision`` into ``output_dir / "_pack"``.

    Manifest contract notes:

    - The artifact manifest's ``parent_artifacts`` includes the pack's
      ``artifact_id`` so the lineage chain is preserved.
    - The training-run manifest's ``parent_artifacts`` matches the
      artifact manifest's.
    - When the pack is resolved via HF, the manifest still records the
      local dir hash; the operator can verify against the upstream
      ``manifest.json`` checksum recorded by the publisher.
    """

    if not isinstance(config, ExecutionTrainConfig):
        raise ExecutionTrainConfigError(
            "config must be an ExecutionTrainConfig instance"
        )
    if seed < 0:
        raise TrainingRunError("seed must be non-negative")

    root_path = Path(root).resolve()
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (root_path / output_path).resolve()

    if output_path.exists() and not overwrite:
        if any(output_path.iterdir()):
            raise TrainingRunError(
                f"output already exists; pass overwrite=True to replace: {output_path}"
            )
    output_path.mkdir(parents=True, exist_ok=True)

    pack_dir, pack_jsonl_path, parent_pack_artifact_id = _resolve_pack(
        config=config,
        output_dir=output_path,
        pack_local_dir=pack_local_dir,
        pack_jsonl_override=pack_jsonl_override,
    )

    # The pack JSONL file is required; the manifest sidecar is optional
    # for fixtures used in tests but recorded when present.
    if not pack_jsonl_path.is_file():
        raise SourceUnavailableError(
            f"execution pack JSONL not found at {pack_jsonl_path}"
        )

    # Lazy torch import so the legacy runner and config tests stay
    # torch-free.
    try:
        import torch  # noqa: PLC0415
        from torch.optim import AdamW  # noqa: PLC0415
    except ImportError as exc:
        raise OptionalDependencyError(
            "torch is required for execution-substrate training; install codelewm[train]"
        ) from exc

    selected_device = _resolve_device(
        requested=device or config.trainer.accelerator, torch_=torch
    )
    tensorboard_flag = (
        config.trainer.tensorboard_enabled if tensorboard is None else bool(tensorboard)
    )

    loader_config = ExecutionPackLoaderConfig(
        pack_jsonl=pack_jsonl_path,
        code_sequence_length=config.loader.code_sequence_length,
        action_sequence_length=config.loader.action_sequence_length,
        output_sequence_length=config.loader.output_sequence_length,
        batch_size=config.loader.batch_size,
        shuffle=config.loader.shuffle,
        shuffle_seed=seed,
    )

    pack_record_count = sum(1 for _ in iter_records(loader_config))
    if pack_record_count == 0:
        raise TrainingRunError(
            f"execution pack at {pack_jsonl_path} contains zero records"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_torch_transition_model(
        TorchCodeTransitionModelConfig(
            action_view="text",
            latent_dim=config.wm.embed_dim,
            state_sequence_length=STATE_SEQUENCE_LENGTH,
            action_sequence_length=TEXT_ACTION_SEQUENCE_LENGTH,
            vocab_size=_DEFAULT_VOCAB_SIZE,
            dropout=0.0,
            action_fusion="conditional_transformer",
            enable_inverse_action_head=(
                config.objective.inverse_action_reconstruction_weight > 0.0
            ),
            state_encoder_type=config.wm.state_encoder_type,
            state_encoder_layers=config.wm.state_encoder_layers,
            state_encoder_heads=config.wm.state_encoder_heads,
        )
    ).to(selected_device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=tuple(config.optimizer.betas),
    )

    objective_config = ObjectiveConfig(
        sigreg_weight=config.objective.sigreg_weight,
        prediction_mse_weight=config.objective.prediction_mse_weight,
        enable_retrieval_loss=config.objective.retrieval_weight > 0.0,
        retrieval_weight=config.objective.retrieval_weight,
        enable_action_swap_contrastive=(
            config.objective.action_swap_contrastive_weight > 0.0
        ),
        action_swap_contrastive_weight=config.objective.action_swap_contrastive_weight,
        action_swap_contrastive_margin=0.05,
        enable_inverse_action_reconstruction=(
            config.objective.inverse_action_reconstruction_weight > 0.0
        ),
        inverse_action_reconstruction_weight=(
            config.objective.inverse_action_reconstruction_weight
        ),
        sigreg_seed=seed,
    )

    # Output layout under run_dir.
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "config.json"
    metrics_path = output_path / "metrics.jsonl"
    collapse_diag_path = (
        reports_dir / "collapse_diagnostics.jsonl"
        if config.trainer.collapse_diagnostics_every_n_steps > 0
        else None
    )

    # Persist the resolved config + per-seed materialization so the
    # artifact directory carries everything a verifier needs.
    config_payload = config.to_dict()
    materialized_config = {
        "execution_train_config": config_payload,
        "seed": seed,
        "device": str(selected_device),
        "pack_jsonl_path": pack_jsonl_path.as_posix(),
        "pack_record_count": pack_record_count,
        "pack_batch_schema_version": EXECUTION_PACK_BATCH_SCHEMA_VERSION,
        "pack_manifest_schema_version": EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
    }
    _write_json(materialized_config, config_path)
    if metrics_path.exists():
        metrics_path.unlink()
    metrics_path.touch()
    if collapse_diag_path is not None and collapse_diag_path.exists():
        collapse_diag_path.unlink()

    batch_iter = _infinite_batch_iter(loader_config)
    accumulation_steps = max(1, config.loader.gradient_accumulation_steps)
    target_steps = config.trainer.max_steps
    optimizer.zero_grad(set_to_none=True)

    started_at = time.perf_counter()
    initial_metrics: dict[str, float] | None = None
    last_metrics: dict[str, float] = {}
    last_checkpoint_metrics: dict[str, float] = {}
    best_checkpoint_metric: float | None = None
    last_collapse_report: dict[str, Any] | None = None

    micro_step = 0
    optimizer_step = 0
    tail_metrics: list[dict[str, float]] = []
    tail_window = max(5, target_steps // 10)

    while optimizer_step < target_steps:
        batch = next(batch_iter)
        terms, no_action_mse_val, margin_val, swap_val = _train_one_step(
            model=model,
            batch=batch,
            objective_config=objective_config,
            pad_token_id=0,
            output_sequence_length=config.loader.output_sequence_length,
            vocab_size=_DEFAULT_VOCAB_SIZE,
            device=selected_device,
            torch_=torch,
        )

        # Scale the loss for gradient accumulation so the effective
        # batch is gradient_accumulation_steps * micro-batch.
        loss = terms.total / accumulation_steps
        loss.backward()
        micro_step += 1
        if micro_step % accumulation_steps != 0:
            continue

        if config.trainer.gradient_clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.trainer.gradient_clip_val
            )
        optimizer_step += 1

        lr_now = _scheduled_lr(
            step=optimizer_step,
            warmup_steps=config.trainer.warmup_steps,
            max_steps=target_steps,
            base_lr=config.optimizer.lr,
            cosine_decay_to=config.trainer.cosine_decay_to,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        scalar = terms.scalars()
        metric_row = {
            "loss_total": float(scalar["loss/total"]),
            "loss_prediction_mse": float(scalar["loss/prediction_mse"]),
            "loss_sigreg": float(scalar["loss/sigreg"]),
            "loss_action_swap_contrastive": (
                float(swap_val) if swap_val is not None else 0.0
            ),
            "no_action_mse": float(no_action_mse_val),
            "margin_no_action_minus_pred": float(margin_val),
            "lr": float(lr_now),
        }
        if initial_metrics is None:
            initial_metrics = dict(metric_row)
        last_metrics = dict(metric_row)
        tail_metrics.append(metric_row)
        if len(tail_metrics) > tail_window:
            tail_metrics.pop(0)

        _append_metrics_row(metrics_path, step=optimizer_step, metrics=metric_row)

        if (
            collapse_diag_path is not None
            and optimizer_step
            % config.trainer.collapse_diagnostics_every_n_steps
            == 0
        ):
            last_collapse_report = _emit_collapse_diagnostics(
                model=model,
                loader_config=loader_config,
                pad_token_id=0,
                output_sequence_length=config.loader.output_sequence_length,
                vocab_size=_DEFAULT_VOCAB_SIZE,
                device=selected_device,
                torch_=torch,
                step=optimizer_step,
                output_path=collapse_diag_path,
            )

        if (
            optimizer_step % config.trainer.checkpoint_every_n_steps == 0
            or optimizer_step == target_steps
        ):
            _write_periodic_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                seed=seed,
                step=optimizer_step,
                metrics=metric_row,
                checkpoint_dir=checkpoint_dir,
                torch_=torch,
            )
            last_checkpoint_metrics = dict(metric_row)
            metric_value = metric_row.get(config.trainer.keep_best_by_metric)
            if metric_value is None:
                # Common alias: callers spell train metrics with a
                # "val_" prefix even when no held-out split exists.
                aliased = config.trainer.keep_best_by_metric.removeprefix("val_")
                metric_value = metric_row.get(aliased)
            if metric_value is not None:
                if best_checkpoint_metric is None or metric_value < best_checkpoint_metric:
                    best_checkpoint_metric = float(metric_value)
                    _copy_checkpoint(
                        src=checkpoint_dir / f"checkpoint_step_{optimizer_step:08d}.pt",
                        dst=checkpoint_dir / "best.pt",
                    )

    if not last_metrics:
        raise TrainingRunError(
            "execution-substrate training did not execute any optimizer steps"
        )

    # Ensure we always have a "last" checkpoint pointer for downloaders.
    last_step_checkpoint = (
        checkpoint_dir / f"checkpoint_step_{optimizer_step:08d}.pt"
    )
    if last_step_checkpoint.exists():
        _copy_checkpoint(src=last_step_checkpoint, dst=checkpoint_dir / "last.pt")

    # Tail-smoothed final metrics so a single noisy step at the end
    # doesn't dominate the reported summary.
    final_metrics = {
        key: _mean(row[key] for row in tail_metrics)
        for key in last_metrics
    }
    deltas = (
        {k: final_metrics[k] - initial_metrics[k] for k in initial_metrics}
        if initial_metrics is not None
        else {}
    )
    elapsed = max(time.perf_counter() - started_at, 1e-12)

    z_diagnostics = _latent_diagnostics(
        model=model,
        loader_config=loader_config,
        pad_token_id=0,
        output_sequence_length=config.loader.output_sequence_length,
        vocab_size=_DEFAULT_VOCAB_SIZE,
        device=selected_device,
        torch_=torch,
    )

    report_path = reports_dir / "execution_train_run_report.json"
    report_payload = {
        "schema_version": EXECUTION_TRAIN_RUN_REPORT_SCHEMA_VERSION,
        "run_name": config.name,
        "seed": seed,
        "device": str(selected_device),
        "step_count": optimizer_step,
        "pack_record_count": pack_record_count,
        "pack_jsonl_path": pack_jsonl_path.as_posix(),
        "pack_dir": pack_dir.as_posix(),
        "wall_time_seconds": round(elapsed, 3),
        "initial_metrics": initial_metrics or {},
        "final_metrics": final_metrics,
        "deltas": deltas,
        "z_diagnostics": z_diagnostics,
        "last_collapse_report": last_collapse_report,
        "objective": {
            "prediction_mse_weight": config.objective.prediction_mse_weight,
            "sigreg_weight": config.objective.sigreg_weight,
            "action_swap_contrastive_weight": config.objective.action_swap_contrastive_weight,
            "inverse_action_reconstruction_weight": (
                config.objective.inverse_action_reconstruction_weight
            ),
        },
        "claim_gates": {
            "retrieval_min_recall_at_1_lift_over_no_action": (
                config.claim_gates.retrieval_min_recall_at_1_lift_over_no_action
            ),
            "retrieval_min_mrr_lift_over_no_action": (
                config.claim_gates.retrieval_min_mrr_lift_over_no_action
            ),
            "collapse_effective_rank_ratio_min": (
                config.claim_gates.collapse_effective_rank_ratio_min
            ),
            "required_seeds": config.claim_gates.required_seeds,
        },
        "claim_boundary": {
            "name": config.claim_boundary.name,
            "scope": config.claim_boundary.scope,
        },
        "step_schema_version": EXECUTION_TRAIN_STEP_SCHEMA_VERSION,
        "training_report_schema_version": EXECUTION_TRAIN_REPORT_SCHEMA_VERSION,
    }
    _write_json(report_payload, report_path)

    # Apply checkpoint retention before recording manifests so the file
    # set on disk matches the artifact manifest.
    retained_checkpoints = _apply_checkpoint_retention(
        checkpoint_dir=checkpoint_dir,
        keep_last_n=config.trainer.keep_last_n_checkpoints,
    )

    tensorboard_report_paths: tuple[Path, ...] = ()
    tensorboard_metadata: dict[str, Any] = {"enabled": False}
    if tensorboard_flag:
        try:
            tensorboard_metadata, tensorboard_report_paths = _export_tensorboard(
                run_id=config.name,
                run_dir=output_path,
                step_count=optimizer_step,
                metrics=final_metrics,
                model=model,
                tensorboard_dir=tensorboard_dir,
            )
        except OptionalDependencyError as exc:
            # Don't fail the run for a missing TensorBoard install — the
            # event-file export is diagnostic only. Surface the reason
            # in the metadata.
            tensorboard_metadata = {
                "enabled": False,
                "skipped_reason": str(exc),
            }
    report_payload["tensorboard_export"] = tensorboard_metadata
    _write_json(report_payload, report_path)

    # Build the final artifact manifest. All paths must live under
    # output_path for the manifest contract to hold.
    artifact_files: list[Path] = [config_path, metrics_path, report_path]
    if collapse_diag_path is not None and collapse_diag_path.is_file():
        artifact_files.append(collapse_diag_path)
    for ckpt_path in retained_checkpoints:
        artifact_files.append(ckpt_path)
        sidecar = ckpt_path.with_name(ckpt_path.name + ".manifest.json")
        if sidecar.is_file():
            artifact_files.append(sidecar)
    for tb_path in tensorboard_report_paths:
        if tb_path.is_file() and _is_under(tb_path, output_path):
            artifact_files.append(tb_path)

    artifact_metadata = {
        "schema_version": TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": config.name,
        "seed": seed,
        "step_count": optimizer_step,
        "final_metrics": dict(final_metrics),
        "pack_repo_id": config.data.pack_repo_id,
        "pack_revision": config.data.pack_revision,
        "pack_jsonl_path": pack_jsonl_path.as_posix(),
        "pack_record_count": pack_record_count,
        "claim_boundary": {
            "name": config.claim_boundary.name,
            "scope": config.claim_boundary.scope,
        },
        "claim_gates": {
            "retrieval_min_recall_at_1_lift_over_no_action": (
                config.claim_gates.retrieval_min_recall_at_1_lift_over_no_action
            ),
            "retrieval_min_mrr_lift_over_no_action": (
                config.claim_gates.retrieval_min_mrr_lift_over_no_action
            ),
            "required_seeds": config.claim_gates.required_seeds,
        },
        "tensorboard_export": tensorboard_metadata,
        "executor": {
            "executor": "execution_torch",
            "torch": str(torch.__version__),
            "device": str(selected_device),
            "precision": config.trainer.precision,
            "pack_batch_schema_version": EXECUTION_PACK_BATCH_SCHEMA_VERSION,
            "pack_manifest_schema_version": EXECUTION_PACK_MANIFEST_SCHEMA_VERSION,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        },
    }
    parent_artifact_ids = (
        (parent_pack_artifact_id,) if parent_pack_artifact_id else ("execution_pack:unmanaged",)
    )
    artifact_manifest = build_artifact_manifest(
        artifact_kind="training_run",
        root=output_path,
        files=artifact_files,
        command=command,
        config=config_payload,
        parent_artifacts=parent_artifact_ids,
        source_git_sha=source_git_sha,
        created_at=created_at,
        metadata=artifact_metadata,
    )
    artifact_manifest_path = output_path / "manifest.json"
    write_artifact_manifest(artifact_manifest, artifact_manifest_path)

    # Build the training-run manifest the rest of the codebase consumes.
    checkpoint_manifest_files = tuple(
        build_manifest_file(path, root=output_path)
        for path in retained_checkpoints
        if path.is_file()
    )
    report_manifest_files = tuple(
        build_manifest_file(path, root=output_path)
        for path in (report_path, *tensorboard_report_paths)
        if path.is_file() and _is_under(path, output_path)
    )
    training_manifest = TrainingRunManifest(
        run_id=config.name,
        config_sha256=artifact_manifest.config_sha256,
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path=_relative_to(artifact_manifest_path, output_path),
        parent_artifacts=parent_artifact_ids,
        dataset_manifest_path=pack_jsonl_path.as_posix(),
        config_path=_relative_to(config_path, output_path),
        metrics_path=_relative_to(metrics_path, output_path),
        metrics_report_path=_relative_to(report_path, output_path),
        checkpoint_files=checkpoint_manifest_files,
        report_files=report_manifest_files,
        final_metrics=dict(final_metrics),
        step_count=optimizer_step,
        seed=seed,
        metadata={
            "executor": {
                "executor": "execution_torch",
                "device": str(selected_device),
                "precision": config.trainer.precision,
                "torch": str(torch.__version__),
            },
            "pack": {
                "repo_id": config.data.pack_repo_id,
                "revision": config.data.pack_revision,
                "record_count": pack_record_count,
                "jsonl_path": pack_jsonl_path.as_posix(),
            },
            "tensorboard_export": tensorboard_metadata,
        },
    )
    training_manifest_path = output_path / "training_manifest.json"
    training_manifest_path.write_text(
        json.dumps(training_manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return ExecutionTrainRunResult(
        training_manifest=training_manifest,
        training_manifest_path=training_manifest_path,
        artifact_manifest_path=artifact_manifest_path,
        checkpoint_paths=retained_checkpoints,
        metrics_path=metrics_path,
        report_path=report_path,
        pack_dir=pack_dir,
        collapse_diagnostics_path=collapse_diag_path,
    )


# --- helpers ---------------------------------------------------------------


def _resolve_pack(
    *,
    config: ExecutionTrainConfig,
    output_dir: Path,
    pack_local_dir: Path | str | None,
    pack_jsonl_override: Path | str | None,
) -> tuple[Path, Path, str | None]:
    """Resolve the execution pack location and return ``(dir, jsonl, parent_id)``.

    See the module docstring for the resolution order.
    """

    if pack_jsonl_override is not None:
        jsonl_path = Path(pack_jsonl_override).resolve()
        return jsonl_path.parent, jsonl_path, _read_pack_parent_artifact(jsonl_path.parent, config)

    env_dir = os.environ.get(_PACK_LOCAL_DIR_ENV)
    if pack_local_dir is None and env_dir:
        pack_local_dir = env_dir

    if pack_local_dir is not None:
        pack_dir = Path(pack_local_dir).resolve()
        if not pack_dir.is_dir():
            raise SourceUnavailableError(
                f"execution pack local dir does not exist: {pack_dir}"
            )
        jsonl_path = pack_dir / config.data.pack_jsonl
        return pack_dir, jsonl_path, _read_pack_parent_artifact(pack_dir, config)

    # Fall back to HF download. Use the existing huggingface_hub helper.
    pack_dir = output_dir / "_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(
            "huggingface_hub is required to download the execution pack; "
            "set CODELEWM_EXECUTION_PACK_LOCAL_DIR or install codelewm[release]"
        ) from exc
    snapshot_download(
        repo_id=config.data.pack_repo_id,
        repo_type="dataset",
        revision=config.data.pack_revision,
        local_dir=str(pack_dir),
    )
    jsonl_path = pack_dir / config.data.pack_jsonl
    return pack_dir, jsonl_path, _read_pack_parent_artifact(pack_dir, config)


def _read_pack_parent_artifact(
    pack_dir: Path, config: ExecutionTrainConfig
) -> str | None:
    """Return the pack's ``pack_id`` so it can be recorded as a parent artifact."""

    manifest_path = pack_dir / config.data.manifest_filename
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    pack_id = payload.get("pack_id") if isinstance(payload, dict) else None
    if isinstance(pack_id, str) and pack_id:
        return pack_id
    return None


def _scheduled_lr(
    *,
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    cosine_decay_to: float,
) -> float:
    """Warmup → cosine decay schedule matching the v0.6 config defaults."""

    if step <= warmup_steps and warmup_steps > 0:
        return base_lr * step / warmup_steps
    if step >= max_steps:
        return base_lr * cosine_decay_to
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (cosine_decay_to + (1.0 - cosine_decay_to) * cosine)


def _resolve_device(*, requested: str, torch_: Any) -> Any:
    """Resolve the requested accelerator string to a torch device."""

    if requested == "auto":
        if torch_.cuda.is_available():
            return torch_.device("cuda")
        if hasattr(torch_.backends, "mps") and torch_.backends.mps.is_available():
            return torch_.device("mps")
        return torch_.device("cpu")
    if requested in {"cuda", "gpu"}:
        if not torch_.cuda.is_available():
            raise TrainingRunError("requested cuda/gpu device but cuda is not available")
        return torch_.device("cuda")
    if requested == "mps":
        if not hasattr(torch_.backends, "mps") or not torch_.backends.mps.is_available():
            raise TrainingRunError("requested mps device but mps is not available")
        return torch_.device("mps")
    if requested == "cpu":
        return torch_.device("cpu")
    raise TrainingRunError(f"unsupported accelerator: {requested}")


def _append_metrics_row(
    path: Path, *, step: int, metrics: Mapping[str, float]
) -> None:
    row = {
        "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
        "step": step,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _emit_collapse_diagnostics(
    *,
    model: Any,
    loader_config: ExecutionPackLoaderConfig,
    pad_token_id: int,
    output_sequence_length: int,
    vocab_size: int,
    device: Any,
    torch_: Any,
    step: int,
    output_path: Path,
) -> dict[str, Any]:
    """Append one collapse-diagnostics row and return its payload."""

    diagnostics = _latent_diagnostics(
        model=model,
        loader_config=loader_config,
        pad_token_id=pad_token_id,
        output_sequence_length=output_sequence_length,
        vocab_size=vocab_size,
        device=device,
        torch_=torch_,
    )
    row = {
        "schema_version": EXECUTION_TRAIN_COLLAPSE_DIAGNOSTICS_SCHEMA_VERSION,
        "step": step,
        "diagnostics": diagnostics,
    }
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return diagnostics


def _write_periodic_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    config: ExecutionTrainConfig,
    seed: int,
    step: int,
    metrics: Mapping[str, float],
    checkpoint_dir: Path,
    torch_: Any,
) -> Path:
    """Write a step-tagged checkpoint plus its manifest sidecar."""

    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:08d}.pt"
    payload = {
        "schema_version": "codelewm.execution_train_checkpoint.v1",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "seed": seed,
        "config_name": config.name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "compatibility_config": _compatibility_payload(config),
        "compatibility_config_hash": compute_config_hash(_compatibility_payload(config)),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    torch_.save(payload, checkpoint_path)
    manifest_path = checkpoint_path.with_name(checkpoint_path.name + ".manifest.json")
    metadata = build_checkpoint_metadata(
        _compatibility_payload(config),
        record_schema_version=EXECUTION_PACK_RECORD_SCHEMA_VERSION,
        latent_dim=config.wm.embed_dim,
        action_view="text",
        model_class="TorchCodeTransitionModel",
    )
    write_checkpoint_manifest(
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
    )
    return checkpoint_path


def _compatibility_payload(config: ExecutionTrainConfig) -> dict[str, Any]:
    """Return the slice of the config that defines checkpoint compatibility.

    A v0.6 execution checkpoint is interchangeable across runs that
    agree on the world-model arch and the objective surface; per-seed
    or schedule differences do not invalidate the checkpoint.
    """

    return {
        "wm": {
            "history_size": config.wm.history_size,
            "num_preds": config.wm.num_preds,
            "embed_dim": config.wm.embed_dim,
            "action_view": "text",
            "state_sequence_length": STATE_SEQUENCE_LENGTH,
            "action_sequence_length": TEXT_ACTION_SEQUENCE_LENGTH,
            # Persist the state-encoder architecture (RFC-0015 WS-C1) so the
            # checkpoint is self-describing: eval rebuilds the matching model
            # without inferring it from the weights.
            "state_encoder_type": config.wm.state_encoder_type,
            "state_encoder_layers": config.wm.state_encoder_layers,
            "state_encoder_heads": config.wm.state_encoder_heads,
        },
        "objective": {
            "prediction_mse_weight": config.objective.prediction_mse_weight,
            "sigreg_weight": config.objective.sigreg_weight,
            "action_swap_contrastive_weight": (
                config.objective.action_swap_contrastive_weight
            ),
            "inverse_action_reconstruction_weight": (
                config.objective.inverse_action_reconstruction_weight
            ),
        },
        "loader": {
            "code_sequence_length": config.loader.code_sequence_length,
            "action_sequence_length": config.loader.action_sequence_length,
            "output_sequence_length": config.loader.output_sequence_length,
        },
    }


def _copy_checkpoint(*, src: Path, dst: Path) -> None:
    """Copy a checkpoint and its manifest sidecar to ``dst``.

    Used for the ``last.pt`` / ``best.pt`` pointers downstream code
    expects in the downloaded artifact directory.
    """

    if not src.exists():
        return
    dst.write_bytes(src.read_bytes())
    src_manifest = src.with_name(src.name + ".manifest.json")
    if src_manifest.is_file():
        dst_manifest = dst.with_name(dst.name + ".manifest.json")
        # Rewrite the manifest with the new checkpoint path so the
        # sidecar matches the file it sits beside (and the sha256 is
        # recomputed against the pointer file).
        try:
            import json as _json

            payload = _json.loads(src_manifest.read_text(encoding="utf-8"))
            payload["checkpoint_path"] = dst.name
            # Recompute the checksum against the new pointer file.
            from codelewm.model.checkpoint import sha256_file

            payload["checkpoint_sha256"] = sha256_file(dst)
            dst_manifest.write_text(
                _json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError):
            # Fall back to a plain copy if the manifest can't be
            # rewritten; the runner still recorded the original sidecar
            # next to the step-tagged checkpoint.
            dst_manifest = dst.with_name(dst.name + ".manifest.json")
            dst_manifest.write_bytes(src_manifest.read_bytes())


def _apply_checkpoint_retention(
    *, checkpoint_dir: Path, keep_last_n: int
) -> tuple[Path, ...]:
    """Apply the keep-last-N policy and return the final retained checkpoints.

    The ``last.pt`` and ``best.pt`` pointers are always retained;
    step-tagged checkpoints beyond the budget are deleted (with their
    manifest sidecars). The returned tuple is the full set of remaining
    checkpoint files that will be recorded on the artifact manifest.
    """

    step_checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: p.name,
    )
    if keep_last_n > 0 and len(step_checkpoints) > keep_last_n:
        to_drop = step_checkpoints[:-keep_last_n]
        for path in to_drop:
            manifest_path = path.with_name(path.name + ".manifest.json")
            if path.is_file():
                path.unlink()
            if manifest_path.is_file():
                manifest_path.unlink()
        step_checkpoints = step_checkpoints[-keep_last_n:]

    retained: list[Path] = list(step_checkpoints)
    pointer_last = checkpoint_dir / "last.pt"
    pointer_best = checkpoint_dir / "best.pt"
    for pointer in (pointer_last, pointer_best):
        if pointer.is_file() and pointer not in retained:
            retained.append(pointer)
    return tuple(retained)


def _export_tensorboard(
    *,
    run_id: str,
    run_dir: Path,
    step_count: int,
    metrics: Mapping[str, float],
    model: Any,
    tensorboard_dir: Path | str | None,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Best-effort TensorBoard export; surfaces metadata even when skipped."""

    from .tensorboard_export import (
        TensorBoardExportError,
        export_tensorboard_training_run,
    )

    try:
        result = export_tensorboard_training_run(
            run_id=run_id,
            run_dir=run_dir,
            step_count=step_count,
            metrics=dict(metrics),
            model=model,
            embeddings=None,
            checkpoint_path=None,
            checkpoint_manifest_path=None,
            log_dir=tensorboard_dir,
        )
    except TensorBoardExportError as exc:
        return {"enabled": False, "skipped_reason": str(exc)}, ()
    paths = (result.report_path, *result.event_files)
    return result.to_metadata(root=run_dir), paths


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "EXECUTION_TRAIN_COLLAPSE_DIAGNOSTICS_SCHEMA_VERSION",
    "EXECUTION_TRAIN_RUN_REPORT_SCHEMA_VERSION",
    "ExecutionTrainRunResult",
    "train_execution_run",
]
