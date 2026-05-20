"""Package-native torch executor for packed CodeLeWM transition datasets."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from codelewm.data import DATASET_SCHEMA_VERSION, OptionalDependencyError
from codelewm.model import (
    CHECKPOINT_SCHEMA_VERSION,
    ABSTRACT_ACTION_SEQUENCE_LENGTH,
    CheckpointCompatibilityError,
    CodeStateBatch,
    ObjectiveConfig,
    STATE_SEQUENCE_LENGTH,
    TEXT_ACTION_SEQUENCE_LENGTH,
    TorchCodeTransitionModel,
    TorchCodeTransitionModelConfig,
    TransitionBatch,
    ActionBatch,
    build_checkpoint_metadata,
    build_torch_transition_model,
    compute_config_hash,
    compute_transition_objective,
    write_checkpoint_manifest,
)
from codelewm.security import CheckpointTrustError, require_trusted_checkpoint

from .config import TrainConfig
from .resume import compatibility_config_payload
from .runner import TrainingExecutorResult, TrainingRunContext, TrainingRunError, train

try:  # pragma: no cover - exercised when torch is installed.
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - lightweight local env.
    torch = None
    DataLoader = None
    Dataset = object


TORCH_TRAINING_REPORT_SCHEMA_VERSION = "codelewm.torch_training_report.v1"
TORCH_CHECKPOINT_SCHEMA_VERSION = "codelewm.torch_checkpoint.v1"
DEFAULT_TRAINING_VOCAB_SIZE = 32768


def train_torch(
    config: TrainConfig | dict[str, Any] | Path | str,
    *,
    root: Path | str = ".",
    source_git_sha: str | None = None,
    created_at: str | None = None,
    overwrite: bool = False,
    resume_from: Path | str | None = None,
    device: str | None = None,
):
    """Run the manifest-backed runner with the package-native torch executor."""

    executor = make_torch_training_executor(device=device)
    return train(
        config,
        executor=executor,
        root=root,
        command=("codelewm", "train", "--executor", "torch"),
        source_git_sha=source_git_sha,
        created_at=created_at,
        overwrite=overwrite,
        resume_from=resume_from,
    )


def make_torch_training_executor(*, device: str | None = None):
    """Return a torch executor with an optional explicit device override."""

    def _executor(context: TrainingRunContext) -> TrainingExecutorResult:
        return torch_training_executor(context, device=device)

    return _executor


def torch_training_executor(
    context: TrainingRunContext,
    *,
    device: str | None = None,
) -> TrainingExecutorResult:
    """Train the package-native transition model over packed HDF5 batches."""

    runtime = _require_torch_runtime()
    _seed_everything(context.config.seed, runtime)
    selected_device = _resolve_device(device or context.config.trainer.accelerator, runtime)
    precision = _precision_dtype(context.config.trainer.precision, selected_device, runtime)
    train_path = _resolve_context_path(context.config.data.train, root=context.root)
    val_path = _resolve_context_path(context.config.data.val, root=context.root)

    train_dataset = PackedTransitionHdf5Dataset(
        train_path,
        action_view=context.config.wm.action_view,
        vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
    )
    val_dataset = PackedTransitionHdf5Dataset(
        val_path,
        action_view=context.config.wm.action_view,
        vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
        allow_empty=True,
    )
    if len(train_dataset) == 0:
        raise TrainingRunError("torch training requires a non-empty train split")

    model = build_torch_transition_model(_model_config(context.config))
    model.to(selected_device)
    optimizer = runtime.optim.AdamW(
        model.parameters(),
        lr=context.config.optimizer.lr,
        weight_decay=context.config.optimizer.weight_decay,
    )
    start_step = 0
    if context.resume is not None:
        start_step = _load_resume_checkpoint(
            context=context,
            model=model,
            optimizer=optimizer,
            device=selected_device,
            runtime=runtime,
        )

    train_loader = _build_dataloader(
        train_dataset,
        context.config,
        runtime=runtime,
        seed=context.config.seed,
    )
    objective_config = _objective_config(context.config)
    objective_payload = context.config.loss.to_dict()
    target_step = start_step + context.config.trainer.max_steps
    step = start_step
    last_terms: dict[str, float] = {}
    last_embeddings = None
    examples_seen = 0
    started_at = time.perf_counter()

    model.train()
    while step < target_step:
        for raw_batch in train_loader:
            if step >= target_step:
                break
            batch = _to_transition_batch(
                raw_batch,
                device=selected_device,
                action_view=context.config.wm.action_view,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(context.config.trainer.precision, selected_device, runtime):
                outputs = model(batch)
                terms = compute_transition_objective(
                    outputs["z_before"],
                    outputs["z_after"],
                    outputs["z_pred_after"],
                    config=objective_config,
                )
            terms.total.backward()
            grad_norm = runtime.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=context.config.trainer.gradient_clip_val,
            )
            optimizer.step()
            step += 1
            batch_size = int(batch.state_before.input_ids.shape[0])
            examples_seen += batch_size
            last_terms = terms.scalars()
            last_terms["train/gradient_norm"] = _finite_float(grad_norm, "train/gradient_norm")
            last_embeddings = _collapse_embeddings(outputs, runtime=runtime)
        if len(train_loader) == 0:
            raise TrainingRunError("torch training dataloader produced no batches")

    if not last_terms or last_embeddings is None:
        raise TrainingRunError("torch training did not execute any optimizer steps")

    elapsed = max(time.perf_counter() - started_at, 1e-12)
    final_metrics = dict(last_terms)
    final_metrics["train/examples"] = float(examples_seen)
    final_metrics["train/examples_per_second"] = float(examples_seen / elapsed)
    if len(val_dataset) > 0:
        final_metrics.update(_evaluate_validation(model, val_dataset, context.config, selected_device, runtime))
    from codelewm.eval import compute_collapse_report

    collapse_report = compute_collapse_report(last_embeddings)
    final_metrics.update(_collapse_metrics(collapse_report))
    _validate_no_collapse_smoke(final_metrics)

    checkpoint_path = context.checkpoint_dir / "checkpoint.pt"
    checkpoint_manifest_path = checkpoint_path.with_name(checkpoint_path.name + ".manifest.json")
    _write_torch_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=context.config,
        step=step,
        metrics=final_metrics,
        runtime=runtime,
    )
    checkpoint_metadata = build_checkpoint_metadata(
        compatibility_config_payload(context.config),
        record_schema_version=DATASET_SCHEMA_VERSION,
        latent_dim=context.config.wm.embed_dim,
        action_view=context.config.wm.action_view,
        model_class="TorchCodeTransitionModel",
    )
    write_checkpoint_manifest(
        metadata=checkpoint_metadata,
        checkpoint_path=checkpoint_path,
        manifest_path=checkpoint_manifest_path,
    )

    report_path = context.run_dir / "reports" / "torch_training_report.json"
    report_payload = {
        "schema_version": TORCH_TRAINING_REPORT_SCHEMA_VERSION,
        "run_id": context.config.name,
        "step_count": step,
        "metrics": final_metrics,
        "collapse_report": collapse_report.to_dict(),
        "dataset": {
            "train_rows": len(train_dataset),
            "val_rows": len(val_dataset),
            "action_view": context.config.wm.action_view,
        },
        "objective": objective_payload,
        "runtime": {
            "device": str(selected_device),
            "precision": context.config.trainer.precision,
            "dtype": str(precision),
            "torch": str(runtime.__version__),
        },
    }
    _write_json(report_payload, report_path)

    return TrainingExecutorResult(
        step_count=step,
        metrics=final_metrics,
        checkpoint_paths=(checkpoint_path, checkpoint_manifest_path),
        report_paths=(report_path,),
        metadata={
            "executor": "torch",
            "device": str(selected_device),
            "precision": context.config.trainer.precision,
            "torch": str(runtime.__version__),
            "train_rows": len(train_dataset),
            "val_rows": len(val_dataset),
            "objective": objective_payload,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        },
    )


class PackedTransitionHdf5Dataset(Dataset):  # type: ignore[misc]
    """Torch dataset for split HDF5 files emitted by `codelewm dataset pack`."""

    def __init__(
        self,
        path: Path | str,
        *,
        action_view: str = "text",
        vocab_size: int = DEFAULT_TRAINING_VOCAB_SIZE,
        allow_empty: bool = False,
    ) -> None:
        runtime = _require_torch_runtime()
        h5py = _require_h5py()
        if action_view not in ("text", "abstract"):
            raise TrainingRunError("patch action is diagnostic only and cannot be used for training")
        if vocab_size <= 1:
            raise TrainingRunError("vocab_size must be greater than 1")
        self.path = Path(path)
        self.action_view = action_view
        self.vocab_size = vocab_size
        if not self.path.is_file():
            raise TrainingRunError(f"packed HDF5 split does not exist: {self.path}")

        with h5py.File(self.path, "r") as handle:
            schema_version = _hdf5_attr_text(handle.attrs.get("schema_version"))
            if schema_version != DATASET_SCHEMA_VERSION:
                raise TrainingRunError(
                    f"packed HDF5 schema_version must be {DATASET_SCHEMA_VERSION!r}; "
                    f"got {schema_version!r}"
                )
            row_count = int(handle.attrs.get("row_count", -1))
            action_group = "action_text" if action_view == "text" else "action_abs"
            state_before = _read_state_group(handle, "state_before", vocab_size=vocab_size)
            state_after = _read_state_group(handle, "state_after", vocab_size=vocab_size)
            action = _read_action_group(handle, action_group, vocab_size=vocab_size)

        observed_rows = int(state_before["input_ids"].shape[0])
        _require_width(state_before, STATE_SEQUENCE_LENGTH, group_name="state_before")
        _require_width(state_after, STATE_SEQUENCE_LENGTH, group_name="state_after")
        _require_width(
            action,
            TEXT_ACTION_SEQUENCE_LENGTH if action_view == "text" else ABSTRACT_ACTION_SEQUENCE_LENGTH,
            group_name=action_group,
        )
        if row_count != observed_rows:
            raise TrainingRunError(
                f"packed HDF5 row_count attr {row_count} does not match rows {observed_rows}"
            )
        if observed_rows == 0 and not allow_empty:
            raise TrainingRunError(f"packed HDF5 split is empty: {self.path}")
        _validate_matching_rows("state_after", observed_rows, state_after)
        _validate_matching_rows(action_group, observed_rows, action)
        self._runtime = runtime
        self._state_before = state_before
        self._state_after = state_after
        self._action = action

    def __len__(self) -> int:
        return int(self._state_before["input_ids"].shape[0])

    def __getitem__(self, index: int) -> dict[str, dict[str, Any]]:
        runtime = self._runtime
        return {
            "state_before": {
                key: runtime.as_tensor(value[index])
                for key, value in self._state_before.items()
            },
            "state_after": {
                key: runtime.as_tensor(value[index])
                for key, value in self._state_after.items()
            },
            "action": {
                key: runtime.as_tensor(value[index])
                for key, value in self._action.items()
            },
        }


def _require_torch_runtime() -> Any:
    if torch is None:
        raise OptionalDependencyError(
            "torch training requires torch; install the train dependency group with "
            "`uv sync --group train --group data --group dev`"
        )
    if importlib.util.find_spec("einops") is None:
        raise OptionalDependencyError(
            "torch training requires einops; install the train dependency group with "
            "`uv sync --group train --group data --group dev`"
        )
    return torch


def _require_h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "torch training requires h5py; install with `uv sync --group train --group data --group dev`"
        ) from exc
    return h5py


def _seed_everything(seed: int, runtime: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    runtime.manual_seed(seed)
    if runtime.cuda.is_available():
        runtime.cuda.manual_seed_all(seed)


def _resolve_device(requested: str, runtime: Any) -> Any:
    if requested == "auto":
        if runtime.cuda.is_available():
            return runtime.device("cuda")
        if hasattr(runtime.backends, "mps") and runtime.backends.mps.is_available():
            return runtime.device("mps")
        return runtime.device("cpu")
    if requested in {"cuda", "gpu"}:
        if not runtime.cuda.is_available():
            raise TrainingRunError("requested cuda/gpu training device is not available")
        return runtime.device("cuda")
    if requested == "mps":
        if not hasattr(runtime.backends, "mps") or not runtime.backends.mps.is_available():
            raise TrainingRunError("requested mps training device is not available")
        return runtime.device("mps")
    if requested == "cpu":
        return runtime.device("cpu")
    raise TrainingRunError(f"unsupported training device: {requested}")


def _precision_dtype(precision: str, device: Any, runtime: Any) -> Any:
    if precision in {"float32", "32-true"}:
        return runtime.float32
    if precision == "bf16-mixed":
        return runtime.bfloat16
    if precision in {"16-mixed", "fp16-mixed"}:
        return runtime.float16 if str(device) != "cpu" else runtime.float32
    raise TrainingRunError(f"unsupported precision: {precision}")


def _autocast_context(precision: str, device: Any, runtime: Any):
    device_type = str(device).split(":", maxsplit=1)[0]
    if precision == "bf16-mixed":
        return runtime.autocast(device_type=device_type, dtype=runtime.bfloat16)
    if precision in {"16-mixed", "fp16-mixed"} and device_type != "cpu":
        return runtime.autocast(device_type=device_type, dtype=runtime.float16)
    return nullcontext()


def _resolve_context_path(value: str | None, *, root: Path) -> Path:
    if value is None:
        raise TrainingRunError("training data path is required")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _model_config(config: TrainConfig) -> TorchCodeTransitionModelConfig:
    return TorchCodeTransitionModelConfig(
        action_view=config.wm.action_view,
        latent_dim=config.wm.embed_dim,
        state_sequence_length=config.wm.state_sequence_length,
        action_sequence_length=config.wm.action_sequence_length,
        vocab_size=DEFAULT_TRAINING_VOCAB_SIZE,
        dropout=0.0,
    )


def _objective_config(config: TrainConfig) -> ObjectiveConfig:
    return ObjectiveConfig(
        sigreg_weight=config.loss.sigreg_weight,
        enable_retrieval_loss=config.loss.enable_retrieval_loss,
        retrieval_weight=config.loss.retrieval_weight,
        retrieval_temperature=config.loss.retrieval_temperature,
        enable_action_use_margin=config.loss.enable_action_use_margin,
        action_use_margin_weight=config.loss.action_use_margin_weight,
        action_use_margin=config.loss.action_use_margin,
        sigreg_knots=config.loss.sigreg_knots,
        sigreg_num_proj=config.loss.sigreg_num_proj,
        sigreg_seed=config.seed,
    )


def _build_dataloader(
    dataset: PackedTransitionHdf5Dataset,
    config: TrainConfig,
    *,
    runtime: Any,
    seed: int,
) -> Any:
    if DataLoader is None:
        raise OptionalDependencyError("torch DataLoader is unavailable")
    generator = runtime.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(config.loader.batch_size, max(len(dataset), 1)),
        shuffle=config.loader.shuffle,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        persistent_workers=config.loader.persistent_workers,
        generator=generator,
    )


def _to_transition_batch(raw: Any, *, device: Any, action_view: str) -> TransitionBatch:
    state_before = _to_state_batch(raw["state_before"], device=device)
    state_after = _to_state_batch(raw["state_after"], device=device)
    action_payload = {
        key: value.to(device=device)
        for key, value in raw["action"].items()
    }
    action = ActionBatch(
        input_ids=action_payload["input_ids"].long(),
        attention_mask=action_payload["attention_mask"].bool(),
        action_view=action_view,  # type: ignore[arg-type]
    )
    return TransitionBatch(state_before=state_before, action=action, state_after=state_after)


def _to_state_batch(raw_state: dict[str, Any], *, device: Any) -> CodeStateBatch:
    payload = {
        key: value.to(device=device)
        for key, value in raw_state.items()
    }
    return CodeStateBatch(
        input_ids=payload["input_ids"].long(),
        attention_mask=payload["attention_mask"].bool(),
        segment_ids=payload["segment_ids"].long(),
        changed_hunk_mask=payload["changed_hunk_mask"].bool(),
    )


def _evaluate_validation(
    model: TorchCodeTransitionModel,
    dataset: PackedTransitionHdf5Dataset,
    config: TrainConfig,
    device: Any,
    runtime: Any,
) -> dict[str, float]:
    loader = DataLoader(  # type: ignore[misc]
        dataset,
        batch_size=min(config.loader.batch_size, max(len(dataset), 1)),
        shuffle=False,
        num_workers=0,
    )
    objective_config = _objective_config(config)
    accum: dict[str, list[float]] = {}
    was_training = model.training
    model.eval()
    with runtime.no_grad():
        for raw_batch in loader:
            batch = _to_transition_batch(
                raw_batch,
                device=device,
                action_view=config.wm.action_view,
            )
            outputs = model(batch)
            terms = compute_transition_objective(
                outputs["z_before"],
                outputs["z_after"],
                outputs["z_pred_after"],
                config=objective_config,
            )
            for key, value in terms.scalars().items():
                accum.setdefault(f"val/{key}", []).append(value)
    if was_training:
        model.train()
    return {key: float(np.mean(values)) for key, values in accum.items()}


def _collapse_embeddings(outputs: dict[str, Any], *, runtime: Any) -> np.ndarray:
    values = runtime.cat(
        (
            outputs["z_before"].detach(),
            outputs["z_after"].detach(),
            outputs["z_pred_after"].detach(),
        ),
        dim=0,
    )
    return values.float().cpu().numpy()


def _collapse_metrics(report: Any) -> dict[str, float]:
    return {
        "collapse/effective_rank": report.effective_rank,
        "collapse/effective_rank_ratio": report.effective_rank_ratio,
        "collapse/per_dim_variance_min": report.per_dim_variance_min,
        "collapse/per_dim_variance_median": report.per_dim_variance_median,
        "collapse/per_dim_variance_max": report.per_dim_variance_max,
        "collapse/pairwise_cosine_mean": report.pairwise_cosine_mean,
        "collapse/embedding_norm_mean": report.embedding_norm_mean,
        "collapse/nearest_neighbor_entropy": report.nearest_neighbor_entropy,
        "collapse/embedding_count": float(report.embedding_count),
        "collapse/latent_dim": float(report.latent_dim),
    }


def _validate_no_collapse_smoke(metrics: dict[str, float]) -> None:
    variance_max = metrics.get("collapse/per_dim_variance_max", 0.0)
    norm_mean = metrics.get("collapse/embedding_norm_mean", 0.0)
    if not math.isfinite(variance_max) or variance_max <= 0.0:
        raise TrainingRunError("torch training collapse smoke failed: embedding variance is zero")
    if not math.isfinite(norm_mean) or norm_mean <= 0.0:
        raise TrainingRunError("torch training collapse smoke failed: embedding norm is zero")


def _write_torch_checkpoint(
    path: Path,
    *,
    model: TorchCodeTransitionModel,
    optimizer: Any,
    config: TrainConfig,
    step: int,
    metrics: dict[str, float],
    runtime: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TORCH_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "compatibility_config": compatibility_config_payload(config),
        "compatibility_config_hash": compute_config_hash(compatibility_config_payload(config)),
        "metrics": metrics,
    }
    runtime.save(payload, path)


def _load_resume_checkpoint(
    *,
    context: TrainingRunContext,
    model: TorchCodeTransitionModel,
    optimizer: Any,
    device: Any,
    runtime: Any,
) -> int:
    if context.resume is None:
        return 0
    try:
        require_trusted_checkpoint(
            context.resume.parent_checkpoint_path,
            manifest_path=context.resume.parent_checkpoint_manifest_path,
        )
    except CheckpointTrustError as exc:
        raise TrainingRunError(f"resume checkpoint rejected: {exc}") from exc
    try:
        payload = runtime.load(
            context.resume.parent_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:  # pragma: no cover - older torch compatibility.
        payload = runtime.load(context.resume.parent_checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise TrainingRunError("resume checkpoint payload must be a mapping")
    if payload.get("schema_version") != TORCH_CHECKPOINT_SCHEMA_VERSION:
        raise TrainingRunError(
            "resume checkpoint schema_version is unsupported: "
            f"{payload.get('schema_version')!r}"
        )
    try:
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (KeyError, RuntimeError, ValueError, CheckpointCompatibilityError) as exc:
        raise TrainingRunError(f"resume checkpoint state could not be loaded: {exc}") from exc
    _move_optimizer_state(optimizer, device=device)
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TrainingRunError("resume checkpoint step must be a non-negative integer")
    return step


def _move_optimizer_state(optimizer: Any, *, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch is not None and torch.is_tensor(value):
                state[key] = value.to(device=device)


def _read_state_group(handle: Any, group_name: str, *, vocab_size: int) -> dict[str, np.ndarray]:
    return {
        "input_ids": _read_token_matrix(handle, f"{group_name}/input_ids", vocab_size=vocab_size),
        "attention_mask": _read_bool_matrix(handle, f"{group_name}/attention_mask"),
        "segment_ids": _read_int_matrix(handle, f"{group_name}/segment_ids"),
        "changed_hunk_mask": _read_bool_matrix(handle, f"{group_name}/changed_hunk_mask"),
    }


def _read_action_group(handle: Any, group_name: str, *, vocab_size: int) -> dict[str, np.ndarray]:
    return {
        "input_ids": _read_token_matrix(handle, f"{group_name}/input_ids", vocab_size=vocab_size),
        "attention_mask": _read_bool_matrix(handle, f"{group_name}/attention_mask"),
    }


def _read_token_matrix(handle: Any, key: str, *, vocab_size: int) -> np.ndarray:
    values = _read_int_matrix(handle, key)
    return np.where(values > 0, ((values - 1) % (vocab_size - 1)) + 1, 0).astype(np.int64)


def _read_int_matrix(handle: Any, key: str) -> np.ndarray:
    if key not in handle:
        raise TrainingRunError(f"packed HDF5 is missing dataset {key!r}")
    values = np.asarray(handle[key], dtype=np.int64)
    if values.ndim != 2:
        raise TrainingRunError(f"packed HDF5 dataset {key!r} must be rank 2")
    if values.size and values.min() < 0:
        raise TrainingRunError(f"packed HDF5 dataset {key!r} contains negative ids")
    return values


def _read_bool_matrix(handle: Any, key: str) -> np.ndarray:
    if key not in handle:
        raise TrainingRunError(f"packed HDF5 is missing dataset {key!r}")
    values = np.asarray(handle[key], dtype=bool)
    if values.ndim != 2:
        raise TrainingRunError(f"packed HDF5 dataset {key!r} must be rank 2")
    return values


def _validate_matching_rows(group_name: str, expected_rows: int, arrays: dict[str, np.ndarray]) -> None:
    for key, value in arrays.items():
        if value.shape[0] != expected_rows:
            raise TrainingRunError(
                f"packed HDF5 group {group_name!r} dataset {key!r} has "
                f"{value.shape[0]} rows, expected {expected_rows}"
            )


def _require_width(arrays: dict[str, np.ndarray], expected_width: int, *, group_name: str) -> None:
    for key, value in arrays.items():
        if value.shape[1] != expected_width:
            raise TrainingRunError(
                f"packed HDF5 group {group_name!r} dataset {key!r} has width "
                f"{value.shape[1]}, expected {expected_width}"
            )


def _hdf5_attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return "" if value is None else str(value)


def _finite_float(value: Any, field: str) -> float:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().item()
    result = float(value)
    if not math.isfinite(result):
        raise TrainingRunError(f"{field} must be finite")
    return result


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
