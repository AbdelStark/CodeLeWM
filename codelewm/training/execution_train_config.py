"""Config schema for the v0.6 execution-substrate training run.

The legacy ``codelewm.train_config.v1`` parser (:mod:`codelewm.training.config`)
expects an HDF5 transition-pack layout: a ``data.train`` and ``data.val``
path on disk plus a parent ``data.manifest`` artifact. The v0.6
execution-substrate run uses a different substrate entirely — a
JSONL execution pack downloaded from Hugging Face at runtime — and its
operator-facing YAML is shaped to match. Rather than overload the v1
schema, we keep the contracts side-by-side and let the CLI route to the
right runner by peeking the ``schema_version`` marker.

This module owns:

- :data:`EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION` — the public schema marker
  the launcher generator (:mod:`codelewm.training.execution_launch_plan`)
  and the production runner (:mod:`codelewm.training.execution_runner`)
  agree on.
- :class:`ExecutionTrainConfig` — strictly-validated config dataclass.
- :func:`load_execution_train_config` — file-level loader.
- :func:`peek_train_config_schema_version` — non-destructive schema peek
  the CLI uses to choose between the legacy HDF5 path and this one.

The YAML is the same YAML the launcher already validates; we keep both
parsers in lockstep so the operator-facing surface stays consistent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION = "codelewm.execution_train_config.v1"

_VALID_SUBSTRATES = frozenset({"execution_trace_v1"})
_VALID_ACCELERATORS = frozenset({"auto", "cpu", "cuda", "gpu", "mps"})
_VALID_PRECISIONS = frozenset(
    {"float32", "32-true", "bf16-mixed", "16-mixed", "fp16-mixed"}
)


class ExecutionTrainConfigError(ValueError):
    """Raised when the v0.6 execution-train config violates its contract."""


@dataclass(frozen=True)
class ExecutionTrainDataConfig:
    """Source of the execution pack the trainer consumes."""

    pack_repo_id: str
    pack_revision: str
    pack_jsonl: str
    manifest_filename: str
    claim_boundary_filename: str
    ingestion_sources: tuple[str, ...]
    held_out_for_eval: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionTrainLoaderConfig:
    """Loader-side knobs (sequence lengths + batch sizing)."""

    code_sequence_length: int
    action_sequence_length: int
    output_sequence_length: int
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    shuffle: bool


@dataclass(frozen=True)
class ExecutionTrainTrainerConfig:
    """Trainer schedule and checkpoint cadence."""

    accelerator: str
    devices: int
    precision: str
    max_steps: int
    warmup_steps: int
    cosine_decay_to: float
    gradient_clip_val: float
    checkpoint_every_n_steps: int
    keep_last_n_checkpoints: int
    keep_best_by_metric: str
    tensorboard_enabled: bool
    collapse_diagnostics_every_n_steps: int
    progress_log_every_n_steps: int = 100


@dataclass(frozen=True)
class ExecutionTrainOptimizerConfig:
    name: str
    lr: float
    betas: tuple[float, float]
    weight_decay: float


@dataclass(frozen=True)
class ExecutionTrainWorldModelConfig:
    history_size: int
    num_preds: int
    embed_dim: int
    # RFC-0015 WS-C1: state encoder backbone. Optional with v0.6 defaults so
    # existing configs parse unchanged.
    state_encoder_type: str = "pool"
    state_encoder_layers: int = 4
    state_encoder_heads: int = 8
    enable_ema_target_encoder: bool = False
    ema_target_decay: float = 0.99


@dataclass(frozen=True)
class ExecutionTrainObjectiveConfig:
    prediction_mse_weight: float
    sigreg_weight: float
    action_swap_contrastive_weight: float
    inverse_action_reconstruction_weight: float
    # RFC-0015 WS-C3: in-batch InfoNCE/retrieval term. Optional, default 0.0
    # (off) so existing configs are unchanged; capped at 0.10 to match the
    # objective's retrieval_weight_cap.
    retrieval_weight: float = 0.0
    p_pass_bce_weight: float = 0.0
    p_pass_bce_pos_weight: float = 1.0
    output_value_ce_weight: float = 0.0


@dataclass(frozen=True)
class ExecutionTrainHfJobsConfig:
    flavor: str
    region: str | None
    timeout_hours: int
    run_name_template: str
    artifact_repo_id: str
    checkpoint_repo_id: str
    checkpoint_revision_template: str
    runtime_image: str | None = None


@dataclass(frozen=True)
class ExecutionTrainClaimGatesConfig:
    """Numeric gates the eval harness checks once a run completes.

    The runner does not enforce these gates — that's the eval harness's
    job (#266-#269). They are recorded on the artifact manifest so the
    benchmark report can quote the gate values without re-reading the
    config.
    """

    retrieval_min_recall_at_1_lift_over_no_action: float
    retrieval_min_mrr_lift_over_no_action: float
    collapse_effective_rank_ratio_min: float
    collapse_per_dim_variance_median_min: float
    collapse_nearest_neighbor_entropy_min: float
    surprise_mutation_auc_min: float
    surprise_same_problem_different_submission_auc_min: float
    surprise_same_code_different_input_auc_min: float
    downstream_rerank_pass_at_1_lift_min: float
    required_seeds: int


@dataclass(frozen=True)
class ExecutionTrainClaimBoundaryConfig:
    name: str
    scope: str


@dataclass(frozen=True)
class ExecutionTrainConfig:
    """The complete v0.6 execution-substrate training config.

    Frozen + JSON-serializable. The runner reads this; it never mutates
    it. Per-seed runs differ only in ``--seed``; the rest of the config
    is shared.
    """

    schema_version: str
    name: str
    substrate: str
    parent_issue: int | None
    implementing_issue: int | None
    target_substrate_run: str
    data: ExecutionTrainDataConfig
    loader: ExecutionTrainLoaderConfig
    trainer: ExecutionTrainTrainerConfig
    optimizer: ExecutionTrainOptimizerConfig
    wm: ExecutionTrainWorldModelConfig
    objective: ExecutionTrainObjectiveConfig
    seeds: tuple[int, ...]
    hf_jobs: ExecutionTrainHfJobsConfig
    claim_gates: ExecutionTrainClaimGatesConfig
    claim_boundary: ExecutionTrainClaimBoundaryConfig

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION:
            raise ExecutionTrainConfigError(
                f"schema_version must be {EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )
        if not self.name.strip():
            raise ExecutionTrainConfigError("name must not be empty")
        if self.substrate not in _VALID_SUBSTRATES:
            allowed = ", ".join(sorted(_VALID_SUBSTRATES))
            raise ExecutionTrainConfigError(
                f"substrate must be one of: {allowed}; got {self.substrate!r}"
            )
        if not self.target_substrate_run.strip():
            raise ExecutionTrainConfigError("target_substrate_run must not be empty")
        if not self.seeds:
            raise ExecutionTrainConfigError("seeds must list at least one seed")
        for seed in self.seeds:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ExecutionTrainConfigError(
                    "seeds[] must contain plain integers"
                )
            if seed < 0:
                raise ExecutionTrainConfigError("seeds[] entries must be non-negative")
        # Loader contract.
        if self.loader.code_sequence_length <= 0:
            raise ExecutionTrainConfigError("loader.code_sequence_length must be positive")
        if self.loader.action_sequence_length <= 0:
            raise ExecutionTrainConfigError("loader.action_sequence_length must be positive")
        if self.loader.output_sequence_length <= 0:
            raise ExecutionTrainConfigError("loader.output_sequence_length must be positive")
        if self.loader.batch_size <= 0:
            raise ExecutionTrainConfigError("loader.batch_size must be positive")
        if self.loader.gradient_accumulation_steps <= 0:
            raise ExecutionTrainConfigError(
                "loader.gradient_accumulation_steps must be positive"
            )
        if self.loader.effective_batch_size != (
            self.loader.batch_size * self.loader.gradient_accumulation_steps
        ):
            raise ExecutionTrainConfigError(
                "loader.effective_batch_size must equal "
                "loader.batch_size * loader.gradient_accumulation_steps"
            )
        # Trainer contract.
        if self.trainer.accelerator not in _VALID_ACCELERATORS:
            allowed = ", ".join(sorted(_VALID_ACCELERATORS))
            raise ExecutionTrainConfigError(
                f"trainer.accelerator must be one of: {allowed}"
            )
        if self.trainer.devices <= 0:
            raise ExecutionTrainConfigError("trainer.devices must be positive")
        if self.trainer.precision not in _VALID_PRECISIONS:
            allowed = ", ".join(sorted(_VALID_PRECISIONS))
            raise ExecutionTrainConfigError(
                f"trainer.precision must be one of: {allowed}"
            )
        if self.trainer.max_steps <= 0:
            raise ExecutionTrainConfigError("trainer.max_steps must be positive")
        if self.trainer.warmup_steps < 0:
            raise ExecutionTrainConfigError("trainer.warmup_steps must be non-negative")
        if self.trainer.warmup_steps >= self.trainer.max_steps:
            raise ExecutionTrainConfigError(
                "trainer.warmup_steps must be less than trainer.max_steps"
            )
        if self.trainer.gradient_clip_val < 0.0:
            raise ExecutionTrainConfigError("trainer.gradient_clip_val must be non-negative")
        if self.trainer.checkpoint_every_n_steps <= 0:
            raise ExecutionTrainConfigError(
                "trainer.checkpoint_every_n_steps must be positive"
            )
        if self.trainer.keep_last_n_checkpoints <= 0:
            raise ExecutionTrainConfigError(
                "trainer.keep_last_n_checkpoints must be positive"
            )
        if not self.trainer.keep_best_by_metric.strip():
            raise ExecutionTrainConfigError(
                "trainer.keep_best_by_metric must not be empty"
            )
        if self.trainer.collapse_diagnostics_every_n_steps <= 0:
            raise ExecutionTrainConfigError(
                "trainer.collapse_diagnostics_every_n_steps must be positive"
            )
        if self.trainer.progress_log_every_n_steps <= 0:
            raise ExecutionTrainConfigError(
                "trainer.progress_log_every_n_steps must be positive"
            )
        # Optimizer contract.
        if self.optimizer.name not in {"adamw", "AdamW"}:
            raise ExecutionTrainConfigError(
                "optimizer.name must be 'adamw' (AdamW) for v0.6"
            )
        if self.optimizer.lr <= 0.0:
            raise ExecutionTrainConfigError("optimizer.lr must be positive")
        if self.optimizer.weight_decay < 0.0:
            raise ExecutionTrainConfigError("optimizer.weight_decay must be non-negative")
        if len(self.optimizer.betas) != 2:
            raise ExecutionTrainConfigError("optimizer.betas must be a length-2 sequence")
        for beta in self.optimizer.betas:
            if not (0.0 <= beta < 1.0):
                raise ExecutionTrainConfigError(
                    "optimizer.betas entries must be in [0.0, 1.0)"
                )
        # World-model contract — match the existing JEPA model contract.
        if self.wm.history_size != 1:
            raise ExecutionTrainConfigError(
                "wm.history_size must be 1 (one-step prediction contract)"
            )
        if self.wm.num_preds != 1:
            raise ExecutionTrainConfigError(
                "wm.num_preds must be 1 (one-step prediction contract)"
            )
        if self.wm.embed_dim <= 0:
            raise ExecutionTrainConfigError("wm.embed_dim must be positive")
        if self.wm.state_encoder_type not in {"pool", "transformer"}:
            raise ExecutionTrainConfigError(
                "wm.state_encoder_type must be 'pool' or 'transformer'"
            )
        if self.wm.state_encoder_layers <= 0:
            raise ExecutionTrainConfigError("wm.state_encoder_layers must be positive")
        if (
            self.wm.state_encoder_heads <= 0
            or self.wm.embed_dim % self.wm.state_encoder_heads != 0
        ):
            raise ExecutionTrainConfigError(
                "wm.state_encoder_heads must be positive and divide wm.embed_dim"
            )
        if (
            not math.isfinite(self.wm.ema_target_decay)
            or not 0.0 <= self.wm.ema_target_decay < 1.0
        ):
            raise ExecutionTrainConfigError(
                "wm.ema_target_decay must be finite and in [0.0, 1.0)"
            )
        # Objective contract.
        if self.objective.prediction_mse_weight < 0.0:
            raise ExecutionTrainConfigError(
                "objective.prediction_mse_weight must be non-negative"
            )
        if self.objective.sigreg_weight < 0.0:
            raise ExecutionTrainConfigError("objective.sigreg_weight must be non-negative")
        if self.objective.action_swap_contrastive_weight < 0.0:
            raise ExecutionTrainConfigError(
                "objective.action_swap_contrastive_weight must be non-negative"
            )
        if not 0.0 <= self.objective.retrieval_weight <= 0.10:
            raise ExecutionTrainConfigError(
                "objective.retrieval_weight must be in [0.0, 0.10]"
            )
        if self.objective.inverse_action_reconstruction_weight < 0.0:
            raise ExecutionTrainConfigError(
                "objective.inverse_action_reconstruction_weight must be non-negative"
            )
        if self.objective.p_pass_bce_weight < 0.0:
            raise ExecutionTrainConfigError(
                "objective.p_pass_bce_weight must be non-negative"
            )
        if self.objective.p_pass_bce_pos_weight <= 0.0:
            raise ExecutionTrainConfigError(
                "objective.p_pass_bce_pos_weight must be positive"
            )
        if self.objective.output_value_ce_weight < 0.0:
            raise ExecutionTrainConfigError(
                "objective.output_value_ce_weight must be non-negative"
            )
        # HF Jobs contract.
        if self.hf_jobs.timeout_hours <= 0:
            raise ExecutionTrainConfigError("hf_jobs.timeout_hours must be positive")
        for key, value in (
            ("flavor", self.hf_jobs.flavor),
            ("run_name_template", self.hf_jobs.run_name_template),
            ("artifact_repo_id", self.hf_jobs.artifact_repo_id),
            ("checkpoint_repo_id", self.hf_jobs.checkpoint_repo_id),
            ("checkpoint_revision_template", self.hf_jobs.checkpoint_revision_template),
        ):
            if not value.strip():
                raise ExecutionTrainConfigError(f"hf_jobs.{key} must not be empty")
        # Claim gates: require_seeds and lift gates are floats / non-negative ints.
        if self.claim_gates.required_seeds <= 0:
            raise ExecutionTrainConfigError(
                "claim_gates.required_seeds must be positive"
            )
        # Claim boundary: name + scope non-empty.
        if not self.claim_boundary.name.strip():
            raise ExecutionTrainConfigError("claim_boundary.name must not be empty")
        if not self.claim_boundary.scope.strip():
            raise ExecutionTrainConfigError("claim_boundary.scope must not be empty")
        # Data: required source-mix non-empty.
        for key, value in (
            ("pack_repo_id", self.data.pack_repo_id),
            ("pack_revision", self.data.pack_revision),
            ("pack_jsonl", self.data.pack_jsonl),
            ("manifest_filename", self.data.manifest_filename),
            ("claim_boundary_filename", self.data.claim_boundary_filename),
        ):
            if not value.strip():
                raise ExecutionTrainConfigError(f"data.{key} must not be empty")
        if not self.data.ingestion_sources:
            raise ExecutionTrainConfigError(
                "data.ingestion_sources must list at least one source"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-native dict representation of the config."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "substrate": self.substrate,
            "parent_issue": self.parent_issue,
            "implementing_issue": self.implementing_issue,
            "target_substrate_run": self.target_substrate_run,
            "data": {
                "pack_repo_id": self.data.pack_repo_id,
                "pack_revision": self.data.pack_revision,
                "pack_jsonl": self.data.pack_jsonl,
                "manifest_filename": self.data.manifest_filename,
                "claim_boundary_filename": self.data.claim_boundary_filename,
                "ingestion_sources": list(self.data.ingestion_sources),
                "held_out_for_eval": list(self.data.held_out_for_eval),
            },
            "loader": {
                "code_sequence_length": self.loader.code_sequence_length,
                "action_sequence_length": self.loader.action_sequence_length,
                "output_sequence_length": self.loader.output_sequence_length,
                "batch_size": self.loader.batch_size,
                "gradient_accumulation_steps": self.loader.gradient_accumulation_steps,
                "effective_batch_size": self.loader.effective_batch_size,
                "shuffle": self.loader.shuffle,
            },
            "trainer": {
                "accelerator": self.trainer.accelerator,
                "devices": self.trainer.devices,
                "precision": self.trainer.precision,
                "max_steps": self.trainer.max_steps,
                "warmup_steps": self.trainer.warmup_steps,
                "cosine_decay_to": self.trainer.cosine_decay_to,
                "gradient_clip_val": self.trainer.gradient_clip_val,
                "checkpoint_every_n_steps": self.trainer.checkpoint_every_n_steps,
                "keep_last_n_checkpoints": self.trainer.keep_last_n_checkpoints,
                "keep_best_by_metric": self.trainer.keep_best_by_metric,
                "tensorboard_enabled": self.trainer.tensorboard_enabled,
                "collapse_diagnostics_every_n_steps": (
                    self.trainer.collapse_diagnostics_every_n_steps
                ),
                "progress_log_every_n_steps": (
                    self.trainer.progress_log_every_n_steps
                ),
            },
            "optimizer": {
                "name": self.optimizer.name,
                "lr": self.optimizer.lr,
                "betas": list(self.optimizer.betas),
                "weight_decay": self.optimizer.weight_decay,
            },
            "wm": {
                "history_size": self.wm.history_size,
                "num_preds": self.wm.num_preds,
                "embed_dim": self.wm.embed_dim,
                "state_encoder_type": self.wm.state_encoder_type,
                "state_encoder_layers": self.wm.state_encoder_layers,
                "state_encoder_heads": self.wm.state_encoder_heads,
                "enable_ema_target_encoder": self.wm.enable_ema_target_encoder,
                "ema_target_decay": self.wm.ema_target_decay,
            },
            "objective": {
                "prediction_mse_weight": self.objective.prediction_mse_weight,
                "sigreg_weight": self.objective.sigreg_weight,
                "action_swap_contrastive_weight": (
                    self.objective.action_swap_contrastive_weight
                ),
                "inverse_action_reconstruction_weight": (
                    self.objective.inverse_action_reconstruction_weight
                ),
                "retrieval_weight": self.objective.retrieval_weight,
                "p_pass_bce_weight": self.objective.p_pass_bce_weight,
                "p_pass_bce_pos_weight": self.objective.p_pass_bce_pos_weight,
                "output_value_ce_weight": self.objective.output_value_ce_weight,
            },
            "seeds": list(self.seeds),
            "hf_jobs": {
                "flavor": self.hf_jobs.flavor,
                "region": self.hf_jobs.region,
                "timeout_hours": self.hf_jobs.timeout_hours,
                "run_name_template": self.hf_jobs.run_name_template,
                "artifact_repo_id": self.hf_jobs.artifact_repo_id,
                "checkpoint_repo_id": self.hf_jobs.checkpoint_repo_id,
                "checkpoint_revision_template": (
                    self.hf_jobs.checkpoint_revision_template
                ),
                "runtime_image": self.hf_jobs.runtime_image,
            },
            "claim_gates": {
                "retrieval_min_recall_at_1_lift_over_no_action": (
                    self.claim_gates.retrieval_min_recall_at_1_lift_over_no_action
                ),
                "retrieval_min_mrr_lift_over_no_action": (
                    self.claim_gates.retrieval_min_mrr_lift_over_no_action
                ),
                "collapse_effective_rank_ratio_min": (
                    self.claim_gates.collapse_effective_rank_ratio_min
                ),
                "collapse_per_dim_variance_median_min": (
                    self.claim_gates.collapse_per_dim_variance_median_min
                ),
                "collapse_nearest_neighbor_entropy_min": (
                    self.claim_gates.collapse_nearest_neighbor_entropy_min
                ),
                "surprise_mutation_auc_min": self.claim_gates.surprise_mutation_auc_min,
                "surprise_same_problem_different_submission_auc_min": (
                    self.claim_gates.surprise_same_problem_different_submission_auc_min
                ),
                "surprise_same_code_different_input_auc_min": (
                    self.claim_gates.surprise_same_code_different_input_auc_min
                ),
                "downstream_rerank_pass_at_1_lift_min": (
                    self.claim_gates.downstream_rerank_pass_at_1_lift_min
                ),
                "required_seeds": self.claim_gates.required_seeds,
            },
            "claim_boundary": {
                "name": self.claim_boundary.name,
                "scope": self.claim_boundary.scope,
            },
        }


def load_execution_train_config(path: Path | str) -> ExecutionTrainConfig:
    """Load and validate a v0.6 execution-train config from YAML or JSON.

    Raises :class:`ExecutionTrainConfigError` if the file does not exist,
    is not parseable, or fails any field-level contract. Use
    :func:`peek_train_config_schema_version` first if you only need the
    schema marker without paying for the full validation pass.
    """

    payload = _load_config_payload(Path(path))
    return _from_payload(payload)


def peek_train_config_schema_version(path: Path | str) -> str | None:
    """Return the ``schema_version`` field of a config file, or ``None``.

    Used by the CLI to decide which training contract to apply. The peek
    is intentionally permissive: it returns ``None`` for any file that
    doesn't carry a string ``schema_version`` field, leaving the caller
    to surface the underlying error from the chosen parser.
    """

    config_path = Path(path)
    if not config_path.is_file():
        return None
    try:
        payload = _load_config_payload(config_path)
    except ExecutionTrainConfigError:
        return None
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("schema_version")
    if not isinstance(raw, str) or not raw:
        return None
    return raw


def _load_config_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExecutionTrainConfigError(
            f"execution-train config does not exist: {path}"
        )
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExecutionTrainConfigError(
                f"execution-train config is not valid JSON: {path}: {exc}"
            ) from exc
    elif suffix in {".yaml", ".yml"}:
        # Re-use the launch-plan generator's strict YAML subset so a
        # config can never validate under one parser and fail under the
        # other.
        from .execution_launch_plan import (
            ExecutionLaunchPlanError,
            _load_yaml_via_safe_loader,
        )

        try:
            payload = _load_yaml_via_safe_loader(text)
        except ExecutionLaunchPlanError as exc:
            raise ExecutionTrainConfigError(
                f"execution-train config is not valid YAML: {path}: {exc}"
            ) from exc
    else:
        raise ExecutionTrainConfigError(
            f"execution-train config has unsupported extension: {path.suffix}"
        )
    if not isinstance(payload, Mapping):
        raise ExecutionTrainConfigError(
            f"execution-train config root must be a mapping: {path}"
        )
    return dict(payload)


def _from_payload(payload: Mapping[str, Any]) -> ExecutionTrainConfig:
    _reject_unknown(
        payload,
        {
            "schema_version",
            "name",
            "substrate",
            "parent_issue",
            "implementing_issue",
            "target_substrate_run",
            "data",
            "loader",
            "trainer",
            "optimizer",
            "wm",
            "objective",
            "seeds",
            "hf_jobs",
            "claim_gates",
            "claim_boundary",
        },
        "config",
    )

    data_payload = _require_mapping(payload, "data", "config")
    _reject_unknown(
        data_payload,
        {
            "pack_repo_id",
            "pack_revision",
            "pack_jsonl",
            "manifest_filename",
            "claim_boundary_filename",
            "ingestion_sources",
            "held_out_for_eval",
        },
        "config.data",
    )

    loader_payload = _require_mapping(payload, "loader", "config")
    _reject_unknown(
        loader_payload,
        {
            "code_sequence_length",
            "action_sequence_length",
            "output_sequence_length",
            "batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "shuffle",
        },
        "config.loader",
    )

    trainer_payload = _require_mapping(payload, "trainer", "config")
    _reject_unknown(
        trainer_payload,
        {
            "accelerator",
            "devices",
            "precision",
            "max_steps",
            "warmup_steps",
            "cosine_decay_to",
            "gradient_clip_val",
            "checkpoint_every_n_steps",
            "keep_last_n_checkpoints",
            "keep_best_by_metric",
            "tensorboard_enabled",
            "collapse_diagnostics_every_n_steps",
            "progress_log_every_n_steps",
        },
        "config.trainer",
    )

    optimizer_payload = _require_mapping(payload, "optimizer", "config")
    _reject_unknown(
        optimizer_payload,
        {"name", "lr", "betas", "weight_decay"},
        "config.optimizer",
    )

    wm_payload = _require_mapping(payload, "wm", "config")
    _reject_unknown(
        wm_payload,
        {
            "history_size",
            "num_preds",
            "embed_dim",
            "state_encoder_type",
            "state_encoder_layers",
            "state_encoder_heads",
            "enable_ema_target_encoder",
            "ema_target_decay",
        },
        "config.wm",
    )

    objective_payload = _require_mapping(payload, "objective", "config")
    _reject_unknown(
        objective_payload,
        {
            "prediction_mse_weight",
            "sigreg_weight",
            "action_swap_contrastive_weight",
            "inverse_action_reconstruction_weight",
            "retrieval_weight",
            "p_pass_bce_weight",
            "p_pass_bce_pos_weight",
            "output_value_ce_weight",
        },
        "config.objective",
    )

    hf_jobs_payload = _require_mapping(payload, "hf_jobs", "config")
    _reject_unknown(
        hf_jobs_payload,
        {
            "flavor",
            "region",
            "timeout_hours",
            "run_name_template",
            "artifact_repo_id",
            "checkpoint_repo_id",
            "checkpoint_revision_template",
            "runtime_image",
        },
        "config.hf_jobs",
    )

    claim_gates_payload = _require_mapping(payload, "claim_gates", "config")
    _reject_unknown(
        claim_gates_payload,
        {
            "retrieval_min_recall_at_1_lift_over_no_action",
            "retrieval_min_mrr_lift_over_no_action",
            "collapse_effective_rank_ratio_min",
            "collapse_per_dim_variance_median_min",
            "collapse_nearest_neighbor_entropy_min",
            "surprise_mutation_auc_min",
            "surprise_same_problem_different_submission_auc_min",
            "surprise_same_code_different_input_auc_min",
            "downstream_rerank_pass_at_1_lift_min",
            "required_seeds",
        },
        "config.claim_gates",
    )

    claim_boundary_payload = _require_mapping(payload, "claim_boundary", "config")
    _reject_unknown(
        claim_boundary_payload,
        {"name", "scope"},
        "config.claim_boundary",
    )

    data = ExecutionTrainDataConfig(
        pack_repo_id=_require_string(data_payload, "pack_repo_id", "config.data"),
        pack_revision=_require_string(data_payload, "pack_revision", "config.data"),
        pack_jsonl=_require_string(data_payload, "pack_jsonl", "config.data"),
        manifest_filename=_require_string(
            data_payload, "manifest_filename", "config.data"
        ),
        claim_boundary_filename=_require_string(
            data_payload, "claim_boundary_filename", "config.data"
        ),
        ingestion_sources=tuple(
            _require_string_sequence(data_payload, "ingestion_sources", "config.data")
        ),
        held_out_for_eval=tuple(
            _require_string_sequence(data_payload, "held_out_for_eval", "config.data")
        ),
    )

    loader = ExecutionTrainLoaderConfig(
        code_sequence_length=_require_int(
            loader_payload, "code_sequence_length", "config.loader"
        ),
        action_sequence_length=_require_int(
            loader_payload, "action_sequence_length", "config.loader"
        ),
        output_sequence_length=_require_int(
            loader_payload, "output_sequence_length", "config.loader"
        ),
        batch_size=_require_int(loader_payload, "batch_size", "config.loader"),
        gradient_accumulation_steps=_require_int(
            loader_payload, "gradient_accumulation_steps", "config.loader"
        ),
        effective_batch_size=_require_int(
            loader_payload, "effective_batch_size", "config.loader"
        ),
        shuffle=_require_bool(loader_payload, "shuffle", "config.loader"),
    )

    progress_log_every_n_steps = _optional_int(
        trainer_payload,
        "progress_log_every_n_steps",
        "config.trainer",
    )
    trainer = ExecutionTrainTrainerConfig(
        accelerator=_require_string(trainer_payload, "accelerator", "config.trainer"),
        devices=_require_int(trainer_payload, "devices", "config.trainer"),
        precision=_require_string(trainer_payload, "precision", "config.trainer"),
        max_steps=_require_int(trainer_payload, "max_steps", "config.trainer"),
        warmup_steps=_require_int(trainer_payload, "warmup_steps", "config.trainer"),
        cosine_decay_to=_require_float(
            trainer_payload, "cosine_decay_to", "config.trainer"
        ),
        gradient_clip_val=_require_float(
            trainer_payload, "gradient_clip_val", "config.trainer"
        ),
        checkpoint_every_n_steps=_require_int(
            trainer_payload, "checkpoint_every_n_steps", "config.trainer"
        ),
        keep_last_n_checkpoints=_require_int(
            trainer_payload, "keep_last_n_checkpoints", "config.trainer"
        ),
        keep_best_by_metric=_require_string(
            trainer_payload, "keep_best_by_metric", "config.trainer"
        ),
        tensorboard_enabled=_require_bool(
            trainer_payload, "tensorboard_enabled", "config.trainer"
        ),
        collapse_diagnostics_every_n_steps=_require_int(
            trainer_payload, "collapse_diagnostics_every_n_steps", "config.trainer"
        ),
        progress_log_every_n_steps=(
            100
            if progress_log_every_n_steps is None
            else progress_log_every_n_steps
        ),
    )

    betas_raw = optimizer_payload.get("betas")
    if not isinstance(betas_raw, list) or len(betas_raw) != 2:
        raise ExecutionTrainConfigError(
            "config.optimizer.betas must be a length-2 list of numbers"
        )
    betas = tuple(float(b) for b in betas_raw)
    optimizer = ExecutionTrainOptimizerConfig(
        name=_require_string(optimizer_payload, "name", "config.optimizer"),
        lr=_require_float(optimizer_payload, "lr", "config.optimizer"),
        betas=betas,  # type: ignore[arg-type]
        weight_decay=_require_float(
            optimizer_payload, "weight_decay", "config.optimizer"
        ),
    )

    ema_target_decay = _optional_float(
        wm_payload, "ema_target_decay", "config.wm"
    )
    wm = ExecutionTrainWorldModelConfig(
        history_size=_require_int(wm_payload, "history_size", "config.wm"),
        num_preds=_require_int(wm_payload, "num_preds", "config.wm"),
        embed_dim=_require_int(wm_payload, "embed_dim", "config.wm"),
        state_encoder_type=_optional_string(
            wm_payload, "state_encoder_type", "config.wm"
        )
        or "pool",
        state_encoder_layers=_optional_int(
            wm_payload, "state_encoder_layers", "config.wm"
        )
        or 4,
        state_encoder_heads=_optional_int(
            wm_payload, "state_encoder_heads", "config.wm"
        )
        or 8,
        enable_ema_target_encoder=_optional_bool(
            wm_payload, "enable_ema_target_encoder", "config.wm"
        ),
        ema_target_decay=0.99 if ema_target_decay is None else ema_target_decay,
    )

    p_pass_bce_pos_weight = _optional_float(
        objective_payload, "p_pass_bce_pos_weight", "config.objective"
    )
    objective = ExecutionTrainObjectiveConfig(
        prediction_mse_weight=_require_float(
            objective_payload, "prediction_mse_weight", "config.objective"
        ),
        sigreg_weight=_require_float(
            objective_payload, "sigreg_weight", "config.objective"
        ),
        action_swap_contrastive_weight=_require_float(
            objective_payload, "action_swap_contrastive_weight", "config.objective"
        ),
        inverse_action_reconstruction_weight=_require_float(
            objective_payload, "inverse_action_reconstruction_weight", "config.objective"
        ),
        retrieval_weight=_optional_float(
            objective_payload, "retrieval_weight", "config.objective"
        )
        or 0.0,
        p_pass_bce_weight=_optional_float(
            objective_payload, "p_pass_bce_weight", "config.objective"
        )
        or 0.0,
        p_pass_bce_pos_weight=(
            1.0 if p_pass_bce_pos_weight is None else p_pass_bce_pos_weight
        ),
        output_value_ce_weight=_optional_float(
            objective_payload, "output_value_ce_weight", "config.objective"
        )
        or 0.0,
    )

    hf_jobs = ExecutionTrainHfJobsConfig(
        flavor=_require_string(hf_jobs_payload, "flavor", "config.hf_jobs"),
        region=_optional_string(hf_jobs_payload, "region", "config.hf_jobs"),
        timeout_hours=_require_int(
            hf_jobs_payload, "timeout_hours", "config.hf_jobs"
        ),
        run_name_template=_require_string(
            hf_jobs_payload, "run_name_template", "config.hf_jobs"
        ),
        artifact_repo_id=_require_string(
            hf_jobs_payload, "artifact_repo_id", "config.hf_jobs"
        ),
        checkpoint_repo_id=_require_string(
            hf_jobs_payload, "checkpoint_repo_id", "config.hf_jobs"
        ),
        checkpoint_revision_template=_require_string(
            hf_jobs_payload, "checkpoint_revision_template", "config.hf_jobs"
        ),
        runtime_image=_optional_string(
            hf_jobs_payload, "runtime_image", "config.hf_jobs"
        ),
    )

    claim_gates = ExecutionTrainClaimGatesConfig(
        retrieval_min_recall_at_1_lift_over_no_action=_require_float(
            claim_gates_payload,
            "retrieval_min_recall_at_1_lift_over_no_action",
            "config.claim_gates",
        ),
        retrieval_min_mrr_lift_over_no_action=_require_float(
            claim_gates_payload,
            "retrieval_min_mrr_lift_over_no_action",
            "config.claim_gates",
        ),
        collapse_effective_rank_ratio_min=_require_float(
            claim_gates_payload,
            "collapse_effective_rank_ratio_min",
            "config.claim_gates",
        ),
        collapse_per_dim_variance_median_min=_require_float(
            claim_gates_payload,
            "collapse_per_dim_variance_median_min",
            "config.claim_gates",
        ),
        collapse_nearest_neighbor_entropy_min=_require_float(
            claim_gates_payload,
            "collapse_nearest_neighbor_entropy_min",
            "config.claim_gates",
        ),
        surprise_mutation_auc_min=_require_float(
            claim_gates_payload,
            "surprise_mutation_auc_min",
            "config.claim_gates",
        ),
        surprise_same_problem_different_submission_auc_min=_require_float(
            claim_gates_payload,
            "surprise_same_problem_different_submission_auc_min",
            "config.claim_gates",
        ),
        surprise_same_code_different_input_auc_min=_require_float(
            claim_gates_payload,
            "surprise_same_code_different_input_auc_min",
            "config.claim_gates",
        ),
        downstream_rerank_pass_at_1_lift_min=_require_float(
            claim_gates_payload,
            "downstream_rerank_pass_at_1_lift_min",
            "config.claim_gates",
        ),
        required_seeds=_require_int(
            claim_gates_payload, "required_seeds", "config.claim_gates"
        ),
    )

    claim_boundary = ExecutionTrainClaimBoundaryConfig(
        name=_require_string(claim_boundary_payload, "name", "config.claim_boundary"),
        scope=_require_string(claim_boundary_payload, "scope", "config.claim_boundary"),
    )

    seeds_raw = payload.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ExecutionTrainConfigError("config.seeds must be a non-empty list")
    for index, seed in enumerate(seeds_raw):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ExecutionTrainConfigError(
                f"config.seeds[{index}] must be an integer; got "
                f"{type(seed).__name__}: {seed!r}"
            )

    return ExecutionTrainConfig(
        schema_version=_require_string(payload, "schema_version", "config"),
        name=_require_string(payload, "name", "config"),
        substrate=_require_string(payload, "substrate", "config"),
        parent_issue=_optional_int(payload, "parent_issue", "config"),
        implementing_issue=_optional_int(payload, "implementing_issue", "config"),
        target_substrate_run=_require_string(
            payload, "target_substrate_run", "config"
        ),
        data=data,
        loader=loader,
        trainer=trainer,
        optimizer=optimizer,
        wm=wm,
        objective=objective,
        seeds=tuple(int(s) for s in seeds_raw),
        hf_jobs=hf_jobs,
        claim_gates=claim_gates,
        claim_boundary=claim_boundary,
    )


# -- mapping helpers (mirror the legacy parser's strictness) --------------


def _reject_unknown(
    payload: Mapping[str, Any], allowed: set[str], section: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ExecutionTrainConfigError(
            f"{section} contains unknown key(s): {joined}"
        )


def _require_mapping(
    payload: Mapping[str, Any], key: str, section: str
) -> Mapping[str, Any]:
    if key not in payload:
        raise ExecutionTrainConfigError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, Mapping):
        raise ExecutionTrainConfigError(f"{section}.{key} must be a mapping")
    return value


def _require_string(
    payload: Mapping[str, Any], key: str, section: str
) -> str:
    if key not in payload:
        raise ExecutionTrainConfigError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise ExecutionTrainConfigError(f"{section}.{key} must be a string")
    return value


def _optional_string(
    payload: Mapping[str, Any], key: str, section: str
) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExecutionTrainConfigError(f"{section}.{key} must be a string")
    return value


def _require_int(payload: Mapping[str, Any], key: str, section: str) -> int:
    if key not in payload:
        raise ExecutionTrainConfigError(f"{section}.{key} is required")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionTrainConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_int(
    payload: Mapping[str, Any], key: str, section: str
) -> int | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionTrainConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_float(
    payload: Mapping[str, Any], key: str, section: str
) -> float | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutionTrainConfigError(f"{section}.{key} must be a number")
    return float(value)


def _require_float(
    payload: Mapping[str, Any], key: str, section: str
) -> float:
    if key not in payload:
        raise ExecutionTrainConfigError(f"{section}.{key} is required")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutionTrainConfigError(f"{section}.{key} must be numeric")
    return float(value)


def _require_bool(payload: Mapping[str, Any], key: str, section: str) -> bool:
    if key not in payload:
        raise ExecutionTrainConfigError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, bool):
        raise ExecutionTrainConfigError(f"{section}.{key} must be true or false")
    return value


def _optional_bool(
    payload: Mapping[str, Any], key: str, section: str, *, default: bool = False
) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise ExecutionTrainConfigError(f"{section}.{key} must be true or false")
    return value


def _require_string_sequence(
    payload: Mapping[str, Any], key: str, section: str
) -> tuple[str, ...]:
    if key not in payload:
        raise ExecutionTrainConfigError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, list):
        raise ExecutionTrainConfigError(f"{section}.{key} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ExecutionTrainConfigError(
                f"{section}.{key}[{index}] must be a string"
            )
        result.append(item)
    return tuple(result)


__all__ = [
    "EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION",
    "ExecutionTrainClaimBoundaryConfig",
    "ExecutionTrainClaimGatesConfig",
    "ExecutionTrainConfig",
    "ExecutionTrainConfigError",
    "ExecutionTrainDataConfig",
    "ExecutionTrainHfJobsConfig",
    "ExecutionTrainLoaderConfig",
    "ExecutionTrainObjectiveConfig",
    "ExecutionTrainOptimizerConfig",
    "ExecutionTrainTrainerConfig",
    "ExecutionTrainWorldModelConfig",
    "load_execution_train_config",
    "peek_train_config_schema_version",
]
