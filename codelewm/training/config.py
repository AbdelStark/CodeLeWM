"""Training configuration contracts for CodeLeWM."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from codelewm.model import (
    LATENT_DIM,
    STATE_SEQUENCE_LENGTH,
    TEXT_ACTION_SEQUENCE_LENGTH,
    ObjectiveConfig,
    expected_action_sequence_length,
)


TRAIN_CONFIG_SCHEMA_VERSION = "codelewm.train_config.v1"
DEFAULT_TINY_TRAIN_CONFIG = Path("config/train/codelewm_tiny.yaml")
DEFAULT_SMALL_TRAIN_CONFIG = Path("config/train/codelewm_small.yaml")

_ACTION_VIEWS = frozenset({"text", "abstract"})
_PRECISIONS = frozenset({"float32", "32-true", "bf16-mixed", "16-mixed", "fp16-mixed"})
_ACCELERATORS = frozenset({"auto", "cpu", "gpu", "cuda", "mps"})
_FORBIDDEN_DATASET_TOKENS = frozenset({"pusht", "dmc", "tworoom", "ogbench", "pixels", "proprio"})


class TrainConfigError(ValueError):
    """Raised when a CodeLeWM training config violates the public schema."""


@dataclass(frozen=True)
class TrainDataConfig:
    train: str
    val: str
    manifest: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainDataConfig":
        _reject_unknown(payload, {"train", "val", "manifest"}, "data")
        return cls(
            train=_require_string(payload, "train", "data"),
            val=_require_string(payload, "val", "data"),
            manifest=_optional_string(payload, "manifest", "data"),
        )

    def __post_init__(self) -> None:
        for field_name, value in (("train", self.train), ("val", self.val)):
            if not value.strip():
                raise TrainConfigError(f"data.{field_name} must not be empty")
        if self.manifest is not None and not self.manifest.strip():
            raise TrainConfigError("data.manifest must not be empty when set")
        _reject_image_control_path("data.train", self.train)
        _reject_image_control_path("data.val", self.val)
        if self.manifest is not None:
            _reject_image_control_path("data.manifest", self.manifest)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"train": self.train, "val": self.val}
        if self.manifest is not None:
            payload["manifest"] = self.manifest
        return payload


@dataclass(frozen=True)
class WorldModelTrainConfig:
    history_size: int = 1
    num_preds: int = 1
    embed_dim: int = LATENT_DIM
    action_view: str = "text"
    state_sequence_length: int = STATE_SEQUENCE_LENGTH
    action_sequence_length: int = TEXT_ACTION_SEQUENCE_LENGTH

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WorldModelTrainConfig":
        _reject_unknown(
            payload,
            {
                "history_size",
                "num_preds",
                "embed_dim",
                "action_view",
                "state_sequence_length",
                "action_sequence_length",
            },
            "wm",
        )
        action_view = _optional_string(payload, "action_view", "wm")
        if action_view is None:
            action_view = "text"
        try:
            default_action_sequence_length = expected_action_sequence_length(action_view)
        except ValueError as exc:
            raise TrainConfigError(str(exc)) from exc
        return cls(
            history_size=_optional_int(payload, "history_size", "wm", default=1),
            num_preds=_optional_int(payload, "num_preds", "wm", default=1),
            embed_dim=_optional_int(payload, "embed_dim", "wm", default=LATENT_DIM),
            action_view=action_view,
            state_sequence_length=_optional_int(
                payload,
                "state_sequence_length",
                "wm",
                default=STATE_SEQUENCE_LENGTH,
            ),
            action_sequence_length=_optional_int(
                payload,
                "action_sequence_length",
                "wm",
                default=default_action_sequence_length,
            ),
        )

    def __post_init__(self) -> None:
        if self.history_size != 1:
            raise TrainConfigError("wm.history_size must be 1 for the v0.1 one-step contract")
        if self.num_preds != 1:
            raise TrainConfigError("wm.num_preds must be 1 for the v0.1 one-step contract")
        if self.embed_dim != LATENT_DIM:
            raise TrainConfigError(f"wm.embed_dim must be {LATENT_DIM}")
        if self.action_view not in _ACTION_VIEWS:
            raise TrainConfigError("wm.action_view must be 'text' or 'abstract'; patch is diagnostic only")
        if self.state_sequence_length != STATE_SEQUENCE_LENGTH:
            raise TrainConfigError(f"wm.state_sequence_length must be {STATE_SEQUENCE_LENGTH}")
        expected_length = expected_action_sequence_length(self.action_view)
        if self.action_sequence_length != expected_length:
            raise TrainConfigError(
                f"wm.action_sequence_length must be {expected_length} for action_view={self.action_view!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_size": self.history_size,
            "num_preds": self.num_preds,
            "embed_dim": self.embed_dim,
            "action_view": self.action_view,
            "state_sequence_length": self.state_sequence_length,
            "action_sequence_length": self.action_sequence_length,
        }


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int
    accelerator: str = "auto"
    devices: int | str = 1
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainerConfig":
        _reject_unknown(
            payload,
            {"max_steps", "accelerator", "devices", "precision", "gradient_clip_val"},
            "trainer",
        )
        accelerator = _optional_string(payload, "accelerator", "trainer")
        precision = _optional_string(payload, "precision", "trainer")
        return cls(
            max_steps=_require_int(payload, "max_steps", "trainer"),
            accelerator="auto" if accelerator is None else accelerator,
            devices=_optional_device(payload, "devices", "trainer", default=1),
            precision="bf16-mixed" if precision is None else precision,
            gradient_clip_val=_optional_float(payload, "gradient_clip_val", "trainer", default=1.0),
        )

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise TrainConfigError("trainer.max_steps must be positive")
        if self.accelerator not in _ACCELERATORS:
            allowed = ", ".join(sorted(_ACCELERATORS))
            raise TrainConfigError(f"trainer.accelerator must be one of: {allowed}")
        if isinstance(self.devices, int):
            if self.devices <= 0:
                raise TrainConfigError("trainer.devices must be positive")
        elif self.devices != "auto":
            raise TrainConfigError("trainer.devices must be a positive integer or 'auto'")
        if self.precision not in _PRECISIONS:
            allowed = ", ".join(sorted(_PRECISIONS))
            raise TrainConfigError(f"trainer.precision must be one of: {allowed}")
        if not math.isfinite(self.gradient_clip_val) or self.gradient_clip_val < 0.0:
            raise TrainConfigError("trainer.gradient_clip_val must be a finite non-negative value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "accelerator": self.accelerator,
            "devices": self.devices,
            "precision": self.precision,
            "gradient_clip_val": self.gradient_clip_val,
        }


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LoaderConfig":
        _reject_unknown(
            payload,
            {"batch_size", "num_workers", "shuffle", "pin_memory", "persistent_workers"},
            "loader",
        )
        return cls(
            batch_size=_require_int(payload, "batch_size", "loader"),
            num_workers=_optional_int(payload, "num_workers", "loader", default=0),
            shuffle=_optional_bool(payload, "shuffle", "loader", default=True),
            pin_memory=_optional_bool(payload, "pin_memory", "loader", default=False),
            persistent_workers=_optional_bool(payload, "persistent_workers", "loader", default=False),
        )

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise TrainConfigError("loader.batch_size must be positive")
        if self.num_workers < 0:
            raise TrainConfigError("loader.num_workers must be non-negative")
        if self.persistent_workers and self.num_workers == 0:
            raise TrainConfigError("loader.persistent_workers requires loader.num_workers > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }


@dataclass(frozen=True)
class OptimizerConfig:
    type: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-3

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OptimizerConfig":
        _reject_unknown(payload, {"type", "lr", "weight_decay"}, "optimizer")
        optimizer_type = _optional_string(payload, "type", "optimizer")
        return cls(
            type="AdamW" if optimizer_type is None else optimizer_type,
            lr=_optional_float(payload, "lr", "optimizer", default=1e-4),
            weight_decay=_optional_float(payload, "weight_decay", "optimizer", default=1e-3),
        )

    def __post_init__(self) -> None:
        if self.type != "AdamW":
            raise TrainConfigError("optimizer.type must be AdamW for the default CodeLeWM configs")
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise TrainConfigError("optimizer.lr must be a finite positive value")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise TrainConfigError("optimizer.weight_decay must be a finite non-negative value")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "lr": self.lr, "weight_decay": self.weight_decay}


@dataclass(frozen=True)
class TrainingLossConfig:
    sigreg_weight: float = 0.09
    enable_retrieval_loss: bool = False
    retrieval_weight: float = 0.0
    retrieval_temperature: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainingLossConfig":
        _reject_unknown(
            payload,
            {
                "sigreg_weight",
                "enable_retrieval_loss",
                "retrieval_weight",
                "retrieval_temperature",
                "sigreg_knots",
                "sigreg_num_proj",
            },
            "loss",
        )
        return cls(
            sigreg_weight=_optional_float(payload, "sigreg_weight", "loss", default=0.09),
            enable_retrieval_loss=_optional_bool(payload, "enable_retrieval_loss", "loss", default=False),
            retrieval_weight=_optional_float(payload, "retrieval_weight", "loss", default=0.0),
            retrieval_temperature=_optional_float(payload, "retrieval_temperature", "loss", default=0.1),
            sigreg_knots=_optional_int(payload, "sigreg_knots", "loss", default=17),
            sigreg_num_proj=_optional_int(payload, "sigreg_num_proj", "loss", default=1024),
        )

    def __post_init__(self) -> None:
        try:
            ObjectiveConfig(
                sigreg_weight=self.sigreg_weight,
                enable_retrieval_loss=self.enable_retrieval_loss,
                retrieval_weight=self.retrieval_weight,
                retrieval_temperature=self.retrieval_temperature,
                sigreg_knots=self.sigreg_knots,
                sigreg_num_proj=self.sigreg_num_proj,
            )
        except ValueError as exc:
            raise TrainConfigError(f"loss config is invalid: {exc}") from exc

    def to_objective_config(self) -> ObjectiveConfig:
        return ObjectiveConfig(
            sigreg_weight=self.sigreg_weight,
            enable_retrieval_loss=self.enable_retrieval_loss,
            retrieval_weight=self.retrieval_weight,
            retrieval_temperature=self.retrieval_temperature,
            sigreg_knots=self.sigreg_knots,
            sigreg_num_proj=self.sigreg_num_proj,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigreg_weight": self.sigreg_weight,
            "enable_retrieval_loss": self.enable_retrieval_loss,
            "retrieval_weight": self.retrieval_weight,
            "retrieval_temperature": self.retrieval_temperature,
            "sigreg_knots": self.sigreg_knots,
            "sigreg_num_proj": self.sigreg_num_proj,
        }


@dataclass(frozen=True)
class OutputConfig:
    run_dir: str
    checkpoint_dir: str
    metrics_path: str
    manifest_path: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OutputConfig":
        _reject_unknown(payload, {"run_dir", "checkpoint_dir", "metrics_path", "manifest_path"}, "output")
        return cls(
            run_dir=_require_string(payload, "run_dir", "output"),
            checkpoint_dir=_require_string(payload, "checkpoint_dir", "output"),
            metrics_path=_require_string(payload, "metrics_path", "output"),
            manifest_path=_require_string(payload, "manifest_path", "output"),
        )

    def __post_init__(self) -> None:
        for field_name, value in (
            ("run_dir", self.run_dir),
            ("checkpoint_dir", self.checkpoint_dir),
            ("metrics_path", self.metrics_path),
            ("manifest_path", self.manifest_path),
        ):
            if not value.strip():
                raise TrainConfigError(f"output.{field_name} must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "metrics_path": self.metrics_path,
            "manifest_path": self.manifest_path,
        }


@dataclass(frozen=True)
class TrainConfig:
    schema_version: str
    name: str
    seed: int
    data: TrainDataConfig
    wm: WorldModelTrainConfig
    trainer: TrainerConfig
    loader: LoaderConfig
    optimizer: OptimizerConfig
    loss: TrainingLossConfig
    output: OutputConfig

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainConfig":
        _reject_unknown(
            payload,
            {
                "schema_version",
                "name",
                "seed",
                "data",
                "wm",
                "trainer",
                "loader",
                "optimizer",
                "loss",
                "output",
            },
            "train config",
        )
        config = cls(
            schema_version=_require_string(payload, "schema_version", "train config"),
            name=_require_string(payload, "name", "train config"),
            seed=_require_int(payload, "seed", "train config"),
            data=TrainDataConfig.from_mapping(_require_mapping(payload, "data", "train config")),
            wm=WorldModelTrainConfig.from_mapping(_require_mapping(payload, "wm", "train config")),
            trainer=TrainerConfig.from_mapping(_require_mapping(payload, "trainer", "train config")),
            loader=LoaderConfig.from_mapping(_require_mapping(payload, "loader", "train config")),
            optimizer=OptimizerConfig.from_mapping(_require_mapping(payload, "optimizer", "train config")),
            loss=TrainingLossConfig.from_mapping(_require_mapping(payload, "loss", "train config")),
            output=OutputConfig.from_mapping(_require_mapping(payload, "output", "train config")),
        )
        return validate_train_config(config)

    def __post_init__(self) -> None:
        if self.schema_version != TRAIN_CONFIG_SCHEMA_VERSION:
            raise TrainConfigError(
                f"schema_version must be {TRAIN_CONFIG_SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if not self.name.strip():
            raise TrainConfigError("name must not be empty")
        if self.seed < 0:
            raise TrainConfigError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "seed": self.seed,
            "data": self.data.to_dict(),
            "wm": self.wm.to_dict(),
            "trainer": self.trainer.to_dict(),
            "loader": self.loader.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "loss": self.loss.to_dict(),
            "output": self.output.to_dict(),
        }


def default_train_config_paths(root: Path | str = ".") -> tuple[Path, Path]:
    """Return repository-relative paths for the default CodeLeWM train configs."""

    base = Path(root)
    return base / DEFAULT_TINY_TRAIN_CONFIG, base / DEFAULT_SMALL_TRAIN_CONFIG


def load_train_config(path: Path | str) -> TrainConfig:
    """Load and validate a CodeLeWM train config from JSON or YAML."""

    path = Path(path)
    payload = _load_config_mapping(path)
    return TrainConfig.from_mapping(payload)


def load_default_train_configs(root: Path | str = ".") -> tuple[TrainConfig, TrainConfig]:
    """Load the tiny and small default CodeLeWM train configs."""

    tiny_path, small_path = default_train_config_paths(root)
    return load_train_config(tiny_path), load_train_config(small_path)


def validate_train_config(config: TrainConfig | Mapping[str, Any]) -> TrainConfig:
    """Validate and normalize a train config object."""

    if isinstance(config, TrainConfig):
        _ensure_json_native(config.to_dict())
        return config
    if isinstance(config, Mapping):
        return TrainConfig.from_mapping(config)
    raise TrainConfigError(f"expected TrainConfig or mapping, got {type(config).__name__}")


def _load_config_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise TrainConfigError(f"training config does not exist: {path}")
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif path.suffix in {".yaml", ".yml"}:
        payload = _load_yaml_mapping(path)
    else:
        raise TrainConfigError(f"unsupported training config extension: {path.suffix}")
    if not isinstance(payload, Mapping):
        raise TrainConfigError(f"training config root must be a mapping: {path}")
    return payload


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return _load_strict_yaml_subset(path)

    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # pragma: no cover - depends on optional runtime.
        raise TrainConfigError(f"failed to load training config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TrainConfigError(f"training config root must be a mapping: {path}")
    return payload


def _load_strict_yaml_subset(path: Path) -> Mapping[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "\t" in raw_line:
            raise TrainConfigError(f"{path}:{lineno}: tabs are not supported in training configs")
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise TrainConfigError(f"{path}:{lineno}: indentation must use two-space levels")
        body = line.strip()
        if ":" not in body:
            raise TrainConfigError(f"{path}:{lineno}: expected 'key: value' mapping entry")
        while indent <= stack[-1][0]:
            stack.pop()
        expected_indent = 0 if stack[-1][0] == -1 else stack[-1][0] + 2
        if indent != expected_indent:
            raise TrainConfigError(f"{path}:{lineno}: indentation must nest under a mapping key")
        key, raw_value = body.split(":", 1)
        key = key.strip()
        if not key:
            raise TrainConfigError(f"{path}:{lineno}: mapping key must not be empty")
        parent = stack[-1][1]
        if key in parent:
            raise TrainConfigError(f"{path}:{lineno}: duplicate key {key!r}")
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        parent[key] = _parse_yaml_scalar(value, path=path, lineno=lineno)
    return root


def _strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None:
            return line[:index]
    return line


def _parse_yaml_scalar(value: str, *, path: Path, lineno: int) -> Any:
    if value.startswith(("[", "{")):
        raise TrainConfigError(f"{path}:{lineno}: lists and inline mappings are not supported")
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if normalized in {"null", "none", "~"}:
        return None
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?((\d+\.\d*)|(\.\d+)|(\d+e[+-]?\d+)|(\d+\.\d*e[+-]?\d+))", normalized):
        return float(value)
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise TrainConfigError(f"{section} contains unknown key(s): {joined}")


def _require_mapping(payload: Mapping[str, Any], key: str, section: str) -> Mapping[str, Any]:
    if key not in payload:
        raise TrainConfigError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, Mapping):
        raise TrainConfigError(f"{section}.{key} must be a mapping")
    return value


def _require_string(payload: Mapping[str, Any], key: str, section: str) -> str:
    if key not in payload:
        raise TrainConfigError(f"{section}.{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise TrainConfigError(f"{section}.{key} must be a string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str, section: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise TrainConfigError(f"{section}.{key} must be a string")
    return value


def _require_int(payload: Mapping[str, Any], key: str, section: str) -> int:
    if key not in payload:
        raise TrainConfigError(f"{section}.{key} is required")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_int(payload: Mapping[str, Any], key: str, section: str, *, default: int) -> int:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_device(payload: Mapping[str, Any], key: str, section: str, *, default: int | str) -> int | str:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        raise TrainConfigError(f"{section}.{key} must be a positive integer or 'auto'")
    if isinstance(value, int):
        return value
    if value == "auto":
        return value
    raise TrainConfigError(f"{section}.{key} must be a positive integer or 'auto'")


def _optional_float(payload: Mapping[str, Any], key: str, section: str, *, default: float) -> float:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainConfigError(f"{section}.{key} must be numeric")
    return float(value)


def _optional_bool(payload: Mapping[str, Any], key: str, section: str, *, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise TrainConfigError(f"{section}.{key} must be true or false")
    return value


def _reject_image_control_path(field_name: str, value: str) -> None:
    normalized = value.lower()
    for token in sorted(_FORBIDDEN_DATASET_TOKENS):
        if token in normalized:
            raise TrainConfigError(f"{field_name} must not reference image-control dataset token {token!r}")


def _ensure_json_native(payload: Any) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TrainConfigError(f"training config must be JSON-native: {exc}") from exc
