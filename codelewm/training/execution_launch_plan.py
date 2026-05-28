"""Launch-plan generator for the v0.6 execution-substrate HF Jobs run.

The launcher is intentionally a *plan* generator, not a live launcher.
It reads the v0.6 config, validates the required fields, and emits one
launch plan per configured seed. An operator then runs the plans via
``hf jobs run`` (or via the existing ``scripts/hf-launch-codelewm-job``
shell pipeline).

The plan is JSON-serializable so CI can dry-run it and assert the
shape; live execution is operator-driven.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXECUTION_LAUNCH_PLAN_SCHEMA_VERSION = "codelewm.execution_launch_plan.v1"


_REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "name",
    "substrate",
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
)
_REQUIRED_DATA_KEYS = (
    "pack_repo_id",
    "pack_revision",
    "pack_jsonl",
    "manifest_filename",
    "claim_boundary_filename",
    "ingestion_sources",
    "held_out_for_eval",
)
_REQUIRED_HF_JOBS_KEYS = (
    "flavor",
    "timeout_hours",
    "run_name_template",
    "artifact_repo_id",
    "checkpoint_repo_id",
    "checkpoint_revision_template",
)
_REQUIRED_OBJECTIVE_KEYS = (
    "prediction_mse_weight",
    "sigreg_weight",
    "action_swap_contrastive_weight",
    "inverse_action_reconstruction_weight",
)


class ExecutionLaunchPlanError(ValueError):
    """Raised when a v0.6 config does not satisfy the launch contract."""


@dataclass(frozen=True)
class LaunchPlan:
    """One seed's launch plan, ready for the operator to fire."""

    schema_version: str
    seed: int
    run_name: str
    config_path: str
    pack_repo_id: str
    pack_revision: str
    flavor: str
    timeout_hours: int
    artifact_repo_id: str
    checkpoint_repo_id: str
    checkpoint_revision: str
    objective: dict[str, float]
    loader: dict[str, Any]
    trainer: dict[str, Any]
    optimizer: dict[str, Any]
    claim_gates: dict[str, float]
    claim_boundary: dict[str, str]
    command: tuple[str, ...]
    issued_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "run_name": self.run_name,
            "config_path": self.config_path,
            "pack_repo_id": self.pack_repo_id,
            "pack_revision": self.pack_revision,
            "flavor": self.flavor,
            "timeout_hours": self.timeout_hours,
            "artifact_repo_id": self.artifact_repo_id,
            "checkpoint_repo_id": self.checkpoint_repo_id,
            "checkpoint_revision": self.checkpoint_revision,
            "objective": dict(self.objective),
            "loader": dict(self.loader),
            "trainer": dict(self.trainer),
            "optimizer": dict(self.optimizer),
            "claim_gates": dict(self.claim_gates),
            "claim_boundary": dict(self.claim_boundary),
            "command": list(self.command),
            "issued_at": self.issued_at,
        }


def load_v0_6_config(path: Path) -> dict[str, Any]:
    """Read the v0.6 YAML or JSON config and validate required keys."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        config = _load_yaml_via_safe_loader(text)
    else:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise ExecutionLaunchPlanError(
            f"config root must be a mapping, got {type(config).__name__}"
        )
    _require_keys(config, _REQUIRED_TOP_LEVEL_KEYS, "config")
    _require_keys(config["data"], _REQUIRED_DATA_KEYS, "config.data")
    _require_keys(config["hf_jobs"], _REQUIRED_HF_JOBS_KEYS, "config.hf_jobs")
    _require_keys(
        config["objective"], _REQUIRED_OBJECTIVE_KEYS, "config.objective"
    )
    if config["schema_version"] != "codelewm.execution_train_config.v1":
        raise ExecutionLaunchPlanError(
            f"unsupported schema_version: {config['schema_version']!r}"
        )
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ExecutionLaunchPlanError("config.seeds must be a non-empty list")
    return config


def build_launch_plans(
    *,
    config: dict[str, Any],
    config_path: Path,
    git_sha: str = "unset",
    date: str | None = None,
) -> tuple[LaunchPlan, ...]:
    """Build one :class:`LaunchPlan` per seed in the config.

    Re-validates the config so callers that hand in a dict directly
    (rather than via :func:`load_v0_6_config`) cannot bypass the schema
    contract.
    """

    _require_keys(config, _REQUIRED_TOP_LEVEL_KEYS, "config")
    _require_keys(config["data"], _REQUIRED_DATA_KEYS, "config.data")
    _require_keys(config["hf_jobs"], _REQUIRED_HF_JOBS_KEYS, "config.hf_jobs")
    _require_keys(
        config["objective"], _REQUIRED_OBJECTIVE_KEYS, "config.objective"
    )
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ExecutionLaunchPlanError("config.seeds must be a non-empty list")
    date_str = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    issued_at = datetime.now(timezone.utc).isoformat()
    name_template: str = config["hf_jobs"]["run_name_template"]
    rev_template: str = config["hf_jobs"]["checkpoint_revision_template"]
    plans: list[LaunchPlan] = []
    for seed in config["seeds"]:
        if not isinstance(seed, int):
            raise ExecutionLaunchPlanError(
                f"seed must be int, got {type(seed).__name__}: {seed!r}"
            )
        run_name = name_template.format(date=date_str, sha=git_sha, seed=seed)
        checkpoint_revision = rev_template.format(seed=seed)
        command = (
            "hf",
            "jobs",
            "run",
            "--flavor",
            str(config["hf_jobs"]["flavor"]),
            "--timeout",
            f"{int(config['hf_jobs']['timeout_hours'])}h",
            "--env",
            f"CODELEWM_HF_RUN_NAME={run_name}",
            "--env",
            f"CODELEWM_EXECUTION_PACK_REPO_ID={config['data']['pack_repo_id']}",
            "--env",
            f"CODELEWM_EXECUTION_PACK_REVISION={config['data']['pack_revision']}",
            "--env",
            f"CODELEWM_TRAIN_SEED={seed}",
            "--env",
            f"CODELEWM_TRAIN_CONFIG={config_path}",
            "abdelstark/codelewm-runtime:v0.6",
            "uv",
            "run",
            "codelewm",
            "train",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
        )
        plans.append(
            LaunchPlan(
                schema_version=EXECUTION_LAUNCH_PLAN_SCHEMA_VERSION,
                seed=seed,
                run_name=run_name,
                config_path=str(config_path),
                pack_repo_id=str(config["data"]["pack_repo_id"]),
                pack_revision=str(config["data"]["pack_revision"]),
                flavor=str(config["hf_jobs"]["flavor"]),
                timeout_hours=int(config["hf_jobs"]["timeout_hours"]),
                artifact_repo_id=str(config["hf_jobs"]["artifact_repo_id"]),
                checkpoint_repo_id=str(config["hf_jobs"]["checkpoint_repo_id"]),
                checkpoint_revision=checkpoint_revision,
                objective=dict(config["objective"]),
                loader=dict(config["loader"]),
                trainer=dict(config["trainer"]),
                optimizer=dict(config["optimizer"]),
                claim_gates=dict(config["claim_gates"]),
                claim_boundary=dict(config["claim_boundary"]),
                command=command,
                issued_at=issued_at,
            )
        )
    return tuple(plans)


def _require_keys(payload: Any, keys: tuple[str, ...], where: str) -> None:
    if not isinstance(payload, dict):
        raise ExecutionLaunchPlanError(
            f"{where} must be a mapping, got {type(payload).__name__}"
        )
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ExecutionLaunchPlanError(
            f"{where} is missing required key(s): {missing}"
        )


def _load_yaml_via_safe_loader(text: str) -> Any:
    """Minimal YAML support without requiring PyYAML.

    The v0.6 config is restricted to the subset we generate ourselves:
    nested mappings, lists of scalars, and scalar values. We parse it
    by hand to avoid pulling PyYAML into the dev dependency group.
    """

    return _parse_simple_yaml(text)


# --- handrolled YAML subset ---------------------------------------------


def _parse_simple_yaml(text: str) -> Any:
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    # Strip comments and blank lines.
    cleaned: list[tuple[int, str]] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        # Inline comments after the first ' #' that is not inside quotes.
        # The configs we generate never use inline comments inside strings,
        # so the simple split works for our subset.
        if " #" in stripped:
            stripped = stripped[: stripped.index(" #")].rstrip()
        cleaned.append((indent, stripped))
    pos = [0]
    return _parse_block(cleaned, pos, 0)


def _parse_block(lines: list[tuple[int, str]], pos: list[int], indent: int) -> Any:
    if pos[0] >= len(lines):
        return {}
    cur_indent = lines[pos[0]][0]
    if lines[pos[0]][1].startswith("- "):
        return _parse_list(lines, pos, cur_indent)
    return _parse_mapping(lines, pos, cur_indent)


def _parse_mapping(
    lines: list[tuple[int, str]], pos: list[int], indent: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    while pos[0] < len(lines):
        line_indent, line = lines[pos[0]]
        if line_indent < indent:
            break
        if line_indent > indent:
            # Should not happen at this level; treat as end of block.
            break
        if line.startswith("- "):
            break
        if ":" not in line:
            raise ExecutionLaunchPlanError(f"invalid YAML line: {line!r}")
        key, _, raw_value = line.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        pos[0] += 1
        if raw_value == "":
            # Block child.
            if pos[0] < len(lines) and lines[pos[0]][0] > indent:
                child = _parse_block(lines, pos, lines[pos[0]][0])
                result[key] = child
            else:
                result[key] = None
        else:
            result[key] = _parse_scalar(raw_value)
    return result


def _parse_list(
    lines: list[tuple[int, str]], pos: list[int], indent: int
) -> list[Any]:
    result: list[Any] = []
    while pos[0] < len(lines):
        line_indent, line = lines[pos[0]]
        if line_indent < indent:
            break
        if not line.startswith("- "):
            break
        if line_indent != indent:
            break
        raw = line[2:].strip()
        pos[0] += 1
        if raw == "":
            child = _parse_block(lines, pos, indent + 2)
            result.append(child)
        else:
            result.append(_parse_scalar(raw))
    return result


def _parse_scalar(token: str) -> Any:
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    if token in {"true", "True", "yes"}:
        return True
    if token in {"false", "False", "no"}:
        return False
    if token in {"null", "~", "None"}:
        return None
    if (token.startswith("'") and token.endswith("'")) or (
        token.startswith('"') and token.endswith('"')
    ):
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token
