"""Scenario fixtures for the local LLM plus world-model demo."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from codelewm.security import parse_python_source_text


DEMO_SCENARIO_SCHEMA_VERSION = "codelewm.harness.demo_scenario.v1"
DEFAULT_DEMO_SCENARIO_ID = "bugfix-edge-case"


class DemoScenarioError(ValueError):
    """Raised when a demo scenario id or fixture is invalid."""


@dataclass(frozen=True)
class DemoScenarioFile:
    """One source file materialized for a demo scenario."""

    path: str
    content: str
    primary: bool = False

    def __post_init__(self) -> None:
        _validate_safe_relative_path(self.path)
        if not self.content:
            raise DemoScenarioError(f"scenario file {self.path!r} must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "primary": self.primary,
            "bytes": len(self.content.encode("utf-8")),
        }


@dataclass(frozen=True)
class DemoScenario:
    """Manifestable metadata for a local LLM plus world-model demo task."""

    scenario_id: str
    title: str
    instruction: str
    files: tuple[DemoScenarioFile, ...]
    prompt_template_id: str
    expected_static_constraints: Mapping[str, Any] = field(default_factory=dict)
    publication_notes: tuple[str, ...] = ()
    check_command_id: str | None = None
    schema_version: str = DEMO_SCENARIO_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DEMO_SCENARIO_SCHEMA_VERSION:
            raise DemoScenarioError(f"schema_version must be {DEMO_SCENARIO_SCHEMA_VERSION}")
        if not self.scenario_id or any(char.isspace() for char in self.scenario_id):
            raise DemoScenarioError("scenario_id must be non-empty and contain no whitespace")
        if not self.title:
            raise DemoScenarioError("scenario title must not be empty")
        if not self.instruction:
            raise DemoScenarioError("scenario instruction must not be empty")
        if not self.prompt_template_id:
            raise DemoScenarioError("prompt_template_id must not be empty")
        if not self.files:
            raise DemoScenarioError("scenario must include at least one before-state file")
        primary_files = [file for file in self.files if file.primary]
        if len(primary_files) != 1:
            raise DemoScenarioError("scenario must include exactly one primary before-state file")
        for file in self.files:
            if file.path.endswith(".py"):
                try:
                    parse_python_source_text(file.content, filename=file.path)
                except SyntaxError as exc:
                    raise DemoScenarioError(
                        f"scenario file {file.path!r} must be parseable Python: {exc.msg}"
                    ) from exc

    @property
    def primary_file(self) -> DemoScenarioFile:
        return next(file for file in self.files if file.primary)

    @property
    def task_id(self) -> str:
        return f"codelewm-llm-world-model-demo-{self.scenario_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "title": self.title,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "context_paths": [file.path for file in self.files],
            "primary_context_path": self.primary_file.path,
            "prompt_template_id": self.prompt_template_id,
            "expected_static_constraints": _json_native_mapping(self.expected_static_constraints),
            "check_command_id": self.check_command_id,
            "publication_notes": list(self.publication_notes),
            "files": [file.to_dict() for file in self.files],
        }


def list_demo_scenarios() -> tuple[DemoScenario, ...]:
    """Return built-in local demo scenarios."""

    return tuple(_SCENARIOS.values())


def get_demo_scenario(scenario_id: str | None = None) -> DemoScenario:
    """Return a built-in scenario by id."""

    resolved = scenario_id or DEFAULT_DEMO_SCENARIO_ID
    try:
        return _SCENARIOS[resolved]
    except KeyError as exc:
        choices = ", ".join(sorted(_SCENARIOS))
        raise DemoScenarioError(f"unknown demo scenario {resolved!r}; available: {choices}") from exc


def materialize_demo_scenario(scenario: DemoScenario, root: Path | str) -> dict[str, Any]:
    """Write scenario before-state files and metadata below ``root``."""

    scenario_root = Path(root) / scenario.scenario_id
    scenario_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for scenario_file in scenario.files:
        target = scenario_root / scenario_file.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(scenario_file.content, encoding="utf-8")
        written.append(
            {
                **scenario_file.to_dict(),
                "path": target.as_posix(),
                "context_path": scenario_file.path,
            }
        )
    metadata = {
        **scenario.to_dict(),
        "scenario_root": scenario_root.as_posix(),
        "before_path": (scenario_root / scenario.primary_file.path).as_posix(),
        "written_files": written,
    }
    (scenario_root / "scenario.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata["metadata_path"] = (scenario_root / "scenario.json").as_posix()
    return metadata


def write_demo_scenario_shell_env(metadata: Mapping[str, Any], path: Path | str) -> None:
    """Write shell-safe scenario defaults for ``scripts/llm-world-model-demo``."""

    env_path = Path(path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "CODELEWM_LLM_DEMO_SCENARIO_ID": metadata["scenario_id"],
        "CODELEWM_LLM_DEMO_SCENARIO_TITLE": metadata["title"],
        "CODELEWM_LLM_DEMO_SCENARIO_TASK_ID": metadata["task_id"],
        "CODELEWM_LLM_DEMO_SCENARIO_INSTRUCTION": metadata["instruction"],
        "CODELEWM_LLM_DEMO_SCENARIO_CONTEXT_PATH": metadata["primary_context_path"],
        "CODELEWM_LLM_DEMO_SCENARIO_BEFORE": metadata["before_path"],
        "CODELEWM_LLM_DEMO_SCENARIO_PROMPT_TEMPLATE_ID": metadata["prompt_template_id"],
        "CODELEWM_LLM_DEMO_SCENARIO_METADATA": metadata["metadata_path"],
    }
    env_path.write_text(
        "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in fields.items()),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="materialize CodeLeWM LLM demo scenarios")
    parser.add_argument(
        "--scenario",
        default=DEFAULT_DEMO_SCENARIO_ID,
        help=f"scenario id to materialize; default: {DEFAULT_DEMO_SCENARIO_ID}",
    )
    parser.add_argument("--root", type=Path, required=True, help="directory for scenario fixtures")
    parser.add_argument("--shell-env-file", type=Path, help="write shell-safe defaults to this file")
    parser.add_argument("--list", action="store_true", help="list available scenarios and exit")
    parser.add_argument("--json", action="store_true", help="emit scenario metadata JSON")
    args = parser.parse_args(argv)

    if args.list:
        payload = {
            "schema_version": DEMO_SCENARIO_SCHEMA_VERSION,
            "default_scenario_id": DEFAULT_DEMO_SCENARIO_ID,
            "scenarios": [scenario.to_dict() for scenario in list_demo_scenarios()],
        }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0

    scenario = get_demo_scenario(args.scenario)
    metadata = materialize_demo_scenario(scenario, args.root)
    if args.shell_env_file is not None:
        write_demo_scenario_shell_env(metadata, args.shell_env_file)
    if args.json:
        print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _validate_safe_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    lowered = value.lower()
    if path.is_absolute() or ".." in path.parts:
        raise DemoScenarioError(f"scenario file path must be safe and relative: {value!r}")
    if not value or lowered.endswith(".env") or ".env" in lowered or "token" in lowered:
        raise DemoScenarioError(f"scenario file path must not be token-bearing: {value!r}")


def _json_native_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(dict(value), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DemoScenarioError(f"scenario metadata must be JSON-native: {exc}") from exc
    return dict(value)


EXECUTION_RERANK_SCENARIO_ID = "execution-rerank-mbpp"


_SCENARIOS: dict[str, DemoScenario] = {
    DEFAULT_DEMO_SCENARIO_ID: DemoScenario(
        scenario_id=DEFAULT_DEMO_SCENARIO_ID,
        title="Bug fix: normalize blank labels",
        instruction=(
            "Fix normalize_label so blank or whitespace-only labels return 'untitled' "
            "and repeated whitespace collapses to one hyphen."
        ),
        files=(
            DemoScenarioFile(
                path="app.py",
                primary=True,
                content=(
                    "def normalize_label(value: str) -> str:\n"
                    "    return value.strip().lower().replace(\" \", \"-\")\n"
                ),
            ),
        ),
        prompt_template_id="codelewm.openrouter.demo_scenario.bugfix_edge_case.v1",
        expected_static_constraints={
            "changed_files": ["app.py"],
            "touched_symbols": ["normalize_label"],
            "non_comment_change_required": True,
            "expected_terms": ["untitled", "split"],
        },
        publication_notes=(
            "Small public-safe single-file bug-fix scenario.",
            "Candidate code is parsed and patch-applied as text; it is not executed.",
            "Demo artifacts remain workflow evidence and do not prove model usefulness.",
        ),
    ),
    EXECUTION_RERANK_SCENARIO_ID: DemoScenario(
        scenario_id=EXECUTION_RERANK_SCENARIO_ID,
        title="Execution-substrate rerank: complete the MBPP-style function",
        instruction=(
            "Implement compute_square so it returns n * n for any integer n. "
            "The harness samples N completions from the configured LLM, scores "
            "each candidate with the v0.6 execution-substrate world model "
            "(conditioned on the example input), and reports pass@1 lift over "
            "the LLM's own sampling order."
        ),
        files=(
            DemoScenarioFile(
                path="app.py",
                primary=True,
                content=(
                    "def compute_square(n):\n"
                    "    # TODO: return n * n for any integer n.\n"
                    "    pass\n"
                ),
            ),
        ),
        prompt_template_id="codelewm.openrouter.demo_scenario.execution_rerank.v1",
        expected_static_constraints={
            "changed_files": ["app.py"],
            "touched_symbols": ["compute_square"],
            "non_comment_change_required": True,
            "expected_terms": ["return", "n"],
            "example_input_repr": "[3]",
            "expected_output_repr": "9",
            "benchmark_id": "mbpp_demo",
        },
        publication_notes=(
            "Single-problem demo of the execution-substrate rerank protocol.",
            (
                "Candidate code is parsed statically and scored with the "
                "execution-substrate world model. Hidden-test execution that "
                "labels candidate correctness runs only through the named "
                "data-prep sandbox subsystem (codelewm.data.sandbox) and only "
                "on the operator-reviewed example input shipped in this "
                "scenario."
            ),
            (
                "The HumanEval / MBPP-Plus full-benchmark rerank is the "
                "operator-driven flow documented in the v0.6 runbook; the "
                "demo is a single-problem walkthrough of the same protocol."
            ),
        ),
    ),
}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
