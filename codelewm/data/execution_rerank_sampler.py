"""Sample and label completions for the v0.6 downstream rerank gate.

This module lives under :mod:`codelewm.data` because it intentionally
executes untrusted candidate code through the data-prep sandbox. The
model, training, scoring, and generic evaluation paths must continue to
consume only the labeled JSONL artifact emitted here.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from codelewm.data.execution_sources import (
    ExecutionSourceError,
    SourceSubmission,
    get_execution_source_adapter,
)
from codelewm.data.sandbox import (
    DEFAULT_SANDBOX_POLICY,
    SandboxPolicy,
    SandboxPolicyError,
    SandboxRunnerError,
    run_one,
)
from codelewm.data.wsd_mutations import generate_mutants
from codelewm.observability import build_artifact_manifest, write_artifact_manifest
from codelewm.security.secret_scan import scan_paths


COMPLETION_LABEL_SCHEMA_VERSION = "codelewm.eval.completion_label.v1"
COMPLETION_LABEL_ARTIFACT_SCHEMA_VERSION = "codelewm.eval.completion_label_artifact.v1"
COMPLETION_SAMPLING_REPORT_SCHEMA_VERSION = (
    "codelewm.eval.completion_sampling_report.v1"
)
COMPLETION_SAMPLING_PROMPT_SCHEMA_VERSION = (
    "codelewm.eval.completion_sampling_prompt.v1"
)
DEFAULT_EXECUTION_RERANK_LLM = "openrouter:anthropic/claude-haiku-4-5"
_OPENROUTER_PREFIX = "openrouter:"
_BENCHMARK_CHOICES = {"humaneval", "mbpp_plus"}


class CompletionSamplingError(RuntimeError):
    """Raised when completion sampling or labeling cannot finish safely."""


@dataclass(frozen=True)
class CompletionSamplingResult:
    """Summary returned by the completion sampler script."""

    schema_version: str
    output_dir: Path
    labels_path: str
    report_path: str
    secret_scan_report_path: str
    artifact_manifest_id: str
    artifact_manifest_path: str
    benchmark_id: str
    problem_count: int
    completion_count: int
    passed_completion_count: int
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "output_dir": str(self.output_dir),
            "labels_path": self.labels_path,
            "report_path": self.report_path,
            "secret_scan_report_path": self.secret_scan_report_path,
            "artifact_manifest_id": self.artifact_manifest_id,
            "artifact_manifest_path": self.artifact_manifest_path,
            "benchmark_id": self.benchmark_id,
            "problem_count": self.problem_count,
            "completion_count": self.completion_count,
            "passed_completion_count": self.passed_completion_count,
            "dry_run": self.dry_run,
        }


def sample_execution_rerank_completions(
    *,
    benchmark: str,
    source_path: Path | str,
    out: Path | str,
    llm: str = DEFAULT_EXECUTION_RERANK_LLM,
    samples_per_problem: int = 10,
    llm_seeds: Sequence[int] = (17, 42, 1729),
    dry_run: bool = True,
    max_problems: int | None = None,
    max_cases_per_problem: int | None = None,
    short_circuit_failures: bool = False,
    sandbox_policy: SandboxPolicy = DEFAULT_SANDBOX_POLICY,
    overwrite: bool = False,
    allow_secret_findings: bool = False,
    temperature: float = 0.2,
    timeout_seconds: int = 90,
    retry_limit: int = 2,
    command: Sequence[str] = ("scripts/sample-execution-rerank-completions",),
    env: Mapping[str, str] | None = None,
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> CompletionSamplingResult:
    """Sample completions, label them with sandboxed hidden tests, and write JSONL.

    ``dry_run=True`` is deterministic and offline. It emits one passing
    canonical solution and failing synthetic completions per problem so CI can
    verify the artifact contract without spending provider budget. Live mode
    requires an explicit OpenRouter LLM slug and ``OPENROUTER_API_KEY``.
    """

    benchmark_id = normalize_completion_benchmark(benchmark)
    _positive_int(samples_per_problem, "samples_per_problem")
    if not llm_seeds:
        raise CompletionSamplingError("llm_seeds must contain at least one seed")
    seeds = tuple(_coerce_int(seed, "llm_seeds") for seed in llm_seeds)
    if max_problems is not None:
        _positive_int(max_problems, "max_problems")
    if max_cases_per_problem is not None:
        _positive_int(max_cases_per_problem, "max_cases_per_problem")
    if temperature < 0.0 or temperature > 2.0:
        raise CompletionSamplingError("temperature must be in [0.0, 2.0]")
    _positive_int(timeout_seconds, "timeout_seconds")
    if retry_limit < 0:
        raise CompletionSamplingError("retry_limit must be non-negative")

    source = Path(source_path)
    output_dir = Path(out).resolve()
    labels_path = output_dir / f"{benchmark_id}_completion_labels.jsonl"
    prompt_path = output_dir / "prompts" / "sampling_prompts.jsonl"
    config_path = output_dir / "config.json"
    report_path = output_dir / "reports" / "completion_sampling_report.json"
    secret_scan_report_path = output_dir / "reports" / "secret_scan_report.json"
    manifest_path = output_dir / "manifest.json"
    _reject_existing(
        output_dir,
        (
            labels_path,
            prompt_path,
            config_path,
            report_path,
            secret_scan_report_path,
            manifest_path,
        ),
        overwrite=overwrite,
    )

    submissions = _load_submissions(
        benchmark_id=benchmark_id,
        source_path=source,
        max_problems=max_problems,
    )
    if max_cases_per_problem is not None:
        submissions = tuple(
            _limit_submission_cases(submission, limit=max_cases_per_problem)
            for submission in submissions
        )
    prompts = _load_problem_prompts(source)
    env_map = os.environ if env is None else env

    config_payload = {
        "schema_version": COMPLETION_LABEL_ARTIFACT_SCHEMA_VERSION,
        "label_schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "source_path": str(source),
        "llm": llm,
        "samples_per_problem": samples_per_problem,
        "llm_seeds": list(seeds),
        "dry_run": dry_run,
        "max_problems": max_problems,
        "max_cases_per_problem": max_cases_per_problem,
        "short_circuit_failures": short_circuit_failures,
        "sandbox_policy": sandbox_policy.as_dict(),
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
        "retry_limit": retry_limit,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    problem_summaries: list[dict[str, Any]] = []
    for submission in submissions:
        prompt_text = render_completion_prompt(
            benchmark_id=benchmark_id,
            submission=submission,
            problem_prompt=prompts.get(submission.source_problem_id),
        )
        prompt_rows.append(
            {
                "schema_version": COMPLETION_SAMPLING_PROMPT_SCHEMA_VERSION,
                "benchmark_id": benchmark_id,
                "problem_id": submission.source_problem_id,
                "prompt_sha256": _sha256_text(prompt_text),
                "prompt_text": prompt_text,
            }
        )
        problem_passes = 0
        for seed_index, seed in enumerate(seeds):
            for sample_rank in range(1, samples_per_problem + 1):
                llm_order_rank = seed_index * samples_per_problem + sample_rank
                completion_text = _sample_completion(
                    submission=submission,
                    prompt_text=prompt_text,
                    llm=llm,
                    sample_seed=seed,
                    sample_rank=sample_rank,
                    dry_run=dry_run,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    retry_limit=retry_limit,
                    env=env_map,
                )
                completion_id = _completion_id(
                    submission.source_problem_id,
                    seed=seed,
                    sample_rank=sample_rank,
                )
                test_results, passed = _label_completion(
                    completion_text,
                    submission=submission,
                    sandbox_policy=sandbox_policy,
                    short_circuit_failures=short_circuit_failures,
                )
                valid_candidate = _valid_candidate_from_results(test_results)
                if passed:
                    problem_passes += 1
                rows.append(
                    {
                        "schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
                        "benchmark_id": benchmark_id,
                        "problem_id": submission.source_problem_id,
                        "completion_id": completion_id,
                        "completion_text": completion_text,
                        # Compatibility field consumed by older rerank loaders.
                        "code": completion_text,
                        "completion_sha256": _sha256_text(completion_text),
                        "llm_order_rank": llm_order_rank,
                        "llm": llm,
                        "llm_seed": seed,
                        "sample_seed": seed,
                        "sample_rank": sample_rank,
                        "dry_run": dry_run,
                        "scoring_inputs": [
                            {
                                "input_id": input_case.input_id,
                                "input_repr": input_case.input_repr,
                                "input_kind": input_case.input_kind,
                                "function_name": input_case.function_name,
                            }
                            for input_case in submission.inputs
                        ],
                        "label": "pass" if passed else "fail",
                        "passed": passed,
                        "valid_candidate": valid_candidate,
                        "test_results": test_results,
                    }
                )
        problem_summaries.append(
            {
                "problem_id": submission.source_problem_id,
                "completion_count": samples_per_problem * len(seeds),
                "input_case_count": len(submission.inputs),
                "passed_completion_count": problem_passes,
            }
        )

    _write_jsonl(labels_path, rows)
    _write_jsonl(prompt_path, prompt_rows)

    passed_count = sum(1 for row in rows if row["passed"])
    valid_count = sum(1 for row in rows if row["valid_candidate"])
    report_payload = {
        "schema_version": COMPLETION_SAMPLING_REPORT_SCHEMA_VERSION,
        "label_schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "problem_count": len(submissions),
        "completion_count": len(rows),
        "passed_completion_count": passed_count,
        "failed_completion_count": len(rows) - passed_count,
        "valid_completion_count": valid_count,
        "invalid_completion_count": len(rows) - valid_count,
        "valid_completion_rate": valid_count / len(rows) if rows else 0.0,
        "test_pass_rate": passed_count / len(rows) if rows else 0.0,
        "llm": llm,
        "samples_per_problem": samples_per_problem,
        "llm_seeds": list(seeds),
        "dry_run": dry_run,
        "max_cases_per_problem": max_cases_per_problem,
        "short_circuit_failures": short_circuit_failures,
        "sandbox_policy": sandbox_policy.as_dict(),
        "source_path": str(source),
        "labels_path": _relative_to_root(labels_path, output_dir),
        "prompts_path": _relative_to_root(prompt_path, output_dir),
        "problem_summaries": problem_summaries,
        "claim_allowed": False,
        "claim_reason": "completion_labels_only_downstream_rerank_not_run",
    }
    report_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    scan_report = scan_paths(
        (labels_path, prompt_path, config_path, report_path),
        include_suffixes=(),
        recursive=False,
    )
    scan_payload = _relative_secret_scan_payload(scan_report.to_dict(), output_dir)
    secret_scan_report_path.write_text(
        json.dumps(scan_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if scan_payload["findings"] and not allow_secret_findings:
        raise CompletionSamplingError(
            "completion-label artifact contains secret-scan findings; refusing to publish"
        )

    artifact_manifest = build_artifact_manifest(
        artifact_kind="downstream_benchmark",
        root=output_dir,
        files=(
            labels_path,
            prompt_path,
            config_path,
            report_path,
            secret_scan_report_path,
        ),
        command=command,
        config=config_payload,
        metadata={
            "schema_version": COMPLETION_LABEL_ARTIFACT_SCHEMA_VERSION,
            "label_schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
            "benchmark_id": benchmark_id,
            "problem_count": len(submissions),
            "completion_count": len(rows),
            "passed_completion_count": passed_count,
            "valid_completion_count": valid_count,
            "dry_run": dry_run,
            "secret_scan_ok": bool(scan_payload["ok"]),
        },
        source_git_sha=source_git_sha,
        created_at=created_at,
    )
    write_artifact_manifest(artifact_manifest, manifest_path)

    return CompletionSamplingResult(
        schema_version=COMPLETION_LABEL_ARTIFACT_SCHEMA_VERSION,
        output_dir=output_dir,
        labels_path=_relative_to_root(labels_path, output_dir),
        report_path=_relative_to_root(report_path, output_dir),
        secret_scan_report_path=_relative_to_root(secret_scan_report_path, output_dir),
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path=_relative_to_root(manifest_path, output_dir),
        benchmark_id=benchmark_id,
        problem_count=len(submissions),
        completion_count=len(rows),
        passed_completion_count=passed_count,
        dry_run=dry_run,
    )


WSD_RERANK_REPORT_SCHEMA_VERSION = "codelewm.eval.wsd_rerank_pack_report.v1"


def _stable_hash(text: str) -> int:
    """Process-stable non-negative hash (Python's ``hash`` is salted per run)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def build_mutation_rerank_pack(
    *,
    benchmark: str,
    source_path: Path | str,
    out: Path | str,
    mutants_per_problem: int = 12,
    pool_size: int = 6,
    seed: int = 17,
    max_problems: int | None = None,
    max_cases_per_problem: int | None = None,
    sandbox_policy: SandboxPolicy = DEFAULT_SANDBOX_POLICY,
    overwrite: bool = False,
    allow_secret_findings: bool = False,
    command: Sequence[str] = ("scripts/build-wsd-rerank-pack",),
    source_git_sha: str | None = None,
    created_at: str | None = None,
) -> CompletionSamplingResult:
    """Build an *unsaturated* WS-D rerank pack from mutation distractors.

    For each problem the known-correct reference solution is labeled, then up
    to ``mutants_per_problem`` single-point mutants are generated and labeled
    in the sandbox. Only problems whose pool ends up with a pass/fail mix (the
    reference passes and at least one mutant fails) are kept, so reranking has
    headroom by construction. Candidates are deterministically shuffled and the
    shuffle position is written as ``llm_order_rank`` so the rerank gate's
    "llm_order" baseline is an honest (non-reference-first) ordering.

    Emits the same ``completion_label.v1`` JSONL + ``downstream_benchmark``
    manifest contract as :func:`sample_execution_rerank_completions`, so the
    existing ``codelewm eval rerank-*`` consumers read it unchanged.
    """

    benchmark_id = normalize_completion_benchmark(benchmark)
    _positive_int(mutants_per_problem, "mutants_per_problem")
    _positive_int(pool_size, "pool_size")
    if pool_size < 2:
        raise CompletionSamplingError("pool_size must be at least 2 (1 pass + 1 fail)")
    if pool_size - 1 > mutants_per_problem:
        raise CompletionSamplingError(
            "mutants_per_problem must be >= pool_size - 1 to fill a pool"
        )
    if max_problems is not None:
        _positive_int(max_problems, "max_problems")
    if max_cases_per_problem is not None:
        _positive_int(max_cases_per_problem, "max_cases_per_problem")

    source = Path(source_path)
    output_dir = Path(out).resolve()
    labels_path = output_dir / f"{benchmark_id}_completion_labels.jsonl"
    config_path = output_dir / "config.json"
    report_path = output_dir / "reports" / "completion_sampling_report.json"
    secret_scan_report_path = output_dir / "reports" / "secret_scan_report.json"
    manifest_path = output_dir / "manifest.json"
    _reject_existing(
        output_dir,
        (labels_path, config_path, report_path, secret_scan_report_path, manifest_path),
        overwrite=overwrite,
    )

    submissions = _load_submissions(
        benchmark_id=benchmark_id, source_path=source, max_problems=max_problems
    )
    if max_cases_per_problem is not None:
        submissions = tuple(
            _limit_submission_cases(submission, limit=max_cases_per_problem)
            for submission in submissions
        )

    config_payload = {
        "schema_version": COMPLETION_LABEL_ARTIFACT_SCHEMA_VERSION,
        "label_schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "source_path": str(source),
        "generator": "wsd_mutation",
        "mutants_per_problem": mutants_per_problem,
        "pool_size": pool_size,
        "seed": seed,
        "max_problems": max_problems,
        "max_cases_per_problem": max_cases_per_problem,
        "sandbox_policy": sandbox_policy.as_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    problem_summaries: list[dict[str, Any]] = []
    kept_problems = 0
    skipped_reference_fail = 0
    skipped_no_mix = 0
    for submission in submissions:
        pid = submission.source_problem_id
        ref_results, ref_passed = _label_completion(
            submission.code, submission=submission, sandbox_policy=sandbox_policy
        )
        if not ref_passed:
            skipped_reference_fail += 1
            continue
        # Collect failing distractors until the pool is full. Short-circuiting
        # once we have pool_size-1 failures keeps the sandbox cost down.
        failing: list[tuple[str, bool, list[dict[str, Any]], str]] = []
        for mutant in generate_mutants(
            submission.code, count=mutants_per_problem, seed=seed
        ):
            m_results, m_passed = _label_completion(
                mutant.source, submission=submission, sandbox_policy=sandbox_policy
            )
            if not m_passed:
                failing.append((mutant.source, False, m_results, mutant.description))
            if len(failing) >= pool_size - 1:
                break
        if len(failing) < pool_size - 1:
            # not enough behavior-changing distractors to fill a fixed pool
            skipped_no_mix += 1
            continue
        # Exactly one passing candidate (the reference) + (pool_size-1) failing
        # distractors -> a fixed-size pool with a single correct answer.
        candidates: list[tuple[str, bool, list[dict[str, Any]], str]] = [
            (submission.code, True, ref_results, "reference"),
            *failing[: pool_size - 1],
        ]

        order = list(range(len(candidates)))
        random.Random(seed + _stable_hash(pid)).shuffle(order)
        rank_by_index = {cand_index: rank for rank, cand_index in enumerate(order)}

        problem_passes = 0
        for cand_index, (code, passed, results, descriptor) in enumerate(candidates):
            valid_candidate = _valid_candidate_from_results(results)
            if passed:
                problem_passes += 1
            rows.append(
                {
                    "schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
                    "benchmark_id": benchmark_id,
                    "problem_id": pid,
                    "completion_id": f"{pid}::wsd::{cand_index}",
                    "completion_text": code,
                    "code": code,
                    "completion_sha256": _sha256_text(code),
                    "llm_order_rank": rank_by_index[cand_index],
                    "llm": "wsd-mutation",
                    "llm_seed": seed,
                    "sample_seed": seed,
                    "sample_rank": cand_index + 1,
                    "dry_run": False,
                    "wsd_mutation": descriptor,
                    "scoring_inputs": [
                        {
                            "input_id": input_case.input_id,
                            "input_repr": input_case.input_repr,
                            "input_kind": input_case.input_kind,
                            "function_name": input_case.function_name,
                        }
                        for input_case in submission.inputs
                    ],
                    "label": "pass" if passed else "fail",
                    "passed": passed,
                    "valid_candidate": valid_candidate,
                    "test_results": results,
                }
            )
        kept_problems += 1
        problem_summaries.append(
            {
                "problem_id": pid,
                "completion_count": len(candidates),
                "input_case_count": len(submission.inputs),
                "passed_completion_count": problem_passes,
            }
        )

    if not rows:
        raise CompletionSamplingError(
            "no problems yielded a pass/fail mix; increase mutants_per_problem "
            "or check the source reference solutions"
        )

    _write_jsonl(labels_path, rows)

    passed_count = sum(1 for row in rows if row["passed"])
    valid_count = sum(1 for row in rows if row["valid_candidate"])
    mixed_problems = sum(
        1
        for s in problem_summaries
        if 0 < s["passed_completion_count"] < s["completion_count"]
    )
    report_payload = {
        "schema_version": WSD_RERANK_REPORT_SCHEMA_VERSION,
        "label_schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "generator": "wsd_mutation",
        "mutants_per_problem": mutants_per_problem,
        "pool_size": pool_size,
        "seed": seed,
        "source_problem_count": len(submissions),
        "kept_problem_count": kept_problems,
        "skipped_reference_fail": skipped_reference_fail,
        "skipped_no_mix": skipped_no_mix,
        "mixed_problem_count": mixed_problems,
        "mixed_problem_rate": mixed_problems / kept_problems if kept_problems else 0.0,
        "completion_count": len(rows),
        "passed_completion_count": passed_count,
        "failed_completion_count": len(rows) - passed_count,
        "valid_completion_count": valid_count,
        "test_pass_rate": passed_count / len(rows) if rows else 0.0,
        "sandbox_policy": sandbox_policy.as_dict(),
        "source_path": str(source),
        "labels_path": _relative_to_root(labels_path, output_dir),
        "problem_summaries": problem_summaries,
        "claim_allowed": False,
        "claim_reason": "wsd_pack_only_downstream_rerank_not_run",
    }
    report_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    scan_report = scan_paths(
        (labels_path, config_path, report_path), include_suffixes=(), recursive=False
    )
    scan_payload = _relative_secret_scan_payload(scan_report.to_dict(), output_dir)
    secret_scan_report_path.write_text(
        json.dumps(scan_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if scan_payload["findings"] and not allow_secret_findings:
        raise CompletionSamplingError(
            "WS-D rerank pack contains secret-scan findings; refusing to publish"
        )

    artifact_manifest = build_artifact_manifest(
        artifact_kind="downstream_benchmark",
        root=output_dir,
        files=(labels_path, config_path, report_path, secret_scan_report_path),
        command=command,
        config=config_payload,
        metadata={
            "schema_version": COMPLETION_LABEL_ARTIFACT_SCHEMA_VERSION,
            "label_schema_version": COMPLETION_LABEL_SCHEMA_VERSION,
            "benchmark_id": benchmark_id,
            "generator": "wsd_mutation",
            "problem_count": kept_problems,
            "completion_count": len(rows),
            "passed_completion_count": passed_count,
            "valid_completion_count": valid_count,
            "mixed_problem_rate": report_payload["mixed_problem_rate"],
            "dry_run": False,
            "secret_scan_ok": bool(scan_payload["ok"]),
        },
        source_git_sha=source_git_sha,
        created_at=created_at,
    )
    write_artifact_manifest(artifact_manifest, manifest_path)

    return CompletionSamplingResult(
        schema_version=COMPLETION_LABEL_ARTIFACT_SCHEMA_VERSION,
        output_dir=output_dir,
        labels_path=_relative_to_root(labels_path, output_dir),
        report_path=_relative_to_root(report_path, output_dir),
        secret_scan_report_path=_relative_to_root(secret_scan_report_path, output_dir),
        artifact_manifest_id=artifact_manifest.artifact_id,
        artifact_manifest_path=_relative_to_root(manifest_path, output_dir),
        benchmark_id=benchmark_id,
        problem_count=kept_problems,
        completion_count=len(rows),
        passed_completion_count=passed_count,
        dry_run=False,
    )


def normalize_completion_benchmark(benchmark: str) -> Literal["humaneval", "mbpp_plus"]:
    value = benchmark.strip().lower().replace("-", "_")
    if value not in _BENCHMARK_CHOICES:
        raise CompletionSamplingError(
            "benchmark must be one of humaneval, mbpp-plus, or mbpp_plus"
        )
    return value  # type: ignore[return-value]


def render_completion_prompt(
    *,
    benchmark_id: str,
    submission: SourceSubmission,
    problem_prompt: str | None,
) -> str:
    """Render the exact prompt used for live reference-LLM sampling."""

    function_name = _entry_point(submission)
    prompt = problem_prompt or submission.code
    return (
        "You are generating one Python solution for a code benchmark.\n"
        "Return only executable Python source code. Do not include Markdown, "
        "analysis, or prose.\n"
        f"Benchmark: {benchmark_id}\n"
        f"Problem id: {submission.source_problem_id}\n"
        f"Required entry point: {function_name}\n\n"
        "Problem:\n"
        f"{prompt.rstrip()}\n"
    )


def _load_submissions(
    *,
    benchmark_id: str,
    source_path: Path,
    max_problems: int | None,
) -> tuple[SourceSubmission, ...]:
    try:
        adapter = get_execution_source_adapter(benchmark_id)
        submissions = tuple(adapter.iter_submissions(source_path=source_path))
    except ExecutionSourceError as exc:
        raise CompletionSamplingError(str(exc)) from exc
    if max_problems is not None:
        submissions = submissions[:max_problems]
    if not submissions:
        raise CompletionSamplingError(
            f"no {benchmark_id} submissions could be parsed from {source_path}"
        )
    for submission in submissions:
        if submission.expected_outputs is None:
            raise CompletionSamplingError(
                f"{submission.source_problem_id}: expected_outputs are required"
            )
    return submissions


def _limit_submission_cases(
    submission: SourceSubmission, *, limit: int
) -> SourceSubmission:
    if len(submission.inputs) <= limit:
        return submission
    expected_outputs = (
        None
        if submission.expected_outputs is None
        else tuple(submission.expected_outputs[:limit])
    )
    return replace(
        submission,
        inputs=tuple(submission.inputs[:limit]),
        expected_outputs=expected_outputs,
        raw_hash="",
    )


def _load_problem_prompts(source_path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if not source_path.is_file():
        return prompts
    with source_path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            task_id = row.get("task_id")
            prompt = row.get("prompt")
            entry_point = row.get("entry_point")
            if not isinstance(task_id, str):
                continue
            if isinstance(prompt, str) and prompt.strip():
                text = prompt
                if isinstance(entry_point, str) and entry_point not in prompt:
                    text = f"{prompt.rstrip()}\n\nEntry point: {entry_point}\n"
                prompts[task_id] = text
    return prompts


def _sample_completion(
    *,
    submission: SourceSubmission,
    prompt_text: str,
    llm: str,
    sample_seed: int,
    sample_rank: int,
    dry_run: bool,
    temperature: float,
    timeout_seconds: int,
    retry_limit: int,
    env: Mapping[str, str],
) -> str:
    if dry_run:
        return _dry_run_completion(
            submission, sample_seed=sample_seed, sample_rank=sample_rank
        )
    return _openrouter_completion(
        prompt_text=prompt_text,
        llm=llm,
        sample_seed=sample_seed,
        sample_rank=sample_rank,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        retry_limit=retry_limit,
        env=env,
    )


def _dry_run_completion(
    submission: SourceSubmission, *, sample_seed: int, sample_rank: int
) -> str:
    if sample_rank == 1:
        return submission.code
    function_name = _entry_point(submission)
    variant = (sample_seed + sample_rank) % 4
    if variant == 0:
        body = "return None"
    elif variant == 1:
        body = "return False"
    elif variant == 2:
        body = "return 0"
    else:
        body = "raise RuntimeError('dry run completion')"
    return f"def {function_name}(*args, **kwargs):\n    {body}\n"


def _openrouter_completion(
    *,
    prompt_text: str,
    llm: str,
    sample_seed: int,
    sample_rank: int,
    temperature: float,
    timeout_seconds: int,
    retry_limit: int,
    env: Mapping[str, str],
) -> str:
    if not llm.startswith(_OPENROUTER_PREFIX):
        raise CompletionSamplingError(
            "live sampling currently supports only openrouter:<model> LLM slugs"
        )
    api_key = env.get("OPENROUTER_API_KEY")
    if not api_key:
        raise CompletionSamplingError(
            "OPENROUTER_API_KEY is required for live completion sampling"
        )
    if env.get("OPENROUTER_DEBUG", "").lower() in {"1", "true", "yes", "on"}:
        raise CompletionSamplingError(
            "OPENROUTER_DEBUG must be disabled for publishable sampling runs"
        )
    try:
        from openrouter import OpenRouter  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CompletionSamplingError(
            "OpenRouter SDK is not installed; install the llm dependency group"
        ) from exc

    model = llm[len(_OPENROUTER_PREFIX) :]
    seeded_prompt = (
        f"{prompt_text.rstrip()}\n\n"
        f"Sampling seed: {sample_seed}\n"
        f"Candidate index for this seed: {sample_rank}\n"
    )
    errors: list[str] = []
    for attempt in range(retry_limit + 1):
        try:
            with OpenRouter(
                api_key=api_key,
                http_referer="https://github.com/abdelstark/CodeLeWM",
                x_open_router_title="CodeLeWM execution rerank sampler",
                timeout_ms=timeout_seconds * 1000,
            ) as open_router:
                response = open_router.chat.send(
                    messages=[{"role": "user", "content": seeded_prompt}],
                    model=model,
                    temperature=temperature,
                    stream=False,
                    timeout_ms=timeout_seconds * 1000,
                )
            return _strip_markdown_code(_extract_response_text(response))
        except Exception as exc:  # pragma: no cover - live provider failure path.
            errors.append(_redacted_error(str(exc)))
            if attempt >= retry_limit:
                raise CompletionSamplingError(
                    f"OpenRouter completion request failed: {errors[-1]}"
                ) from exc
    raise CompletionSamplingError("OpenRouter completion request failed")


def _label_completion(
    code: str,
    *,
    submission: SourceSubmission,
    sandbox_policy: SandboxPolicy,
    short_circuit_failures: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    if submission.expected_outputs is None:
        raise CompletionSamplingError(
            f"{submission.source_problem_id}: expected outputs are required"
        )
    results: list[dict[str, Any]] = []
    all_passed = True
    for input_case, expected_output in zip(
        submission.inputs, submission.expected_outputs, strict=True
    ):
        try:
            if input_case.input_kind == "function_call":
                sandbox_result = run_one(
                    code,
                    input_repr=input_case.input_repr,
                    function_name=input_case.function_name,
                    policy=sandbox_policy,
                )
            else:
                sandbox_result = run_one(
                    code,
                    stdin_text=input_case.input_repr,
                    policy=sandbox_policy,
                )
        except (SandboxPolicyError, SandboxRunnerError) as exc:
            raise CompletionSamplingError(
                f"sandbox labeling failed for {submission.source_problem_id}: {exc}"
            ) from exc
        passed = bool(
            sandbox_result.ok and sandbox_result.output_repr == expected_output
        )
        all_passed = all_passed and passed
        results.append(
            {
                "input_id": input_case.input_id,
                "input_kind": input_case.input_kind,
                "function_name": input_case.function_name,
                "exit_code": sandbox_result.exit_code.value,
                "passed": passed,
                "output_repr_sha256": _sha256_optional(sandbox_result.output_repr),
                "expected_output_sha256": _sha256_text(expected_output),
                "output_type": sandbox_result.output_type,
                "output_kind": sandbox_result.output_kind,
                "exception_class": sandbox_result.exception_class,
                "policy_violations": list(sandbox_result.policy_violations),
                "wall_time_ms": sandbox_result.wall_time_ms,
                "determinism_check": sandbox_result.determinism_check,
            }
        )
        if short_circuit_failures and not passed:
            break
    return results, all_passed


def _valid_candidate_from_results(results: Sequence[Mapping[str, Any]]) -> bool:
    return bool(results) and all(result.get("exit_code") == "ok" for result in results)


def _completion_id(problem_id: str, *, seed: int, sample_rank: int) -> str:
    return f"{problem_id}::seed-{seed}::rank-{sample_rank}"


def _entry_point(submission: SourceSubmission) -> str:
    for input_case in submission.inputs:
        if input_case.function_name:
            return input_case.function_name
    return "solution"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _reject_existing(
    output_dir: Path, paths: Sequence[Path], *, overwrite: bool
) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        rel = ", ".join(_relative_to_root(path, output_dir) for path in existing)
        raise CompletionSamplingError(
            f"output already exists; pass --overwrite to replace: {rel}"
        )


def _relative_secret_scan_payload(
    payload: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "ok": payload["ok"],
        "paths_scanned": [
            _relative_to_root(Path(path), root)
            for path in payload.get("paths_scanned", [])
        ],
        "findings": [
            {
                **dict(finding),
                "path": _relative_to_root(Path(str(finding["path"])), root),
            }
            for finding in payload.get("findings", [])
        ],
    }


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CompletionSamplingError(f"{name} must be a positive integer")


def _coerce_int(value: object, name: str) -> int:
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CompletionSamplingError(f"{name} must contain integers") from exc
    return coerced


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_optional(text: str | None) -> str | None:
    return None if text is None else _sha256_text(text)


_FENCE_RE = re.compile(r"^```(?:python)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _strip_markdown_code(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip() + "\n"
    if "```" not in stripped:
        return stripped + ("\n" if stripped else "")
    parts = stripped.split("```")
    for part in parts:
        candidate = part.strip()
        if candidate.lower().startswith("python"):
            candidate = candidate[6:].lstrip()
        if "def " in candidate:
            return candidate.rstrip() + "\n"
    return stripped + "\n"


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            return _message_content(choices[0])
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return _message_content(choices[0])
    content = getattr(response, "content", None)
    return content if isinstance(content, str) else str(response)


def _message_content(choice: Any) -> str:
    if isinstance(choice, Mapping):
        message = choice.get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(choice.get("content"), str):
            return choice["content"]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    content = getattr(choice, "content", None)
    return content if isinstance(content, str) else str(choice)


_SECRETISH_RE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|"
    r"(?:api[_-]?key|token|secret)[^\s]{0,20})"
)


def _redacted_error(detail: str) -> str:
    return _SECRETISH_RE.sub("<redacted>", detail)[:500]


def parse_llm_seed_csv(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise CompletionSamplingError("llm seed list must contain integers") from exc
    if not seeds:
        raise CompletionSamplingError("llm seed list must not be empty")
    return seeds


def policy_from_args(
    *,
    timeout_ms: int,
    memory_mb: int,
    cpu_seconds: int,
    determinism_check: bool,
) -> SandboxPolicy:
    try:
        return SandboxPolicy(
            import_allowlist="stdlib_only",
            timeout_ms=timeout_ms,
            memory_mb=memory_mb,
            cpu_seconds=cpu_seconds,
            determinism_check=determinism_check,
        )
    except SandboxPolicyError as exc:
        raise CompletionSamplingError(f"invalid sandbox policy: {exc}") from exc


def script_command(argv: Sequence[str] | None = None) -> tuple[str, ...]:
    args = tuple(sys.argv[1:] if argv is None else argv)
    return ("scripts/sample-execution-rerank-completions", *args)
