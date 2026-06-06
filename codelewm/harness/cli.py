"""Command-line entry point for CodeLeWM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from codelewm import __version__
from codelewm.data import (
    DatasetBuildConfigError,
    DatasetBuildError,
    OptionalDependencyError,
    PackError,
    SourceRecordError,
    SourceUnavailableError,
    build_dataset_from_config_path,
    pack_dataset_from_manifest,
)
from codelewm.eval import (
    ActionAblationError,
    ActionViewPolicyError,
    CrashPredictionError,
    ExecutionEvalError,
    LatentMatrixError,
    LatentProbeError,
    P_PASS_DATASET_KINDS,
    P_PASS_DEFAULT_BASELINES,
    PPassCalibrationError,
    RetrievalEvalError,
    SurpriseEvalError,
    DownstreamBenchmarkPackError,
    DownstreamRerankEvalError,
    ExecutionRerankEvalError,
    SemanticDecoyPackError,
    build_downstream_benchmark_pack,
    build_semantic_decoy_pack,
    run_downstream_rerank_evaluation,
    run_action_ablation_suite,
    run_p_pass_calibration_evaluation,
    run_crash_prediction_evaluation,
    run_execution_rerank_evaluation,
    run_execution_probe_evaluation,
    run_execution_retrieval_evaluation,
    run_execution_surprise_evaluation,
    run_latent_matrix_evaluation,
    run_latent_probe_evaluation,
    run_retrieval_evaluation,
    run_surprise_evaluation,
)
from codelewm.harness.index_runner import build_transition_index_artifact
from codelewm.harness.llm_demo import (
    LLMWorldModelDemoError,
    run_llm_world_model_demo,
)
from codelewm.harness.openrouter_adapter import (
    OpenRouterAdapterError,
    register_openrouter_byok_credential,
)
from codelewm.harness.quality import (
    ScorerQualityError,
    run_scorer_quality_evaluation,
)
from codelewm.harness.scorer import ScoreError, load_scorer
from codelewm.harness.transition_index import TransitionIndexError
from codelewm.model.inspection import CheckpointInspectionError, inspect_checkpoint
from codelewm.observability import (
    ArtifactManifestError,
    LogEvent,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_log_event_jsonl,
)
from codelewm.security import CheckpointTrustError, scan_paths
from codelewm.training import (
    EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
    ExecutionTrainConfigError,
    TrainConfigError,
    TrainingRunError,
    cpu_smoke_training_executor,
    load_execution_train_config,
    load_train_config,
    make_torch_training_executor,
    peek_train_config_schema_version,
    train,
    train_execution_run,
    validate_train_config,
)


MANIFEST_VERIFY_REPORT_SCHEMA_VERSION = "codelewm.manifest_verify.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codelewm",
        description=(
            "CodeLeWM command-line interface. Implementation commands are "
            "introduced incrementally by the spec-tracked issues."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    score = subparsers.add_parser("score", help="score one candidate after-state")
    score.add_argument(
        "--before", type=Path, required=True, help="before-state Python file"
    )
    score.add_argument(
        "--instruction", required=True, help="instruction text or path to a text file"
    )
    score.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="candidate after-state Python file",
    )
    score.add_argument("--checkpoint", type=Path, required=True, help="checkpoint file")
    score.add_argument(
        "--device", default="auto", choices=("cpu", "cuda", "mps", "auto")
    )
    score.add_argument(
        "--index",
        type=Path,
        help="transition index directory for retrieval-prior scoring",
    )
    score.add_argument(
        "--retrieval-prior-weight",
        type=float,
        default=0.0,
        help="non-negative weight applied to the retrieval prior",
    )
    score.add_argument(
        "--retrieval-prior-k",
        type=int,
        default=10,
        help="nearest index hits used for the prior",
    )
    score.add_argument("--json", action="store_true", help="emit JSON output")
    score.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    score.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="load the checkpoint without verifying its manifest (trusted local use only)",
    )
    score.set_defaults(func=_score_command)
    rerank = subparsers.add_parser(
        "rerank", help="rerank candidate after-states or patches"
    )
    rerank.add_argument(
        "--before", type=Path, required=True, help="before-state Python file"
    )
    rerank.add_argument(
        "--instruction", required=True, help="instruction text or path to a text file"
    )
    rerank.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="candidate after-state file, patch file, or directory",
    )
    rerank.add_argument(
        "--checkpoint", type=Path, required=True, help="checkpoint file"
    )
    rerank.add_argument(
        "--device", default="auto", choices=("cpu", "cuda", "mps", "auto")
    )
    rerank.add_argument(
        "--index",
        type=Path,
        help="transition index directory for retrieval-prior scoring",
    )
    rerank.add_argument(
        "--retrieval-prior-weight",
        type=float,
        default=0.0,
        help="non-negative weight applied to the retrieval prior",
    )
    rerank.add_argument(
        "--retrieval-prior-k",
        type=int,
        default=10,
        help="nearest index hits used for the prior",
    )
    rerank.add_argument("--json", action="store_true", help="emit JSON output")
    rerank.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    rerank.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="load the checkpoint without verifying its manifest (trusted local use only)",
    )
    rerank.set_defaults(func=_rerank_command)
    llm_demo = subparsers.add_parser(
        "llm-demo", help="run LLM candidate generation plus CodeLeWM reranking demo"
    )
    llm_demo.add_argument(
        "--before", type=Path, required=True, help="before-state Python file"
    )
    llm_demo.add_argument(
        "--instruction", required=True, help="instruction text or path to a text file"
    )
    llm_demo.add_argument("--checkpoint", type=Path, required=True, help="checkpoint file")
    llm_demo.add_argument("--out", type=Path, required=True, help="demo artifact directory")
    llm_demo.add_argument("--task-id", default="codelewm-demo", help="stable demo task id")
    llm_demo.add_argument(
        "--context-path",
        help="repository-relative path label for the before file sent to the LLM",
    )
    llm_demo.add_argument(
        "--device", default="auto", choices=("cpu", "cuda", "mps", "auto")
    )
    llm_demo.add_argument(
        "--index",
        type=Path,
        help="transition index directory for retrieval-prior scoring",
    )
    llm_demo.add_argument(
        "--retrieval-prior-weight",
        type=float,
        default=0.0,
        help="non-negative weight applied to the retrieval prior",
    )
    llm_demo.add_argument(
        "--retrieval-prior-k",
        type=int,
        default=10,
        help="nearest index hits used for the prior",
    )
    llm_demo.add_argument(
        "--parent-manifest",
        action="append",
        type=Path,
        default=[],
        help="parent artifact manifest to verify and record; may be repeated",
    )
    llm_demo.add_argument(
        "--checkpoint-inspection-manifest",
        type=Path,
        help="manifest for a model checkpoint inspection diagnostic artifact",
    )
    llm_demo.add_argument(
        "--checkpoint-inspection-report",
        type=Path,
        help="model checkpoint inspection report; inferred from manifest metadata when omitted",
    )
    llm_demo.add_argument(
        "--latent-matrix-manifest",
        type=Path,
        help="manifest for a latent matrix diagnostic artifact",
    )
    llm_demo.add_argument(
        "--latent-matrix-report",
        type=Path,
        help="latent matrix diagnostic report; inferred from manifest metadata when omitted",
    )
    llm_demo.add_argument(
        "--tensorboard-manifest",
        type=Path,
        help="manifest for a TensorBoard export diagnostic artifact",
    )
    llm_demo.add_argument(
        "--tensorboard-export",
        type=Path,
        help="TensorBoard export metadata report; inferred from manifest metadata when omitted",
    )
    llm_demo.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="load the checkpoint without verifying its manifest (trusted local use only)",
    )
    llm_demo.add_argument(
        "--require-learned-scorer",
        action="store_true",
        help="fail instead of using the deterministic fixture scorer when the checkpoint is not a learned torch model",
    )
    llm_demo.add_argument("--overwrite", action="store_true", help="replace existing output")
    llm_demo.add_argument("--json", action="store_true", help="emit JSON output")
    llm_demo.add_argument(
        "--tui",
        action="store_true",
        help="open the optional Textual TUI after the demo artifact is written",
    )
    llm_demo.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    llm_demo.set_defaults(func=_llm_demo_command)
    llm_demo_tui = subparsers.add_parser(
        "llm-demo-tui", help="open a Textual TUI for an existing llm-demo artifact"
    )
    llm_demo_tui_source = llm_demo_tui.add_mutually_exclusive_group(required=True)
    llm_demo_tui_source.add_argument(
        "--view-model", type=Path, help="path to reports/visual_view_model.json"
    )
    llm_demo_tui_source.add_argument(
        "--demo-dir", type=Path, help="demo artifact directory containing reports/visual_view_model.json"
    )
    llm_demo_tui.add_argument(
        "--snapshot-json",
        action="store_true",
        help="emit a deterministic JSON snapshot instead of opening Textual",
    )
    llm_demo_tui.add_argument("--json", action="store_true", help="emit JSON errors")
    llm_demo_tui.set_defaults(func=_llm_demo_tui_command)
    execution_demo = subparsers.add_parser(
        "execution-rerank-demo",
        help="run the v0.6 execution-rerank tour and write the showcase artifact set",
    )
    execution_demo.add_argument("--checkpoint", type=Path, required=True)
    execution_demo.add_argument("--out", type=Path, required=True, help="run directory")
    execution_demo.add_argument("--tour", type=int, default=5, help="number of built-in problems")
    execution_demo.add_argument("--scenario", default="execution-rerank-mbpp")
    execution_demo.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    execution_demo.add_argument(
        "--html", type=Path, help="optional extra self-contained HTML export path"
    )
    execution_demo.add_argument("--allow-unsafe-checkpoint", action="store_true")
    execution_demo.add_argument(
        "--fixture-scorer",
        action="store_true",
        help="allow a deterministic fixture scorer for tests/demos",
    )
    execution_demo.add_argument("--overwrite", action="store_true")
    execution_demo.add_argument(
        "--tui", action="store_true", help="open the Textual TUI after the run"
    )
    execution_demo.add_argument("--json", action="store_true")
    execution_demo.set_defaults(func=_execution_rerank_demo_command)
    execution_tui = subparsers.add_parser(
        "execution-rerank-tui",
        help="open a Textual TUI for an existing execution-rerank tour",
    )
    execution_tui_source = execution_tui.add_mutually_exclusive_group(required=True)
    execution_tui_source.add_argument(
        "--view-model",
        type=Path,
        help="path to reports/execution_rerank_view_model.json",
    )
    execution_tui_source.add_argument(
        "--demo-dir",
        type=Path,
        help="demo directory containing reports/execution_rerank_view_model.json",
    )
    execution_tui.add_argument(
        "--snapshot-json",
        action="store_true",
        help="emit a deterministic JSON snapshot instead of opening Textual",
    )
    execution_tui.add_argument("--json", action="store_true", help="emit JSON errors")
    execution_tui.set_defaults(func=_execution_rerank_tui_command)
    openrouter = subparsers.add_parser("openrouter", help="OpenRouter helper utilities")
    openrouter_subcommands = openrouter.add_subparsers(dest="openrouter_command")
    byok_register = openrouter_subcommands.add_parser(
        "byok-register",
        help="create an OpenRouter BYOK provider credential from local env secrets",
    )
    byok_register.add_argument("--provider", default=None, help="provider slug; default: anthropic")
    byok_register.add_argument(
        "--key-env",
        default=None,
        help="environment variable containing the raw provider key; default: ANTHROPIC_API_KEY",
    )
    byok_register.add_argument(
        "--management-key-env",
        default=None,
        help=(
            "environment variable containing the OpenRouter management key; "
            "default: OPENROUTER_MANAGEMENT_KEY"
        ),
    )
    byok_register.add_argument("--name", default=None, help="OpenRouter BYOK key name")
    byok_register.add_argument(
        "--allowed-model",
        action="append",
        default=None,
        help="model slug allowlist for the BYOK credential; may be repeated",
    )
    byok_register.add_argument("--workspace-id", default=None, help="OpenRouter workspace UUID")
    byok_register.add_argument(
        "--fallback",
        action="store_true",
        default=None,
        help="register as a fallback key",
    )
    byok_register.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="validate inputs without sending keys",
    )
    byok_register.add_argument("--json", action="store_true", help="emit JSON output")
    byok_register.set_defaults(func=_openrouter_byok_register_command)
    openrouter.set_defaults(func=_openrouter_help_command, openrouter_parser=openrouter)
    train_parser = subparsers.add_parser(
        "train", help="run manifest-backed CodeLeWM training"
    )
    train_parser.add_argument(
        "--config", type=Path, required=True, help="training config JSON or YAML path"
    )
    train_parser.add_argument(
        "--out", type=Path, help="override output.run_dir and write run artifacts here"
    )
    train_parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        help="override trainer.accelerator for this run",
    )
    train_parser.add_argument(
        "--executor",
        default="torch",
        choices=("torch", "cpu-smoke"),
        help="training executor to run",
    )
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help="parent training_manifest.json to resume from after compatibility checks",
    )
    train_parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="emit optional TensorBoard-compatible training/checkpoint event logs",
    )
    train_parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        help="event-log directory relative to output.run_dir; implies --tensorboard",
    )
    train_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite an existing CodeLeWM run output",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "training seed override; required for "
            "codelewm.execution_train_config.v1 configs, "
            "ignored by the legacy HDF5 path"
        ),
    )
    train_parser.add_argument(
        "--pack-local-dir",
        type=Path,
        default=None,
        help=(
            "override CODELEWM_EXECUTION_PACK_LOCAL_DIR for execution-config "
            "runs; the directory must already contain the pack.jsonl and "
            "manifest.json"
        ),
    )
    train_parser.add_argument("--json", action="store_true", help="emit JSON output")
    train_parser.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    train_parser.set_defaults(func=_train_command)
    model = subparsers.add_parser("model", help="model inspection utilities")
    model_subcommands = model.add_subparsers(dest="model_command")
    inspect_ckpt = model_subcommands.add_parser(
        "inspect-checkpoint",
        help="write a manifest-backed checkpoint tensor/layer inspection report",
    )
    inspect_ckpt.add_argument(
        "--checkpoint", type=Path, required=True, help="trusted training checkpoint path"
    )
    inspect_ckpt.add_argument(
        "--checkpoint-manifest",
        type=Path,
        help="checkpoint manifest path; defaults to <checkpoint>.manifest.json",
    )
    inspect_ckpt.add_argument(
        "--out", type=Path, required=True, help="checkpoint inspection artifact directory"
    )
    inspect_ckpt.add_argument(
        "--parent-manifest",
        type=Path,
        action="append",
        default=[],
        help="parent artifact manifest to verify and record; may be repeated",
    )
    inspect_ckpt.add_argument(
        "--histogram-bins",
        type=int,
        default=16,
        help="number of bins for selected tensor histograms",
    )
    inspect_ckpt.add_argument(
        "--max-histogram-tensors",
        type=int,
        default=24,
        help="maximum tensors that receive histogram summaries",
    )
    inspect_ckpt.add_argument(
        "--max-histogram-values",
        type=int,
        default=8192,
        help="maximum finite tensor values sampled per histogram",
    )
    inspect_ckpt.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="load the checkpoint without verifying its manifest (trusted local use only)",
    )
    inspect_ckpt.add_argument(
        "--overwrite", action="store_true", help="overwrite existing inspection output files"
    )
    inspect_ckpt.add_argument("--json", action="store_true", help="emit JSON output")
    inspect_ckpt.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    inspect_ckpt.set_defaults(func=_model_inspect_checkpoint_command)
    model.set_defaults(func=_model_help_command, model_parser=model)
    eval_parser = subparsers.add_parser("eval", help="evaluation report utilities")
    eval_subcommands = eval_parser.add_subparsers(dest="eval_command")
    retrieval = eval_subcommands.add_parser(
        "retrieval", help="run retrieval evaluation"
    )
    retrieval.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted training checkpoint path",
    )
    retrieval.add_argument(
        "--data",
        type=Path,
        required=True,
        help="packed dataset directory or manifest.json",
    )
    retrieval.add_argument(
        "--out", type=Path, required=True, help="retrieval report artifact directory"
    )
    retrieval.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    retrieval.add_argument(
        "--max-candidates",
        type=int,
        default=1000,
        help="maximum easy held-out candidates",
    )
    retrieval.add_argument(
        "--hard-negatives",
        type=int,
        default=1000,
        help="maximum hard negatives per query",
    )
    retrieval.add_argument(
        "--seed", type=int, default=0, help="deterministic evaluation seed"
    )
    retrieval.add_argument(
        "--report-scope",
        default="headline",
        choices=("headline", "ablation", "diagnostic"),
        help="action-view policy scope for this report",
    )
    retrieval.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing retrieval output files",
    )
    retrieval.add_argument("--json", action="store_true", help="emit JSON output")
    retrieval.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    retrieval.set_defaults(func=_eval_retrieval_command)
    execution_retrieval = eval_subcommands.add_parser(
        "execution-retrieval",
        help="run retrieval evaluation over a v0.6 JSONL execution pack",
    )
    execution_retrieval.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted v0.6 execution training checkpoint path",
    )
    execution_retrieval.add_argument(
        "--pack",
        type=Path,
        required=True,
        help="execution pack directory, pack.jsonl, manifest.json, or artifact_manifest.json",
    )
    execution_retrieval.add_argument(
        "--baselines",
        default="random,no_action,shuffled_action",
        help="comma-separated baselines: random, lexical, no_action, shuffled_action",
    )
    execution_retrieval.add_argument(
        "--out",
        type=Path,
        required=True,
        help="retrieval report artifact directory",
    )
    execution_retrieval.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    execution_retrieval.add_argument(
        "--max-candidates",
        type=int,
        default=1000,
        help="maximum val/test execution records to evaluate",
    )
    execution_retrieval.add_argument(
        "--seed", type=int, default=0, help="deterministic evaluation seed"
    )
    execution_retrieval.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing execution retrieval output files",
    )
    execution_retrieval.add_argument("--json", action="store_true", help="emit JSON output")
    execution_retrieval.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    execution_retrieval.set_defaults(func=_eval_execution_retrieval_command)
    execution_surprise = eval_subcommands.add_parser(
        "execution-surprise",
        help="run surprise evaluation over a v0.6 JSONL execution pack",
    )
    execution_surprise.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted v0.6 execution training checkpoint path",
    )
    execution_surprise.add_argument(
        "--pack",
        type=Path,
        required=True,
        help="execution pack directory, pack.jsonl, manifest.json, or artifact_manifest.json",
    )
    execution_surprise.add_argument(
        "--decoys",
        default="mutation,same_problem_different_submission,same_code_different_input",
        help="comma-separated decoys: mutation, same_problem_different_submission, same_code_different_input",
    )
    execution_surprise.add_argument(
        "--out",
        type=Path,
        required=True,
        help="surprise report artifact directory",
    )
    execution_surprise.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    execution_surprise.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="maximum val/test execution records to evaluate",
    )
    execution_surprise.add_argument(
        "--seed", type=int, default=0, help="deterministic decoy seed"
    )
    execution_surprise.add_argument(
        "--semantic-decoy-manifest",
        type=Path,
        help="optional semantic decoy pack manifest to use for same-problem decoys",
    )
    execution_surprise.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing execution surprise output files",
    )
    execution_surprise.add_argument("--json", action="store_true", help="emit JSON output")
    execution_surprise.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    execution_surprise.set_defaults(func=_eval_execution_surprise_command)
    semantic_decoy_pack = eval_subcommands.add_parser(
        "semantic-decoy-pack",
        help="build a manifest-backed same-problem semantic decoy pack",
    )
    semantic_decoy_pack.add_argument(
        "--pack",
        type=Path,
        required=True,
        help="execution pack directory, pack.jsonl, manifest.json, or artifact_manifest.json",
    )
    semantic_decoy_pack.add_argument(
        "--out", type=Path, required=True, help="semantic decoy pack artifact directory"
    )
    semantic_decoy_pack.add_argument(
        "--splits",
        default="val,test",
        help="comma-separated execution-pack splits used for decoy construction",
    )
    semantic_decoy_pack.add_argument("--seed", type=int, default=0)
    semantic_decoy_pack.add_argument(
        "--max-pairs-per-query",
        type=int,
        default=3,
        help="maximum semantic decoys retained per query record",
    )
    semantic_decoy_pack.add_argument(
        "--min-pairs-for-claim",
        type=int,
        default=100,
        help="minimum same-problem semantic pairs required for claim eligibility",
    )
    semantic_decoy_pack.add_argument(
        "--min-distinct-problems-for-claim",
        type=int,
        default=30,
        help="minimum distinct problems required for claim eligibility",
    )
    semantic_decoy_pack.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing semantic decoy pack output files",
    )
    semantic_decoy_pack.add_argument("--json", action="store_true", help="emit JSON output")
    semantic_decoy_pack.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    semantic_decoy_pack.set_defaults(func=_eval_semantic_decoy_pack_command)
    execution_probe = eval_subcommands.add_parser(
        "execution-probe",
        help="run execution-specific frozen latent probes over a v0.6 JSONL pack",
    )
    execution_probe.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted v0.6 execution training checkpoint path",
    )
    execution_probe.add_argument(
        "--pack",
        type=Path,
        required=True,
        help="execution pack directory, pack.jsonl, manifest.json, or artifact_manifest.json",
    )
    execution_probe.add_argument(
        "--targets",
        default="output_type,will_raise,output_magnitude_bucket,output_length_bucket",
        help="comma-separated execution probe targets",
    )
    execution_probe.add_argument(
        "--out",
        type=Path,
        required=True,
        help="latent probe report artifact directory",
    )
    execution_probe.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    execution_probe.add_argument(
        "--max-examples-per-split",
        type=int,
        default=1000,
        help="maximum execution rows per split",
    )
    execution_probe.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="bootstrap samples used for probe confidence intervals",
    )
    execution_probe.add_argument(
        "--seed", type=int, default=0, help="deterministic probe seed"
    )
    execution_probe.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing execution probe output files",
    )
    execution_probe.add_argument("--json", action="store_true", help="emit JSON output")
    execution_probe.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    execution_probe.set_defaults(func=_eval_execution_probe_command)
    crash_prediction = eval_subcommands.add_parser(
        "crash-prediction",
        help="run the v0.6 execution-substrate crash-prediction fallback eval",
    )
    crash_prediction.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted v0.6 execution training checkpoint path",
    )
    crash_prediction.add_argument(
        "--pack",
        type=Path,
        required=True,
        help="execution pack directory, pack.jsonl, manifest.json, or artifact_manifest.json",
    )
    crash_prediction.add_argument(
        "--out",
        type=Path,
        required=True,
        help="crash prediction report artifact directory",
    )
    crash_prediction.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    crash_prediction.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="maximum val/test execution records to evaluate",
    )
    crash_prediction.add_argument(
        "--seed", type=int, default=0, help="deterministic evaluation seed"
    )
    crash_prediction.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing crash prediction output files",
    )
    crash_prediction.add_argument("--json", action="store_true", help="emit JSON output")
    crash_prediction.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    crash_prediction.set_defaults(func=_eval_crash_prediction_command)
    latent_probe = eval_subcommands.add_parser(
        "latent-probe", help="run frozen latent representation probes"
    )
    latent_probe.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted training checkpoint path",
    )
    latent_probe.add_argument(
        "--data",
        type=Path,
        required=True,
        help="packed dataset directory or manifest.json",
    )
    latent_probe.add_argument(
        "--out", type=Path, required=True, help="latent probe report artifact directory"
    )
    latent_probe.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    latent_probe.add_argument(
        "--max-examples-per-split",
        type=int,
        default=1000,
        help="maximum rows to probe from each train/val/test split",
    )
    latent_probe.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="bootstrap samples used for accuracy confidence intervals",
    )
    latent_probe.add_argument(
        "--seed", type=int, default=0, help="deterministic probe seed"
    )
    latent_probe.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing latent probe output files",
    )
    latent_probe.add_argument("--json", action="store_true", help="emit JSON output")
    latent_probe.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    latent_probe.set_defaults(func=_eval_latent_probe_command)
    latent_matrix = eval_subcommands.add_parser(
        "latent-matrix", help="run latent representation matrix diagnostics"
    )
    latent_matrix.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted training checkpoint path",
    )
    latent_matrix.add_argument(
        "--data",
        type=Path,
        required=True,
        help="packed dataset directory or manifest.json",
    )
    latent_matrix.add_argument(
        "--out", type=Path, required=True, help="latent matrix report artifact directory"
    )
    latent_matrix.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    latent_matrix.add_argument(
        "--max-examples-per-split",
        type=int,
        default=1000,
        help="maximum rows to inspect from each train/val/test split",
    )
    latent_matrix.add_argument(
        "--matrix-dimension-limit",
        type=int,
        default=32,
        help="maximum latent dimensions included in heatmap-ready matrix previews",
    )
    latent_matrix.add_argument(
        "--top-dimensions",
        type=int,
        default=16,
        help="top per-target dimensions retained for label-association diagnostics",
    )
    latent_matrix.add_argument(
        "--max-pairwise-rows",
        type=int,
        default=512,
        help="maximum rows used for mean pairwise cosine diagnostics",
    )
    latent_matrix.add_argument(
        "--latent-probe-report",
        type=Path,
        help="optional codelewm.eval.latent_probe_report.v1 JSON to link probe controls",
    )
    latent_matrix.add_argument(
        "--seed", type=int, default=0, help="deterministic matrix diagnostic seed"
    )
    latent_matrix.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing latent matrix output files",
    )
    latent_matrix.add_argument("--json", action="store_true", help="emit JSON output")
    latent_matrix.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    latent_matrix.set_defaults(func=_eval_latent_matrix_command)
    surprise = eval_subcommands.add_parser(
        "surprise", help="run patch-surprise evaluation"
    )
    surprise.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted training checkpoint path",
    )
    surprise.add_argument(
        "--data",
        type=Path,
        required=True,
        help="packed dataset directory or manifest.json",
    )
    surprise.add_argument(
        "--out", type=Path, required=True, help="surprise report artifact directory"
    )
    surprise.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    surprise.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="maximum held-out examples to score",
    )
    surprise.add_argument(
        "--random-decoys", type=int, default=1, help="random decoys per example"
    )
    surprise.add_argument(
        "--same-file-decoys", type=int, default=1, help="same-file decoys per example"
    )
    surprise.add_argument(
        "--mutation-decoys", type=int, default=1, help="mutation decoys per example"
    )
    surprise.add_argument(
        "--action-cluster-decoys",
        type=int,
        default=1,
        help="action-cluster decoys per example",
    )
    surprise.add_argument(
        "--seed", type=int, default=0, help="deterministic evaluation seed"
    )
    surprise.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing surprise output files",
    )
    surprise.add_argument("--json", action="store_true", help="emit JSON output")
    surprise.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    surprise.set_defaults(func=_eval_surprise_command)
    ablation = eval_subcommands.add_parser(
        "ablation", help="build an action-view ablation report"
    )
    ablation.add_argument(
        "--retrieval-artifact",
        type=Path,
        required=True,
        help="retrieval artifact manifest.json",
    )
    ablation.add_argument(
        "--training-artifact",
        type=Path,
        required=True,
        help="training artifact manifest.json",
    )
    ablation.add_argument(
        "--out", type=Path, required=True, help="ablation report artifact directory"
    )
    ablation.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing ablation output files",
    )
    ablation.add_argument("--json", action="store_true", help="emit JSON output")
    ablation.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    ablation.set_defaults(func=_eval_ablation_command)
    scorer_quality = eval_subcommands.add_parser(
        "scorer-quality",
        help="run scorer/reranker quality evaluation",
    )
    scorer_quality.add_argument(
        "--config", type=Path, required=True, help="scorer quality config JSON path"
    )
    scorer_quality.add_argument(
        "--checkpoint", type=Path, required=True, help="checkpoint file"
    )
    scorer_quality.add_argument(
        "--out", type=Path, required=True, help="quality report artifact directory"
    )
    scorer_quality.add_argument(
        "--device", default="auto", choices=("cpu", "cuda", "mps", "auto")
    )
    scorer_quality.add_argument(
        "--index",
        type=Path,
        help="transition index directory for retrieval-prior scoring",
    )
    scorer_quality.add_argument(
        "--retrieval-prior-weight",
        type=float,
        default=0.0,
        help="non-negative weight applied to the retrieval prior",
    )
    scorer_quality.add_argument(
        "--retrieval-prior-k",
        type=int,
        default=10,
        help="nearest index hits used for the prior",
    )
    scorer_quality.add_argument(
        "--parent-manifest",
        action="append",
        type=Path,
        default=[],
        help="required parent artifact manifest to verify and record; may be repeated",
    )
    scorer_quality.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing quality output files",
    )
    scorer_quality.add_argument("--json", action="store_true", help="emit JSON output")
    scorer_quality.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    scorer_quality.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="load the checkpoint without verifying its manifest (trusted local use only)",
    )
    scorer_quality.set_defaults(func=_eval_scorer_quality_command)
    downstream_pack = eval_subcommands.add_parser(
        "downstream-pack",
        help="build a manifest-backed downstream candidate-reranking benchmark pack",
    )
    downstream_pack.add_argument(
        "--config", type=Path, required=True, help="downstream benchmark pack config JSON"
    )
    downstream_pack.add_argument(
        "--out", type=Path, required=True, help="benchmark pack artifact directory"
    )
    downstream_pack.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing benchmark pack output files",
    )
    downstream_pack.add_argument("--json", action="store_true", help="emit JSON output")
    downstream_pack.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    downstream_pack.set_defaults(func=_eval_downstream_pack_command)
    downstream_rerank = eval_subcommands.add_parser(
        "downstream-rerank",
        help="run downstream candidate-reranking evaluation and claim gate",
    )
    downstream_rerank.add_argument(
        "--benchmark-manifest",
        type=Path,
        required=True,
        help="downstream benchmark artifact manifest.json",
    )
    downstream_rerank.add_argument("--checkpoint", type=Path, required=True, help="checkpoint file")
    downstream_rerank.add_argument(
        "--out", type=Path, required=True, help="downstream rerank report artifact directory"
    )
    downstream_rerank.add_argument(
        "--candidate-pack-manifest",
        action="append",
        type=Path,
        default=[],
        help="candidate-pack artifact manifest to verify and record; may be repeated",
    )
    downstream_rerank.add_argument(
        "--device", default="auto", choices=("cpu", "cuda", "mps", "auto")
    )
    downstream_rerank.add_argument(
        "--index",
        type=Path,
        help="transition index directory for retrieval-prior scoring",
    )
    downstream_rerank.add_argument(
        "--retrieval-prior-weight",
        type=float,
        default=0.0,
        help="non-negative weight applied to the retrieval prior",
    )
    downstream_rerank.add_argument(
        "--retrieval-prior-k",
        type=int,
        default=10,
        help="nearest index hits used for the prior",
    )
    downstream_rerank.add_argument("--pass-at-k", type=int, default=5, help="k for pass@k")
    downstream_rerank.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="bootstrap samples for confidence intervals when sample count permits",
    )
    downstream_rerank.add_argument("--seed", type=int, default=0, help="deterministic evaluation seed")
    downstream_rerank.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="load the checkpoint without verifying its manifest (trusted local use only)",
    )
    downstream_rerank.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing downstream rerank output files",
    )
    downstream_rerank.add_argument("--json", action="store_true", help="emit JSON output")
    downstream_rerank.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    downstream_rerank.set_defaults(func=_eval_downstream_rerank_command)
    p_pass_calibration = eval_subcommands.add_parser(
        "p-pass-calibration",
        help="build a held-out p_pass correctness calibration report",
    )
    p_pass_calibration.add_argument(
        "--scores",
        action="append",
        type=Path,
        required=True,
        help="completion score JSONL path; repeatable",
    )
    p_pass_calibration.add_argument(
        "--parent-manifest",
        action="append",
        type=Path,
        required=True,
        help="verified parent artifact manifest for score-row lineage; repeatable",
    )
    p_pass_calibration.add_argument(
        "--dataset-kind",
        required=True,
        choices=P_PASS_DATASET_KINDS,
        help="kind of held-out rows represented by the score file",
    )
    p_pass_calibration.add_argument(
        "--out",
        type=Path,
        required=True,
        help="p_pass calibration report artifact directory",
    )
    p_pass_calibration.add_argument(
        "--baseline",
        action="append",
        default=[],
        help=(
            "score key to evaluate; repeatable. Defaults to "
            + ",".join(P_PASS_DEFAULT_BASELINES)
        ),
    )
    p_pass_calibration.add_argument(
        "--benchmark",
        help="default benchmark id for score rows that omit benchmark metadata",
    )
    p_pass_calibration.add_argument(
        "--calibration-bin-count",
        type=int,
        default=10,
        help="number of equal-width probability bins used for ECE",
    )
    p_pass_calibration.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing p_pass calibration output files",
    )
    p_pass_calibration.add_argument("--json", action="store_true", help="emit JSON output")
    p_pass_calibration.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    p_pass_calibration.set_defaults(func=_eval_p_pass_calibration_command)
    _add_execution_completion_rerank_parser(
        eval_subcommands,
        name="rerank-humaneval",
        benchmark="humaneval",
        help_text="score and rerank HumanEval completion-label artifacts",
    )
    _add_execution_completion_rerank_parser(
        eval_subcommands,
        name="rerank-mbpp-plus",
        benchmark="mbpp_plus",
        help_text="score and rerank MBPP-Plus completion-label artifacts",
    )
    eval_parser.set_defaults(func=_eval_help_command, eval_parser=eval_parser)
    index = subparsers.add_parser("index", help="build a transition index artifact")
    index.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted training checkpoint path",
    )
    index.add_argument(
        "--data",
        type=Path,
        required=True,
        help="packed dataset directory or manifest.json",
    )
    index.add_argument(
        "--out", type=Path, required=True, help="transition index artifact directory"
    )
    index.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    index.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="number of train rows to embed per index batch",
    )
    index.add_argument(
        "--distance",
        default="l2",
        choices=("l2", "cosine"),
        help="index distance metric",
    )
    index.add_argument(
        "--name",
        default="codelewm-train-index",
        help="index name recorded in index.json",
    )
    index.add_argument(
        "--overwrite", action="store_true", help="overwrite existing index output files"
    )
    index.add_argument("--json", action="store_true", help="emit JSON output")
    index.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    index.set_defaults(func=_index_command)
    dataset = subparsers.add_parser(
        "dataset", help="dataset build and packing utilities"
    )
    dataset_subcommands = dataset.add_subparsers(dest="dataset_command")
    build = dataset_subcommands.add_parser(
        "build", help="build a transition dataset artifact"
    )
    build.add_argument(
        "--config",
        type=Path,
        required=True,
        help="dataset build config JSON or YAML path",
    )
    build.add_argument(
        "--out", type=Path, required=True, help="empty output artifact directory"
    )
    build.add_argument("--json", action="store_true", help="emit JSON output")
    build.set_defaults(func=_dataset_build_command)
    pack = dataset_subcommands.add_parser(
        "pack", help="pack built transition rows for training"
    )
    pack.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="dataset build artifact manifest path",
    )
    pack.add_argument(
        "--out", type=Path, required=True, help="empty packed dataset output directory"
    )
    pack.add_argument("--json", action="store_true", help="emit JSON output")
    pack.set_defaults(func=_dataset_pack_command)
    execute = dataset_subcommands.add_parser(
        "execute",
        help=(
            "run one (code, input) pair in the sandboxed deterministic Python "
            "executor and emit a JSON result"
        ),
    )
    execute.add_argument(
        "--code-file",
        type=Path,
        required=True,
        help="path to a Python source file containing the payload",
    )
    execute.add_argument(
        "--input-file",
        type=Path,
        help=(
            "path to a JSON file holding the function arguments. Required when "
            "--function-name is provided."
        ),
    )
    execute.add_argument(
        "--function-name",
        type=str,
        default=None,
        help=(
            "name of a function defined by the payload to invoke. If omitted, the "
            "payload runs script-style and its stdout becomes the captured output."
        ),
    )
    execute.add_argument(
        "--stdin-file",
        type=Path,
        default=None,
        help="path to a text file fed to the payload's stdin (script-style only)",
    )
    execute.add_argument(
        "--policy",
        choices=("stdlib-only",),
        default="stdlib-only",
        help="sandbox policy preset (only stdlib-only is supported)",
    )
    execute.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="wall-clock timeout in milliseconds (10..60000)",
    )
    execute.add_argument(
        "--memory-mb",
        type=int,
        default=256,
        help="RSS cap in megabytes (16..4096)",
    )
    execute.add_argument(
        "--cpu-seconds",
        type=int,
        default=10,
        help="CPU-time cap in seconds (1..120)",
    )
    execute.add_argument(
        "--no-determinism-check",
        dest="determinism_check",
        action="store_false",
        default=True,
        help="skip the determinism re-run (default: enabled)",
    )
    execute.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help=(
            "directory the sandboxed process may write to. A fresh temporary "
            "directory is used when omitted."
        ),
    )
    execute.add_argument("--json", action="store_true", help="emit JSON output")
    execute.set_defaults(func=_dataset_execute_command)
    ingest = dataset_subcommands.add_parser(
        "ingest",
        help=(
            "ingest one execution-substrate upstream source into a normalized "
            "JSONL of SourceSubmission records (codenet/mbpp/mbpp_plus/apps/humaneval)"
        ),
    )
    ingest.add_argument(
        "--source",
        choices=("codenet", "mbpp", "mbpp_plus", "apps", "humaneval"),
        required=True,
        help="upstream dataset name",
    )
    ingest.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        required=True,
        help="path to the upstream flattened JSONL or directory the adapter expects",
    )
    ingest.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output JSONL path for normalized SourceSubmission records",
    )
    ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of submissions written (default: no limit)",
    )
    ingest.add_argument("--json", action="store_true", help="emit JSON output")
    ingest.set_defaults(func=_dataset_ingest_command)
    execution_pack = dataset_subcommands.add_parser(
        "execution-pack",
        help=(
            "build the execution-substrate pack (pack.jsonl + manifest + sidecars) "
            "from one or more ingestion JSONL files"
        ),
    )
    execution_pack.add_argument(
        "--ingestion",
        dest="ingestion_paths",
        type=Path,
        action="append",
        required=True,
        help="ingestion JSONL path (from `codelewm dataset ingest`); repeatable",
    )
    execution_pack.add_argument(
        "--output",
        type=Path,
        required=True,
        help="empty output directory for the pack",
    )
    execution_pack.add_argument(
        "--seed", type=int, default=42, help="split RNG seed"
    )
    execution_pack.add_argument(
        "--train-frac", type=float, default=0.85, help="fraction of problems in train"
    )
    execution_pack.add_argument(
        "--val-frac", type=float, default=0.05, help="fraction of problems in val"
    )
    execution_pack.add_argument(
        "--max-inputs-per-problem",
        type=int,
        default=None,
        help="cap inputs kept per problem (default: no cap)",
    )
    execution_pack.add_argument(
        "--target-records",
        type=int,
        default=None,
        help="stop early once this many records have been written",
    )
    execution_pack.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="sandbox per-invocation wall-clock budget (ms)",
    )
    execution_pack.add_argument(
        "--memory-mb", type=int, default=1024, help="sandbox memory cap (MB)"
    )
    execution_pack.add_argument(
        "--cpu-seconds",
        type=int,
        default=10,
        help="sandbox CPU-time cap (seconds)",
    )
    execution_pack.add_argument(
        "--no-determinism-check",
        dest="determinism_check",
        action="store_false",
        default=True,
        help="skip the sandbox determinism re-run (default: enabled)",
    )
    execution_pack.add_argument(
        "--json", action="store_true", help="emit JSON output"
    )
    execution_pack.set_defaults(func=_dataset_execution_pack_command)
    dataset.set_defaults(func=_dataset_help_command, dataset_parser=dataset)
    manifest = subparsers.add_parser("manifest", help="artifact manifest utilities")
    manifest_subcommands = manifest.add_subparsers(dest="manifest_command")
    verify = manifest_subcommands.add_parser(
        "verify", help="verify an artifact manifest"
    )
    verify.add_argument(
        "--manifest", type=Path, required=True, help="artifact manifest JSON path"
    )
    verify.add_argument(
        "--root",
        type=Path,
        help="artifact root directory; defaults to the manifest parent",
    )
    verify.add_argument(
        "--parent-manifest",
        type=Path,
        action="append",
        default=None,
        help="parent artifact manifest required by this artifact; repeatable",
    )
    verify.add_argument("--json", action="store_true", help="emit JSON output")
    verify.set_defaults(func=_manifest_verify_command)
    manifest.set_defaults(func=_manifest_help_command, manifest_parser=manifest)
    secret_scan = subparsers.add_parser(
        "secret-scan",
        help="scan files for secret patterns and emit a redacted JSON report",
    )
    secret_scan.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="files or directories to scan",
    )
    secret_scan.add_argument(
        "--include-suffix",
        action="append",
        default=None,
        help="restrict scan to files with this suffix; repeatable. Defaults match log/report files.",
    )
    secret_scan.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        default=True,
        help="do not recurse into subdirectories",
    )
    secret_scan.add_argument(
        "--json",
        action="store_true",
        help="emit JSON output",
    )
    secret_scan.set_defaults(func=_secret_scan_command)
    parser.set_defaults(func=_print_help)
    return parser


def _add_execution_completion_rerank_parser(
    eval_subcommands: Any,
    *,
    name: str,
    benchmark: str,
    help_text: str,
) -> None:
    rerank = eval_subcommands.add_parser(name, help=help_text)
    rerank.add_argument(
        "--completion-manifest",
        type=Path,
        required=True,
        help="completion-label artifact manifest.json",
    )
    rerank.add_argument(
        "--labels",
        type=Path,
        help="override completion labels JSONL path; defaults to the manifest-listed file",
    )
    rerank.add_argument("--checkpoint", type=Path, required=True, help="checkpoint file")
    rerank.add_argument(
        "--checkpoint-manifest",
        type=Path,
        help="trusted checkpoint manifest; defaults to <checkpoint>.manifest.json",
    )
    rerank.add_argument(
        "--out", type=Path, required=True, help="execution rerank report artifact directory"
    )
    rerank.add_argument(
        "--device", default="auto", choices=("cpu", "cuda", "mps", "auto")
    )
    rerank.add_argument(
        "--index",
        type=Path,
        help="transition index directory for retrieval-prior scoring",
    )
    rerank.add_argument(
        "--retrieval-prior-weight",
        type=float,
        default=0.0,
        help="non-negative weight applied to the retrieval prior",
    )
    rerank.add_argument(
        "--retrieval-prior-k",
        type=int,
        default=10,
        help="nearest index hits used for the prior",
    )
    rerank.add_argument("--pass-at-k", type=int, default=5, help="k for pass@k")
    rerank.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="bootstrap samples for confidence intervals",
    )
    rerank.add_argument("--seed", type=int, default=17, help="deterministic bootstrap seed")
    rerank.add_argument(
        "--min-lift-for-claim",
        type=float,
        default=3.0,
        help="minimum pass@1 lift in percentage points required for a usefulness claim",
    )
    rerank.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="load the checkpoint without verifying its manifest (trusted local use only)",
    )
    rerank.add_argument(
        "--require-learned-scorer",
        action="store_true",
        help="fail instead of using the deterministic fixture scorer",
    )
    rerank.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    rerank.add_argument("--json", action="store_true", help="emit JSON output")
    rerank.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    rerank.set_defaults(
        func=_eval_execution_completion_rerank_command,
        benchmark=benchmark,
    )


def _print_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0


def _manifest_help_command(args: argparse.Namespace) -> int:
    args.manifest_parser.print_help()
    return 0


def _dataset_help_command(args: argparse.Namespace) -> int:
    args.dataset_parser.print_help()
    return 0


def _eval_help_command(args: argparse.Namespace) -> int:
    args.eval_parser.print_help()
    return 0


def _model_help_command(args: argparse.Namespace) -> int:
    args.model_parser.print_help()
    return 0


def _openrouter_help_command(args: argparse.Namespace) -> int:
    args.openrouter_parser.print_help()
    return 0


def _openrouter_byok_register_command(args: argparse.Namespace) -> int:
    try:
        result = register_openrouter_byok_credential(
            provider=args.provider,
            key_env=args.key_env,
            management_key_env=args.management_key_env,
            name=args.name,
            allowed_models=tuple(args.allowed_model or ()),
            workspace_id=args.workspace_id,
            is_fallback=args.fallback,
            dry_run=args.dry_run,
        )
    except OpenRouterAdapterError as exc:
        if args.json:
            print(json.dumps(exc.to_error_report(), indent=2, sort_keys=True))
        else:
            print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        status = "dry-run" if result.dry_run else "registered"
        print(f"status: {status}")
        print(f"provider: {result.provider}")
        print(f"key_env: {result.key_env}")
        print(f"management_key_env: {result.management_key_env}")
        if result.credential_name:
            print(f"name: {result.credential_name}")
    return 0


def _score_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    try:
        instruction = _instruction_arg_to_text(args.instruction)
        _emit_cli_log(
            args,
            LogEvent(
                event="harness.score.start",
                level="info",
                run_id=run_id,
                step="score",
                message="score command started",
                fields={
                    "before": str(args.before),
                    "candidate": str(args.candidate),
                    "checkpoint": str(args.checkpoint),
                    "device": args.device,
                    "index": None if args.index is None else str(args.index),
                    "retrieval_prior_weight": args.retrieval_prior_weight,
                    "retrieval_prior_k": args.retrieval_prior_k,
                    "instruction_sha256": _sha256_text(instruction),
                },
            ),
        )
        scorer = load_scorer(
            args.checkpoint,
            device=args.device,
            allow_unsafe=args.allow_unsafe_checkpoint,
            index=args.index,
            retrieval_prior_weight=args.retrieval_prior_weight,
            retrieval_prior_k=args.retrieval_prior_k,
        )
        result = scorer.score_files(
            before=args.before,
            instruction=instruction,
            candidate=args.candidate,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="harness.score.complete",
                level="info",
                run_id=run_id,
                step="score",
                message="score command completed",
                fields={"result": result.to_dict()},
            ),
        )
    except ScoreError as exc:
        _emit_error_log(
            args, run_id=run_id, step="score", event="harness.score.error", exc=exc
        )
        if args.json:
            print(json.dumps(exc.to_error_report().to_dict(), indent=2, sort_keys=True))
        else:
            print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"candidate: {result.candidate}")
        print(f"transition_energy: {result.transition_energy:.6g}")
        print(f"final_score: {result.final_score:.6g}")
    return 0


def _llm_demo_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    if args.json and args.tui:
        error = ScoreError(
            "`codelewm llm-demo --json --tui` is ambiguous",
            error_type="config_error",
            remediation="use --json for machine output or --tui for the interactive Textual viewer",
            artifact=str(args.out),
        )
        _emit_error(args, error, json_output=True)
        return 2
    try:
        instruction = _instruction_arg_to_text(args.instruction)
        _emit_cli_log(
            args,
            LogEvent(
                event="harness.llm_demo.start",
                level="info",
                run_id=run_id,
                step="llm_demo",
                message="LLM demo command started",
                fields={
                    "before": str(args.before),
                    "checkpoint": str(args.checkpoint),
                    "out": str(args.out),
                    "task_id": args.task_id,
                    "context_path": args.context_path,
                    "device": args.device,
                    "index": None if args.index is None else str(args.index),
                    "retrieval_prior_weight": args.retrieval_prior_weight,
                    "retrieval_prior_k": args.retrieval_prior_k,
                    "checkpoint_inspection_manifest": None
                    if args.checkpoint_inspection_manifest is None
                    else str(args.checkpoint_inspection_manifest),
                    "latent_matrix_manifest": None
                    if args.latent_matrix_manifest is None
                    else str(args.latent_matrix_manifest),
                    "tensorboard_manifest": None
                    if args.tensorboard_manifest is None
                    else str(args.tensorboard_manifest),
                    "require_learned_scorer": args.require_learned_scorer,
                    "instruction_sha256": _sha256_text(instruction),
                },
            ),
        )
        result = run_llm_world_model_demo(
            before=args.before,
            instruction=instruction,
            checkpoint=args.checkpoint,
            out=args.out,
            task_id=args.task_id,
            context_path=args.context_path,
            device=args.device,
            index=args.index,
            retrieval_prior_weight=args.retrieval_prior_weight,
            retrieval_prior_k=args.retrieval_prior_k,
            parent_manifests=tuple(args.parent_manifest or ()),
            checkpoint_inspection_manifest=args.checkpoint_inspection_manifest,
            checkpoint_inspection_report=args.checkpoint_inspection_report,
            latent_matrix_manifest=args.latent_matrix_manifest,
            latent_matrix_report=args.latent_matrix_report,
            tensorboard_manifest=args.tensorboard_manifest,
            tensorboard_export=args.tensorboard_export,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            require_learned_scorer=args.require_learned_scorer,
            overwrite=args.overwrite,
            command=(
                "codelewm",
                "llm-demo",
                "--before",
                str(args.before),
                "--checkpoint",
                str(args.checkpoint),
                "--out",
                str(args.out),
            ),
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="harness.llm_demo.complete",
                level="info",
                run_id=run_id,
                step="llm_demo",
                message="LLM demo command completed",
                fields=result.to_dict(),
            ),
        )
    except (LLMWorldModelDemoError, ScoreError) as exc:
        error = ScoreError(
            f"LLM demo failed: {exc}",
            error_type="scoring_error",
            remediation="inspect the demo inputs, candidate pack, checkpoint, and output directory",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="llm_demo", event="harness.llm_demo.error", exc=error
        )
        if args.json:
            print(json.dumps(error.to_error_report().to_dict(), indent=2, sort_keys=True))
        else:
            print(str(error), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    elif args.tui:
        try:
            from codelewm.harness.demo_tui import TextualDemoTuiError, run_demo_tui

            return run_demo_tui(args.out / result.visual_view_model_path)
        except TextualDemoTuiError as exc:
            error = ScoreError(
                f"LLM demo TUI failed: {exc}",
                error_type=exc.error_type,  # type: ignore[arg-type]
                remediation=exc.remediation,
                artifact=str(args.out / result.visual_view_model_path),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            )
            _emit_error(args, error, json_output=False)
            return 2
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"demo_report: {args.out / result.report_path}")
        print(f"candidate_pack_manifest: {args.out / result.candidate_pack_manifest_path}")
        print(f"success: {result.success}")
    return 0


def _llm_demo_tui_command(args: argparse.Namespace) -> int:
    try:
        from codelewm.harness.demo_tui import (
            TextualDemoTuiError,
            build_demo_tui_snapshot,
            load_demo_tui_view_model,
            resolve_demo_tui_view_model_path,
            run_demo_tui,
        )

        path = resolve_demo_tui_view_model_path(
            view_model=args.view_model,
            demo_dir=args.demo_dir,
        )
        if args.snapshot_json:
            payload = build_demo_tui_snapshot(load_demo_tui_view_model(path))
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        return run_demo_tui(path)
    except TextualDemoTuiError as exc:
        error = ScoreError(
            f"LLM demo TUI failed: {exc}",
            error_type=exc.error_type,  # type: ignore[arg-type]
            remediation=exc.remediation,
            artifact=str(args.view_model or args.demo_dir or ""),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=bool(args.json or args.snapshot_json))
        return 2


def _execution_rerank_demo_command(args: argparse.Namespace) -> int:
    if args.json and args.tui:
        error = ScoreError(
            "`codelewm execution-rerank-demo --json --tui` is ambiguous",
            error_type="config_error",
            remediation="use --json for machine output or --tui for the interactive Textual viewer",
            artifact=str(args.out),
        )
        _emit_error(args, error, json_output=True)
        return 2
    try:
        from codelewm.harness.execution_rerank_demo import (
            ExecutionRerankDemoError,
            render_execution_rerank_tour_terminal,
            run_execution_rerank_tour,
        )

        result = run_execution_rerank_tour(
            checkpoint=args.checkpoint,
            out=args.out,
            tour_count=args.tour,
            scenario_id=args.scenario,
            device=args.device,
            html_export=args.html,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            require_learned_scorer=not args.fixture_scorer,
            overwrite=args.overwrite,
            command=(
                "codelewm",
                "execution-rerank-demo",
                "--scenario",
                args.scenario,
                "--tour",
                str(args.tour),
            ),
        )
    except (ExecutionRerankDemoError, ScoreError) as exc:
        error = ScoreError(
            f"execution-rerank tour failed: {exc}",
            error_type="scoring_error",
            remediation="inspect the tour inputs, checkpoint, and output directory",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.tui:
        try:
            from codelewm.harness.execution_rerank_tui import (
                ExecutionRerankTuiError,
                run_execution_rerank_tui,
            )

            return run_execution_rerank_tui(args.out / result.view_model_path)
        except ExecutionRerankTuiError as exc:
            error = ScoreError(
                f"execution-rerank TUI failed: {exc}",
                error_type=exc.error_type,  # type: ignore[arg-type]
                remediation=exc.remediation,
                artifact=str(args.out / result.view_model_path),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            )
            _emit_error(args, error, json_output=False)
            return 2
    report_path = args.out / result.report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(render_execution_rerank_tour_terminal(report))
    print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
    print(f"demo_report: {report_path}")
    print(f"view_model: {args.out / result.view_model_path}")
    print(f"html: {args.out / result.html_path}")
    if result.html_export_path is not None:
        print(f"html_export: {result.html_export_path}")
    return 0


def _execution_rerank_tui_command(args: argparse.Namespace) -> int:
    try:
        from codelewm.harness.execution_rerank_tui import (
            ExecutionRerankTuiError,
            build_execution_rerank_tui_snapshot,
            load_execution_rerank_tui_view_model,
            resolve_execution_rerank_tui_view_model_path,
            run_execution_rerank_tui,
        )

        path = resolve_execution_rerank_tui_view_model_path(
            view_model=args.view_model,
            demo_dir=args.demo_dir,
        )
        if args.snapshot_json:
            payload = build_execution_rerank_tui_snapshot(
                load_execution_rerank_tui_view_model(path)
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        return run_execution_rerank_tui(path)
    except ExecutionRerankTuiError as exc:
        error = ScoreError(
            f"execution-rerank TUI failed: {exc}",
            error_type=exc.error_type,  # type: ignore[arg-type]
            remediation=exc.remediation,
            artifact=str(args.view_model or args.demo_dir or ""),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=bool(args.json or args.snapshot_json))
        return 2


def _dataset_build_command(args: argparse.Namespace) -> int:
    command = (
        "codelewm",
        "dataset",
        "build",
        "--config",
        str(args.config),
        "--out",
        str(args.out),
        *(("--json",) if args.json else ()),
    )
    try:
        result = build_dataset_from_config_path(
            config_path=args.config,
            output_dir=args.out,
            command=command,
        )
    except DatasetBuildConfigError as exc:
        error = ScoreError(
            f"dataset build config is invalid: {exc}",
            error_type="config_error",
            remediation="repair the dataset build config and retry",
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except SourceUnavailableError as exc:
        error = ScoreError(
            f"dataset source is unavailable: {exc}",
            error_type="source_unavailable",
            remediation="check source paths and adapter options, then retry",
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 3
    except SourceRecordError as exc:
        error = ScoreError(
            f"dataset source row is invalid: {exc}",
            error_type="dataset_build_error",
            remediation="repair or filter the malformed source row and retry",
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 4
    except DatasetBuildError as exc:
        error_type = (
            "empty_dataset"
            if "zero kept transitions" in str(exc)
            else "dataset_build_error"
        )
        error = ScoreError(
            f"dataset build failed: {exc}",
            error_type=error_type,
            remediation="inspect the dataset reports, filters, and split policy, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 4
    except OSError as exc:
        error = ScoreError(
            f"dataset build failed while writing artifacts: {exc}",
            error_type="dataset_build_error",
            remediation="choose a writable output directory and retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 4

    if args.json:
        print(json.dumps(result.to_report(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {result.output_dir / 'manifest.json'}")
        print(f"dataset_manifest: {result.output_dir / 'dataset_manifest.json'}")
        print(f"row_count: {result.dataset_manifest.row_count}")
    return 0


def _train_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _train_command_tuple(args)
    tensorboard_enabled = bool(args.tensorboard or args.tensorboard_dir is not None)
    schema_version = peek_train_config_schema_version(args.config)
    if schema_version == EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION:
        return _train_execution_command(args, run_id=run_id, command=command)
    try:
        config = _load_cli_train_config(args)
        _validate_train_cli_executor(args)
        _emit_cli_log(
            args,
            LogEvent(
                event="training.start",
                level="info",
                run_id=run_id,
                step="train",
                message="train command started",
                fields={
                    "config": str(args.config),
                    "out": None if args.out is None else str(args.out),
                    "executor": args.executor,
                    "device": args.device or "config",
                    "resume_from": None
                    if args.resume_from is None
                    else str(args.resume_from),
                    "tensorboard": tensorboard_enabled,
                    "tensorboard_dir": None
                    if args.tensorboard_dir is None
                    else str(args.tensorboard_dir),
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        executor = (
            make_torch_training_executor(
                device=args.device,
                tensorboard=tensorboard_enabled,
                tensorboard_dir=args.tensorboard_dir,
            )
            if args.executor == "torch"
            else cpu_smoke_training_executor
        )
        manifest = train(
            config,
            root=Path.cwd(),
            executor=executor,
            command=command,
            overwrite=args.overwrite,
            resume_from=args.resume_from,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="training.complete",
                level="info",
                run_id=run_id,
                artifact_id=manifest.artifact_manifest_id,
                step="train",
                message="train command completed",
                fields={
                    "run_id": manifest.run_id,
                    "step_count": manifest.step_count,
                    "final_metrics": dict(manifest.final_metrics),
                    "artifact_manifest_path": manifest.artifact_manifest_path,
                },
            ),
        )
    except TrainConfigError as exc:
        error = ScoreError(
            f"training config is invalid: {exc}",
            error_type="config_error",
            remediation="repair the training config and retry",
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except OptionalDependencyError as exc:
        remediation = (
            "install the required groups with `uv sync --group train --group data --group dev --group observability`"
            if tensorboard_enabled
            else "install the required groups with `uv sync --group train --group data --group dev`"
        )
        error = ScoreError(
            f"training optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation=remediation,
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except SourceUnavailableError as exc:
        error = ScoreError(
            f"training source artifact is unavailable: {exc}",
            error_type="source_unavailable",
            remediation="check data.train, data.val, and data.manifest, then retry",
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 3
    except (ArtifactManifestError, json.JSONDecodeError, OSError) as exc:
        error = ScoreError(
            f"training manifest or artifact validation failed: {exc}",
            error_type="manifest_error",
            remediation="verify the dataset or parent training manifest and retry",
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except TrainingRunError as exc:
        error, exit_code = _training_run_error(args, exc)
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return exit_code
    except Exception as exc:
        error = ScoreError(
            f"training failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the training logs and retry with a corrected config",
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    else:
        run_dir = Path(config.output.run_dir)
        print(f"run_id: {manifest.run_id}")
        print(f"artifact_manifest: {run_dir / manifest.artifact_manifest_path}")
        print(f"training_manifest: {config.output.manifest_path}")
        print(f"metrics: {config.output.metrics_path}")
        print(f"step_count: {manifest.step_count}")
    return 0


def _train_execution_command(
    args: argparse.Namespace, *, run_id: str, command: tuple[str, ...]
) -> int:
    """Dispatch ``codelewm train`` against a ``codelewm.execution_train_config.v1`` file.

    The execution-substrate path is intentionally distinct from the
    legacy HDF5 path: different pack contract, different runner, and a
    different artifact directory layout. The two paths share the manifest
    + checkpoint schemas so manifest verify and the publish scripts work
    against either.
    """

    if args.resume_from is not None:
        error = ScoreError(
            "--resume-from is not supported for execution-substrate configs in v0.6",
            error_type="config_error",
            remediation=(
                "remove --resume-from; checkpoint resume for the execution "
                "runner ships in a follow-on issue"
            ),
            artifact=str(args.config),
            caused_by="ExecutionTrainConfigError: --resume-from not implemented",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    if args.executor != "torch":
        error = ScoreError(
            "execution-substrate configs only run with --executor torch",
            error_type="config_error",
            remediation="drop --executor cpu-smoke or pass --executor torch explicitly",
            artifact=str(args.config),
            caused_by="ExecutionTrainConfigError: cpu-smoke executor not supported",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 2

    try:
        execution_config = load_execution_train_config(args.config)
    except ExecutionTrainConfigError as exc:
        error = ScoreError(
            f"execution-train config is invalid: {exc}",
            error_type="config_error",
            remediation="repair the execution-train config and retry",
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 2

    seed = args.seed if args.seed is not None else execution_config.seeds[0]
    if seed not in execution_config.seeds:
        # Allow off-list seeds but emit a warning so the operator can
        # tell from the log whether the run will count toward the
        # claim-gate variance requirement.
        _emit_cli_log(
            args,
            LogEvent(
                event="training.warning",
                level="warning",
                run_id=run_id,
                step="train",
                message=(
                    f"seed {seed} is not in the config's declared seeds list "
                    f"{list(execution_config.seeds)!r}; this run will not count "
                    "toward the required-seeds claim gate"
                ),
                fields={"seed": seed, "declared_seeds": list(execution_config.seeds)},
            ),
        )

    tensorboard_override: bool | None
    if args.tensorboard:
        tensorboard_override = True
    elif args.tensorboard_dir is not None:
        tensorboard_override = True
    else:
        tensorboard_override = None  # let the config decide

    output_dir = args.out if args.out is not None else (
        Path("runs") / "v0_6" / f"seed-{seed}"
    )

    _emit_cli_log(
        args,
        LogEvent(
            event="training.start",
            level="info",
            run_id=run_id,
            step="train",
            message="execution-substrate train command started",
            fields={
                "config": str(args.config),
                "schema_version": EXECUTION_TRAIN_CONFIG_SCHEMA_VERSION,
                "seed": seed,
                "out": str(output_dir),
                "pack_local_dir": (
                    None if args.pack_local_dir is None else str(args.pack_local_dir)
                ),
                "device": args.device or execution_config.trainer.accelerator,
                "tensorboard": (
                    execution_config.trainer.tensorboard_enabled
                    if tensorboard_override is None
                    else tensorboard_override
                ),
            },
        ),
    )

    try:
        result = train_execution_run(
            execution_config,
            seed=seed,
            output_dir=output_dir,
            root=Path.cwd(),
            command=command,
            overwrite=args.overwrite,
            pack_local_dir=args.pack_local_dir,
            device=args.device,
            tensorboard=tensorboard_override,
            tensorboard_dir=args.tensorboard_dir,
        )
    except OptionalDependencyError as exc:
        remediation = (
            "install the required groups with "
            "`uv sync --group train --group data --group dev --group observability --group release`"
        )
        error = ScoreError(
            f"execution-substrate training optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation=remediation,
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except SourceUnavailableError as exc:
        error = ScoreError(
            f"execution pack is unavailable: {exc}",
            error_type="source_unavailable",
            remediation=(
                "set CODELEWM_EXECUTION_PACK_LOCAL_DIR to the pack directory, "
                "or ensure HF_TOKEN is exported and the pack repo is reachable"
            ),
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 3
    except TrainingRunError as exc:
        runtime_error, exit_code = _training_run_error(args, exc)
        _emit_error_log(
            args,
            run_id=run_id,
            step="train",
            event="training.error",
            exc=runtime_error,
        )
        _emit_error(args, runtime_error, json_output=args.json)
        return exit_code
    except Exception as exc:
        error = ScoreError(
            f"execution-substrate training failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation=(
                "inspect the training logs and the config; verify pack contents "
                "are well-formed"
            ),
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="train", event="training.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    manifest = result.training_manifest
    _emit_cli_log(
        args,
        LogEvent(
            event="training.complete",
            level="info",
            run_id=run_id,
            artifact_id=manifest.artifact_manifest_id,
            step="train",
            message="execution-substrate train command completed",
            fields={
                "run_id": manifest.run_id,
                "seed": seed,
                "step_count": manifest.step_count,
                "final_metrics": dict(manifest.final_metrics),
                "artifact_manifest_path": str(result.artifact_manifest_path),
                "training_manifest_path": str(result.training_manifest_path),
            },
        ),
    )

    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"run_id: {manifest.run_id}")
        print(f"seed: {seed}")
        print(f"artifact_manifest: {result.artifact_manifest_path}")
        print(f"training_manifest: {result.training_manifest_path}")
        print(f"metrics: {result.metrics_path}")
        print(f"step_count: {manifest.step_count}")
        for ckpt in result.checkpoint_paths:
            print(f"checkpoint: {ckpt}")
    return 0


def _model_inspect_checkpoint_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _model_inspect_checkpoint_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="model.checkpoint_inspection.start",
                level="info",
                run_id=run_id,
                step="model.inspect_checkpoint",
                message="checkpoint inspection started",
                fields={
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_manifest": None
                    if args.checkpoint_manifest is None
                    else str(args.checkpoint_manifest),
                    "out": str(args.out),
                    "parent_manifest_count": len(args.parent_manifest),
                    "allow_unsafe_checkpoint": bool(args.allow_unsafe_checkpoint),
                    "histogram_bins": args.histogram_bins,
                    "max_histogram_tensors": args.max_histogram_tensors,
                    "max_histogram_values": args.max_histogram_values,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = inspect_checkpoint(
            checkpoint=args.checkpoint,
            checkpoint_manifest=args.checkpoint_manifest,
            out=args.out,
            parent_manifests=tuple(args.parent_manifest),
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            overwrite=args.overwrite,
            histogram_bins=args.histogram_bins,
            max_histogram_tensors=args.max_histogram_tensors,
            max_histogram_values=args.max_histogram_values,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="model.checkpoint_inspection.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="model.inspect_checkpoint",
                message="checkpoint inspection completed",
                fields=result.to_dict(),
            ),
        )
    except OptionalDependencyError as exc:
        error = ScoreError(
            f"checkpoint inspection optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation="install the required groups with `uv sync --group train --group dev`",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="model.inspect_checkpoint",
            event="model.checkpoint_inspection.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except CheckpointTrustError as exc:
        error = ScoreError(
            f"checkpoint inspection rejected checkpoint: {exc}",
            error_type="checkpoint_error",
            remediation="provide a trusted checkpoint with a matching checkpoint manifest or explicitly pass --allow-unsafe-checkpoint for trusted local inspection",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="model.inspect_checkpoint",
            event="model.checkpoint_inspection.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 5
    except (ArtifactManifestError, json.JSONDecodeError, OSError) as exc:
        error = ScoreError(
            f"checkpoint inspection artifact validation failed: {exc}",
            error_type="manifest_error",
            remediation="verify the checkpoint, parent artifact manifests, and output directory, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="model.inspect_checkpoint",
            event="model.checkpoint_inspection.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except CheckpointInspectionError as exc:
        error = ScoreError(
            f"checkpoint inspection failed: {exc}",
            error_type="scoring_error",
            remediation="inspect the checkpoint payload, manifest, and output directory",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="model.inspect_checkpoint",
            event="model.checkpoint_inspection.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except Exception as exc:
        error = ScoreError(
            f"checkpoint inspection failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the checkpoint inputs and retry with a corrected request",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="model.inspect_checkpoint",
            event="model.checkpoint_inspection.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print("CodeLeWM checkpoint inspection")
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"report: {args.out / result.report_path.relative_to(result.output_dir)}")
        print(f"tensors: {result.tensor_count}")
        print(f"modules: {result.module_count}")
        print(f"parameters: {result.parameter_count}")
        print("claim_gate: diagnostic_only")
    return 0


def _eval_retrieval_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_retrieval_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.retrieval.start",
                level="info",
                run_id=run_id,
                step="eval.retrieval",
                message="retrieval evaluation started",
                fields={
                    "checkpoint": str(args.checkpoint),
                    "data": str(args.data),
                    "out": str(args.out),
                    "device": args.device,
                    "max_candidates": args.max_candidates,
                    "hard_negatives": args.hard_negatives,
                    "seed": args.seed,
                    "report_scope": args.report_scope,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = run_retrieval_evaluation(
            checkpoint=args.checkpoint,
            data=args.data,
            out=args.out,
            device=args.device,
            max_candidates=args.max_candidates,
            hard_negatives=args.hard_negatives,
            seed=args.seed,
            report_scope=args.report_scope,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.retrieval.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.retrieval",
                message="retrieval evaluation completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "metrics": result.metrics.to_dict(),
                    "parent_artifacts": list(result.parent_artifacts),
                },
            ),
        )
    except OptionalDependencyError as exc:
        error = ScoreError(
            f"retrieval evaluation optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation="install the required groups with `uv sync --group train --group data --group dev`",
            artifact=str(args.data),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.retrieval",
            event="evaluation.retrieval.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except CheckpointTrustError as exc:
        error = ScoreError(
            f"retrieval checkpoint rejected: {exc}",
            error_type="checkpoint_error",
            remediation="provide a trusted checkpoint with a matching checkpoint manifest",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.retrieval",
            event="evaluation.retrieval.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 5
    except (ArtifactManifestError, json.JSONDecodeError, OSError) as exc:
        error = ScoreError(
            f"retrieval artifact validation failed: {exc}",
            error_type="manifest_error",
            remediation="verify the checkpoint, training run, and dataset manifests, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.retrieval",
            event="evaluation.retrieval.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except (RetrievalEvalError, ActionViewPolicyError) as exc:
        error, exit_code = _retrieval_eval_error(args, exc)
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.retrieval",
            event="evaluation.retrieval.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return exit_code
    except Exception as exc:
        error = ScoreError(
            f"retrieval evaluation failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the retrieval inputs and retry with a corrected request",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.retrieval",
            event="evaluation.retrieval.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"retrieval_report: {args.out / result.report_path}")
        print(f"query_count: {result.metrics.query_count}")
        print(f"recall_at_1: {result.metrics.recall_at_1:.6g}")
        print(f"recall_at_5: {result.metrics.recall_at_5:.6g}")
        print(f"mrr: {result.metrics.mrr:.6g}")
    return 0


def _eval_execution_retrieval_command(args: argparse.Namespace) -> int:
    return _run_execution_eval_cli(
        args,
        step="eval.execution_retrieval",
        start_event="evaluation.execution_retrieval.start",
        complete_event="evaluation.execution_retrieval.complete",
        error_event="evaluation.execution_retrieval.error",
        command_builder=_eval_execution_retrieval_command_tuple,
        runner=lambda command: run_execution_retrieval_evaluation(
            checkpoint=args.checkpoint,
            pack=args.pack,
            out=args.out,
            baselines=(args.baselines,),
            device=args.device,
            max_candidates=args.max_candidates,
            seed=args.seed,
            overwrite=args.overwrite,
            command=command,
        ),
        start_fields={
            "checkpoint": str(args.checkpoint),
            "pack": str(args.pack),
            "out": str(args.out),
            "baselines": args.baselines,
            "device": args.device,
            "max_candidates": args.max_candidates,
            "seed": args.seed,
            "overwrite": bool(args.overwrite),
        },
    )


def _eval_execution_surprise_command(args: argparse.Namespace) -> int:
    return _run_execution_eval_cli(
        args,
        step="eval.execution_surprise",
        start_event="evaluation.execution_surprise.start",
        complete_event="evaluation.execution_surprise.complete",
        error_event="evaluation.execution_surprise.error",
        command_builder=_eval_execution_surprise_command_tuple,
        runner=lambda command: run_execution_surprise_evaluation(
            checkpoint=args.checkpoint,
            pack=args.pack,
            out=args.out,
            decoys=(args.decoys,),
            device=args.device,
            max_examples=args.max_examples,
            seed=args.seed,
            semantic_decoy_manifest=args.semantic_decoy_manifest,
            overwrite=args.overwrite,
            command=command,
        ),
        start_fields={
            "checkpoint": str(args.checkpoint),
            "pack": str(args.pack),
            "out": str(args.out),
            "decoys": args.decoys,
            "device": args.device,
            "max_examples": args.max_examples,
            "seed": args.seed,
            "semantic_decoy_manifest": None
            if args.semantic_decoy_manifest is None
            else str(args.semantic_decoy_manifest),
            "overwrite": bool(args.overwrite),
        },
    )


def _eval_semantic_decoy_pack_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_semantic_decoy_pack_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.semantic_decoy_pack.start",
                level="info",
                run_id=run_id,
                step="eval.semantic_decoy_pack",
                message="semantic decoy pack build started",
                fields={
                    "pack": str(args.pack),
                    "out": str(args.out),
                    "splits": args.splits,
                    "seed": args.seed,
                    "max_pairs_per_query": args.max_pairs_per_query,
                    "min_pairs_for_claim": args.min_pairs_for_claim,
                    "min_distinct_problems_for_claim": args.min_distinct_problems_for_claim,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = build_semantic_decoy_pack(
            pack=args.pack,
            out=args.out,
            splits=tuple(part.strip() for part in args.splits.split(",") if part.strip()),
            seed=args.seed,
            max_pairs_per_query=args.max_pairs_per_query,
            min_pairs_for_claim=args.min_pairs_for_claim,
            min_distinct_problems_for_claim=args.min_distinct_problems_for_claim,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.semantic_decoy_pack.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.semantic_decoy_pack",
                message="semantic decoy pack build completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "pair_rows_path": result.pair_rows_path,
                    "summary_path": result.summary_path,
                    "parent_artifacts": list(result.parent_artifacts),
                    "pair_count": result.pair_count,
                    "distinct_problem_count": result.distinct_problem_count,
                    "claim_allowed": result.claim_allowed,
                },
            ),
        )
    except (
        ArtifactManifestError,
        SemanticDecoyPackError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        error = ScoreError(
            f"semantic decoy pack build failed: {exc}",
            error_type="evaluation_gate_error",
            remediation="verify the execution pack manifest, split policy, and output directory",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.semantic_decoy_pack",
            event="evaluation.semantic_decoy_pack.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 6
    except Exception as exc:
        error = ScoreError(
            f"semantic decoy pack build failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the semantic decoy pack inputs and retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.semantic_decoy_pack",
            event="evaluation.semantic_decoy_pack.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"semantic_decoy_pairs: {args.out / result.pair_rows_path}")
        print(f"summary: {args.out / result.summary_path}")
        print(f"pair_count: {result.pair_count}")
        print(f"distinct_problem_count: {result.distinct_problem_count}")
        print(f"claim_allowed: {result.claim_allowed}")
    return 0


def _eval_execution_probe_command(args: argparse.Namespace) -> int:
    return _run_execution_eval_cli(
        args,
        step="eval.execution_probe",
        start_event="evaluation.execution_probe.start",
        complete_event="evaluation.execution_probe.complete",
        error_event="evaluation.execution_probe.error",
        command_builder=_eval_execution_probe_command_tuple,
        runner=lambda command: run_execution_probe_evaluation(
            checkpoint=args.checkpoint,
            pack=args.pack,
            out=args.out,
            targets=(args.targets,),
            device=args.device,
            max_examples_per_split=args.max_examples_per_split,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            overwrite=args.overwrite,
            command=command,
        ),
        start_fields={
            "checkpoint": str(args.checkpoint),
            "pack": str(args.pack),
            "out": str(args.out),
            "targets": args.targets,
            "device": args.device,
            "max_examples_per_split": args.max_examples_per_split,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "overwrite": bool(args.overwrite),
        },
    )


def _eval_crash_prediction_command(args: argparse.Namespace) -> int:
    return _run_execution_eval_cli(
        args,
        step="eval.crash_prediction",
        start_event="evaluation.crash_prediction.start",
        complete_event="evaluation.crash_prediction.complete",
        error_event="evaluation.crash_prediction.error",
        command_builder=_eval_crash_prediction_command_tuple,
        runner=lambda command: run_crash_prediction_evaluation(
            checkpoint=args.checkpoint,
            pack=args.pack,
            out=args.out,
            device=args.device,
            max_examples=args.max_examples,
            seed=args.seed,
            overwrite=args.overwrite,
            command=command,
        ),
        start_fields={
            "checkpoint": str(args.checkpoint),
            "pack": str(args.pack),
            "out": str(args.out),
            "device": args.device,
            "max_examples": args.max_examples,
            "seed": args.seed,
            "overwrite": bool(args.overwrite),
        },
    )


def _run_execution_eval_cli(
    args: argparse.Namespace,
    *,
    step: str,
    start_event: str,
    complete_event: str,
    error_event: str,
    command_builder: Any,
    runner: Any,
    start_fields: Mapping[str, Any],
) -> int:
    run_id = _run_id()
    command = command_builder(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event=start_event,
                level="info",
                run_id=run_id,
                step=step,
                message=f"{step} started",
                fields=start_fields,
            ),
        )
        result = runner(command)
        _emit_cli_log(
            args,
            LogEvent(
                event=complete_event,
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step=step,
                message=f"{step} completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "parent_artifacts": list(result.parent_artifacts),
                    "metadata": dict(result.metadata),
                },
            ),
        )
    except OptionalDependencyError as exc:
        error = ScoreError(
            f"execution evaluation optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation="install the required groups with `uv sync --group train --group data --group dev`",
            artifact=str(args.pack),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(args, run_id=run_id, step=step, event=error_event, exc=error)
        _emit_error(args, error, json_output=args.json)
        return 2
    except CheckpointTrustError as exc:
        error = ScoreError(
            f"execution evaluation checkpoint rejected: {exc}",
            error_type="checkpoint_error",
            remediation="provide a trusted checkpoint with a matching checkpoint manifest",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(args, run_id=run_id, step=step, event=error_event, exc=error)
        _emit_error(args, error, json_output=args.json)
        return 5
    except (ArtifactManifestError, json.JSONDecodeError, OSError) as exc:
        error = ScoreError(
            f"execution evaluation artifact validation failed: {exc}",
            error_type="manifest_error",
            remediation="verify the checkpoint, training-run, and execution-pack manifests, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(args, run_id=run_id, step=step, event=error_event, exc=error)
        _emit_error(args, error, json_output=args.json)
        return 2
    except (
        CrashPredictionError,
        ExecutionEvalError,
        LatentProbeError,
        RetrievalEvalError,
        SemanticDecoyPackError,
        SurpriseEvalError,
    ) as exc:
        error = ScoreError(
            f"execution evaluation failed: {exc}",
            error_type="evaluation_gate_error",
            remediation="inspect the execution pack, checkpoint, requested targets/decoys, and output directory",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(args, run_id=run_id, step=step, event=error_event, exc=error)
        _emit_error(args, error, json_output=args.json)
        return 6
    except Exception as exc:
        error = ScoreError(
            f"execution evaluation failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the execution eval inputs and retry with a corrected request",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(args, run_id=run_id, step=step, event=error_event, exc=error)
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"report: {args.out / result.report_path}")
        for key in ("query_count", "example_count", "row_count", "sample_count"):
            if key in result.metadata:
                print(f"{key}: {result.metadata[key]}")
    return 0


def _eval_latent_probe_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_latent_probe_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.latent_probe.start",
                level="info",
                run_id=run_id,
                step="eval.latent_probe",
                message="latent probe evaluation started",
                fields={
                    "checkpoint": str(args.checkpoint),
                    "data": str(args.data),
                    "out": str(args.out),
                    "device": args.device,
                    "max_examples_per_split": args.max_examples_per_split,
                    "bootstrap_samples": args.bootstrap_samples,
                    "seed": args.seed,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = run_latent_probe_evaluation(
            checkpoint=args.checkpoint,
            data=args.data,
            out=args.out,
            device=args.device,
            max_examples_per_split=args.max_examples_per_split,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.latent_probe.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.latent_probe",
                message="latent probe evaluation completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "row_count": result.row_count,
                    "split_counts": dict(result.split_counts),
                    "claim_boundary": dict(result.claim_boundary),
                    "parent_artifacts": list(result.parent_artifacts),
                },
            ),
        )
    except OptionalDependencyError as exc:
        error = ScoreError(
            f"latent probe optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation="install the required groups with `uv sync --group train --group data --group dev`",
            artifact=str(args.data),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_probe",
            event="evaluation.latent_probe.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except CheckpointTrustError as exc:
        error = ScoreError(
            f"latent probe checkpoint rejected: {exc}",
            error_type="checkpoint_error",
            remediation="provide a trusted checkpoint with a matching checkpoint manifest",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_probe",
            event="evaluation.latent_probe.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 5
    except (ArtifactManifestError, json.JSONDecodeError, OSError) as exc:
        error = ScoreError(
            f"latent probe artifact validation failed: {exc}",
            error_type="manifest_error",
            remediation="verify the checkpoint, training run, and dataset manifests, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_probe",
            event="evaluation.latent_probe.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except LatentProbeError as exc:
        error, exit_code = _latent_probe_eval_error(args, exc)
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_probe",
            event="evaluation.latent_probe.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return exit_code
    except Exception as exc:
        error = ScoreError(
            f"latent probe evaluation failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the latent probe inputs and retry with a corrected request",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_probe",
            event="evaluation.latent_probe.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"latent_probe_report: {args.out / result.report_path}")
        print(f"row_count: {result.row_count}")
        print(f"semantic_structure_status: {result.claim_boundary.get('semantic_structure_status')}")
    return 0


def _eval_latent_matrix_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_latent_matrix_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.latent_matrix.start",
                level="info",
                run_id=run_id,
                step="eval.latent_matrix",
                message="latent matrix evaluation started",
                fields={
                    "checkpoint": str(args.checkpoint),
                    "data": str(args.data),
                    "out": str(args.out),
                    "device": args.device,
                    "max_examples_per_split": args.max_examples_per_split,
                    "matrix_dimension_limit": args.matrix_dimension_limit,
                    "top_dimensions": args.top_dimensions,
                    "max_pairwise_rows": args.max_pairwise_rows,
                    "latent_probe_report": None
                    if args.latent_probe_report is None
                    else str(args.latent_probe_report),
                    "seed": args.seed,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = run_latent_matrix_evaluation(
            checkpoint=args.checkpoint,
            data=args.data,
            out=args.out,
            device=args.device,
            max_examples_per_split=args.max_examples_per_split,
            matrix_dimension_limit=args.matrix_dimension_limit,
            top_dimensions=args.top_dimensions,
            max_pairwise_rows=args.max_pairwise_rows,
            latent_probe_report=args.latent_probe_report,
            seed=args.seed,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.latent_matrix.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.latent_matrix",
                message="latent matrix evaluation completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "row_count": result.row_count,
                    "split_counts": dict(result.split_counts),
                    "view_shapes": dict(result.view_shapes),
                    "claim_boundary": dict(result.claim_boundary),
                    "parent_artifacts": list(result.parent_artifacts),
                },
            ),
        )
    except OptionalDependencyError as exc:
        error = ScoreError(
            f"latent matrix optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation="install the required groups with `uv sync --group train --group data --group dev`",
            artifact=str(args.data),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_matrix",
            event="evaluation.latent_matrix.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except CheckpointTrustError as exc:
        error = ScoreError(
            f"latent matrix checkpoint rejected: {exc}",
            error_type="checkpoint_error",
            remediation="provide a trusted checkpoint with a matching checkpoint manifest",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_matrix",
            event="evaluation.latent_matrix.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 5
    except (ArtifactManifestError, json.JSONDecodeError, OSError) as exc:
        error = ScoreError(
            f"latent matrix artifact validation failed: {exc}",
            error_type="manifest_error",
            remediation="verify the checkpoint, training run, dataset manifests, and optional latent-probe report, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_matrix",
            event="evaluation.latent_matrix.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except (LatentMatrixError, LatentProbeError) as exc:
        error, exit_code = _latent_matrix_eval_error(args, exc)
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_matrix",
            event="evaluation.latent_matrix.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return exit_code
    except Exception as exc:
        error = ScoreError(
            f"latent matrix evaluation failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the latent matrix inputs and retry with a corrected request",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.latent_matrix",
            event="evaluation.latent_matrix.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"latent_matrix_report: {args.out / result.report_path}")
        print(f"row_count: {result.row_count}")
        for view, shape in sorted(result.view_shapes.items()):
            print(f"{view}: rows={shape.get('rows')} dimensions={shape.get('dimensions')}")
        print(f"semantic_axis_claim_allowed: {result.claim_boundary.get('semantic_axis_claim_allowed')}")
    return 0


def _eval_surprise_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_surprise_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.surprise.start",
                level="info",
                run_id=run_id,
                step="eval.surprise",
                message="surprise evaluation started",
                fields={
                    "checkpoint": str(args.checkpoint),
                    "data": str(args.data),
                    "out": str(args.out),
                    "device": args.device,
                    "max_examples": args.max_examples,
                    "random_decoys": args.random_decoys,
                    "same_file_decoys": args.same_file_decoys,
                    "mutation_decoys": args.mutation_decoys,
                    "action_cluster_decoys": args.action_cluster_decoys,
                    "seed": args.seed,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = run_surprise_evaluation(
            checkpoint=args.checkpoint,
            data=args.data,
            out=args.out,
            device=args.device,
            max_examples=args.max_examples,
            random_decoys=args.random_decoys,
            same_file_decoys=args.same_file_decoys,
            mutation_decoys=args.mutation_decoys,
            action_cluster_decoys=args.action_cluster_decoys,
            seed=args.seed,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.surprise.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.surprise",
                message="surprise evaluation completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "metrics": result.metrics.to_dict(),
                    "parent_artifacts": list(result.parent_artifacts),
                },
            ),
        )
    except OptionalDependencyError as exc:
        error = ScoreError(
            f"surprise evaluation optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation="install the required groups with `uv sync --group train --group data --group dev`",
            artifact=str(args.data),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.surprise",
            event="evaluation.surprise.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except CheckpointTrustError as exc:
        error = ScoreError(
            f"surprise checkpoint rejected: {exc}",
            error_type="checkpoint_error",
            remediation="provide a trusted checkpoint with a matching checkpoint manifest",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.surprise",
            event="evaluation.surprise.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 5
    except (ArtifactManifestError, json.JSONDecodeError, OSError) as exc:
        error = ScoreError(
            f"surprise artifact validation failed: {exc}",
            error_type="manifest_error",
            remediation="verify the checkpoint, training run, and dataset manifests, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.surprise",
            event="evaluation.surprise.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except (SurpriseEvalError, RetrievalEvalError) as exc:
        error, exit_code = _surprise_eval_error(args, exc)
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.surprise",
            event="evaluation.surprise.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return exit_code
    except Exception as exc:
        error = ScoreError(
            f"surprise evaluation failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the surprise inputs and retry with a corrected request",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.surprise",
            event="evaluation.surprise.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"surprise_report: {args.out / result.report_path}")
        print(f"example_count: {result.metrics.example_count}")
        print(f"pairwise_auc_overall: {result.metrics.pairwise_auc_overall:.6g}")
        print(f"mean_true_rank: {result.metrics.mean_true_rank:.6g}")
        print(f"recall_at_1: {result.metrics.recall_at_1:.6g}")
    return 0


def _eval_ablation_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_ablation_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.ablation.start",
                level="info",
                run_id=run_id,
                step="eval.ablation",
                message="action-view ablation report started",
                fields={
                    "retrieval_artifact": str(args.retrieval_artifact),
                    "training_artifact": str(args.training_artifact),
                    "out": str(args.out),
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = run_action_ablation_suite(
            retrieval_artifact=args.retrieval_artifact,
            training_artifact=args.training_artifact,
            out=args.out,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.ablation.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.ablation",
                message="action-view ablation report completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "parent_artifacts": list(result.parent_artifacts),
                    "row_count": len(result.rows),
                },
            ),
        )
    except (
        ArtifactManifestError,
        json.JSONDecodeError,
        OSError,
        ActionAblationError,
    ) as exc:
        error = ScoreError(
            f"action-view ablation report failed: {exc}",
            error_type="manifest_error",
            remediation="verify retrieval/training artifacts and rerun with a clean output directory",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.ablation",
            event="evaluation.ablation.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except Exception as exc:
        error = ScoreError(
            f"action-view ablation report failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the ablation inputs and retry with corrected artifacts",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.ablation",
            event="evaluation.ablation.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"ablation_report: {args.out / result.report_path}")
        print(f"completed: {result.to_dict()['completed']}")
        print(f"blocked: {result.to_dict()['blocked']}")
    return 0


def _eval_scorer_quality_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_scorer_quality_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.scorer_quality.start",
                level="info",
                run_id=run_id,
                step="eval.scorer_quality",
                message="scorer/reranker quality evaluation started",
                fields={
                    "config": str(args.config),
                    "checkpoint": str(args.checkpoint),
                    "out": str(args.out),
                    "device": args.device,
                    "index": None if args.index is None else str(args.index),
                    "retrieval_prior_weight": args.retrieval_prior_weight,
                    "retrieval_prior_k": args.retrieval_prior_k,
                    "parent_manifests": [str(path) for path in args.parent_manifest],
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = run_scorer_quality_evaluation(
            config=args.config,
            checkpoint=args.checkpoint,
            out=args.out,
            device=args.device,
            index=args.index,
            retrieval_prior_weight=args.retrieval_prior_weight,
            retrieval_prior_k=args.retrieval_prior_k,
            parent_manifests=args.parent_manifest,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.scorer_quality.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.scorer_quality",
                message="scorer/reranker quality evaluation completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "parent_artifacts": list(result.parent_artifacts),
                },
            ),
        )
    except (
        ArtifactManifestError,
        ScorerQualityError,
        ScoreError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        error = ScoreError(
            f"scorer/reranker quality evaluation failed: {exc}",
            error_type="scoring_error",
            remediation="repair the scorer quality config or input artifacts and retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.scorer_quality",
            event="evaluation.scorer_quality.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except Exception as exc:
        error = ScoreError(
            f"scorer/reranker quality evaluation failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the quality inputs and retry with corrected artifacts",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.scorer_quality",
            event="evaluation.scorer_quality.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"quality_report: {args.out / result.report_path}")
    return 0


def _eval_downstream_pack_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_downstream_pack_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.downstream_pack.start",
                level="info",
                run_id=run_id,
                step="eval.downstream_pack",
                message="downstream benchmark pack build started",
                fields={
                    "config": str(args.config),
                    "out": str(args.out),
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = build_downstream_benchmark_pack(
            config_path=args.config,
            out=args.out,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.downstream_pack.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.downstream_pack",
                message="downstream benchmark pack build completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "benchmark_path": result.benchmark_path,
                    "readiness_report_path": result.readiness_report_path,
                    "example_count": result.example_count,
                    "labeled_example_count": result.labeled_example_count,
                    "scaled_evaluation_ready": result.scaled_evaluation_ready,
                    "downstream_claim_allowed": result.downstream_claim_allowed,
                },
            ),
        )
    except (
        ArtifactManifestError,
        DownstreamBenchmarkPackError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        error = ScoreError(
            f"downstream benchmark pack build failed: {exc}",
            error_type="manifest_error",
            remediation="repair the benchmark pack config or source files and retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.downstream_pack",
            event="evaluation.downstream_pack.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except Exception as exc:
        error = ScoreError(
            f"downstream benchmark pack build failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the benchmark pack inputs and retry with corrected artifacts",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.downstream_pack",
            event="evaluation.downstream_pack.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"benchmark: {args.out / result.benchmark_path}")
        print(f"readiness_report: {args.out / result.readiness_report_path}")
        print(f"example_count: {result.example_count}")
        print(f"labeled_example_count: {result.labeled_example_count}")
        print(f"scaled_evaluation_ready: {result.scaled_evaluation_ready}")
        print(f"downstream_claim_allowed: {result.downstream_claim_allowed}")
    return 0


def _eval_downstream_rerank_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_downstream_rerank_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.downstream_rerank.start",
                level="info",
                run_id=run_id,
                step="eval.downstream_rerank",
                message="downstream rerank evaluation started",
                fields={
                    "benchmark_manifest": str(args.benchmark_manifest),
                    "checkpoint": str(args.checkpoint),
                    "out": str(args.out),
                    "candidate_pack_manifests": [
                        str(path) for path in args.candidate_pack_manifest
                    ],
                    "device": args.device,
                    "index": None if args.index is None else str(args.index),
                    "retrieval_prior_weight": args.retrieval_prior_weight,
                    "retrieval_prior_k": args.retrieval_prior_k,
                    "pass_at_k": args.pass_at_k,
                    "bootstrap_samples": args.bootstrap_samples,
                    "seed": args.seed,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = run_downstream_rerank_evaluation(
            benchmark_manifest=args.benchmark_manifest,
            checkpoint=args.checkpoint,
            out=args.out,
            device=args.device,
            index=args.index,
            retrieval_prior_weight=args.retrieval_prior_weight,
            retrieval_prior_k=args.retrieval_prior_k,
            candidate_pack_manifests=args.candidate_pack_manifest,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            pass_at_k=args.pass_at_k,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.downstream_rerank.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.downstream_rerank",
                message="downstream rerank evaluation completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "parent_artifacts": list(result.parent_artifacts),
                    "example_count": result.example_count,
                    "claim_allowed": result.claim_allowed,
                },
            ),
        )
    except (
        ArtifactManifestError,
        DownstreamRerankEvalError,
        ScoreError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        error = ScoreError(
            f"downstream rerank evaluation failed: {exc}",
            error_type="scoring_error",
            remediation="verify the benchmark, candidate-pack, and checkpoint artifacts, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.downstream_rerank",
            event="evaluation.downstream_rerank.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except Exception as exc:
        error = ScoreError(
            f"downstream rerank evaluation failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the downstream rerank inputs and retry with corrected artifacts",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.downstream_rerank",
            event="evaluation.downstream_rerank.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"downstream_report: {args.out / result.report_path}")
        print(f"example_count: {result.example_count}")
        print(f"claim_allowed: {result.claim_allowed}")
    return 0


def _eval_p_pass_calibration_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_p_pass_calibration_command_tuple(args)
    baselines = args.baseline or P_PASS_DEFAULT_BASELINES
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.p_pass_calibration.start",
                level="info",
                run_id=run_id,
                step="eval.p_pass_calibration",
                message="p_pass calibration evaluation started",
                fields={
                    "scores": [str(path) for path in args.scores],
                    "parent_manifests": [
                        str(path) for path in args.parent_manifest
                    ],
                    "dataset_kind": args.dataset_kind,
                    "out": str(args.out),
                    "baselines": list(baselines),
                    "benchmark": args.benchmark,
                    "calibration_bin_count": args.calibration_bin_count,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = run_p_pass_calibration_evaluation(
            scores=args.scores,
            out=args.out,
            dataset_kind=args.dataset_kind,
            parent_manifests=args.parent_manifest,
            baselines=baselines,
            benchmark=args.benchmark,
            calibration_bin_count=args.calibration_bin_count,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.p_pass_calibration.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.p_pass_calibration",
                message="p_pass calibration evaluation completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "parent_artifacts": list(result.parent_artifacts),
                    "dataset_kind": result.dataset_kind,
                    "row_count": result.row_count,
                    "claim_allowed": result.claim_allowed,
                },
            ),
        )
    except (
        ArtifactManifestError,
        PPassCalibrationError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        error = ScoreError(
            f"p_pass calibration evaluation failed: {exc}",
            error_type="scoring_error",
            remediation="verify the score rows and parent manifests, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.p_pass_calibration",
            event="evaluation.p_pass_calibration.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except Exception as exc:
        error = ScoreError(
            f"p_pass calibration evaluation failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the p_pass calibration inputs and retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.p_pass_calibration",
            event="evaluation.p_pass_calibration.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"p_pass_calibration_report: {args.out / result.report_path}")
        print(f"dataset_kind: {result.dataset_kind}")
        print(f"row_count: {result.row_count}")
        print(f"claim_allowed: {result.claim_allowed}")
    return 0


def _eval_execution_completion_rerank_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _eval_execution_completion_rerank_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.execution_rerank.start",
                level="info",
                run_id=run_id,
                step="eval.execution_rerank",
                message="execution completion rerank evaluation started",
                fields={
                    "benchmark": args.benchmark,
                    "completion_manifest": str(args.completion_manifest),
                    "labels": None if args.labels is None else str(args.labels),
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_manifest": None
                    if args.checkpoint_manifest is None
                    else str(args.checkpoint_manifest),
                    "out": str(args.out),
                    "device": args.device,
                    "index": None if args.index is None else str(args.index),
                    "retrieval_prior_weight": args.retrieval_prior_weight,
                    "retrieval_prior_k": args.retrieval_prior_k,
                    "pass_at_k": args.pass_at_k,
                    "bootstrap_samples": args.bootstrap_samples,
                    "seed": args.seed,
                    "min_lift_for_claim": args.min_lift_for_claim,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = run_execution_rerank_evaluation(
            completion_manifest=args.completion_manifest,
            labels_path=args.labels,
            benchmark=args.benchmark,
            checkpoint=args.checkpoint,
            checkpoint_manifest=args.checkpoint_manifest,
            out=args.out,
            device=args.device,
            index=args.index,
            retrieval_prior_weight=args.retrieval_prior_weight,
            retrieval_prior_k=args.retrieval_prior_k,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            require_learned_scorer=args.require_learned_scorer,
            pass_at_k=args.pass_at_k,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            min_lift_for_claim=args.min_lift_for_claim,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="evaluation.execution_rerank.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="eval.execution_rerank",
                message="execution completion rerank evaluation completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "report_path": result.report_path,
                    "score_rows_path": result.score_rows_path,
                    "parent_artifacts": list(result.parent_artifacts),
                    "problem_count": result.problem_count,
                    "completion_count": result.completion_count,
                    "claim_allowed": result.claim_allowed,
                },
            ),
        )
    except (
        ArtifactManifestError,
        ExecutionRerankEvalError,
        ScoreError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        error = ScoreError(
            f"execution rerank evaluation failed: {exc}",
            error_type="scoring_error",
            remediation="verify the completion-label artifact and checkpoint, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.execution_rerank",
            event="evaluation.execution_rerank.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except Exception as exc:
        error = ScoreError(
            f"execution rerank evaluation failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the execution rerank inputs and retry with corrected artifacts",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args,
            run_id=run_id,
            step="eval.execution_rerank",
            event="evaluation.execution_rerank.error",
            exc=error,
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"execution_rerank_report: {args.out / result.report_path}")
        print(f"completion_scores: {args.out / result.score_rows_path}")
        print(f"problem_count: {result.problem_count}")
        print(f"completion_count: {result.completion_count}")
        print(f"claim_allowed: {result.claim_allowed}")
    return 0


def _index_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    command = _index_command_tuple(args)
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event="index.start",
                level="info",
                run_id=run_id,
                step="index",
                message="index command started",
                fields={
                    "checkpoint": str(args.checkpoint),
                    "data": str(args.data),
                    "out": str(args.out),
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "distance": args.distance,
                    "name": args.name,
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        result = build_transition_index_artifact(
            checkpoint=args.checkpoint,
            data=args.data,
            out=args.out,
            device=args.device,
            batch_size=args.batch_size,
            distance=args.distance,
            name=args.name,
            overwrite=args.overwrite,
            command=command,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="index.complete",
                level="info",
                run_id=run_id,
                artifact_id=result.artifact_manifest_id,
                step="index",
                message="index command completed",
                fields={
                    "artifact_manifest_path": result.artifact_manifest_path,
                    "index_path": result.index_path,
                    "count": result.count,
                    "dim": result.dim,
                    "parent_artifacts": list(result.parent_artifacts),
                },
            ),
        )
    except OptionalDependencyError as exc:
        error = ScoreError(
            f"index optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation="install the required groups with `uv sync --group train --group data --group dev`",
            artifact=str(args.data),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="index", event="index.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except CheckpointTrustError as exc:
        error = ScoreError(
            f"index checkpoint rejected: {exc}",
            error_type="checkpoint_error",
            remediation="provide a trusted checkpoint with a matching checkpoint manifest",
            artifact=str(args.checkpoint),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="index", event="index.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 5
    except (ArtifactManifestError, json.JSONDecodeError, OSError) as exc:
        error = ScoreError(
            f"index artifact validation failed: {exc}",
            error_type="manifest_error",
            remediation="verify the checkpoint, training run, and dataset manifests, then retry",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="index", event="index.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except (TransitionIndexError, RetrievalEvalError) as exc:
        error, exit_code = _index_error(args, exc)
        _emit_error_log(
            args, run_id=run_id, step="index", event="index.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return exit_code
    except Exception as exc:
        error = ScoreError(
            f"index build failed unexpectedly: {exc}",
            error_type="scoring_error",
            remediation="inspect the index inputs and retry with a corrected request",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error_log(
            args, run_id=run_id, step="index", event="index.error", exc=error
        )
        _emit_error(args, error, json_output=args.json)
        return 70

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {args.out / result.artifact_manifest_path}")
        print(f"index: {args.out / result.index_path}")
        print(f"vectors: {args.out / result.vectors_path}")
        print(f"entries: {args.out / result.entries_path}")
        print(f"count: {result.count}")
        print(f"dim: {result.dim}")
        print(f"distance: {result.distance}")
    return 0


def _dataset_pack_command(args: argparse.Namespace) -> int:
    command = (
        "codelewm",
        "dataset",
        "pack",
        "--manifest",
        str(args.manifest),
        "--out",
        str(args.out),
        *(("--json",) if args.json else ()),
    )
    try:
        result = pack_dataset_from_manifest(
            manifest_path=args.manifest,
            output_dir=args.out,
            command=command,
        )
    except OptionalDependencyError as exc:
        error = ScoreError(
            f"dataset pack optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation="install the data dependency group with `uv sync --group data --group dev`",
            artifact=str(args.manifest),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    except (ArtifactManifestError, PackError, OSError, json.JSONDecodeError) as exc:
        error = ScoreError(
            f"dataset pack failed: {exc}",
            error_type="dataset_build_error",
            remediation="verify the input manifest, checksums, and output path, then retry",
            artifact=str(args.manifest),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 4

    if args.json:
        print(json.dumps(result.to_report(), indent=2, sort_keys=True))
    else:
        print(f"artifact_manifest: {result.output_dir / 'manifest.json'}")
        print(f"dataset_manifest: {result.output_dir / 'dataset_manifest.json'}")
        print(f"row_count: {result.dataset_manifest.row_count}")
    return 0


def _dataset_execute_command(args: argparse.Namespace) -> int:
    """Run one (code, input) pair in the sandbox and emit a JSON result.

    The sandbox is loaded lazily so that ``codelewm --help`` and unrelated
    commands do not pay the import cost.
    """

    # Lazy import: the sandbox is a data-prep module and we keep its
    # surface out of the import graph of unrelated CLI commands.
    from codelewm.data.sandbox import (  # noqa: PLC0415
        SandboxPolicy,
        SandboxPolicyError,
        SandboxRunnerError,
        run_one,
    )

    code_path: Path = args.code_file
    if not code_path.is_file():
        error = ScoreError(
            f"code file does not exist: {code_path}",
            error_type="input_missing",
            remediation="pass --code-file pointing to a valid Python source file",
            artifact=str(code_path),
            caused_by="FileNotFoundError",
        )
        _emit_error(args, error, json_output=args.json)
        return 2

    code = code_path.read_text(encoding="utf-8")

    input_repr: str | None = None
    if args.input_file is not None:
        if not args.input_file.is_file():
            error = ScoreError(
                f"input file does not exist: {args.input_file}",
                error_type="input_missing",
                remediation="pass --input-file pointing to a JSON file",
                artifact=str(args.input_file),
                caused_by="FileNotFoundError",
            )
            _emit_error(args, error, json_output=args.json)
            return 2
        input_repr = args.input_file.read_text(encoding="utf-8")

    if args.function_name is not None and input_repr is None:
        error = ScoreError(
            "--function-name requires --input-file",
            error_type="invalid_arguments",
            remediation="pass --input-file when --function-name is used",
            artifact=str(code_path),
            caused_by="ValueError",
        )
        _emit_error(args, error, json_output=args.json)
        return 2

    stdin_text = ""
    if args.stdin_file is not None:
        if not args.stdin_file.is_file():
            error = ScoreError(
                f"stdin file does not exist: {args.stdin_file}",
                error_type="input_missing",
                remediation="pass --stdin-file pointing to a text file",
                artifact=str(args.stdin_file),
                caused_by="FileNotFoundError",
            )
            _emit_error(args, error, json_output=args.json)
            return 2
        stdin_text = args.stdin_file.read_text(encoding="utf-8")

    try:
        policy = SandboxPolicy(
            import_allowlist="stdlib_only",
            timeout_ms=args.timeout_ms,
            memory_mb=args.memory_mb,
            cpu_seconds=args.cpu_seconds,
            determinism_check=args.determinism_check,
        )
    except SandboxPolicyError as exc:
        error = ScoreError(
            f"invalid sandbox policy: {exc}",
            error_type="invalid_arguments",
            remediation="check timeout, memory, and cpu bounds against docs/operations/sandbox_policy.md",
            artifact=str(code_path),
            caused_by=f"SandboxPolicyError: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 2

    try:
        result = run_one(
            code,
            input_repr=input_repr,
            function_name=args.function_name,
            stdin_text=stdin_text,
            policy=policy,
            scratch_dir=args.scratch_dir,
        )
    except SandboxRunnerError as exc:
        error = ScoreError(
            f"sandbox runner error: {exc}",
            error_type="sandbox_runner_error",
            remediation="rerun with a fresh scratch dir or inspect the child stderr",
            artifact=str(code_path),
            caused_by=f"SandboxRunnerError: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 4

    payload = result.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"exit_code: {payload['exit_code']}")
        print(f"output_type: {payload['output_type']}")
        print(f"output_kind: {payload['output_kind']}")
        if payload["output_repr"] is not None:
            print(f"output_repr: {payload['output_repr']}")
        if payload["exception_class"]:
            print(f"exception: {payload['exception_class']}: {payload['exception_message']}")
        if payload["policy_violations"]:
            print("policy_violations:")
            for violation in payload["policy_violations"]:
                print(f"  - {violation}")
        print(f"wall_time_ms: {payload['wall_time_ms']:.2f}")
        print(f"determinism_check: {payload['determinism_check']}")
    return 0 if result.ok else 1


def _dataset_ingest_command(args: argparse.Namespace) -> int:
    """Normalize one upstream dataset's JSONL into SourceSubmission records."""

    from codelewm.data.execution_sources import (  # noqa: PLC0415
        ExecutionSourceError,
        load_execution_source,
    )

    if not args.input_path.exists():
        error = ScoreError(
            f"upstream source path does not exist: {args.input_path}",
            error_type="input_missing",
            remediation="pass --input pointing to the adapter's expected file or directory",
            artifact=str(args.input_path),
            caused_by="FileNotFoundError",
        )
        _emit_error(args, error, json_output=args.json)
        return 2
    if args.limit is not None and args.limit < 1:
        error = ScoreError(
            "--limit must be a positive integer",
            error_type="invalid_arguments",
            remediation="omit --limit or pass a value >= 1",
            artifact=str(args.input_path),
            caused_by="ValueError",
        )
        _emit_error(args, error, json_output=args.json)
        return 2

    try:
        result = load_execution_source(
            source=args.source,
            source_path=args.input_path,
            output_path=args.output,
            limit=args.limit,
        )
    except ExecutionSourceError as exc:
        error = ScoreError(
            f"ingestion failed: {exc}",
            error_type="dataset_build_error",
            remediation="verify the upstream JSONL format matches the adapter docstring",
            artifact=str(args.input_path),
            caused_by=f"ExecutionSourceError: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 4

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"source: {result['source']}")
        print(f"submission_count: {result['submission_count']}")
        print(f"unique_problem_count: {result['unique_problem_count']}")
        print(f"unique_submission_count: {result['unique_submission_count']}")
        print(f"held_out_for_eval: {result['held_out_for_eval']}")
        print(f"license: {result['license']}")
        print(f"output_path: {result['output_path']}")
    return 0


def _dataset_execution_pack_command(args: argparse.Namespace) -> int:
    """Build an execution-substrate pack from one or more ingestion JSONLs."""

    from codelewm.data.execution_pack import (  # noqa: PLC0415
        ExecutionPackBuilderError,
        build_execution_pack,
    )
    from codelewm.data.sandbox import (  # noqa: PLC0415
        SandboxPolicy,
        SandboxPolicyError,
    )

    paths: list[Path] = list(args.ingestion_paths or ())
    missing = [p for p in paths if not p.is_file()]
    if missing:
        error = ScoreError(
            f"ingestion file(s) missing: {missing}",
            error_type="input_missing",
            remediation="pass --ingestion <path> for each ingestion JSONL produced by `codelewm dataset ingest`",
            artifact=str(missing[0]),
            caused_by="FileNotFoundError",
        )
        _emit_error(args, error, json_output=args.json)
        return 2

    try:
        policy = SandboxPolicy(
            import_allowlist="stdlib_only",
            timeout_ms=args.timeout_ms,
            memory_mb=args.memory_mb,
            cpu_seconds=args.cpu_seconds,
            determinism_check=args.determinism_check,
        )
    except SandboxPolicyError as exc:
        error = ScoreError(
            f"invalid sandbox policy: {exc}",
            error_type="invalid_arguments",
            remediation="check the sandbox policy bounds against docs/operations/sandbox_policy.md",
            artifact=str(paths[0]) if paths else "",
            caused_by=f"SandboxPolicyError: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 2

    try:
        result = build_execution_pack(
            ingestion_paths=paths,
            output_dir=args.output,
            sandbox_policy=policy,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            max_inputs_per_problem=args.max_inputs_per_problem,
            target_records=args.target_records,
        )
    except ExecutionPackBuilderError as exc:
        error = ScoreError(
            f"execution-pack build failed: {exc}",
            error_type="dataset_build_error",
            remediation="verify --output is empty and ingestion files are valid",
            artifact=str(args.output),
            caused_by=f"ExecutionPackBuilderError: {exc}",
        )
        _emit_error(args, error, json_output=args.json)
        return 4

    if args.json:
        payload = {
            "pack_dir": str(result.output_dir),
            "record_count": result.record_count,
            "sandbox_reject_counts": dict(result.sandbox_reject_counts),
            "manifest": result.manifest.as_dict(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"pack_dir: {result.output_dir}")
        print(f"record_count: {result.record_count}")
        print("splits:")
        for split, count in sorted(result.manifest.split_counts.items()):
            print(f"  {split}: {count}")
        if result.sandbox_reject_counts:
            print("sandbox_reject_counts:")
            for reason, count in sorted(result.sandbox_reject_counts.items()):
                print(f"  {reason}: {count}")
    return 0


def _manifest_verify_command(args: argparse.Namespace) -> int:
    parent_manifest_paths: tuple[Path, ...] = tuple(args.parent_manifest or ())
    try:
        manifest = read_artifact_manifest(args.manifest)
        root = args.root or args.manifest.parent
        checked_files = validate_artifact_checksums(manifest, root=root)
        parent_ids: list[str] = []
        for parent_manifest_path in parent_manifest_paths:
            parent_manifest = read_artifact_manifest(parent_manifest_path)
            validate_artifact_checksums(
                parent_manifest, root=parent_manifest_path.parent
            )
            parent_ids.append(parent_manifest.artifact_id)
        accepted_parent_ids = set(parent_ids)
        accepted_parent_ids.update(
            f"execution_pack:{parent_id}" for parent_id in parent_ids
        )
        missing_parents = sorted(set(manifest.parent_artifacts) - accepted_parent_ids)
        if missing_parents:
            raise ScoreError(
                "manifest parent artifact(s) were not provided: "
                + ", ".join(missing_parents),
                error_type="manifest_error",
                remediation="pass each required parent with --parent-manifest",
                artifact=str(args.manifest),
            )
    except (ArtifactManifestError, OSError, json.JSONDecodeError) as exc:
        error = ScoreError(
            f"manifest verification failed: {exc}",
            error_type="manifest_error",
            remediation="repair the artifact or regenerate its manifest",
            artifact=str(args.manifest),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        )
        if args.json:
            print(
                json.dumps(error.to_error_report().to_dict(), indent=2, sort_keys=True)
            )
        else:
            print(str(error), file=sys.stderr)
        return 2
    except ScoreError as exc:
        if args.json:
            print(json.dumps(exc.to_error_report().to_dict(), indent=2, sort_keys=True))
        else:
            print(str(exc), file=sys.stderr)
        return 2

    report = {
        "schema_version": MANIFEST_VERIFY_REPORT_SCHEMA_VERSION,
        "ok": True,
        "manifest": str(args.manifest),
        "artifact_id": manifest.artifact_id,
        "artifact_kind": manifest.artifact_kind,
        "files_checked": len(checked_files),
        "parent_artifacts": list(manifest.parent_artifacts),
        "parents_checked": parent_ids,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"manifest: {manifest.artifact_id}")
        print(f"files_checked: {len(checked_files)}")
        print("ok: true")
    return 0


def _emit_error(
    args: argparse.Namespace, exc: ScoreError, *, json_output: bool
) -> None:
    if json_output:
        print(json.dumps(exc.to_error_report().to_dict(), indent=2, sort_keys=True))
    else:
        print(str(exc), file=sys.stderr)


def _load_cli_train_config(args: argparse.Namespace):
    config = load_train_config(args.config)
    if args.out is None:
        return config
    output_dir = args.out
    payload = config.to_dict()
    payload["output"] = {
        "run_dir": str(output_dir),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "metrics_path": str(output_dir / "metrics.jsonl"),
        "manifest_path": str(output_dir / "training_manifest.json"),
    }
    return validate_train_config(payload)


def _validate_train_cli_executor(args: argparse.Namespace) -> None:
    if (
        args.executor == "cpu-smoke"
        and args.device is not None
        and args.device not in {"cpu", "auto"}
    ):
        raise TrainConfigError(
            "cpu-smoke executor only supports --device cpu or --device auto"
        )
    if args.executor == "cpu-smoke" and (args.tensorboard or args.tensorboard_dir is not None):
        raise TrainConfigError("TensorBoard export currently requires --executor torch")


def _train_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "train",
        "--config",
        str(args.config),
        "--executor",
        str(args.executor),
    ]
    if args.device is not None:
        command.extend(("--device", str(args.device)))
    if args.out is not None:
        command.extend(("--out", str(args.out)))
    if args.resume_from is not None:
        command.extend(("--resume-from", str(args.resume_from)))
    if args.tensorboard:
        command.append("--tensorboard")
    if args.tensorboard_dir is not None:
        command.extend(("--tensorboard-dir", str(args.tensorboard_dir)))
    if args.overwrite:
        command.append("--overwrite")
    if getattr(args, "seed", None) is not None:
        command.extend(("--seed", str(args.seed)))
    if getattr(args, "pack_local_dir", None) is not None:
        command.extend(("--pack-local-dir", str(args.pack_local_dir)))
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _model_inspect_checkpoint_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "model",
        "inspect-checkpoint",
        "--checkpoint",
        str(args.checkpoint),
        "--out",
        str(args.out),
    ]
    if args.checkpoint_manifest is not None:
        command.extend(("--checkpoint-manifest", str(args.checkpoint_manifest)))
    for parent_manifest in args.parent_manifest:
        command.extend(("--parent-manifest", str(parent_manifest)))
    command.extend(("--histogram-bins", str(args.histogram_bins)))
    command.extend(("--max-histogram-tensors", str(args.max_histogram_tensors)))
    command.extend(("--max-histogram-values", str(args.max_histogram_values)))
    if args.allow_unsafe_checkpoint:
        command.append("--allow-unsafe-checkpoint")
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_retrieval_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "retrieval",
        "--checkpoint",
        str(args.checkpoint),
        "--data",
        str(args.data),
        "--out",
        str(args.out),
        "--device",
        str(args.device),
        "--max-candidates",
        str(args.max_candidates),
        "--hard-negatives",
        str(args.hard_negatives),
        "--seed",
        str(args.seed),
        "--report-scope",
        str(args.report_scope),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_execution_retrieval_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "execution-retrieval",
        "--checkpoint",
        str(args.checkpoint),
        "--pack",
        str(args.pack),
        "--baselines",
        str(args.baselines),
        "--out",
        str(args.out),
        "--device",
        str(args.device),
        "--max-candidates",
        str(args.max_candidates),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_execution_surprise_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "execution-surprise",
        "--checkpoint",
        str(args.checkpoint),
        "--pack",
        str(args.pack),
        "--decoys",
        str(args.decoys),
        "--out",
        str(args.out),
        "--device",
        str(args.device),
        "--max-examples",
        str(args.max_examples),
        "--seed",
        str(args.seed),
    ]
    if args.semantic_decoy_manifest is not None:
        command.extend(("--semantic-decoy-manifest", str(args.semantic_decoy_manifest)))
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_semantic_decoy_pack_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "semantic-decoy-pack",
        "--pack",
        str(args.pack),
        "--out",
        str(args.out),
        "--splits",
        str(args.splits),
        "--seed",
        str(args.seed),
        "--max-pairs-per-query",
        str(args.max_pairs_per_query),
        "--min-pairs-for-claim",
        str(args.min_pairs_for_claim),
        "--min-distinct-problems-for-claim",
        str(args.min_distinct_problems_for_claim),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_execution_probe_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "execution-probe",
        "--checkpoint",
        str(args.checkpoint),
        "--pack",
        str(args.pack),
        "--targets",
        str(args.targets),
        "--out",
        str(args.out),
        "--device",
        str(args.device),
        "--max-examples-per-split",
        str(args.max_examples_per_split),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_crash_prediction_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "crash-prediction",
        "--checkpoint",
        str(args.checkpoint),
        "--pack",
        str(args.pack),
        "--out",
        str(args.out),
        "--device",
        str(args.device),
        "--max-examples",
        str(args.max_examples),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_latent_probe_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "latent-probe",
        "--checkpoint",
        str(args.checkpoint),
        "--data",
        str(args.data),
        "--out",
        str(args.out),
        "--device",
        str(args.device),
        "--max-examples-per-split",
        str(args.max_examples_per_split),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_latent_matrix_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "latent-matrix",
        "--checkpoint",
        str(args.checkpoint),
        "--data",
        str(args.data),
        "--out",
        str(args.out),
        "--device",
        str(args.device),
        "--max-examples-per-split",
        str(args.max_examples_per_split),
        "--matrix-dimension-limit",
        str(args.matrix_dimension_limit),
        "--top-dimensions",
        str(args.top_dimensions),
        "--max-pairwise-rows",
        str(args.max_pairwise_rows),
        "--seed",
        str(args.seed),
    ]
    if args.latent_probe_report is not None:
        command.extend(("--latent-probe-report", str(args.latent_probe_report)))
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_surprise_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "surprise",
        "--checkpoint",
        str(args.checkpoint),
        "--data",
        str(args.data),
        "--out",
        str(args.out),
        "--device",
        str(args.device),
        "--max-examples",
        str(args.max_examples),
        "--random-decoys",
        str(args.random_decoys),
        "--same-file-decoys",
        str(args.same_file_decoys),
        "--mutation-decoys",
        str(args.mutation_decoys),
        "--action-cluster-decoys",
        str(args.action_cluster_decoys),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_ablation_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "ablation",
        "--retrieval-artifact",
        str(args.retrieval_artifact),
        "--training-artifact",
        str(args.training_artifact),
        "--out",
        str(args.out),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_scorer_quality_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "scorer-quality",
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--out",
        str(args.out),
        "--device",
        args.device,
    ]
    if args.index is not None:
        command.extend(("--index", str(args.index)))
    command.extend(("--retrieval-prior-weight", str(args.retrieval_prior_weight)))
    command.extend(("--retrieval-prior-k", str(args.retrieval_prior_k)))
    for parent_manifest in args.parent_manifest:
        command.extend(("--parent-manifest", str(parent_manifest)))
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    if args.allow_unsafe_checkpoint:
        command.append("--allow-unsafe-checkpoint")
    return tuple(command)


def _eval_downstream_pack_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "downstream-pack",
        "--config",
        str(args.config),
        "--out",
        str(args.out),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_downstream_rerank_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "downstream-rerank",
        "--benchmark-manifest",
        str(args.benchmark_manifest),
        "--checkpoint",
        str(args.checkpoint),
        "--out",
        str(args.out),
        "--device",
        args.device,
    ]
    for manifest in args.candidate_pack_manifest:
        command.extend(("--candidate-pack-manifest", str(manifest)))
    if args.index is not None:
        command.extend(("--index", str(args.index)))
    command.extend(("--retrieval-prior-weight", str(args.retrieval_prior_weight)))
    command.extend(("--retrieval-prior-k", str(args.retrieval_prior_k)))
    command.extend(("--pass-at-k", str(args.pass_at_k)))
    command.extend(("--bootstrap-samples", str(args.bootstrap_samples)))
    command.extend(("--seed", str(args.seed)))
    if args.allow_unsafe_checkpoint:
        command.append("--allow-unsafe-checkpoint")
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_p_pass_calibration_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "p-pass-calibration",
        "--dataset-kind",
        args.dataset_kind,
        "--out",
        str(args.out),
    ]
    for score_path in args.scores:
        command.extend(("--scores", str(score_path)))
    for manifest in args.parent_manifest:
        command.extend(("--parent-manifest", str(manifest)))
    for baseline in args.baseline:
        command.extend(("--baseline", str(baseline)))
    if args.benchmark is not None:
        command.extend(("--benchmark", str(args.benchmark)))
    command.extend(("--calibration-bin-count", str(args.calibration_bin_count)))
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _eval_execution_completion_rerank_command_tuple(
    args: argparse.Namespace,
) -> tuple[str, ...]:
    command = [
        "codelewm",
        "eval",
        "rerank-humaneval" if args.benchmark == "humaneval" else "rerank-mbpp-plus",
        "--completion-manifest",
        str(args.completion_manifest),
        "--checkpoint",
        str(args.checkpoint),
        "--out",
        str(args.out),
        "--device",
        args.device,
        "--pass-at-k",
        str(args.pass_at_k),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.seed),
        "--min-lift-for-claim",
        str(args.min_lift_for_claim),
    ]
    if args.labels is not None:
        command.extend(("--labels", str(args.labels)))
    if args.checkpoint_manifest is not None:
        command.extend(("--checkpoint-manifest", str(args.checkpoint_manifest)))
    if args.index is not None:
        command.extend(("--index", str(args.index)))
    command.extend(("--retrieval-prior-weight", str(args.retrieval_prior_weight)))
    command.extend(("--retrieval-prior-k", str(args.retrieval_prior_k)))
    if args.allow_unsafe_checkpoint:
        command.append("--allow-unsafe-checkpoint")
    if args.require_learned_scorer:
        command.append("--require-learned-scorer")
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _index_command_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "codelewm",
        "index",
        "--checkpoint",
        str(args.checkpoint),
        "--data",
        str(args.data),
        "--out",
        str(args.out),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--distance",
        str(args.distance),
        "--name",
        str(args.name),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.json:
        command.append("--json")
    if args.log_jsonl is not None:
        command.extend(("--log-jsonl", str(args.log_jsonl)))
    return tuple(command)


def _retrieval_eval_error(
    args: argparse.Namespace, exc: Exception
) -> tuple[ScoreError, int]:
    message = str(exc)
    normalized = message.lower()
    if (
        "output already exists" in normalized
        or "must be a positive integer" in normalized
        or "--data" in normalized
        or "device" in normalized
        or "report_scope" in normalized
    ):
        return (
            ScoreError(
                f"retrieval evaluation request is invalid: {exc}",
                error_type="config_error",
                remediation="repair the command flags or choose a clean output directory",
                artifact=str(args.out),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ),
            2,
        )
    return (
        ScoreError(
            f"retrieval evaluation gate failed: {exc}",
            error_type="evaluation_gate_error",
            remediation="inspect the retrieval report inputs, baselines, and action-view policy",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        ),
        6,
    )


def _latent_probe_eval_error(
    args: argparse.Namespace, exc: Exception
) -> tuple[ScoreError, int]:
    message = str(exc)
    normalized = message.lower()
    if (
        "output already exists" in normalized
        or "must be a positive integer" in normalized
        or "must be a non-negative integer" in normalized
        or "--data" in normalized
        or "device" in normalized
    ):
        return (
            ScoreError(
                f"latent probe request is invalid: {exc}",
                error_type="config_error",
                remediation="repair the command flags or choose a clean output directory",
                artifact=str(args.out),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ),
            2,
        )
    return (
        ScoreError(
            f"latent probe gate failed: {exc}",
            error_type="evaluation_gate_error",
            remediation="inspect probe labels, split coverage, and checkpoint artifacts",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        ),
        6,
    )


def _latent_matrix_eval_error(
    args: argparse.Namespace, exc: Exception
) -> tuple[ScoreError, int]:
    message = str(exc)
    normalized = message.lower()
    if (
        "output already exists" in normalized
        or "must be a positive integer" in normalized
        or "must be a non-negative integer" in normalized
        or "--data" in normalized
        or "device" in normalized
        or "row count mismatch" in normalized
        or "rank 2" in normalized
    ):
        return (
            ScoreError(
                f"latent matrix request is invalid: {exc}",
                error_type="config_error",
                remediation="repair the command flags or choose a clean output directory",
                artifact=str(args.out),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ),
            2,
        )
    return (
        ScoreError(
            f"latent matrix gate failed: {exc}",
            error_type="evaluation_gate_error",
            remediation="inspect latent matrices, probe links, split coverage, and checkpoint artifacts",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        ),
        6,
    )


def _index_error(args: argparse.Namespace, exc: Exception) -> tuple[ScoreError, int]:
    message = str(exc)
    normalized = message.lower()
    if (
        "output already exists" in normalized
        or "--data" in normalized
        or "batch_size" in normalized
        or "device" in normalized
        or "distance" in normalized
        or "name must not be empty" in normalized
    ):
        return (
            ScoreError(
                f"index request is invalid: {exc}",
                error_type="config_error",
                remediation="repair the command flags or choose a clean output directory",
                artifact=str(args.out),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ),
            2,
        )
    return (
        ScoreError(
            f"index build gate failed: {exc}",
            error_type="evaluation_gate_error",
            remediation="inspect the index inputs, training split, and scores",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        ),
        6,
    )


def _surprise_eval_error(
    args: argparse.Namespace, exc: Exception
) -> tuple[ScoreError, int]:
    message = str(exc)
    normalized = message.lower()
    if (
        "output already exists" in normalized
        or "must be a positive integer" in normalized
        or "must be a non-negative integer" in normalized
        or "at least one decoy" in normalized
        or "--data" in normalized
        or "device" in normalized
    ):
        return (
            ScoreError(
                f"surprise evaluation request is invalid: {exc}",
                error_type="config_error",
                remediation="repair the command flags or choose a clean output directory",
                artifact=str(args.out),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ),
            2,
        )
    return (
        ScoreError(
            f"surprise evaluation gate failed: {exc}",
            error_type="evaluation_gate_error",
            remediation="inspect the surprise report inputs, decoys, and scores",
            artifact=str(args.out),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        ),
        6,
    )


def _training_run_error(
    args: argparse.Namespace, exc: TrainingRunError
) -> tuple[ScoreError, int]:
    message = str(exc)
    normalized = message.lower()
    if (
        "checkpoint" in normalized
        or "resume" in normalized
        or "parent training" in normalized
    ):
        return (
            ScoreError(
                f"training checkpoint compatibility failed: {exc}",
                error_type="checkpoint_error",
                remediation="choose a compatible resume checkpoint or start a fresh run",
                artifact=None if args.resume_from is None else str(args.resume_from),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ),
            5,
        )
    if (
        "output" in normalized
        or "device" in normalized
        or "config" in normalized
        or "patch action" in normalized
    ):
        return (
            ScoreError(
                f"training request is invalid: {exc}",
                error_type="config_error",
                remediation="repair the command flags or training config and retry",
                artifact=str(args.config),
                caused_by=f"{exc.__class__.__name__}: {exc}",
            ),
            2,
        )
    return (
        ScoreError(
            f"training failed: {exc}",
            error_type="scoring_error",
            remediation="inspect the training reports and retry with a corrected config",
            artifact=str(args.config),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        ),
        70,
    )


def _rerank_command(args: argparse.Namespace) -> int:
    run_id = _run_id()
    try:
        instruction = _instruction_arg_to_text(args.instruction)
        _emit_cli_log(
            args,
            LogEvent(
                event="harness.rerank.start",
                level="info",
                run_id=run_id,
                step="rerank",
                message="rerank command started",
                fields={
                    "before": str(args.before),
                    "candidates": str(args.candidates),
                    "checkpoint": str(args.checkpoint),
                    "device": args.device,
                    "index": None if args.index is None else str(args.index),
                    "retrieval_prior_weight": args.retrieval_prior_weight,
                    "retrieval_prior_k": args.retrieval_prior_k,
                    "instruction_sha256": _sha256_text(instruction),
                },
            ),
        )
        scorer = load_scorer(
            args.checkpoint,
            device=args.device,
            allow_unsafe=args.allow_unsafe_checkpoint,
            index=args.index,
            retrieval_prior_weight=args.retrieval_prior_weight,
            retrieval_prior_k=args.retrieval_prior_k,
        )
        result = scorer.rerank_files(
            before=args.before,
            instruction=instruction,
            candidates=args.candidates,
        )
        _emit_cli_log(
            args,
            LogEvent(
                event="harness.rerank.complete",
                level="info",
                run_id=run_id,
                step="rerank",
                message="rerank command completed",
                fields={
                    "result_count": len(result.results),
                    "valid_count": sum(
                        1 for item in result.results if hasattr(item, "final_score")
                    ),
                    "error_count": sum(
                        1 for item in result.results if hasattr(item, "error_type")
                    ),
                    "warnings": list(result.warnings),
                },
            ),
        )
    except ScoreError as exc:
        _emit_error_log(
            args, run_id=run_id, step="rerank", event="harness.rerank.error", exc=exc
        )
        if args.json:
            print(json.dumps(exc.to_error_report().to_dict(), indent=2, sort_keys=True))
        else:
            print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        for rank, item in enumerate(result.results, start=1):
            if hasattr(item, "final_score"):
                print(f"{rank}. {item.candidate} final_score={item.final_score:.6g}")
            else:
                print(
                    f"{rank}. {item.artifact or item.record_id} error={item.error_type}: {item.message}"
                )
    return 0


def _secret_scan_command(args: argparse.Namespace) -> int:
    include_suffixes = (
        None if args.include_suffix is None else tuple(args.include_suffix)
    )
    report = scan_paths(
        args.paths,
        include_suffixes=include_suffixes,
        recursive=args.recursive,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"paths_scanned: {len(report.paths_scanned)}")
        print(f"findings: {len(report.findings)}")
        for finding in report.findings:
            print(
                f"  {finding.path}:{finding.line} {finding.pattern} {finding.redacted}"
            )
        print(f"ok: {'true' if report.ok else 'false'}")
    return 0 if report.ok else 2


def _instruction_arg_to_text(value: str) -> str:
    path = Path(value)
    if path.exists():
        if not path.is_file():
            raise ScoreError(f"instruction path is not a file: {path}")
        return path.read_text()
    return value


def _emit_cli_log(args: argparse.Namespace, event: LogEvent) -> None:
    log_path = getattr(args, "log_jsonl", None)
    if log_path is None:
        return
    try:
        write_log_event_jsonl(event, log_path)
    except OSError as exc:
        raise ScoreError(
            f"failed to write structured log: {log_path}",
            remediation="provide a writable --log-jsonl path",
            artifact=str(log_path),
            caused_by=f"{exc.__class__.__name__}: {exc}",
        ) from exc


def _emit_error_log(
    args: argparse.Namespace,
    *,
    run_id: str,
    step: str,
    event: str,
    exc: ScoreError,
) -> None:
    try:
        _emit_cli_log(
            args,
            LogEvent(
                event=event,
                level="error",
                run_id=run_id,
                step=step,
                message=str(exc),
                fields={"error": exc.to_error_report().to_dict()},
            ),
        )
    except ScoreError:
        pass


def _run_id() -> str:
    return uuid.uuid4().hex


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.parser = parser
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
