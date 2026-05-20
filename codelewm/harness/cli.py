"""Command-line entry point for CodeLeWM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

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
    RetrievalEvalError,
    SurpriseEvalError,
    run_action_ablation_suite,
    run_retrieval_evaluation,
    run_surprise_evaluation,
)
from codelewm.harness.index_runner import build_transition_index_artifact
from codelewm.harness.quality import (
    ScorerQualityError,
    run_scorer_quality_evaluation,
)
from codelewm.harness.scorer import ScoreError, load_scorer
from codelewm.harness.transition_index import TransitionIndexError
from codelewm.observability import (
    ArtifactManifestError,
    LogEvent,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_log_event_jsonl,
)
from codelewm.security import CheckpointTrustError, scan_paths
from codelewm.training import (
    TrainConfigError,
    TrainingRunError,
    cpu_smoke_training_executor,
    load_train_config,
    make_torch_training_executor,
    train,
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
        "--overwrite",
        action="store_true",
        help="overwrite an existing CodeLeWM run output",
    )
    train_parser.add_argument("--json", action="store_true", help="emit JSON output")
    train_parser.add_argument(
        "--log-jsonl", type=Path, help="append structured JSONL logs to this local file"
    )
    train_parser.set_defaults(func=_train_command)
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
                    "overwrite": bool(args.overwrite),
                },
            ),
        )
        executor = (
            make_torch_training_executor(device=args.device)
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
        error = ScoreError(
            f"training optional dependency is missing: {exc}",
            error_type="optional_dependency_missing",
            remediation="install the required groups with `uv sync --group train --group data --group dev`",
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
        missing_parents = sorted(set(manifest.parent_artifacts) - set(parent_ids))
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
