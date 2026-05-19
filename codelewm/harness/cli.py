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
    SourceRecordError,
    SourceUnavailableError,
    build_dataset_from_config_path,
)
from codelewm.harness.scorer import ScoreError, load_scorer
from codelewm.observability import (
    ArtifactManifestError,
    LogEvent,
    read_artifact_manifest,
    validate_artifact_checksums,
    write_log_event_jsonl,
)
from codelewm.security import scan_paths


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
    score.add_argument("--before", type=Path, required=True, help="before-state Python file")
    score.add_argument("--instruction", required=True, help="instruction text or path to a text file")
    score.add_argument("--candidate", type=Path, required=True, help="candidate after-state Python file")
    score.add_argument("--checkpoint", type=Path, required=True, help="checkpoint file")
    score.add_argument("--device", default="auto", choices=("cpu", "cuda", "mps", "auto"))
    score.add_argument("--json", action="store_true", help="emit JSON output")
    score.add_argument("--log-jsonl", type=Path, help="append structured JSONL logs to this local file")
    score.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="load the checkpoint without verifying its manifest (trusted local use only)",
    )
    score.set_defaults(func=_score_command)
    rerank = subparsers.add_parser("rerank", help="rerank candidate after-states or patches")
    rerank.add_argument("--before", type=Path, required=True, help="before-state Python file")
    rerank.add_argument("--instruction", required=True, help="instruction text or path to a text file")
    rerank.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="candidate after-state file, patch file, or directory",
    )
    rerank.add_argument("--checkpoint", type=Path, required=True, help="checkpoint file")
    rerank.add_argument("--device", default="auto", choices=("cpu", "cuda", "mps", "auto"))
    rerank.add_argument("--json", action="store_true", help="emit JSON output")
    rerank.add_argument("--log-jsonl", type=Path, help="append structured JSONL logs to this local file")
    rerank.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="load the checkpoint without verifying its manifest (trusted local use only)",
    )
    rerank.set_defaults(func=_rerank_command)
    dataset = subparsers.add_parser("dataset", help="dataset build and packing utilities")
    dataset_subcommands = dataset.add_subparsers(dest="dataset_command")
    build = dataset_subcommands.add_parser("build", help="build a transition dataset artifact")
    build.add_argument("--config", type=Path, required=True, help="dataset build config JSON or YAML path")
    build.add_argument("--out", type=Path, required=True, help="empty output artifact directory")
    build.add_argument("--json", action="store_true", help="emit JSON output")
    build.set_defaults(func=_dataset_build_command)
    dataset.set_defaults(func=_dataset_help_command, dataset_parser=dataset)
    manifest = subparsers.add_parser("manifest", help="artifact manifest utilities")
    manifest_subcommands = manifest.add_subparsers(dest="manifest_command")
    verify = manifest_subcommands.add_parser("verify", help="verify an artifact manifest")
    verify.add_argument("--manifest", type=Path, required=True, help="artifact manifest JSON path")
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
                    "instruction_sha256": _sha256_text(instruction),
                },
            ),
        )
        scorer = load_scorer(args.checkpoint, device=args.device, allow_unsafe=args.allow_unsafe_checkpoint)
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
        _emit_error_log(args, run_id=run_id, step="score", event="harness.score.error", exc=exc)
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
        error_type = "empty_dataset" if "zero kept transitions" in str(exc) else "dataset_build_error"
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


def _manifest_verify_command(args: argparse.Namespace) -> int:
    parent_manifest_paths: tuple[Path, ...] = tuple(args.parent_manifest or ())
    try:
        manifest = read_artifact_manifest(args.manifest)
        root = args.root or args.manifest.parent
        checked_files = validate_artifact_checksums(manifest, root=root)
        parent_ids: list[str] = []
        for parent_manifest_path in parent_manifest_paths:
            parent_manifest = read_artifact_manifest(parent_manifest_path)
            validate_artifact_checksums(parent_manifest, root=parent_manifest_path.parent)
            parent_ids.append(parent_manifest.artifact_id)
        missing_parents = sorted(set(manifest.parent_artifacts) - set(parent_ids))
        if missing_parents:
            raise ScoreError(
                "manifest parent artifact(s) were not provided: " + ", ".join(missing_parents),
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
            print(json.dumps(error.to_error_report().to_dict(), indent=2, sort_keys=True))
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


def _emit_error(args: argparse.Namespace, exc: ScoreError, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(exc.to_error_report().to_dict(), indent=2, sort_keys=True))
    else:
        print(str(exc), file=sys.stderr)


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
                    "instruction_sha256": _sha256_text(instruction),
                },
            ),
        )
        scorer = load_scorer(args.checkpoint, device=args.device, allow_unsafe=args.allow_unsafe_checkpoint)
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
                    "valid_count": sum(1 for item in result.results if hasattr(item, "final_score")),
                    "error_count": sum(1 for item in result.results if hasattr(item, "error_type")),
                    "warnings": list(result.warnings),
                },
            ),
        )
    except ScoreError as exc:
        _emit_error_log(args, run_id=run_id, step="rerank", event="harness.rerank.error", exc=exc)
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
                print(f"{rank}. {item.artifact or item.record_id} error={item.error_type}: {item.message}")
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
            print(f"  {finding.path}:{finding.line} {finding.pattern} {finding.redacted}")
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
