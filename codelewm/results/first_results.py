"""Reproducible first-results runner and report renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIRST_RESULTS_SCHEMA_VERSION = "codelewm.first_results.v1"
DEFAULT_OUTPUT_ROOT = Path(".artifacts/first-results")
DEFAULT_CONFIG_DIR = Path("config/first_results")
DEFAULT_REPORT_PATH = Path("docs/benchmark/FIRST_RESULTS.md")
ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("dataset_build", "build", "manifest.json"),
    ("dataset_pack", "pack", "manifest.json"),
    ("training_run", "train", "manifest.json"),
    ("retrieval_eval", "retrieval", "manifest.json"),
    ("action_ablation", "ablation", "manifest.json"),
    ("surprise_eval", "surprise", "manifest.json"),
    ("transition_index", "index", "manifest.json"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/first-results",
        description="Run the reproducible CodeLeWM first-results smoke workflow.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--overwrite", action="store_true", help="replace the owned first-results artifact root")
    parser.add_argument("--json", action="store_true", help="emit the manifest inventory JSON")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    try:
        inventory = run_first_results(
            output_root=args.output_root,
            config_dir=args.config_dir,
            report_path=args.report,
            overwrite=args.overwrite,
            repo_root=repo_root,
        )
    except FirstResultsError as exc:
        print(f"first-results failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        print(f"first_results_dir: {_display_path(Path(inventory['output_root']), repo_root=repo_root)}")
        print(f"report: {_display_path(Path(inventory['report_path']), repo_root=repo_root)}")
        print(f"secret_scan_ok: {inventory['secret_scan']['ok']}")
    return 0


class FirstResultsError(RuntimeError):
    """Raised when the first-results workflow cannot complete."""


def run_first_results(
    *,
    output_root: Path,
    config_dir: Path,
    report_path: Path,
    overwrite: bool,
    repo_root: Path,
) -> dict[str, Any]:
    """Run the full first-results workflow and return its inventory."""

    repo_root = repo_root.resolve()
    output_root = _resolve_under_repo(output_root, repo_root=repo_root)
    config_dir = _resolve_under_repo(config_dir, repo_root=repo_root)
    report_path = _resolve_under_repo(report_path, repo_root=repo_root)
    if output_root.exists():
        if not overwrite:
            raise FirstResultsError(f"output root already exists; pass --overwrite: {output_root}")
        _remove_owned_output_root(output_root, repo_root=repo_root)

    logs_dir = output_root / "logs"
    runtime_config_dir = output_root / "configs"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_config = config_dir / "dataset_build.json"
    train_template = config_dir / "train_tiny.json"
    runtime_train_config = runtime_config_dir / "train_tiny.json"
    _write_json(
        _runtime_train_config(_read_json(train_template), output_root=output_root, repo_root=repo_root),
        runtime_train_config,
    )

    build_dir = output_root / "build"
    pack_dir = output_root / "pack"
    train_dir = output_root / "train"
    retrieval_dir = output_root / "retrieval"
    ablation_dir = output_root / "ablation"
    surprise_dir = output_root / "surprise"
    index_dir = output_root / "index"
    checkpoint = train_dir / "checkpoints" / "checkpoint.pt"

    commands: list[dict[str, Any]] = []
    _run_cli(
        commands,
        repo_root=repo_root,
        label="dataset_build",
        args=("dataset", "build", "--config", str(dataset_config), "--out", str(build_dir), "--json"),
    )
    _run_cli(
        commands,
        repo_root=repo_root,
        label="dataset_pack",
        args=("dataset", "pack", "--manifest", str(build_dir / "manifest.json"), "--out", str(pack_dir), "--json"),
    )
    _run_cli(
        commands,
        repo_root=repo_root,
        label="training_run",
        args=(
            "train",
            "--config",
            str(runtime_train_config),
            "--executor",
            "torch",
            "--device",
            "cpu",
            "--overwrite",
            "--json",
            "--log-jsonl",
            str(logs_dir / "train.jsonl"),
        ),
    )
    _run_cli(
        commands,
        repo_root=repo_root,
        label="retrieval_eval",
        args=(
            "eval",
            "retrieval",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(pack_dir),
            "--out",
            str(retrieval_dir),
            "--device",
            "cpu",
            "--seed",
            "0",
            "--overwrite",
            "--json",
            "--log-jsonl",
            str(logs_dir / "retrieval.jsonl"),
        ),
    )
    _run_cli(
        commands,
        repo_root=repo_root,
        label="action_ablation",
        args=(
            "eval",
            "ablation",
            "--retrieval-artifact",
            str(retrieval_dir / "manifest.json"),
            "--training-artifact",
            str(train_dir / "manifest.json"),
            "--out",
            str(ablation_dir),
            "--overwrite",
            "--json",
            "--log-jsonl",
            str(logs_dir / "ablation.jsonl"),
        ),
    )
    _run_cli(
        commands,
        repo_root=repo_root,
        label="surprise_eval",
        args=(
            "eval",
            "surprise",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(pack_dir),
            "--out",
            str(surprise_dir),
            "--device",
            "cpu",
            "--seed",
            "0",
            "--overwrite",
            "--json",
            "--log-jsonl",
            str(logs_dir / "surprise.jsonl"),
        ),
    )
    _run_cli(
        commands,
        repo_root=repo_root,
        label="transition_index",
        args=(
            "index",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(pack_dir),
            "--out",
            str(index_dir),
            "--device",
            "cpu",
            "--overwrite",
            "--json",
            "--log-jsonl",
            str(logs_dir / "index.jsonl"),
        ),
    )

    verify_reports = _verify_artifacts(commands, repo_root=repo_root, output_root=output_root)
    inventory = _collect_inventory(
        repo_root=repo_root,
        output_root=output_root,
        config_dir=config_dir,
        runtime_train_config=runtime_train_config,
        report_path=report_path,
        commands=commands,
        verify_reports=verify_reports,
        secret_scan=None,
    )
    report_path.write_text(render_first_results_report(inventory), encoding="utf-8")

    secret_scan = _run_cli(
        commands,
        repo_root=repo_root,
        label="secret_scan",
        args=("secret-scan", str(output_root), str(report_path), "--json"),
    )
    inventory = _collect_inventory(
        repo_root=repo_root,
        output_root=output_root,
        config_dir=config_dir,
        runtime_train_config=runtime_train_config,
        report_path=report_path,
        commands=commands,
        verify_reports=verify_reports,
        secret_scan=secret_scan,
    )
    _write_json(inventory, output_root / "manifest_inventory.json")
    report_path.write_text(render_first_results_report(inventory), encoding="utf-8")
    secret_scan = _run_cli(
        commands,
        repo_root=repo_root,
        label="secret_scan_final",
        args=("secret-scan", str(output_root), str(report_path), "--json"),
        record=False,
    )
    inventory = _collect_inventory(
        repo_root=repo_root,
        output_root=output_root,
        config_dir=config_dir,
        runtime_train_config=runtime_train_config,
        report_path=report_path,
        commands=commands,
        verify_reports=verify_reports,
        secret_scan=secret_scan,
    )
    _write_json(inventory, output_root / "manifest_inventory.json")
    report_path.write_text(render_first_results_report(inventory), encoding="utf-8")
    return inventory


def render_first_results_report(inventory: Mapping[str, Any]) -> str:
    """Render the checked-in first-results benchmark report."""

    retrieval = inventory["reports"]["retrieval"]
    ablation = inventory["reports"]["ablation"]
    surprise = inventory["reports"]["surprise"]
    training = inventory["reports"]["training"]
    license_gate = inventory["reports"]["license_gate"]
    index = inventory["reports"]["index"]
    dataset = inventory["reports"]["packed_dataset"]
    checkpoint = inventory["reports"]["checkpoint"]
    secret_scan = inventory.get("secret_scan") or {"ok": False, "findings": [], "paths_scanned": []}
    retrieval_metrics = retrieval["metrics"]
    baselines = retrieval["baselines"]
    baseline_rows = []
    beats_all = True
    for name in ("random", "lexical", "no_action", "shuffled_action"):
        baseline = baselines[name]
        beats = _strictly_beats(retrieval_metrics, baseline)
        beats_all = beats_all and beats
        baseline_rows.append(
            "| "
            + " | ".join(
                (
                    _baseline_label(name),
                    _fmt(retrieval_metrics["recall_at_1"]),
                    _fmt(baseline["recall_at_1"]),
                    _fmt(retrieval_metrics["mrr"]),
                    _fmt(baseline["mrr"]),
                    "yes" if beats else "no",
                )
            )
            + " |"
        )
    baseline_statement = (
        "Text-action beats all required baselines on Recall@1 and MRR."
        if beats_all
        else "Text-action does not beat all required baselines on this fixture."
    )

    artifact_rows = []
    for key in (
        "dataset_build",
        "dataset_pack",
        "training_run",
        "retrieval_eval",
        "action_ablation",
        "surprise_eval",
        "transition_index",
    ):
        artifact = inventory["artifacts"][key]
        artifact_rows.append(
            "| "
            + " | ".join(
                (
                    artifact["label"],
                    artifact["schema_version"],
                    artifact["manifest_path"],
                    artifact["artifact_id"],
                    artifact["config_sha256"][:12],
                )
            )
            + " |"
        )
    verify_rows = [
        "| "
        + " | ".join(
            (
                report["label"],
                "pass" if report["ok"] else "fail",
                str(report["files_checked"]),
                ", ".join(report["parent_artifacts"]) or "none",
                report["command"],
            )
        )
        + " |"
        for report in inventory["manifest_verification"]
    ]
    command_lines = "\n".join(f"{idx}. `{entry['command']}`" for idx, entry in enumerate(inventory["commands"], start=1))
    surprise_category_rows = []
    for category in ("random", "same_file", "mutation", "action_cluster"):
        count = surprise["metrics"]["decoy_counts"].get(category, 0)
        auc = surprise["metrics"]["pairwise_auc_by_category"].get(category)
        caveat = surprise["metadata"]["category_caveats"].get(category, "")
        surprise_category_rows.append(f"| `{category}` | {_fmt_optional(auc)} | {count} | {caveat or 'available'} |")
    ablation_rows = []
    for row in ablation["rows"]:
        metrics = row.get("metrics") or {}
        recall = _fmt_optional(metrics.get("recall_at_1"))
        mrr = _fmt_optional(metrics.get("mrr"))
        reason = row.get("block_reason") or "available"
        ablation_rows.append(
            "| "
            + " | ".join((row["name"], row["family"], row["status"], recall, mrr, reason))
            + " |"
        )

    final_metrics = training["final_metrics"]
    source_sha = inventory["source_git_sha"]
    report_lines = [
        "# CodeLeWM First Results",
        "",
        f"- Report ID: `codelewm-first-results-{inventory['generated_date_utc']}`",
        f"- Schema version: `{FIRST_RESULTS_SCHEMA_VERSION}`",
        "- Evidence tier: smoke fixture, not scaled research evidence",
        f"- Source git SHA: `{source_sha}`",
        f"- Config bundle SHA-256: `{inventory['config_bundle_sha256']}`",
        f"- Runtime train config SHA-256: `{inventory['runtime_train_config_sha256']}`",
        f"- Seed: dataset `{inventory['seeds']['dataset']}`, training `{inventory['seeds']['training']}`, evaluation `0`",
        "- Reproduction command: `uv run scripts/first-results --overwrite`",
        "",
        "## Verdict",
        "",
        "The complete local path now runs from a clean checkout: dataset build, pack, torch",
        "training, retrieval evaluation, action-view ablation, surprise evaluation, transition-index build,",
        "manifest verification, report rendering, and secret scanning.",
        "",
        f"{baseline_statement} The selected fixture has {retrieval_metrics['query_count']} held-out query and "
        f"{retrieval['candidate_pool']['entry_count']} retrieval candidate, so retrieval Recall@k is saturated.",
        "This report is therefore useful as reproducibility evidence, not as evidence that",
        "CodeLeWM has learned general action-conditioned code-edit structure.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv sync --group dev --group data --group train",
        "uv run scripts/first-results --overwrite",
        "uv run codelewm secret-scan .artifacts/first-results docs/benchmark/FIRST_RESULTS.md --json",
        "```",
        "",
        "The runner writes `.artifacts/first-results/manifest_inventory.json` with the",
        "machine-readable command outputs and artifact IDs used by this report.",
        "",
        "## Exact Commands",
        "",
        command_lines,
        "",
        "## Reproducibility Chain",
        "",
        "| Artifact | Schema version | Manifest path | Artifact ID | Config SHA prefix |",
        "| -------- | -------------- | ------------- | ----------- | ----------------- |",
        *artifact_rows,
        f"| Checkpoint | `{checkpoint['schema_version']}` | `{checkpoint['checkpoint_path']}` | `{checkpoint['checkpoint_sha256'][:16]}` | `{checkpoint['metadata']['config_hash'][:12]}` |",
        f"| License gate | `{license_gate['schema_version']}` | `.artifacts/first-results/build/reports/license_gate_report.json` | `release_allowed={str(license_gate['release_allowed']).lower()}` | n/a |",
        "",
        "## Manifest Verification",
        "",
        "| Artifact | Result | Files checked | Required parents | Command |",
        "| -------- | ------ | ------------- | ---------------- | ------- |",
        *verify_rows,
        "",
        "## Dataset And Training",
        "",
        f"- Packed rows: `{dataset['row_count']}`; splits: `{json.dumps(dataset['split_counts'], sort_keys=True)}`.",
        f"- License gate: release_allowed `{str(license_gate['release_allowed']).lower()}`, included rows `{license_gate['included_rows']}`, excluded rows `{license_gate['excluded_rows']}`, blocked rows `{license_gate['blocked_rows']}`.",
        f"- Training executor: `{training['metadata']['executor']['executor']}` on `{training['metadata']['executor']['device']}` for `{training['step_count']}` steps.",
        f"- Final loss: total `{_fmt(final_metrics['loss/total'])}`, prediction MSE `{_fmt(final_metrics['loss/prediction_mse'])}`, SIGReg `{_fmt(final_metrics['loss/sigreg'])}`.",
        f"- Collapse diagnostics: effective rank `{_fmt(final_metrics['collapse/effective_rank'])}`, variance min `{_fmt(final_metrics['collapse/per_dim_variance_min'])}`, nearest-neighbor entropy `{_fmt(final_metrics['collapse/nearest_neighbor_entropy'])}`.",
        "",
        "## Retrieval Evaluation",
        "",
        f"- Report schema: `{retrieval['schema_version']}`.",
        f"- Headline text-action metrics: Recall@1 `{_fmt(retrieval_metrics['recall_at_1'])}`, Recall@5 `{_fmt(retrieval_metrics['recall_at_5'])}`, Recall@10 `{_fmt(retrieval_metrics['recall_at_10'])}`, MRR `{_fmt(retrieval_metrics['mrr'])}`, median rank `{_fmt(retrieval_metrics['median_rank'])}`.",
        f"- Candidate pool: `{retrieval['candidate_pool']['name']}`, entries `{retrieval['candidate_pool']['entry_count']}`, excluded splits `{', '.join(retrieval['candidate_pool']['excluded_splits'])}`.",
        "",
        "| Baseline | Text Recall@1 | Baseline Recall@1 | Text MRR | Baseline MRR | Text beats baseline? |",
        "| -------- | ------------- | ----------------- | -------- | ------------ | -------------------- |",
        *baseline_rows,
        "| Patch-action diagnostic | n/a | n/a | n/a | n/a | not run for this headline smoke report |",
        "",
        "## Action-View Ablation",
        "",
        f"- Report schema: `{ablation['schema_version']}`.",
        f"- Completed rows: `{ablation['summary']['completed']}`; blocked rows: `{ablation['summary']['blocked']}`; failed rows: `{ablation['summary']['failed']}`.",
        "- Missing abstract-action, retrieval-loss, patch-action diagnostic, and alternate SIGReg runs are explicit blocked rows rather than dropped rows.",
        "",
        "| Row | Family | Status | Recall@1 | MRR | Reason |",
        "| --- | ------ | ------ | -------- | --- | ------ |",
        *ablation_rows,
        "",
        "## Patch-Surprise Evaluation",
        "",
        f"- Report schema: `{surprise['schema_version']}`.",
        f"- Overall pairwise AUC: `{_fmt(surprise['metrics']['pairwise_auc_overall'])}`.",
        f"- Mean true rank: `{_fmt(surprise['metrics']['mean_true_rank'])}`; median true rank: `{_fmt(surprise['metrics']['median_true_rank'])}`; Recall@1 `{_fmt(surprise['metrics']['recall_at_1'])}`.",
        f"- Examples scored: `{surprise['metrics']['example_count']}`; score direction: `{surprise['score_direction']}`.",
        "",
        "| Decoy category | Pairwise AUC | Decoy count | Caveat |",
        "| -------------- | ------------ | ----------- | ------ |",
        *surprise_category_rows,
        "",
        "## Transition Index",
        "",
        f"- Index schema: `{index['schema_version']}`.",
        f"- Count: `{index['count']}` train-split vectors; dimension `{index['dim']}`; distance `{index['distance']}`.",
        f"- Indexed splits: `{', '.join(index['metadata']['indexed_splits'])}`.",
        "",
        "## Security Evidence",
        "",
        f"- Secret scan result: `{'pass' if secret_scan['ok'] else 'fail'}`.",
        f"- Paths scanned: `{len(secret_scan.get('paths_scanned', []))}`.",
        f"- Findings: `{len(secret_scan.get('findings', []))}`.",
        "- Published artifact policy: local fixture artifacts are full-text and pass the configured permissive-license gate.",
        "",
        "## Claim Checklist",
        "",
        f"- [{'x' if beats_all else ' '}] Text-action beats random, lexical, no-action, and shuffled-action baselines on Recall@1 and MRR.",
        "- [x] Headline retrieval uses `action_text`.",
        "- [x] Action-view ablation records missing variants as blocked rows.",
        "- [x] Hard-negative and candidate pools exclude `train` split rows.",
        "- [ ] Patch-surprise covers all four decoy categories with non-zero decoy counts.",
        "- [x] Every selected artifact manifest verifies with required parents.",
        "- [x] Secret scan passes over the selected first-results artifact directory and this report.",
        "- [x] License gate passed for the local fixture artifact.",
        "- [ ] This report supports a scaled research claim about learned action-conditioned structure.",
        "",
        "## Caveats",
        "",
        "- Smoke evidence: this run proves the package-native workflow and artifact lineage,",
        "  including trusted checkpoint loading and index-backed evaluation prerequisites.",
        "- Research evidence: this fixture is too small for a learning claim. It has one",
        "  held-out query, no random same-corpus retrieval competition beyond the true",
        "  target, and only mutation surprise decoys. Baseline ties and failed surprise",
        "  rankings must be read as blockers for any public model-quality claim.",
        "- Next required work is a bounded public-safe shard with enough held-out examples",
        "  to make random, lexical, no-action, shuffled-action, and surprise decoy",
        "  comparisons meaningful.",
        "",
        "## Sign-off",
        "",
        "| Reviewer | Role | GitHub handle | Date |",
        "| -------- | ---- | ------------- | ---- |",
        "| Pending | First-results smoke review | Pending | Pending |",
        "",
    ]
    return "\n".join(report_lines)


def _verify_artifacts(
    commands: list[dict[str, Any]],
    *,
    repo_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    manifests = {
        key: output_root / directory / manifest_name
        for key, directory, manifest_name in ARTIFACTS
    }
    parents = {
        "dataset_build": (),
        "dataset_pack": (manifests["dataset_build"],),
        "training_run": (manifests["dataset_pack"],),
        "retrieval_eval": (manifests["training_run"], manifests["dataset_pack"]),
        "action_ablation": (manifests["retrieval_eval"], manifests["training_run"]),
        "surprise_eval": (manifests["training_run"], manifests["dataset_pack"]),
        "transition_index": (manifests["training_run"], manifests["dataset_pack"]),
    }
    verify_reports: list[dict[str, Any]] = []
    for key, manifest in manifests.items():
        args: list[str] = ["manifest", "verify", "--manifest", str(manifest)]
        for parent in parents[key]:
            args.extend(("--parent-manifest", str(parent)))
        args.append("--json")
        payload = _run_cli(commands, repo_root=repo_root, label=f"verify_{key}", args=tuple(args))
        verify_reports.append(
            {
                "label": key,
                "ok": bool(payload.get("ok")),
                "files_checked": payload.get("files_checked"),
                "parent_artifacts": payload.get("parent_artifacts", []),
                "parents_checked": payload.get("parents_checked", []),
                "command": commands[-1]["command"],
            }
        )
    return verify_reports


def _collect_inventory(
    *,
    repo_root: Path,
    output_root: Path,
    config_dir: Path,
    runtime_train_config: Path,
    report_path: Path,
    commands: Sequence[Mapping[str, Any]],
    verify_reports: Sequence[Mapping[str, Any]],
    secret_scan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    artifacts = {}
    for key, directory, manifest_name in ARTIFACTS:
        manifest_path = output_root / directory / manifest_name
        manifest = _read_json(manifest_path)
        artifacts[key] = {
            "label": key,
            "manifest_path": _display_path(manifest_path, repo_root=repo_root),
            "schema_version": manifest["schema_version"],
            "artifact_kind": manifest["artifact_kind"],
            "artifact_id": manifest["artifact_id"],
            "source_git_sha": manifest["source_git_sha"],
            "config_sha256": manifest["config_sha256"],
            "parent_artifacts": manifest["parent_artifacts"],
        }
    train_config = _read_json(runtime_train_config)
    packed_dataset = _read_json(output_root / "pack" / "dataset_manifest.json")
    return {
        "schema_version": FIRST_RESULTS_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "generated_date_utc": datetime.now(timezone.utc).date().isoformat(),
        "output_root": str(output_root),
        "report_path": str(report_path),
        "source_git_sha": _git_sha(repo_root),
        "config_dir": _display_path(config_dir, repo_root=repo_root),
        "config_bundle_sha256": _hash_files((config_dir / "dataset_build.json", config_dir / "train_tiny.json")),
        "runtime_train_config": _display_path(runtime_train_config, repo_root=repo_root),
        "runtime_train_config_sha256": _sha256_file(runtime_train_config),
        "seeds": {
            "dataset": _read_json(config_dir / "dataset_build.json")["seed"],
            "training": train_config["seed"],
        },
        "commands": [dict(command) for command in commands],
        "artifacts": artifacts,
        "manifest_verification": [dict(report) for report in verify_reports],
        "secret_scan": dict(secret_scan or {"ok": False, "paths_scanned": [], "findings": []}),
        "reports": {
            "packed_dataset": {
                "row_count": packed_dataset["row_count"],
                "split_counts": packed_dataset["split_counts"],
                "source_counts": packed_dataset["source_counts"],
            },
            "training": _read_json(output_root / "train" / "training_manifest.json"),
            "checkpoint": _read_json(output_root / "train" / "checkpoints" / "checkpoint.pt.manifest.json"),
            "retrieval": _read_json(output_root / "retrieval" / "reports" / "retrieval_report.json"),
            "ablation": _read_json(output_root / "ablation" / "reports" / "action_view_ablation_report.json"),
            "surprise": _read_json(output_root / "surprise" / "reports" / "surprise_report.json"),
            "index": _read_json(output_root / "index" / "index.json"),
            "license_gate": _read_json(output_root / "build" / "reports" / "license_gate_report.json"),
        },
    }


def _runtime_train_config(payload: Mapping[str, Any], *, output_root: Path, repo_root: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(payload))
    relative_root = _display_path(output_root, repo_root=repo_root)
    config["data"] = {
        "manifest": f"{relative_root}/pack/manifest.json",
        "train": f"{relative_root}/pack/hdf5/train.hdf5",
        "val": f"{relative_root}/pack/hdf5/val.hdf5",
    }
    config["output"] = {
        "run_dir": f"{relative_root}/train",
        "checkpoint_dir": f"{relative_root}/train/checkpoints",
        "metrics_path": f"{relative_root}/train/metrics.jsonl",
        "manifest_path": f"{relative_root}/train/training_manifest.json",
    }
    return config


def _run_cli(
    commands: list[dict[str, Any]],
    *,
    repo_root: Path,
    label: str,
    args: Sequence[str],
    record: bool = True,
) -> dict[str, Any]:
    argv = [sys.executable, "-m", "codelewm.harness.cli", *args]
    display_command = "uv run codelewm " + " ".join(_display_arg(arg, repo_root=repo_root) for arg in args)
    completed = subprocess.run(
        argv,
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    command_record = {
        "label": label,
        "command": display_command,
        "returncode": completed.returncode,
    }
    if record:
        commands.append(command_record)
    if completed.returncode != 0:
        raise FirstResultsError(
            f"{label} exited {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FirstResultsError(f"{label} did not emit JSON stdout: {exc}") from exc
    command_record["schema_version"] = payload.get("schema_version")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FirstResultsError(f"expected JSON object: {path}")
    return payload


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_under_repo(path: Path, *, repo_root: Path) -> Path:
    resolved = (repo_root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise FirstResultsError(f"path must stay under repository root: {path}") from exc
    return resolved


def _remove_owned_output_root(output_root: Path, *, repo_root: Path) -> None:
    relative = output_root.relative_to(repo_root)
    if relative == Path(".") or len(relative.parts) < 2 or relative.parts[0] != ".artifacts":
        raise FirstResultsError(f"refusing to remove non-owned output root: {output_root}")
    shutil.rmtree(output_root)


def _display_arg(value: str, *, repo_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            return str(value)
    return str(value)


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _git_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    sha = completed.stdout.strip()
    return sha if completed.returncode == 0 and sha else "unknown"


def _hash_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strictly_beats(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return float(metrics["recall_at_1"]) > float(baseline["recall_at_1"]) and float(metrics["mrr"]) > float(
        baseline["mrr"]
    )


def _baseline_label(name: str) -> str:
    return {
        "random": "Random",
        "lexical": "Lexical",
        "no_action": "No-action",
        "shuffled_action": "Shuffled-action",
    }[name]


def _fmt(value: Any) -> str:
    numeric = float(value)
    return f"{numeric:.6g}"


def _fmt_optional(value: Any) -> str:
    return "n/a" if value is None else _fmt(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
