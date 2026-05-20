"""Build release provenance evidence for package artifacts and audit reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELEASE_PROVENANCE_SCHEMA_VERSION = "codelewm.release_provenance.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ReleaseProvenanceError(ValueError):
    """Raised when release provenance inputs or payloads are invalid."""


def build_release_provenance(
    *,
    dist_dir: Path | str,
    root: Path | str = ".",
    lockfile: Path | str = "uv.lock",
    audit_report: Path | str | None = None,
    include_paths: Sequence[Path | str] = (),
    source_git_sha: str | None = None,
    command: Sequence[str] = ("scripts/release-provenance",),
    require_clean_tracked_tree: bool = False,
) -> dict[str, Any]:
    """Return a JSON-native release provenance report.

    Paths are recorded relative to ``root`` so the report is shareable across
    machines and does not disclose local absolute paths.
    """

    root_path = Path(root).resolve()
    dist_path = _resolve_under_root(dist_dir, root=root_path)
    lockfile_path = _resolve_under_root(lockfile, root=root_path)

    if not dist_path.is_dir():
        raise ReleaseProvenanceError(f"dist directory does not exist: {_display_path(dist_path, root=root_path)}")
    if not lockfile_path.is_file():
        raise ReleaseProvenanceError(f"lockfile does not exist: {_display_path(lockfile_path, root=root_path)}")

    distributions = _distribution_entries(dist_path, root=root_path)
    tracked_status = _tracked_git_status(root_path)
    tracked_dirty = bool(tracked_status)
    if require_clean_tracked_tree and tracked_dirty:
        raise ReleaseProvenanceError("tracked git tree is dirty; commit or discard tracked changes before release")

    resolved_source_git_sha = source_git_sha or _detect_git_sha(root_path)
    if resolved_source_git_sha != "unknown" and not _GIT_SHA_RE.fullmatch(resolved_source_git_sha):
        raise ReleaseProvenanceError("source_git_sha must be a 40-character lowercase git SHA or 'unknown'")

    audit_payload = None
    if audit_report is not None:
        audit_payload = _file_entry(_resolve_under_root(audit_report, root=root_path), root=root_path)

    included_evidence = []
    seen_paths = {
        entry["path"] for entry in distributions
    } | {_file_entry(lockfile_path, root=root_path)["path"]}
    if audit_payload is not None:
        seen_paths.add(audit_payload["path"])
    for include_path in include_paths:
        entry = _file_entry(_resolve_under_root(include_path, root=root_path), root=root_path)
        if entry["path"] not in seen_paths:
            included_evidence.append(entry)
            seen_paths.add(entry["path"])

    payload = {
        "schema_version": RELEASE_PROVENANCE_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "source_git_sha": resolved_source_git_sha,
        "tracked_git_dirty": tracked_dirty,
        "tracked_git_status": tracked_status,
        "command": list(command),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "lockfile": _file_entry(lockfile_path, root=root_path),
        "built_distributions": distributions,
        "dependency_audit": {
            "tool": "pip-audit",
            "scope": "installed base plus dev and release dependency groups",
            "report": audit_payload,
            "failure_policy": "nonzero pip-audit exit blocks release unless a signed waiver is recorded",
        },
        "included_evidence": included_evidence,
    }
    return validate_release_provenance_payload(payload)


def write_release_provenance(payload: Mapping[str, Any], output_path: Path | str) -> Path:
    """Validate and write a release provenance report."""

    checked = validate_release_provenance_payload(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_release_provenance_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a release provenance v1 payload and return a plain dict."""

    required = {
        "schema_version",
        "generated_at",
        "source_git_sha",
        "tracked_git_dirty",
        "tracked_git_status",
        "command",
        "environment",
        "lockfile",
        "built_distributions",
        "dependency_audit",
        "included_evidence",
    }
    _reject_unknown(payload, required, "release provenance")
    checked = dict(payload)
    if checked["schema_version"] != RELEASE_PROVENANCE_SCHEMA_VERSION:
        raise ReleaseProvenanceError("unsupported release provenance schema_version")
    if not isinstance(checked["generated_at"], str):
        raise ReleaseProvenanceError("generated_at must be a string")
    datetime.fromisoformat(checked["generated_at"].replace("Z", "+00:00"))
    if not isinstance(checked["source_git_sha"], str):
        raise ReleaseProvenanceError("source_git_sha must be a string")
    if checked["source_git_sha"] != "unknown" and not _GIT_SHA_RE.fullmatch(checked["source_git_sha"]):
        raise ReleaseProvenanceError("source_git_sha must be a 40-character lowercase git SHA or 'unknown'")
    if not isinstance(checked["tracked_git_dirty"], bool):
        raise ReleaseProvenanceError("tracked_git_dirty must be a bool")
    _require_string_list(checked["tracked_git_status"], "tracked_git_status")
    _require_string_list(checked["command"], "command", min_items=1)
    _validate_environment(checked["environment"])
    checked["lockfile"] = _validate_file_entry(checked["lockfile"], "lockfile")
    distributions = _validate_file_entry_list(checked["built_distributions"], "built_distributions", min_items=2)
    if sum(1 for entry in distributions if entry["path"].endswith(".whl")) != 1:
        raise ReleaseProvenanceError("built_distributions must contain exactly one wheel")
    if sum(1 for entry in distributions if entry["path"].endswith(".tar.gz")) != 1:
        raise ReleaseProvenanceError("built_distributions must contain exactly one source distribution")
    checked["built_distributions"] = distributions
    checked["dependency_audit"] = _validate_dependency_audit(checked["dependency_audit"])
    checked["included_evidence"] = _validate_file_entry_list(checked["included_evidence"], "included_evidence")
    return checked


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root; defaults to the cwd")
    parser.add_argument("--dist", type=Path, required=True, help="directory containing one wheel and one sdist")
    parser.add_argument("--lockfile", type=Path, default=Path("uv.lock"), help="lockfile path relative to root")
    parser.add_argument("--audit-report", type=Path, help="pip-audit JSON report path relative to root")
    parser.add_argument("--include", dest="includes", action="append", type=Path, default=[], help="extra evidence file")
    parser.add_argument("--out", type=Path, required=True, help="output provenance JSON path")
    parser.add_argument("--source-git-sha", help="override detected git SHA")
    parser.add_argument(
        "--require-clean-tracked-tree",
        action="store_true",
        help="fail if tracked files have uncommitted changes",
    )
    parser.add_argument("--json", action="store_true", help="print the provenance payload to stdout")
    args = parser.parse_args(argv)

    command = ["scripts/release-provenance", *(argv if argv is not None else sys.argv[1:])]
    try:
        payload = build_release_provenance(
            root=args.root,
            dist_dir=args.dist,
            lockfile=args.lockfile,
            audit_report=args.audit_report,
            include_paths=tuple(args.includes),
            source_git_sha=args.source_git_sha,
            command=command,
            require_clean_tracked_tree=args.require_clean_tracked_tree,
        )
        output_path = write_release_provenance(payload, args.out)
    except ReleaseProvenanceError as exc:
        print(f"release-provenance: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"wrote {output_path}")
    return 0


def _distribution_entries(dist_path: Path, *, root: Path) -> list[dict[str, Any]]:
    wheels = sorted(path for path in dist_path.iterdir() if path.name.endswith(".whl"))
    sdists = sorted(path for path in dist_path.iterdir() if path.name.endswith(".tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseProvenanceError("dist directory must contain exactly one wheel and one source distribution")
    return [_file_entry(path, root=root) for path in (*wheels, *sdists)]


def _resolve_under_root(path: Path | str, *, root: Path) -> Path:
    raw_path = Path(path)
    resolved = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReleaseProvenanceError(f"path is outside repository root: {raw_path}") from exc
    return resolved


def _file_entry(path: Path, *, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseProvenanceError(f"evidence file does not exist: {_display_path(path, root=root)}")
    return {
        "path": _display_path(path, root=root),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _display_path(path: Path, *, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_git_sha(root: Path) -> str:
    output = _git_output(root, "rev-parse", "HEAD")
    return output if output and _GIT_SHA_RE.fullmatch(output) else "unknown"


def _tracked_git_status(root: Path) -> list[str]:
    output = _git_output(root, "status", "--porcelain", "--untracked-files=no")
    return [] if not output else output.splitlines()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    missing = sorted(allowed - payload.keys())
    extra = sorted(payload.keys() - allowed)
    if missing:
        raise ReleaseProvenanceError(f"{label} missing required key(s): {', '.join(missing)}")
    if extra:
        raise ReleaseProvenanceError(f"{label} contains unknown key(s): {', '.join(extra)}")


def _require_string_list(value: Any, label: str, *, min_items: int = 0) -> list[str]:
    if not isinstance(value, list):
        raise ReleaseProvenanceError(f"{label} must be a list")
    if len(value) < min_items:
        raise ReleaseProvenanceError(f"{label} must contain at least {min_items} item(s)")
    if any(not isinstance(item, str) or not item for item in value):
        raise ReleaseProvenanceError(f"{label} must contain non-empty strings")
    return list(value)


def _validate_environment(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ReleaseProvenanceError("environment must be an object")
    _reject_unknown(value, {"python", "platform"}, "environment")
    environment = dict(value)
    if not isinstance(environment["python"], str) or not environment["python"]:
        raise ReleaseProvenanceError("environment.python must be a non-empty string")
    if not isinstance(environment["platform"], str) or not environment["platform"]:
        raise ReleaseProvenanceError("environment.platform must be a non-empty string")
    return environment


def _validate_file_entry(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseProvenanceError(f"{label} must be an object")
    _reject_unknown(value, {"path", "sha256", "bytes"}, label)
    entry = dict(value)
    if not isinstance(entry["path"], str) or not entry["path"]:
        raise ReleaseProvenanceError(f"{label}.path must be a non-empty string")
    path = Path(entry["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ReleaseProvenanceError(f"{label}.path must be a repository-relative path")
    if not isinstance(entry["sha256"], str) or not _SHA256_RE.fullmatch(entry["sha256"]):
        raise ReleaseProvenanceError(f"{label}.sha256 must be a SHA-256 hex digest")
    if not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
        raise ReleaseProvenanceError(f"{label}.bytes must be a non-negative integer")
    return entry


def _validate_file_entry_list(value: Any, label: str, *, min_items: int = 0) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ReleaseProvenanceError(f"{label} must be a list")
    if len(value) < min_items:
        raise ReleaseProvenanceError(f"{label} must contain at least {min_items} item(s)")
    entries = [_validate_file_entry(item, f"{label}[]") for item in value]
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ReleaseProvenanceError(f"{label} must not contain duplicate paths")
    return entries


def _validate_dependency_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseProvenanceError("dependency_audit must be an object")
    _reject_unknown(value, {"tool", "scope", "report", "failure_policy"}, "dependency_audit")
    audit = dict(value)
    for key in ("tool", "scope", "failure_policy"):
        if not isinstance(audit[key], str) or not audit[key]:
            raise ReleaseProvenanceError(f"dependency_audit.{key} must be a non-empty string")
    if audit["report"] is not None:
        audit["report"] = _validate_file_entry(audit["report"], "dependency_audit.report")
    return audit


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
