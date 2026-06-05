#!/usr/bin/env bash
# CodeLeWM v0.8 runtime container entrypoint.
#
# Responsibilities:
#   1. Pre-download the execution pack from HF so the runner sees a local path
#      and does not need to authenticate with the Hub itself.
#   2. Forward the operator's CMD verbatim, typically
#      codelewm train --config ... --seed ...
#   3. After the operator's command exits cleanly, upload the run output
#      directory to a HF dataset repo so checkpoints, metrics JSONL, and reports
#      survive the container's ephemeral filesystem.
#
# Skipping rules for the pack pre-download:
#   * If $CODELEWM_EXECUTION_PACK_LOCAL_DIR already contains .populated, the
#     download is skipped.
#   * If CODELEWM_EXECUTION_PACK_REPO_ID or CODELEWM_EXECUTION_PACK_REVISION is
#     missing, the download is skipped and the runner fallback path takes over.
#   * If HF_TOKEN is not set, the entrypoint skips so the runner fallback can
#     surface the auth error directly.
#
# Skipping rules for the post-run upload:
#   * If CODELEWM_UPLOAD_REPO_ID is unset, the upload is skipped.
#   * If the operator's command exited non-zero, the upload is skipped.
#   * If $CODELEWM_RUN_OUTPUT_DIR does not exist after the run, the upload is
#     skipped.
#   * HF_TOKEN is required for the upload too.
#
# This script is intentionally short and dependency-free; it only expects hf
# (provided by huggingface_hub) on PATH.

set -euo pipefail

resolve_runtime_python() {
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    return 1
}

runtime_now_s() {
    date +%s
}

runtime_elapsed_s() {
    local started_at="$1"
    local now_s
    now_s="$(runtime_now_s)"
    echo "$((now_s - started_at))"
}

emit_runtime_event() {
    local event_name="$1"
    shift
    "${RUNTIME_PYTHON}" - "$event_name" "$@" <<'PY' >&2
import json
import math
import os
import re
import sys


def _coerce(value: str):
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    if re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
        return int(value)
    if re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?", value):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return value


event_name = sys.argv[1]
fields = {}
for raw_pair in sys.argv[2:]:
    key, sep, value = raw_pair.partition("=")
    if not sep or not key:
        continue
    fields[key] = _coerce(value)

payload = {
    "schema_version": "codelewm.log_event.v1",
    "event": event_name,
    "level": "info",
    "run_id": os.environ.get("CODELEWM_HF_RUN_NAME", "codelewm_runtime"),
    "step": "runtime",
    "message": event_name,
    "fields": fields,
}
print("CODELEWM_JOB_EVENT " + json.dumps(payload, sort_keys=True), flush=True)
PY
}

RUNTIME_PYTHON="$(resolve_runtime_python)" || {
    echo "[codelewm-runtime] python3 or python is required for structured runtime events" >&2
    exit 127
}
RUN_STARTED_AT="$(runtime_now_s)"

if [[ -z "${CODELEWM_EXECUTION_PACK_LOCAL_DIR:-}" ]]; then
    export CODELEWM_EXECUTION_PACK_LOCAL_DIR="/workspace/pack"
fi

mkdir -p "${CODELEWM_EXECUTION_PACK_LOCAL_DIR}"
POPULATED_MARKER="${CODELEWM_EXECUTION_PACK_LOCAL_DIR}/.populated"
COMMAND_NAME="${1:-}"

emit_runtime_event "runtime.start" \
    "run_name=${CODELEWM_HF_RUN_NAME:-unset}" \
    "pack_local_dir=${CODELEWM_EXECUTION_PACK_LOCAL_DIR}" \
    "command_name=${COMMAND_NAME:-unset}" \
    "command_arg_count=$#" \
    "has_pack_repo_id=$([[ -n "${CODELEWM_EXECUTION_PACK_REPO_ID:-}" ]] && echo true || echo false)" \
    "has_pack_revision=$([[ -n "${CODELEWM_EXECUTION_PACK_REVISION:-}" ]] && echo true || echo false)" \
    "has_upload_repo_id=$([[ -n "${CODELEWM_UPLOAD_REPO_ID:-}" ]] && echo true || echo false)"

if [[ -f "${POPULATED_MARKER}" ]]; then
    echo "[codelewm-runtime] pack already populated at ${CODELEWM_EXECUTION_PACK_LOCAL_DIR}"
    emit_runtime_event "runtime.pack_skip" \
        "reason=already_populated" \
        "pack_local_dir=${CODELEWM_EXECUTION_PACK_LOCAL_DIR}"
elif [[ -n "${CODELEWM_EXECUTION_PACK_REPO_ID:-}" \
        && -n "${CODELEWM_EXECUTION_PACK_REVISION:-}" \
        && -n "${HF_TOKEN:-}" ]]; then
    echo "[codelewm-runtime] downloading ${CODELEWM_EXECUTION_PACK_REPO_ID}@${CODELEWM_EXECUTION_PACK_REVISION}"
    PACK_DOWNLOAD_STARTED_AT="$(runtime_now_s)"
    emit_runtime_event "runtime.pack_download_start" \
        "pack_repo_id=${CODELEWM_EXECUTION_PACK_REPO_ID}" \
        "pack_revision=${CODELEWM_EXECUTION_PACK_REVISION}" \
        "pack_local_dir=${CODELEWM_EXECUTION_PACK_LOCAL_DIR}"
    set +e
    hf download \
        "${CODELEWM_EXECUTION_PACK_REPO_ID}" \
        --repo-type dataset \
        --revision "${CODELEWM_EXECUTION_PACK_REVISION}" \
        --local-dir "${CODELEWM_EXECUTION_PACK_LOCAL_DIR}"
    PACK_DOWNLOAD_EXIT_CODE=$?
    set -e
    if [[ "${PACK_DOWNLOAD_EXIT_CODE}" -ne 0 ]]; then
        emit_runtime_event "runtime.pack_download_failed" \
            "exit_code=${PACK_DOWNLOAD_EXIT_CODE}" \
            "elapsed_seconds=$(runtime_elapsed_s "${PACK_DOWNLOAD_STARTED_AT}")"
        exit "${PACK_DOWNLOAD_EXIT_CODE}"
    fi
    touch "${POPULATED_MARKER}"
    emit_runtime_event "runtime.pack_download_complete" \
        "elapsed_seconds=$(runtime_elapsed_s "${PACK_DOWNLOAD_STARTED_AT}")" \
        "pack_local_dir=${CODELEWM_EXECUTION_PACK_LOCAL_DIR}"
else
    echo "[codelewm-runtime] skipping HF pack download (pack-repo, revision, or HF_TOKEN unset)"
    emit_runtime_event "runtime.pack_skip" \
        "reason=missing_pack_repo_revision_or_token" \
        "has_pack_repo_id=$([[ -n "${CODELEWM_EXECUTION_PACK_REPO_ID:-}" ]] && echo true || echo false)" \
        "has_pack_revision=$([[ -n "${CODELEWM_EXECUTION_PACK_REVISION:-}" ]] && echo true || echo false)" \
        "has_hf_token=$([[ -n "${HF_TOKEN:-}" ]] && echo true || echo false)"
fi

# Run the operator-supplied command. We deliberately do not exec here so the
# entrypoint regains control afterwards and can upload the run output.
COMMAND_STARTED_AT="$(runtime_now_s)"
emit_runtime_event "runtime.command_start" \
    "command_name=${COMMAND_NAME:-unset}" \
    "command_arg_count=$#"
set +e
"$@"
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo "[codelewm-runtime] training exited with code ${EXIT_CODE}; skipping artifact upload"
    emit_runtime_event "runtime.command_failed" \
        "exit_code=${EXIT_CODE}" \
        "elapsed_seconds=$(runtime_elapsed_s "${COMMAND_STARTED_AT}")" \
        "upload_skipped=true"
    exit "${EXIT_CODE}"
fi
emit_runtime_event "runtime.command_complete" \
    "exit_code=${EXIT_CODE}" \
    "elapsed_seconds=$(runtime_elapsed_s "${COMMAND_STARTED_AT}")"

if [[ -z "${CODELEWM_UPLOAD_REPO_ID:-}" ]]; then
    echo "[codelewm-runtime] CODELEWM_UPLOAD_REPO_ID not set; skipping artifact upload"
    emit_runtime_event "runtime.upload_skip" \
        "reason=missing_upload_repo_id" \
        "elapsed_seconds=$(runtime_elapsed_s "${RUN_STARTED_AT}")"
    exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[codelewm-runtime] HF_TOKEN not set; cannot upload artifacts"
    emit_runtime_event "runtime.upload_skip" \
        "reason=missing_hf_token" \
        "elapsed_seconds=$(runtime_elapsed_s "${RUN_STARTED_AT}")"
    exit 0
fi

# Resolve the output dir the runner wrote to. The launcher always passes
# --out <CODELEWM_RUN_OUTPUT_DIR> so the dir is known.
RUN_DIR="${CODELEWM_RUN_OUTPUT_DIR:-/tmp/runs}"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[codelewm-runtime] run output dir ${RUN_DIR} does not exist; skipping artifact upload"
    emit_runtime_event "runtime.upload_skip" \
        "reason=missing_run_output_dir" \
        "run_output_dir=${RUN_DIR}" \
        "elapsed_seconds=$(runtime_elapsed_s "${RUN_STARTED_AT}")"
    exit 0
fi

# Where to write inside the HF repo. Default to the HF Jobs run name so the two
# seed runs do not collide.
UPLOAD_PATH_IN_REPO="${CODELEWM_UPLOAD_PATH_IN_REPO:-${CODELEWM_HF_RUN_NAME:-run}}"

echo "[codelewm-runtime] uploading ${RUN_DIR} -> ${CODELEWM_UPLOAD_REPO_ID}:${UPLOAD_PATH_IN_REPO}"
UPLOAD_STARTED_AT="$(runtime_now_s)"
emit_runtime_event "runtime.upload_start" \
    "upload_repo_id=${CODELEWM_UPLOAD_REPO_ID}" \
    "upload_path_in_repo=${UPLOAD_PATH_IN_REPO}" \
    "run_output_dir=${RUN_DIR}"
set +e
hf upload \
    "${CODELEWM_UPLOAD_REPO_ID}" \
    "${RUN_DIR}" \
    "${UPLOAD_PATH_IN_REPO}" \
    --repo-type dataset \
    --revision main \
    --commit-message "codelewm v0.8 run ${CODELEWM_HF_RUN_NAME:-unset}"
UPLOAD_EXIT_CODE=$?
set -e
if [[ "${UPLOAD_EXIT_CODE}" -ne 0 ]]; then
    emit_runtime_event "runtime.upload_failed" \
        "exit_code=${UPLOAD_EXIT_CODE}" \
        "elapsed_seconds=$(runtime_elapsed_s "${UPLOAD_STARTED_AT}")" \
        "upload_path_in_repo=${UPLOAD_PATH_IN_REPO}"
    exit "${UPLOAD_EXIT_CODE}"
fi
echo "[codelewm-runtime] upload complete"
emit_runtime_event "runtime.upload_complete" \
    "elapsed_seconds=$(runtime_elapsed_s "${UPLOAD_STARTED_AT}")" \
    "upload_path_in_repo=${UPLOAD_PATH_IN_REPO}"
emit_runtime_event "runtime.complete" \
    "elapsed_seconds=$(runtime_elapsed_s "${RUN_STARTED_AT}")"
exit 0
