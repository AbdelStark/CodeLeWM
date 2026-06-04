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

if [[ -z "${CODELEWM_EXECUTION_PACK_LOCAL_DIR:-}" ]]; then
    export CODELEWM_EXECUTION_PACK_LOCAL_DIR="/workspace/pack"
fi

mkdir -p "${CODELEWM_EXECUTION_PACK_LOCAL_DIR}"
POPULATED_MARKER="${CODELEWM_EXECUTION_PACK_LOCAL_DIR}/.populated"

if [[ -f "${POPULATED_MARKER}" ]]; then
    echo "[codelewm-runtime] pack already populated at ${CODELEWM_EXECUTION_PACK_LOCAL_DIR}"
elif [[ -n "${CODELEWM_EXECUTION_PACK_REPO_ID:-}" \
        && -n "${CODELEWM_EXECUTION_PACK_REVISION:-}" \
        && -n "${HF_TOKEN:-}" ]]; then
    echo "[codelewm-runtime] downloading ${CODELEWM_EXECUTION_PACK_REPO_ID}@${CODELEWM_EXECUTION_PACK_REVISION}"
    hf download \
        "${CODELEWM_EXECUTION_PACK_REPO_ID}" \
        --repo-type dataset \
        --revision "${CODELEWM_EXECUTION_PACK_REVISION}" \
        --local-dir "${CODELEWM_EXECUTION_PACK_LOCAL_DIR}"
    touch "${POPULATED_MARKER}"
else
    echo "[codelewm-runtime] skipping HF pack download (pack-repo, revision, or HF_TOKEN unset)"
fi

# Run the operator-supplied command. We deliberately do not exec here so the
# entrypoint regains control afterwards and can upload the run output.
set +e
"$@"
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo "[codelewm-runtime] training exited with code ${EXIT_CODE}; skipping artifact upload"
    exit "${EXIT_CODE}"
fi

if [[ -z "${CODELEWM_UPLOAD_REPO_ID:-}" ]]; then
    echo "[codelewm-runtime] CODELEWM_UPLOAD_REPO_ID not set; skipping artifact upload"
    exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[codelewm-runtime] HF_TOKEN not set; cannot upload artifacts"
    exit 0
fi

# Resolve the output dir the runner wrote to. The launcher always passes
# --out <CODELEWM_RUN_OUTPUT_DIR> so the dir is known.
RUN_DIR="${CODELEWM_RUN_OUTPUT_DIR:-/tmp/runs}"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[codelewm-runtime] run output dir ${RUN_DIR} does not exist; skipping artifact upload"
    exit 0
fi

# Where to write inside the HF repo. Default to the HF Jobs run name so the two
# seed runs do not collide.
UPLOAD_PATH_IN_REPO="${CODELEWM_UPLOAD_PATH_IN_REPO:-${CODELEWM_HF_RUN_NAME:-run}}"

echo "[codelewm-runtime] uploading ${RUN_DIR} -> ${CODELEWM_UPLOAD_REPO_ID}:${UPLOAD_PATH_IN_REPO}"
hf upload \
    "${CODELEWM_UPLOAD_REPO_ID}" \
    "${RUN_DIR}" \
    "${UPLOAD_PATH_IN_REPO}" \
    --repo-type dataset \
    --revision main \
    --commit-message "codelewm v0.8 run ${CODELEWM_HF_RUN_NAME:-unset}"
echo "[codelewm-runtime] upload complete"
exit 0
