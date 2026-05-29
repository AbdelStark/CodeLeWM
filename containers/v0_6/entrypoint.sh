#!/usr/bin/env bash
# CodeLeWM v0.6 runtime container entrypoint.
#
# Responsibilities:
#   1. Pre-download the execution pack from HF so the runner sees a
#      local path and does not need to authenticate with the Hub
#      itself. This keeps the in-process runner off the network for
#      the dataset round-trip and matches the
#      ``CODELEWM_EXECUTION_PACK_LOCAL_DIR`` resolution path the
#      production runner (#293) already checks first.
#   2. Forward the operator's CMD verbatim (typically
#      ``uv run codelewm train --config ... --seed ...``).
#
# Skipping rules:
#   * If ``$CODELEWM_EXECUTION_PACK_LOCAL_DIR`` already contains
#     ``.populated`` (e.g. a bind-mounted pre-built pack), the
#     download is skipped.
#   * If ``CODELEWM_EXECUTION_PACK_REPO_ID`` or
#     ``CODELEWM_EXECUTION_PACK_REVISION`` is missing, the download
#     is skipped — the runner's HF fallback path takes over.
#   * If ``HF_TOKEN`` is not set, the entrypoint also skips so the
#     runner's fallback can surface the auth error directly.
#
# This script is intentionally short and dependency-free; it only
# expects ``hf`` (provided by ``huggingface_hub``) on PATH.

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

# Hand off to the operator-supplied command (CMD or `docker run` argv).
exec "$@"
