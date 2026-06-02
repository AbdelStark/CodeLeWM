# CodeLeWM v0.7 execution-substrate runtime image.
#
# Identical contract to the v0.6 runtime (see
# docs/operations/V0_6_RUNTIME_CONTAINER.md) but baked from the v0.7
# source tree, which carries the RFC-0015 WS-C architecture levers:
#   - the transformer state encoder (wm.state_encoder_type=transformer),
#   - the in-batch InfoNCE retrieval term (objective.retrieval_weight),
#   - the applied prediction_mse_weight.
# Because the package source is COPYed in at build time, v0.7 training
# REQUIRES a v0.7 image — the v0.6 image does not contain this code.
#
# Build:    scripts/build-codelewm-runtime \
#             --tag ghcr.io/abdelstark/codelewm-runtime:v0.7 \
#             --dockerfile containers/v0_7/Dockerfile
# Smoke:    docker run --rm ghcr.io/abdelstark/codelewm-runtime:v0.7 codelewm --version
# Publish:  see docs/operations/V0_6_RUNTIME_CONTAINER.md (same flow, v0.7 tag)

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE} AS base

# Documentary env vars — the launcher (#280) sets these per-run.
# Cache dirs are pinned to /tmp because HF Jobs runs the container as a
# non-root UID with HOME read-only (a /root/.cache write fails with
# Permission denied even though the USER directive says root). /tmp is
# always writable.
ENV CODELEWM_HF_RUN_NAME="" \
    CODELEWM_EXECUTION_PACK_REPO_ID="" \
    CODELEWM_EXECUTION_PACK_REVISION="" \
    CODELEWM_TRAIN_SEED="" \
    CODELEWM_TRAIN_CONFIG="" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    HF_HOME=/tmp/huggingface \
    HF_HUB_CACHE=/tmp/huggingface/hub

ARG CODELEWM_GIT_SHA=unset
LABEL org.opencontainers.image.title="codelewm-runtime"
LABEL org.opencontainers.image.version="v0.7"
LABEL org.opencontainers.image.source="https://github.com/AbdelStark/CodeLeWM"
LABEL org.opencontainers.image.description="CodeLeWM v0.7 execution-substrate runtime (HF Jobs)"
LABEL org.opencontainers.image.revision=${CODELEWM_GIT_SHA}

# Install ``uv`` (the package manager the project standardizes on) and
# the minimum OS toolchain. ``git`` is required because some optional
# train deps install from VCS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace

# Copy the project in. ``pyproject.toml`` first lets the dep-install
# layer cache when only source changes.
COPY pyproject.toml uv.lock README.md LICENSE /workspace/
COPY codelewm /workspace/codelewm/
COPY config /workspace/config/
COPY scripts /workspace/scripts/
COPY containers/v0_7/entrypoint.sh /usr/local/bin/codelewm-runtime-entrypoint
RUN chmod +x /usr/local/bin/codelewm-runtime-entrypoint

# Install codelewm with the runtime extras the launch plan + smoke
# runner need. ``--system`` installs into the base image's Python,
# matching the existing torch + cuda toolchain. ``--no-deps .`` keeps
# the project layer cacheable; the explicit dep list pins the runtime
# surface so a pyproject change can't silently drift the image.
RUN uv pip install --system --no-cache --no-deps . \
    && uv pip install --system --no-cache \
        "numpy>=1.24" "h5py>=3.10" "pyarrow>=15" \
        "einops>=0.7" "hydra-core>=1.3" "omegaconf>=2.3" \
        "tensorboard>=2.16" \
        "datasets>=3" "huggingface_hub>=1.0"

# Smoke-test that the package imports cleanly in this image.
RUN python -c "import codelewm; print('codelewm', codelewm.__version__)"

# Pack workspace + cache dirs the entrypoint will populate.
RUN mkdir -p /workspace/pack /workspace/runs

# Run as a non-root user.
RUN useradd -m -u 1000 codelewm \
    && chown -R codelewm:codelewm /workspace

# Document the pack-local-dir env var the runner reads first. The
# production runner (#293) short-circuits its HF download when this
# directory holds the pack JSONL + manifest.
ENV CODELEWM_EXECUTION_PACK_LOCAL_DIR="/workspace/pack" \
    CODELEWM_RUN_OUTPUT_DIR="/workspace/runs"

USER codelewm

# Default entry pre-downloads the pack (if not already populated)
# and then forwards the operator's CMD. The launcher's HF Jobs
# invocation overrides CMD with ``uv run codelewm train ...``.
ENTRYPOINT ["/usr/local/bin/codelewm-runtime-entrypoint"]
CMD ["codelewm", "--help"]
