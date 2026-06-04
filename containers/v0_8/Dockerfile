# CodeLeWM v0.8 correctness-aware execution-substrate runtime image.
#
# Identical operational contract to the v0.7 runtime, but baked from
# the v0.8 source tree, which carries the correctness co-training stack:
#   - transformer state encoder (WS-C1),
#   - EMA target encoder extension (WS-C2),
#   - in-batch InfoNCE retrieval term (WS-C3),
#   - output-value auxiliary CE head (WS-C4),
#   - applied prediction_mse_weight (WS-C5),
#   - supervised p_pass BCE head and scorer compatibility.
# Because the package source is COPYed in at build time, v0.8 training
# REQUIRES a v0.8 image; earlier images do not contain the full head/objective
# stack.
#
# Build:    scripts/build-codelewm-runtime \
#             --tag ghcr.io/abdelstark/codelewm-runtime:v0.8 \
#             --dockerfile containers/v0_8/Dockerfile
# Smoke:    docker run --rm ghcr.io/abdelstark/codelewm-runtime:v0.8 codelewm --version
# Publish:  scripts/build-codelewm-runtime \
#             --tag ghcr.io/abdelstark/codelewm-runtime:v0.8 \
#             --dockerfile containers/v0_8/Dockerfile --push

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE} AS base

# Documentary env vars - the launcher sets these per run. Cache dirs are pinned
# to /tmp because HF Jobs runs the container as a non-root UID with HOME
# read-only. /tmp is writable.
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
LABEL org.opencontainers.image.version="v0.8"
LABEL org.opencontainers.image.source="https://github.com/AbdelStark/CodeLeWM"
LABEL org.opencontainers.image.description="CodeLeWM v0.8 correctness-aware execution-substrate runtime (HF Jobs)"
LABEL org.opencontainers.image.revision=${CODELEWM_GIT_SHA}

# Install uv and the minimum OS toolchain. git is required because some optional
# train deps install from VCS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace

# Copy the project in. pyproject.toml first lets the dependency install layer
# cache when only source changes.
COPY pyproject.toml uv.lock README.md LICENSE /workspace/
COPY codelewm /workspace/codelewm/
COPY config /workspace/config/
COPY scripts /workspace/scripts/
COPY containers/v0_8/entrypoint.sh /usr/local/bin/codelewm-runtime-entrypoint
RUN chmod +x /usr/local/bin/codelewm-runtime-entrypoint

# Install codelewm with the runtime extras the launch plan and smoke runner
# need. --system installs into the base image's Python, matching the existing
# torch + cuda toolchain. --no-deps . keeps the project layer cacheable; the
# explicit dep list pins the runtime surface so a pyproject change cannot
# silently drift the image.
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

# Document the pack-local-dir env var the runner reads first. The production
# runner short-circuits its HF download when this directory holds pack.jsonl +
# manifest.json.
ENV CODELEWM_EXECUTION_PACK_LOCAL_DIR="/workspace/pack" \
    CODELEWM_RUN_OUTPUT_DIR="/workspace/runs"

USER codelewm

# Default entry pre-downloads the pack if not already populated and then
# forwards the operator's CMD. HF Jobs overrides CMD with codelewm train ...
ENTRYPOINT ["/usr/local/bin/codelewm-runtime-entrypoint"]
CMD ["codelewm", "--help"]
