# CodeLeWM v0.6 execution-substrate runtime image.
#
# Hugging Face Jobs runs each invocation inside a container. This image:
#
# - is rooted at a CUDA-runtime PyTorch base so A10G runs work out of
#   the box;
# - has codelewm + train / data / observability / llm extras installed
#   via uv pip install (mirroring the local dev contract);
# - exposes `uv`, `codelewm`, and `codelewm-execution-train-smoke` on
#   PATH for the launch plan;
# - runs as a non-root `codelewm` user;
# - documents the env vars the launcher (#280) passes.
#
# Build:    docker build -t abdelstark/codelewm-runtime:v0.6 \
#             --build-arg CODELEWM_GIT_SHA=$(git rev-parse HEAD) \
#             -f containers/v0_6/Dockerfile .
# Smoke:    docker run --rm abdelstark/codelewm-runtime:v0.6 codelewm --version
# Publish:  see docs/operations/V0_6_RUNTIME_CONTAINER.md

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE} AS base

# Documentary env vars — the launcher (#280) sets these per-run.
ENV CODELEWM_HF_RUN_NAME="" \
    CODELEWM_EXECUTION_PACK_REPO_ID="" \
    CODELEWM_EXECUTION_PACK_REVISION="" \
    CODELEWM_TRAIN_SEED="" \
    CODELEWM_TRAIN_CONFIG="" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy

ARG CODELEWM_GIT_SHA=unset
LABEL org.opencontainers.image.title="codelewm-runtime"
LABEL org.opencontainers.image.version="v0.6"
LABEL org.opencontainers.image.source="https://github.com/AbdelStark/CodeLeWM"
LABEL org.opencontainers.image.description="CodeLeWM v0.6 execution-substrate runtime (HF Jobs)"
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

# Install codelewm with the runtime extras the launch plan + smoke
# runner need. ``--system`` installs into the base image's Python,
# matching the existing torch + cuda toolchain.
RUN uv pip install --system --no-cache --no-deps . \
    && uv pip install --system --no-cache \
        "numpy>=1.24" "h5py>=3.10" "pyarrow>=15" \
        "einops>=0.7" "hydra-core>=1.3" "omegaconf>=2.3" \
        "tensorboard>=2.16" \
        "datasets>=3" "huggingface_hub>=0.24"

# Smoke-test that the package imports cleanly in this image.
RUN python -c "import codelewm; print('codelewm', codelewm.__version__)"

# Run as a non-root user.
RUN useradd -m -u 1000 codelewm \
    && chown -R codelewm:codelewm /workspace

USER codelewm

# Default entry is the project CLI so an ``hf jobs run ... codelewm
# train ...`` invocation works without an explicit interpreter shell.
ENTRYPOINT []
CMD ["codelewm", "--help"]
