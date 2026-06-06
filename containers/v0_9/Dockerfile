# CodeLeWM v0.9 correctness-aware execution-substrate runtime image.
#
# v0.9 keeps the v0.8 correctness co-training stack and adds the repaired
# cross-benchmark data/eval surfaces. Because the package source and configs are
# COPYed into the image, v0.9 jobs must use a v0.9 image built from the merged
# #391 source and pinned by digest in the launch plan. The operational launch
# config is config/train/scaled/codelewm_execution_v0_9_short_a10g.yaml.
#
# Build:    scripts/build-codelewm-runtime \
#             --tag ghcr.io/abdelstark/codelewm-runtime:v0.9 \
#             --dockerfile containers/v0_9/Dockerfile
# Smoke:    docker run --rm ghcr.io/abdelstark/codelewm-runtime:v0.9 codelewm --version
# Publish:  scripts/build-codelewm-runtime \
#             --tag ghcr.io/abdelstark/codelewm-runtime:v0.9 \
#             --dockerfile containers/v0_9/Dockerfile --push

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE} AS base

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
LABEL org.opencontainers.image.version="v0.9"
LABEL org.opencontainers.image.source="https://github.com/AbdelStark/CodeLeWM"
LABEL org.opencontainers.image.description="CodeLeWM v0.9 correctness-aware execution-substrate runtime (HF Jobs)"
LABEL org.opencontainers.image.revision=${CODELEWM_GIT_SHA}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md LICENSE /workspace/
COPY codelewm /workspace/codelewm/
COPY config /workspace/config/
COPY scripts /workspace/scripts/
COPY containers/v0_9/entrypoint.sh /usr/local/bin/codelewm-runtime-entrypoint
RUN chmod +x /usr/local/bin/codelewm-runtime-entrypoint

RUN uv pip install --system --no-cache --no-deps . \
    && uv pip install --system --no-cache \
        "numpy>=1.24" "h5py>=3.10" "pyarrow>=15" \
        "einops>=0.7" "hydra-core>=1.3" "omegaconf>=2.3" \
        "tensorboard>=2.16" \
        "datasets>=3" "huggingface_hub>=1.0"

RUN python -c "import codelewm; print('codelewm', codelewm.__version__)"
RUN mkdir -p /workspace/pack /workspace/runs
RUN useradd -m -u 1000 codelewm \
    && chown -R codelewm:codelewm /workspace

ENV CODELEWM_EXECUTION_PACK_LOCAL_DIR="/workspace/pack" \
    CODELEWM_RUN_OUTPUT_DIR="/workspace/runs"

USER codelewm

ENTRYPOINT ["/usr/local/bin/codelewm-runtime-entrypoint"]
CMD ["codelewm", "--help"]
