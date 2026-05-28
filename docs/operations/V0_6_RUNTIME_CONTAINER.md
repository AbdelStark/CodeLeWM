# v0.6 Runtime Container

The `abdelstark/codelewm-runtime:v0.6` container image is the HF Jobs
runtime referenced by the v0.6 launch plan
(`scripts/hf-launch-execution-run`). This doc captures the build,
smoke, and publish workflow.

## What's In The Image

- Base: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (overridable via
  `--build-arg BASE_IMAGE=...`).
- `uv` package manager installed at `/usr/local/bin/uv`.
- The CodeLeWM repository checked out under `/workspace/` and installed
  via `uv pip install --system --no-deps .` plus the runtime extras
  (`numpy`, `h5py`, `pyarrow`, `einops`, `hydra-core`, `omegaconf`,
  `tensorboard`, `datasets`, `huggingface_hub`).
- The `codelewm` console script and the project's helper scripts on
  PATH for the launcher's command vector.
- Non-root `codelewm` user (uid 1000) by default.
- Empty `ENTRYPOINT` so the launcher's exact `codelewm train ...`
  command vector runs unmodified.
- OCI labels recording the image title, version, source URL, and
  build-time `CODELEWM_GIT_SHA`.

The expected env vars the launcher passes are declared as documentary
`ENV X=""` in the Dockerfile so `docker inspect` surfaces them:

| Variable | Set by | Used by |
|----------|--------|---------|
| `CODELEWM_HF_RUN_NAME` | launcher | run-name embedded into the manifest |
| `CODELEWM_EXECUTION_PACK_REPO_ID` | launcher | dataset download |
| `CODELEWM_EXECUTION_PACK_REVISION` | launcher | dataset download |
| `CODELEWM_TRAIN_SEED` | launcher | seed override |
| `CODELEWM_TRAIN_CONFIG` | launcher | config path |

## Build

```bash
# Default tag: abdelstark/codelewm-runtime:v0.6
scripts/build-codelewm-runtime

# Custom tag + linux/amd64 platform (HF Jobs only runs amd64):
scripts/build-codelewm-runtime --tag myreg/codelewm-runtime:v0.6-dev --platform linux/amd64

# Build + push in one step (requires logged-in registry):
scripts/build-codelewm-runtime --push
```

The script prints the resulting image digest. Capture it for the
artifact lineage.

## Smoke Test

```bash
# 1. Console script resolves.
docker run --rm abdelstark/codelewm-runtime:v0.6 codelewm --version

# 2. Execution-substrate smoke runner is on PATH.
docker run --rm abdelstark/codelewm-runtime:v0.6 \
  codelewm-execution-train-smoke --help

# 3. Full smoke train on the in-image MBPP fixture (no GPU needed).
docker run --rm abdelstark/codelewm-runtime:v0.6 \
  codelewm-execution-train-smoke \
  --ingestion tests/data/execution_sources/fixtures/mbpp_tiny.jsonl \
  --out /tmp/smoke --batch-size 2 --max-steps 80 \
  --device cpu --json
```

The third command runs the same end-to-end smoke as
`docs/benchmark/EXECUTION_V0_6_LOCAL_SMOKE_2026-05-28.md` but from
inside the container, validating that the image really can run the
v0.6 path.

## Publish To A Registry

HF Jobs accepts images from public registries. Two common options:

### Hugging Face Hub Spaces

```bash
# Authenticate
hf auth login

# Tag with the HF registry namespace and push
docker tag abdelstark/codelewm-runtime:v0.6 \
    registry.hf.co/abdelstark/codelewm-runtime:v0.6
docker push registry.hf.co/abdelstark/codelewm-runtime:v0.6
```

### Docker Hub

```bash
docker login
docker push abdelstark/codelewm-runtime:v0.6
```

After pushing, confirm the v0.6 launch plan still resolves:

```bash
PYTHONPATH=. python scripts/hf-launch-execution-run --json | \
    jq '.[0].command'
```

The emitted `hf jobs run ...` command vector includes
`abdelstark/codelewm-runtime:v0.6` as the image reference. If the
registry path needs to change, update
`config/train/scaled/codelewm_execution_v0_6_a10g.yaml`'s
`hf_jobs.runtime_image` (the field is not present in the v0.6 config
today; adding it is a follow-on if the published image path differs
from the launcher default).

## Threat Model

- The image bakes the repo at the build-time git SHA. Re-run the build
  to pick up new code.
- The container runs as a non-root user; no `sudo`, no privileged
  capabilities required.
- The sandbox subsystem (`codelewm.data.sandbox`) and the LLM demo
  scenario (`execution-rerank-mbpp`) still rely on the existing
  in-process subprocess isolation. The container is a higher-level
  boundary, not a replacement for the sandbox.

## Out-Of-Scope

- arm64 build (HF Jobs is amd64 only).
- TensorRT / DeepSpeed extras.
- A multi-stage strip-down for image-size optimization. The base
  PyTorch image is ~5 GB; a multi-stage build can shave off the
  toolchain layers but is not urgent.

## Reference

- Tracker: #289 (v0.6 end-to-end completion)
- Implementation issue: #292
- Launcher: `scripts/hf-launch-execution-run`
- Launch-plan generator: `codelewm.training.execution_launch_plan.build_launch_plans`
- Pack publish: `docs/benchmark/EXECUTION_PACK_PUBLISH_2026-05-28.md`
- Operator runbook: `docs/operations/V0_6_EXECUTION_RUN_RUNBOOK.md`
