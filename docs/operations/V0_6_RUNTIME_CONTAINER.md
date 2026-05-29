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
- A thin entrypoint
  (`/usr/local/bin/codelewm-runtime-entrypoint`) that pre-downloads
  the execution pack into `$CODELEWM_EXECUTION_PACK_LOCAL_DIR`
  (default `/workspace/pack`) when
  `CODELEWM_EXECUTION_PACK_REPO_ID`, `CODELEWM_EXECUTION_PACK_REVISION`,
  and `HF_TOKEN` are all set. The production runner (#293)
  short-circuits its own HF download when that directory holds the
  pack; this keeps the runner off the network for the dataset
  round-trip even when the operator's CMD does not pass through
  `HF_TOKEN`. A bind-mounted pre-built pack with a `.populated`
  marker skips the download.
- OCI labels recording the image title, version, source URL, and
  build-time `CODELEWM_GIT_SHA`.

The expected env vars the launcher passes are declared as documentary
`ENV X=""` in the Dockerfile so `docker inspect` surfaces them:

| Variable | Set by | Used by |
|----------|--------|---------|
| `CODELEWM_HF_RUN_NAME` | launcher | run-name embedded into the manifest |
| `CODELEWM_EXECUTION_PACK_REPO_ID` | launcher | dataset download (entrypoint) |
| `CODELEWM_EXECUTION_PACK_REVISION` | launcher | dataset download (entrypoint) |
| `CODELEWM_TRAIN_SEED` | launcher | seed override |
| `CODELEWM_TRAIN_CONFIG` | launcher | config path |
| `CODELEWM_EXECUTION_PACK_LOCAL_DIR` | image default `/workspace/pack` | runner reads this directory directly |
| `CODELEWM_RUN_OUTPUT_DIR` | image default `/workspace/runs` | the operator-supplied `--out` should write under here |
| `HF_TOKEN` | launcher | required by entrypoint to hit the Hub |

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

### End-to-end with the production runner

To exercise the path that HF Jobs takes (entrypoint pre-downloads the
pack, then runs `codelewm train`), pass `HF_TOKEN` and the pack env
vars:

```bash
docker run --rm \
  -e HF_TOKEN \
  -e CODELEWM_EXECUTION_PACK_REPO_ID=abdelstark/codelewm-execution-pack \
  -e CODELEWM_EXECUTION_PACK_REVISION=v0.6.0 \
  abdelstark/codelewm-runtime:v0.6 \
  uv run codelewm train \
    --config config/train/scaled/codelewm_execution_v0_6_a10g.yaml \
    --seed 42
```

The entrypoint logs `[codelewm-runtime] downloading <repo>@<rev>` and
then hands control to the operator's command. The runner reads
`CODELEWM_EXECUTION_PACK_LOCAL_DIR=/workspace/pack` and skips its
own HF download.

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
