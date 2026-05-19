# Scaled Training Configs And Runbook

This runbook is the #119 operator contract for moving from smoke evidence to a
bounded scaled training run. It does not make a learning claim by itself; the
claim still depends on the ablation, scorer/reranker, card, and HF Jobs evidence
tracked by #120, #121, #122, and #138.

## Config Matrix

All scaled configs keep the v0.1 one-step contract: `history_size=1`,
`num_preds=1`, `embed_dim=256`, `action_view=text`, and retrieval loss disabled.
Patch-action remains diagnostic-only and is rejected by training config
validation.

| Profile | Config | Seed | Steps | Batch | Precision | Budget |
| --- | --- | --- | ---: | ---: | --- | --- |
| CPU rehearsal | `config/train/scaled/codelewm_scaled_cpu.yaml` | `240119` | `2048` | `8` | `float32` | 6-18h on 8-12 CPU cores, 8-12 GiB RAM, 1-3 GiB artifacts |
| Apple MPS development | `config/train/scaled/codelewm_scaled_mps.yaml` | `240119` | `10000` | `32` | `float32` | 4-12h on M2/M3 Max class hardware, 16-32 GiB unified memory, 1-4 GiB artifacts |
| HF A10G headline | `config/train/scaled/codelewm_scaled_gpu_a10g.yaml` | `240119` | `60000` | `64` | `bf16-mixed` | 12-24h on `a10g-small`, <=24 GiB device memory, 2-8 GiB artifacts |

The A10G profile is the default candidate for the first meaningful HF Jobs run.
The CPU and MPS profiles are bounded rehearsal/debug profiles, not headline
research claims.

## Config Validation

Validate every checked-in scaled config before launching local or remote
training:

```bash
uv run scripts/validate-training-configs
```

The command emits `codelewm.train_config_validation.v1` with each config path,
seed, action view, hardware profile, batch size, max steps, and deterministic
config SHA-256. A non-zero exit means the config must be fixed before launch.

## Public Shard Preconditions

The first checked-in public shard build config is:

```text
config/data/codelewm_public_shard_commitpackft_python.json
```

It consumes the Python shard from `bigcode/commitpackft` after the HF CLI
downloads it to:

```text
.artifacts/hf-sources/commitpackft/data/python/data.jsonl
```

Preflight the source path:

```bash
hf download bigcode/commitpackft \
  data/python/data.jsonl \
  --repo-type dataset \
  --local-dir .artifacts/hf-sources/commitpackft \
  --dry-run
```

The scaled configs expect the packed public-safe dataset under:

```text
data/codelewm_public_shard_v0_3/
  manifest.json
  hdf5/train.hdf5
  hdf5/val.hdf5
```

Before using that path, build and pack the shard through `codelewm dataset
build` and `codelewm dataset pack`. The build artifact must include
`reports/source_acquisition_report.json` with `release_allowed=true`, a source
mix, counted exclusions, and redacted private source paths. Do not publish raw
restricted data; publish only the manifest-backed artifacts allowed by the
license gate.

## Bounded Local Rehearsal

Use a tiny packed fixture to exercise the scaled CPU config shape without
spending the full CPU budget:

```bash
rm -rf .artifacts/scaled-rehearsal

uv run codelewm dataset build \
  --config config/first_results/dataset_build.json \
  --out .artifacts/scaled-rehearsal/build \
  --json

uv run codelewm dataset pack \
  --manifest .artifacts/scaled-rehearsal/build/manifest.json \
  --out .artifacts/scaled-rehearsal/pack \
  --json

uv run python - <<'PY'
import json
from pathlib import Path
from codelewm.training import load_train_config

root = Path(".artifacts/scaled-rehearsal")
payload = load_train_config("config/train/scaled/codelewm_scaled_cpu.yaml").to_dict()
payload["name"] = "scaled_cpu_rehearsal"
payload["trainer"]["max_steps"] = 1
payload["loader"]["batch_size"] = 2
payload["loader"]["shuffle"] = False
payload["data"] = {
    "manifest": str(root / "pack" / "manifest.json"),
    "train": str(root / "pack" / "hdf5" / "train.hdf5"),
    "val": str(root / "pack" / "hdf5" / "val.hdf5"),
}
payload["output"] = {
    "run_dir": str(root / "train"),
    "checkpoint_dir": str(root / "train" / "checkpoints"),
    "metrics_path": str(root / "train" / "metrics.jsonl"),
    "manifest_path": str(root / "train" / "training_manifest.json"),
}
target = root / "configs" / "scaled_cpu_rehearsal.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

uv run codelewm train \
  --config .artifacts/scaled-rehearsal/configs/scaled_cpu_rehearsal.json \
  --executor torch \
  --device cpu \
  --overwrite \
  --json \
  --log-jsonl .artifacts/scaled-rehearsal/logs/train.jsonl

uv run codelewm manifest verify \
  --manifest .artifacts/scaled-rehearsal/train/manifest.json \
  --parent-manifest .artifacts/scaled-rehearsal/pack/manifest.json \
  --json
```

This rehearsal proves config parsing, seed/config-hash recording, checkpoint
manifest creation, and artifact verification. It is not a scaled-result claim.

## Resume Policy

Resume into a fresh run directory with the same architecture and loss surface:

```bash
uv run codelewm train \
  --config <runtime-train-config.json> \
  --out <new-run-dir> \
  --executor torch \
  --device <cpu|mps|cuda|auto> \
  --resume-from <previous-run-dir>/training_manifest.json \
  --json
```

Resume validation checks the parent training manifest, parent artifact manifest,
checkpoint manifest, checkpoint SHA-256, action view, latent dimension, and
compatibility config hash before writing new outputs. If validation fails,
preserve the parent run ID, child config path, and error JSON; do not edit or
reuse the failed output directory.

## HF Jobs Launch

Use the A10G config for the real remote candidate once the public shard config is
merged and the remaining evidence gates allow spending GPU compute:

```bash
CODELEWM_HF_JOBS_DRY_RUN=0 \
CODELEWM_HF_PIPELINE_MODE=scaled \
CODELEWM_HF_JOBS_FLAVOR=a10g-small \
CODELEWM_HF_JOBS_TIMEOUT=24h \
CODELEWM_HF_PUBLISH_DRY_RUN=0 \
CODELEWM_HF_REF=<merged-sha-or-main> \
CODELEWM_DATASET_BUILD_CONFIG=config/data/codelewm_public_shard_commitpackft_python.json \
CODELEWM_TRAIN_CONFIG=config/train/scaled/codelewm_scaled_gpu_a10g.yaml \
CODELEWM_HF_SCORER_QUALITY_CONFIG=config/first_results/scorer_quality.json \
CODELEWM_HF_RETRIEVAL_PRIOR_WEIGHT=1.0 \
CODELEWM_HF_SOURCE_DATASET_REPO_ID=bigcode/commitpackft \
CODELEWM_HF_SOURCE_DATASET_REPO_TYPE=dataset \
CODELEWM_HF_SOURCE_DATASET_PATH=data/python/data.jsonl \
CODELEWM_HF_SOURCE_DATASET_REVISION=main \
CODELEWM_HF_SOURCE_LOCAL_DIR=.artifacts/hf-sources/commitpackft \
uv run scripts/hf-launch-codelewm-job
```

Monitor and inspect only through the HF CLI:

```bash
hf jobs ps
hf jobs inspect <job-id>
hf jobs logs <job-id>
hf jobs stats <job-id>
```

If the job fails, record the job ID, commit SHA, run ID, config paths, hardware
flavor, timeout, failure phase, and a short log excerpt in the relevant issue
before patching or relaunching.

## Post-Run Verification

After a remote run succeeds, download the private artifacts with `hf download`
and verify manifests before writing cards or benchmark prose:

```bash
hf download "$CODELEWM_HF_RESULTS_REPO_ID" \
  --repo-type dataset \
  --local-dir .artifacts/hf-download/results

hf download "$CODELEWM_HF_MODEL_REPO_ID" \
  "checkpoints/<run-id>" \
  --repo-type model \
  --local-dir .artifacts/hf-download/model

uv run codelewm manifest verify \
  --manifest .artifacts/hf-download/results/runs/<run-id>/pack/manifest.json \
  --json
```

Only artifact-backed numbers from downloaded, verified outputs can be used in
`docs/benchmark/`, dataset cards, or model cards.
