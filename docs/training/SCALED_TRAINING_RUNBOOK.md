# Scaled Training Configs And Runbook

This runbook is the #119 operator contract for moving from smoke evidence to a
bounded scaled training run. The first HF A10G run completed under #138 and is
documented in `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md`. The primary
action-use follow-up completed under #154 and is documented in
`docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`. The #159
margin+retrieval remediation run and #172 v0.2 action-swap/inverse-action run
also completed negative/diagnostic, documented in
`docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md` and
`docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`. These runs prove
the remote systems path but do not support a positive action-conditioned
quality claim because text-action loses to no-action on headline retrieval and
v0.2 action-contrast gates. Verify downloaded artifacts before any positive
public claim.

## Config Matrix

All scaled configs keep the v0.1 one-step contract: `history_size=1`,
`num_preds=1`, `embed_dim=256`, and `action_view=text`. Patch-action remains diagnostic-only.
Training config validation rejects patch-action. The baseline CPU/MPS/A10G
configs keep retrieval and action-use margin disabled; the follow-up A10G sweep
explicitly enables the no-action margin objective. The v0.2 A10G intervention
adds gated residual action fusion plus action-swap and inverse-action
auxiliaries behind explicit config gates.

| Profile | Config | Seed | Steps | Batch | Precision | Budget |
| --- | --- | --- | ---: | ---: | --- | --- |
| CPU rehearsal | `config/train/scaled/codelewm_scaled_cpu.yaml` | `240119` | `2048` | `8` | `float32` | 6-18h on 8-12 CPU cores, 8-12 GiB RAM, 1-3 GiB artifacts |
| Apple MPS development | `config/train/scaled/codelewm_scaled_mps.yaml` | `240119` | `10000` | `32` | `float32` | 4-12h on M2/M3 Max class hardware, 16-32 GiB unified memory, 1-4 GiB artifacts |
| HF A10G baseline | `config/train/scaled/codelewm_scaled_gpu_a10g.yaml` | `240119` | `60000` | `64` | `bf16-mixed` | Prior systems proof from #138; rerun only for regression comparison |
| HF A10G primary action-use | `config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml` | `240119` | `60000` | `64` | `bf16-mixed` | Completed in #154; negative claim gate |
| HF A10G margin+retrieval fallback | `config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml` | `240119` | `60000` | `64` | `bf16-mixed` | Completed in #159; negative claim gate |
| HF A10G v0.2 action-swap+inverse | `config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml` | `240119` | `60000` | `64` | `bf16-mixed` | Completed in #172; negative action-use, representation, and downstream gates |
| HF A10G v0.9 correctness short | `config/train/scaled/codelewm_execution_v0_9_short_a10g.yaml` | `42`, `1729` | `12000` | `64` | `bf16-mixed` | #391 guarded run; launch only after v0.9 pack publication and digest-pinned runtime dry run |

The primary follow-up candidate was
`codelewm_scaled_action_use_margin_gpu_a10g.yaml`. It directly targeted the
observed failure mode by penalizing predictions whose after-state latent error
does not beat the no-action identity baseline by `action_use_margin=0.02`, with
`action_use_margin_weight=0.25`. It still lost to no-action in #154. The
fallback adds the existing retrieval auxiliary at `retrieval_weight=0.05`; it
completed in #159 and still lost to no-action.
The v0.2 intervention uses `action_fusion=gated_residual`,
`action_swap_contrastive_weight=0.20`,
`action_swap_contrastive_margin=0.05`, and
`inverse_action_reconstruction_weight=0.10` in addition to the no-action
margin. It completed in #172 and also failed the action-use gate; keep it as a
comparison baseline, not a job to relaunch.
The CPU and MPS profiles are bounded rehearsal/debug profiles, not headline
research claims.

## Config Validation

Validate every checked-in scaled config before launching local or remote
training:

```bash
uv run scripts/validate-training-configs
```

The command emits `codelewm.train_config_validation.v1` with each config path,
seed, action view, action fusion, hardware profile, batch size, max steps,
retrieval objective settings, action-use margin settings, action-swap and
inverse-action objective settings, and deterministic config SHA-256. A
non-zero exit means the config must be fixed before launch.

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
mix, counted exclusions, and redacted private source paths. The build and pack
artifacts must also include
`reports/action_discriminative_shard_report.json`; the follow-up action-use run
must record whether that report's shard-level claim-readiness gate is true and
which hard-negative pools are available. Do not publish raw restricted data;
publish only the manifest-backed artifacts allowed by the
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

The primary action-use A10G launch command used for #154 was:

```bash
CODELEWM_HF_JOBS_DRY_RUN=0 \
CODELEWM_HF_PIPELINE_MODE=scaled \
CODELEWM_HF_JOBS_FLAVOR=a10g-small \
CODELEWM_HF_JOBS_TIMEOUT=24h \
CODELEWM_HF_PUBLISH_DRY_RUN=0 \
CODELEWM_HF_REF=<merged-sha-or-main> \
CODELEWM_DATASET_BUILD_CONFIG=config/data/codelewm_public_shard_commitpackft_python.json \
CODELEWM_TRAIN_CONFIG=config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml \
CODELEWM_HF_SCORER_QUALITY_CONFIG=config/first_results/scorer_quality.json \
CODELEWM_HF_RETRIEVAL_PRIOR_WEIGHT=1.0 \
CODELEWM_HF_INDEX_BATCH_SIZE=64 \
CODELEWM_HF_SOURCE_DATASET_REPO_ID=bigcode/commitpackft \
CODELEWM_HF_SOURCE_DATASET_REPO_TYPE=dataset \
CODELEWM_HF_SOURCE_DATASET_PATH=data/python/data.jsonl \
CODELEWM_HF_SOURCE_DATASET_REVISION=main \
CODELEWM_HF_SOURCE_LOCAL_DIR=.artifacts/hf-sources/commitpackft \
uv run scripts/hf-launch-codelewm-job
```

The old `config/train/scaled/codelewm_scaled_gpu_a10g.yaml` baseline remains
available only for regression comparison against the #138 systems result. The
primary action-use result and failure mode are recorded in #154 and
`docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`; #159 records the
margin+retrieval escalation with
`config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml`.
#172 records the v0.2 intervention launch with
`config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml`.

The v0.9 #391 execution-substrate run uses
`config/train/scaled/codelewm_execution_v0_9_short_a10g.yaml`. Before launch,
the v0.9 pack must be published to
`abdelstark/codelewm-execution-pack@v0.9.0-rc1`, the runtime image must be built
from `containers/v0_9/Dockerfile`, and the launch plan must pin that image with
`--runtime-image-digest sha256:<digest> --require-runtime-image-digest`:

Pack-build preflight is observable: `scripts/build-passfail-pack` writes
`reports/passfail_pack_progress.jsonl` by default and mirrors
`CODELEWM_JOB_EVENT` progress to stderr. Use
`uv run scripts/hf-job-event-status --from-file <pack-dir>/reports/passfail_pack_progress.jsonl`
to check rows, sandboxed inputs, ETA, reject counts, and completion status
without reading raw JSONL.

```bash
uv run scripts/hf-launch-execution-run \
  --config config/train/scaled/codelewm_execution_v0_9_short_a10g.yaml \
  --git-sha <merged-sha> \
  --date <YYYYMMDD> \
  --runtime-image-digest sha256:<published-image-digest> \
  --require-runtime-image-digest \
  --json
```

Review the emitted per-seed commands before running them. Each plan must include
`CODELEWM_EXECUTION_PACK_REVISION=v0.9.0-rc1`, the config path, seed, digest
pinned image reference, `CODELEWM_UPLOAD_REPO_ID`, and
`CODELEWM_UPLOAD_PATH_IN_REPO`.

Monitor and inspect only through the HF CLI:

```bash
hf jobs ps
hf jobs inspect <job-id>
hf jobs logs <job-id>
hf jobs stats <job-id>
```

For a compact status view, use the CodeLeWM event parser instead of reading the
raw log stream by hand:

```bash
uv run scripts/hf-job-event-status <job-id> [<job-id> ...]
uv run scripts/hf-job-event-status --watch 300 <job-id> [<job-id> ...]
```

For execution-substrate jobs, the runtime container and trainer emit structured
stderr lines with the prefix `CODELEWM_JOB_EVENT `. Filter for that prefix in
`hf jobs logs` to see runtime lifecycle events such as `runtime.start`,
`runtime.pack_download_start`, `runtime.pack_download_complete`,
`runtime.command_start`, `runtime.upload_start`, and `runtime.upload_complete`,
plus training events such as `execution_training.start`,
`execution_training.progress`, `execution_training.collapse_diagnostics`,
`execution_training.checkpoint`, and `execution_training.complete`. The training
stream is also persisted after upload as `reports/job_progress.jsonl` in the run
artifact. The same status parser can summarize a downloaded progress log without
contacting HF:

```bash
uv run scripts/hf-job-event-status --from-file <run-dir>/reports/job_progress.jsonl
```

If the job fails, record the job ID, commit SHA, run ID, config paths, hardware
flavor, timeout, failure phase, and a short log excerpt in the relevant issue
before patching or relaunching.

The transition index builder embeds the train split in bounded batches. Keep
`CODELEWM_HF_INDEX_BATCH_SIZE` explicit for HF Jobs relaunches so the index
phase cannot accidentally embed the full train split as one CUDA batch.

## Post-Run Verification

After a remote run succeeds, download the artifacts with `hf download`
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
