# Hugging Face Jobs And ml-intern Training Runbook

This runbook is the operator path for moving CodeLeWM from the local
first-results smoke loop to a Hugging Face Hub-backed training, publication, and
evaluation run. It is designed to be usable by a human operator or by
`ml-intern` in headless mode.

Current status: the smoke path, scaled HF Jobs path, public publication path,
and downloaded-artifact verification path are implemented. The first scaled run
is documented in `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md`; the primary
action-use follow-up is documented in
`docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`; the second-stage
margin+retrieval remediation run is documented in
`docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md`. All three are
useful systems evidence, but none is a positive action-conditioning claim
because the no-action baseline beats text-action on headline retrieval. Issue
#126 froze the private diagnostic release boundary, and #159 confirms that the
current completion boundary remains negative/diagnostic. Training config details
remain in `docs/training/SCALED_TRAINING_RUNBOOK.md`.

## Upstream Contract

`ml-intern` supports a single headless prompt and `--max-iterations`, so the
project prompt lives in `docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md`.

Hugging Face Jobs can run arbitrary container commands through the `hf jobs run`
CLI. The launcher uses:

- `--flavor` for hardware;
- `--timeout` for long training runs;
- `--secrets HF_TOKEN` so the token reaches the job without appearing in logs;
- repeated `--env KEY=value` entries for non-secret CodeLeWM settings;
- `--detach` by default so the job ID can be monitored separately.

Do not rely on unsupported local `ml-intern` flags. The locally validated CLI
surface is:

```bash
ml-intern --max-iterations -1 "$(cat docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md)"
```

## Environment

Copy `.env.example` to `.env` and fill real local values. `.env` is ignored by
git. Shell exports always win over values in `.env`, which lets operators keep a
safe default file and override a single run at launch time.

Required secret:

```bash
HF_TOKEN=<token with repository create/upload access>
```

Required repository targets:

```bash
CODELEWM_HF_DATASET_REPO_ID=abdelstark/codelewm-public-shard
CODELEWM_HF_MODEL_REPO_ID=abdelstark/codelewm-transition-model
CODELEWM_HF_RESULTS_REPO_ID=abdelstark/codelewm-runs
CODELEWM_HF_PRIVATE=0
```

The CodeLeWM dataset/model/results repositories are public diagnostic artifact
repositories. Public publication is allowed after the public source/license
gate, card evidence, manifest verification, secret scan, and checkpoint-trust
checks pass. Public visibility does not permit a positive model-quality claim
unless the relevant benchmark gate passes.

Job defaults:

```bash
CODELEWM_HF_GITHUB_REPO=https://github.com/AbdelStark/CodeLeWM.git
CODELEWM_HF_REF=main
CODELEWM_HF_JOBS_DRY_RUN=1
CODELEWM_HF_JOBS_FLAVOR=a10g-small
CODELEWM_HF_JOBS_TIMEOUT=6h
CODELEWM_HF_JOBS_DETACH=1
CODELEWM_HF_JOB_IMAGE=python:3.13-bookworm
CODELEWM_HF_PIPELINE_MODE=smoke
CODELEWM_HF_PUBLISH=1
CODELEWM_HF_PUBLISH_DRY_RUN=1
CODELEWM_HF_PRIVATE=1
```

Scaled mode additionally requires checked-in configs:

```bash
CODELEWM_HF_PIPELINE_MODE=scaled
CODELEWM_DATASET_BUILD_CONFIG=config/data/codelewm_public_shard_commitpackft_python.json
CODELEWM_TRAIN_CONFIG=config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml
CODELEWM_HF_SCORER_QUALITY_CONFIG=config/first_results/scorer_quality.json
CODELEWM_HF_RETRIEVAL_PRIOR_WEIGHT=1.0
CODELEWM_HF_RETRIEVAL_PRIOR_K=10
CODELEWM_HF_INDEX_DEVICE=auto
CODELEWM_HF_INDEX_BATCH_SIZE=64
```

The first public shard candidate is the Python subset of
`bigcode/commitpackft`. The HF job downloads it with the HF CLI before
`codelewm dataset build`:

```bash
CODELEWM_HF_SOURCE_DATASET_REPO_ID=bigcode/commitpackft
CODELEWM_HF_SOURCE_DATASET_REPO_TYPE=dataset
CODELEWM_HF_SOURCE_DATASET_PATH=data/python/data.jsonl
CODELEWM_HF_SOURCE_DATASET_REVISION=main
CODELEWM_HF_SOURCE_LOCAL_DIR=.artifacts/hf-sources/commitpackft
```

Preflight the source path without spending GPU compute:

```bash
hf download bigcode/commitpackft \
  data/python/data.jsonl \
  --repo-type dataset \
  --local-dir .artifacts/hf-sources/commitpackft \
  --dry-run
```

The build and pack stages now emit
`reports/action_discriminative_shard_report.json` with
`schema_version=codelewm.data.action_discriminative_shard_report.v1`. Treat that
report as the data preflight for the follow-up action-use run: record
`claim_readiness.positive_action_use_claim_ready`, unavailable hard-negative
pools, and same-file/near-before/action-cluster pair counts before launching a
real GPU job.

## Scripts

`scripts/hf-launch-codelewm-job` builds the `hf jobs run` command. It loads
`.env` as defaults, redacts nothing by printing no secret values, and defaults
to dry-run.

`scripts/hf-run-codelewm-pipeline` runs inside the job container. It supports:

- `smoke`: run `scripts/first-results` into `.artifacts/hf/<run-id>`;
- `scaled`: optionally download the configured HF source shard with
  `hf download`, build the dataset, pack it, train, run retrieval evaluation,
  build the action-view ablation report, run surprise evaluation, build the
  transition index with bounded batches, run the scorer/reranker quality report
  with retrieval-prior settings, verify manifests, and scan the run root for
  secrets.

`scripts/hf-publish-codelewm-artifacts` publishes the resulting directories:

- packed dataset artifacts to the dataset repository under
  `runs/<run-id>/pack`;
- training artifacts and checkpoint to the model repository under
  `checkpoints/<run-id>`;
- the full evidence bundle to the results dataset under `runs/<run-id>`.

The publisher emits `codelewm.hf_publish_plan.v1` for both dry-run and real
publication.

`scripts/hf-verify-codelewm-run` is the local post-publication gate. It loads
project defaults from `.env`, downloads the published results/model/dataset
artifacts with `hf download`, verifies the manifest chain, reruns retrieval,
latent-probe, ablation, surprise, scorer-quality, score, and rerank checks from
the downloaded checkpoint and index, and runs `codelewm secret-scan` over the
download root. Use `--dry-run --json` before final artifacts exist to inspect
the exact command plan without printing secrets.

## Local Dry Runs

Validate the publisher against an existing first-results artifact:

```bash
uv run scripts/hf-publish-codelewm-artifacts \
  --artifact-root .artifacts/first-results \
  --run-id local-first-results \
  --dry-run \
  --json
```

Validate the launcher command without starting a remote job:

```bash
CODELEWM_HF_JOBS_DRY_RUN=1 uv run scripts/hf-launch-codelewm-job
```

Validate the in-job pipeline locally with publish dry-run:

```bash
CODELEWM_HF_PIPELINE_MODE=smoke \
CODELEWM_HF_RUN_ID=local-smoke \
CODELEWM_HF_OUTPUT_ROOT=.artifacts/hf-local \
CODELEWM_HF_PUBLISH=1 \
CODELEWM_HF_PUBLISH_DRY_RUN=1 \
uv run scripts/hf-run-codelewm-pipeline
```

## Remote Smoke Job

Use this before the scaled run if the HF Jobs environment or repository token was
changed. It spends only the smoke workflow budget and publishes only when
`CODELEWM_HF_PUBLISH_DRY_RUN=0` is explicitly set.

```bash
CODELEWM_HF_JOBS_DRY_RUN=0 \
CODELEWM_HF_PIPELINE_MODE=smoke \
CODELEWM_HF_PUBLISH_DRY_RUN=1 \
uv run scripts/hf-launch-codelewm-job
```

Monitor the job:

```bash
hf jobs ps
hf jobs inspect <job-id>
hf jobs logs <job-id>
hf jobs stats <job-id>
```

## Remote Scaled Training And Publication

The command below is the primary #154 follow-up profile already executed in run
`codelewm-action-use-20260520-6650183`. It used the #153 no-action margin
objective to target the observed failure mode from the first scaled run, but the
claim gate remained negative. The #159 remediation job is
`codelewm-action-use-retrieval-20260520-7895d18` / job
`6a0da3a08229e585f969c3f7`, launched from
`7895d185e165a917af0956a313d8948c04b33638` with
`config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml`.
It completed, published HF artifacts, and was downloaded and verified
locally. The claim gate remained negative: text-action Recall@1 `0.597` and MRR
`0.674500` versus no-action Recall@1 `0.650` and MRR `0.708037`. Do not
relaunch it as release cleanup. The original
`config/train/scaled/codelewm_scaled_gpu_a10g.yaml` baseline remains available
only for regression comparison against #138. The v0.2 #170 intervention config
is
`config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml`;
it adds gated residual action fusion, action-swap contrastive loss, and
inverse-action reconstruction for the #172 action-contrast sweep.

If a future research issue justifies a new GPU run and the recorded decision
does not require another code or config repair after #170 lands, use this v0.2
template as a starting point with a new run ID and issue:

```bash
CODELEWM_HF_JOBS_DRY_RUN=0 \
CODELEWM_HF_PIPELINE_MODE=scaled \
CODELEWM_HF_JOBS_FLAVOR=a10g-small \
CODELEWM_HF_JOBS_TIMEOUT=24h \
CODELEWM_HF_PUBLISH_DRY_RUN=0 \
CODELEWM_HF_PRIVATE=0 \
CODELEWM_HF_RUN_ID=<new-run-id> \
CODELEWM_HF_REF=<merged-sha-or-main> \
CODELEWM_DATASET_BUILD_CONFIG=config/data/codelewm_public_shard_commitpackft_python.json \
CODELEWM_TRAIN_CONFIG=config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml \
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

Definition of done for any further scaled run:

- the HF job exits successfully;
- the source prefetch logs show the expected `hf download` path and the dataset
  build source-acquisition report names `bigcode-commitpackft-python`;
- dataset, model, and results repositories contain the expected `run-id`;
- every published manifest verifies locally after download;
- retrieval includes headline baselines and action-view ablations;
- the ablation report records missing variants as explicit blocked rows;
- surprise evaluation includes enough decoy coverage for a useful result;
- scorer/reranker quality report records ranking metrics, calibration slices,
  parse/patch failure counts, retrieval-prior settings, and non-execution
  policy evidence;
- `codelewm score` and `codelewm rerank` run from the downloaded checkpoint;
- dataset and model cards are filled from the artifacts before any public flip.

## Download And Post-Run Verification

Prefer the scripted verifier so the `hf download` paths, manifest parent chain,
local eval reruns, latent-probe report, score/rerank smokes, and secret scan
stay consistent:

```bash
CODELEWM_HF_RUN_ID=<run-id> \
uv run scripts/hf-verify-codelewm-run --dry-run --json

CODELEWM_HF_RUN_ID=<run-id> \
uv run scripts/hf-verify-codelewm-run --json
```

The equivalent manual core is below.

Download the evidence bundle:

```bash
hf download "$CODELEWM_HF_RESULTS_REPO_ID" \
  --repo-type dataset \
  --local-dir ".artifacts/hf-download/results"
```

Download the model artifacts:

```bash
hf download "$CODELEWM_HF_MODEL_REPO_ID" \
  "checkpoints/<run-id>" \
  --repo-type model \
  --local-dir ".artifacts/hf-download/model"
```

Verify manifests and run local inference/evaluation checks against the
downloaded checkpoint:

```bash
uv run codelewm manifest verify \
  --manifest .artifacts/hf-download/results/runs/<run-id>/pack/manifest.json \
  --json

uv run codelewm eval retrieval \
  --checkpoint .artifacts/hf-download/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --data .artifacts/hf-download/results/runs/<run-id>/pack \
  --out .artifacts/hf-download/retrieval \
  --overwrite \
  --json

uv run codelewm eval surprise \
  --checkpoint .artifacts/hf-download/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --data .artifacts/hf-download/results/runs/<run-id>/pack \
  --out .artifacts/hf-download/surprise \
  --overwrite \
  --json

uv run codelewm eval scorer-quality \
  --config config/first_results/scorer_quality.json \
  --checkpoint .artifacts/hf-download/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --out .artifacts/hf-download/scorer-quality \
  --index .artifacts/hf-download/results/runs/<run-id>/index \
  --retrieval-prior-weight 1.0 \
  --parent-manifest .artifacts/hf-download/results/runs/<run-id>/train/manifest.json \
  --parent-manifest .artifacts/hf-download/results/runs/<run-id>/index/manifest.json \
  --overwrite \
  --json
```

For a scorer smoke check, use a real before/instruction/candidate triple from
the downloaded result bundle or a checked-in fixture with the same action view:

```bash
uv run codelewm score \
  --before tests/fixtures/codestate/class_method_before.py \
  --instruction "add an input guard" \
  --candidate tests/fixtures/codestate/class_method_after.py \
  --checkpoint .artifacts/hf-download/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --json
```

## Issue Gate

Completed gates that must be preserved:

- #118 produced the public source acquisition report and license gate.
- #119 added scaled dataset/training configs and this runbook.
- #120 added the action-view ablation suite.
- #121 added scorer calibration and reranker quality evidence.
- #122 filled dataset and model cards from real artifacts.
- #138 executed the first scaled HF Jobs run, artifact publication, download
  verification, inference, and eval loop.

Completed action-use gates:

- #151 has added no-action dominance diagnostics and a claim gate.
- #152 has added action-discriminative shard diagnostics and hard negatives.
- #153 has added the action-use margin objective and scaled sweep configs.
- #170 adds the v0.2 action-swap/inverse-action training intervention config
  `config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml`
  for the next action-contrast sweep.
- #154 executed the follow-up HF Jobs run with
  `config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml`,
  verified downloaded artifacts, and recorded a negative claim gate.
- #126 froze the private diagnostic release boundary without enabling public
  positive claims.
- #159 executed the second-stage action-use remediation sweep as
  `codelewm-action-use-retrieval-20260520-7895d18` / job
  `6a0da3a08229e585f969c3f7`; HF monitoring, download, manifest verification,
  local eval/inference checks, docs/cards, and claim gate closure are complete.

Do not update public positive claims or mark a result as a positive
action-conditioning result until those gates are satisfied by artifact-backed
evidence. The current follow-up still loses to no-action, so public artifacts
must be framed explicitly as negative/diagnostic.
