# Hugging Face Jobs And ml-intern Training Runbook

This runbook is the operator path for moving CodeLeWM from the local
first-results smoke loop to a Hugging Face Hub-backed training, publication, and
evaluation run. It is designed to be usable by a human operator or by
`ml-intern` in headless mode.

Current status: the smoke path is ready for dry-run validation. The scaled path
uses the source-acquisition gate from #118 and the training config/runbook
contract in `docs/training/SCALED_TRAINING_RUNBOOK.md`. It must still wait for
the remaining ablation, quality-report, and card gates before publishing a
public claim. The full remote execution and publication run is tracked by #138.

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
CODELEWM_HF_PRIVATE=1
```

Keep those repositories private until the public source/license gate in #118 and
the card population work in #122 are complete.

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
```

Scaled mode additionally requires checked-in configs:

```bash
CODELEWM_HF_PIPELINE_MODE=scaled
CODELEWM_DATASET_BUILD_CONFIG=config/<public-shard-build>.json
CODELEWM_TRAIN_CONFIG=config/train/scaled/codelewm_scaled_gpu_a10g.yaml
CODELEWM_HF_SCORER_QUALITY_CONFIG=config/first_results/scorer_quality.json
CODELEWM_HF_RETRIEVAL_PRIOR_WEIGHT=1.0
CODELEWM_HF_RETRIEVAL_PRIOR_K=10
```

## Scripts

`scripts/hf-launch-codelewm-job` builds the `hf jobs run` command. It loads
`.env` as defaults, redacts nothing by printing no secret values, and defaults
to dry-run.

`scripts/hf-run-codelewm-pipeline` runs inside the job container. It supports:

- `smoke`: run `scripts/first-results` into `.artifacts/hf/<run-id>`;
- `scaled`: build the dataset, pack it, train, run retrieval evaluation, build
  the action-view ablation report, run surprise evaluation, build the transition
  index, run the scorer/reranker quality report with retrieval-prior settings,
  verify manifests, and scan the run root for secrets.

`scripts/hf-publish-codelewm-artifacts` publishes the resulting directories:

- packed dataset artifacts to the dataset repository under
  `runs/<run-id>/pack`;
- training artifacts and checkpoint to the model repository under
  `checkpoints/<run-id>`;
- the full evidence bundle to the results dataset under `runs/<run-id>`.

The publisher emits `codelewm.hf_publish_plan.v1` for both dry-run and real
publication.

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

Run this only after #118 and #119 have landed, the public shard config exists on
the published ref, and the remaining issue gates allow spending GPU compute.

```bash
CODELEWM_HF_JOBS_DRY_RUN=0 \
CODELEWM_HF_PIPELINE_MODE=scaled \
CODELEWM_HF_JOBS_FLAVOR=a10g-small \
CODELEWM_HF_JOBS_TIMEOUT=24h \
CODELEWM_HF_PUBLISH_DRY_RUN=0 \
CODELEWM_HF_REF=<merged-sha-or-main> \
CODELEWM_DATASET_BUILD_CONFIG=config/<public-shard-build>.json \
CODELEWM_TRAIN_CONFIG=config/train/scaled/codelewm_scaled_gpu_a10g.yaml \
CODELEWM_HF_SCORER_QUALITY_CONFIG=config/first_results/scorer_quality.json \
CODELEWM_HF_RETRIEVAL_PRIOR_WEIGHT=1.0 \
uv run scripts/hf-launch-codelewm-job
```

Definition of done for the scaled run:

- the HF job exits successfully;
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

Use the existing ordered backlog as the gate for a meaningful remote run:

- #118 must produce a public source acquisition report and license gate.
- #119 must add scaled dataset/training configs and update this runbook if the
  command shape changes.
- #120 must run the action-view ablation suite.
- #121 must produce scorer calibration and reranker quality evidence.
- #122 must fill dataset and model cards from the actual artifacts.
- #123 through #126 remain release gates after the model/data artifacts exist.
- #138 tracks the actual HF Jobs scaled run, private publication, download
  verification, inference smoke, and final evidence update.

Do not flip repositories public, update public claims, or mark a result as
meaningful until those gates are satisfied by artifact-backed evidence.
