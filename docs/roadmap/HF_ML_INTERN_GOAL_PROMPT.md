# HF ml-intern `/goal` Prompt

Use this prompt from the repository root. It is written for a headless
`ml-intern` run that must close the remaining action-use gap and orchestrate the
remote training/publication loop through the Hugging Face CLI.

```text
/goal Complete CodeLeWM's remaining path to a claim-eligible first scaled
training result and release-ready artifact set. Use Hugging Face Hub
infrastructure for remote execution, private artifact publication, download
verification, inference, and evaluation.

You are operating in the CodeLeWM repository. The current gap is not smoke
infrastructure and not first HF Jobs automation. Issues #118, #119, #120, #121,
#122, #137, and #138 are closed. The completed scaled run
`codelewm-scaled-20260520-9699b53` / job `6a0d43c92dc5b1243da50bba` published
private artifacts and verified them after `hf download`, but it is not a
positive action-conditioning result because text-action lost to no-action on the
headline retrieval metrics.

The remaining gap is action-use evidence from remote artifacts: diagnostics and
claim gates around no-action dominance are implemented, the
action-discriminative shard/hard-negative support is implemented, and the
action-use margin training configs are implemented. The next step is a follow-up
HF Jobs run from a merged SHA, downloaded-artifact inference/evals, then release
packaging/provenance/docs/freeze gates.

Start by grounding in the current repo and issue tracker. Do not assume this
prompt's issue status is current. Run:

- `git status --short`
- `git branch --show-current`
- `git fetch origin`
- `gh issue view 150`
- `gh issue view 151`
- `gh issue view 152`
- `gh issue view 153`
- `gh issue view 154`
- `gh issue view 123`
- `gh issue view 124`
- `gh issue view 125`
- `gh issue view 126`
- `gh issue view 138`
- `gh pr list --state open --search "repo:AbdelStark/CodeLeWM"`

Before editing or launching compute, read:

- AGENTS.md
- SPEC.md
- CONTRIBUTING.md
- docs/roadmap/FULL_COMPLETION.md
- docs/roadmap/IMPLEMENTATION.md
- docs/roadmap/NEXT_GOAL_PROMPT.md
- docs/operations/HF_ML_INTERN_TRAINING.md
- docs/training/SCALED_TRAINING_RUNBOOK.md
- docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md
- docs/spec/05-observability.md
- docs/spec/06-security.md
- docs/spec/07-testing-strategy.md
- docs/spec/09-release-and-versioning.md
- docs/rfcs/RFC-0002-edit-transition-dataset.md
- docs/rfcs/RFC-0005-model-objective-and-collapse-diagnostics.md
- docs/rfcs/RFC-0006-training-runtime-and-configs.md
- docs/rfcs/RFC-0007-retrieval-and-surprise-evaluation.md
- docs/rfcs/RFC-0008-agent-harness-scorer-reranker.md
- docs/rfcs/RFC-0010-security-licensing-trust-boundaries.md
- docs/rfcs/RFC-0012-release-ci-and-governance.md

Security rule: Do not print, commit, paste, or summarize token values. Do not
log token values either. Treat `.env` as local secret state. Load it only as
defaults. Prefer shell overrides for one run. Do not disable secret scans,
license gates, checkpoint trust gates, manifest verification, or
private-publication defaults.

Primary execution rule: orchestrate Hugging Face work with the `hf` CLI. Use the
checked-in scripts where they encode CodeLeWM policy, but the remote job
lifecycle, monitoring, download, and verification boundary must be explicit HF
CLI commands:

- `hf auth whoami`
- `hf jobs run`
- `hf jobs ps`
- `hf jobs inspect <job-id>`
- `hf jobs logs <job-id>`
- `hf jobs stats <job-id>`
- `hf download`

Use `hf jobs uv run` only for small ad hoc HF-hosted probes. Use
`scripts/hf-launch-codelewm-job` for the project pipeline because it builds the
policy-compliant `hf jobs run` command with `--secrets HF_TOKEN`, `--env`,
`--label`, `--timeout`, `--flavor`, and `--detach`.

Do not treat local-only execution as completion. Local runs are preflight only.

Work in this order, skipping only issues already closed by merged PRs and
verifying their artifacts before moving on:

1. #154: run follow-up HF Jobs action-use training, private publication,
   download verification, inference, and evals.
2. #123: release: add uv build and package publishing gates.
3. #124: release: add dependency audit and provenance evidence.
4. #125: docs: refresh public docs against first-results evidence.
5. #126: release: run final artifact freeze and checklist.

Historical closed gates to preserve, not redo: #118 source acquisition, #119
scaled configs/runbook, #120 action-view ablation, #121 scorer quality, #122
artifact cards, #138 first scaled HF execution, #151 no-action claim gates, #152
action-discriminative diagnostics, and #153 action-use margin training configs.

For each unfinished issue, use one branch and one PR. Re-read the issue body and
linked spec/RFC, inspect current code before editing, implement the smallest
complete change, add focused tests/docs, run the strongest relevant validation,
commit, push, open the PR, wait for checks when available, merge after passing,
return to main, pull latest main, and continue. Do not mix unrelated issue work
in the same PR.

Before spending GPU compute, verify the HF CLI and dry-run infrastructure:

hf auth whoami

CODELEWM_HF_JOBS_DRY_RUN=1 uv run scripts/hf-launch-codelewm-job

CODELEWM_HF_PIPELINE_MODE=smoke \
CODELEWM_HF_RUN_ID=local-smoke \
CODELEWM_HF_OUTPUT_ROOT=.artifacts/hf-local \
CODELEWM_HF_PUBLISH=1 \
CODELEWM_HF_PUBLISH_DRY_RUN=1 \
uv run scripts/hf-run-codelewm-pipeline

The dry-run launcher must show an `hf jobs run` command. Confirm it includes
`--secrets HF_TOKEN`, project labels, the selected flavor, timeout, detached
mode, repo/ref env vars, dataset/model/results repo env vars when configured,
and the chosen pipeline mode.

Do not launch the real follow-up scaled job until #152 and #153 are closed by
merged PRs on `main`. If the issue tracker disagrees with this prompt, trust the
tracker and update the roadmap before spending compute.

For the real follow-up HF Jobs run, use a merged SHA or `main` for
`CODELEWM_HF_REF`, checked-in dataset and training configs, private publishing,
HF CLI source prefetch, and a detached job. The current public shard source is
`config/data/codelewm_public_shard_commitpackft_python.json`, which expects
`bigcode/commitpackft:data/python/data.jsonl` under
`.artifacts/hf-sources/commitpackft`. Preflight the source path first:

hf download bigcode/commitpackft \
  data/python/data.jsonl \
  --repo-type dataset \
  --local-dir .artifacts/hf-sources/commitpackft \
  --dry-run

Start the run through the launcher with the primary action-use margin config
from #153:

CODELEWM_HF_JOBS_DRY_RUN=0 \
CODELEWM_HF_PIPELINE_MODE=scaled \
CODELEWM_HF_JOBS_FLAVOR=a10g-small \
CODELEWM_HF_JOBS_TIMEOUT=24h \
CODELEWM_HF_PUBLISH=1 \
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

Capture the job ID from the detached launch. Monitor and triage only with HF
CLI commands:

hf jobs ps
hf jobs inspect <job-id>
hf jobs logs <job-id>
hf jobs stats <job-id>

Record the job ID, commit SHA, run ID, config paths, hardware flavor, timeout,
repository targets, and command line in #154. If the job fails, preserve the job
ID, log excerpt, failure phase, and config paths in the relevant issue before
patching. Do not relaunch blindly.

After the job succeeds, use `hf download` to fetch the published private
artifacts into a clean local artifact directory. Do not validate against the
job's working directory. Validate the downloaded artifacts:

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

uv run codelewm manifest verify \
  --manifest .artifacts/hf-download/results/runs/<run-id>/train/manifest.json \
  --parent-manifest .artifacts/hf-download/results/runs/<run-id>/pack/manifest.json \
  --json

uv run codelewm eval retrieval \
  --checkpoint .artifacts/hf-download/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --data .artifacts/hf-download/results/runs/<run-id>/pack \
  --out .artifacts/hf-download/retrieval \
  --overwrite \
  --json

uv run codelewm eval ablation \
  --retrieval-artifact .artifacts/hf-download/retrieval/manifest.json \
  --training-artifact .artifacts/hf-download/results/runs/<run-id>/train/manifest.json \
  --out .artifacts/hf-download/ablation \
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

Then run scorer and reranker smoke checks from the downloaded checkpoint and, if
the transition index is published, from the downloaded index. Use real examples
from the downloaded result bundle when available; otherwise use checked-in
fixtures and label them as fixture checks.

Update benchmark docs, dataset card, model card, and release evidence only with
artifact-backed numbers. Every claim must name the command, data source, commit
SHA, run ID, job ID when applicable, and artifact path that produced it.

Do not call the result meaningful or claim-eligible unless all of these are
true:

- the action-use claim gate from #151 passes;
- text-action improves over no-action on the agreed headline metrics;
- text-action still improves over random, shuffled-action, and lexical
  baselines;
- action-view ablation includes completed baseline rows and explicit blocked
  rows for unavailable variants;
- collapse gates pass;
- surprise/decoy coverage is adequate for the claim being made;
- scorer calibration and reranker quality reports are present;
- license, source-acquisition, secret-scan, and checkpoint-trust gates pass;
- the published private artifacts can be downloaded with `hf download` and
  verified locally from a clean checkout.

Keep Hugging Face repositories private until #126 is complete and the public
visibility gate is explicitly satisfied. If any prerequisite is missing or the
result is negative, stop the public-claim path, keep repositories private, and
open or update an evidence-backed GitHub issue with the exact blocker, job ID
when applicable, and next validation command.

Completion criteria:

- #151, #152, #153, #154, #123, #124, #125, and #126 are closed by merged PRs,
  or any remaining one is updated with an explicit blocker that prevents
  completion.
- At least one follow-up scaled HF Jobs run has a recorded job ID, commit SHA,
  run ID, config paths, hardware flavor, timeout, and repo targets.
- Dataset, model, and results artifacts are published privately to the
  configured Hugging Face repositories.
- Published artifacts are downloaded with `hf download` and verified locally.
- Retrieval, action-view ablation, surprise, score, and rerank checks run from
  the downloaded checkpoint/artifacts.
- Dataset/model cards and benchmark docs reflect only verified artifacts.
- The final response reports the HF job ID, published repo paths, validation
  commands, metrics summary, caveats, and any remaining release blockers.
```

Launch command:

```bash
ml-intern --max-iterations -1 "$(cat docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md)"
```
