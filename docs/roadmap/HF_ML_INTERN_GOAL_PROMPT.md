# HF ml-intern `/goal` Prompt

Use this prompt from the repository root. It is written for a headless
`ml-intern` run that must either close the remaining action-use gap with a
second-stage HF Jobs run or complete the release as explicitly negative
diagnostic evidence.

```text
/goal Complete CodeLeWM's remaining path to a release-ready first scaled
artifact set, with a positive action-conditioned claim only if the evidence
actually passes the claim gate.

You are operating in the CodeLeWM repository. The project already has a working
package runtime, HF Jobs automation, private Hugging Face publication, and
downloaded-artifact verification.

Do not redo closed infrastructure work. The completed scaled systems run is
`codelewm-scaled-20260520-9699b53` / job `6a0d43c92dc5b1243da50bba`. The
completed primary action-use follow-up is
`codelewm-action-use-20260520-6650183` / job
`6a0d7a763aba298b21d147a9`, source `6650183`, train config
`config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml`.

The current gap is precise: #154 executed and verified the primary no-action
margin recipe, but the action-use claim gate is still negative. Text-action
Recall@1 is `0.363` and MRR is `0.467875`; no-action Recall@1 is `0.469` and
MRR is `0.549624`. The result is documented in
`docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`. The positive-claim path
is blocked until text-action beats no-action on the agreed headline metrics.

Start by grounding in the current repo and issue tracker. Do not assume this
prompt's issue status is current. Run:

- `git status --short`
- `git branch --show-current`
- `git fetch origin`
- `gh issue view 150`
- `gh issue view 154`
- `gh issue view 159`
- `gh issue view 123`
- `gh issue view 124`
- `gh issue view 125`
- `gh issue view 126`
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
- docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md
- docs/cards/codelewm-action-use-dataset-2026-05-20.md
- docs/cards/codelewm-action-use-model-2026-05-20.md
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
license gates, checkpoint trust gates, manifest verification, or private
publication defaults.

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

Use `scripts/hf-launch-codelewm-job` for project pipeline launches because it
builds the policy-compliant `hf jobs run` command with `--secrets HF_TOKEN`,
`--env`, `--label`, `--timeout`, `--flavor`, and `--detach`.

Work in this order, skipping only issues already closed by merged PRs and
verifying their artifacts before moving on:

1. #159: run the second-stage action-use remediation sweep if the project is
   still pursuing a positive claim. Prefer the checked-in fallback config
   `config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml`
   unless side-by-side analysis of #138 and #154 shows a smaller correction is
   required first.
2. #123: release: add uv build and package publishing gates.
3. #124: release: add dependency audit and provenance evidence.
4. #125: docs: refresh public docs against verified scaled evidence and the
   final claim boundary.
5. #126: release: run final artifact freeze and checklist.

Historical closed gates to preserve, not redo: #118 source acquisition, #119
scaled configs/runbook, #120 action-view ablation, #121 scorer quality, #122
artifact cards, #137 HF Jobs automation, #138 first scaled HF execution, #151
no-action claim gates, #152 action-discriminative diagnostics, #153 action-use
margin training configs, and #154 primary action-use HF execution.

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

For #159, first compare the baseline and action-use reports:

- `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md`
- `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`

If no code/config correction is needed, launch the checked-in fallback profile.
The current public shard source is `config/data/codelewm_public_shard_commitpackft_python.json`,
which expects `bigcode/commitpackft:data/python/data.jsonl` under
`.artifacts/hf-sources/commitpackft`. Preflight the source path first:

hf download bigcode/commitpackft \
  data/python/data.jsonl \
  --repo-type dataset \
  --local-dir .artifacts/hf-sources/commitpackft \
  --dry-run

Start the #159 run through the launcher:

CODELEWM_HF_JOBS_DRY_RUN=0 \
CODELEWM_HF_PIPELINE_MODE=scaled \
CODELEWM_HF_JOBS_FLAVOR=a10g-small \
CODELEWM_HF_JOBS_TIMEOUT=24h \
CODELEWM_HF_PUBLISH=1 \
CODELEWM_HF_PUBLISH_DRY_RUN=0 \
CODELEWM_HF_REF=<merged-sha-or-main> \
CODELEWM_DATASET_BUILD_CONFIG=config/data/codelewm_public_shard_commitpackft_python.json \
CODELEWM_TRAIN_CONFIG=config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml \
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
repository targets, and command line in #159. If the job fails, preserve the job
ID, log excerpt, failure phase, and config paths in the relevant issue before
patching. Do not relaunch blindly.

After the job succeeds, use `hf download` to fetch the published private
artifacts into a clean local artifact directory. Do not validate against the
job's working directory. Validate the downloaded checkpoint and artifacts:

hf download "$CODELEWM_HF_RESULTS_REPO_ID" \
  --repo-type dataset \
  --include "runs/<run-id>/**" \
  --local-dir .artifacts/hf-download/<run-id>/results

hf download "$CODELEWM_HF_MODEL_REPO_ID" \
  --repo-type model \
  --include "checkpoints/<run-id>/**" \
  --local-dir .artifacts/hf-download/<run-id>/model

hf download "$CODELEWM_HF_DATASET_REPO_ID" \
  --repo-type dataset \
  --include "runs/<run-id>/pack/**" \
  --local-dir .artifacts/hf-download/<run-id>/dataset

uv run codelewm manifest verify \
  --manifest .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/manifest.json \
  --parent-manifest .artifacts/hf-download/<run-id>/dataset/runs/<run-id>/pack/manifest.json \
  --json

uv run codelewm eval retrieval \
  --checkpoint .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --data .artifacts/hf-download/<run-id>/dataset/runs/<run-id>/pack \
  --out .artifacts/hf-download/<run-id>/local-checks/retrieval \
  --device cpu \
  --seed 0 \
  --overwrite \
  --json

uv run codelewm eval ablation \
  --retrieval-artifact .artifacts/hf-download/<run-id>/local-checks/retrieval/manifest.json \
  --training-artifact .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/manifest.json \
  --out .artifacts/hf-download/<run-id>/local-checks/ablation \
  --overwrite \
  --json

uv run codelewm eval surprise \
  --checkpoint .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --data .artifacts/hf-download/<run-id>/dataset/runs/<run-id>/pack \
  --out .artifacts/hf-download/<run-id>/local-checks/surprise \
  --device cpu \
  --seed 0 \
  --overwrite \
  --json

uv run codelewm eval scorer-quality \
  --config config/first_results/scorer_quality.json \
  --checkpoint .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --out .artifacts/hf-download/<run-id>/local-checks/scorer_quality \
  --device cpu \
  --index .artifacts/hf-download/<run-id>/results/runs/<run-id>/index \
  --retrieval-prior-weight 1.0 \
  --retrieval-prior-k 10 \
  --parent-manifest .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/manifest.json \
  --parent-manifest .artifacts/hf-download/<run-id>/results/runs/<run-id>/index/manifest.json \
  --overwrite \
  --json

Then run scorer and reranker smoke checks from the downloaded checkpoint and
downloaded index. Use real examples from the downloaded result bundle when
available; otherwise use checked-in fixtures and label them as fixture checks.

Update benchmark docs, dataset card, model card, roadmap, release checklist, and
the #150 tracker only with artifact-backed numbers. Every claim must name the
command, data source, commit SHA, run ID, job ID when applicable, and artifact
path that produced it.

Do not call the result claim-eligible unless all of these are true:

- the action-use claim gate from #151 passes;
- text-action improves over no-action on Recall@1 and MRR;
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
visibility gate is explicitly satisfied. If #159 remains negative, stop the
positive-claim path, keep repositories private or frame the release explicitly
as negative/diagnostic, and update #150 with the exact blocker, job ID, and next
validation command.

Completion criteria:

- #159, #123, #124, #125, and #126 are closed by merged PRs, or any remaining
  one is updated with an explicit blocker that prevents completion.
- At least one second-stage HF Jobs run or explicitly justified no-run decision
  is recorded in #159.
- Dataset, model, and results artifacts selected for release are published
  privately to the configured Hugging Face repositories.
- Published artifacts are downloaded with `hf download` and verified locally.
- Retrieval, action-view ablation, surprise, score, and rerank checks run from
  the downloaded checkpoint/artifacts.
- Dataset/model cards and benchmark docs reflect only verified artifacts.
- The final response reports the HF job ID if one ran, published repo paths,
  validation commands, metrics summary, caveats, and remaining release blockers.
```

Launch command:

```bash
ml-intern --max-iterations -1 "$(cat docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md)"
```
