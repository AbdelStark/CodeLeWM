# HF ml-intern Goal Prompt

Use this prompt from the repository root after the HF automation branch is on the
ref that `CODELEWM_HF_REF` will check out.

```text
You are operating CodeLeWM toward a Hugging Face Hub-backed scaled training run.

Start from the current default branch. Before editing or launching compute, read
AGENTS.md, SPEC.md, docs/roadmap/FULL_COMPLETION.md,
docs/roadmap/IMPLEMENTATION.md, docs/roadmap/NEXT_GOAL_PROMPT.md,
docs/operations/HF_ML_INTERN_TRAINING.md, CONTRIBUTING.md, the relevant
docs/spec file, and the relevant docs/rfcs file.

Do not print, commit, paste, or summarize token values. Treat .env as local
secret state. Use .env values only as defaults, and prefer shell overrides for a
single run.

Goal: complete #138 by producing one meaningful CodeLeWM scaled training result
and publishing the artifact bundle to private Hugging Face repositories first,
then verifying the downloaded model with inference and evaluations before any
public claim.

Work in the existing issue order:
1. Finish #118: data: document and gate public source acquisition.
2. Finish #119: train: add scaled training configs and runbook.
3. Finish #120: eval: add action-view ablation suite.
4. Finish #121: eval: add scorer calibration and reranker quality report.
5. Finish #122: docs: fill dataset and model cards from artifacts.

For each issue, use one branch and one PR. Re-read the issue body and linked
spec/RFC, inspect current code before editing, implement the smallest complete
change, add focused tests/docs, run the strongest relevant validation, commit,
push, open the PR, wait for checks when available, merge after passing, return
to main, pull latest main, and continue.

After #118 and #119 are merged, run only dry-run infrastructure checks first:

CODELEWM_HF_JOBS_DRY_RUN=1 uv run scripts/hf-launch-codelewm-job

CODELEWM_HF_PIPELINE_MODE=smoke \
CODELEWM_HF_RUN_ID=local-smoke \
CODELEWM_HF_OUTPUT_ROOT=.artifacts/hf-local \
CODELEWM_HF_PUBLISH=1 \
CODELEWM_HF_PUBLISH_DRY_RUN=1 \
uv run scripts/hf-run-codelewm-pipeline

If dry-run checks pass and the scaled configs are ready on the merged ref, launch
the real scaled job:

CODELEWM_HF_JOBS_DRY_RUN=0 \
CODELEWM_HF_PIPELINE_MODE=scaled \
CODELEWM_HF_PUBLISH_DRY_RUN=0 \
CODELEWM_HF_REF=<merged-sha-or-main> \
CODELEWM_DATASET_BUILD_CONFIG=<checked-in-public-shard-build-config> \
CODELEWM_TRAIN_CONFIG=<checked-in-scaled-train-config> \
uv run scripts/hf-launch-codelewm-job

Monitor with hf jobs ps, hf jobs inspect <job-id>, hf jobs logs <job-id>, and
hf jobs stats <job-id>. If the job fails, preserve the job ID, log excerpt,
commit SHA, config paths, and failure phase in the relevant issue before
patching.

After the job succeeds, download the results/model repositories, verify all
manifests, rerun retrieval and surprise evaluation locally against the model
artifact, run a codelewm score/rerank smoke check from the downloaded checkpoint,
and update the benchmark report/cards only with artifact-backed numbers.

Do not call the result meaningful unless text-action improves over the required
baselines on held-out examples, collapse gates pass, decoy coverage is adequate,
license/security gates pass, and the published private artifacts can be
downloaded and verified from a clean checkout.

If any prerequisite is missing or the result is negative, stop the publish/public
claim path, keep repositories private, and open or update an evidence-backed
GitHub issue with the exact blocker and next validation command.
```

Launch command:

```bash
ml-intern --max-iterations -1 "$(cat docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md)"
```
