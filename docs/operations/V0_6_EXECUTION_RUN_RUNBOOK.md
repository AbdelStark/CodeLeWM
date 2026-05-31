# v0.6 Execution-Substrate Training Run — Operator Runbook

This is the operator-facing runbook for the v0.6 execution-substrate
training run defined by RFC-0014 (#259) and tracked under #265.

## Preconditions

- Local `main` has the substrate-pivot PR stack merged (#274-#279 and
  the rest of #259's subtasks).
- The execution-pack dataset is published to Hugging Face under
  `abdelstark/codelewm-execution-pack@v0.6.0` (#263).
- The smoke pipeline passes locally (#264):
  `uv run scripts/smoke-execution-train --json | jq '.passed == true'`.
- `HF_TOKEN` is set in the operator's environment.
- `hf auth whoami` returns the expected account.
- The operator has reviewed and approved the claim-gate language in
  `config/train/scaled/codelewm_execution_v0_6_a10g.yaml`.

## Step 1 — Dry-run the launch plan

```bash
uv run scripts/hf-launch-execution-run \
  --config config/train/scaled/codelewm_execution_v0_6_a10g.yaml \
  --json
```

Inspect the printed JSON. Two plans should appear — one per seed
(42 and 1729). Each plan carries the run name, pack revision, HF
flavor, command vector, objective weights, and the resolved
claim-gate values. The script exits 0 without contacting HF.

## Step 2 — Fire the seed-42 run

```bash
# Take the command vector from the dry-run plan and run it.
hf auth whoami
hf jobs run \
  --flavor a10g-small \
  --timeout 24h \
  --secrets HF_TOKEN \
  --env CODELEWM_HF_RUN_NAME=<run_name from plan> \
  --env CODELEWM_EXECUTION_PACK_REPO_ID=abdelstark/codelewm-execution-pack \
  --env CODELEWM_EXECUTION_PACK_REVISION=v0.6.0 \
  --env CODELEWM_TRAIN_SEED=42 \
  --env CODELEWM_TRAIN_CONFIG=config/train/scaled/codelewm_execution_v0_6_a10g.yaml \
  ghcr.io/abdelstark/codelewm-runtime:v0.6 \
  /usr/local/bin/codelewm-runtime-entrypoint \
    codelewm train --config config/train/scaled/codelewm_execution_v0_6_a10g.yaml --seed 42
```

The entrypoint script is invoked explicitly because HF Jobs overrides
the image's `ENTRYPOINT` when a command is supplied. The entrypoint
pre-downloads the execution pack into
`$CODELEWM_EXECUTION_PACK_LOCAL_DIR` (default `/workspace/pack`), then
hands off to `codelewm train`. The `codelewm` console script is
installed system-wide at image build time, so `uv run` is unnecessary
(and its cache dir cannot be created under HF Jobs' read-only HOME).

Capture the job ID. Verify state with `hf jobs inspect <id>` and tail
the logs with `hf jobs logs <id>`.

### How the container resolves the pack

The runtime container's entrypoint pre-downloads the pack to
`/workspace/pack` and exports `CODELEWM_EXECUTION_PACK_LOCAL_DIR=/workspace/pack`
before invoking `codelewm train`. The runner short-circuits the HF
download in that case. When running locally, pass `--pack-local-dir
/path/to/pack` or set `CODELEWM_EXECUTION_PACK_LOCAL_DIR` to the same
value; the runner falls back to `huggingface_hub.snapshot_download`
only when neither is set.

## Step 3 — Fire the seed-1729 run

Repeat Step 2 with `--seed 1729`. The two seeds are required by the
#265 claim gates (variance bounds).

## Step 4 — Download artifacts

```bash
for SEED in 42 1729; do
  hf download abdelstark/codelewm-runs \
    codelewm-v0-6-execution-<date>-<sha>-seed-${SEED} \
    --local-dir .artifacts/v0_6/seed-${SEED}
done
```

## Step 5 — Local verify

```bash
for SEED in 42 1729; do
  uv run codelewm manifest verify \
    --manifest .artifacts/v0_6/seed-${SEED}/manifest.json --json
  uv run codelewm secret-scan .artifacts/v0_6/seed-${SEED} --json
  uv run codelewm model inspect-checkpoint \
    .artifacts/v0_6/seed-${SEED}/checkpoints/last.pt
done
```

## Step 6 — Publish checkpoints

```bash
for SEED in 42 1729; do
  uv run scripts/hf-publish-codelewm-artifacts \
    --artifact-root .artifacts/v0_6/seed-${SEED} \
    --run-id codelewm-v0-6-execution-<date>-<sha>-seed-${SEED} \
    --dataset-repo-id abdelstark/codelewm-execution-pack \
    --model-repo-id abdelstark/codelewm-transition-model \
    --results-repo-id abdelstark/codelewm-runs \
    --no-dry-run
done
```

## Step 7 — Run the headline evaluations

```bash
# Retrieval / collapse / probes / surprise:
for SEED in 42 1729; do
  uv run codelewm eval execution-retrieval \
    --checkpoint .artifacts/v0_6/seed-${SEED}/checkpoints/last.pt \
    --pack .artifacts/v0_6/execution-pack \
    --baselines random,lexical,no_action,shuffled_action \
    --out results/v0_6/seed-${SEED}/execution_retrieval \
    --json
  uv run codelewm eval execution-surprise \
    --checkpoint .artifacts/v0_6/seed-${SEED}/checkpoints/last.pt \
    --pack .artifacts/v0_6/execution-pack \
    --decoys mutation,same_problem_different_submission,same_code_different_input \
    --out results/v0_6/seed-${SEED}/execution_surprise \
    --json
  uv run codelewm eval execution-probe \
    --checkpoint .artifacts/v0_6/seed-${SEED}/checkpoints/last.pt \
    --pack .artifacts/v0_6/execution-pack \
    --targets output_type,will_raise,output_magnitude_bucket,output_length_bucket \
    --out results/v0_6/seed-${SEED}/execution_probe \
    --json
  uv run codelewm eval crash-prediction \
    --checkpoint .artifacts/v0_6/seed-${SEED}/checkpoints/last.pt \
    --pack .artifacts/v0_6/execution-pack \
    --out results/v0_6/seed-${SEED}/crash_prediction \
    --json
done

# Downstream rerank completion labeling (#304 operator step):
uv run scripts/sample-execution-rerank-completions \
  --benchmark humaneval \
  --source data/raw/humaneval.jsonl \
  --out results/v0_6/completion_labels/humaneval \
  --llm openrouter:anthropic/claude-haiku-4-5 \
  --samples-per-problem 10 \
  --llm-seeds 17,42,1729 \
  --live \
  --json
uv run scripts/sample-execution-rerank-completions \
  --benchmark mbpp-plus \
  --source data/raw/mbpp_plus.jsonl \
  --out results/v0_6/completion_labels/mbpp_plus \
  --llm openrouter:anthropic/claude-haiku-4-5 \
  --samples-per-problem 10 \
  --llm-seeds 17,42,1729 \
  --live \
  --json

uv run codelewm manifest verify \
  --manifest results/v0_6/completion_labels/humaneval/manifest.json \
  --json
uv run codelewm manifest verify \
  --manifest results/v0_6/completion_labels/mbpp_plus/manifest.json \
  --json

# Offline smoke for the same artifact contract, not claim evidence:
uv run scripts/sample-execution-rerank-completions \
  --benchmark humaneval \
  --source tests/data/execution_sources/fixtures/humaneval_tiny.jsonl \
  --out .artifacts/v0_6/rerank-label-smoke/humaneval \
  --samples-per-problem 2 \
  --llm-seeds 17 \
  --json
```

The sampler writes `codelewm.eval.completion_label.v1` JSONL rows plus
`codelewm.eval.completion_sampling_report.v1`, `codelewm.secret_scan.v1`,
and `codelewm.artifact_manifest.v1` reports. Candidate code is never run
directly in the evaluator; labeling goes through `codelewm.data.sandbox`
with the stdlib-only policy documented in `docs/operations/sandbox_policy.md`.
Live mode requires `OPENROUTER_API_KEY`; do not enable `OPENROUTER_DEBUG`
for publishable runs.

## Step 8 — Write the benchmark report

Use the template at `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`
as a starting point. The new report lives at
`docs/benchmark/EXECUTION_V0_6_RESULTS_<date>.md` and is the artifact
the #265 claim-gate review reads. The two-substrate paper outline
(#272) cites this report.

## Claim Gates (Recap from the Config)

A positive headline claim is allowed only when **all** of:

- text-action Recall@1 / MRR exceed no-action by ≥0.05 across both
  seeds;
- effective rank ratio ≥0.20; per-dim variance median ≥1e-8;
  nearest-neighbor entropy ≥0.10;
- ≥1 latent probe target beats every control across both seeds;
- mutation-decoy surprise AUC ≥0.65;
- same-problem-different-submission AUC ≥0.60;
- HumanEval/MBPP-Plus pass@1 lift ≥3 absolute points with bootstrap
  95% CI excluding zero across ≥3 LLM sampling seeds;
- checkpoint trust, manifest verify, downloaded-artifact verify,
  secret scan all pass.

If any gate fails, public claims remain limited to negative or
diagnostic evidence. The two-substrate comparison (commit-edit
v0.2 vs. execution v0.6) is itself a publishable finding regardless
of outcome.

## Reference

- Config: `config/train/scaled/codelewm_execution_v0_6_a10g.yaml`
- Launch plan generator: `scripts/hf-launch-execution-run`
- RFC: `docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`
- Roadmap: `docs/roadmap/EXECUTION_TRACE_WORLD_MODEL.md`
- Tracker: #259
- Implementation issue: #265
