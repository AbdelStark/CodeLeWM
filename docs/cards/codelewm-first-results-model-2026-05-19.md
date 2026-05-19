# CodeLeWM First-Results Model Card

- Model name: `codelewm-first-results-tiny`
- Checkpoint SHA-256: `b7c541c2057ec3318c47c047ad6c6ac3d2093ef88a4c2c0d9f46b449088655cf`
- Checkpoint manifest path: `.artifacts/first-results/train/checkpoints/checkpoint.pt.manifest.json`
- Training run manifest id: `training_run-e1d34484c3d2e5a7`
- Training run manifest path: `.artifacts/first-results/train/manifest.json`
- Dataset artifact id: `dataset-0bfb8da3ed5a70e5`
- Benchmark report: `docs/benchmark/FIRST_RESULTS.md`
- Source git SHA: `ce687cbd501fd3b113f7286e19a4ffc018f66949`
- Release status: smoke evidence, not a public model release
- Author: `@AbdelStark`
- Card date: `2026-05-19`

## Summary

This card describes the tiny CPU-trained CodeLeWM checkpoint produced by
`scripts/first-results`. The model uses a JEPA-style latent transition objective
with text actions, MSE prediction loss, and SIGReg regularization. Its intended
use is local scorer/reranker and artifact-lineage validation. It is not evidence
that CodeLeWM has learned general action-conditioned code-edit structure.

## Evidence Tiers

| Tier | Status | Interpretation |
| ---- | ------ | -------------- |
| Smoke evidence | present | The package-native path runs end to end from dataset build through scorer-quality reporting. |
| First-results evidence | present | Metrics are recorded in `docs/benchmark/FIRST_RESULTS.md` from local artifacts. |
| Scaled evidence | absent | No scaled HF Jobs run or downloaded private Hub artifact has been verified yet. |

## Schema Versions

| Surface | Schema version |
| ------- | -------------- |
| Checkpoint manifest | `codelewm.checkpoint.v1` |
| Training run manifest | `codelewm.training_run.v1` |
| Artifact manifest | `codelewm.artifact_manifest.v1` |
| Retrieval report | `codelewm.eval.retrieval_report.v1` |
| Action ablation report | `codelewm.eval.action_ablation_report.v1` |
| Surprise report | `codelewm.eval.surprise_report.v1` |
| Scorer quality report | `codelewm.harness.scorer_quality_report.v1` |

## Architecture

| Field | Value |
| ----- | ----- |
| Model class | `TorchCodeTransitionModel` |
| Action view | `text` |
| Latent dim | 256 |
| State sequence length | 1024 |
| Action sequence length | 256 |
| History size | 1 |
| Prediction horizon | 1 |

## Training

| Field | Value |
| ----- | ----- |
| Objective | MSE + SIGReg |
| Retrieval loss gate | disabled |
| Seed | 1337 |
| Optimizer | AdamW |
| Step count | 4 |
| Final loss/total | 0.654402 |
| Final prediction MSE | 0.507995 |
| Final SIGReg | 1.62675 |
| Compute | CPU, float32 |

Collapse diagnostics from the training manifest: effective rank `1.12557`,
minimum per-dimension variance `0.000130734`, nearest-neighbor entropy
`1.56071`.

## Intended Use

- Validate CodeLeWM's local training, manifest, evaluation, scoring, and
  reranking surfaces.
- Score or rerank candidate after-states for a single before-state and
  instruction in smoke tests.
- Exercise retrieval-prior plumbing through the train-split transition index.

## Out-of-Scope Use

- Generating new code.
- Running candidate code through tests.
- Modifying the user's working tree.
- Loading checkpoints without their `checkpoint.pt.manifest.json`.
- Claiming scaled or general model quality.

## Evaluation Evidence

### Retrieval (headline)

| Metric | Value | Manifest field |
| ------ | ----- | -------------- |
| Recall@1 | 1.0 | `metrics.recall_at_1` |
| Recall@5 | 1.0 | `metrics.recall_at_5` |
| Recall@10 | 1.0 | `metrics.recall_at_10` |
| MRR | 1.0 | `metrics.mrr` |
| Median rank | 1.0 | `metrics.median_rank` |

The benchmark report records that text-action ties random, lexical, no-action,
and shuffled-action baselines on the tiny fixture. It does not beat the required
baselines.

### Patch Surprise

| Metric | Value | Manifest field |
| ------ | ----- | -------------- |
| Pairwise AUC overall | 0.0 | `metrics.pairwise_auc_overall` |
| Recall@1 | 0.0 | `metrics.recall_at_1` |
| Mean true rank | 2.0 | `metrics.mean_true_rank` |
| Examples scored | 1 | `metrics.example_count` |

Only the mutation decoy category has non-zero coverage in the fixture.

### Action-View Diagnostic

The headline report uses `action_text`. The action-view ablation report has
`completed=7`, `blocked=5`, and `failed=0`; missing abstract-action,
retrieval-loss, alternate SIGReg, and patch-action diagnostic variants are
blocked rows rather than omitted evidence.

### Scorer / Reranker Quality

| Metric | Value | Report field |
| ------ | ----- | ------------ |
| Recall@1 | 1.0 | `summary.recall_at_1` |
| MRR | 1.0 | `summary.mrr` |
| Mean true rank | 1.0 | `summary.mean_true_rank` |
| Candidates | 4 | `summary.candidate_count` |
| Valid candidates | 2 | `summary.valid_count` |
| Error candidates | 2 | `summary.error_count` |
| Failure counts | `invalid_syntax=1`, `patch_apply_failed=1` | `summary.failure_counts` |

The scorer-quality report uses retrieval-prior weight `1.0` and records that
candidate code is parsed and diff-applied as text but never executed.

## Limitations And Risks

- The run has one held-out query and one retrieval candidate, so retrieval
  metrics are saturated.
- Surprise evaluation fails the headline expectation on the fixture and has
  sparse decoy coverage.
- Scorer/reranker quality uses one labeled fixture example.
- There is no scaled HF Jobs result, no downloaded Hub checkpoint verification,
  and no public release evidence yet.

## Reproduction

| Step | Command |
| ---- | ------- |
| Run first results | `uv run scripts/first-results --overwrite` |
| Verify training run | `uv run codelewm manifest verify --manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json` |
| Evaluate scorer quality | `uv run codelewm eval scorer-quality --config config/first_results/scorer_quality.json --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --out .artifacts/first-results/scorer_quality --index .artifacts/first-results/index --retrieval-prior-weight 1.0 --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/index/manifest.json --overwrite --json` |
| Verify scorer quality | `uv run codelewm manifest verify --manifest .artifacts/first-results/scorer_quality/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/index/manifest.json --json` |
| Secret scan | `uv run codelewm secret-scan .artifacts/first-results docs/benchmark/FIRST_RESULTS.md --json` |

## Caveats

This card is a filled smoke card so release checks can validate concrete card
content. Replace it with a scaled artifact-backed card before publishing a model
or making performance claims.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| -------- | ---- | ------------- | ---- |
| AbdelStark | Model owner | @AbdelStark | 2026-05-19 |
