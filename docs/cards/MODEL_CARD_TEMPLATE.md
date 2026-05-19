# CodeLeWM Model Card Template

> Copy this file into `docs/cards/<model>-<release>.md` and fill in
> every section. The release checklist rejects cards that omit
> required sections, skip manifest IDs, or claim performance without
> backing benchmark evidence.

- Model name: `<short slug>`
- Checkpoint manifest id: `<artifact_id from codelewm.checkpoint.v1>`
- Checkpoint manifest path: `<docs-relative path>`
- Training run manifest id: `<artifact_id from codelewm.training_run.v1>`
- Dataset manifest id: `<artifact_id from codelewm.dataset.v1>`
- Source git SHA: `<40-char SHA matching manifest.source_git_sha>`
- Release tag: `<v0.x.y | v1.x.y>`
- Author: `<github-handle>`

## Summary

> One paragraph describing the model architecture (JEPA-style
> latent transition), the action view it consumes, the training
> objective, and the intended downstream use (local scorer / reranker).
> No marketing copy.

## Schema Versions

| Surface | Schema version |
| ------- | -------------- |
| Checkpoint manifest | `codelewm.checkpoint.v1` |
| Training run manifest | `codelewm.training_run.v1` |
| Artifact manifest | `codelewm.artifact_manifest.v1` |
| Retrieval report | `codelewm.eval.retrieval_report.v1` |
| Surprise report | `codelewm.eval.surprise_report.v1` |

## Architecture

| Field | Value |
| ----- | ----- |
| Model class | `<CodeTransitionModel | other>` |
| Action view | `<text | abstract>` |
| Latent dim | `<int>` |
| State sequence length | `<int>` |
| Action sequence length | `<int>` |
| History size | `<int>` |
| Prediction horizon | `<int>` |

Match every value against
`checkpoint_manifest.metadata` and the resolved training config.

## Training

| Field | Value |
| ----- | ----- |
| Objective | MSE + SIGReg |
| Retrieval loss gate | `<enabled | disabled>` |
| Seed | `<int>` |
| Optimizer | `<adam / adamw / ...>` |
| Step count | `<int>` |
| Final loss/total | `<float>` |
| Compute | `<cpu | cuda | mps>` |

Values must match `training_run_manifest.final_metrics`,
`training_run_manifest.step_count`, `training_run_manifest.seed`,
and the resolved config recorded under `config.json` in the run
directory.

## Intended Use

- Score / rerank candidate after-states for a single before-state
  and instruction.
- Optional nearest-historical-edit retrieval via the local
  transition index.

## Out-of-Scope Use

- Generating new code.
- Running candidate code through tests.
- Modifying the user's working tree.
- Loading checkpoints without their `<checkpoint>.manifest.json`.

## Evaluation Evidence

### Retrieval (headline)

| Metric | Value | Manifest field |
| ------ | ----- | -------------- |
| Recall@1 | | `recall_at_1` |
| Recall@5 | | `recall_at_5` |
| Recall@10 | | `recall_at_10` |
| MRR | | `mrr` |
| Median rank | | `median_rank` |

Required baselines (random, lexical, no-action, shuffled-action,
patch-action diagnostic) must be reported in the benchmark report
linked above. The model card may summarise but must not omit.

### Patch Surprise

| Metric | Value | Manifest field |
| ------ | ----- | -------------- |
| Pairwise AUC overall | | `metrics.pairwise_auc_overall` |
| Recall@1 | | `metrics.recall_at_1` |

By-category AUC table is required in the benchmark report; the
card may summarize one or two highlights.

### Action-View Diagnostic

State explicitly which action view the headline used (`action_text`
is the only allowed value for headline claims) and which views
appear only as diagnostics.

## Limitations And Risks

> Honest bullets: data biases inherited from the dataset, failure
> modes that have been observed (collapse, miscalibration on long
> diffs, lexical-baseline dominance on commit-message-style
> actions, ...), known overfitting regimes, hardware quirks.

## Reproduction

| Step | Command |
| ---- | ------- |
| Train | `codelewm train --config <yaml>` |
| Evaluate retrieval | `codelewm eval retrieval --checkpoint <ckpt> --data <pack_dir> --out <retrieval_dir>` |
| Evaluate surprise | `codelewm eval surprise --checkpoint <ckpt> --data <pack_dir> --out <surprise_dir>` |
| Verify | `codelewm manifest verify --manifest <run_dir>/manifest.json` |
| Score one candidate | `codelewm score --before <before.py> --instruction <text> --candidate <after.py> --checkpoint <ckpt>` |

The trust gate refuses to load the checkpoint without its
`<checkpoint>.manifest.json`. Releases must verify the trust gate
fires on the published checkpoint.

## Caveats

> Anything a downstream user needs to know that doesn't fit the
> categories above. Distribution-shift warnings, recommended
> calibration steps, etc.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| -------- | ---- | ------------- | ---- |
| | | | |

At least one model owner sign-off is required.
