# CodeLeWM Benchmark Report Template

> Copy this file into `docs/benchmark/<release>-<date>.md` and fill in
> every section. Reports that leave required tables blank or do not
> reference the backing manifest IDs are rejected by the release gate.

- Report ID: `codelewm-benchmark-<release>-<date>`
- Schema version: `codelewm.eval.retrieval_report.v1` for retrieval
  tables; `codelewm.eval.surprise_report.v1` for surprise tables;
  `codelewm.public_license_gate.v1` for license claims.
- Release: `<v0.1 | v1.0 | other>`
- Date (UTC): `<YYYY-MM-DD>`
- Checkpoint manifest: `<artifact_id from codelewm.checkpoint.v1>`
- Dataset manifest: `<artifact_id from codelewm.dataset.v1>`
- Index manifest: `<artifact_id from codelewm.transition_index.v1>`
- Source git SHA: `<40-char SHA from manifest.source_git_sha>`
- Run reproduction command: `<exact CLI line that produced the run>`
- Author / on-call: `<github-handle>`

## Reproducibility Chain

| Artifact | Schema version | Manifest path | Artifact ID |
| -------- | -------------- | ------------- | ----------- |
| Dataset  | `codelewm.dataset.v1` | `<path>` | `<id>` |
| Training run | `codelewm.training_run.v1` | `<path>` | `<id>` |
| Checkpoint | `codelewm.checkpoint.v1` | `<path>` | `<id>` |
| Index | `codelewm.transition_index.v1` | `<path>` | `<id>` |
| Retrieval report | `codelewm.eval.retrieval_report.v1` | `<path>` | `<id>` |
| Surprise report | `codelewm.eval.surprise_report.v1` | `<path>` | `<id>` |
| License gate | `codelewm.public_license_gate.v1` | `<path>` | `<id>` |

Every manifest above must pass `codelewm manifest verify
--manifest <path>` before this report is accepted.

## Retrieval Evaluation

### Headline metrics (action_text)

| Metric | Value | Manifest field |
| ------ | ----- | -------------- |
| Recall@1 | `<float>` | `recall_at_1` |
| Recall@5 | `<float>` | `recall_at_5` |
| Recall@10 | `<float>` | `recall_at_10` |
| MRR | `<float>` | `mrr` |
| Median rank | `<float>` | `median_rank` |
| Candidate pool name | `<string>` | `candidate_pool.name` |
| Candidate pool size | `<int>` | `candidate_pool.entry_count` |
| Excluded splits | `train` (required) | `candidate_pool.excluded_splits` |
| Held-out source mix | `<source -> count>` | `metadata` |

Headline retrieval reports use `action_text`. Reports that use
`action_patch` are diagnostic only and must be tagged with
`action_view_policy=diagnostic_only` per
`codelewm.public_license_gate.v1`-derived policy in
`docs/spec/02-public-api.md`.

### Required baselines

Every headline retrieval report **must** include all four baselines.
Leave the cell blank only if the baseline is provably impossible for
the dataset; the absence-justification column is required.

| Baseline | Recall@1 | Recall@5 | Recall@10 | MRR | Absence justification |
| -------- | -------- | -------- | --------- | --- | --------------------- |
| Random | | | | | |
| Lexical (BM25 / TF-IDF) | | | | | |
| No-action | | | | | |
| Shuffled-action | | | | | |
| Patch-action (diagnostic) | | | | | |

If a row is empty and the absence-justification cell is empty, the
release gate refuses to publish this report.

### Slices

| Slice | Recall@1 | Recall@5 | Recall@10 | MRR | Sample count |
| ----- | -------- | -------- | --------- | --- | ------------ |
| Source: GitHub | | | | | |
| Source: CommitPackFT | | | | | |
| Source: Local repo | | | | | |
| Edit size 1–10 lines | | | | | |
| Edit size 11–50 lines | | | | | |
| Edit size 51+ lines | | | | | |
| Action view: text | | | | | |
| Action view: abstract | | | | | |

### Hard-negative pool report

| Field | Value |
| ----- | ----- |
| Pool name | `<string>` |
| Sampler config (hash) | `<sha256 prefix>` |
| Excluded splits | `train` |
| Hardness metric | `<weak action cluster | edit-size bucket | ...>` |

## Patch-Surprise Evaluation

### Headline surprise metrics

| Metric | Value | Manifest field |
| ------ | ----- | -------------- |
| Pairwise AUC (overall) | `<float>` | `metrics.pairwise_auc_overall` |
| Mean true rank | `<float>` | `metrics.mean_true_rank` |
| Median true rank | `<float>` | `metrics.median_true_rank` |
| Recall@1 | `<float>` | `metrics.recall_at_1` |
| Examples scored | `<int>` | `metrics.example_count` |
| Decoy seed | `<int>` | `decoy_seed` |
| Score direction | `lower_is_better` | `score_direction` |

### Pairwise AUC by decoy category

| Decoy category | Pairwise AUC | Decoy count |
| -------------- | ------------ | ----------- |
| random | | |
| same_file | | |
| mutation | | |
| action_cluster | | |

A category row may be omitted only when the corpus produced zero
decoys for that category. State the reason in the comments section.

## License And Source Policy

| Field | Value |
| ----- | ----- |
| Public artifact policy | `<exclude | metadata_only | embeddings | full_text>` |
| License gate result | `<pass | fail>` |
| Included rows | `<int>` |
| Excluded rows | `<int>` |
| Blocked rows | `<int>` |
| Permissive licenses included | `<license -> count>` |
| Non-allowlisted licenses excluded | `<license -> count>` |

The license gate report must accompany the dataset manifest before
this report is accepted.

## Claim Checklist

Tick every claim this report makes and link the supporting manifest
field. Unchecked claims must not appear in any release announcement,
README, or external communication.

- [ ] **Action conditioning beats no-action.** Evidence: retrieval
      report shows `no_action` baseline strictly below the headline
      result on Recall@1 and MRR. Manifest fields:
      `baselines.no_action.recall_at_1`, `mrr`.
- [ ] **Action conditioning beats shuffled-action.** Evidence:
      `baselines.shuffled_action` baseline strictly below the headline
      result on Recall@1 and MRR.
- [ ] **Headline retrieval uses `action_text`.** Evidence:
      `action_view_policy=headline_text_only` in the retrieval report.
- [ ] **Hard negatives exclude the true target.** Evidence:
      hard-negative sampler report's `excluded_targets` matches the
      pool's `entry_count` and excludes the `train` split.
- [ ] **Patch-surprise reports pairwise AUC over four decoy
      categories.** Evidence:
      `metrics.pairwise_auc_by_category` covers the configured
      categories with non-zero decoy counts.
- [ ] **Every manifest referenced verifies cleanly.** Evidence:
      `codelewm manifest verify --manifest <path>` returns exit 0 for
      every artifact above.
- [ ] **No secret-pattern leakage in published artifacts.** Evidence:
      `codelewm secret-scan <reports_dir>` returns exit 0.
- [ ] **License gate passed before publication.** Evidence: license
      gate report attached above.

## Caveats

> Use this section to note anything a reader of the headline numbers
> would otherwise miss: known dataset gaps, dropped baselines and the
> reason, deviations from the documented evaluation policy, or
> hardware notes that affect reproducibility.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| -------- | ---- | ------------- | ---- |
| | | | |

Reports without at least one reviewer sign-off are rejected by the
release gate.
