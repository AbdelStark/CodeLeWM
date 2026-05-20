# CodeLeWM Benchmark Report Template

> Copy this file into `docs/benchmark/<release>-<date>.md` and fill in
> every section. Reports that leave required tables blank or do not
> reference the backing manifest IDs are rejected by the release gate.

- Report ID: `codelewm-benchmark-<release>-<date>`
- Schema version: `codelewm.eval.retrieval_report.v1` for retrieval
  tables; `codelewm.eval.action_ablation_report.v1` for action-view
  ablations; `codelewm.eval.surprise_report.v1` for surprise tables;
  `codelewm.harness.scorer_quality_report.v1` for scorer/reranker quality;
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
| Action ablation report | `codelewm.eval.action_ablation_report.v1` | `<path>` | `<id>` |
| Surprise report | `codelewm.eval.surprise_report.v1` | `<path>` | `<id>` |
| Scorer quality report | `codelewm.harness.scorer_quality_report.v1` | `<path>` | `<id>` |
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

### Action-use claim gate

Every headline report must include a machine-readable
`action_use_claim_gate` with text-action versus baseline deltas. Positive
action-conditioning claims are allowed only when
`action_use_claim_gate.claim_allowed=true`.

| Field | Value |
| ----- | ----- |
| Gate schema | `codelewm.eval.action_use_claim_gate.v1` |
| Claim allowed | `<true|false>` |
| Failure reasons | `<none | no_action_dominance:... | missing_baseline:...>` |
| No-action Recall@1 delta | `<text_action - no_action>` |
| No-action MRR delta | `<text_action - no_action>` |

### v0.2 Action-contrast gate

For v0.2 action-use claims, the benchmark must include
`reports/action_contrast_pool_report.json` with schema
`codelewm.eval.action_contrast_pool_report.v1`. Positive action-use claims are
blocked unless downloaded HF artifacts satisfy the v0.2 gate on action-contrast
pools:

| Pool | Text-action Recall@1 | No-action Recall@1 | Delta | Text-action MRR | No-action MRR | Delta | Gate |
| ---- | -------------------- | ------------------ | ----- | --------------- | ------------- | ----- | ---- |
| exact_same_before | | | `>= 0.10` | | | `>= 0.08` | `<pass|fail>` |
| near_before | | | `>= 0.10` | | | `>= 0.08` | `<pass|fail>` |

The action-contrast report must show `leakage.selected_train_rows=0`, list
unavailable-pool reasons, and preserve split-membership proofs for every
candidate pool. Treat random, same-file, action-cluster, edit-shape, and
mutation pools as controls unless the report explicitly scopes the claim to a
different slice.

## Latent Representation Probes

Representation claims require `reports/latent_probe_report.json` with schema
`codelewm.eval.latent_probe_report.v1`. Fill this section from downloaded HF
artifacts, not local-only scratch runs.

| Probe target | Best latent view | Test accuracy | Best control | Control accuracy | Status |
| ------------ | ---------------- | ------------- | ------------ | ---------------- | ------ |
| edit_class | | | | | `<supported|unsupported|not_evaluable>` |
| ast_node_kind | | | | | `<supported|unsupported|not_evaluable>` |
| symbol_kind | | | | | `<supported|unsupported|not_evaluable>` |
| edit_size_bucket | | | | | `<supported|unsupported|not_evaluable>` |
| action_cluster | | | | | `<supported|unsupported|not_evaluable>` |
| source_family | | | | | `<supported|unsupported|not_evaluable>` |

Per-dimension associations are diagnostic only. Do not name semantic axes unless
the report shows stable dimensions across seeds and splits and records
`dimension_claims_allowed=true`. State the report-level
`semantic_structure_status` exactly as emitted.

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

### Action-discriminative shard report

| Field | Value |
| ----- | ----- |
| Schema | `codelewm.data.action_discriminative_shard_report.v1` |
| Claim-ready shard coverage | `<true | false>` |
| Held-out rows | `<integer>` |
| Same-file / near-before pairs | `<integer>` |
| Action-cluster pairs | `<integer>` |
| Unavailable hard-negative pools | `<list>` |

## Action-View Ablation

Every expected variant must appear as `completed`, `blocked`, or
`failed`. Do not drop missing runs from the table.

| Row | Family | Status | Recall@1 | MRR | Artifact / reason |
| --- | ------ | ------ | -------- | --- | ----------------- |
| text_action | action_view | | | | |
| abstract_action | action_view | | | | |
| patch_action_diagnostic | action_view | | | | diagnostic upper bound only |
| no_action | baseline | | | | |
| shuffled_action | baseline | | | | |
| retrieval_loss_enabled | retrieval_loss | | | | |
| retrieval_loss_disabled | retrieval_loss | | | | |
| collapse_sigreg_0.05 | collapse | | | | |
| collapse_sigreg_0.09 | collapse | | | | |
| collapse_sigreg_0.15 | collapse | | | | |

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

## Scorer And Reranker Quality

The quality report must include true after-states, hard negatives, syntax
failures, and patch failures. The harness must parse and dry-run-apply
candidates as text without executing candidate code.

| Metric | Value | Report field |
| ------ | ----- | ------------ |
| Recall@1 | `<float>` | `summary.recall_at_1` |
| MRR | `<float>` | `summary.mrr` |
| Mean true rank | `<float>` | `summary.mean_true_rank` |
| Median true rank | `<float>` | `summary.median_true_rank` |
| Valid candidates | `<int>` | `summary.valid_count` |
| Error candidates | `<int>` | `summary.error_count` |
| Failure counts | `<error_type -> count>` | `summary.failure_counts` |
| Retrieval prior weight | `<float>` | `scoring_policy.retrieval_prior_weight` |
| Retrieval prior k | `<int>` | `scoring_policy.retrieval_prior_k` |

| Slice | Candidates | Valid | Errors | Mean final score | Mean retrieval prior |
| ----- | ---------- | ----- | ------ | ---------------- | -------------------- |
| true_after | | | | | |
| hard_negative | | | | | |
| syntax_failure | | | | | |
| patch_failure | | | | | |

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
- [ ] **Scorer/reranker quality report includes failure and
      calibration evidence.** Evidence:
      `codelewm.harness.scorer_quality_report.v1` records ranking
      metrics, calibration slices, parse/patch failure counts, and
      `candidate code is parsed and diff-applied as text but never
      executed`.
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
