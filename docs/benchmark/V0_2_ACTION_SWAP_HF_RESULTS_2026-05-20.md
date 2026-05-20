# CodeLeWM v0.2 Action-Swap HF Results 2026-05-20

- Report ID: `codelewm-v0-2-action-swap-hf-results-2026-05-20`
- Run ID: `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b`
- Job ID: `6a0dea258229e585f969c808`
- Job URL: `https://huggingface.co/jobs/abdelstark/6a0dea258229e585f969c808`
- Source git SHA: `7c7cb0b8fe132e4819f05a77585c254267e77574`
- Evidence tier: scaled public HF artifact evidence, downloaded-artifact verification
- Hardware: HF Jobs `a10g-small`, timeout `24h`
- Dataset card: `docs/cards/codelewm-v0-2-action-swap-dataset-2026-05-20.md`
- Model card: `docs/cards/codelewm-v0-2-action-swap-model-2026-05-20.md`

## Verdict

The v0.2 action-swap/inverse-action intervention completed on Hugging Face
Jobs, published public dataset/model/results artifacts, downloaded those
artifacts with `hf download`, and passed local verification from the downloaded
files.

The result is negative/diagnostic. The intervention did not produce a positive
action-use, representation, or downstream-reranking claim:

- headline text-action retrieval loses to no-action;
- exact-same-before and near-before action-contrast slices do not beat the
  v0.2 no-action margins;
- latent probes are explicitly reported as unsupported for semantic
  representation claims;
- downstream scoring/reranking remains a one-example smoke path, not a scaled
  usefulness result.

## Published Public Artifacts

| Surface | Repository | Path |
| --- | --- | --- |
| Dataset pack | `abdelstark/codelewm-public-shard` | `runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/pack` |
| Model checkpoint | `abdelstark/codelewm-transition-model` | `checkpoints/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` |
| Run evidence | `abdelstark/codelewm-runs` | `runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` |

Downloaded local root:
`.artifacts/hf-download/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b`.

## Artifact Chain

| Artifact | Path | Artifact ID |
| --- | --- | --- |
| Build | `results/runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/build/manifest.json` | `dataset-d67b1cd46dc05bea` |
| Pack | `dataset/runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/pack/manifest.json` | `dataset-daecac9f9965c563` |
| Training run | `model/checkpoints/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/manifest.json` | `training_run-0a41863d1da33737` |
| Remote retrieval | `results/runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/retrieval/manifest.json` | `eval_report-54860bf47fa58d03` |
| Local retrieval rerun | `local-checks/retrieval/manifest.json` | `eval_report-7106be1f701c8a90` |
| Local latent probe | `local-checks/latent_probe/manifest.json` | `eval_report-642bf385ae9bc9bc` |
| Local action ablation | `local-checks/ablation/manifest.json` | `eval_report-3c83cf8b3ac03d65` |
| Local surprise | `local-checks/surprise/manifest.json` | `eval_report-8e16f7fe3e00ecbb` |
| Index | `results/runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/index/manifest.json` | `index-4a0e11baa24f9c49` |
| Local scorer quality | `local-checks/scorer_quality/manifest.json` | `score_report-034108c90ec57676` |

## Dataset

| Field | Value |
| --- | --- |
| Source | `bigcode/commitpackft`, `data/python/data.jsonl`, revision `main` |
| Raw rows loaded | 56,025 |
| License-included rows before filters | 23,015 |
| License-excluded rows | 33,010 |
| Packed transitions | 20,645 |
| Train / val / test | 18,019 / 1,291 / 1,335 |
| License gate | `release_allowed=true`, `blocked_rows=0` |

## v0.2 Action-Contrast Pools

The downloaded retrieval rerun regenerated
`reports/action_contrast_pool_report.json` with schema
`codelewm.eval.action_contrast_pool_report.v1`.

| Pool | Selected candidates |
| --- | ---: |
| `exact_same_before` | 6 |
| `near_before` | 186 |
| `same_file` | 218 |
| `action_cluster` | 281 |
| `edit_shape` | 14,619 |
| `mutation` | 0 |
| `random` | 16,000 |

Leakage report: `selected_train_rows=0`, `query_train_rows=0`,
`input_train_rows=0`, `leakage_detected=false`.

No-action challenge report:

- `exact_same_before_query_count=6`;
- `same_before_multi_action_query_count=4`;
- `synthetic_controlled_same_before_query_count=0`;
- `no_action_prior_insufficient=true`.

Synthetic controlled transforms are implemented and covered by #171 fixture
tests. This public CommitPackFT shard did not contribute synthetic controlled
same-before rows to the scaled run.

## Training

| Field | Value |
| --- | --- |
| Config | `config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml` |
| Executor | `torch` |
| Device | `cuda` |
| Precision | `bf16-mixed` |
| Torch | `2.12.0+cu130` |
| Seed | 240119 |
| Steps | 60,000 |
| Checkpoint SHA-256 | `f2c5ba50ee0ec5e32ff5c3ceed848020e989ebdb1c98a917f17589ee523c6d7e` |
| Objective | MSE + SIGReg + no-action margin + action-swap contrastive + inverse-action reconstruction |
| Action-use margin weight | 0.25 |
| Action-swap contrastive weight | 0.2 |
| Inverse-action reconstruction weight | 0.1 |
| Loss total | 0.123456 |
| Validation loss total | 0.426304 |
| Validation prediction MSE | 0.149414 |
| Validation swap-distance gap | 0.035137 |
| Collapse effective rank | 4.034709 |
| Collapse effective-rank ratio | 0.015761 |
| Examples per second | 1323.917 |

## Headline Retrieval

Downloaded local CPU rerun, 1,000 held-out queries with 1,000 candidates:

| Variant | Recall@1 | Recall@5 | Recall@10 | MRR | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text action | 0.263 | 0.478 | 0.596 | 0.370048 | 6.5 |
| Random | 0.001 | 0.004 | 0.008 | 0.007118 | 502 |
| Shuffled action | 0.001 | 0.007 | 0.012 | 0.007563 | 502.5 |
| Lexical | 0.045 | 0.130 | 0.190 | 0.093745 | 152 |
| No action | 0.441 | 0.638 | 0.712 | 0.533105 | 2 |

The headline action-use claim gate is `claim_allowed=false`.

Failure reason:

```text
no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action
```

## Action-Contrast Retrieval Gate

Positive v0.2 action-use claims require text-action to beat no-action by at
least `0.10` Recall@1 and `0.08` MRR on exact-same-before and near-before
action-contrast pools. This run fails that gate.

| Pool | Queries | Text Recall@1 | No-action Recall@1 | Delta | Text MRR | No-action MRR | Delta | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exact_same_before` | 6 | 0.333333 | 0.333333 | 0.000000 | 0.666667 | 0.666667 | 0.000000 | fail |
| `near_before` | 112 | 0.678571 | 0.723214 | -0.044643 | 0.828423 | 0.854464 | -0.026042 | fail |
| `same_file` | 139 | 0.553957 | 0.553957 | 0.000000 | 0.762590 | 0.766187 | -0.003597 | fail |
| `action_cluster` | 70 | 0.857143 | 0.928571 | -0.071429 | 0.923810 | 0.964286 | -0.040476 | fail |
| `edit_shape` | 987 | 0.780142 | 0.843972 | -0.063830 | 0.862312 | 0.903093 | -0.040781 | fail |
| `random` | 1,000 | 0.755000 | 0.833000 | -0.078000 | 0.842392 | 0.893789 | -0.051397 | fail |

Interpretation: no-action remains as strong or stronger not only on headline
hard-1k retrieval but also on the v0.2 contrast pools. This supports a
negative action-use result, not an action-sensitive model-quality claim.

## Latent Probe

Downloaded local CPU rerun wrote
`reports/latent_probe_report.json` with schema
`codelewm.eval.latent_probe_report.v1`.

Report-level claim boundary:

- `semantic_structure_status=unsupported`;
- `positive_representation_claim_allowed=false`;
- `dimension_claims_allowed=false`;
- reason: latent probes do not consistently beat listed controls on available
  targets.

| Target | Best latent view / accuracy | Best listed control / accuracy | Status |
| --- | ---: | ---: | --- |
| `edit_class` | `z_before` / 0.003 | `metadata_only` / 0.103 | unsupported |
| `ast_node_kind` | `z_pred_after` / 0.456 | `lexical` / 0.521 | unsupported |
| `symbol_kind` | `z_before` / 0.358 | `lexical` / 0.482 | unsupported |
| `edit_size_bucket` | `z_after` / 0.119 | `metadata_only` / 0.686 | unsupported |
| `action_cluster` | `z_before` / 0.000 | `metadata_only` / 0.023 | unsupported |
| `source_family` | not evaluable | fewer than two train labels | not evaluable |

Some dimensions show high exploratory associations, but the report explicitly
blocks naming semantic axes without seed/split-stable dimension evidence.

## Surprise

Downloaded local CPU rerun:

| Metric | Value |
| --- | ---: |
| Example count | 1,000 |
| Pairwise AUC overall | 0.732763 |
| Recall@1 | 0.476 |
| Mean true rank | 1.562 |
| Median true rank | 2 |

| Decoy category | Count | Pairwise AUC |
| --- | ---: | ---: |
| Random | 1,000 | 0.961 |
| Mutation | 1,000 | 0.501 |
| Same file | 63 | 0.634921 |
| Action cluster | 40 | 0.975 |

## Scorer And Reranker

The scorer-quality report has one labeled example, four candidates, two valid
candidates, and two expected error candidates. It is a smoke path for command
and artifact integration, not scaled downstream evidence.

| Field | Value |
| --- | ---: |
| Examples | 1 |
| Valid candidates | 2 |
| Error candidates | 2 |
| Final-score Recall@1 | 1.0 |
| Final-score MRR | 1.0 |
| Transition-energy-only Recall@1 | 1.0 |
| Retrieval-prior-only Recall@1 | 1.0 |

Benchmark readiness remains blocked:

```text
scaled downstream benchmark requires at least 100 labeled examples; got 1
```

## Verification Command

The full clean-download verifier completed successfully:

```bash
CODELEWM_HF_RUN_ID=codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b \
  uv run scripts/hf-verify-codelewm-run --json
```

It completed all planned commands:

- `hf download` for results, model, and dataset pack;
- dataset and model manifest verification;
- regenerated retrieval, latent-probe, ablation, surprise, and scorer-quality
  reports from downloaded artifacts;
- index manifest verification;
- `codelewm score` and `codelewm rerank` smoke checks from the downloaded
  checkpoint and transition index;
- secret scan over the downloaded artifact root.

Secret scan result: `ok=true`, zero findings.

`hf jobs stats` was attempted during monitoring, but the local HF CLI stats
command behaved like a long-running stream in this environment and had to be
stopped. Job inspect/logs, HF metrics API samples, public artifact listing,
`hf download`, and local verification supplied the durable evidence.

## Claim Checklist

- [x] v0.2 HF Jobs run completed on the recorded source SHA.
- [x] Dataset, model, and result artifacts were published to public HF repos.
- [x] Published artifacts were downloaded with `hf download`.
- [x] Manifest verification, checkpoint trust, source/license gates, and secret
      scan passed.
- [x] Retrieval, latent-probe, ablation, surprise, scorer-quality, score, and
      rerank checks ran from downloaded artifacts.
- [x] Action-contrast pools are present, schema-versioned, leakage-checked, and
      include unavailable-pool evidence.
- [ ] Text action beats no-action on headline retrieval.
- [ ] Text action beats no-action by the v0.2 exact-same-before and near-before
      action-contrast margins.
- [ ] Latent probes support a positive semantic representation claim.
- [ ] Downstream reranking supports a scaled coding-usefulness claim.

## Completion Boundary

Issue #172 closes as a completed public HF v0.2 sweep with a negative
diagnostic result. The current evidence invalidates the v0.2 action-swap /
inverse-action intervention as a positive action-use fix. It also does not
validate semantic latent-axis or downstream usefulness claims.

Future work should be scoped as a new research hypothesis, not as release
cleanup.
