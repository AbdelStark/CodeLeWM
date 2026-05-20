# CodeLeWM v0.2 Action-Swap Model Card

- Model name: `codelewm-v0-2-action-swap-inverse-gpu-a10g-text-action`
- Run ID: `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b`
- Training artifact id: `training_run-0a41863d1da33737`
- Checkpoint SHA-256: `f2c5ba50ee0ec5e32ff5c3ceed848020e989ebdb1c98a917f17589ee523c6d7e`
- Model repo: `abdelstark/codelewm-transition-model`
- Model repo path: `checkpoints/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b`
- Dataset artifact id: `dataset-daecac9f9965c563`
- Source git SHA: `7c7cb0b8fe132e4819f05a77585c254267e77574`
- Benchmark report: `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`
- Release status: public negative v0.2 research evidence, not public positive quality release
- Card date: `2026-05-20`

## Summary

This card describes the v0.2 CodeLeWM checkpoint trained on Hugging Face Jobs
with the no-action margin, action-swap contrastive loss, inverse-action
reconstruction, and gated residual action fusion. The checkpoint was published
to Hugging Face, downloaded with `hf download`, verified locally with its parent
dataset artifact, and used for downloaded-artifact retrieval, latent-probe,
ablation, surprise, scorer-quality, score, and rerank checks.

The run is a completed negative diagnostic result. It does not support a
positive action-conditioned model-quality claim, a semantic latent-axis claim,
or a scaled downstream coding-usefulness claim.

## Architecture

| Field | Value |
| --- | --- |
| Model class | `TorchCodeTransitionModel` |
| Action view | `text` |
| Action fusion | `gated_residual` |
| History size | 1 |
| Prediction horizon | 1 |
| Latent dimension | 256 |
| State sequence length | 1024 |
| Action sequence length | 256 |

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
| Train rows | 18,019 |
| Validation rows | 1,291 |
| Optimizer | AdamW |
| Objective | MSE + SIGReg + no-action margin + action-swap contrastive + inverse-action reconstruction |
| Action-use margin weight | 0.25 |
| Action-swap contrastive weight | 0.2 |
| Inverse-action reconstruction weight | 0.1 |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss/total` | 0.123456 |
| `loss/prediction_mse` | 0.004847 |
| `loss/action_use_margin` | 0.007081 |
| `loss/action_swap_contrastive` | 0.004875 |
| `loss/inverse_action_reconstruction` | 0.211372 |
| `val/loss/total` | 0.426304 |
| `val/loss/prediction_mse` | 0.149414 |
| `val/action_diagnostics/swap_distance_gap` | 0.035137 |
| `collapse/effective_rank` | 4.034709 |
| `collapse/effective_rank_ratio` | 0.015761 |
| `train/examples_per_second` | 1323.917 |

## Evaluation

### Headline Retrieval

Downloaded local CPU rerun, 1,000 held-out queries with 1,000 candidates:

| Variant | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Text action | 0.263 | 0.478 | 0.596 | 0.370048 |
| Random | 0.001 | 0.004 | 0.008 | 0.007118 |
| Shuffled action | 0.001 | 0.007 | 0.012 | 0.007563 |
| Lexical | 0.045 | 0.130 | 0.190 | 0.093745 |
| No action | 0.441 | 0.638 | 0.712 | 0.533105 |

The action-use claim gate is `claim_allowed=false`.

### Action-Contrast Slices

| Pool | Text Recall@1 | No-action Recall@1 | Text MRR | No-action MRR | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `exact_same_before` | 0.333333 | 0.333333 | 0.666667 | 0.666667 | fail |
| `near_before` | 0.678571 | 0.723214 | 0.828423 | 0.854464 | fail |
| `same_file` | 0.553957 | 0.553957 | 0.762590 | 0.766187 | fail |
| `action_cluster` | 0.857143 | 0.928571 | 0.923810 | 0.964286 | fail |
| `edit_shape` | 0.780142 | 0.843972 | 0.862312 | 0.903093 | fail |

No-action remains equal or stronger on the v0.2 action-contrast slices.

### Latent Probe

The downloaded latent-probe rerun reports
`semantic_structure_status=unsupported`,
`positive_representation_claim_allowed=false`, and
`dimension_claims_allowed=false`.

| Target | Best latent view / accuracy | Best listed control / accuracy | Status |
| --- | ---: | ---: | --- |
| `edit_class` | `z_before` / 0.003 | `metadata_only` / 0.103 | unsupported |
| `ast_node_kind` | `z_pred_after` / 0.456 | `lexical` / 0.521 | unsupported |
| `symbol_kind` | `z_before` / 0.358 | `lexical` / 0.482 | unsupported |
| `edit_size_bucket` | `z_after` / 0.119 | `metadata_only` / 0.686 | unsupported |
| `action_cluster` | `z_before` / 0.000 | `metadata_only` / 0.023 | unsupported |
| `source_family` | not evaluable | fewer than two train labels | not evaluable |

### Surprise

| Metric | Value |
| --- | ---: |
| Example count | 1,000 |
| Pairwise AUC overall | 0.732763 |
| Recall@1 | 0.476 |
| Mean true rank | 1.562 |
| Median true rank | 2 |

### Scorer And Reranker

The scorer-quality path has one labeled example. It validates command and
artifact plumbing, not scaled downstream usefulness.

| Component | Recall@1 | MRR | Evaluable examples |
| --- | ---: | ---: | ---: |
| Final score | 1.0 | 1.0 | 1 |
| Transition energy only | 1.0 | 1.0 | 1 |
| Retrieval prior only | 1.0 | 1.0 | 1 |

Readiness remains blocked:
`scaled downstream benchmark requires at least 100 labeled examples; got 1`.

## Verification

Downloaded checkpoint path:
`.artifacts/hf-download/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/model/checkpoints/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/checkpoints/checkpoint.pt`.

Verified by:

```bash
CODELEWM_HF_RUN_ID=codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b \
  uv run scripts/hf-verify-codelewm-run --json
```

Result: model manifest verification `ok=true`, parent
`dataset-daecac9f9965c563`. Secret scan over the downloaded artifact root
returned `ok=true` with zero findings.

## Intended Use

- Reproduce the v0.2 action-swap/inverse-action negative diagnostic run.
- Compare future action-use interventions against the current failed v0.2
  baseline.
- Inspect downloaded artifact manifests, retrieval reports, latent-probe
  reports, and scorer-quality reports.

## Out-of-Scope Use

- Generating or executing code.
- Treating this checkpoint as a public positive-quality release model.
- Claiming action-conditioned superiority over no-action baselines.
- Claiming named semantic latent dimensions.
- Claiming scaled downstream patch-ranking usefulness.

## Limitations

- No-action is stronger than text action on headline retrieval.
- No-action is equal or stronger on exact-same-before, near-before, same-file,
  action-cluster, and edit-shape action-contrast slices.
- Latent probes do not support the representation claim gate.
- Downstream scorer-quality evidence has one labeled example.
- Public artifact visibility does not imply a positive model-quality claim.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Model owner | @AbdelStark | 2026-05-20 |
