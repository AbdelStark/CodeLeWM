# CodeLeWM Action-Use Model Card

- Model name: `codelewm-action-use-margin-gpu-a10g-text-action`
- Run ID: `codelewm-action-use-20260520-6650183`
- Training artifact id: `training_run-ce98fe8768af2143`
- Checkpoint SHA-256: `1e361498c722893c9754abcc9c2efa4499a615590572b77c7f0de939e789ac66`
- Model repo: `abdelstark/codelewm-transition-model`
- Model repo path: `checkpoints/codelewm-action-use-20260520-6650183`
- Dataset artifact id: `dataset-67895f8dc3e217c4`
- Source git SHA: `6650183`
- Benchmark report: `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`
- Release status: private negative action-use evidence, not public positive quality release
- Card date: `2026-05-20`

## Summary

This card describes the #154 CodeLeWM checkpoint trained on Hugging Face Jobs
with the no-action margin objective. The checkpoint was published privately,
downloaded with `hf download`, verified locally with its parent dataset
artifact, and used for downloaded-artifact retrieval, ablation, surprise,
scorer-quality, score, and rerank checks.

The run is not sufficient for a public positive action-conditioning claim.
Text-action beats random, shuffled-action, and lexical baselines, but it still
loses to no-action on Recall@1 and MRR.

## Architecture

| Field | Value |
| --- | --- |
| Model class | `TorchCodeTransitionModel` |
| Action view | `text` |
| History size | 1 |
| Prediction horizon | 1 |
| Latent dimension | 256 |
| State sequence length | 1024 |
| Action sequence length | 256 |

## Training

| Field | Value |
| --- | --- |
| Config | `config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml` |
| Executor | `torch` |
| Device | `cuda` |
| Precision | `bf16-mixed` |
| Torch | `2.12.0+cu130` |
| Seed | 240119 |
| Steps | 60,000 |
| Train rows | 18,019 |
| Validation rows | 1,291 |
| Optimizer | AdamW |
| Objective | MSE + SIGReg + no-action margin |
| Action-use margin weight | 0.25 |
| Action-use margin | 0.02 |
| Retrieval loss | disabled |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss/total` | 0.099949 |
| `loss/prediction_mse` | 0.005575 |
| `loss/action_use_margin` | 0.006401 |
| `val/loss/total` | 0.398456 |
| `val/loss/action_use_margin` | 0.055218 |
| `collapse/effective_rank` | 5.913854 |
| `collapse/effective_rank_ratio` | 0.023101 |
| `train/examples_per_second` | 1088.744 |

## Evaluation

### Retrieval

Remote headline retrieval, 1,000 held-out queries with 1,000 candidates:

| Variant | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Text action | 0.363 | 0.589 | 0.673 | 0.467875 |
| Random | 0.001 | 0.004 | 0.008 | 0.007118 |
| Shuffled action | 0.001 | 0.004 | 0.010 | 0.007474 |
| Lexical | 0.045 | 0.130 | 0.190 | 0.093745 |
| No action | 0.469 | 0.640 | 0.700 | 0.549624 |

Interpretation: the no-action margin objective did not fix no-action dominance.
The action-use claim gate is `claim_allowed=false`.

### Action Ablation

| Field | Value |
| --- | ---: |
| Completed rows | 7 |
| Blocked rows | 5 |
| Failed rows | 0 |

Unavailable abstract-action, patch-action diagnostic, retrieval-loss-enabled,
and alternate SIGReg variants are explicit blocked rows.

### Surprise

| Metric | Value |
| --- | ---: |
| Example count | 1,000 |
| Pairwise AUC overall | 0.746553 |
| Recall@1 | 0.495 |
| Mean true rank | 1.533 |
| Median true rank | 2 |

Decoy coverage: random `1000`, mutation `1000`, same-file `63`,
action-cluster `40`.

### Scorer And Reranker

The scorer-quality report has one labeled example, four candidates, two valid
candidates, and two expected error candidates. It records Recall@1 `0.0`, MRR
`0.5`, mean true rank `2.0`, and failure counts `invalid_syntax=1`,
`patch_apply_failed=1`.

`codelewm score` and `codelewm rerank` both ran from the downloaded checkpoint
and downloaded transition index with retrieval-prior weight `1.0` and k `10`.
The fixture true-after candidate scored `25.630047`; rerank placed the hard
negative before the true-after candidate, so this remains a smoke check.

## Verification

Downloaded checkpoint path:
`.artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/checkpoints/checkpoint.pt`.

Verified command:

```bash
uv run codelewm manifest verify \
  --manifest .artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/manifest.json \
  --parent-manifest .artifacts/hf-download/codelewm-action-use-20260520-6650183/dataset/runs/codelewm-action-use-20260520-6650183/pack/manifest.json \
  --json
```

Result: `ok=true`, parent `dataset-67895f8dc3e217c4`. Secret scan over the
downloaded artifact root returned `ok=true` with zero findings.

## Intended Use

- Reproduce #154 action-use margin evaluation from a trusted checkpoint.
- Diagnose no-action dominance against text-action retrieval.
- Compare against the completed #159 margin+retrieval remediation run.

## Out-of-Scope Use

- Generating or executing code.
- Treating this checkpoint as a public positive-quality release model.
- Claiming action-conditioned superiority over no-action baselines.
- Loading without `checkpoint.pt.manifest.json` trust verification.

## Limitations

- No-action is stronger than text action on headline retrieval.
- The no-action margin auxiliary did not improve the headline claim gate enough.
- Scorer-quality has one labeled example and failed to rank the fixture true
  after-state first.
- Repositories remain private; #159 also failed the positive action-use claim
  gate.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Model owner | @AbdelStark | 2026-05-20 |
