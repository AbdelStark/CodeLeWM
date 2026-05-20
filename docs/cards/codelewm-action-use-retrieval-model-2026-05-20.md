# CodeLeWM Action-Use Retrieval Model Card

- Model name: `codelewm-action-use-margin-retrieval-gpu-a10g-text-action`
- Run ID: `codelewm-action-use-retrieval-20260520-7895d18`
- Training artifact id: `training_run-924cd056375f11ea`
- Checkpoint SHA-256: `0cb4daf1500495579f5c59cc9fd8aa39f5f70e88f55c0c121320d023b43ddeda`
- Model repo: `abdelstark/codelewm-transition-model`
- Model repo path: `checkpoints/codelewm-action-use-retrieval-20260520-7895d18`
- Dataset artifact id: `dataset-5695087296ce4a97`
- Source git SHA: `7895d185e165a917af0956a313d8948c04b33638`
- Benchmark report: `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md`
- Release status: public negative action-use remediation evidence, not public positive quality release
- Card date: `2026-05-20`

## Summary

This card describes the #159 CodeLeWM checkpoint trained on Hugging Face Jobs
with the no-action margin objective plus retrieval loss. The checkpoint was
published to Hugging Face, downloaded with `hf download`, verified locally with its
parent dataset artifact, and used for downloaded-artifact retrieval, ablation,
surprise, scorer-quality, score, and rerank checks.

The run improves over the #154 margin-only run, but it is still not sufficient
for a public positive action-conditioning claim. Text-action beats random,
shuffled-action, and lexical baselines, but it still loses to no-action on
Recall@1 and MRR.

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
| Config | `config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml` |
| Executor | `torch` |
| Device | `cuda` |
| Precision | `bf16-mixed` |
| Torch | `2.12.0+cu130` |
| Seed | 240119 |
| Steps | 60,000 |
| Train rows | 18,019 |
| Validation rows | 1,291 |
| Optimizer | AdamW |
| Objective | MSE + SIGReg + no-action margin + retrieval loss |
| Action-use margin weight | 0.25 |
| Action-use margin | 0.02 |
| Retrieval loss weight | 0.05 |
| Retrieval temperature | 0.1 |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss/total` | 0.100775 |
| `loss/prediction_mse` | 0.006972 |
| `loss/action_use_margin` | 0.005475 |
| `loss/retrieval` | 0.208046 |
| `val/loss/total` | 0.412853 |
| `val/loss/action_use_margin` | 0.051669 |
| `val/loss/retrieval` | 0.915677 |
| `collapse/effective_rank` | 10.542142 |
| `collapse/effective_rank_ratio` | 0.041180 |
| `train/examples_per_second` | 1064.506 |

## Evaluation

### Retrieval

Remote headline retrieval, 1,000 held-out queries with 1,000 candidates:

| Variant | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Text action | 0.597 | 0.770 | 0.813 | 0.674500 |
| Random | 0.001 | 0.004 | 0.008 | 0.007118 |
| Shuffled action | 0.000 | 0.003 | 0.010 | 0.006567 |
| Lexical | 0.045 | 0.130 | 0.190 | 0.093745 |
| No action | 0.650 | 0.774 | 0.816 | 0.708037 |

Interpretation: retrieval loss improved text-action metrics, but no-action
dominance remains. The action-use claim gate is `claim_allowed=false`.

### Action Ablation

| Field | Value |
| --- | ---: |
| Completed rows | 7 |
| Blocked rows | 5 |
| Failed rows | 0 |

Unavailable abstract-action, patch-action diagnostic, retrieval-loss-disabled,
and alternate SIGReg variants are explicit blocked rows.

### Surprise

| Metric | Value |
| --- | ---: |
| Example count | 1,000 |
| Pairwise AUC overall | 0.757965 |
| Recall@1 | 0.511 |
| Mean true rank | 1.509 |
| Median true rank | 1 |

Decoy coverage: random `1000`, mutation `1000`, same-file `63`,
action-cluster `40`.

### Scorer And Reranker

The scorer-quality report has one labeled example, four candidates, two valid
candidates, and two expected error candidates. It records Recall@1 `0.0`, MRR
`0.5`, mean true rank `2.0`, and failure counts `invalid_syntax=1`,
`patch_apply_failed=1`.

`codelewm score` and `codelewm rerank` both ran from the downloaded checkpoint
and downloaded transition index with retrieval-prior weight `1.0` and k `10`.
The public CLI reports the deterministic lightweight scorer backend plus
retrieval prior; this is a harness smoke check, not proof of calibrated neural
patch scoring.

## Verification

Downloaded checkpoint path:
`.artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/checkpoints/checkpoint.pt`.

Verified command:

```bash
uv run codelewm manifest verify \
  --manifest .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/manifest.json \
  --parent-manifest .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/dataset/runs/codelewm-action-use-retrieval-20260520-7895d18/pack/manifest.json \
  --json
```

Result: `ok=true`, parent `dataset-5695087296ce4a97`. Secret scan over the
downloaded artifact root returned `ok=true` with zero findings.

## Intended Use

- Reproduce #159 margin+retrieval evaluation from a trusted checkpoint.
- Diagnose no-action dominance against text-action retrieval.
- Compare future research iterations against the strongest current negative
  action-use run.

## Out-of-Scope Use

- Generating or executing code.
- Treating this checkpoint as a public positive-quality release model.
- Claiming action-conditioned superiority over no-action baselines.
- Loading without `checkpoint.pt.manifest.json` trust verification.

## Limitations

- No-action is still stronger than text action on headline retrieval.
- Scorer-quality has one labeled example and failed to rank the fixture true
  after-state first.
- The public `score` and `rerank` commands currently report the deterministic
  lightweight scorer backend plus retrieval prior.
- Repositories are public diagnostic artifact repositories; positive
  model-quality claims remain blocked by the no-action baseline result.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Model owner | @AbdelStark | 2026-05-20 |
