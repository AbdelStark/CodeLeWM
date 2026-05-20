# CodeLeWM Scaled Model Card

- Model name: `codelewm-scaled-gpu-a10g-text-action`
- Run ID: `codelewm-scaled-20260520-9699b53`
- Training artifact id: `training_run-d9074199c0d58911`
- Checkpoint SHA-256: `09bf8d3880ec272a858dd9b19f2b29622a66a5ebbef6dbd1f8e4ebeb8b6392b8`
- Model repo: `abdelstark/codelewm-transition-model`
- Model repo path: `checkpoints/codelewm-scaled-20260520-9699b53`
- Dataset artifact id: `dataset-ef8ad3f4f48dea9e`
- Source git SHA: `9699b5309e43a3278f272663ef60cda23040d92a`
- Benchmark report: `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md`
- Release status: public scaled diagnostic evidence, not public positive quality release
- Card date: `2026-05-20`

## Summary

This card describes the first scaled CodeLeWM checkpoint trained on Hugging Face
Jobs with the A10G profile. The checkpoint was published to Hugging Face, downloaded
with `hf download`, verified locally with its parent dataset artifact, and used
for downloaded-artifact retrieval, ablation, surprise, scorer-quality, score,
and rerank checks.

The run is useful scaled systems evidence. It is not sufficient for a public
positive action-conditioning claim because the no-action retrieval baseline
beats text-action on the headline metric.

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
| Config | `config/train/scaled/codelewm_scaled_gpu_a10g.yaml` |
| Executor | `torch` |
| Device | `cuda` |
| Precision | `bf16-mixed` |
| Torch | `2.12.0+cu130` |
| Seed | 240119 |
| Steps | 60,000 |
| Train rows | 18,019 |
| Validation rows | 1,291 |
| Optimizer | AdamW |
| Objective | MSE + SIGReg |
| Retrieval loss | disabled |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss/total` | 0.113565 |
| `loss/prediction_mse` | 0.005167 |
| `val/loss/total` | 0.355089 |
| `collapse/effective_rank` | 5.437365 |
| `collapse/effective_rank_ratio` | 0.021240 |
| `train/examples_per_second` | 1103.204 |

## Evaluation

### Retrieval

Remote headline retrieval, 1,000 held-out queries with 1,000 candidates:

| Variant | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Text action | 0.371 | 0.586 | 0.672 | 0.472984 |
| Random | 0.001 | 0.004 | 0.008 | 0.007118 |
| Shuffled action | 0.001 | 0.006 | 0.011 | 0.007518 |
| Lexical | 0.045 | 0.130 | 0.190 | 0.093745 |
| No action | 0.459 | 0.641 | 0.712 | 0.546116 |

Interpretation: text-action clearly beats random, shuffled-action, and lexical
baselines, but it does not beat the no-action baseline. This blocks a public
claim that the model has learned useful action-conditioned code-edit structure.

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
| Pairwise AUC overall | 0.755587 |
| Recall@1 | 0.514 |
| Mean true rank | 1.514 |
| Median true rank | 1 |

Decoy coverage: random `1000`, mutation `1000`, same-file `63`,
action-cluster `40`.

### Scorer And Reranker

The scorer-quality report has one labeled example, four candidates, two valid
candidates, and two expected error candidates. It records Recall@1 `1.0`, MRR
`1.0`, mean true rank `1.0`, and failure counts `invalid_syntax=1`,
`patch_apply_failed=1`.

`codelewm score` and `codelewm rerank` both ran from the downloaded checkpoint
and downloaded transition index with retrieval-prior weight `1.0` and k `10`.

## Verification

Downloaded checkpoint path:
`.artifacts/hf-download/codelewm-scaled-20260520-9699b53/model/checkpoints/codelewm-scaled-20260520-9699b53/checkpoints/checkpoint.pt`.

Verified command:

```bash
uv run codelewm manifest verify \
  --manifest .artifacts/hf-download/codelewm-scaled-20260520-9699b53/model/checkpoints/codelewm-scaled-20260520-9699b53/manifest.json \
  --parent-manifest .artifacts/hf-download/codelewm-scaled-20260520-9699b53/results/runs/codelewm-scaled-20260520-9699b53/pack/manifest.json \
  --json
```

Result: `ok=true`, files checked `6`, parent `dataset-ef8ad3f4f48dea9e`.
Secret scan over the downloaded model repo artifact returned `ok=true` with
zero findings.

## Intended Use

- Reproduce scaled CodeLeWM evaluation from a trusted checkpoint.
- Score and rerank candidate after-states with an index-backed retrieval prior.
- Diagnose whether future training changes improve action-use over the no-action
  baseline.

## Out-of-Scope Use

- Generating or executing code.
- Treating this checkpoint as a public release model.
- Claiming action-conditioned superiority over no-action baselines.
- Loading without `checkpoint.pt.manifest.json` trust verification.

## Limitations

- The no-action baseline is stronger than text action on headline retrieval.
- Scorer-quality has one labeled example and should be treated as a smoke
  quality gate, not a broad calibration benchmark.
- Same-file and action-cluster surprise decoy counts are lower than random and
  mutation decoys because of source-shard availability.
- Repositories are public diagnostic artifact repositories; positive
  model-quality claims remain blocked by the no-action baseline result.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Model owner | @AbdelStark | 2026-05-20 |
