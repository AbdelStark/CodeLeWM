# CodeLeWM v0.6 Execution Model Card — Seed 1729

- Model name: `codelewm-v0-6-execution-seed-1729`
- HF model repo target: `abdelstark/codelewm-transition-model@v0.6.0-seed-1729`
- Run artifact repo path:
  `abdelstark/codelewm-runs/runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729`
- Training artifact ID: `training_run-d0b59108447c9c4a`
- Checkpoint path: `checkpoints/last.pt`
- Checkpoint SHA-256:
  `b453cb1f70868301dc75d94ad08c40fccc8a94147c460db4e0563f406f752de8`
- Source git SHA: `af1a114`
- Dataset: `abdelstark/codelewm-execution-pack@v0.6.0`
- Dataset artifact ID: `codelewm-execution-pack-20260528T102625Z`
- Release status: public v0.6 partial-positive research artifact; HF card
  publication waits for the #306 arXiv URL.
- Card date: `2026-05-31`

## Summary

This checkpoint is the second-seed v0.6 execution-substrate model. It uses the
same JEPA-style architecture and training recipe as the seed-42 checkpoint and
exists to validate that the substrate-pivot result is not a single-seed artifact.

## Architecture

| Field | Value |
| --- | --- |
| Model class | `TorchCodeTransitionModel` |
| Substrate | `execution_trace_v1` |
| Action view | `text` |
| Action fusion | conditional transformer |
| Latent dimension | 256 |
| State sequence length | 1,024 |
| Action sequence length | 256 |
| Output sequence length | 256 |
| History size | 1 |
| Prediction horizon | 1 |

## Training

| Field | Value |
| --- | --- |
| Config | `codelewm_execution_v0_6_a10g` |
| HF Jobs flavor | `a10g-small` |
| Device | CUDA |
| Precision | `bf16-mixed` |
| Torch | `2.6.0+cu124` |
| Seed | 1729 |
| Steps | 50,000 |
| Pack records | 1,605 |
| Optimizer | AdamW, lr `3e-4`, weight decay `0.1` |
| Batch / accumulation | 64 / 4 |
| Objective | MSE + SIGReg + action-swap contrastive + inverse-action reconstruction |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss_total` | 0.0072837 |
| `loss_prediction_mse` | 0.0006005 |
| `loss_sigreg` | 0.0348204 |
| `loss_action_swap_contrastive` | 0.0001183 |
| `margin_no_action_minus_pred` | +1.2433804 |
| `no_action_mse` | 1.2439808 |
| Predicted effective rank | 122.0498 |
| Predicted effective rank ratio | 0.4768 |
| Mean predicted latent norm | 14.9432 |
| Mean pairwise cosine | 0.000119 |

## Evaluation Evidence

| Surface | Result |
| --- | --- |
| Execution-pack retrieval | Recall@1 `0.6483`, Recall@5 `0.8941`, Recall@10 `0.9703`, MRR `0.7587` |
| No-action retrieval control | Recall@1 `0.0381`, MRR `0.1040` |
| Generated-decoy surprise | Recall@1 `1.0000`, pairwise AUC `1.0000` |
| Latent probe | Claim blocked: only `output_type` evaluable and lexical control is stronger |
| Crash prediction | Not evaluable: zero crash-positive val/test rows |
| Downstream HumanEval / MBPP-Plus rerank | Not run as a scaled 100-example benchmark |

The seed-1729 results agree with seed 42: execution-pack retrieval and
generated-decoy surprise pass, while broader downstream utility remains
unsupported.

## Intended Use

- Reproduce the cross-seed v0.6 execution-pack retrieval and surprise evidence.
- Compare seed sensitivity against seed 42.
- Run diagnostic local scoring only after checkpoint trust-gate verification.

## Out-of-Scope Use

- Generating code.
- Claiming HumanEval / MBPP-Plus improvement.
- Claiming named semantic latent dimensions.
- Treating the demo tour as a scaled downstream benchmark.
- Loading the checkpoint without trust-gate and manifest verification.

## Verification

```bash
hf download abdelstark/codelewm-runs \
  --repo-type dataset \
  --local-dir /tmp/codelewm-v0-6-seed-1729 \
  --include 'runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729/**'
uv run codelewm manifest verify \
  --manifest /tmp/codelewm-v0-6-seed-1729/runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729/manifest.json \
  --parent-manifest /tmp/codelewm-execution-pack-v0-6-0/artifact_manifest.json \
  --json
uv run codelewm secret-scan /tmp/codelewm-v0-6-seed-1729 --json
```

Expected result: manifest verification `ok=true`; secret scan `ok=true` with
zero findings.

## Limitations

- The positive result is scoped to execution-pack retrieval and
  generated-decoy surprise.
- The downstream rerank utility gate remains unrun at the required scale.
- The arXiv URL is not yet available in-repo; final HF card publication waits
  for the operator upload from #306.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Model owner | @AbdelStark | 2026-05-31 |
