# CodeLeWM v0.8 Execution Model Card - Seed 1729

- Model name: `codelewm-v0-8-execution-seed-1729`
- HF checkpoint surface:
  `abdelstark/codelewm-transition-model/checkpoints/codelewm-v0-8-short-execution-20260605-1b737e4-seed-1729/checkpoints/last.pt`
- Run artifact repo path:
  `abdelstark/codelewm-runs/codelewm-v0-8-short-execution-20260605-1b737e4-seed-1729`
- Training artifact ID: `training_run-951983cbf59f6fa6`
- Checkpoint path: `checkpoints/last.pt`
- Checkpoint SHA-256:
  `dafc3bf8022d1f3a9560b63f63c620873a2f2b3922e4f773a350e3f88c15ecfe`
- Source git SHA: `1b737e4`
- Dataset: `abdelstark/codelewm-execution-pack@v0.8.0-rc1`
- Dataset artifact ID: `codelewm-passfail-execution-pack-20260604T185716Z`
- Release status: public v0.8 diagnostic artifact; HumanEval WS-D positive,
  overall downstream claim closed.
- Card date: `2026-06-05`

## Summary

This checkpoint is the seed-1729 replicate for the v0.8 correctness-aware
execution transition model. It uses the same transformer execution substrate,
supervised `p_pass` head, and output-value head as seed 42. It is intended for
diagnostic scoring and reranking experiments, not for code generation or
unchecked checkpoint loading.

The execution-pack data artifact is the deterministic output of running
licensed public Python submissions in an isolated sandbox under a stdlib-only
policy at data-build time. The artifact contains no executable payload; it
contains tokenized code, tokenized inputs, tokenized outputs, and metadata.
Training and inference never execute code. The sandbox is reused only in the
dedicated downstream-evaluation scenario (`execution-rerank`) to label
completion correctness against hidden tests, and only on inputs the operator has
reviewed.

## Architecture

| Field | Value |
| --- | --- |
| Model class | `TorchCodeTransitionModel` |
| Substrate | `execution_trace_v1` |
| Action view | `text` |
| State encoder | transformer, 4 layers, 8 heads |
| Latent dimension | 256 |
| State sequence length | 1,024 |
| Action sequence length | 256 |
| Output sequence length | 256 |
| History size | 1 |
| Prediction horizon | 1 |
| Auxiliary heads | `p_pass`, output value |

## Training

| Field | Value |
| --- | --- |
| Config | `codelewm_execution_v0_8_short_a10g` |
| HF Jobs flavor | `a10g-small` |
| Device | CUDA |
| Precision | `bf16-mixed` |
| Torch | `2.6.0+cu124` |
| Seed | 1729 |
| Steps | 12,000 |
| Pack records | 1,882 |
| Optimizer | AdamW, lr `3e-4`, weight decay `0.1` |
| Batch / accumulation | 64 / 4 |
| Objective | MSE + SIGReg + retrieval + action-swap contrastive + inverse-action reconstruction + `p_pass` BCE + output-value CE |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss_total` | 0.1117948 |
| `loss_prediction_mse` | 0.0074480 |
| `loss_p_pass_bce` | 0.000000038 |
| `loss_output_value_ce` | 0.0000126 |
| `loss_sigreg` | 0.1303913 |
| `loss_action_swap_contrastive` | 0.0017883 |
| `margin_no_action_minus_pred` | +1.3870633 |
| `no_action_mse` | 1.3945113 |
| Predicted effective rank | 100.5493 |
| Predicted effective rank ratio | 0.3928 |
| Mean predicted latent norm | 15.1861 |
| Mean pairwise cosine | 0.002948 |

## Evaluation Evidence

| Surface | Result |
| --- | --- |
| Execution-pack retrieval | Recall@1 `0.2924`, Recall@5 `0.5678`, Recall@10 `0.6441`, MRR `0.4213` |
| No-action retrieval control | Recall@1 `0.0169`, MRR `0.0708` |
| Semantic-decoy surprise | Mutation pairwise AUC `1.0000`, Recall@1 `1.0000`; broad semantic gate closed because aligned same-problem/same-code pair counts are zero |
| Latent probe | `passed` z_pred_after test accuracy `0.5892`; best control `0.6486`; `output_magnitude_bucket` not evaluable |
| HumanEval WS-D rerank | codelewm pass@1 `0.9787`; no_action `0.8723`; lift CI `[2.13, 19.15]`; claim allowed |
| MBPP-Plus WS-D rerank | codelewm pass@1 `0.3529`; lexical `1.0000`; claim closed |
| Completion-level ROC-AUC diagnostic | HumanEval `0.9699`; MBPP-Plus `0.6941` |

HumanEval WS-D is positive for this seed. The overall v0.8 claim remains closed
because MBPP-Plus and representation-probe gates do not clear.

## Intended Use

- Reproduce the v0.8 HumanEval WS-D rerank diagnostic.
- Compare correctness-head reranking against v0.7 and no-action controls.
- Inspect non-collapse, retrieval, surprise, and pass-head behavior.

## Out-of-Scope Use

- Generating code.
- Claiming general HumanEval / MBPP-Plus benchmark improvement.
- Claiming robust named semantic latent dimensions.
- Treating `loss_p_pass_bce` as standalone benchmark evidence.
- Loading the checkpoint without trust-gate and manifest verification.

## Verification

```bash
hf download abdelstark/codelewm-transition-model \
  --local-dir /tmp/codelewm-v0-8-model \
  --include 'checkpoints/codelewm-v0-8-short-execution-20260605-1b737e4-seed-1729/**'
uv run codelewm manifest verify \
  --manifest /tmp/codelewm-v0-8-model/checkpoints/codelewm-v0-8-short-execution-20260605-1b737e4-seed-1729/manifest.json \
  --parent-manifest /tmp/codelewm-v0-8-pack/artifact_manifest.json \
  --json
uv run codelewm secret-scan /tmp/codelewm-v0-8-model --json
```

Expected result: manifest verification `ok=true`; secret scan `ok=true` with
zero findings.

## Limitations

- The supervised pass/fail pack is HumanEval-only.
- MBPP-Plus rerank does not clear the claim gate, and lexical is perfect on the
  current 17-problem MBPP-Plus pack.
- The frozen latent probe does not show a robust `passed` representation.
- `output_magnitude_bucket` is not evaluable because the v0.8 pack has zero val
  labels for that target after filtering.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Model owner | @AbdelStark | 2026-06-05 |
