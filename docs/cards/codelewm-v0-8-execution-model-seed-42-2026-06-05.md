# CodeLeWM v0.8 Execution Model Card - Seed 42

- Model name: `codelewm-v0-8-execution-seed-42`
- HF checkpoint surface:
  `abdelstark/codelewm-transition-model/checkpoints/codelewm-v0-8-short-execution-20260605-1b737e4-seed-42/checkpoints/last.pt`
- Run artifact repo path:
  `abdelstark/codelewm-runs/codelewm-v0-8-short-execution-20260605-1b737e4-seed-42`
- Training artifact ID: `training_run-e2a757caf75cbcf2`
- Checkpoint path: `checkpoints/last.pt`
- Checkpoint SHA-256:
  `03707bc09d1e60d74bdd94d649f4179632ea57fc06b6f0640d05083664aa7136`
- Source git SHA: `1b737e4`
- Dataset: `abdelstark/codelewm-execution-pack@v0.8.0-rc1`
- Dataset artifact ID: `codelewm-passfail-execution-pack-20260604T185716Z`
- Release status: public v0.8 diagnostic artifact; HumanEval WS-D positive,
  overall downstream claim closed.
- Card date: `2026-06-05`

## Summary

This checkpoint is a correctness-aware JEPA-style execution transition model.
It consumes tokenized code and action text, uses a transformer state encoder,
predicts the after-output latent, and adds supervised `p_pass` and output-value
heads. It is intended for diagnostic scoring and reranking experiments, not for
code generation or unchecked checkpoint loading.

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
| Seed | 42 |
| Steps | 12,000 |
| Pack records | 1,882 |
| Optimizer | AdamW, lr `3e-4`, weight decay `0.1` |
| Batch / accumulation | 64 / 4 |
| Objective | MSE + SIGReg + retrieval + action-swap contrastive + inverse-action reconstruction + `p_pass` BCE + output-value CE |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss_total` | 0.1159468 |
| `loss_prediction_mse` | 0.0076881 |
| `loss_p_pass_bce` | 0.000000013 |
| `loss_output_value_ce` | 0.0000118 |
| `loss_sigreg` | 0.1553529 |
| `loss_action_swap_contrastive` | 0.0012013 |
| `margin_no_action_minus_pred` | +1.3264009 |
| `no_action_mse` | 1.3340891 |
| Predicted effective rank | 98.3859 |
| Predicted effective rank ratio | 0.3843 |
| Mean predicted latent norm | 15.2819 |
| Mean pairwise cosine | 0.002551 |

## Evaluation Evidence

| Surface | Result |
| --- | --- |
| Execution-pack retrieval | Recall@1 `0.2754`, Recall@5 `0.5593`, Recall@10 `0.6568`, MRR `0.4125` |
| No-action retrieval control | Recall@1 `0.0085`, MRR `0.0497` |
| Semantic-decoy surprise | Mutation pairwise AUC `1.0000`, Recall@1 `1.0000`; broad semantic gate closed because aligned same-problem/same-code pair counts are zero |
| Latent probe | `passed` z_pred_after test accuracy `0.5081`; best control `0.6216`; `output_magnitude_bucket` not evaluable |
| HumanEval WS-D rerank | codelewm pass@1 `0.9787`; no_action `0.8723`; lift CI `[2.13, 19.15]`; claim allowed |
| MBPP-Plus WS-D rerank | codelewm pass@1 `0.3529`; lexical `1.0000`; claim closed |
| Completion-level ROC-AUC diagnostic | HumanEval `0.9622`; MBPP-Plus `0.5772` |

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
  --include 'checkpoints/codelewm-v0-8-short-execution-20260605-1b737e4-seed-42/**'
uv run codelewm manifest verify \
  --manifest /tmp/codelewm-v0-8-model/checkpoints/codelewm-v0-8-short-execution-20260605-1b737e4-seed-42/manifest.json \
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
