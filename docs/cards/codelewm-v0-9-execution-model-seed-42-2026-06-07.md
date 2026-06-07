# CodeLeWM v0.9 Execution Model Card - Seed 42

- Model name: `codelewm-v0-9-execution-seed-42`
- Run artifact repo path:
  `abdelstark/codelewm-runs/codelewm-v0-9-short-execution-20260606-69f798a-seed-42`
- Training artifact ID: `training_run-992f7757f2780da4`
- Checkpoint path: `checkpoints/checkpoint_step_00012000.pt`
- Checkpoint SHA-256:
  `c783fa0dbe5da6bd072ff0b2f2753bdbac9fe684b49bf82e70ab6a2f69d513da`
- Checkpoint inspection artifact: `eval_report-54b696c9cd038493`
- Source git SHA: `69f798a620fdf8e50c1773228428e80ffee1a6ef`
- Dataset: `abdelstark/codelewm-execution-pack@v0.9.0-rc1`
- Dataset artifact ID: `codelewm-passfail-execution-pack-20260606T122240Z`
- Release status: public v0.9 diagnostic artifact; HumanEval WS-D positive,
  overall downstream claim closed.
- Card date: `2026-06-07`

## Summary

This checkpoint is a correctness-aware execution transition model. It consumes
tokenized code and action text, uses a transformer state encoder, predicts the
after-output latent, and includes supervised `p_pass` and output-value heads.
It is intended for diagnostic scoring and reranking experiments, not for code
generation or unchecked checkpoint loading.

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
| Auxiliary heads | `p_pass`, output value |
| Parameters | 44,978,967 |

## Training

| Field | Value |
| --- | --- |
| Config | `codelewm_execution_v0_9_short_a10g` |
| HF job | `6a241a07ece949d7b3dca989` |
| Device | CUDA |
| Precision | `bf16-mixed` |
| Torch | `2.6.0+cu124` |
| Seed | 42 |
| Steps | 12,000 |
| Pack records | 2,188 |
| Batch / accumulation | 64 / 4 |
| Objective | MSE + SIGReg + retrieval + action-swap contrastive + inverse-action reconstruction + `p_pass` BCE + output-value CE |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss_total` | 0.1139949 |
| `loss_prediction_mse` | 0.0097263 |
| `loss_p_pass_bce` | 0.000000121 |
| `loss_output_value_ce` | 0.0001002 |
| `loss_sigreg` | 0.1430375 |
| `loss_action_swap_contrastive` | 0.0020818 |
| `margin_no_action_minus_pred` | +1.3184099 |
| `no_action_mse` | 1.3281362 |
| Predicted effective-rank ratio | 0.3883 |
| Mean predicted latent norm | 15.2112 |
| Mean pairwise cosine | 0.002485 |

## Evaluation Evidence

| Surface | Result |
| --- | --- |
| Execution-pack retrieval | Recall@1 `0.2538`, Recall@5 `0.5115`, Recall@10 `0.6038`, MRR `0.3804` |
| No-action retrieval control | Recall@1 `0.0115`, MRR `0.0604` |
| Semantic-decoy surprise | Mutation pairwise AUC `1.0000`; broad semantic gate closed by zero scorable same-problem/same-code pairs |
| Latent probe | `passed` z_pred_after test accuracy `0.6552`, weaker than no-action/before-state control `0.6897`; `output_magnitude_bucket` tied with `z_after` at `0.5000` |
| HumanEval WS-D rerank | codelewm pass@1 `0.9787`; no_action `0.8723`; lift CI `[2.13, 21.28]`; claim allowed for this benchmark |
| MBPP-Plus WS-D rerank | codelewm pass@1 `1.0000`; no_action `1.0000`; lexical `1.0000`; claim closed |
| Downstream calibration | codelewm ROC-AUC `0.9352`; HumanEval `0.9196`; MBPP-Plus `0.9730`; `p_pass` score key missing in score rows |

The overall v0.9 claim remains closed because cross-benchmark downstream lift
over controls and representation gates do not jointly pass.

## Intended Use

- Reproduce the v0.9 HumanEval WS-D rerank diagnostic.
- Compare cross-benchmark correctness-aware scoring against no-action,
  shuffled-action, lexical, random, and LLM-order controls.
- Inspect non-collapse, retrieval, surprise, calibration, and probe behavior.

## Out-of-Scope Use

- Generating code.
- Claiming general HumanEval / MBPP-Plus benchmark improvement.
- Claiming robust named semantic latent dimensions.
- Treating `loss_p_pass_bce` as standalone benchmark evidence.
- Loading the checkpoint without trust-gate and manifest verification.

## Verification

```bash
hf download abdelstark/codelewm-runs \
  --repo-type dataset \
  --include 'codelewm-v0-9-short-execution-20260606-69f798a-seed-42/**' \
  --local-dir .artifacts/v0_9/hf-runs-download
uv run codelewm manifest verify \
  --manifest .artifacts/v0_9/hf-runs-download/codelewm-v0-9-short-execution-20260606-69f798a-seed-42/manifest.json \
  --parent-manifest .artifacts/v0_9/hf-pack-download/artifact_manifest.json \
  --json
uv run codelewm secret-scan .artifacts/v0_9/hf-runs-download/codelewm-v0-9-short-execution-20260606-69f798a-seed-42 --json
```

Expected result: manifest verification `ok=true`; secret scan `ok=true` with
zero findings.

## Limitations

- MBPP-Plus WS-D is saturated and gives zero lift over no-action and lexical.
- Broad semantic-decoy surprise has zero scorable semantic pairs against this
  pack.
- Frozen latent probes do not support a positive representation claim.
- Downstream score rows do not expose a distinct `p_pass` probability key.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Model owner | @AbdelStark | 2026-06-07 |
