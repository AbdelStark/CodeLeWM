# CodeLeWM v0.9 Execution Model Card - Seed 1729

- Model name: `codelewm-v0-9-execution-seed-1729`
- Run artifact repo path:
  `abdelstark/codelewm-runs/codelewm-v0-9-short-execution-20260606-69f798a-seed-1729`
- Training artifact ID: `training_run-91e9cf7c645379b3`
- Checkpoint path: `checkpoints/checkpoint_step_00012000.pt`
- Checkpoint SHA-256:
  `34ebb282b284580dd123c781ae77c93cc36bbffc4eeeee9f0bd4cdf8042001eb`
- Checkpoint inspection artifact: `eval_report-5c9de2e6f492809c`
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
| HF job | `6a241a07368e0b5dc8064efb` |
| Device | CUDA |
| Precision | `bf16-mixed` |
| Torch | `2.6.0+cu124` |
| Seed | 1729 |
| Steps | 12,000 |
| Pack records | 2,188 |
| Batch / accumulation | 64 / 4 |
| Objective | MSE + SIGReg + retrieval + action-swap contrastive + inverse-action reconstruction + `p_pass` BCE + output-value CE |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss_total` | 0.1152192 |
| `loss_prediction_mse` | 0.0106948 |
| `loss_p_pass_bce` | 0.0017163 |
| `loss_output_value_ce` | 0.0001082 |
| `loss_sigreg` | 0.1534895 |
| `loss_action_swap_contrastive` | 0.0020599 |
| `margin_no_action_minus_pred` | +1.2773338 |
| `no_action_mse` | 1.2880286 |
| Predicted effective-rank ratio | 0.4056 |
| Mean predicted latent norm | 15.1381 |
| Mean pairwise cosine | 0.002525 |

## Evaluation Evidence

| Surface | Result |
| --- | --- |
| Execution-pack retrieval | Recall@1 `0.2731`, Recall@5 `0.5346`, Recall@10 `0.5923`, MRR `0.3982` |
| No-action retrieval control | Recall@1 `0.0192`, MRR `0.0725` |
| Semantic-decoy surprise | Mutation pairwise AUC `1.0000`; broad semantic gate closed by zero scorable same-problem/same-code pairs |
| Latent probe | `passed` z_pred_after test accuracy `0.6158`, weaker than no-action/before-state control `0.7980`; `output_magnitude_bucket` tied with `z_after` at `0.8421` |
| HumanEval WS-D rerank | codelewm pass@1 `0.9787`; no_action `0.8936`; lift CI `[2.13, 17.02]`; claim allowed for this benchmark |
| MBPP-Plus WS-D rerank | codelewm pass@1 `1.0000`; no_action `1.0000`; lexical `1.0000`; claim closed |
| Downstream calibration | codelewm ROC-AUC `0.9746`; HumanEval `0.9717`; MBPP-Plus `0.9945`; `p_pass` score key missing in score rows |

The overall v0.9 claim remains closed because cross-benchmark downstream lift
over controls and representation gates do not jointly pass.

## Intended Use

- Reproduce the v0.9 HumanEval WS-D rerank diagnostic.
- Compare cross-benchmark correctness-aware scoring against no-action,
  shuffled-action, lexical, random, and LLM-order controls.
- Inspect non-collapse, retrieval, surprise, calibration, and probe behavior.
- Reproduce the final v1.0 paper-demo replay documented in
  `docs/cards/codelewm-v1-0-paper-demo-2026-06-08.md`,
  `docs/cards/codelewm-v1-0-final-release-2026-06-08.md`, and
  `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md`.

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
  --include 'codelewm-v0-9-short-execution-20260606-69f798a-seed-1729/**' \
  --local-dir .artifacts/v0_9/hf-runs-download
uv run codelewm manifest verify \
  --manifest .artifacts/v0_9/hf-runs-download/codelewm-v0-9-short-execution-20260606-69f798a-seed-1729/manifest.json \
  --parent-manifest .artifacts/v0_9/hf-pack-download/artifact_manifest.json \
  --json
uv run codelewm secret-scan .artifacts/v0_9/hf-runs-download/codelewm-v0-9-short-execution-20260606-69f798a-seed-1729 --json
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
