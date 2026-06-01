# CodeLeWM v0.6 Execution Model Card — Seed 42

- Model name: `codelewm-v0-6-execution-seed-42`
- HF checkpoint surface:
  `abdelstark/codelewm-runs/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42/checkpoints/last.pt`
- Run artifact repo path:
  `abdelstark/codelewm-runs/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42`
- Training artifact ID: `training_run-cb62408f881eff8c`
- Checkpoint path: `checkpoints/last.pt`
- Checkpoint SHA-256:
  `c4aaae5ad5ebae7a32c7a79520496a2ca039b3a1b9a4a6d3c6199f1274fc2b20`
- Source git SHA: `af1a114`
- Dataset: `abdelstark/codelewm-execution-pack@v0.6.0`
- Dataset artifact ID: `codelewm-execution-pack-20260528T102625Z`
- Release status: public v0.6 partial-positive research artifact; the
  `codelewm-runs` path is the canonical public checkpoint surface, while final
  HF README refresh waits for the #306 arXiv URL.
- Card date: `2026-05-31`

## Summary

This card is the repo-side model card for the seed-42 checkpoint. The
checkpoint is published as part of the public `abdelstark/codelewm-runs`
artifact tree rather than as a `codelewm-transition-model` v0.6 tag.

This checkpoint is a JEPA-style latent transition model trained on deterministic
Python execution traces. It consumes candidate program text plus an execution
input representation and predicts the output latent. It is intended for
diagnostic scoring and retrieval experiments, not for code generation.

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
| Seed | 42 |
| Steps | 50,000 |
| Pack records | 1,605 |
| Optimizer | AdamW, lr `3e-4`, weight decay `0.1` |
| Batch / accumulation | 64 / 4 |
| Objective | MSE + SIGReg + action-swap contrastive + inverse-action reconstruction |

## Training Metrics

| Metric | Value |
| --- | ---: |
| `loss_total` | 0.0080448 |
| `loss_prediction_mse` | 0.0006388 |
| `loss_sigreg` | 0.0364093 |
| `loss_action_swap_contrastive` | 0.0001802 |
| `margin_no_action_minus_pred` | +1.2307509 |
| `no_action_mse` | 1.2313897 |
| Predicted effective rank | 119.5278 |
| Predicted effective rank ratio | 0.4669 |
| Mean predicted latent norm | 14.9692 |
| Mean pairwise cosine | 0.000329 |

## Evaluation Evidence

| Surface | Result |
| --- | --- |
| Execution-pack retrieval | Recall@1 `0.6568`, Recall@5 `0.9025`, Recall@10 `0.9703`, MRR `0.7670` |
| No-action retrieval control | Recall@1 `0.0381`, MRR `0.1042` |
| Semantic-decoy surprise score gates | Recall@1 `1.0000`, pairwise AUC `1.0000`; mutation `236` pairs, same-code/different-input `352` pairs, same-problem/different-submission `6` pairs |
| Semantic-decoy claim gate | Closed: same-problem/different-submission count is `6/30` |
| Latent probe | Claim blocked: only `output_type` evaluable and lexical control is stronger |
| Crash prediction | Not evaluable: zero crash-positive val/test rows |
| Downstream HumanEval / MBPP-Plus rerank | Not run as a scaled 100-example benchmark |

The narrow execution-pack retrieval gate passes for this seed. Semantic-decoy
score diagnostics pass, but broad semantic surprise, coding-usefulness, and
semantic-axis gates remain closed.

## Intended Use

- Reproduce the v0.6 execution-pack retrieval and semantic-decoy surprise score
  diagnostics.
- Run `codelewm score` or `codelewm rerank` as a diagnostic scorer for
  candidate programs and input representations.
- Inspect the substrate-pivot comparison against the v0.2 commit-edit result.

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
  --local-dir /tmp/codelewm-v0-6-seed-42 \
  --include 'runs/codelewm-v0-6-execution-20260530-af1a114-seed-42/**'
uv run codelewm manifest verify \
  --manifest /tmp/codelewm-v0-6-seed-42/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42/manifest.json \
  --parent-manifest /tmp/codelewm-execution-pack-v0-6-0/artifact_manifest.json \
  --json
uv run codelewm secret-scan /tmp/codelewm-v0-6-seed-42 --json
```

Expected result: manifest verification `ok=true`; secret scan `ok=true` with
zero findings.

## Limitations

- This is positive evidence for the execution-pack retrieval substrate, not for
  general generated-code utility.
- Semantic-decoy surprise has a stronger same-code/different-input slice
  (`n=352`), but the same-problem/different-submission slice remains too small
  for a broad semantic claim (`n=6`, gate requires `30`).
- The live tour in `docs/demo/` had no pass@1 lift because all sampled first
  completions passed.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Model owner | @AbdelStark | 2026-05-31 |
