# CodeLeWM WS-A / A1.5 Full-Latent Correctness Probe (2026-06-04)

A1 showed the model's *scalar* scores decode correctness at chance. But A3's
head reads the **full latent vectors**, which a scalar energy could collapse
away. Before committing to the expensive A3 build (data pipeline + schema
bumps + GPU training), A1.5 tests A3's make-or-break assumption directly and
cheaply: **do the full v0.7 latents linearly encode correctness?**

## Method

For every WS-D candidate (282 completions, 47 problems, 1-correct-among-6),
encode through the v0.7 short checkpoint via the execution scoring backend and
extract the full 256-dim vectors `z_code = encode_state(candidate)` and
`z_pred_after = predict_after(z_code, encode_action(input_repr))`. Fit a
grouped 5-fold (by problem) logistic probe to the sandbox `passed` label and
report out-of-fold ROC-AUC. Probe code: `docs/benchmark/wsa/latent_probe.py`.

## Result (both seeds)

| latent view | seed 42 AUC | seed 1729 AUC |
|-------------|-------------|---------------|
| z_code | 0.538 | 0.514 |
| z_pred_after | 0.474 | 0.481 |
| z_pred − z_code | 0.524 | 0.494 |
| concat(z_code, z_pred) | 0.511 | 0.504 |

Every view is at chance on both seeds (0.47–0.54).

## Verdict — the SSL representation does not encode correctness

Correctness is not recoverable from the v0.7 latents at all — not the scalar
energy (A1: 0.500) and not the full 256-dim vectors (A1.5: 0.47–0.54). The
information is simply not in the representation.

**This rules out A3 as a cheap second stage on the frozen trunk.** A `p_pass`
head reading these latents cannot learn what the latents do not contain; a
frozen-trunk A3 run would burn the data build + GPU only to land at chance.

The correctness signal must instead be put **into the trunk during
training**:

- **A3 co-trained from scratch** — the SSL objective + the BCE `p_pass` term
  trained jointly on pass/fail-labeled data, so the trunk learns a
  correctness-bearing representation. Cost: the pass/fail data build (re-exec
  the mutation completions to recover outputs + tokenize + split) AND a full
  GPU pretraining run, not a cheap second stage.
- **WS-C architecture first** — the chance-level latent suggests the current
  encoder/objective may be the bottleneck (no mechanism rewards encoding
  "is this output correct for the spec"); WS-C levers (EMA target, output-
  value aux head C4) + co-training may be prerequisite.

Either way, the cheap end of WS-A (A1 calibrator, A2 prior, A3 frozen-trunk
head) is closed. A1.5 — a few minutes of CPU — saved the entire A3
frozen-trunk data-build + GPU spend by proving up front it would not work.

## Why this is plausible (not a bug)

The substrate is `(code, input) → output` with a self-supervised next-state
objective. Nothing in training rewards the latent for encoding *whether the
produced output is the one the problem specifies* — the model never sees the
expected output / spec as a target. So "correct vs incorrect" is genuinely
outside what the current representation is trained to capture. This also
explains the WS-D result (rerank at chance) and points the roadmap at
injecting a correctness/spec signal at training time, not at read-out.
