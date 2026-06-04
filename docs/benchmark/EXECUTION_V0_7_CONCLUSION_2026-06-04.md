# CodeLeWM v0.7 — Consolidated Conclusion (2026-06-04)

This is the executive summary of the v0.7 cycle: what was built, what the
evidence says, and the concrete v0.8 plan. Detailed reports are linked per
section.

## One-paragraph verdict

v0.7 made the execution-substrate world model demonstrably better at
representing execution **structure** — and proved, with a purpose-built
unsaturated benchmark and direct latent probes, that it represents execution
**correctness** not at all. The structure wins are real and reproducible
across 2 seeds; the correctness gap is the binding limitation and defines
v0.8.

## What shipped (24 PRs this cycle)

- **v0.7 recipe + training** — transformer state encoder (WS-C1), in-batch
  InfoNCE (WS-C3), applied prediction-MSE (WS-C5); 2-seed A10G run on the
  bucket-augmented mbpp pack. The 50k-step full runs hit the 24h wall; a
  right-sized 15k "short" hedge delivered the results.
- **Eval pipeline unblocked** — three model-load sites couldn't load
  transformer checkpoints and the eval encoded whole splits at once (OOM);
  fixed (arch resolver + chunked encoding).
- **WS-D unsaturated rerank benchmark** — deterministic mutation-distractor
  packs + builder + leak-free baselines.
- **WS-A diagnosis** — feature-subset calibrator (A1) + full-latent probe
  (A1.5).

## The positives (reproduced, 2 seeds)

| signal | result |
|--------|--------|
| Representation collapse | none — eff-rank ratio 0.30–0.33, pairwise cosine ≈ 0 |
| Surprise (mutation + same-code-diff-input) | AUC **1.0** |
| Retrieval recall@1 lift over no-action | **+0.48 / +0.51** |
| Probe: output-magnitude decodable | z_pred **0.75 / 0.63** vs every control 0.54 / 0.49 (**+0.21 / +0.15**) — newly measurable vs v0.6 |

See `EXECUTION_V0_7_RESULTS_2026-06-04.md`.

## The binding limitation: no correctness signal

- **WS-D** (`EXECUTION_V0_7_WSD_RESULTS_2026-06-04.md`): on a benchmark where
  every problem has a pass/fail mix (random pass@1 = 1/6), v0.7 reranks at
  **chance** (codelewm pass@1 0.06–0.17), far below a trivial lexical baseline
  (0.30–0.66). So the flat downstream rerank is not only benchmark saturation.
- **WS-A / A1** (`EXECUTION_V0_7_WSA_A1_RESULTS_2026-06-04.md`): the model's
  scalar scores decode `passed` at chance (codelewm univariate AUC 0.500).
- **WS-A / A1.5** (`EXECUTION_V0_7_WSA_A15_LATENT_PROBE_2026-06-04.md`): the
  **full 256-dim latents** decode `passed` at chance too (0.47–0.54, both
  seeds). The signal isn't there to read.

**Root cause.** The `(code, input) → output` self-supervised objective never
sees the problem's expected output / spec, so "is this output *correct*" is
outside what the representation is trained to capture. This single fact
explains WS-D and A1/A1.5 together.

## What we can / cannot claim

- **Can claim (2-seed, evidence-backed):** a non-collapsed execution-transition
  latent that separates true outcomes from decoys (surprise AUC 1.0),
  retrieves the true after-state far above baselines, and predicts output
  magnitude above every control.
- **Cannot claim:** any correctness-reranking ability, or a downstream
  pass@1 lift. The RFC-0014 downstream gate stays closed.

## v0.8 plan (correctness co-trained into the trunk)

The cheap read-out fixes (A1 calibrator, A2 prior, frozen-trunk A3) are ruled
out by A1.5. v0.8 injects correctness at training time:

1. **Pass/fail data build** — `completion_label.v1 → pack` adapter
   (re-execute WS-D mutation completions to recover outputs, tokenize, split,
   write `passed`) + `record.py`/loader/executor schema bumps.
2. **A3 co-trained from scratch** — `p_pass` BCE head trained jointly with the
   SSL objective (head `torch_transition.py:97-106`, BCE
   `objective.py:217-229`, fusion `scorer.py:614-653`).
3. **WS-C2/C4 enablers** — EMA target encoder + output-value auxiliary head to
   force the latent to encode output/spec semantics.
4. **Measure on WS-D** — success = calibrated p_pass reranks above the lexical
   baseline (~0.30) with bootstrap-CI clearance.

The full rationale and verified change points are in RFC-0015 §"Outcomes and
v0.8 Direction".
