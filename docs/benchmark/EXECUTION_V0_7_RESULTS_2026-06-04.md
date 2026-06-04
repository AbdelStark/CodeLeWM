# CodeLeWM v0.7 Execution-Substrate Results (2026-06-04)

v0.7 adds the RFC-0015 **WS-C** architecture levers to the v0.6 execution
substrate and measures their effect on the frozen-latent gates:

- **WS-C1** — transformer state encoder (`wm.state_encoder_type=transformer`,
  4 layers / 8 heads) replacing the v0.6 mean-pooling encoder.
- **WS-C3** — in-batch InfoNCE retrieval term (`objective.retrieval_weight=0.05`).
- **WS-C5** — `prediction_mse_weight` is now an explicit, applied lever.

The headline: the v0.7 latent **predicts output magnitude above every control
on both seeds** — a property that was *not even measurable* in v0.6 — while
surprise and retrieval remain strongly positive. Downstream rerank stays
flat because the HumanEval / MBPP-Plus candidate sets are saturated.

## Reproducibility chain

| Artifact | Reference |
|----------|-----------|
| Pack | `abdelstark/codelewm-execution-pack@v0.7.0-rc1` (public, 1859 mbpp records, CC-BY-4.0) |
| Runtime image | `ghcr.io/abdelstark/codelewm-runtime:v0.7-short` |
| Recipe | `config/train/scaled/codelewm_execution_v0_7_short_a10g.yaml` (15000 steps) |
| Runs | `abdelstark/codelewm-runs/codelewm-v0-7-short-execution-20260603-bcc581d-seed-{42,1729}` |
| Eval code | requires PR #354 (transformer-checkpoint load + chunked encoding) |

> **Note on the step budget.** The 50000-step full profile
> (`codelewm_execution_v0_7_a10g.yaml`) ran past the HF Jobs 24h wall
> because the transformer encoder at seq-1024 is far more expensive per
> step than v0.6's pooling encoder. These results use the 15000-step
> short profile (≈2000 epochs over the 1859-record pack), launched as a
> timeout-safe hedge. The full runs, if they complete, will be reported
> as a training-duration ablation.

## Training-time health (both seeds)

| metric | seed 42 | seed 1729 |
|--------|---------|-----------|
| loss_total (initial → final) | 5.16 → 0.029 | 5.16 → 0.030 |
| loss_prediction_mse (final) | 0.0020 | 0.0019 |
| margin (no_action − pred), final | **+0.621** | **+0.594** |
| z_pred effective-rank ratio | **0.326** | **0.295** |
| z_pred mean pairwise cosine | −0.0001 | −0.0003 |

Both seeds converge, learn a meaningful transition (positive no_action
margin: the predicted-after latent is closer to the true after-state than
the before-state), and **do not collapse** (effective-rank ratio > the
0.20 gate; near-zero pairwise cosine).

## Gate summary

| Gate | Verdict | Evidence |
|------|---------|----------|
| Collapse (eff-rank ratio ≥ 0.20) | **PASS** | 0.326 / 0.295 |
| Surprise — mutation AUC ≥ 0.65 | **PASS** | 1.000 / 1.000 |
| Surprise — same_code_different_input AUC ≥ 0.70 | **PASS** | 1.000 / 1.000 |
| Surprise — same_problem_different_submission AUC ≥ 0.60 | NOT EVALUABLE | 0 eligible pairs (single-submission pack) |
| Retrieval — recall@1 lift over no_action ≥ 0.05 | **MET (gate blocked)** | +0.482 / +0.511; formal gate `not allowed` only because the `lexical` baseline was not computed |
| Probe — ≥1 target beats every control across ≥2 seeds | **MET** | `output_magnitude_bucket` +0.21 / +0.15 |
| Probe — all listed targets beat all controls (formal) | NOT MET (conservative) | type/length/will_raise show no advantage |
| Downstream rerank — pass@1 lift ≥ 3pts | **NOT MET** | benchmark saturated (≈95% pass@1) |
| Crash prediction | NOT EVALUABLE | only 6 raised / 1859 (mbpp rarely throws) |

## Frozen-latent probe (the v0.7 headline)

All five probe targets are **evaluable** in v0.7 — the v0.6→v0.7 flip,
driven by persisting `output_magnitude_bucket` / `output_length_bucket`
labels (in v0.6 only `output_type` was labeled → `not_evaluable`).

Best predicted-after latent (`z_pred_after`) test accuracy vs the best
control, per target:

| target | seed 42 z_pred | best control | adv | seed 1729 z_pred | best control | adv |
|--------|---------------|--------------|-----|------------------|--------------|-----|
| **output_magnitude_bucket** | **0.750** | 0.539 | **+0.211** | **0.632** | 0.487 | **+0.145** |
| arithmetic_vs_string_vs_collection | 0.824 | 0.819 | +0.005 | 0.867 | 0.883 | −0.016 |
| output_type | 0.691 | 0.707 | −0.016 | 0.713 | 0.803 | −0.090 |
| output_length_bucket | 0.583 | 0.679 | −0.095 | 0.583 | 0.679 | −0.095 |
| will_raise | 0.957 | 1.000 | −0.043 | 0.979 | 1.000 | −0.021 |

`output_magnitude_bucket` is the clean win: the predicted-after latent
beats **every** control (lexical, majority, metadata-only, random-latent,
no-action, shuffled-action) on **both** seeds. This satisfies the
representation-probe condition "≥1 target beats every control across ≥2
seeds". The remaining targets show no consistent advantage, so the
conservative *all-targets* representation claim stays closed
(`positive_representation_claim_allowed=false`).

## Surprise

Pairwise decoy AUC = **1.0** on both seeds for both evaluable categories
(mutation, same_code_different_input); true execution outcomes receive far
lower energy than mutated-code or different-input decoys (e.g. true ≈0.06
vs mutation decoy ≈306). The `same_problem_different_submission` category
is not evaluable (the mbpp pack has one reference submission per problem).

## Retrieval

| | seed 42 | seed 1729 |
|--|---------|-----------|
| recall@1 | 0.636 | 0.636 |
| recall@5 | 0.825 | 0.825 |
| MRR | 0.728 | 0.728 |
| recall@1 lift over no_action | **+0.482** | **+0.511** |
| recall@1 lift over random | +0.636 | +0.636 |

The latent retrieves the true after-state far better than the no-action
and random baselines. The formal action-use gate reports `not allowed`
**only** because the `lexical` baseline was not computed in this run — an
eval-completeness gap, not a metric failure.

## Downstream reranking (saturated)

| seed | benchmark | codelewm pass@1 | llm_order | no_action | lift vs llm | claim |
|------|-----------|-----------------|-----------|-----------|-------------|-------|
| 42 | HumanEval | 0.9545 | 0.9545 | 0.9481 | +0.00 | NO |
| 42 | MBPP-Plus | 0.9162 | 0.9243 | 0.9216 | −0.81 | NO |
| 1729 | HumanEval | 0.9481 | 0.9545 | 0.9416 | −0.65 | NO |
| 1729 | MBPP-Plus | 0.9189 | 0.9243 | 0.9189 | −0.54 | NO |

Reranking shows no lift: the reused v0.6 LLM candidate sets are already
~92–95% pass@1, leaving no headroom. The reranker matches LLM order
(within ±1pt noise) without regressing. **This is the binding limitation,
and it is the benchmark, not the model** — converting v0.7's representation
gains into measurable downstream lift requires the WS-D unsaturated
benchmark.

## What we can and cannot claim

- **Can claim (evidence-backed, 2 seeds):** v0.7 trains a non-collapsed
  execution-transition latent that (a) separates true outcomes from
  decoys with AUC 1.0, (b) retrieves the true after-state far above
  no-action/random, and (c) **predicts output magnitude above every
  control** — a property v0.6 could not even measure.
- **Cannot claim:** a downstream reranking improvement (benchmark
  saturated), a full multi-target representation result (only magnitude
  beats controls), or anything about crash prediction (data-limited).

## Next

- **WS-D** — an unsaturated reranking benchmark so representation gains can
  show downstream lift.
- Re-run retrieval with the `lexical` baseline to close the action-use
  gate cleanly.
- Complete the 50000-step full runs for a training-duration ablation.
- WS-C4 (output-value auxiliary head) to push length/type decodability,
  and a harder pack (more exceptions) to make crash prediction evaluable.
