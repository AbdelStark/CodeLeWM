# Paper Outline: Two Substrates for a JEPA-style Code World Model

> This is the structured outline that turns the v0.2 commit-edit
> work and the v0.6 execution-substrate work into a single,
> publishable comparison. It commits to a narrative that ships
> regardless of whether v0.6 passes its claim gates.

- Status: outline draft (post #272)
- Tracker: #259, RFC-0014
- Substrate A (commit-edit, v0.2) benchmark:
  `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`
- Substrate B (execution-trace, v0.6) benchmark template:
  `docs/benchmark/EXECUTION_V0_6_RESULTS_TEMPLATE.md`
- Working title: *"The substrate is the bottleneck: a JEPA
  world-model recipe applied to two flavors of code data"*
- Target venue: code/ML workshop or COLM. Tighten to a venue once the
  v0.6 results land.

## Abstract

We apply the same JEPA latent-transition recipe — encoder, predictor,
SIGReg, action-swap contrastive, inverse-action reconstruction — to
two substrates of code data. Substrate A is
commit-message-conditioned code edits over CommitPackFT-Python.
Substrate B is input-conditioned program execution over CodeNet,
MBPP, and APPS, with outputs captured by a sandboxed deterministic
executor at data-build time. On Substrate A, four scaled training
runs all fail the headline retrieval gate against the no-action
baseline and produce severely rank-collapsed latents (effective rank
ratio 0.016 of 256 dimensions). On Substrate B, the same architecture
and objective registry produce non-collapsed action-discriminative
latents: across two training seeds, the no-action margin flips from
−0.77 to +1.24, SIGReg drops 1200× (44 → 0.036), and the predicted
latents' effective-rank ratio settles at 0.47 — 2.3× the collapse
gate. The comparison isolates the substrate as the controlling
variable; we argue the result implies a general lesson for
JEPA-style modeling of code: substrate signal-to-noise dominates
objective tuning. Downstream-utility evaluations (retrieval,
surprise, rerank) are deferred to a follow-up evaluation harness
wiring.

## 1. Introduction

- JEPA-style world models have produced strong results in vision and
  video; the question of whether the same recipe transfers to code
  remains open.
- "Code" is an umbrella term: the right substrate matters. We frame
  the choice as a controlled variable.
- Contribution sketch:
  1. A reproducible JEPA latent-transition recipe applied to two
     substrates.
  2. A negative result on commit-edit transitions, with
     diagnostics that isolate the failure to substrate signal, not
     architecture.
  3. A positive (or diagnostic) result on input-conditioned
     execution, with downstream-reranking evidence on HumanEval
     and MBPP-Plus.
  4. Publicly published datasets, model checkpoints, and run
     manifests for both substrates.

## 2. Background And Related Work

- I-JEPA, V-JEPA, and the LeWorldModel line.
- Code-representation models: CodeBERT, GraphCodeBERT, UnixCoder,
  CodeT5.
- Execution-based code modeling: TRACED, execution-conditioned
  training, neural execution.
- Reranking for code generation: CodeT, pass@k via critic models,
  ranker fine-tuning.
- Anti-collapse SSL: SIGReg, VICReg, Barlow Twins.

## 3. Architecture (Shared Across Substrates)

- The latent-transition contract:
  `latent(after) = predictor(latent(before), action)`.
- Encoder: token + segment + changed-hunk-mask + positional
  embeddings, mean-pool, 2-layer GELU MLP to latent_dim=256.
- Action encoder: same pool architecture; carries the conditioning
  signal whose semantics differ across substrates.
- Predictor: 6-layer Transformer (`ARPredictor`).
- Target encoder: EMA of the state encoder.
- Objective registry:
  - prediction MSE (weight 1.0);
  - SIGReg (weight 0.09);
  - action-swap contrastive (weight 0.1);
  - inverse-action reconstruction (weight 0.05).

## 4. Substrate A — Commit-Message-Conditioned Code Edits

- Data: CommitPackFT-Python, ≈20k packed transitions, license-gated
  to permissive-only.
- Action: commit message (free-form natural language).
- Training: four scaled HF Jobs runs over a v0.2 lineage.
- Results: every run loses to the no-action baseline on Recall@1 and
  MRR. The v0.2 action-swap run reports effective rank ratio
  0.015761 (vs. threshold 0.20), mutation surprise AUC 0.501
  (chance), and lexical-baseline-beats-latent on all six probe
  targets.
- Diagnosis: the substrate carries three compounding signal failures
  — `after ≈ before` Bayes-optimal prior, vague commit messages,
  multi-purpose edits. The objective registry cannot compensate.

## 5. Substrate B — Input-Conditioned Execution

- Substrate definition: `(code, input) → output` triples captured by
  a sandboxed deterministic Python executor (#260) at data-build
  time. Source datasets: CodeNet (Python), MBPP (function-call),
  APPS (stdin). HumanEval and MBPP-Plus held out for downstream
  evaluation.
- Substrate properties (vs. Substrate A):
  - output and code live in disjoint token distributions: no
    "do-nothing" prior;
  - input is a typed deterministic value: precise action signal;
  - each record is single-purpose: one transformation per row.
- Implementation: `codelewm.data.execution_pack.v1` builder,
  sandbox policy, HF dataset publish flow.
- Training: v0.6 HF Jobs runs, two seeds, 50k steps, identical
  objective registry to Substrate A.

## 6. Results

### 6.1 Retrieval

| Substrate | Run | Recall@1 | No-action Δ | MRR | No-action Δ |
|-----------|-----|---------:|------------:|----:|------------:|
| A | #138 | 0.371 | -0.088 | 0.473 | -0.073 |
| A | #154 | 0.363 | -0.106 | 0.468 | -0.082 |
| A | #159 | 0.597 | -0.053 | 0.675 | -0.034 |
| A | #172 | 0.263 | -0.178 | 0.370 | -0.163 |
| B | v0.6 seed 42 | *deferred* | *deferred* | *deferred* | *deferred* |
| B | v0.6 seed 1729 | *deferred* | *deferred* | *deferred* | *deferred* |

The Substrate B retrieval numbers are deferred to a follow-up wiring
of the v0.6 JSONL execution-pack to the `codelewm eval retrieval`
CLI (currently HDF5-only). The training-time **no-action margin
proxy** for Substrate B is +1.23 on seed 42 and +1.24 on seed 1729
(positive across both, with spread 0.013 ≪ mean 1.24) — the model
clearly uses the action signal rather than relying on a do-nothing
prior. The headline-claim posture is documented in
`docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md`.

### 6.2 Collapse And Surprise

| Substrate | Run | Effective rank ratio | Mean ‖z‖₂ | Mean pairwise cosine |
|-----------|-----|---------------------:|----------:|---------------------:|
| A | scaled v0.2 | 0.016 | n/a | n/a |
| B | v0.6 seed 42 | **0.467** | 14.97 | 3e-4 |
| B | v0.6 seed 1729 | **0.477** | 14.94 | 1e-4 |

Substrate A: effective rank ratio 0.016. **Substrate B clears the
0.20 collapse gate by 2.3× on both seeds (`EXECUTION_V0_6_RESULTS_2026-05-30.md`),
inverting Substrate A's collapse failure.** Surprise AUCs deferred
to the eval-suite follow-up.

### 6.3 Latent Probes

Substrate A: every target beaten by lexical or metadata-only
controls (Table from `V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`).
Substrate B: per-target table per
`docs/benchmark/EXECUTION_V0_6_RESULTS_TEMPLATE.md`.

### 6.4 Downstream Reranking

Substrate A has no positive downstream rerank evidence at scale
(only the one-fixture smoke). Substrate B:

| Benchmark | LLM-order pass@1 | CodeLeWM pass@1 | Lift | 95% CI |
| --------- | ---------------: | --------------: | ---: | -----: |
| HumanEval |  |  |  |  |
| MBPP-Plus |  |  |  |  |

### 6.5 Crash Prediction (Scoped Fallback)

Filled in from the v0.6 crash-prediction report.

## 7. Discussion

- Substrate signal-to-noise dominates objective tuning.
- The four-run sweep on Substrate A is a controlled negative result:
  every run uses the same architecture, varying only the objective
  weights. None of the variations make the latent useful.
- Substrate B inverts the failure modes of Substrate A by
  construction. Whether it produces a positive headline result, a
  partial-positive (e.g., crash-prediction-only), or a negative
  result, the comparison itself is the contribution.
- Implications for related code-modeling work: pick the substrate
  before picking the architecture.

## 8. Limitations And Threats To Validity

- Reference LLM (Claude Haiku 4.5) version-dependence on downstream
  rerank results.
- Python-only and stdlib-only sandbox policy: scope is narrower than
  "code in general."
- Two training seeds: lower bound on the variance estimate.
- License-clean source-dataset envelope: results do not generalize
  outside MIT/Apache/CC-BY/permissive sources.

## 9. Conclusion

Pick the substrate before picking the architecture. We provide two
public artifact lines (datasets + checkpoints + manifests) and the
end-to-end reproducibility steps so the next paper in this space can
start from where ours leaves off.

## 10. Reproducibility

- Substrate A: `abdelstark/codelewm-public-shard`,
  `abdelstark/codelewm-transition-model`, `abdelstark/codelewm-runs`.
- Substrate B: `abdelstark/codelewm-execution-pack@v0.6.0` plus the
  v0.6 model checkpoints and run artifacts.
- All artifacts publish with `codelewm manifest verify` round-trip
  evidence and `codelewm secret-scan` reports.
- Code path:
  - data: `codelewm.data.execution_sources`,
    `codelewm.data.execution_pack`, `codelewm.data.sandbox`.
  - train: `codelewm.training.execution_pack_loader`,
    `codelewm.training.execution_launch_plan`,
    `config/train/scaled/codelewm_execution_v0_6_a10g.yaml`.
  - eval: `codelewm.eval.execution_probe_targets`,
    `codelewm.eval.execution_surprise_decoys`,
    `codelewm.eval.execution_rerank`,
    `codelewm.eval.crash_prediction`.
  - harness: `codelewm.harness.execution_rerank_view_model`,
    `codelewm.harness.demo_scenarios.EXECUTION_RERANK_SCENARIO_ID`.

## 11. Two Framings, Two Endings

**Framing as of 2026-05-30 (post-v0.6 first end-to-end run):**
*partial positive — substrate-pivot's headline-shape claim confirmed,
downstream-utility evaluations deferred.*

The substrate-pivot's headline prediction — that the
execution-trace substrate produces a non-collapsed
action-discriminative latent transition model — is confirmed by the
v0.6 HF Jobs runs (`EXECUTION_V0_6_RESULTS_2026-05-30.md`):

- Across two training seeds, prediction MSE drops three orders of
  magnitude (0.95 → 6e-4), SIGReg drops 1200× (44 → 0.036), and the
  no-action margin flips from −0.77 to +1.24.
- The effective-rank ratio of predicted latents climbs to 0.47,
  clearing the 0.20 collapse gate by 2.3×. Substrate A failed the
  same gate at 0.016.
- Cross-seed variance is small (margin spread = 0.013, ~1% of mean).

The downstream-utility gates (retrieval lift, surprise AUC, latent
probe, downstream rerank) are deferred to a follow-up CLI wiring of
the v0.6 JSONL execution pack to the existing evaluator libraries
(`codelewm.eval.execution_probe_targets`,
`codelewm.eval.execution_surprise_decoys`,
`codelewm.eval.execution_rerank`, `codelewm.eval.crash_prediction`).
Until that follow-up lands, the paper's framing is:

> *"A controlled substrate comparison: Substrate A fails the
> collapse gate at 0.016 effective rank ratio; Substrate B passes it
> at 0.47, with the no-action margin flipping from −0.77 to +1.24.
> The same architecture and objective registry yields qualitatively
> different latents on the two substrates. Downstream-utility
> claims (retrieval, surprise, rerank) are deferred to a follow-up
> evaluation harness wiring."*

The published artifact set is sufficient to fill the deferred tables
once the eval harness ships; both framings remain available from the
same artifacts.

## References

To be filled in. Key citations: I-JEPA, V-JEPA, LeWorldModel,
CodeBERT, GraphCodeBERT, UnixCoder, CodeT5, CodeNet, MBPP,
HumanEval, MBPP-Plus, APPS, SIGReg, VICReg, Barlow Twins.
