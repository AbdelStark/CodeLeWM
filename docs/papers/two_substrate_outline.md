# Paper Outline: Two Substrates for a JEPA-style Code World Model

> This is the structured outline that turns the v0.2 commit-edit
> work and the v0.6 execution-substrate work into a single,
> publishable comparison. It commits to a narrative that ships
> regardless of whether v0.6 passes its claim gates.

- Status: outline draft (post #272); full #306 paper draft in
  `docs/papers/two_substrate_paper.tex`
- Tracker: #259, RFC-0014
- Substrate A (commit-edit, v0.2) benchmark:
  `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`
- Substrate B (execution-trace, v0.6) benchmark:
  `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md`
  plus per-seed eval artifacts in `docs/benchmark/v0_6/`
- Substrate B report template:
  `docs/benchmark/EXECUTION_V0_6_RESULTS_TEMPLATE.md`
- Working title: *"The substrate is the bottleneck: a JEPA
  world-model recipe applied to two flavors of code data"*
- Target venue: code/ML workshop or COLM. Tighten during the #306
  draft pass.

## Abstract

We apply the same CodeLeWM latent-transition recipe family — encoder,
predictor, SIGReg, action-swap contrastive, inverse-action
reconstruction — to two substrates of code data. Substrate A is
commit-message-conditioned code edits over CommitPackFT-Python.
Substrate B is input-conditioned program execution over CodeNet,
MBPP, and APPS, with outputs captured by a sandboxed deterministic
executor at data-build time. On Substrate A, four scaled training
runs all fail the headline retrieval gate against the no-action
baseline and produce severely rank-collapsed latents (effective rank
ratio 0.016 of 256 dimensions). On Substrate B, the same
latent-transition model family and objective registry produce
non-collapsed action-discriminative latents: across two training seeds,
the no-action margin flips from
−0.77 to +1.24, SIGReg drops 1200× (44 → 0.036), and the predicted
latents' effective-rank ratio settles at 0.47 — 2.3× the collapse
gate. The comparison makes the substrate the primary explanatory
variable, while disclosing run-level configuration differences; we
argue the result implies a general lesson for JEPA-style modeling of
code: substrate signal-to-noise dominates objective tuning. The #305
downloaded-artifact eval pass strengthens but narrows the positive
result: execution-pack retrieval beats
no-action by +61.4 Recall@1 points and +65.9 MRR points on average
across two seeds, and generated-decoy surprise AUC is 1.0 on mutation,
same-code-different-input, and same-problem-different-submission
decoys. Broader downstream utility is not established: the latent
probe claim is blocked by lexical controls, crash prediction is not
evaluable because the val/test slice has no positives, and
HumanEval/MBPP-Plus reranking still lacks live labeled-completion
artifacts.

## 1. Introduction

- JEPA-style world models have produced strong results in vision and
  video; the question of whether the same recipe transfers to code
  remains open.
- "Code" is an umbrella term: the right substrate matters. We frame
  the choice as the primary variable, while explicitly noting that the
  v0.6 execution runner and schedule are not a perfect ablation of
  the v0.2 commit-edit run.
- Contribution sketch:
  1. A reproducible JEPA latent-transition recipe applied to two
     substrates.
  2. A negative result on commit-edit transitions, with
     diagnostics that make the substrate-signal explanation more
     plausible than another objective-only fix.
  3. A partial-positive result on input-conditioned execution:
     training-shape, internal retrieval, and generated-decoy surprise
     gates pass, while semantic-probe, crash, and downstream-rerank
     utility claims remain blocked.
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
- Predictor: shared CodeLeWM latent predictor family. The v0.2
  action-swap run uses the gated-residual fusion profile; the v0.6
  execution runner uses the conditional-transformer fusion surface.
- Target encoder: no separate EMA target encoder in the v0.6
  execution runner; checkpoint compatibility is governed by the
  shared state/action/predictor contract.
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
- Training: v0.6 HF Jobs runs, two seeds, 50k steps, same objective
  family as the v0.2 action-swap run but different weights and
  execution-specific loader/schedule.

## 6. Results

### 6.1 Retrieval

| Substrate | Run | Recall@1 | No-action Δ | MRR | No-action Δ |
|-----------|-----|---------:|------------:|----:|------------:|
| A | #138 | 0.371 | -0.088 | 0.473 | -0.073 |
| A | #154 | 0.363 | -0.106 | 0.468 | -0.082 |
| A | #159 | 0.597 | -0.053 | 0.675 | -0.034 |
| A | #172 | 0.263 | -0.178 | 0.370 | -0.163 |
| B | v0.6 seed 42 | **0.657** | **+0.619** | **0.767** | **+0.663** |
| B | v0.6 seed 1729 | **0.648** | **+0.610** | **0.759** | **+0.655** |

Substrate B reverses Substrate A's headline retrieval failure on the
execution-pack val/test slice. Mean Recall@1 is 0.653 with mean lift
+0.614 over no-action; mean MRR is 0.763 with mean lift +0.659. The
seed-to-seed spread is below one point for Recall@1 and MRR. This is
internal execution-pack retrieval evidence, not HumanEval/MBPP-Plus
downstream reranking evidence. The claim posture is documented in
`docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md`.

### 6.2 Collapse And Surprise

| Substrate | Run | Effective rank ratio | Mean ‖z‖₂ | Mean pairwise cosine |
|-----------|-----|---------------------:|----------:|---------------------:|
| A | scaled v0.2 | 0.016 | n/a | n/a |
| B | v0.6 seed 42 | **0.467** | 14.97 | 3e-4 |
| B | v0.6 seed 1729 | **0.477** | 14.94 | 1e-4 |

Substrate A: effective rank ratio 0.016. **Substrate B clears the
0.20 collapse gate by 2.3× on both seeds (`EXECUTION_V0_6_RESULTS_2026-05-30.md`),
inverting Substrate A's collapse failure.**

Surprise on generated execution-pack decoys:

| Run | Mutation AUC / pairs | Same-code-different-input AUC / pairs | Same-problem-different-submission AUC / pairs |
|-----|---------------------:|--------------------------------------:|-----------------------------------------------:|
| v0.6 seed 42 | **1.000 / 236** | **1.000 / 195** | **1.000 / 6** |
| v0.6 seed 1729 | **1.000 / 236** | **1.000 / 195** | **1.000 / 6** |

The same-problem row clears the configured numeric gate but has only
six generated pairs after filtering, so the paper should frame it as
a generated-decoy diagnostic rather than broad semantic surprise.

### 6.3 Latent Probes

Substrate A: every target beaten by lexical or metadata-only
controls (Table from `V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`).
Substrate B:

| Run | Target | z_pred_after test acc | No-action test acc | Lexical test acc | Claim |
|-----|--------|----------------------:|-------------------:|-----------------:|-------|
| v0.6 seed 42 | `output_type` | 0.497 | 0.439 | **0.662** | blocked |
| v0.6 seed 1729 | `output_type` | 0.599 | 0.541 | **0.618** | blocked |

Only `output_type` has train/val/test labels. The latent view beats
no-action by 5.7 points on both seeds, but lexical controls remain
stronger. The positive representation claim is therefore closed:
`positive_representation_claim_allowed=false`,
`semantic_structure_status=not_evaluable`.

### 6.4 Downstream Reranking

Substrate A has no positive downstream rerank evidence at scale
(only the one-fixture smoke). Substrate B:

| Benchmark | LLM-order pass@1 | CodeLeWM pass@1 | Lift | 95% CI |
| --------- | ---------------: | --------------: | ---: | -----: |
| HumanEval | n/a | n/a | n/a | n/a |
| MBPP-Plus | n/a | n/a | n/a | n/a |

The completion-label sampler and `codelewm.eval.completion_label.v1`
contract exist, but the full live HumanEval / MBPP-Plus labeled
artifacts are not present. No downstream-rerank claim is allowed.

### 6.5 Crash Prediction (Scoped Fallback)

| Run | Samples | Positives | Negatives | Claim |
|-----|--------:|----------:|----------:|-------|
| v0.6 seed 42 | 236 | 0 | 236 | not evaluable |
| v0.6 seed 1729 | 236 | 0 | 236 | not evaluable |

The execution-pack val/test slice has no crash-positive examples, so
the crash-prediction fallback is not evaluable.

## 7. Discussion

- Substrate signal-to-noise dominates objective tuning.
- The four-run sweep on Substrate A is a negative result across
  objective variants. None of the variations make the latent useful.
- Substrate B inverts the training-shape and internal retrieval
  failure modes of Substrate A. It is not a full downstream-utility
  win: lexical controls still beat the latent probe and HumanEval /
  MBPP-Plus rerank evidence is absent.
- Implications for related code-modeling work: pick the substrate
  before picking the architecture.

## 8. Limitations And Threats To Validity

- The v0.2-v0.6 comparison is not a perfect single-variable ablation:
  v0.6 changes the loader, optimizer schedule, effective batch size,
  and action-fusion surface along with the substrate. The paper should
  claim a substrate-pivot result, not proof that substrate is the only
  changed variable.
- Reference LLM (Claude Haiku 4.5) version-dependence on future
  downstream rerank results.
- Python-only and stdlib-only sandbox policy: scope is narrower than
  "code in general."
- Two training seeds: lower bound on the variance estimate.
- License-clean source-dataset envelope: results do not generalize
  outside MIT/Apache/CC-BY/permissive sources.
- Generated surprise decoys are not a substitute for adversarial or
  independently sampled semantic decoys; the same-problem decoy row has
  only six pairs.
- HumanEval / MBPP-Plus reranking remains an operator-run dependency:
  the sampler contract exists, but live labeled-completion artifacts
  are not part of the current evidence set.

## 9. Conclusion

Pick the substrate before picking the architecture. We provide two
public artifact lines (datasets + checkpoints + manifests) and the
end-to-end reproducibility steps so the next paper in this space can
start from where ours leaves off.

## 10. Reproducibility

- Substrate A: `abdelstark/codelewm-public-shard`,
  `abdelstark/codelewm-transition-model`, `abdelstark/codelewm-runs`.
- Substrate B: `abdelstark/codelewm-execution-pack@v0.6.0` plus the
  v0.6 model checkpoints, run artifacts, and committed per-seed eval
  reports in `docs/benchmark/v0_6/`.
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

## 11. Final Framing

**Framing as of 2026-05-31 (post-#305 eval pass):**
*partial positive — substrate-pivot, internal retrieval, and
generated-decoy surprise gates pass; broader downstream utility is not
established.*

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
- Execution-pack retrieval passes the no-action gates across both
  seeds: Recall@1 lift is +0.619 / +0.610 and MRR lift is
  +0.663 / +0.655.
- Generated-decoy surprise passes the configured mutation,
  same-code-different-input, and same-problem-different-submission
  gates at AUC 1.0 on both seeds, with the caveat that
  same-problem has only six generated pairs.

The broader downstream-utility story is negative or incomplete:

- Latent probes do not support a semantic-representation claim. Only
  `output_type` is evaluable, and lexical controls beat the latent
  view on both seeds.
- Crash prediction is not evaluable because the val/test slice has
  zero crash-positive examples.
- HumanEval / MBPP-Plus reranking is not run because live
  completion-label artifacts are absent.

The paper's framing should therefore be:

> *"A substrate-pivot comparison: Substrate A fails the collapse
> gate at 0.016 effective-rank ratio and loses to no-action retrieval;
> Substrate B passes the collapse gate at 0.47, flips the no-action
> margin from −0.77 to +1.24, beats no-action retrieval by +61.4
> Recall@1 points and +65.9 MRR points on average, and passes
> generated-decoy surprise diagnostics. The same model family and
> objective registry yields qualitatively different latents on the two
> substrates, with run-level configuration differences disclosed. The
> result is partial, not a broad code-generation or
> downstream-reranking claim: semantic probes remain blocked by lexical
> controls and HumanEval / MBPP-Plus reranking remains unscored."*

## References

To be filled in. Key citations: I-JEPA, V-JEPA, LeWorldModel,
CodeBERT, GraphCodeBERT, UnixCoder, CodeT5, CodeNet, MBPP,
HumanEval, MBPP-Plus, APPS, SIGReg, VICReg, Barlow Twins.
