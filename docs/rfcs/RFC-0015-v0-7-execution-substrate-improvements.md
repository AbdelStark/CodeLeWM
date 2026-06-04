# RFC-0015: v0.7 Execution-Substrate Improvements

- Status: Draft
- Authors: CodeLeWM maintainers
- Created: 2026-06-02
- Target milestone: v0.7

## Summary

The v0.6 execution-trace substrate (RFC-0014) turned the broad JEPA recipe from
a negative result into a substrate-level partial positive. Three acceptance
gates are **open** across both seeds and must be preserved:

- execution-pack retrieval R@1 **+0.6144** / MRR **+0.6588** over no-action;
- generated/mutation-decoy surprise AUC **1.0** (score gates);
- anti-collapse effective-rank ratio **~0.47** (threshold 0.20).

Four gates remain **closed**. This RFC scopes the v0.7 work to close them
without overstating the claim boundary. The closed gates and their verified
mechanical causes are:

1. **Downstream rerank** (pass@1 lift ≥ 3.0 pts, bootstrap CI > 0, ≥ 3 LLM
   seeds): the rerank score is a self-supervised transition *plausibility*
   (`pred_norm + |cos sim|`) that is never trained or calibrated against pass/fail.
2. **Representation/latent probe** (≥ 1 target beats every control across
   ≥ 2 seeds; ≥ 5 labeled targets): only one of four declared targets had
   labels, and on that target a lexical control beats the latent.
3. **Surprise same-problem/different-submission count** (≥ 30 scored pairs):
   only 6 pairs exist in the pack.
4. **Crash prediction**: not evaluable — the pack contains zero exception
   records.

v0.7 attacks these with four workstreams: correctness-aware scoring, a data
overhaul, targeted architecture/recipe upgrades, and an unsaturated rerank
benchmark. The claim boundary does not change: a positive downstream
coding-usefulness claim still requires the full RFC-0014 gate set; v0.7 aims to
make that gate *reachable and honestly measured*, not to assert it.

## Motivation

The v0.6 evidence (`docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md`,
`docs/benchmark/V0_6_RERANK_FULL_2026-06-01.md`) shows the latent geometry is
good (retrieval and surprise are strong) but two things are missing: the model
is never connected to *correctness*, and the data cannot exercise the semantic
probe and decoy gates. The root causes are concrete and code-level.

### Why downstream rerank is flat/negative

The `codelewm` rerank score is the negated transition energy of
`before → candidate` (`codelewm/eval/execution_rerank.py:807, 840-848`). For the
execution backend the energy is a latent norm plus a self-similarity term
(`codelewm/harness/scorer.py:606-645`):

```
z_pred_after = predict_after(encode_state(code), encode_action(input))
energy = ‖z_pred_after‖₂ + |cos(z_pred_after, z_code)|
```

The training objective (`codelewm/model/objective.py:154-251`) is pure
self-supervised JEPA — next-latent MSE + SIGReg + a single action-swap negative
+ inverse-action reconstruction. **No term is a function of execution outcome**,
and `passed` labels are read only post-hoc on the eval side
(`execution_rerank.py:73, 282-285`). Ranking on transition plausibility is, to
first order, uncorrelated with correctness — and since `pred_norm` rewards large
latent moves, it can be mildly anti-correlated. The scorer's own warnings admit
this (`scorer.py:514-515, 534-535`). The benchmarks also saturate (~0.95 pass@1),
leaving little lift headroom.

### Why the representation probe is closed

Gate code (`codelewm/eval/latent_probe.py:516-518`) requires
`available_targets >= min(5, len(config.targets))`. v0.6 declared four targets
(`output_type, will_raise, output_magnitude_bucket, output_length_bucket`,
`codelewm/eval/execution_runner.py:505-510`) but only `output_type` had
train/val/test labels. `will_raise` is always `False` (no exception records);
`output_magnitude_bucket`/`output_length_bucket` get 0 labels because CodeNet
stdout text is `output_type="str"` and not parseable as a typed value
(`codelewm/eval/execution_probe_targets.py:151, 180-187`;
`codelewm/data/sandbox/_child.py:393-411`). On the one labeled target the lexical
control beats the latent (0.66 vs 0.50 test acc), so the latent encodes no signal
beyond surface tokens. (`positive_representation_claim_allowed` is additionally
hard-coded `False` at `latent_probe.py:530`.)

### Why the surprise count and crash gates fail

The same-problem/different-submission generator
(`codelewm/eval/execution_surprise_decoys.py:94-186`) drops candidate pairs whose
outputs are identical. Because `scripts/dataset/flatten-codenet` keeps only
`accepted` verdicts with few submissions per problem and one I/O case, only 6
qualifying pairs exist (threshold 30, `execution_runner.py:107`). The pack has no
exception records, so crash prediction has no positive class.

### Underlying recipe enablers

The state encoder is a bag-of-embeddings mean-pool (`codelewm/model/state.py:44-104`)
while the action encoder and predictor are transformers — a capacity asymmetry.
There is no EMA/target encoder (`execution_runner.py:26-29`), no value/output or
correctness head, weak negatives (one batch-roll swap; InfoNCE implemented but
off, `objective.py:23-24`), tiny data (~1,605 transitions, Python-only,
stdout-dominated, no dedup, train consumes all splits), and two latent bugs:
`precision: bf16-mixed` is declared but no autocast is wired, and
`prediction_mse_weight` is declared but never applied (`objective.py:175`).

## Goals

Close the gaps while keeping every open gate. Target gate states for v0.7:

| Gate | v0.6 | v0.7 target | Primary workstream |
|---|---|---|---|
| Surprise same-problem/diff-submission count | 6 pairs | ≥ 30 pairs | WS-B |
| Crash prediction | not evaluable | evaluable (≥1 pos + neg) | WS-B |
| Representation probe | closed | ≥ 1 latent target beats lexical by > 0.05 on 2 seeds; ≥ 5 labeled targets | WS-B + WS-C |
| Downstream rerank | flat/neg | calibrated score beats no-action; ≥ +3 pt lift (CI > 0) on an unsaturated split, ≥ 3 LLM seeds | WS-A + WS-D |
| Retrieval R@1/MRR, surprise AUC, anti-collapse, 2-seed | open | keep open | — |

## Non-Goals

- Do not abandon the JEPA self-supervised representation; correctness signal is
  added *on top of* it, not in place of it.
- Do not assert a broad coding-usefulness claim from v0.7 alone. The claim gate
  stays closed unless the full RFC-0014 downstream gate passes.
- Do not extend to non-Python languages or execute candidate code outside the
  existing operator-reviewed sandbox boundary (RFC-0010, RFC-0014).
- Do not change the open-gate thresholds to manufacture a pass.

## Design

### WS-A — Correctness-aware scoring (targets the rerank gate)

The fusion hook already exists: `final_score = energy + retrieval_prior_weight *
retrieval_prior` (`scorer.py:770-772`) with a reserved, always-`None`
`risk_penalty` slot (`scorer.py:184, 197-198`). Three additions, increasing in
fidelity:

- **A1 — label-trained calibrator (cheapest, no retraining).** Every eval run
  already writes per-completion features
  (`transition_energy, pred_norm, similarity, retrieval_prior, lexical,
  llm_order_rank`) plus `passed` to `completion_scores.jsonl`
  (`execution_rerank.py:817-827`). Fit a logistic / pairwise-ranking calibrator
  on existing v0.6 artifacts and fuse it through `final_score`. This answers the
  make-or-break question — *is correctness decodable from current signals?* —
  before any GPU spend.
- **A2 — label-aware retrieval prior.** Rebuild the transition index over
  **known-passing** transitions, query with the **learned encoder** (not the
  current hashed bag-of-tokens, `scorer.py:794`), and define the prior as
  similarity to / fraction of nearest *passing* neighbors. Requires adding a
  `passed` field to `TransitionIndexEntry.metadata`
  (`codelewm/harness/transition_index.py:63`). This converts the strong R@1
  signal into a correctness signal and is the most promising single lever.
- **A3 — execution-outcome head (the real fix).** Add
  `p_pass = σ(W·[z_code, action, z_pred_after])` trained with BCE on sandbox
  `passed` labels, as a new gated term in `compute_transition_objective`
  (mirroring `enable_inverse_action_reconstruction`, `objective.py:32-33,
  214-226`). The execution backend returns `-p_pass` (or its logit) as the
  energy so `final_score` becomes a calibrated pass-probability. Run as a
  supervised second stage on the SSL trunk to preserve retrieval/surprise.

### WS-B — Data overhaul (unblocks probe, surprise count, crash; adds headroom)

- **B1** — add function-call sources (`flatten-mbpp`, `flatten-apps`) so outputs
  are typed `repr(value)`, unblocking `output_magnitude/length/arithmetic` probe
  targets.
- **B2** — `flatten-codenet --keep-verdicts=accepted,wrong_answer,runtime_error`,
  raise `--max-submissions-per-problem`, and emit multiple I/O cases, yielding
  differing outputs (≥ 30 surprise pairs) and exception records (crash +
  `exception_type` target).
- **B3** — scale the pack, add `raw_hash`-based dedup in
  `codelewm/data/execution_sources/base.py`, and restrict training to the train
  split (today the runner consumes train+val+test, `execution_runner.py:216-224`).
- **B4** — expand the default probe target set to ≥ 5 and label them; add an
  `exception_type` target in `codelewm/eval/execution_probe_targets.py`.

### WS-C — Architecture/recipe upgrades (raise the ceiling so latent > lexical)

- **C1** — replace the bag-of-embeddings state encoder with a transformer
  (`codelewm/model/state.py`).
- **C2** — add an EMA target encoder + stop-grad (proper JEPA asymmetry).
- **C3** — enable multi-negative InfoNCE (already implemented, off) and use
  same-problem/different-submission pairs as hard negatives at train time.
- **C4** — add an output-value auxiliary head at train time (predict the output
  bucket from `z_pred_after`) to force the latent to encode output semantics so
  it can beat the lexical control on probes.
- **C5** — fix the two latent bugs: wire `torch.autocast` for `bf16-mixed`;
  apply `prediction_mse_weight`. Consider larger `latent_dim`/model.

### WS-D — Unsaturated rerank benchmark

Build a harder split (or weaker generation model / larger candidate pools) so a
real pass/fail mix and ≥ 3 pt lift headroom exist. Live experiments on the
v0.6 model confirmed the failure mode: a frontier model passes every candidate
(no discrimination) and a weak model fails the patch format (no valid
candidates) — the sweet spot requires a controlled, unsaturated set.

## Open Questions

- Does a calibrator over existing features already recover correctness (WS-A1),
  or is a retrained trunk required? Phase 0 decides this.
- Is the lexical control beatable on `output_type`, or does the probe need a
  harder semantic target (e.g. `output_value` bucket) added via WS-B/WS-C4?
- Can the downstream lift reach +3 pt at all, or is the honest v0.7 outcome a
  calibrated score that beats no-action on an unsaturated split without clearing
  the full headline gate?
- Should the retrieval index store `passed` (schema change) or maintain a
  separate passing-only index?

## Implementation

- **Phase 0 (no retraining):** WS-A1 calibrator + WS-A2 prior on existing v0.6
  checkpoints/artifacts; WS-C5 bug fixes; build the WS-D harder split. Decides
  viability before GPU spend.
- **Phase 1 (data):** WS-B → publish execution-pack v0.7; re-run the eval-only
  gates against the v0.6 model (should flip surprise-count + crash-evaluable and
  make the probe evaluable).
- **Phase 2 (model):** WS-C → train v0.7 (2 seeds), targeting probe-beats-lexical.
- **Phase 3 (correctness):** WS-A3 outcome head supervised stage → calibrated
  rerank; eval on the WS-D split with ≥ 3 LLM seeds.
- **Phase 4:** full 2-seed substrate + 3-seed rerank eval; publish results, model
  cards, and a claim audit.

## Acceptance

- All currently-open gates remain open across 2 seeds (retrieval, surprise AUC,
  anti-collapse).
- High-confidence flips: surprise same-problem/different-submission count ≥ 30;
  crash prediction evaluable; representation probe evaluable with ≥ 5 labeled
  targets.
- Stretch: ≥ 1 latent probe target beats every control by > 0.05 on both seeds.
- Hard target (honestly reported, may remain partial): a calibrated rerank score
  that beats no-action on an unsaturated split, advancing toward the ≥ 3 pt /
  CI > 0 / 3-seed downstream gate.
- Every v0.7 artifact passes manifest verification and secret scans; the public
  claim boundary is preserved and any partial-positive shape is scoped.

## Outcomes and v0.8 Direction (2026-06-04)

v0.7 was executed end-to-end (2-seed A10G training on the bucket-augmented
mbpp pack, full eval, the WS-D unsaturated rerank benchmark, and the WS-A
correctness probes). Verdict by work-stream:

| WS | Result |
|----|--------|
| WS-B (data: probe labels, dedup, buckets) | **Shipped.** Probe gate flipped `not_evaluable` → evaluable; `output_magnitude_bucket` / `output_length_bucket` now labeled. |
| WS-C (transformer encoder, InfoNCE, applied pred-MSE) | **Shipped + trained.** Non-collapsed latent (eff-rank ratio 0.30–0.33), surprise AUC 1.0, retrieval recall@1 +0.48/+0.51, and a reproducible **positive**: the latent **predicts output magnitude** above every control on both seeds (+0.21 / +0.15) — newly measurable vs v0.6. |
| WS-D (unsaturated rerank benchmark) | **Shipped.** Deterministic mutation-distractor packs (mix rate 1.0 vs 0.039). Revealed v0.7 reranks at **chance** (codelewm pass@1 0.06–0.17 vs random 0.17), so the flat downstream result was not only saturation. |
| WS-A (correctness-aware scoring) | **Diagnosed, not shipped.** A1: scalar features decode correctness at chance (AUC 0.500). A1.5: the **full 256-dim latents** decode correctness at chance too (0.47–0.54, both seeds). The SSL representation does not encode correctness — so A1/A2 and a frozen-trunk A3 head are all ruled out. |

**The v0.7 conclusion.** The execution substrate learns genuine execution
*structure* (retrieval, surprise, output-magnitude prediction — all positive,
2-seed) but carries **no correctness signal**: the `(code, input) → output`
self-supervised objective never sees the problem's expected output / spec, so
"is this output the correct one" is outside what the representation is trained
to capture. This is the root cause of both the WS-D rerank-at-chance result
and the A1/A1.5 probes.

**v0.8 direction — inject correctness at training time.** The cheap read-out
fixes are dead; correctness must be co-trained into the trunk:

1. **Pass/fail training data** — a new `completion_label.v1 → pack` adapter
   that re-executes the WS-D mutation completions to recover `output_repr`,
   tokenizes, assigns splits, and writes a `passed` label (the execution pack
   has no correctness labels by construction). Plus the `record.py` / loader /
   executor schema bumps.
2. **A3 co-trained from scratch** — the `p_pass = σ(W·[z_code, action,
   z_pred_after])` BCE head trained *jointly* with the SSL objective (not a
   frozen-trunk second stage, which A1.5 rules out), so the trunk learns a
   correctness-bearing representation. Verified change points: head at
   `torch_transition.py:97-106`, BCE term at `objective.py:217-229`, scorer
   fusion at `scorer.py:614-653`.
3. **WS-C2/C4 as enablers** — EMA target encoder and the output-value
   auxiliary head (predict the output bucket from `z_pred_after`) to force the
   latent to encode output semantics / spec alignment, a likely prerequisite
   for correctness to be learnable at all.
4. **Measure on WS-D** — the unsaturated benchmark is now the instrument:
   success = a calibrated p_pass that reranks above the lexical baseline
   (~0.30) with bootstrap-CI clearance.

Full evidence: `docs/benchmark/EXECUTION_V0_7_RESULTS_2026-06-04.md`,
`EXECUTION_V0_7_WSD_RESULTS_2026-06-04.md`,
`EXECUTION_V0_7_WSA_A1_RESULTS_2026-06-04.md`,
`EXECUTION_V0_7_WSA_A15_LATENT_PROBE_2026-06-04.md`, and the consolidated
`EXECUTION_V0_7_CONCLUSION_2026-06-04.md`.

## Related

- RFC-0014 (execution-trace substrate), RFC-0005 (objective and collapse
  diagnostics), RFC-0007 (retrieval and surprise evaluation), RFC-0008 (scorer
  and reranker), RFC-0006 (training runtime and configs).
- `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md`,
  `docs/benchmark/V0_6_RERANK_FULL_2026-06-01.md`,
  `docs/benchmark/V0_6_RERANK_PILOT_2026-06-01.md`.
- `docs/spec/11-llm-world-model-harness.md` (publication boundary).
