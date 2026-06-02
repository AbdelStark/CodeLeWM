# v0.7 Phase 1 — Representation Probe Results (RFC-0015)

- Date: 2026-06-02
- Scope: RFC-0015 Phase 1; issues #338, #339, #341
- Compute: CPU only (no training). Frozen-encoder probes of the **existing v0.6
  execution checkpoints** (seed-42, seed-1729; both trusted) over a rebuilt,
  correctly-typed execution pack.

## Headline

Re-running the latent representation probe on a **correctly-typed** execution
pack reveals a **consistent 2-seed partial-positive that the v0.6 probe-data
bugs had masked**. On two targets — `output_type` and
`arithmetic_vs_string_vs_collection` — the predicted-after latent
(`z_pred_after`) beats **every** control by >0.05 on **both** seeds. The
published v0.6 verdict ("representation closed / `not_evaluable`") was an
artifact of the data/extractor bugs fixed in #344/#345/#346, not a property of
the model.

This is a **scoped partial-positive** per RFC-0014 (lines 178-180), not a full
semantic-axis claim and not a coding-usefulness claim.

## Method

Pack: rebuilt from MBPP (974 → 930 after dedup) through the hardened pipeline
(opt-in dedup #343, graceful sandbox resource-kill #346, privacy-safe bucket
precompute #345, `will_raise` field fix #344) → **1,859 records**, splits
train/val/test = 1579/92/188, with `output_magnitude_bucket` (832) and
`output_length_bucket` (786) precomputed. `codelewm eval execution-probe` with
all five RFC-0014 targets, against each v0.6 checkpoint, `--device cpu`.

## Probe evaluability: 2 → 5 targets

| pack | available targets | status |
|---|---|---|
| v0.6 published (stdout assumption / stripped output_repr) | **2/5** (`output_type`, `arithmetic`) | `not_evaluable` |
| v0.7 typed + bucket-precompute | **5/5** | `unsupported` (evaluable) |

The gate moved off `not_evaluable` purely from data fixes — the WS-B milestone.

## Per-target test accuracy (latent `z_pred_after` vs best control)

`beats_all` = latent exceeds the best of {lexical, majority, metadata_only,
no_action, random_latent, shuffled_action} by >0.05.

| target | seed-42 latent / best ctrl | seed-1729 latent / best ctrl | beats all, both seeds |
|---|---|---|---|
| **output_type** | 0.596 / 0.468 (lexical) | 0.580 / 0.468 (lexical) | **yes** |
| **arithmetic_vs_string_vs_collection** | 0.777 / 0.691 (no_action) | 0.718 / 0.665 (lexical) | **yes** |
| output_magnitude_bucket | 0.474 / 0.487 (majority) | 0.434 / 0.487 (majority) | no (beats lexical, not majority) |
| output_length_bucket | 0.619 / 0.679 (no_action) | 0.440 / 0.643 (lexical) | no |
| will_raise | 0.723 / 0.830 (no_action) | 0.755 / 0.824 (lexical) | no (degenerate: 6 exceptions) |

So **2 of 5 targets meet RFC-0014's "beat every control across ≥2 seeds"
condition.** `output_magnitude_bucket` beats the lexical control on both seeds
(+0.18 / +0.15) but not the trivial majority/metadata baselines, so it is not
claimed.

## Why the formal gate still reports `unsupported`

1. `codelewm/eval/latent_probe.py` hard-codes
   `positive_representation_claim_allowed = False` — the gate cannot formally
   open through this path regardless of the numbers.
2. The aggregate `semantic_structure_status` reason is "latent probes do not
   consistently beat listed controls on available targets" — 2/5 targets pass,
   3/5 do not, so the conservative aggregate stays `unsupported`.

The substantive per-target evidence (2 targets, every control, 2 seeds) is
nonetheless the strongest representation signal recorded for the substrate, and
it directly contradicts the "closed" framing that the buggy pack produced.

## Caveats (do not over-read)

- This is the **v0.6 model** on a **v0.7 MBPP-only** pack (the published v0.6
  pack was MBPP+APPS). One model family, two seeds.
- Test split n=188 (fewer per target); `output_type` is multi-class.
- `will_raise` is degenerate here (6 exception records); a robust binary target
  needs broadened source verdicts (wrong-answer / runtime-error submissions),
  which MBPP's single canonical solution per task cannot provide — APPS/CodeNet
  multi-submission data is the lever.
- This is **representation** evidence, scoped. It is **not** a downstream
  coding-usefulness claim (Phase-0 showed that axis remains flat/closed).

## Implications for v0.7

- **WS-B is validated**: the data overhaul makes the probe evaluable and
  surfaces a real, previously-hidden signal.
- **WS-C target sharpened**: the v0.7 architecture (transformer state encoder,
  EMA target, output-value auxiliary head) should push `output_magnitude` and
  `output_length` past the majority/metadata baselines too, and lift the
  margins, so more targets clear "beat every control".
- **A claim-gate question for the maintainer**: given consistent 2-seed evidence
  on 2 targets, the hard-coded `positive_representation_claim_allowed=False` and
  the aggregate logic warrant review — but flipping a public-claim gate is a
  deliberate claim-boundary decision and is **not** done here.

## Related

- RFC-0015; `V0_7_PHASE1_PROBE_DIAGNOSIS_2026-06-02.md`;
  `V0_7_PHASE0_RERANK_CALIBRATION_2026-06-02.md`
- PRs #343 (dedup), #344 (`will_raise`), #345 (bucket precompute), #346
  (sandbox resource-kill)
- Reproduce: build an MBPP pack via the hardened pipeline, then
  `codelewm eval execution-probe --checkpoint <v0.6 last.pt> --pack <pack.jsonl>
  --targets output_type,will_raise,output_magnitude_bucket,output_length_bucket,arithmetic_vs_string_vs_collection`
