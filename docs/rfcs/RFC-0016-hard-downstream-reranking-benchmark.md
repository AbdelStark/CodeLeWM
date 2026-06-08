# RFC-0016: Hard Anti-Saturation Downstream Reranking Benchmark

- Status: Draft
- Authors: CodeLeWM maintainers
- Created: 2026-06-08
- Target milestone: v1.5

## Summary

The final v1.0 paper leaves CodeLeWM in a mixed state: HumanEval WS-D shows a
narrow downstream reranking win, but MBPP-Plus is saturated because CodeLeWM,
no-action, and lexical controls all reach pass@1 `1.0000`. That result is
honest, but it cannot answer the stronger question:

Can a learned code world-model score add value when simple surface and
before-state baselines are not already enough?

This RFC defines a harder follow-up benchmark, the hard anti-saturation
downstream reranking benchmark. It is a downstream candidate-ranking experiment,
not a new positive claim. The benchmark deliberately selects or constructs
candidate pools where no-action and lexical baselines have headroom, where
wrong candidates are plausible, and where the correct ranking requires behavior
or action sensitivity rather than surface similarity alone.

## Motivation

The v1.0 paper-demo replay is useful but not decisive. It has one positive
HumanEval WS-D slice and one saturated MBPP-Plus counterexample. Saturation
means a slice cannot measure added value: when no-action or lexical controls
already choose a passing candidate at rank 1 on nearly every problem, CodeLeWM
cannot demonstrate lift even if its score is good.

The next experiment must therefore make saturation a first-class failure mode.
It should reject candidate pools where simple baselines are perfect, not report
them as model wins. A credible benchmark should make the following cases
visible:

- no-action baits, where the unchanged or minimally changed before-state looks
  plausible but is wrong;
- lexical baits, where the wrong patch has high prompt/reference overlap;
- partial fixes, where the candidate edits the right region but misses an edge
  case;
- wrong-symbol or wrong-branch fixes, where the patch is locally plausible but
  changes the wrong behavior;
- over-broad fixes, where the patch solves the visible example while damaging
  neighboring behavior;
- LLM candidates that are valid, parseable, and realistic enough that original
  LLM order is a meaningful baseline.

## Scientific Question

Does CodeLeWM improve downstream candidate-patch reranking over no-action,
lexical, LLM-order, random, shuffled-action, retrieval-prior, and static
heuristic controls on public-safe candidate pools that are explicitly
non-saturated for those controls?

## Goals

- Build a public-safe, manifest-backed reranking benchmark with at least `100`
  locked test problems per headline slice.
- Include candidate pools with at least one passing candidate and at least two
  plausible failing candidates per problem.
- Record difficulty diagnostics before model scoring, including no-action,
  lexical, LLM-order, random, static-check, and pass-rate headroom.
- Reject or quarantine slices where no-action or lexical pass@1 exceeds the
  saturation ceiling.
- Evaluate CodeLeWM from downloaded or checked-in trusted artifacts against the
  exact same candidate pools and labels used by every baseline.
- Publish a claim gate that can open only if CodeLeWM beats the required
  baselines with confidence intervals excluding zero across the locked slices
  and seeds.

## Non-Goals

- Do not claim broad coding improvement from fixture, dev, prompt-tuning, or
  saturated slices.
- Do not execute candidate code during model scoring.
- Do not send private repository content, secrets, `.env` files, or unscanned
  prompts to an LLM provider.
- Do not tune prompts, candidate generators, or claim thresholds on the final
  locked test split.
- Do not hide failed slices. Saturated, invalid, or missing-baseline slices are
  explicit benchmark outcomes.

## Benchmark Design

### Candidate Sources

The benchmark may combine several candidate sources, but every source must be
recorded per candidate:

- deterministic semantic mutants from accepted references;
- no-action and near-no-action baits;
- partial-fix and edge-case-missing candidates;
- wrong-symbol, wrong-branch, and over-broad candidates;
- LLM-generated unified diffs captured as `codelewm.llm_candidate_pack.v1`
  artifacts after context redaction and secret scanning;
- optional project-owned synthetic scenarios when licensing or public-source
  availability blocks a needed failure mode.

Each problem pool must include:

- one task prompt and before-state;
- between `6` and `12` candidates;
- at least one passing candidate after label construction;
- at least two failing candidates from distinct hard-negative classes when
  source coverage permits;
- parser/apply status for every candidate;
- label provenance and source/license status.

### Anti-Saturation Filter

The benchmark writes `codelewm.downstream_anti_saturation_report.v1` before
CodeLeWM scoring. A slice is eligible for headline claims only when:

- `problem_count >= 100` on the locked test split;
- random pass@1 is within the configured pool-size expectation;
- no-action pass@1 is below `0.85`;
- lexical pass@1 is below `0.85`;
- LLM-order pass@1 is below `0.90`;
- at least `70%` of problems contain two or more failing candidate classes;
- candidate parser/apply failure rates are reported and do not dominate the
  slice;
- no split-leakage, source-license, manifest, or secret-scan gate is open.

Slices that fail these checks remain publishable as diagnostic evidence but
cannot open a positive downstream usefulness claim.

### Required Baselines

Every headline slice must report:

- random order;
- LLM original order;
- lexical/token-overlap order;
- static heuristic order, including parser/apply/check summaries when
  available;
- no-action score;
- shuffled-action score;
- CodeLeWM transition-energy order;
- retrieval-prior-only order when a verified index is available;
- final CodeLeWM score/ensemble order when a configured ensemble is used;
- `p_pass` order only if a standalone score key is serialized in every
  downstream row.

Missing baselines must be typed `blocked` or `not_recorded`; they must not be
silently omitted.

### Metrics

The primary metrics are:

- pass@1;
- pass@k for the configured candidate-pool size;
- MRR to the first passing candidate;
- median rank of the first passing candidate;
- CodeLeWM lift over no-action, lexical, and LLM-order;
- bootstrap confidence intervals over problem-level lift;
- valid candidate rate;
- parser/apply failure rate;
- check-pass rate where sandbox labels are available;
- per-source and per-hard-negative-class slices.

Secondary diagnostics include score calibration, ROC-AUC over candidate labels,
expected calibration error when a probability score exists, and disagreement
tables showing where CodeLeWM improves or regresses relative to each baseline.

## Data And Split Policy

The benchmark must have explicit train/dev/test roles:

- training data may be used to fit candidate-generation heuristics or
  calibrators;
- dev data may be used to tune prompt templates, saturation thresholds, and
  static heuristics;
- test data is locked before final model scoring and claim review.

No source problem, reference solution, generated negative, or LLM candidate pack
may appear in more than one split. LLM candidates generated after seeing test
labels are invalid for headline claims. Any live provider use must record model
slug, SDK version, provider routing, retry policy, redaction status, and
secret-scan evidence.

## Execution And Security Boundary

Candidate code is untrusted. The model-scoring path remains text-only and must
not import, execute, or test-run candidate patches. Pass/fail labels may be
constructed only in an explicit sandbox-labeling step using the existing
allowlist, disposable checkout, timeout, output-limit, redaction, manifest, and
secret-scan contracts.

Published benchmark artifacts must include:

- source/license gate report;
- split-leakage report;
- anti-saturation report;
- benchmark manifest;
- label-construction report;
- rerank report;
- claim-gate report;
- secret-scan report;
- artifact index and checksums.

## Claim Gate

The aggregate downstream claim may open only when all headline slices satisfy:

- `problem_count >= 100`;
- anti-saturation gate `eligible=true`;
- CodeLeWM pass@1 is strictly above no-action, lexical, and LLM-order;
- CodeLeWM MRR is strictly above no-action, lexical, and LLM-order;
- lift confidence intervals over no-action, lexical, and LLM-order exclude
  zero;
- the result holds on at least two model seeds or one model seed plus a
  predeclared reproducibility check when compute is explicitly bounded;
- no required baseline is missing without a typed blocker;
- manifest verification, checkpoint trust, source/license, split-leakage, and
  secret-scan gates pass.

If any condition fails, public wording remains diagnostic:

> The hard downstream benchmark executed and identified which baselines or
> slices block a positive claim; CodeLeWM does not yet support a broad
> downstream coding-usefulness claim.

## Reviewer-Facing Evaluation Matrix

| Claim | Metric | Dataset/env | Baselines | Controls | Expected result | Falsifying result |
| --- | --- | --- | --- | --- | --- | --- |
| CodeLeWM adds downstream value beyond simple controls | pass@1 and MRR lift | locked anti-saturation test split, `>=100` problems per headline slice | no-action, lexical, LLM-order, random, shuffled-action, retrieval-prior | identical candidate pools, two seeds, bootstrap CIs | CodeLeWM beats no-action, lexical, and LLM-order with CIs above zero | any required baseline ties or beats CodeLeWM |
| The benchmark avoids saturation | anti-saturation report | dev and locked test split | no-action, lexical, LLM-order, random | saturation ceilings and candidate-class coverage | no-action and lexical below `0.85`; LLM-order below `0.90` | simple baselines are near-perfect or candidate pools lack failing classes |
| Improvements are not invalid-candidate artifacts | valid/apply/check rates and slices | all test candidates | static heuristic, parser/apply status | candidate-level failure accounting | lift remains on valid candidate slices | lift comes only from invalid, unparsable, or unchecked candidates |
| Action signal matters | shuffled-action and no-action deltas | same-before and near-before pools | shuffled-action, no-action, lexical | hard-negative classes tied to action necessity | CodeLeWM beats no-action and shuffled-action | before-state or shuffled-action controls explain the result |

## Rollout

1. Lock this RFC, the spec text, roadmap, issue tracker, and docs tests.
2. Implement the benchmark pack schema/config and anti-saturation diagnostics.
3. Build deterministic hard-negative candidate packs with public-safe labels.
4. Add LLM candidate-pack ingestion under the existing OpenRouter/candidate-pack
   security contract.
5. Run baseline and CodeLeWM scoring from trusted downloaded or checked-in
   artifacts.
6. Publish the benchmark artifact set, claim audit, and paper addendum.

## References

- `docs/spec/11-llm-world-model-harness.md`
- `docs/spec/05-observability.md`
- `docs/spec/06-security.md`
- `docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md`
- `docs/rfcs/RFC-0015-v0-7-execution-substrate-improvements.md`
- `docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md`
- `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md`
- `docs/papers/codelewm_final_paper.tex`
