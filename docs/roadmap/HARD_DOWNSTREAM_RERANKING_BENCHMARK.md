# Hard Downstream Reranking Benchmark Roadmap

Last updated: 2026-06-08

Tracker: #417. RFC:
`docs/rfcs/RFC-0016-hard-downstream-reranking-benchmark.md`.

This roadmap defines the follow-up experiment after the final v1.0 paper. The
goal is to test whether CodeLeWM adds downstream ranking value on candidate
pools that are deliberately harder than the saturated MBPP-Plus WS-D replay.

## Why This Exists

The v1.0 paper-demo evidence is mixed:

- HumanEval WS-D shows a narrow positive reranking slice;
- MBPP-Plus WS-D is saturated because CodeLeWM, no-action, and lexical controls
  all reach pass@1 `1.0000`;
- the aggregate downstream claim remains closed.

The hard benchmark answers the next falsifiable question: can CodeLeWM beat
simple controls when those controls are not already perfect?

## Benchmark Thesis

The benchmark should make easy shortcuts fail. A headline slice is useful only
when no-action, lexical, and LLM-order baselines leave measurable headroom.
Candidate pools must contain realistic, plausible wrong patches, not only easy
syntax failures or one-token mutants that a surface heuristic can reject.

## Required Candidate Classes

Each headline slice should include candidate pools with as many of these classes
as source coverage permits:

- passing reference or accepted candidate;
- no-action or near-no-action bait;
- partial fix that misses an edge case;
- wrong-symbol or wrong-branch fix;
- over-broad fix that changes neighboring behavior;
- deterministic semantic mutant;
- LLM-generated valid candidate;
- parser/apply failure candidate, recorded only for failure accounting and not
  as the source of a positive claim.

## Minimum Headline Gate

A positive downstream claim may open only when:

- locked test `problem_count >= 100` per headline slice;
- candidate pools contain `6` to `12` candidates;
- no-action pass@1 is below `0.85`;
- lexical pass@1 is below `0.85`;
- LLM-order pass@1 is below `0.90`;
- CodeLeWM beats no-action, lexical, and LLM-order on pass@1 and MRR;
- bootstrap confidence intervals over all three lifts exclude zero;
- results hold across at least two model seeds or an explicitly documented
  compute-bounded reproducibility check;
- source/license, split-leakage, manifest verification, checkpoint trust, and
  secret scans pass.

If any condition fails, the result remains diagnostic and the paper wording must
say which gate blocked the claim.

## Deliverables

| Order | Issue | Slice | Status |
| --- | --- | --- | --- |
| 0 | #418 | Spec, roadmap, and tracker lock | open |
| 1 | #419 | Benchmark schema/config and anti-saturation report | open |
| 2 | #420 | Public-safe hard-negative candidate-pack builder | open |
| 3 | #421 | LLM candidate-pack ingestion and redaction-preserving capture | open |
| 4 | #422 | Baseline and CodeLeWM scoring/evaluation gate | open |
| 5 | #423 | Artifact publication, claim audit, and paper addendum | open |

## Acceptance Matrix

| Claim | Metric | Dataset/env | Baselines | Controls | Expected result | Falsifying result |
| --- | --- | --- | --- | --- | --- | --- |
| CodeLeWM adds value beyond simple controls | pass@1 and MRR lift | locked anti-saturation test split | no-action, lexical, LLM-order, random, shuffled-action, retrieval-prior | identical candidate pools, two seeds, bootstrap CIs | CodeLeWM beats no-action, lexical, and LLM-order with CIs above zero | any required baseline ties or beats CodeLeWM |
| Benchmark has headroom | anti-saturation report | dev and test slices | no-action, lexical, LLM-order, random | saturation ceilings and hard-negative class coverage | no-action and lexical below `0.85`, LLM-order below `0.90` | simple baselines are near-perfect |
| Lift is not a failure-accounting artifact | valid/apply/check rates | all candidates | parser/apply/static baselines | candidate-level failure slices | lift remains on valid candidate slices | lift appears only on invalid or unchecked candidates |
| Action signal matters | no-action and shuffled-action deltas | same-before and near-before pools | no-action, shuffled-action, lexical | action-necessity hard negatives | CodeLeWM beats no-action and shuffled-action | before-state or shuffled-action controls explain the result |

## Security And Publication Boundary

Candidate code remains untrusted. Label construction may use sandbox execution
only in the existing allowlisted disposable-checkout path. Model scoring remains
text-only and must not import or execute candidate patches.

Public wording may say the benchmark was designed to test the current v1.0
claim blocker. It may not say CodeLeWM improves generated code until the hard
benchmark claim gate opens.

## Validation Template

Representative local gates for implementation PRs:

```bash
uv run pytest tests/
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests
uv run codelewm manifest verify --manifest <artifact>/manifest.json --json
uv run codelewm secret-scan <artifact> --json
git diff --check
```

HF or remote scoring work must use downloaded artifacts, not transient job
directories, before any public claim audit is written.
