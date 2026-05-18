# RFC-0007: Retrieval And Surprise Evaluation

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

The primary evaluation is action-conditioned after-state retrieval. The secondary
evaluation is patch-surprise ranking. Every headline report must include random,
lexical, no-action, and shuffled-action baselines.

## Motivation

CodeLeWM claims learned latent edit dynamics only if the action-conditioned
prediction ranks the true after-state above hard negatives and beats models that
ignore or corrupt the action.

## Goals

- Report `Recall@1`, `Recall@5`, `Recall@10`, MRR, and median rank.
- Build easy and hard candidate pools.
- Include no-action and shuffled-action ablations.
- Use patch-surprise AUC for candidate after-state scoring.
- Slice reports by source, edit size, and action view.

## Non-Goals

- Use hidden test pass rate as the first scientific metric.
- Claim patch generation quality.
- Rely on random negatives only.

## Proposed Design

Retrieval task:

```text
query:      state_before + action
target:     true state_after
candidates: target + hard negatives
score:      -||P(E(before), A(action)) - E(candidate_after)||^2
```

Metrics:

```python
@dataclass(frozen=True)
class RetrievalReport:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    median_rank: float
    baselines: Mapping[str, RetrievalMetrics]
    slices: Mapping[str, RetrievalMetrics]
```

Candidate pools:

- easy-1k: random held-out Python after-states;
- hard-1k: same source, edit-size bucket, and weak action cluster;
- hard-10k: larger held-out pool for v1.0;
- repo-heldout: only examples from unseen repos.

Patch surprise:

```python
energy = squared_l2(z_pred_after, z_candidate_after)
```

Report pairwise AUC and true-after rank against decoys:

- random after-state;
- same-file wrong after-state;
- same action cluster wrong after-state;
- mutated after-state;
- syntax-valid bug injection fixture.

Baselines:

- random rank;
- BM25/TF-IDF lexical retrieval;
- no-action transition model;
- shuffled-action transition model;
- abstract-action model;
- patch-action upper bound when available.

Failure modes:

- hard-negative pool too easy: report rejected and regenerated;
- baseline missing: headline report invalid;
- split leakage detected: evaluation aborts.

## Alternatives Considered

- Clone retrieval: rejected as primary because it can be solved by static
  similarity.
- Code classification probes: rejected as primary because they do not test edit
  transitions.
- Full benchmark harness first: rejected because candidate generation and tests
  would add uncontrolled variables.

## Drawbacks

- Retrieval may understate usefulness for reranking small candidate sets.
- Hard negatives are sensitive to construction quality.
- Lexical baselines can be strong on commit-message-like actions.

## Migration / Rollout

1. Implement easy-1k retrieval.
2. Add hard-negative sampler.
3. Add lexical, no-action, and shuffled-action baselines.
4. Add patch-surprise fixtures.
5. Add v1.0 hard-10k and repo-heldout reports.

## Testing Strategy

- Metric unit tests with known rank arrays.
- Negative sampler tests that exclude the true target.
- Leakage test that rejects train examples in candidate pools.
- Fixture test where true action beats shuffled action.
- Report schema validation.

## Open Questions

- Owner: maintainers. Target: 2026-07-01. Which frozen code embedding baselines
  should be included in v1.0? Resolution: add after data contracts are stable and
  report exact model identifiers in the benchmark manifest.

## References

- `docs/spec/07-testing-strategy.md#evaluation-gates`
- `docs/spec/00-overview.md#v10-research-artifact`
- `docs/rfcs/RFC-0004-action-views-and-encoders.md`
