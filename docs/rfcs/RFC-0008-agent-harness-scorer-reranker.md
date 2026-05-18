# RFC-0008: Agent Harness Scorer And Reranker

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

The v0.1 harness exposes CodeLeWM as a local scorer and reranker for candidate
code changes. It accepts candidates produced elsewhere and ranks them by latent
transition energy plus optional retrieval and risk terms.

## Motivation

The project's credible first showcase is deciding which candidate edit best
matches a requested transition. This isolates the learned world-model component
from patch generation and makes the output mechanically testable.

## Goals

- Provide `codelewm score` and `codelewm rerank`.
- Return JSON score records with artifact lineage.
- Batch candidate scoring.
- Optionally include nearest historical edits from a local index.
- Never execute candidate code.

## Non-Goals

- Generate patches.
- Run project test suites.
- Modify the user's working tree.
- Upload code to remote services.

## Proposed Design

Request schemas:

```python
@dataclass(frozen=True)
class ScoreRequest:
    before: CodeState
    action_text: str
    candidate_after: CodeState
    checkpoint: Path
    index: Path | None = None

@dataclass(frozen=True)
class RerankRequest:
    before: CodeState
    action_text: str
    candidates: tuple[CandidatePatch, ...]
    checkpoint: Path
    index: Path | None = None
```

Score:

```python
transition_energy = squared_l2(
    model.predict_after(model.encode_state(before), model.encode_action(action)),
    model.encode_state(candidate_after),
)
final_score = transition_energy + alpha * retrieval_prior + beta * risk_penalty
```

Default `alpha=0.0` and `beta=0.0` for v0.1 until index and risk reports are
validated.

CLI output:

```json
{
  "schema_version": "codelewm.score.v1",
  "candidate": "candidate_003.patch",
  "transition_energy": 0.42,
  "retrieval_prior": null,
  "risk_penalty": null,
  "final_score": 0.42,
  "warnings": []
}
```

Failure modes:

- candidate cannot apply cleanly: score command rejects candidate with structured
  error;
- candidate after-state cannot parse: candidate receives error result and is
  ranked after valid candidates;
- checkpoint incompatible: command aborts.

## Alternatives Considered

- Direct patch generation: rejected because it turns the project into a coding
  model instead of a transition scorer.
- Test execution harness: deferred because it violates the non-execution default
  and adds benchmark-specific dependencies.
- Retrieval-only showcase: rejected because scoring candidates is a more direct
  use of the transition model.

## Drawbacks

- Usefulness depends on candidate quality from external generators.
- Applying patches safely requires careful file handling.
- Energy values need calibration before they can be interpreted globally.

## Migration / Rollout

1. Implement scoring from before/instruction/after files.
2. Add patch directory reranking with dry-run patch application.
3. Add local index lookup.
4. Add risk penalty only after anomaly metrics are validated.

## Testing Strategy

- CLI fixture score test.
- Rerank test with one true after-state and decoys.
- Invalid patch and invalid syntax tests.
- JSON schema test.
- Non-execution test that malicious code is never imported or run.

## Open Questions

- Owner: maintainers. Target: 2026-07-15. Which candidate patch format should be
  the stable v1.0 input: unified diff, after-file directory, or both? Resolution:
  support after-file fixtures first, then add unified diff once patch application
  safety tests pass.

## References

- `docs/spec/02-public-api.md#cli`
- `docs/spec/06-security.md#non-execution-policy`
- `docs/rfcs/RFC-0007-retrieval-and-surprise-evaluation.md`
