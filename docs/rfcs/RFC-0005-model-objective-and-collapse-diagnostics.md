# RFC-0005: Model Objective And Collapse Diagnostics

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

The primary objective is next-latent MSE plus SIGReg. In-batch retrieval loss
and no-action margin regularization are allowed only after the base objective is
instrumented and collapse/action-use diagnostics show why they are needed.

## Motivation

The core research claim depends on simple action-conditioned latent prediction.
Adding many auxiliary losses before measuring collapse and retrieval would make
results harder to interpret.

## Goals

- Train with `loss_pred + lambda_sig * loss_sig` for the initial smoke path.
- Log collapse diagnostics every validation interval.
- Define kill thresholds before large runs.
- Add retrieval loss only through a config flag and report it separately.
- Add no-action margin regularization only through a config flag and report it
  separately.

## Non-Goals

- EMA or frozen-teacher target as the default v0.1 path.
- LLM-generated reward or judge loss.
- Test-execution reward as a training signal.

## Proposed Design

Loss:

```python
loss_pred = torch.mean((z_pred_after - z_after) ** 2)
loss_sig = sigreg(torch.stack([z_before, z_after, z_pred_after], dim=0))
loss = loss_pred + cfg.loss.sigreg_weight * loss_sig
```

Optional retrieval auxiliary:

```python
scores = cosine_similarity(z_pred_after[:, None, :], z_after[None, :, :]) / temperature
loss_retrieval = cross_entropy(scores, torch.arange(batch_size, device=scores.device))
loss = loss + cfg.loss.retrieval_weight * loss_retrieval
```

Optional action-use margin auxiliary:

```python
no_action_dist = torch.mean((z_before.detach() - z_after.detach()) ** 2, dim=-1)
pred_dist = torch.mean((z_pred_after - z_after.detach()) ** 2, dim=-1)
loss_action_use = torch.relu(pred_dist - no_action_dist + cfg.loss.action_use_margin).mean()
loss = loss + cfg.loss.action_use_margin_weight * loss_action_use
```

Diagnostics:

```python
@dataclass(frozen=True)
class CollapseReport:
    effective_rank: float
    effective_rank_ratio: float
    per_dim_variance_min: float
    per_dim_variance_median: float
    pairwise_cosine_mean: float
    embedding_norm_mean: float
    nearest_neighbor_entropy: float
```

Kill thresholds:

- `effective_rank_ratio < 0.20`;
- median per-dimension variance near zero for two validation windows;
- nearest-neighbor entropy collapse relative to the fixture baseline;
- NaN or inf in loss, embeddings, or gradients.

Default sweep:

```text
sigreg_weight: 0.05, 0.09, 0.15
retrieval_weight: 0.00 for base, 0.05 only after base diagnostics
action_use_margin_weight: 0.00 for base, 0.25 only after no-action dominance diagnostics
action_use_margin: 0.02 for the first action-use follow-up sweep
```

Failure modes:

- collapse: stop run and write kill report;
- action-insensitive embeddings: trigger shuffled-action analysis;
- retrieval loss dominates MSE: fail config validation if retrieval weight exceeds
  configured cap.

## Alternatives Considered

- Contrastive-only training: rejected because it removes the next-latent
  prediction contract.
- EMA target encoder: deferred because it complicates the LeWM-style claim.
- Patch correctness reward: rejected for v1 because it mixes representation
  learning with task-specific execution.

## Drawbacks

- MSE in latent space may be too weak without retrieval or no-action margin
  regularization.
- SIGReg hyperparameters still need tuning.
- Kill thresholds are conservative and may stop runs that could recover.

## Migration / Rollout

1. Add collapse report computation.
2. Add base MSE+SIGReg training.
3. Add kill report output.
4. Add retrieval loss behind config flag after base smoke report.
5. Add action-use margin behind a config flag after no-action dominance evidence.

## Testing Strategy

- Unit test finite SIGReg on random embeddings.
- Unit test effective rank on known low-rank tensors.
- Fixture training test that loss decreases on deterministic transforms.
- Kill test with forced constant embeddings.
- Config test that retrieval weight cannot be enabled without report output.
- Config test that action-use margin cannot be enabled without explicit margin
  and weight fields.

## Open Questions

- Owner: maintainers. Target: 2026-06-30. Should state-after encoder gradients be
  detached for the first smoke runs? Resolution: run paired tiny experiments and
  record collapse reports.

## References

- `docs/spec/05-observability.md#metrics`
- `docs/spec/07-testing-strategy.md#ml-regression-tests`
- `docs/rfcs/RFC-0001-lewm-compatible-code-transition-model.md`
