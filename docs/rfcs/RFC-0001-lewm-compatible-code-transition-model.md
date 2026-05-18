# RFC-0001: LeWM-Compatible Code Transition Model

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

CodeLeWM keeps the LeWM observation/action/next-observation architecture and
replaces image observations with Python `CodeState` observations and robot
actions with edit actions. This locks the project as a latent transition model
instead of a supervised edit classifier or patch generator.

## Motivation

The project thesis in `docs/spec/00-overview.md#thesis` requires a model that
predicts after-state latents from before-state latents and actions. The existing
repository already contains `JEPA`, `ARPredictor`, `Embedder`, `SIGReg`,
Hydra-style configs, and Lightning-style training glue. Reusing that shape gives
the project a narrow implementation path and a clear ablation surface.

## Goals

- Preserve encoder, action encoder, predictor, projector, prediction projector,
  and SIGReg boundaries.
- Support one-step Python edit transitions for v0.1.
- Keep action-conditioned prediction testable against no-action and
  shuffled-action baselines.
- Make image-control components replaceable without changing the public training
  contract.

## Non-Goals

- Reimplement a generic sequence-to-sequence patch generator.
- Preserve image-specific data loaders as public CodeLeWM APIs.
- Add multi-step edit trajectories before one-step retrieval works.

## Proposed Design

Public model interface:

```python
class CodeTransitionModel(nn.Module):
    encoder: CodeStateEncoder
    action_encoder: TextActionEncoder | AbstractActionEncoder
    predictor: nn.Module
    projector: nn.Module
    pred_proj: nn.Module

    def encode_state(self, batch: CodeStateBatch) -> Tensor: ...
    def encode_action(self, batch: ActionBatch) -> Tensor: ...
    def predict_after(self, z_before: Tensor, action_emb: Tensor) -> Tensor: ...
    def transition_energy(self, z_pred: Tensor, z_after: Tensor) -> Tensor: ...
```

Tensor contract:

```text
state input_ids:        int64 [B, 1024]
state attention_mask:   bool  [B, 1024]
state segment_ids:      int64 [B, 1024]
action input_ids:       int64 [B, 256] for text, [B, 192] for abstract
z_before:               float [B, D]
action_emb:             float [B, D]
z_pred_after:           float [B, D]
z_after:                float [B, D]
D:                      256 for v0.1
```

Training computes:

```python
z_b = model.encode_state(batch.state_before)
z_a = model.encode_state(batch.state_after)
a = model.encode_action(batch.action_text)
z_pred = model.predict_after(z_b, a)
loss_pred = mse(z_pred, z_a.detach_or_not_per_config)
loss_sig = sigreg(stack(z_b, z_a, z_pred))
loss = loss_pred + cfg.loss.sigreg_weight * loss_sig
```

For v0.1, `history_size=1` and `num_preds=1`. Multi-step edit sequences are a
future extension because commit histories are irregular and not fixed-rate.

Failure modes:

- shape mismatch: raise `SchemaError`;
- device mismatch: move inputs through a single `to_device` utility or raise
  `CheckpointCompatibilityError` when impossible;
- collapse: fail the training gate through RFC-0005 diagnostics;
- action leakage: fail dataset validation if `state_after` appears in headline
  action fields.

## Alternatives Considered

- Intent classifier: rejected because fixed edit menus are weaker than natural
  text and abstract scripts for open-ended code edits.
- Patch generator: rejected for v0.1 because generation would obscure whether
  the latent transition model learns useful dynamics.
- Static code-change encoder: rejected as the main architecture because it does
  not test action-conditioned next-state prediction.

## Drawbacks

- One-step transitions may underfit edits that require broader history.
- A LeWM-compatible architecture may need adaptation for long code tokens.
- Keeping the model compact limits headline raw capacity.

## Migration / Rollout

1. Move project-specific modules under `codelewm.model`.
2. Add adapter code so existing `JEPA`/`ARPredictor` roots can remain importable.
3. Add tiny config `config/train/codelewm_tiny.yaml`.
4. Retire image-control defaults from project docs once CodeLeWM configs exist.

## Testing Strategy

- Unit test tensor shapes for state/action encoders.
- Unit test `transition_energy` against hand-computed MSE.
- CPU one-batch training test with finite loss.
- Regression test that shuffled actions score worse than true actions on
  deterministic synthetic fixtures.
- Checkpoint load/save test with manifest compatibility.

## Open Questions

- Owner: maintainers. Target: 2026-06-30. Should `z_after` be detached in the
  prediction loss for early training stability? Resolution: run a fixture
  experiment in RFC-0005's collapse diagnostics before v0.1 release.

## References

- `docs/spec/00-overview.md#thesis`
- `docs/spec/01-architecture.md#system-shape`
- `docs/spec/03-data-model.md#core-types`
- `docs/rfcs/RFC-0005-model-objective-and-collapse-diagnostics.md`
