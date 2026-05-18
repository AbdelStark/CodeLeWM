# RFC-0004: Action Views And Encoders

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

CodeLeWM uses three action views: `action_text` for headline inference,
`action_abs` for deterministic structural supervision and ablation, and
`action_patch` for diagnostic upper bounds. Fixed edit-intent labels are not part
of v0.1.

## Motivation

Action design is the central difference between robotics trajectories and code
edit trajectories. Code does not have continuous control vectors, so the model
needs action views that separate realistic inference signals from leaky
diagnostics.

## Goals

- Use natural-language text actions as the user-facing inference action.
- Derive abstract action scripts from AST/CST before/after differences.
- Treat patch actions as leaky and diagnostic.
- Keep all action encoders projected to latent dim `256` in v0.1.
- Validate that action conditioning improves retrieval.

## Non-Goals

- Hand-authored edit taxonomy labels.
- Patch-action headline inference.
- Human annotation pipeline.
- Full semantic program differencing in v0.1.

## Proposed Design

Action schema:

```python
@dataclass(frozen=True)
class EditAction:
    text: str
    abstract: tuple[str, ...]
    patch: str | None
```

Text action sources:

- commit subject;
- first 256 characters of commit body;
- synthetic transform template;
- future user instruction.

Abstract action token format:

```text
OP_UPDATE NODE_Return PATH_DEPTH_4 OLD_Call NEW_CallWithKeyword SIZE_SMALL
OP_INSERT NODE_ExceptHandler PATH_DEPTH_2 SIZE_MEDIUM
OP_DELETE NODE_Assign PATH_DEPTH_3 SIZE_SMALL
```

Rules:

- abstract actions include operation type, AST/CST node type, path depth bucket,
  old shape token, new shape token, scope token, and size bucket;
- abstract actions must not include full inserted after-code lines;
- patch actions can include diff markers and changed spans but are marked leaky.

Encoder interfaces:

```python
class TextActionEncoder(nn.Module):
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor: ...

class AbstractActionEncoder(nn.Module):
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor: ...
```

v0.1 architecture:

```text
TextActionEncoder:     4 layers, dim 256, heads 8, max length 256
AbstractActionEncoder: 3 layers, dim 256, heads 8, max length 192
PatchActionEncoder:    optional, max length 512
```

Failure modes:

- empty text action: row is filtered unless synthetic template is present;
- abstract extraction failure: row can remain for text-action training but is
  excluded from abstract-action ablations;
- patch action configured for headline eval: evaluation fails.

## Alternatives Considered

- Eight fixed intent labels: rejected because they under-specify real edit
  actions and do not match the transition-model thesis.
- Patch-only actions: rejected because they leak the answer and make inference
  claims invalid.
- Text-only actions: rejected because abstract scripts provide a stronger
  diagnostic for whether the architecture can learn edit dynamics.

## Drawbacks

- Abstract action extraction can be brittle.
- Text actions from commits may be noisy or underspecified.
- Multiple encoders add evaluation matrix complexity.

## Migration / Rollout

1. Implement text action tokenization.
2. Implement abstract operation extraction for common Python nodes.
3. Add action-view selector to training config.
4. Add patch-action diagnostic only after text and abstract paths pass fixtures.

## Testing Strategy

- Unit tests that abstract actions omit inserted after-code text.
- Fixture tests for insert, delete, update, rename, and exception-handler edits.
- Retrieval tests for true, no-action, and shuffled-action variants.
- Eval test that patch action cannot be selected for headline reports.

## Open Questions

- Owner: maintainers. Target: 2026-07-15. Should learned action prototypes be
  exposed in the harness? Resolution: cluster `action_abs` only after v1.0
  retrieval reports show stable action neighborhoods.

## References

- `docs/spec/03-data-model.md#core-types`
- `docs/spec/00-overview.md#goals`
- `docs/rfcs/RFC-0007-retrieval-and-surprise-evaluation.md`
