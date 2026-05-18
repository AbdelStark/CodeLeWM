# RFC-0003: CodeState Schema And Normalization

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

`CodeState` is a bounded Python context capsule containing the changed primary
symbol or region plus deterministic local context. It is designed to preserve
semantic cues while preventing whole-repository sequence blowup.

## Motivation

The PRD-derived design rejects whole-repository encoding for v1. The architecture
spec requires a stable input contract that can be parsed, tokenized, truncated,
and reproduced without human judgment.

## Goals

- Encode the changed function/method first, then class, region, or small file.
- Preserve imports, signatures, enclosing class context, and local callee
  signatures.
- Keep state length at 1024 tokens in v0.1.
- Avoid arbitrary tail clipping.
- Emit masks for changed hunks, segments, identifiers, and literals when
  available.

## Non-Goals

- Full data-flow graph construction in v0.1.
- Whole-repository embeddings.
- Identifier anonymization as the only state view.
- Including full caller/callee bodies by default.

## Proposed Design

Builder interface:

```python
def build_code_state(
    source: str,
    path: str,
    changed_ranges: Sequence[LineRange],
    cfg: CodeStateConfig,
) -> CodeState: ...
```

Priority order:

```text
1. changed function or method
2. changed class
3. changed top-level file region
4. whole small file if within token budget
```

Pack format:

```text
<LANG python>
<PATH package/module.py>
<SYMBOL package.module.Class.method>
<KIND method>
<IMPORTS> ...
<ENCLOSING_CLASS> ...
<SIBLING_SIGNATURES> ...
<CALLEE_SIGNATURES> ...
<PRIMARY> ...
```

Truncation order:

1. keep signature and decorators;
2. keep changed hunk vicinity;
3. keep return statements and exception handlers;
4. drop long literals;
5. drop long docstrings and comments unless the row is a documentation edit;
6. drop lower-priority sibling/callee signatures.

Normalization:

- parse with Python AST and a CST parser where formatting preservation matters;
- normalize whitespace;
- preserve public and local identifiers in the main view;
- replace large string and number literals with typed placeholders;
- remove comments/docstrings for main model input unless configured otherwise;
- optionally emit an identifier-normalized auxiliary view.

Failure modes:

- parse failure: row is filtered;
- no changed symbol found: fall back to region or small file;
- over budget after truncation: row is filtered;
- ambiguous multi-symbol edit: choose primary by changed-line overlap and record
  `multi_symbol=true`.

## Alternatives Considered

- Raw whole file: rejected because sequence length and noise hide the changed
  transition.
- AST-only serialization: rejected because code tokens and formatting-adjacent
  cues still matter for edit retrieval.
- Full anonymization: rejected because identifiers, imports, and API names carry
  semantic action signal.

## Drawbacks

- Context capsules may miss distant dependencies.
- Parser behavior can differ across Python syntax versions.
- Truncation policy introduces a modeling prior that must be reported.

## Migration / Rollout

1. Implement function/method and small-file extraction.
2. Add enclosing class and import context.
3. Add sibling and callee signatures.
4. Add token masks and segment IDs.
5. Add optional auxiliary normalized identifier view.

## Testing Strategy

- Fixture tests for functions, methods, classes, top-level regions, decorators,
  async functions, nested functions, and syntax errors.
- Token budget tests that preserve signatures and changed hunks.
- Snapshot tests for stable pack text.
- Property test that normalization is deterministic.

## Open Questions

- Owner: maintainers. Target: 2026-06-30. Should v1.0 include selected caller
  signatures? Resolution: evaluate failure slices where local callee context is
  insufficient before adding more context.

## References

- `docs/spec/03-data-model.md#core-types`
- `docs/spec/03-data-model.md#hdf5-layout`
- `docs/rfcs/RFC-0002-edit-transition-dataset.md`
