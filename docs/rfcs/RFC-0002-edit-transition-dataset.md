# RFC-0002: Edit Transition Dataset

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

CodeLeWM builds schema-versioned Python edit transition datasets from real commit
sources and deterministic synthetic transforms. The dataset unit is a single
parse-valid transition with repository-level split assignment and explicit
filter, deduplication, and license metadata.

## Motivation

`docs/spec/03-data-model.md#core-types` defines `TransitionRecord` as the shared
contract for training, evaluation, and harness scoring. The old fixed-intent
dataset direction is replaced by action-conditioned transitions, so data quality
depends on deterministic extraction and leakage control rather than human labels.

## Goals

- Produce v0.1 smoke data with 40k training transitions and 5k validation
  transitions.
- Support real commit sources and deterministic synthetic transforms.
- Enforce parse-valid Python before and after states.
- Split by repository or source identity before tokenization.
- Emit manifests with filter counts and deduplication evidence.

## Non-Goals

- Public redistribution of raw source text before license policy is proven.
- Multi-language data.
- Human annotation as a required construction step.
- Whole-repository state extraction.

## Proposed Design

Source adapter interface:

```python
class SourceAdapter(Protocol):
    source: SourceKind
    def iter_records(self, spec: SourceSpec) -> Iterator[RawEditRecord]: ...
```

Build pipeline:

```python
def build_dataset(spec: DatasetBuildSpec) -> DatasetBuildReport:
    raw = chain(adapter.iter_records(source) for source in spec.sources)
    filtered = parse_filter(raw, spec.filter_policy)
    transitions = map(build_transition, filtered)
    transitions = deduplicate(transitions, spec.dedup_policy)
    transitions = assign_splits(transitions, spec.split_policy)
    shards = write_parquet(transitions, spec.parquet_dir)
    hdf5 = pack_hdf5(shards, spec.hdf5_dir)
    return write_manifest(shards, hdf5)
```

v0.1 source priority:

```text
1. CommitPackFT-compatible Python rows
2. deterministic synthetic transforms over permissive Python files
3. filtered CommitPack-compatible rows if available
4. gated or unavailable sources only when credentials are explicitly configured
```

Synthetic transforms:

- safe lint rewrites;
- syntax modernization;
- controlled local variable rename;
- exception-handling modernization;
- deterministic loop/comprehension rewrites where semantics are constrained.

Every synthetic row stores `synthetic_transform_id`, source digest, and transform
version.

Failure modes:

- source unavailable: fall back only if configured;
- parse failure: drop row with reason;
- high filter rejection: complete build but fail gate if above configured budget;
- split leakage: abort build.

## Alternatives Considered

- Random row split: rejected because it leaks repository and near-duplicate code
  across evaluation boundaries.
- Full commit as one row: rejected because mixed commits obscure the transition.
- Synthetic-only v1.0: rejected because it risks learning codemod artifacts
  rather than real edit dynamics.

## Drawbacks

- Strict filters reduce available data.
- Repository-level splits can produce uneven source distributions.
- License-aware public artifacts require more metadata work than private local
  experiments.

## Migration / Rollout

1. Implement fixture-only synthetic builder.
2. Add CommitPackFT-compatible adapter and filter report.
3. Add Parquet staging and HDF5 packing.
4. Add source mix config for v0.1.
5. Add optional gated-source adapter only after fallback builds pass.

## Testing Strategy

- Fixture records covering keep/drop filter reasons.
- Property test deterministic split assignment.
- HDF5 round-trip test for row counts and masks.
- Near-duplicate rejection test across train/test.
- License policy test that denied rows do not appear in public artifact manifests.

## Open Questions

- Owner: maintainers. Target: 2026-06-15. Which gated real-edit source is
  accessible for v1.0? Resolution: run source availability checks and document
  fallback mix in the dataset manifest.

## References

- `docs/spec/03-data-model.md#filtering-rules`
- `docs/spec/03-data-model.md#split-policy`
- `docs/spec/06-security.md#dataset-licensing`
- `docs/rfcs/RFC-0010-security-licensing-trust-boundaries.md`
