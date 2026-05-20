# RFC-0006: Training Runtime And Configs

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

Training remains config-driven and reproducible. CodeLeWM adds project-specific
Hydra-style configs, deterministic seeding, artifact manifests, and CPU smoke
support before any large GPU run.

## Motivation

The current repository has working LeWM-style training scripts but image-control
configs. CodeLeWM needs configs that target code transitions, enforce one-step
prediction, and write reproducibility metadata.

## Goals

- Add `config/train/codelewm_tiny.yaml` and `config/train/codelewm_small.yaml`.
- Use `history_size=1`, `num_preds=1` for v0.1.
- Support CPU smoke and single-GPU bf16 training.
- Save config, manifest, checkpoint hash, and metrics report for each run.
- Resume only from compatible checkpoint and schema versions.

## Non-Goals

- Distributed training in v0.1.
- Hyperparameter sweep service integration.
- GPU-only validation path.

## Proposed Design

Training config shape:

```yaml
seed: 1337
data:
  train: data/codelewm_v0_1/train.hdf5
  val: data/codelewm_v0_1/val.hdf5
wm:
  history_size: 1
  num_preds: 1
  embed_dim: 256
  action_view: text
trainer:
  max_steps: 10000
  accelerator: auto
  devices: 1
  precision: bf16-mixed
loss:
  sigreg_weight: 0.09
  enable_retrieval_loss: false
  retrieval_weight: 0.0
  enable_action_use_margin: false
  action_use_margin_weight: 0.0
  action_use_margin: 0.0
```

Runtime contract:

```python
def train(cfg: TrainConfig) -> TrainingRunManifest: ...
```

The manifest stores:

- dataset manifest IDs;
- config hash;
- git SHA;
- seed;
- checkpoint files;
- validation metrics;
- collapse reports;
- objective/intervention settings in executor metadata;
- device and dtype.

Determinism:

- seed Python, NumPy, and PyTorch;
- set dataloader generator seeds;
- log deterministic algorithm settings;
- report nondeterministic backends when used.

Failure modes:

- missing dataset: `SourceUnavailableError`;
- schema mismatch: `SchemaError`;
- incompatible checkpoint: `CheckpointCompatibilityError`;
- gate failure: `EvaluationGateError`.

## Alternatives Considered

- Keep only root `train.py`: rejected because project-specific configs and
  manifests would remain implicit.
- Require GPU for all tests: rejected because contributors need local smoke
  validation.
- Distributed-first runtime: rejected until a single-device baseline is stable.

## Drawbacks

- CPU smoke is not representative of throughput.
- Config compatibility work adds upfront complexity.
- bf16 behavior differs by hardware and must be reported.

## Migration / Rollout

1. Add project configs beside existing LeWM configs.
2. Add config dataclasses and schema validation.
3. Add manifest writing.
4. Add CPU smoke fixture.
5. Add single-GPU training docs after the smoke path passes.

## Testing Strategy

- Config validation unit tests.
- CPU one-step training integration test.
- Resume compatibility test.
- Manifest checksum test.
- Dtype/device test for CPU and available accelerator.

## Open Questions

- Owner: maintainers. Target: 2026-07-01. Should v1.0 use gradient accumulation
  by default? Resolution: measure memory and throughput on the v0.1 model before
  locking the v1.0 config.

## References

- `docs/spec/08-performance-budget.md#model`
- `docs/spec/07-testing-strategy.md#required-commands`
- `docs/rfcs/RFC-0005-model-objective-and-collapse-diagnostics.md`
