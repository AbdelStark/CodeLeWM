# Testing Strategy

## Test Pyramid

Unit tests:

- schema validation;
- parser filters;
- normalization;
- abstract action extraction;
- dedup keys;
- split assignment;
- tensor shape/device/dtype behavior;
- CLI argument validation.

Property tests:

- formatting changes do not alter stable normalization unexpectedly;
- split assignment is deterministic for a seed;
- HDF5 round trips preserve record counts and masks;
- action abstraction never includes full after-code lines;
- scorer order is stable under identical inputs.

Integration tests:

- build a tiny synthetic dataset from fixture files;
- pack it to HDF5;
- run a one-batch train step on CPU;
- compute collapse metrics;
- run retrieval evaluation;
- run `codelewm score` and `codelewm rerank` on fixture candidates.

ML regression tests:

- smoke training loss is finite;
- embeddings have non-zero variance;
- shuffled actions underperform true actions on deterministic fixtures;
- patch-action upper bound outperforms no-action on deterministic fixtures.

## Required Commands

The release gate must expose:

```bash
python -m pytest
python -m pytest tests/data/test_codestate_fixtures.py
python -m pytest tests/integration
python -m codelewm dataset build --config tests/fixtures/tiny_dataset.yaml --out .artifacts/tiny
python -m codelewm train --config tests/fixtures/tiny_train.yaml
python -m codelewm eval retrieval --config tests/fixtures/tiny_retrieval.yaml
python -m codelewm score --before tests/fixtures/before.py --instruction tests/fixtures/instruction.txt --candidate tests/fixtures/after.py --checkpoint .artifacts/tiny/checkpoint.pt --json
```

## Evaluation Gates

v0.1 gates:

- dataset smoke build succeeds;
- tiny CPU training step succeeds;
- collapse metrics are finite and above minimum variance;
- deterministic synthetic retrieval beats random;
- JSON schemas validate for dataset, checkpoint, eval report, and score output.

v1.0 gates:

- real/synthetic dataset manifest validates;
- hard retrieval reports all baselines;
- no-action and shuffled-action ablations are present;
- patch-surprise AUC is reported;
- model card and dataset card match artifact manifests.

## CI Policy

CI must run on every pull request and push to the default branch:

- formatting/linting;
- unit tests;
- integration smoke tests;
- docs link checks;
- manifest/schema validation;
- security scan for accidental secrets.

GPU tests are optional before v1.0, but CPU smoke tests are mandatory.
