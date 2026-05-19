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
- retrieval auxiliary loss can only run behind an explicit config gate and logs
  separately from prediction and SIGReg losses;
- patch-action upper bound outperforms no-action on deterministic fixtures.
- headline evaluation reports reject `action_patch`; patch-action reports must
  be tagged as diagnostic upper bounds.

## Required Commands

The release gate must expose:

```bash
uv sync --group dev
uv sync --group dev --group data
uv run python -m pytest
uv run python -m pytest tests/data/test_codestate_fixtures.py
uv run python -m pytest tests/eval/test_baselines.py
uv run python -m pytest tests/eval/test_hard_negatives.py
uv run python -m pytest tests/eval/test_retrieval_metrics.py
uv run python -m pytest tests/integration/test_action_conditioning.py
uv run python -m pytest tests/integration/test_cpu_train_smoke.py
uv run python -m pytest tests/integration
uv run codelewm dataset build --config tests/fixtures/dataset_build/config.json --out .artifacts/tiny-build --json
uv run codelewm secret-scan .artifacts/tiny-build --json
uv run codelewm dataset pack --manifest .artifacts/tiny-build/manifest.json --out .artifacts/tiny-pack --json
uv run codelewm manifest verify --manifest .artifacts/tiny-pack/manifest.json --parent-manifest .artifacts/tiny-build/manifest.json --json
uv run codelewm train --config tests/fixtures/tiny_train.json --out .artifacts/tiny-train --executor torch --device cpu --json
uv run codelewm eval retrieval --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt --data .artifacts/tiny-pack --out .artifacts/tiny-retrieval --json
uv run codelewm manifest verify --manifest .artifacts/tiny-retrieval/manifest.json --parent-manifest .artifacts/tiny-train/manifest.json --parent-manifest .artifacts/tiny-pack/manifest.json --json
uv run codelewm eval ablation --retrieval-artifact .artifacts/tiny-retrieval/manifest.json --training-artifact .artifacts/tiny-train/manifest.json --out .artifacts/tiny-ablation --json
uv run codelewm manifest verify --manifest .artifacts/tiny-ablation/manifest.json --parent-manifest .artifacts/tiny-retrieval/manifest.json --parent-manifest .artifacts/tiny-train/manifest.json --json
uv run codelewm eval surprise --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt --data .artifacts/tiny-pack --out .artifacts/tiny-surprise --json
uv run codelewm manifest verify --manifest .artifacts/tiny-surprise/manifest.json --parent-manifest .artifacts/tiny-train/manifest.json --parent-manifest .artifacts/tiny-pack/manifest.json --json
uv run codelewm index --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt --data .artifacts/tiny-pack --out .artifacts/tiny-index --json
uv run codelewm manifest verify --manifest .artifacts/tiny-index/manifest.json --parent-manifest .artifacts/tiny-train/manifest.json --parent-manifest .artifacts/tiny-pack/manifest.json --json
uv run scripts/first-results --overwrite
uv run codelewm secret-scan .artifacts/first-results docs/benchmark/FIRST_RESULTS.md --json
uv run scripts/validate-training-configs
CODELEWM_HF_JOBS_DRY_RUN=1 uv run scripts/hf-launch-codelewm-job
CODELEWM_HF_PIPELINE_MODE=smoke CODELEWM_HF_RUN_ID=local-smoke CODELEWM_HF_OUTPUT_ROOT=.artifacts/hf-local CODELEWM_HF_PUBLISH=1 CODELEWM_HF_PUBLISH_DRY_RUN=1 uv run scripts/hf-run-codelewm-pipeline
uv run codelewm score --before tests/fixtures/before.py --instruction tests/fixtures/instruction.txt --candidate tests/fixtures/after.py --checkpoint .artifacts/tiny/checkpoint.pt --json
```

## Evaluation Gates

v0.1 gates:

- dataset smoke build succeeds;
- dataset build emits `codelewm.source_acquisition.v1` with source mix,
  redacted private paths, and a passing embedded license gate;
- tiny CPU training step succeeds;
- collapse metrics are finite and above minimum variance;
- deterministic synthetic retrieval beats random;
- retrieval metric reports validate `Recall@1`, `Recall@5`, `Recall@10`, MRR,
  median rank, and held-out candidate-pool lineage;
- hard-negative pools exclude true targets and `train` split rows;
- headline retrieval reports include random, lexical, no-action, and
  shuffled-action baselines;
- action-view ablation reports include completed baseline rows and explicit
  blocked rows for missing abstract-action, retrieval-loss, collapse-setting,
  and patch-action diagnostic variants;
- headline retrieval reports use `action_text`, not `action_patch`;
- patch-surprise reports include pairwise AUC, true ranks, per-category counts,
  and explicit caveats for unavailable decoy categories;
- transition indexes include only train-split entries and verify both training
  and dataset parent manifests;
- `scripts/first-results` regenerates `docs/benchmark/FIRST_RESULTS.md` from
  local artifacts and keeps smoke evidence separate from research claims;
- Hugging Face Jobs launch scripts must pass dry-run locally before scaled
  remote compute is launched, and real publication requires an explicit
  `CODELEWM_HF_PUBLISH_DRY_RUN=0` override;
- checked-in scaled training configs must validate through
  `scripts/validate-training-configs`, keep `action_text` as the headline
  training path, and preserve trusted checkpoint/resume compatibility;
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

The pull-request workflow lives at `.github/workflows/pr.yml`. The
workflow contract is asserted by
`tests/ci/test_workflow_contract.py`, which ensures the workflow installs
through `uv sync --group dev`, keeps running
`uv run python -m pytest`, compiles real Python sources while excluding
intentionally invalid parser fixtures, verifies an artifact manifest with
`codelewm manifest verify`, runs `codelewm secret-scan` over public docs
and generated CI artifacts, builds and packs the committed tiny dataset fixture
with the data dependency group, verifies the spec-doc tree is present, and pins
the GitHub Actions versions it depends on. Local
`uv run python -m pytest tests/` runs the same lightweight test suite; the
dataset pack fixture additionally requires
`uv sync --group dev --group data`.
