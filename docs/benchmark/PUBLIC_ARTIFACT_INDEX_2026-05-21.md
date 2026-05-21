# CodeLeWM Public Artifact Index 2026-05-21

This index lists the public Hugging Face artifact paths that back the current
CodeLeWM preliminary results. It is an index, not a model-quality claim.

Current claim boundary:

- the HF artifact pipeline is validated;
- the tested action-use interventions are negative/diagnostic;
- no current checkpoint supports positive action-conditioned quality, semantic
  latent-axis, or downstream coding-usefulness claims.

## Hugging Face Repositories

| Surface | Repository | Repo type |
| --- | --- | --- |
| Dataset packs | `abdelstark/codelewm-public-shard` | dataset |
| Model checkpoints | `abdelstark/codelewm-transition-model` | model |
| Run evidence | `abdelstark/codelewm-runs` | dataset |

## Indexed Runs

| Issue | Run ID | Job ID | Dataset pack | Model checkpoint | Run evidence | Report | Dataset card | Model card | Claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #138 | `codelewm-scaled-20260520-9699b53` | `6a0d43c92dc5b1243da50bba` | `abdelstark/codelewm-public-shard/runs/codelewm-scaled-20260520-9699b53/pack` | `abdelstark/codelewm-transition-model/checkpoints/codelewm-scaled-20260520-9699b53` | `abdelstark/codelewm-runs/runs/codelewm-scaled-20260520-9699b53` | `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md` | `docs/cards/codelewm-scaled-dataset-2026-05-20.md` | `docs/cards/codelewm-scaled-model-2026-05-20.md` | systems evidence, action-use negative |
| #154 | `codelewm-action-use-20260520-6650183` | `6a0d7a763aba298b21d147a9` | `abdelstark/codelewm-public-shard/runs/codelewm-action-use-20260520-6650183/pack` | `abdelstark/codelewm-transition-model/checkpoints/codelewm-action-use-20260520-6650183` | `abdelstark/codelewm-runs/runs/codelewm-action-use-20260520-6650183` | `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md` | `docs/cards/codelewm-action-use-dataset-2026-05-20.md` | `docs/cards/codelewm-action-use-model-2026-05-20.md` | action-use negative |
| #159 | `codelewm-action-use-retrieval-20260520-7895d18` | `6a0da3a08229e585f969c3f7` | `abdelstark/codelewm-public-shard/runs/codelewm-action-use-retrieval-20260520-7895d18/pack` | `abdelstark/codelewm-transition-model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18` | `abdelstark/codelewm-runs/runs/codelewm-action-use-retrieval-20260520-7895d18` | `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md` | `docs/cards/codelewm-action-use-retrieval-dataset-2026-05-20.md` | `docs/cards/codelewm-action-use-retrieval-model-2026-05-20.md` | action-use negative |
| #172 | `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` | `6a0dea258229e585f969c808` | `abdelstark/codelewm-public-shard/runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/pack` | `abdelstark/codelewm-transition-model/checkpoints/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` | `abdelstark/codelewm-runs/runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` | `docs/cards/codelewm-v0-2-action-swap-dataset-2026-05-20.md` | `docs/cards/codelewm-v0-2-action-swap-model-2026-05-20.md` | action-use, representation, and downstream gates negative |

## Verification Boundary

Every indexed run is backed by:

- a published dataset pack;
- a published model checkpoint;
- a published results artifact tree;
- an in-repo report;
- dataset/model cards;
- manifest verification;
- downloaded-artifact checks with `hf download`;
- secret-scan evidence before publication.

The index deliberately excludes local-only first-results smoke artifacts because
they are not public HF artifacts.

## Publication Copy Boundary

Use this index when sharing preliminary results. Do not use it to imply that a
checkpoint improves coding or beats no-action baselines. The current public
result is an artifact-backed negative result and a validated pipeline.

