# CodeLeWM Preliminary Results 2026-05-21

This page is the public preliminary-results boundary for CodeLeWM after the
completed v0.2 action-use intervention.

## Short Version

CodeLeWM now has a working public artifact pipeline for code-edit world-model
research: dataset construction, HF Jobs training, checkpoint publication,
downloaded-artifact verification, retrieval/surprise/ablation/scorer evals,
model cards, and dataset cards.

The first scaled action-conditioned hypotheses failed. No current checkpoint
supports a public positive action-conditioned model-quality claim. The current
result is still publishable because it is useful negative evidence: the pipeline
works, the artifacts are public and verified, and the no-action baseline remains
stronger than the action-conditioned variants we tested.

The next milestone is a downstream harness where an LLM proposes candidate
patches and CodeLeWM scores or reranks them.

## Validated

- Public-safe Python edit datasets can be built from CommitPackFT-style sources
  with source, license, split, manifest, and checksum evidence.
- Package-native CodeLeWM training runs on HF Jobs and writes checkpoint
  manifests.
- Dataset, model, and results artifacts can be published to public Hugging Face
  repositories.
- Published artifacts can be downloaded with `hf download` and verified locally.
- Retrieval, action ablation, surprise, scorer-quality, score, rerank, manifest,
  checkpoint-trust, and secret-scan checks can be rerun from downloaded
  artifacts.
- The docs/cards/report pipeline can preserve exact run IDs, job IDs, source
  SHAs, metrics, caveats, and claim gates.

## Run Ledger

| Evidence | Issue | Run ID | Job ID | Source SHA | Report | Cards |
| --- | --- | --- | --- | --- | --- | --- |
| Scaled systems run | #138 | `codelewm-scaled-20260520-9699b53` | `6a0d43c92dc5b1243da50bba` | `9699b5309e43a3278f272663ef60cda23040d92a` | `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md` | `docs/cards/codelewm-scaled-dataset-2026-05-20.md`, `docs/cards/codelewm-scaled-model-2026-05-20.md` |
| Action-use margin | #154 | `codelewm-action-use-20260520-6650183` | `6a0d7a763aba298b21d147a9` | `6650183` | `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md` | `docs/cards/codelewm-action-use-dataset-2026-05-20.md`, `docs/cards/codelewm-action-use-model-2026-05-20.md` |
| Margin + retrieval | #159 | `codelewm-action-use-retrieval-20260520-7895d18` | `6a0da3a08229e585f969c3f7` | `7895d185e165a917af0956a313d8948c04b33638` | `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md` | `docs/cards/codelewm-action-use-retrieval-dataset-2026-05-20.md`, `docs/cards/codelewm-action-use-retrieval-model-2026-05-20.md` |
| v0.2 action-swap | #172 | `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` | `6a0dea258229e585f969c808` | `7c7cb0b8fe132e4819f05a77585c254267e77574` | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` | `docs/cards/codelewm-v0-2-action-swap-dataset-2026-05-20.md`, `docs/cards/codelewm-v0-2-action-swap-model-2026-05-20.md` |

All listed HF artifact sets were published to the public diagnostic Hugging
Face repositories, downloaded with `hf download`, and checked locally through
manifest verification and secret scanning before the corresponding reports and
cards were written.

## Invalidated Or Unsupported

- The #154 action-use margin run did not beat no-action on headline retrieval.
- The #159 margin+retrieval remediation improved text-action retrieval but still
  lost to no-action.
- The #172 v0.2 action-swap/inverse-action run failed headline action-use,
  exact-same-before, near-before, latent-probe, and downstream-readiness gates.
- Named semantic latent-axis claims are unsupported by the current latent probe
  evidence.
- Scaled downstream coding-usefulness claims are unsupported because the current
  scorer-quality path has one labeled example, not the required 100+ examples.

## Key Metrics

| Run | Text-action Recall@1 | No-action Recall@1 | Delta | Text-action MRR | No-action MRR | Delta | Claim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| #138 scaled systems | 0.371 | 0.459 | -0.088 | 0.472984 | 0.546116 | -0.073132 | negative |
| #154 action-use margin | 0.363 | 0.469 | -0.106 | 0.467875 | 0.549624 | -0.081749 | negative |
| #159 margin+retrieval | 0.597 | 0.650 | -0.053 | 0.674500 | 0.708037 | -0.033537 | negative |
| #172 v0.2 action-swap | 0.263 | 0.441 | -0.178 | 0.370048 | 0.533105 | -0.163057 | negative |

For the v0.2 action-contrast pools, no-action remained equal or stronger on
the exact-same-before and near-before slices that were meant to stress action
use.

The #172 latent-probe report sets
`semantic_structure_status=unsupported`,
`positive_representation_claim_allowed=false`, and
`dimension_claims_allowed=false`. The #172 scorer-quality report remains a
smoke path because it has one labeled example; the benchmark readiness gate says
`scaled downstream benchmark requires at least 100 labeled examples; got 1`.

## Public Artifacts

| Surface | Repository |
| --- | --- |
| Dataset artifacts | `abdelstark/codelewm-public-shard` |
| Model checkpoints | `abdelstark/codelewm-transition-model` |
| Run evidence | `abdelstark/codelewm-runs` |

Artifact-backed reports:

- `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md`
- `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`
- `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md`
- `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`

Matching dataset/model cards:

- `docs/cards/codelewm-scaled-dataset-2026-05-20.md`
- `docs/cards/codelewm-scaled-model-2026-05-20.md`
- `docs/cards/codelewm-action-use-dataset-2026-05-20.md`
- `docs/cards/codelewm-action-use-model-2026-05-20.md`
- `docs/cards/codelewm-action-use-retrieval-dataset-2026-05-20.md`
- `docs/cards/codelewm-action-use-retrieval-model-2026-05-20.md`
- `docs/cards/codelewm-v0-2-action-swap-dataset-2026-05-20.md`
- `docs/cards/codelewm-v0-2-action-swap-model-2026-05-20.md`

## What We Can Publish Now

We can publish CodeLeWM as:

- a public, reproducible code-edit world-model research harness;
- a verified HF Jobs and Hugging Face artifact pipeline;
- a negative result for the tested action-use interventions;
- a foundation for a downstream LLM-candidate reranking evaluation.

We cannot publish it as:

- a model that improves coding;
- a model with validated high-level semantic latent dimensions;
- a checkpoint that beats no-action on action-conditioned retrieval;
- a downstream patch-ranking system with proven usefulness.

## Blog / README Summary

CodeLeWM is a code-edit world-model research harness. The first public scaled
artifact pipeline works end to end: it builds public-safe edit datasets, trains
transition checkpoints on HF Jobs, publishes dataset/model/results artifacts,
downloads them again, and verifies manifests, checkpoints, evals, and secret
scans locally.

The first scientific result is negative. Action-conditioned variants beat random
and weak lexical controls, but they do not beat the no-action baseline on the
agreed headline metrics. The current latent-probe and downstream scorer-quality
evidence also block semantic-axis and coding-usefulness claims.

The next test is downstream candidate reranking: let an LLM generate multiple
candidate patches, then ask whether CodeLeWM can improve the ordering over
LLM-order and no-action baselines.

## Next Test

The next public milestone is the LLM + world-model harness:

```text
task + bounded repo context
  -> OpenRouter LLM candidate generation
  -> codelewm.llm_candidate_pack.v1
  -> CodeLeWM score/rerank
  -> demo report
  -> downstream benchmark gate
```

This is the right showcase because it tests the use case CodeLeWM was designed
for: not generating code directly, but scoring candidate transitions proposed
by another system.

The demo is tracked by #183 and the scaled downstream benchmark is tracked by
#184. Public preliminary-results packaging is tracked by #185.

The downstream benchmark must pass #192 before public wording can say that
CodeLeWM improves candidate patch ranking.
