# CodeLeWM Preliminary Results 2026-05-21

This page is the public preliminary-results boundary for CodeLeWM after the
completed v0.2 action-use intervention.

## Short Version

CodeLeWM now has a working public artifact pipeline for code-edit world-model
research: dataset construction, HF Jobs training, checkpoint publication,
downloaded-artifact verification, retrieval/surprise/ablation/scorer evals,
model cards, and dataset cards.

The first action-conditioned hypotheses failed. No current checkpoint supports
a public positive action-conditioned model-quality claim. The next milestone is
a downstream harness where an LLM proposes candidate patches and CodeLeWM scores
or reranks them.

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

| Run | Text-action Recall@1 | Text-action MRR | No-action Recall@1 | No-action MRR | Claim |
| --- | ---: | ---: | ---: | ---: | --- |
| #154 action-use margin | 0.363 | 0.467875 | 0.469 | 0.549624 | negative |
| #159 margin+retrieval | 0.597 | 0.674500 | 0.650 | 0.708037 | negative |
| #172 v0.2 action-swap | 0.263 | 0.370048 | 0.441 | 0.533105 | negative |

For the v0.2 action-contrast pools, no-action remained equal or stronger on
the exact-same-before and near-before slices that were meant to stress action
use.

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

