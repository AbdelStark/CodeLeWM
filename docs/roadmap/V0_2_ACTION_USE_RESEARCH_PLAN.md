# v0.2 Action-Use Research Intervention

Last updated: 2026-05-20

Parent tracker: #167.

## Current Evidence Boundary

CodeLeWM has validated a real end-to-end research pipeline: public-source data
build, public/license filtering, manifest-backed packing, package-native
training, HF Jobs execution, Hugging Face artifact publication, `hf download`
verification, retrieval/ablation/surprise/scorer-quality evaluation, score/rerank
smokes, secret scanning, and artifact-backed reports/cards.

The current model evidence is narrower:

- The latent path carries useful code-state signal. Both text-action and
  no-action variants retrieve true after-states far above random, shuffled, and
  lexical baselines, and surprise evaluation separates true after-states from
  random/action-cluster decoys better than chance.
- The current evidence does **not** prove interpretable high-level semantic
  dimensions. No probe suite, per-dimension stability analysis, or causal
  latent-axis intervention has been run.
- The current evidence does **not** prove useful action-conditioned world-model
  behavior. In #159, text-action reached Recall@1 `0.597` and MRR `0.674500`,
  but no-action remained stronger at Recall@1 `0.650` and MRR `0.708037`.
- The current evidence does **not** prove downstream coding usefulness. The
  scorer/reranker path runs from downloaded artifacts, but the public CLI still
  reports the deterministic lightweight scorer backend plus retrieval prior, and
  the checked fixture ranks a hard negative above the true after-state.

Interpretation: the action-conditioned failure does not invalidate the entire
latent-code-world-model direction. It invalidates the current positive claim
that the model uses edit actions better than a before-state prior on the tested
benchmarks. v0.2 must directly test semantic representation structure and
downstream usefulness instead of inferring them from retrieval alone.

## Research Question

Can CodeLeWM learn latent code-edit structure that is both:

1. action-sensitive beyond a no-action before-state prior; and
2. useful for representation probes or downstream patch-ranking tasks?

## Hypotheses

| ID | Hypothesis | Mechanism | Minimal falsifying experiment |
| --- | --- | --- | --- |
| H1 | The current no-action dominance is partly an evaluation/data artifact. | Many real commits make the after-state predictable from the before-state and local context. | Build exact-same-before and near-before action-contrast pools. If no-action still wins, the model is not using actions enough. |
| H2 | Stronger action supervision can force useful action sensitivity. | Action-swap contrastive loss and inverse-action auxiliaries penalize ignoring action text. | Train one v0.2 intervention against #159 replay on identical pools. If text-action does not beat no-action on action-contrast metrics, the intervention fails. |
| H3 | CodeLeWM latents contain measurable semantic/edit structure, but this is unproven today. | Encoders may organize AST kind, symbol kind, edit class, and edit-size information even when action use is weak. | Frozen-latent probes must beat lexical/metadata/random controls with stable estimates across splits/seeds. |
| H4 | Downstream patch ranking can improve only if transition energy contributes beyond retrieval priors. | A useful world model should prefer true after-states over plausible hard negatives, not just nearest indexed examples. | On a labeled reranking set, model transition energy plus retrieval prior must beat retrieval-prior-only and no-action controls. |

## v0.2 Claim Gates

No public positive claim is allowed unless all relevant gates pass from
downloaded HF artifacts.

### Action-Use Gate

Headline action-contrast retrieval must satisfy:

- text-action Recall@1 >= no-action Recall@1 + `0.10`;
- text-action MRR >= no-action MRR + `0.08`;
- text-action beats random, lexical, shuffled-action, and #159 checkpoint
  replay on Recall@1 and MRR;
- shuffled-action is at least `2x` worse than text-action on Recall@1 or MRR;
- no split leakage or missing-baseline warning is present.

Secondary hard-1k retrieval must not regress below #159 by more than `0.03`
Recall@1 unless the report explicitly scopes the claim to action-contrast pools.

### Representation Gate

Representation claims require:

- frozen-latent probes for `z_before`, `z_after`, and `z_pred_after`;
- probe targets for edit class, AST node kind, symbol kind, edit-size bucket,
  action cluster, and source family where available;
- lexical, metadata-only, random-latent, no-action, and shuffled-action
  controls;
- uncertainty over seeds or bootstrap intervals;
- per-dimension associations reported only as diagnostic unless stable across
  seeds, splits, and model variants.

Dimension-level semantic claims are blocked unless the same axis or small axis
set recurs under at least two seeds and two held-out splits with consistent sign
and effect direction.

### Downstream Reranking Gate

Downstream usefulness claims require:

- at least 100 labeled candidate-ranking examples or a documented blocker;
- true after-state, hard negative, syntax failure, patch failure, and plausible
  wrong-edit candidates where available;
- top-1, MRR, Recall@k, error counts, and calibration slices;
- controls for random, lexical, no-action, #159 checkpoint replay, and
  retrieval-prior-only;
- explicit separation of model transition energy from retrieval prior.

## Experiment Matrix

| Claim | Metric | Dataset/env | Baselines | Controls | Expected result | Falsifying result |
| --- | --- | --- | --- | --- | --- | --- |
| Action text matters | Action-contrast Recall@1/MRR | same-before and near-before pools | #159, lexical, no-action, shuffled-action, random | leakage check, shuffled actions | text-action beats no-action by the v0.2 gate | no-action still equal or better |
| Latents encode edit structure | Probe accuracy/F1 or R2 | frozen latents from held-out splits | lexical, metadata-only, random-latent | seed/split bootstrap | probes beat controls with stable intervals | probes match controls or unstable estimates |
| Dimensions have interpretable axes | per-axis association and stability | frozen latents across seeds | random axes, shuffled labels | sign/effect stability | a small axis set is stable across seeds/splits | axes are unstable or multiple-testing artifact |
| Transition energy helps reranking | top-1/MRR | labeled candidate set | retrieval-only, lexical, no-action, #159 | non-execution parser checks | transition score improves over retrieval-only | retrieval-only or lexical wins |

## Implementation Plan

### Phase 0: Publication Policy

The user approved public Hugging Face visibility for CodeLeWM datasets, models,
and run evidence. The existing HF repositories are public:

- `https://huggingface.co/datasets/abdelstark/codelewm-public-shard`
- `https://huggingface.co/abdelstark/codelewm-transition-model`
- `https://huggingface.co/datasets/abdelstark/codelewm-runs`

Public visibility does not relax gates. Every artifact still needs source,
license, manifest, secret-scan, checkpoint-trust, and claim-boundary evidence.

### Phase 1: Action-Contrast Data And Eval (#171)

Deliver:

- exact-same-before candidate pools;
- near-before candidate pools;
- same-file/action-cluster/edit-shape controls;
- synthetic controlled transforms with multiple valid actions from one
  before-state;
- schema-versioned pool report with pair counts and unavailable-pool reasons;
- retrieval integration without breaking the current report schema.

### Phase 2: Representation Probe Suite (#168)

Deliver:

- probe CLI or eval subcommand;
- frozen-latent extraction manifests;
- probe targets and baseline controls;
- per-dimension association diagnostics;
- stability criteria for semantic-axis claims.

### Phase 3: Action-Fusion And Objective Intervention (#170)

Deliver:

- action-swap contrastive loss;
- inverse-action or action-reconstruction auxiliary;
- gated state-action fusion option behind config flags;
- config validation and CPU fixture tests;
- A10G configs for #159 replay and one v0.2 intervention:
  `config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml`
  and
  `config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml`.

### Phase 4: Downstream Reranking Benchmark (#169)

Deliver:

- labeled candidate-ranking corpus;
- retrieval-prior-only and transition-energy comparisons;
- non-execution candidate validation;
- `benchmark_readiness` gate requiring 100 labeled examples or a documented
  blocker;
- report separating fixture evidence from scaled benchmark evidence in
  `docs/benchmark/DOWNSTREAM_RERANKING_BENCHMARK.md`.

### Phase 5: Public HF v0.2 Sweep (#172)

Deliver:

- HF Jobs launch, logs, stats attempts, and downloads through the `hf` CLI;
- public dataset/model/results artifacts;
- downloaded-artifact local verification;
- benchmark report, dataset card, model card, and claim checklist;
- explicit positive/negative/diagnostic verdict.

## Expected Outcomes

Useful positive outcomes:

- text-action beats no-action on action-contrast pools;
- representation probes show stable edit/AST/symbol structure;
- transition energy improves reranking over retrieval-only controls.

Useful negative outcomes:

- no-action still wins even on action-contrast pools;
- probes fail to beat lexical/metadata controls;
- reranking remains retrieval-prior dominated.

Both outcomes are publishable as evidence if the reports keep the claim boundary
explicit.

## Non-Goals

- claiming code generation;
- using patch-action as a headline input;
- claiming named semantic dimensions from one seed;
- using LLM-as-judge as the primary metric;
- flipping claim wording because artifacts are public.

## Validation Commands

Future v0.2 HF work must use the same lifecycle:

```bash
hf auth whoami
hf jobs run ...
hf jobs ps
hf jobs inspect <job-id>
hf jobs logs <job-id>
hf jobs stats <job-id>
hf download ...
uv run codelewm manifest verify ...
uv run codelewm secret-scan ...
```

Local docs/spec changes must pass:

```bash
uv run pytest tests/docs -q
uv run codelewm secret-scan AGENTS.md README.md docs/roadmap docs/operations docs/benchmark docs/cards docs/release docs/usage tests/docs --json
```
