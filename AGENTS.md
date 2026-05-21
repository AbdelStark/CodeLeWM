# CodeLeWM Agent Context

This repository is a spec-driven research artifact for learning latent
transition models over Python code edits. Treat the spec corpus, RFCs, GitHub
issues, roadmap docs, and benchmark evidence as the contract before making code
changes.

## Current State

As of 2026-05-21, CodeLeWM has a working package runtime, a reproducible local
first-results smoke loop, four completed scaled Hugging Face Jobs runs, and a
post-v0.2 roadmap for the next public harness and benchmark milestone.

Implemented foundations:

- `codelewm.data`: source adapters, filtering, license decisions, split and
  dedup policy, CodeState extraction, action extraction, staging and pack
  helpers, dataset manifests, and a public CommitPackFT Python shard config.
- `codelewm.model`: transition interfaces, action encoders, predictor modules,
  transition energy, objective helpers, retrieval-loss gate, no-action margin
  action-use objective, checkpoint compatibility manifests, and checkpoint trust
  gates.
- `codelewm.training`: manifest-backed training runner, resume compatibility,
  default and scaled configs, CPU smoke executor, and a package-native torch
  executor over packed CodeLeWM transition batches.
- `codelewm.eval`: retrieval metrics, hard-negative pools, required baselines,
  action-view policy, collapse diagnostics, action ablation, surprise reports,
  scorer-quality reports, and downstream reranking benchmark pack contracts.
- `codelewm.harness`: package CLI entry point with `dataset build`, `dataset
  pack`, `train`, `eval retrieval`, `eval ablation`, `eval surprise`, `eval
  scorer-quality`, `eval downstream-pack`, `llm-demo`, `index`, `score`,
  `rerank`, `manifest verify`, and `secret-scan`.
- `codelewm.observability` and `codelewm.security`: artifact manifests,
  structured logs, redaction, public license gates, checkpoint trust checks,
  non-execution guards, and secret scanning.
- Release/package gates: `uv build` wheel/sdist checks, `twine check`, a clean
  wheel-install smoke, typed package marker coverage, `pip-audit`, release
  provenance reports, and manual-only TestPyPI/PyPI publication instructions.

Current evidence:

- `scripts/first-results` runs the local build, pack, torch train, retrieval,
  surprise, index, manifest verification, report rendering, and secret scan
  loop. `docs/benchmark/FIRST_RESULTS.md` is smoke evidence only.
- The scaled HF Jobs run `codelewm-scaled-20260520-9699b53` completed on job
  `6a0d43c92dc5b1243da50bba` from source SHA
  `9699b5309e43a3278f272663ef60cda23040d92a`.
- The action-use follow-up run `codelewm-action-use-20260520-6650183`
  completed on job `6a0d7a763aba298b21d147a9` from source SHA `6650183`.
- HF artifacts were published to `abdelstark/codelewm-public-shard`,
  `abdelstark/codelewm-transition-model`, and `abdelstark/codelewm-runs`, then
  downloaded with `hf download` and verified locally. The repositories are now
  public diagnostic artifact repositories.
- `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md`,
  `docs/cards/codelewm-scaled-dataset-2026-05-20.md`, and
  `docs/cards/codelewm-scaled-model-2026-05-20.md` are the artifact-backed
  record for that run.
- `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`,
  `docs/cards/codelewm-action-use-dataset-2026-05-20.md`, and
  `docs/cards/codelewm-action-use-model-2026-05-20.md` are the
  artifact-backed record for the #154 follow-up run.
- The #159 remediation run `codelewm-action-use-retrieval-20260520-7895d18`
  completed on job `6a0da3a08229e585f969c3f7` from source SHA
  `7895d185e165a917af0956a313d8948c04b33638`, published artifacts,
  downloaded them with `hf download`, and verified retrieval, ablation,
  surprise, scorer-quality, score, rerank, manifest, and secret-scan checks
  locally.
- `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md`,
  `docs/cards/codelewm-action-use-retrieval-dataset-2026-05-20.md`, and
  `docs/cards/codelewm-action-use-retrieval-model-2026-05-20.md` are the
  artifact-backed record for the #159 run.
- The v0.2 action-swap/inverse-action run
  `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` completed on job
  `6a0dea258229e585f969c808` from source SHA
  `7c7cb0b8fe132e4819f05a77585c254267e77574`, published artifacts,
  downloaded them with `hf download`, and verified retrieval, latent-probe,
  ablation, surprise, scorer-quality, score, rerank, manifest, and secret-scan
  checks locally.
- `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`,
  `docs/cards/codelewm-v0-2-action-swap-dataset-2026-05-20.md`, and
  `docs/cards/codelewm-v0-2-action-swap-model-2026-05-20.md` are the
  artifact-backed record for the v0.2 run.
- `config/benchmark/downstream_rerank_fixture.json` and
  `codelewm eval downstream-pack` define the first public-safe downstream
  benchmark pack. The checked-in fixture has one labeled task, emits manifest,
  source-license, split-leakage, readiness, and secret-scan reports, and stays
  claim-blocked below the 100-example gate.

Current blocker:

- The scaled pipeline is proven, but no scaled checkpoint is a positive
  action-conditioned quality result.
- The #154 action-use margin run beats random, shuffled-action, and lexical
  baselines, but loses to no-action on headline retrieval: Recall@1 `0.363`
  and MRR `0.467875` versus no-action Recall@1 `0.469` and MRR `0.549624`.
- The #159 margin+retrieval run improves text-action retrieval to Recall@1
  `0.597` and MRR `0.674500`, but no-action is still stronger at Recall@1
  `0.650` and MRR `0.708037`. Its action-use claim gate is
  `claim_allowed=false` with
  `no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action`.
- The v0.2 action-swap/inverse-action run reaches text-action Recall@1
  `0.263` and MRR `0.370048`, while no-action reaches Recall@1 `0.441` and
  MRR `0.533105`. It also fails the exact-same-before and near-before
  action-contrast margins, the latent-probe representation gate, and the
  scaled downstream-reranking gate.
- Public positive model-quality claims remain blocked. The project is complete
  as public negative/diagnostic evidence unless a new research hypothesis is
  opened for a future positive claim.
- The completed v0.2 research specification lives in
  `docs/roadmap/V0_2_ACTION_USE_RESEARCH_PLAN.md` and was tracked by #167.
- The next public milestone is the LLM + world-model harness and downstream
  reranking benchmark. It is specified by
  `docs/spec/11-llm-world-model-harness.md`,
  `docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md`, and
  `docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md`.
- Stream trackers: #183 LLM + world-model harness demo, #184 downstream
  candidate-reranking benchmark, and #185 preliminary results publication
  package. #183 and #185 are complete; #184 remains open only for #192.

Root `train.py`, root `eval.py`, and the Hydra configs are inherited from the
original image/LeWM seed. They are compatibility artifacts, not the source of
truth for CodeLeWM's code-edit training path.

## Required Reading

Before editing, read:

- `SPEC.md`
- the relevant file under `docs/spec/`
- the relevant RFC under `docs/rfcs/`
- `docs/roadmap/FULL_COMPLETION.md`
- `docs/roadmap/IMPLEMENTATION.md`
- `docs/roadmap/V0_2_ACTION_USE_RESEARCH_PLAN.md`
- `docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md`
- `CONTRIBUTING.md`

If security, manifests, checkpoints, logs, licensing, candidate code, configs,
or reports are touched, also read `docs/spec/06-security.md` and
`docs/spec/05-observability.md`.

If Hugging Face Jobs, ml-intern, publication, downloaded-artifact validation, or
training recipes are touched, also read:

- `docs/operations/HF_ML_INTERN_TRAINING.md`
- `docs/training/SCALED_TRAINING_RUNBOOK.md`
- `docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md`

If LLM candidate generation, OpenRouter, candidate packs, downstream reranking,
or preliminary publication wording are touched, also read:

- `docs/spec/11-llm-world-model-harness.md`
- `docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md`
- `docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md`

## Work Rules

- One GitHub issue per branch and PR.
- Keep public docs direct and evidence-backed. Do not add unverifiable claims.
- Preserve schema versions unless the issue explicitly changes a public schema.
- Candidate code, source data, configs, and checkpoints are untrusted input.
- Do not import, execute, or test-run candidate code while parsing or scoring it.
- Keep optional training/data dependencies host-owned and explicitly grouped.
- Prefer explicit typed failures over silent row drops, silent clipping, or best
  effort artifact writes.
- Every published artifact must be schema-versioned, finite, JSON-native where
  applicable, checksum-verifiable, and secret-scanned.
- Do not print, commit, paste, or summarize Hugging Face token values. Treat
  `.env` as local secret state.
- Hugging Face dataset/model/results repositories may be public by default after
  license/source, manifest, secret-scan, and checkpoint-trust gates pass. Public
  visibility does not permit unsupported positive model-quality claims.

## Implementation Order

Use GitHub issues as the authoritative queue. The closed #109 through #122 and
#137 through #138 issues are completed evidence, not the next queue.

Current completion order:

1. #192 eval: run downstream reranking comparison and claim gate.

Issues #186, #187, #188, #189, #190, #191, #193, and #194 are completed
preconditions for the downstream benchmark stream and publication package.

The OpenRouter public adapter uses `OPENROUTER_API_KEY` and model slugs such as
`anthropic/claude-4.5-sonnet`. Do not silently read raw provider keys in that
adapter. If direct Anthropic API key support is required, open a separate
adapter issue.

Issues #152 and #153 are completed preconditions for action-use remediation: the
dataset pipeline now emits action-discriminative diagnostics and the training
config matrix includes the primary action-use margin A10G profile plus a
margin+retrieval fallback. Issue #154 executed the primary profile and recorded
a negative claim gate; #159 executed the next remediation sweep and also closed
negative/diagnostic. Issue #123 closed the package build and manual publishing
gate. Issue #124 closed the dependency audit and release provenance gate. Issue
#125 closed the public docs refresh against the first-results, scaled systems,
and negative action-use evidence. Issue #126 closed the private diagnostic
release-freeze checkpoint in `docs/release/RELEASE_FREEZE_2026-05-20.md`; it
does not permit public positive action-conditioning claims.

Issue #150 is the completed negative/diagnostic scaled-result milestone. Issue
#167 completed the v0.2 action-use and representation research iteration as
negative/diagnostic evidence.

## Validation

Current lightweight validation:

```bash
uv sync --group dev
uv run pytest tests/
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests
uv run codelewm --help
```

HF Jobs orchestration must use the `hf` CLI for launch, monitoring, logs, stats,
download, and local verification:

```bash
hf auth whoami
CODELEWM_HF_JOBS_DRY_RUN=1 uv run scripts/hf-launch-codelewm-job
hf jobs inspect <job-id>
hf jobs logs <job-id>
hf jobs stats <job-id>
hf download ...
```

Do not relaunch #159 or #172. Job `6a0da3a08229e585f969c3f7` completed #159
and job `6a0dea258229e585f969c808` completed #172; both downloaded artifact
sets documented negative/diagnostic claim boundaries. Any future positive claim
requires a new research issue, a new hypothesis, and the same HF download,
manifest, eval, secret-scan, checkpoint-trust, and claim review gates.
