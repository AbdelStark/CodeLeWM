# CodeLeWM Agent Context

This repository is a spec-driven research artifact for learning latent
transition models over Python code edits. Treat the spec corpus, RFCs, GitHub
issues, roadmap docs, and benchmark evidence as the contract before making code
changes.

## Current State

As of 2026-05-20, CodeLeWM has a working package runtime, a reproducible local
first-results smoke loop, and two completed scaled Hugging Face Jobs runs.

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
  and scorer-quality reports.
- `codelewm.harness`: package CLI entry point with `dataset build`, `dataset
  pack`, `train`, `eval retrieval`, `eval ablation`, `eval surprise`, `eval
  scorer-quality`, `index`, `score`, `rerank`, `manifest verify`, and
  `secret-scan`.
- `codelewm.observability` and `codelewm.security`: artifact manifests,
  structured logs, redaction, public license gates, checkpoint trust checks,
  non-execution guards, and secret scanning.

Current evidence:

- `scripts/first-results` runs the local build, pack, torch train, retrieval,
  surprise, index, manifest verification, report rendering, and secret scan
  loop. `docs/benchmark/FIRST_RESULTS.md` is smoke evidence only.
- The scaled HF Jobs run `codelewm-scaled-20260520-9699b53` completed on job
  `6a0d43c92dc5b1243da50bba` from source SHA
  `9699b5309e43a3278f272663ef60cda23040d92a`.
- The action-use follow-up run `codelewm-action-use-20260520-6650183`
  completed on job `6a0d7a763aba298b21d147a9` from source SHA `6650183`.
- Private HF artifacts were published to `abdelstark/codelewm-public-shard`,
  `abdelstark/codelewm-transition-model`, and `abdelstark/codelewm-runs`, then
  downloaded with `hf download` and verified locally.
- `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md`,
  `docs/cards/codelewm-scaled-dataset-2026-05-20.md`, and
  `docs/cards/codelewm-scaled-model-2026-05-20.md` are the artifact-backed
  record for that run.
- `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`,
  `docs/cards/codelewm-action-use-dataset-2026-05-20.md`, and
  `docs/cards/codelewm-action-use-model-2026-05-20.md` are the
  artifact-backed record for the #154 follow-up run.

Current blocker:

- The scaled pipeline is proven, but neither scaled checkpoint is a positive
  action-conditioned quality result.
- The #154 action-use margin run beats random, shuffled-action, and lexical
  baselines, but loses to no-action on headline retrieval: Recall@1 `0.363`
  and MRR `0.467875` versus no-action Recall@1 `0.469` and MRR `0.549624`.
- Public model-quality claims remain blocked until a follow-up run passes an
  explicit action-use gate or the release is deliberately framed as a
  negative/diagnostic artifact.

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
- `CONTRIBUTING.md`

If security, manifests, checkpoints, logs, licensing, candidate code, configs,
or reports are touched, also read `docs/spec/06-security.md` and
`docs/spec/05-observability.md`.

If Hugging Face Jobs, ml-intern, publication, downloaded-artifact validation, or
training recipes are touched, also read:

- `docs/operations/HF_ML_INTERN_TRAINING.md`
- `docs/training/SCALED_TRAINING_RUNBOOK.md`
- `docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md`

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

## Implementation Order

Use GitHub issues as the authoritative queue. The closed #109 through #122 and
#137 through #138 issues are completed evidence, not the next queue.

Current completion order:

1. #159 run the second-stage action-use remediation sweep through the `hf` CLI
   if the project is still pursuing a positive model-quality claim.
2. #123 add package build and publishing gates.
3. #124 add dependency audit and provenance evidence.
4. #125 refresh public docs against the scaled evidence and claim boundary.
5. #126 run the final artifact freeze and release checklist.

Issues #152 and #153 are completed preconditions for action-use remediation: the
dataset pipeline now emits action-discriminative diagnostics and the training
config matrix includes the primary action-use margin A10G profile plus a
margin+retrieval fallback. Issue #154 executed the primary profile and recorded
a negative claim gate; #159 owns the next remediation sweep.

Tracking issue #150 owns the action-conditioned scaled-result milestone.

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

Do not mark the project complete unless the release-candidate artifacts can be
downloaded from Hugging Face, verified locally, and either pass the action-use
claim gate or explicitly document a negative/diagnostic claim boundary.
