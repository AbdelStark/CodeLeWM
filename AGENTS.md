# CodeLeWM Agent Context

This repository is a spec-driven research artifact for learning latent
transition models over Python code edits. Treat the spec corpus, RFCs, GitHub
issues, roadmap docs, and benchmark evidence as the contract before making code
changes.

## Current State

As of 2026-05-20, CodeLeWM has a working package runtime, a reproducible local
first-results smoke loop, and three completed scaled Hugging Face Jobs runs.

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
- Public positive model-quality claims remain blocked. The project is complete
  as public negative/diagnostic evidence unless a new research iteration is
  opened for a future positive claim.
- The v0.2 research specification lives in
  `docs/roadmap/V0_2_ACTION_USE_RESEARCH_PLAN.md` and is tracked by #167.

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
- Hugging Face dataset/model/results repositories may be public by default after
  license/source, manifest, secret-scan, and checkpoint-trust gates pass. Public
  visibility does not permit unsupported positive model-quality claims.

## Implementation Order

Use GitHub issues as the authoritative queue. The closed #109 through #122 and
#137 through #138 issues are completed evidence, not the next queue.

Current completion order:

1. Execute the v0.2 action-use research intervention through #167 only after
   the relevant child issue has a concrete hypothesis, config, and validation
   gate.

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
#167 owns the next v0.2 action-use and representation research iteration.

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

Do not relaunch #159. Job `6a0da3a08229e585f969c3f7` completed and its
downloaded artifacts documented a negative/diagnostic claim boundary. Any future
positive claim requires #167 or a child issue, a new research hypothesis, and
the same HF download, manifest, eval, secret-scan, checkpoint-trust, and claim
review gates.
