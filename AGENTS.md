# CodeLeWM Agent Context

This repository is a spec-driven research artifact for learning latent
transition models over Python code edits. Treat the spec corpus, RFCs, GitHub
issues, roadmap docs, and benchmark evidence as the contract before making code
changes.

## Current State

As of 2026-05-21, CodeLeWM has a working package runtime, a reproducible local
first-results smoke loop, four completed scaled Hugging Face Jobs runs, a
fixture-proven LLM + world-model harness, and explicit OpenRouter BYOK support
for live demo experiments.

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
  scorer-quality reports, downstream reranking benchmark pack contracts, and
  downstream reranking reports.
- `codelewm.harness`: package CLI entry point with `dataset build`, `dataset
  pack`, `train`, `eval retrieval`, `eval ablation`, `eval surprise`, `eval
  scorer-quality`, `eval downstream-pack`, `eval downstream-rerank`, `llm-demo`,
  `openrouter byok-register`, `index`, `score`, `rerank`, `manifest verify`,
  and `secret-scan`.
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
- `codelewm eval downstream-rerank` runs the benchmark comparison and writes
  `codelewm.downstream_rerank_report.v1` plus
  `codelewm.downstream_rerank_claim_gate.v1`. On the checked-in fixture, the
  claim gate remains closed because `example_count=1`.
- `uv run scripts/llm-world-model-demo` runs the local LLM + world-model fixture
  path, writes `demo.html`, verifies manifests, and secret-scans outputs. Live
  mode uses `OPENROUTER_API_KEY`; Anthropic BYOK is explicit through
  `codelewm openrouter byok-register` or
  `CODELEWM_OPENROUTER_BYOK_REGISTER=1`.
- The meaningful default scenario is now `bugfix-edge-case`. A live
  OpenRouter/BYOK run on 2026-05-21 completed end to end with 4/4 valid
  candidates, learned torch scoring, manifest verification, and secret scans.
  It also exposed the next model-observability gap: CodeLeWM ranked an
  incomplete whitespace-handling patch above more semantically complete
  candidates. Treat that as diagnostic workflow evidence, not a positive model
  result.

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
- The post-v0.2 LLM + world-model harness and downstream reranking benchmark
  milestone is complete as a diagnostic workflow. It is specified by
  `docs/spec/11-llm-world-model-harness.md`,
  `docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md`, and
  `docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md`.
- Stream trackers #183, #184, and #185 are complete. The current public
  boundary remains negative/diagnostic.
- Issue #206 completed the public BYOK/local-demo/readme usability pass. Issues
  #220 and #222 completed learned world-model scoring and terminal-first demo
  output. Issues #207/#208 are closed as superseded because the comment-style
  toy task should not be the public live artifact target. The open next streams
  are the meaningful harness demo (#224 through #231), scaled downstream
  benchmarking (#209/#210/#211), the visual model observability and TUI stream
  (#235 through #245), and a future positive-model research hypothesis (#212,
  related to #178).

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
- `docs/roadmap/MEANINGFUL_HARNESS_DEMO.md`
- `docs/roadmap/MODEL_OBSERVABILITY_TUI_ROADMAP.md`
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
- `docs/roadmap/MEANINGFUL_HARNESS_DEMO.md`
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
- Do not print, commit, paste, or summarize OpenRouter or Anthropic provider
  token values. BYOK helpers may read `ANTHROPIC_API_KEY` only when explicitly
  configured; reports must serialize only redacted BYOK metadata.
- Hugging Face dataset/model/results repositories may be public by default after
  license/source, manifest, secret-scan, and checkpoint-trust gates pass. Public
  visibility does not permit unsupported positive model-quality claims.

## Implementation Order

Use GitHub issues as the authoritative queue. The closed #109 through #122 and
#137 through #138 issues are completed evidence, not the next queue.

Current completion order:

1. #227 through #231 for the meaningful LLM + world-model harness demo. Issues
   #225 and #226 are complete. Continue in order: task-solving prompt path
   (#227), static patch analysis (#228), scorer traces and compact diff
   previews (#229), opt-in sandbox checks (#230), then the live public
   diagnostic artifact run (#231).
2. #237 through #245 for visual model observability and the Textual TUI stream
   under tracker #235: TensorBoard-compatible export (#237), checkpoint tensor
   inspection (#238), latent matrix diagnostics (#239), run timelines (#240),
   non-interactive report parity (#242), optional Textual TUI (#241), demo
   diagnostic links (#243), diagnostics-driven model experiment planning
   (#244), and final visual artifact publication (#245).
3. #210 then #211 for the scaled downstream reranking benchmark gate.
4. #178/#212 for CWM comparison and the next falsifiable positive-model
   research hypothesis.

Issues #186, #187, #188, #189, #190, #191, #192, #193, and #194 are completed
preconditions for the downstream benchmark stream and publication package.
Issue #206 is the completed BYOK/local-demo/readme usability pass.
Issue #224 is the open meaningful harness demo tracker. It supersedes #207/#208
for the next public live artifact because a comment/no-op task is not a useful
showcase. Issue #226 adds the `bugfix-edge-case` default scenario and selector
through `--scenario` / `CODELEWM_LLM_DEMO_SCENARIO`. Candidate code remains
untrusted. Do not execute candidate code by default; only #230 may add opt-in
sandbox checks, and those checks must use scenario allowlists, disposable
checkouts, scrubbed environments, timeouts, manifests, and secret scans.

The OpenRouter public adapter uses `OPENROUTER_API_KEY` and model slugs such as
`anthropic/claude-4.5-sonnet`. Do not silently read raw provider keys in that
adapter. Anthropic BYOK is allowed only through the explicit registration helper
and redacted request metadata.

Issue #235 is the open visual model observability and Textual TUI tracker.
Issue #236 locked its roadmap and backlog. #235 does not supersede #224; it
extends the project after the meaningful-demo scenario path by adding optional
TensorBoard-compatible event exports, checkpoint tensor/layer inspection,
latent representation matrix diagnostics, run timeline artifacts, shared report
view models, an optional Textual TUI, and diagnostics-driven model-improvement
planning. Visualization dependencies must remain optional and must not affect
base imports, JSON reports, fixture tests, or non-interactive CLI usage.

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
uv run codelewm openrouter byok-register --dry-run --json
uv run scripts/llm-world-model-demo
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
