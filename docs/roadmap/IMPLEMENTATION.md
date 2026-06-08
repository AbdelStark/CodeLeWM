# Implementation Tracker

Source of truth for work state: GitHub issues. This document is the
human-readable index over the spec corpus and is updated whenever an issue is
filed, closed, or superseded. See `CONTRIBUTING.md` and
`docs/spec/09-release-and-versioning.md` for how this tracker fits into the
release lifecycle.

- Last updated: 2026-06-08
- Spec corpus: `docs/spec/00-overview.md` through `docs/spec/10-glossary.md`
- Post-v0.2 spec: `docs/spec/11-llm-world-model-harness.md`
- RFCs: `docs/rfcs/RFC-0001-*.md` through `docs/rfcs/RFC-0016-*.md`
- Full completion roadmap: `docs/roadmap/FULL_COMPLETION.md`
- Model observability/TUI roadmap:
  `docs/roadmap/MODEL_OBSERVABILITY_TUI_ROADMAP.md`
- Final v1.0 completion record: `docs/roadmap/NEXT_GOAL_PROMPT.md`
- Hard downstream benchmark roadmap:
  `docs/roadmap/HARD_DOWNSTREAM_RERANKING_BENCHMARK.md`

## How This Tracker Is Maintained

1. Every implementation issue lives on GitHub with `type:*`, `area:*`,
   `priority:*`, `effort:*`, and `spec:rfc-NNNN` labels.
2. When an issue is filed, add a row in the matching milestone table.
3. When an issue closes, set its `Status` to `Closed` and link the merging PR
   from the issue or PR body.
4. When the spec, an RFC, or benchmark evidence changes, audit the
   cross-cutting dependency list at the bottom of this file.
5. A `roadmap:` PR updates this file; the tracker contract is asserted by
   `tests/docs/test_implementation_tracker.py`.

## Milestone: v0.1

Foundation: dataset, model, training smoke, retrieval eval, scorer CLI,
observability manifests, public license gates, CI workflow, and governance docs.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #14 | api: add pyproject package and console script | api | p0 | m | RFC-0011 | Closed |
| #15 | api: move project modules under codelewm package | api | p0 | l | RFC-0011 | Closed |
| #16 | core: create code transition model interfaces | core | p0 | m | RFC-0001 | Closed |
| #17 | model: port predictor to code latent tensors | model | p0 | m | RFC-0001 | Closed |
| #18 | model: add checkpoint compatibility checks | core | p1 | m | RFC-0001 | Closed |
| #19 | data: implement source adapter interfaces | data | p0 | m | RFC-0002 | Closed |
| #20 | data: implement CommitPackFT-compatible loader | data | p0 | m | RFC-0002 | Closed |
| #21 | data: implement filter and license reports | data | p0 | m | RFC-0002 | Closed |
| #22 | data: implement split and deduplication pipeline | data | p0 | l | RFC-0002 | Closed |
| #23 | data: pack transition shards to HDF5 | data | p0 | l | RFC-0002 | Closed |
| #24 | data: generate deterministic synthetic transforms | data | p1 | m | RFC-0002 | Closed |
| #25 | data: implement CodeState extraction for Python symbols | data | p0 | l | RFC-0003 | Closed |
| #26 | data: implement normalization and structured truncation | data | p0 | m | RFC-0003 | Closed |
| #27 | data: emit segment and changed-hunk masks | data | p1 | m | RFC-0003 | Closed |
| #28 | test: add CodeState fixture corpus | data | p0 | m | RFC-0003 | Closed |
| #29 | data: implement text and abstract action extraction | data | p0 | l | RFC-0004 | Closed |
| #30 | model: implement text action encoder | model | p0 | m | RFC-0004 | Closed |
| #31 | model: implement abstract action encoder | model | p1 | m | RFC-0004 | Closed |
| #32 | eval: enforce patch-action diagnostic boundary | evaluation | p1 | s | RFC-0004 | Closed |
| #33 | model: implement MSE plus SIGReg objective | model | p0 | m | RFC-0005 | Closed |
| #34 | eval: implement collapse diagnostics and kill reports | evaluation | p0 | m | RFC-0005 | Closed |
| #36 | test: add action-conditioning regression fixtures | model | p0 | m | RFC-0005 | Closed |
| #37 | train: add CodeLeWM training configs | model | p0 | m | RFC-0006 | Closed |
| #38 | train: implement manifest-backed training runner | model | p0 | l | RFC-0006 | Closed |
| #39 | train: add CPU smoke training path | model | p0 | m | RFC-0006 | Closed |
| #41 | eval: implement retrieval metrics and candidate pools | evaluation | p0 | l | RFC-0007 | Closed |
| #42 | eval: implement hard-negative sampler | evaluation | p0 | m | RFC-0007 | Closed |
| #43 | eval: implement lexical no-action and shuffled baselines | evaluation | p0 | m | RFC-0007 | Closed |
| #46 | harness: implement score command and scorer API | harness | p0 | l | RFC-0008 | Closed |
| #47 | harness: implement rerank command with safe parsing | harness | p0 | m | RFC-0008 | Closed |
| #49 | harness: add JSON schemas and invalid candidate handling | harness | p0 | m | RFC-0008 | Closed |
| #50 | observability: implement artifact manifest schemas | observability | p0 | m | RFC-0009 | Closed |
| #51 | observability: add JSONL logging and redaction | observability | p0 | m | RFC-0009 | Closed |
| #52 | observability: add manifest verifier command | observability | p1 | s | RFC-0009 | Closed |
| #53 | security: enforce non-execution parsing boundary | security | p0 | m | RFC-0010 | Closed |
| #54 | security: implement license decisions and public gates | security | p0 | m | RFC-0010 | Closed |
| #55 | security: add secret scan and unsafe checkpoint refusal | security | p1 | m | RFC-0010 | Closed |
| #56 | api: add CLI help and JSON schema tests | api | p0 | m | RFC-0011 | Closed |
| #58 | ci: add pull request workflow for tests and docs | ci | p0 | m | RFC-0012 | Closed |
| #59 | docs: add contributing security changelog and PR template | docs | p1 | m | RFC-0012 | Closed |
| #61 | roadmap: maintain implementation tracker | docs | p0 | s | RFC-0012 | Closed |

## Milestone: v1.0

Research artifact: package-native runtime, scaled HF evidence, action-use
remediation, publishing, provenance, docs refresh, and final artifact freeze.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #35 | model: add retrieval loss behind config gate | model | p1 | m | RFC-0005 | Closed |
| #40 | train: implement checkpoint resume compatibility | model | p1 | m | RFC-0006 | Closed |
| #44 | eval: implement patch-surprise evaluation | evaluation | p1 | m | RFC-0007 | Closed |
| #45 | docs: add benchmark report template | docs | p1 | s | RFC-0007 | Closed |
| #48 | harness: implement local transition index | harness | p1 | m | RFC-0008 | Closed |
| #57 | docs: add public API usage docs | docs | p1 | s | RFC-0011 | Closed |
| #60 | release: add release checklist and card templates | release | p1 | m | RFC-0012 | Closed |
| #109 | build: migrate dependency management and CI to uv | api | p1 | m | RFC-0011 | Closed |
| #110 | data: expose dataset build CLI | data | p1 | m | RFC-0002 | Closed |
| #111 | data: expose dataset pack CLI and tiny first-results fixture | data | p1 | m | RFC-0002 | Closed |
| #112 | train: add package-native torch training executor | model | p1 | l | RFC-0006 | Closed |
| #113 | train: expose codelewm train CLI | api | p1 | m | RFC-0011 | Closed |
| #114 | eval: expose retrieval evaluation CLI | evaluation | p1 | m | RFC-0007 | Closed |
| #115 | eval: expose surprise evaluation CLI | evaluation | p1 | m | RFC-0007 | Closed |
| #116 | harness: expose index CLI and retrieval-prior scoring | harness | p1 | m | RFC-0008 | Closed |
| #117 | results: add reproducible first-results runner and report | evaluation | p1 | m | RFC-0007 | Closed |
| #118 | data: document and gate public source acquisition | data | p1 | m | RFC-0002 | Closed |
| #119 | train: add scaled training configs and runbook | model | p1 | m | RFC-0006 | Closed |
| #120 | eval: add action-view ablation suite | evaluation | p1 | m | RFC-0007 | Closed |
| #121 | eval: add scorer calibration and reranker quality report | harness | p1 | m | RFC-0008 | Closed |
| #122 | docs: fill dataset and model cards from artifacts | release | p1 | m | RFC-0012 | Closed |
| #137 | ops: add HF Jobs ml-intern training automation | ci | p1 | m | RFC-0012 | Closed |
| #138 | run: execute HF Jobs scaled training and publish artifacts | release | p1 | l | RFC-0012 | Closed |
| #150 | [Tracking] Action-conditioned scaled result and release readiness | release | p1 | l | RFC-0012 | Closed |
| #151 | eval: add no-action dominance diagnostics and claim gates | evaluation | p1 | m | RFC-0007 | Closed |
| #152 | data: add action-discriminative shard diagnostics and hard negatives | data | p1 | l | RFC-0002 | Closed |
| #153 | train: add action-use objective and scaled sweep configs | model | p1 | l | RFC-0006 | Closed |
| #154 | run: execute follow-up HF Jobs action-use training and verify artifacts | release | p1 | l | RFC-0012 | Closed |
| #159 | run: execute second-stage action-use remediation sweep | evaluation | p1 | l | RFC-0007 | Closed |
| #123 | release: add uv build and package publishing gates | release | p1 | m | RFC-0011 | Closed |
| #124 | release: add dependency audit and provenance evidence | security | p1 | m | RFC-0012 | Closed |
| #125 | docs: refresh public docs against first-results evidence | docs | p1 | m | RFC-0011 | Closed |
| #126 | release: run final artifact freeze and checklist | release | p1 | l | RFC-0012 | Closed |
| #401 | [TRACKER] v1.0 paper/demo release: downstream learned-world-model evidence and final CodeLeWM conclusions | evaluation/harness/docs/results | p1 | l | follow-up | Closed |
| #402 | v1.0 hygiene: reconcile stale v0.6 issues and final queue state | docs/results | p1 | s | follow-up | Closed |
| #403 | v1.0 demo: define fixed downstream learned-world-model paper-demo contract | evaluation/harness | p1 | m | follow-up | Closed |
| #404 | v1.0 demo: implement one-command downstream learned-world-model paper demo | evaluation/harness/observability | p1 | l | follow-up | Closed |
| #405 | v1.0 run: publish final downstream paper-demo artifacts | evaluation/harness/results | p1 | m | follow-up | Closed |
| #406 | v1.0 results: consolidate benchmark tables and final claim audit | evaluation/results/docs | p1 | m | follow-up | Closed |
| #407 | v1.0 paper: rewrite CodeLeWM paper around final downstream evidence | docs/results | p1 | l | follow-up | Closed |
| #408 | v1.0 release: publish final artifact index, cards, README, and announcement package | docs/release/results | p1 | m | follow-up | Closed |

## Milestone: v1.1

Post-v0.2 showcase: OpenRouter LLM candidate generation, world-model scoring
and reranking, downstream candidate-reranking benchmark, and preliminary
negative-results publication.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #183 | [Tracking] v0.3 LLM + world-model harness demo | harness | p1 | l | RFC-0013 | Closed |
| #184 | [Tracking] v0.3 downstream candidate-reranking benchmark | evaluation | p1 | l | RFC-0013 | Closed |
| #185 | [Tracking] preliminary results publication package | docs | p1 | m | RFC-0013 | Closed |
| #186 | spec: lock OpenRouter LLM candidate harness contract | harness | p1 | m | RFC-0013 | Closed |
| #187 | harness: add OpenRouter candidate generation adapter | harness | p1 | m | RFC-0013 | Closed |
| #188 | harness: add candidate pack schema and safe patch capture | harness | p1 | m | RFC-0013 | Closed |
| #189 | harness: build end-to-end LLM plus CodeLeWM demo report | harness | p1 | l | RFC-0013 | Closed |
| #190 | benchmark: define downstream task schema and claim gates | evaluation | p1 | m | RFC-0013 | Closed |
| #191 | benchmark: build public-safe labeled candidate reranking set | evaluation | p1 | l | RFC-0013 | Closed |
| #192 | eval: run downstream reranking comparison and claim gate | evaluation | p1 | l | RFC-0013 | Closed |
| #193 | docs: publish preliminary negative-results report | docs | p1 | m | RFC-0013 | Closed |
| #194 | docs: prepare public artifact index and announcement package | docs | p1 | m | RFC-0013 | Closed |
| #206 | harness: add OpenRouter BYOK demo task and public README polish | harness/docs | p1 | m | RFC-0013 | Closed |

## Milestone: v1.2

Post-demo evidence streams. These are historical follow-ups, not prerequisites
for the current v0.9 data/eval repair epic.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #207 | [Tracking] v1.2 live LLM plus world-model harness evidence | harness/results | p1 | l | RFC-0013 | Closed |
| #208 | run: execute live OpenRouter BYOK harness demo and publish diagnostic artifacts | harness/results | p1 | m | RFC-0013 | Closed |
| #209 | [Tracking] v1.2 scaled downstream reranking benchmark | evaluation/data | p1 | l | RFC-0013 | Closed |
| #210 | data: build public-safe 100-example downstream reranking set | data/evaluation | p1 | l | RFC-0013 | Closed |
| #211 | eval: run scaled downstream reranking comparison and claim gate | evaluation/harness | p1 | m | RFC-0013 | Closed |
| #212 | [Tracking] v1.2 next positive-model research hypothesis | model/evaluation | p2 | l | follow-up | Closed |
| #214 | harness: render visual LLM world-model demo report | harness/docs | p1 | m | RFC-0013 | Closed |
| #216 | harness: require OpenRouter management key for BYOK registration | harness | p1 | s | RFC-0013 | Closed |
| #218 | harness: surface OpenRouter provider errors and accept fenced live diffs | harness | p1 | s | RFC-0013 | Closed |
| #220 | harness: use learned world-model inference in LLM demo | harness/model/runtime | p1 | m | RFC-0013 | Closed |
| #222 | harness: make LLM demo terminal output visual by default | harness/docs | p1 | m | RFC-0013 | Closed |

## Milestone: v1.3

Meaningful harness demo: scenario-driven code-edit tasks, richer candidate
analysis, visible scorer traces, optional sandbox checks, and one live public
diagnostic artifact set. This milestone improves the demo, not the model claim
boundary.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #224 | [Tracking] v1.3 meaningful LLM plus world-model harness demo | harness/evaluation/results | p1 | l | RFC-0013 | Closed |
| #225 | docs: lock meaningful harness demo roadmap and backlog | docs/harness | p1 | s | RFC-0013 | Closed |
| #226 | harness: add meaningful demo scenarios and selector | harness | p1 | m | RFC-0013 | Closed |
| #227 | harness: upgrade demo prompt for task-solving patches | harness | p1 | m | RFC-0013 | Closed |
| #228 | harness: add static patch analysis to demo candidates | harness/security | p1 | l | RFC-0013 | Closed |
| #229 | harness: show scorer traces and diff previews in the demo | harness/docs | p1 | m | RFC-0013 | Closed |
| #230 | security: add opt-in sandbox checks for harness demos | security/harness | p1 | l | RFC-0013 | Closed |
| #231 | run: publish meaningful live harness demo artifacts | results/harness | p1 | m | RFC-0013 | Closed |

## Milestone: v1.4

Visual model observability and TUI harness: TensorBoard-compatible model
generation traces, checkpoint tensor/layer inspection, latent representation
matrix diagnostics, run timelines, shared report view models, optional Textual
TUI, and diagnostics-driven model-improvement planning. This milestone improves
debuggability and demo observability; it does not unlock positive model-quality
claims by itself.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #235 | [Tracking] v1.4 visual model observability and TUI harness | observability/harness/model/evaluation | p1 | l | RFC-0009 | Closed |
| #236 | docs: lock visual model observability and TUI roadmap | docs/observability/harness | p1 | s | RFC-0009 | Closed |
| #237 | observability: add TensorBoard event export for training and checkpoints | observability/model/runtime | p1 | m | RFC-0009 | Closed |
| #238 | model: add checkpoint tensor and layer inspection reports | model/observability | p1 | m | RFC-0009 | Closed |
| #239 | eval: add latent representation matrix diagnostics | evaluation/model/observability | p1 | l | RFC-0007 | Closed |
| #240 | observability: add run timeline and monitoring reports | observability/runtime/results | p1 | m | RFC-0009 | Closed |
| #241 | harness: build optional Textual TUI for demo inspection | harness/runtime/observability | p1 | l | RFC-0013 | Closed |
| #242 | harness: keep rich terminal and JSON report parity with the TUI | harness/docs/observability | p1 | m | RFC-0013 | Closed |
| #243 | harness: connect model and latent diagnostics to demo reports | harness/model/evaluation/observability | p1 | m | RFC-0013 | Closed |
| #244 | research: define diagnostics-driven code model improvement experiment | model/evaluation/results | p1 | l | follow-up | Closed |
| #245 | run: publish visual observability harness artifact set | results/harness/observability | p1 | m | RFC-0013 | Closed |
| #256 | [Tracking] production cleanup and optimization pass | core/docs | p1 | m | RFC-0012 | Closed |
| #257 | core: fix audit-backed cleanup findings | core/evaluation/observability/docs | p1 | m | RFC-0009 | Closed |

## Milestone: v1.5

Hard anti-saturation downstream reranking benchmark: a follow-up benchmark
designed to test whether CodeLeWM adds ranking value when no-action, lexical,
and LLM-order controls are not already saturated.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #417 | [TRACKER] v1.5 hard anti-saturation downstream reranking benchmark | evaluation/data/harness/results | p1 | l | RFC-0016 | Open |
| #418 | docs: lock hard downstream benchmark spec and tracker | docs/evaluation | p1 | s | RFC-0016 | Open |
| #419 | data: add anti-saturation benchmark schema and readiness diagnostics | data/evaluation/observability | p1 | m | RFC-0016 | Open |
| #420 | data: build public-safe hard-negative candidate pools | data/security/evaluation | p1 | l | RFC-0016 | Open |
| #421 | harness: ingest LLM candidate packs into the hard benchmark | harness/evaluation/security | p1 | m | RFC-0016 | Open |
| #422 | eval: score hard benchmark baselines and CodeLeWM claim gate | evaluation/model/harness | p1 | l | RFC-0016 | Open |
| #423 | results: publish hard benchmark artifacts and claim audit | results/release/evaluation | p1 | m | RFC-0016 | Open |

## Milestone: v0.9

Data/eval repair for cross-benchmark correctness claims. This milestone is
complete: it started from the v0.8 diagnostic result, fixed the data,
calibration, semantic-decoy, and representation-coverage evaluability gaps,
ran guarded two-seed A10G HF Jobs, and published a final claim audit that keeps
the overall public model-quality claim closed.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #385 | [TRACKER] v0.9 data/eval repair for cross-benchmark correctness claims | data/evaluation/model/results | p1 | l | RFC-0015 | Closed |
| #386 | v0.9 hygiene: reconcile stale trackers and roadmap queue state | docs/results | p1 | s | RFC-0015 | Closed |
| #387 | v0.9 data: build cross-benchmark pass/fail execution pack with stratified labels | data/evaluation | p1 | l | RFC-0015 | Closed |
| #388 | v0.9 eval: emit held-out p_pass ROC-AUC and calibration reports | evaluation/model/results | p1 | m | RFC-0015 | Closed |
| #389 | v0.9 eval: repair semantic-decoy alignment and coverage gates | evaluation/data/results | p1 | m | RFC-0015 | Closed |
| #390 | v0.9 eval: enforce probe-label coverage and representation gates | evaluation/data/results | p1 | m | RFC-0015 | Closed |
| #391 | v0.9 train: guarded 2-seed HF Jobs run after data/eval preflight | model/runtime/observability | p1 | l | RFC-0015 | Closed |
| #392 | v0.9 eval/report: run full gate suite and publish claim audit | evaluation/results/release | p1 | l | RFC-0015 | Closed |

## Tracking Issues

Tracking issues group child implementation issues by subsystem and remain open
until every child issue closes or is explicitly superseded.

- #2 [Tracking] Code transition model — RFC-0001
- #3 [Tracking] Edit transition dataset — RFC-0002
- #4 [Tracking] CodeState schema — RFC-0003
- #5 [Tracking] Action views and encoders — RFC-0004
- #6 [Tracking] Objective and collapse diagnostics — RFC-0005
- #7 [Tracking] Training runtime — RFC-0006
- #8 [Tracking] Retrieval and surprise evaluation — RFC-0007
- #9 [Tracking] Harness scorer and reranker — RFC-0008
- #10 [Tracking] Observability and artifact lineage — RFC-0009
- #11 [Tracking] Security and licensing boundaries — RFC-0010
- #12 [Tracking] Public API and packaging — RFC-0011
- #13 [Tracking] Release CI and governance — RFC-0012

Additional completion trackers: #150 [Tracking] Action-conditioned scaled
result and release readiness, closed with the negative/diagnostic #159
boundary; #167 [Tracking] v0.2 action-use and representation research
intervention, closed with the negative/diagnostic #172 boundary; #183
[Tracking] v0.3 LLM + world-model harness demo; #184 [Tracking] v0.3
downstream candidate-reranking benchmark; #185 [Tracking] preliminary results
publication package; #224 [Tracking] v1.3 meaningful LLM plus world-model
harness demo; #235 [Tracking] v1.4 visual model observability and TUI harness;
#256 [Tracking] production cleanup and optimization pass; #364 [Tracking] v0.8
execution-trace world-model results; #385 [Tracking] v0.9 data/eval repair for
cross-benchmark correctness claims; #417 [Tracking] v1.5 hard
anti-saturation downstream reranking benchmark.

## Cross-Cutting Dependencies

- #14 was the package boundary precondition for every later command and
  module-level change.
- #15 was the module-move precondition for new packages such as
  `codelewm.eval`, `codelewm.security`, and `codelewm.observability`.
- #16 and #17 defined the latent transition interface and were the precondition
  for objective, encoders, scorer, and checkpoint work.
- #19, #20, #21, and #22 form the dataset ingestion chain that precedes HDF5
  packing in #23 and downstream training.
- #25 and #26 precede mask extraction, action extraction, synthetic transforms,
  and CodeState fixture coverage.
- #29 precedes text/abstract action encoders and patch-action diagnostic
  boundary work.
- #33 and #34 precede smoke training, action-conditioning regression, retrieval
  evaluation, and scorer integration.
- #50 is the shared artifact manifest schema that #18, #23, #38, #48, #52, and
  later HF/release evidence build on.
- #53 precedes #47 because reranking must preserve the non-execution boundary.
- #109 through #117 closed the package-native first-results runtime.
- #118 through #122 closed the scaled-data, scaled-training, ablation,
  scorer-quality, and card population work needed before the first HF run.
- #137 and #138 proved the HF Jobs/artifact-publication/downloaded-artifact path.
- #151 is now the landed claim-gate precondition for any positive public
  action-conditioning language.
- #152 landed the data/eval precondition for hard negatives that can stress
  action use instead of before-state priors.
- #153 landed the model/training precondition for the follow-up action-use
  scaled run.
- #154 executed the primary action-use margin follow-up and verified the
  downloaded artifacts, but the result remained negative.
- #159 executed the second-stage remediation run
  `codelewm-action-use-retrieval-20260520-7895d18` on HF Jobs job
  `6a0da3a08229e585f969c3f7`; artifacts were downloaded with
  `hf download`, manifest-verified, locally re-evaluated, secret-scanned, and
  recorded as negative/diagnostic because no-action still beat text-action.
- #171 built v0.2 action-contrast benchmark pools.
- #168 added latent representation probes and axis diagnostics.
- #170 added the v0.2 action-swap/inverse-action objective intervention.
- #169 added the downstream reranking benchmark contract.
- #216 fixes the OpenRouter BYOK registration boundary so provider-key
  registration uses an OpenRouter management key while normal generation keeps
  using `OPENROUTER_API_KEY`.
- #218 surfaces redacted OpenRouter provider-routing failures such as missing
  ZDR endpoints and accepts markdown-fenced unified diffs from live LLM output.
- #220 makes the LLM demo use the trusted package-native torch checkpoint
  scorer instead of the deterministic hashing fixture scorer, while preserving
  explicit fixture fallback for non-model tests.
- #222 makes the local demo script default to a visual terminal walkthrough
  while keeping raw JSON output as an explicit non-interactive mode.
- #172 executed the v0.2 action-swap run
  `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` on HF Jobs job
  `6a0dea258229e585f969c808`; artifacts were downloaded with `hf download`,
  manifest-verified, locally re-evaluated, secret-scanned, and recorded as
  negative/diagnostic because action-use, representation, and downstream gates
  did not pass.
- #123 closed the package build and manual publishing gate for wheel/sdist
  artifacts.
- #124 closed the dependency-audit and release-provenance gate for package
  release candidates.
- #125 closed the public docs refresh against the current artifact evidence and
  legacy-script boundary.
- #126 closed the diagnostic release freeze in
  `docs/release/RELEASE_FREEZE_2026-05-20.md`; public HF diagnostic artifacts
  are allowed, but public positive model-quality claims remain blocked because
  #159 did not supply claim-eligible evidence.
- #183 through #194 define the post-v0.2 v1.1 path. The harness stream uses the
  OpenRouter Python SDK to generate candidate patches, stores them as untrusted
  candidate packs, and composes CodeLeWM scoring/reranking into a demo report.
  The benchmark stream is the first path that can support coding-usefulness
  claims, and only if it beats no-action and LLM-order baselines with at least
  100 labeled examples. The publication stream is allowed to publish current
  negative/diagnostic results but not positive model-quality claims.
- #206 completes the public BYOK/local-demo usability pass. #207/#208 were
  closed as superseded after the local terminal demo showed that the toy
  comment task should not be the public live artifact target.
- #224 through #231 define the meaningful harness demo stream: scenario
  fixtures and selector, task-solving prompts, static patch analysis, scorer
  traces and compact diff previews, opt-in sandbox checks, and one live
  diagnostic artifact publication.
- #235 through #245 define the visual observability stream. #237 adds optional
  TensorBoard-compatible training/checkpoint traces, #238 adds trusted
  checkpoint tensor/layer inspection, and #239 adds manifest-backed latent
  matrix diagnostics with bounded heatmap previews and closed semantic-axis
  claim gates, and #240 adds manifest-backed run timelines for the LLM demo and
  latent-matrix eval paths. The remaining queue is report view-model parity,
  #242 adds the schema-versioned visual view model consumed by JSON, rich
  terminal, and HTML outputs. #241 adds the optional Textual TUI and
  deterministic TUI snapshot. #243 connects manifest-backed model/latent/tensor
  diagnostic links into demo reports and visual surfaces. #244 defines the
  diagnostics-driven candidate-contrast action training experiment. #245
  publishes the public visual observability artifact set documented in
  `docs/benchmark/VISUAL_OBSERVABILITY_ARTIFACTS_2026-05-21.md`.
- #256/#257 define the first audit-backed production cleanup pass: concrete
  static-dead-code cleanup, warning-free collapse diagnostics, and stale
  architecture/roadmap context refresh without removing legacy compatibility
  surfaces outside the deprecation policy.
- #364 completed the v0.8 execution-trace result publication. The final
  diagnostic boundary is recorded in
  `docs/benchmark/EXECUTION_V0_8_RESULTS_2026-06-05.md`.
- #385/#386-#392 completed the v0.9 dependency chain. The final claim boundary
  is recorded in `docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md`: both
  seeds clear HumanEval WS-D reranking, MBPP-Plus has zero lift over no-action,
  and the overall public claim remains closed.
- #401/#402-#408 completed the final v1.0 paper/demo release chain. This chain
  supersedes the stale v0.6-era #293/#306/#308/#309 queue and preserves the
  v0.9 boundary: HumanEval WS-D is a narrow positive slice, while
  MBPP/generalization and broad coding-improvement claims remain closed.
- #417/#418-#423 define the v1.5 hard anti-saturation downstream benchmark.
  This stream tests whether CodeLeWM adds value only after no-action, lexical,
  and LLM-order controls are proven below saturation.

## Cross-Reference Map

Each implementation issue links to:

- one spec section under `docs/spec/`;
- one RFC under `docs/rfcs/`;
- one tracking issue (#2 through #13, plus #150 for the completion milestone).

The reverse links are documented in the RFC References sections. When a new spec
section or RFC is accepted, file the tracking issue first, derive child issues,
then update this file.
