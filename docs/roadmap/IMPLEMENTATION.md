# Implementation Tracker

Source of truth for work state: GitHub issues. This document is the
human-readable index over the spec corpus and is updated whenever an issue is
filed, closed, or superseded. See `CONTRIBUTING.md` and
`docs/spec/09-release-and-versioning.md` for how this tracker fits into the
release lifecycle.

- Last updated: 2026-05-21
- Spec corpus: `docs/spec/00-overview.md` through `docs/spec/10-glossary.md`
- Post-v0.2 spec: `docs/spec/11-llm-world-model-harness.md`
- RFCs: `docs/rfcs/RFC-0001-*.md` through `docs/rfcs/RFC-0013-*.md`
- Full completion roadmap: `docs/roadmap/FULL_COMPLETION.md`
- Next executable prompt: `docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md`

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

## Milestone: v1.1

Post-v0.2 showcase: OpenRouter LLM candidate generation, world-model scoring
and reranking, downstream candidate-reranking benchmark, and preliminary
negative-results publication.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #183 | [Tracking] v0.3 LLM + world-model harness demo | harness | p1 | l | RFC-0013 | Open |
| #184 | [Tracking] v0.3 downstream candidate-reranking benchmark | evaluation | p1 | l | RFC-0013 | Open |
| #185 | [Tracking] preliminary results publication package | docs | p1 | m | RFC-0013 | Open |
| #186 | spec: lock OpenRouter LLM candidate harness contract | harness | p1 | m | RFC-0013 | Open |
| #187 | harness: add OpenRouter candidate generation adapter | harness | p1 | m | RFC-0013 | Open |
| #188 | harness: add candidate pack schema and safe patch capture | harness | p1 | m | RFC-0013 | Open |
| #189 | harness: build end-to-end LLM plus CodeLeWM demo report | harness | p1 | l | RFC-0013 | Open |
| #190 | benchmark: define downstream task schema and claim gates | evaluation | p1 | m | RFC-0013 | Open |
| #191 | benchmark: build public-safe labeled candidate reranking set | evaluation | p1 | l | RFC-0013 | Open |
| #192 | eval: run downstream reranking comparison and claim gate | evaluation | p1 | l | RFC-0013 | Open |
| #193 | docs: publish preliminary negative-results report | docs | p1 | m | RFC-0013 | Open |
| #194 | docs: prepare public artifact index and announcement package | docs | p1 | m | RFC-0013 | Open |

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
publication package.

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

## Cross-Reference Map

Each implementation issue links to:

- one spec section under `docs/spec/`;
- one RFC under `docs/rfcs/`;
- one tracking issue (#2 through #13, plus #150 for the completion milestone).

The reverse links are documented in the RFC References sections. When a new spec
section or RFC is accepted, file the tracking issue first, derive child issues,
then update this file.
