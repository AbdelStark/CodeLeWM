# Implementation Tracker — 2026-05-18

Generated from the spec corpus in PR #1:
https://github.com/AbdelStark/CodeLeWM/pull/1

Every implementable unit of work in the accepted spec/RFC corpus is filed below.
Each child issue is intended to be independently shippable; dependency comments
have been added on GitHub for the cross-issue links.

## Milestone: v0.1

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #14 | api: add pyproject package and console script | api | p0 | m | RFC-0011 | Open |
| #15 | api: move project modules under codelewm package | api | p0 | l | RFC-0011 | Open |
| #16 | core: create code transition model interfaces | core | p0 | m | RFC-0001 | Open |
| #17 | model: port predictor to code latent tensors | model | p0 | m | RFC-0001 | Open |
| #18 | model: add checkpoint compatibility checks | core | p1 | m | RFC-0001 | Open |
| #19 | data: implement source adapter interfaces | data | p0 | m | RFC-0002 | Open |
| #20 | data: implement CommitPackFT-compatible loader | data | p0 | m | RFC-0002 | Open |
| #21 | data: implement filter and license reports | data | p0 | m | RFC-0002 | Open |
| #22 | data: implement split and deduplication pipeline | data | p0 | l | RFC-0002 | Open |
| #23 | data: pack transition shards to HDF5 | data | p0 | l | RFC-0002 | Open |
| #24 | data: generate deterministic synthetic transforms | data | p1 | m | RFC-0002 | Open |
| #25 | data: implement CodeState extraction for Python symbols | data | p0 | l | RFC-0003 | Open |
| #26 | data: implement normalization and structured truncation | data | p0 | m | RFC-0003 | Open |
| #27 | data: emit segment and changed-hunk masks | data | p1 | m | RFC-0003 | Open |
| #28 | test: add CodeState fixture corpus | data | p0 | m | RFC-0003 | Open |
| #29 | data: implement text and abstract action extraction | data | p0 | l | RFC-0004 | Open |
| #30 | model: implement text action encoder | model | p0 | m | RFC-0004 | Open |
| #31 | model: implement abstract action encoder | model | p1 | m | RFC-0004 | Open |
| #32 | eval: enforce patch-action diagnostic boundary | evaluation | p1 | s | RFC-0004 | Open |
| #33 | model: implement MSE plus SIGReg objective | model | p0 | m | RFC-0005 | Open |
| #34 | eval: implement collapse diagnostics and kill reports | evaluation | p0 | m | RFC-0005 | Open |
| #36 | test: add action-conditioning regression fixtures | model | p0 | m | RFC-0005 | Open |
| #37 | train: add CodeLeWM training configs | model | p0 | m | RFC-0006 | Open |
| #38 | train: implement manifest-backed training runner | model | p0 | l | RFC-0006 | Open |
| #39 | train: add CPU smoke training path | model | p0 | m | RFC-0006 | Open |
| #41 | eval: implement retrieval metrics and candidate pools | evaluation | p0 | l | RFC-0007 | Open |
| #42 | eval: implement hard-negative sampler | evaluation | p0 | m | RFC-0007 | Open |
| #43 | eval: implement lexical no-action and shuffled baselines | evaluation | p0 | m | RFC-0007 | Open |
| #46 | harness: implement score command and scorer API | harness | p0 | l | RFC-0008 | Open |
| #47 | harness: implement rerank command with safe parsing | harness | p0 | m | RFC-0008 | Open |
| #49 | harness: add JSON schemas and invalid candidate handling | harness | p0 | m | RFC-0008 | Open |
| #50 | observability: implement artifact manifest schemas | observability | p0 | m | RFC-0009 | Open |
| #51 | observability: add JSONL logging and redaction | observability | p0 | m | RFC-0009 | Open |
| #52 | observability: add manifest verifier command | observability | p1 | s | RFC-0009 | Open |
| #53 | security: enforce non-execution parsing boundary | security | p0 | m | RFC-0010 | Open |
| #54 | security: implement license decisions and public gates | security | p0 | m | RFC-0010 | Open |
| #55 | security: add secret scan and unsafe checkpoint refusal | security | p1 | m | RFC-0010 | Open |
| #56 | api: add CLI help and JSON schema tests | api | p0 | m | RFC-0011 | Open |
| #58 | ci: add pull request workflow for tests and docs | ci | p0 | m | RFC-0012 | Open |
| #59 | docs: add contributing security changelog and PR template | docs | p1 | m | RFC-0012 | Open |
| #61 | roadmap: maintain implementation tracker | docs | p0 | s | RFC-0012 | Open |

## Milestone: v1.0

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #35 | model: add retrieval loss behind config gate | model | p1 | m | RFC-0005 | Open |
| #40 | train: implement checkpoint resume compatibility | model | p1 | m | RFC-0006 | Open |
| #44 | eval: implement patch-surprise evaluation | evaluation | p1 | m | RFC-0007 | Open |
| #45 | docs: add benchmark report template | docs | p1 | s | RFC-0007 | Open |
| #48 | harness: implement local transition index | harness | p1 | m | RFC-0008 | Open |
| #57 | docs: add public API usage docs | docs | p1 | s | RFC-0011 | Open |
| #60 | release: add release checklist and card templates | release | p1 | m | RFC-0012 | Open |

## Tracking Issues

- #2 [Tracking] Code transition model
- #3 [Tracking] Edit transition dataset
- #4 [Tracking] CodeState schema
- #5 [Tracking] Action views and encoders
- #6 [Tracking] Objective and collapse diagnostics
- #7 [Tracking] Training runtime
- #8 [Tracking] Retrieval and surprise evaluation
- #9 [Tracking] Harness scorer and reranker
- #10 [Tracking] Observability and artifact lineage
- #11 [Tracking] Security and licensing boundaries
- #12 [Tracking] Public API and packaging
- #13 [Tracking] Release CI and governance

## Cross-Cutting Dependencies

- #14 blocks package-level work such as #15 and #52 because the CLI and package
  entry point must exist before command-level validation.
- #15 blocks package-scoped implementation work such as #16, #19, and #50 because
  new modules need a stable `codelewm/` package boundary.
- #16 and #17 block model objective, action encoders, scorer, and checkpoint work
  because they define the core latent transition interface.
- #19, #20, #21, and #22 form the dataset ingestion chain that must precede HDF5
  packing in #23 and downstream training.
- #25 and #26 block mask extraction, action extraction, synthetic transforms, and
  CodeState fixture coverage because those tasks depend on stable state packs.
- #29 blocks text/abstract action encoders and patch-action policy validation.
- #33 and #34 block smoke training, action-conditioning regression, retrieval
  evaluation, and scorer integration because they define the trainable objective
  and collapse gate.
- #50 blocks #18, #23, #38, and #48 because checkpoints, datasets, training runs,
  and indexes need the shared artifact manifest schema.
- #53 blocks #47 because reranking must preserve the non-execution boundary.
- #39, #52, #55, and #56 block #58 because CI must run the CPU smoke path,
  manifest verifier, secret/checkpoint safety checks, and CLI schema contract.
- #30, #39, #41, #42, and #43 are the minimum evidence chain for the v0.1 claim
  that text actions improve transition retrieval over no-action and shuffled
  baselines.
