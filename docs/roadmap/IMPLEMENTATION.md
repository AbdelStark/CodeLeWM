# Implementation Tracker

Source of truth for work state: GitHub issues. This document is the
human-readable index over the spec corpus and is updated whenever an
issue is filed, closed, or superseded. See `CONTRIBUTING.md` and
`docs/spec/09-release-and-versioning.md` for how this tracker fits
into the release lifecycle.

- Last updated: 2026-05-18
- Spec corpus: `docs/spec/00-overview.md` through
  `docs/spec/10-glossary.md`
- RFCs: `docs/rfcs/RFC-0001-*.md` through `docs/rfcs/RFC-0012-*.md`

## How This Tracker Is Maintained

1. Every implementation issue lives on GitHub with `type:*`, `area:*`,
   `priority:*`, `effort:*`, and `spec:rfc-NNNN` labels.
2. When an issue is filed, add a row in the matching milestone table.
3. When an issue closes, set its `Status` to `Closed` and link the
   merging PR.
4. When the spec or an RFC changes, audit the cross-cutting
   dependency list at the bottom of this file.
5. A `roadmap:` PR updates this file; the tracker contract is
   asserted by `tests/docs/test_implementation_tracker.py`.

## Milestone: v0.1

Foundation: dataset, model, training smoke, retrieval eval, scorer
CLI, observability manifests, public license gates, CI workflow,
governance docs.

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
| #52 | observability: add manifest verifier command | observability | p1 | s | RFC-0009 | Open |
| #53 | security: enforce non-execution parsing boundary | security | p0 | m | RFC-0010 | Closed |
| #54 | security: implement license decisions and public gates | security | p0 | m | RFC-0010 | Closed |
| #55 | security: add secret scan and unsafe checkpoint refusal | security | p1 | m | RFC-0010 | Open |
| #56 | api: add CLI help and JSON schema tests | api | p0 | m | RFC-0011 | Closed |
| #58 | ci: add pull request workflow for tests and docs | ci | p0 | m | RFC-0012 | Open |
| #59 | docs: add contributing security changelog and PR template | docs | p1 | m | RFC-0012 | Open |
| #61 | roadmap: maintain implementation tracker | docs | p0 | s | RFC-0012 | Open |

## Milestone: v1.0

Research artifact: mixed data, ablations, reranking, cards, release
gates.

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #35 | model: add retrieval loss behind config gate | model | p1 | m | RFC-0005 | Closed |
| #40 | train: implement checkpoint resume compatibility | model | p1 | m | RFC-0006 | Open |
| #44 | eval: implement patch-surprise evaluation | evaluation | p1 | m | RFC-0007 | Open |
| #45 | docs: add benchmark report template | docs | p1 | s | RFC-0007 | Open |
| #48 | harness: implement local transition index | harness | p1 | m | RFC-0008 | Open |
| #57 | docs: add public API usage docs | docs | p1 | s | RFC-0011 | Open |
| #60 | release: add release checklist and card templates | release | p1 | m | RFC-0012 | Open |

## Tracking Issues

Tracking issues group child implementation issues by subsystem and
remain open until every child issue closes.

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

## Cross-Cutting Dependencies

- #14 was the package boundary precondition for every later command
  and module-level change.
- #15 was the module-move precondition for new packages such as
  `codelewm.eval`, `codelewm.security`, and `codelewm.observability`.
- #16 and #17 defined the latent transition interface and were the
  precondition for objective, encoders, scorer, and checkpoint work.
- #19, #20, #21, and #22 form the dataset ingestion chain that
  precedes HDF5 packing in #23 and downstream training.
- #25 and #26 precede mask extraction, action extraction, synthetic
  transforms, and CodeState fixture coverage.
- #29 precedes text/abstract action encoders and patch-action
  diagnostic boundary work.
- #33 and #34 precede smoke training, action-conditioning regression,
  retrieval evaluation, and scorer integration.
- #50 is the shared artifact manifest schema that #18, #23, #38, #48,
  and #52 build on.
- #53 precedes #47 because reranking must preserve the non-execution
  boundary.
- #39, #52, #55, and #56 precede #58 because CI must run the CPU
  smoke path, manifest verifier, secret/checkpoint safety checks, and
  CLI schema contract before it can stand alone.
- #30, #39, #41, #42, and #43 are the minimum evidence chain for the
  v0.1 claim that text actions improve transition retrieval over
  no-action and shuffled baselines.

## Cross-Reference Map

Each implementation issue links to:

- one spec section (`docs/spec/...`);
- one RFC (`docs/rfcs/RFC-NNNN-*.md`);
- one tracking issue (#2..#13).

The reverse links are documented in the RFC References sections.
When a new spec section or RFC is accepted, file the tracking issue
first, then derive the child issues, then update this file.
