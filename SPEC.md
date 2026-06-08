# CodeLeWM Specification

Created: 2026-05-18

This corpus is the execution contract for CodeLeWM: a latent transition model for
Python code edits. The project ports the LeWM observation/action/next-observation
framing to code by treating an edit episode as `(state_before, action,
state_after)`. The first useful artifact is a scorer and reranker for candidate
code changes, not a patch generator.

## Source Of Truth

- `docs/spec/00-overview.md` defines the thesis, goals, non-goals, milestones,
  and success criteria.
- `docs/spec/01-architecture.md` defines module boundaries and data flow.
- `docs/spec/02-public-api.md` defines CLI, Python API, artifact contracts, and
  versioning expectations.
- `docs/spec/03-data-model.md` defines schemas, invariants, split policy, and
  artifact manifests.
- `docs/spec/04-error-model.md` defines failures, typed exceptions, and recovery.
- `docs/spec/05-observability.md` defines logs, metrics, lineage, and redaction.
- `docs/spec/06-security.md` defines trust boundaries, licensing, and secret
  handling.
- `docs/spec/07-testing-strategy.md` defines the validation pyramid and release
  gates.
- `docs/spec/08-performance-budget.md` defines preprocessing, training,
  indexing, and inference budgets.
- `docs/spec/09-release-and-versioning.md` defines release, deprecation, and
  compatibility policy.
- `docs/spec/10-glossary.md` defines canonical terms.
- `docs/spec/11-llm-world-model-harness.md` defines the post-v0.2 LLM
  candidate-generation, world-model reranking, downstream benchmark, and
  publication boundary.

## RFC Index

- `docs/rfcs/RFC-0001-lewm-compatible-code-transition-model.md`
- `docs/rfcs/RFC-0002-edit-transition-dataset.md`
- `docs/rfcs/RFC-0003-codestate-schema-and-normalization.md`
- `docs/rfcs/RFC-0004-action-views-and-encoders.md`
- `docs/rfcs/RFC-0005-model-objective-and-collapse-diagnostics.md`
- `docs/rfcs/RFC-0006-training-runtime-and-configs.md`
- `docs/rfcs/RFC-0007-retrieval-and-surprise-evaluation.md`
- `docs/rfcs/RFC-0008-agent-harness-scorer-reranker.md`
- `docs/rfcs/RFC-0009-observability-artifact-lineage.md`
- `docs/rfcs/RFC-0010-security-licensing-trust-boundaries.md`
- `docs/rfcs/RFC-0011-public-api-cli-and-packaging.md`
- `docs/rfcs/RFC-0012-release-ci-and-governance.md`
- `docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md`
- `docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`
- `docs/rfcs/RFC-0015-v0-7-execution-substrate-improvements.md`
- `docs/rfcs/RFC-0016-hard-downstream-reranking-benchmark.md`

## Milestone Boundary

`v0.1` must prove the end-to-end local path: ingest a bounded Python transition
set, produce schema-versioned shards, train a tiny model without collapse, run
hard retrieval evaluation, and expose a local scorer CLI. `v1.0` is the research
artifact: mixed real/synthetic training, ablations, baselines, patch reranking,
reproducible manifests, model card, dataset card, and release automation. `v1.1`
is the post-v0.2 showcase and evaluation layer: an LLM generates candidate
patches, CodeLeWM scores/reranks them, and a downstream benchmark decides
whether the score adds value over no-action and LLM-order baselines.

## Open Questions

Open questions are allowed only when they include an owner and resolution path.
The current open questions live inside the RFCs and are summarized in
`docs/spec/00-overview.md#open-questions`.
