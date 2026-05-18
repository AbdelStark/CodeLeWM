# Overview

## Thesis

CodeLeWM learns latent edit dynamics over Python code. A transition is
`(CodeState_before, EditAction, CodeState_after)`. The model predicts the latent
after-state from the before-state and action, then uses transition energy to
retrieve, score, and rerank candidate code changes.

The project is not a code generator in its first shippable milestone. Generation
can be supplied by deterministic codemods, search-and-replace patches, or an
external editor. CodeLeWM is responsible for the learned transition model and the
evidence that action-conditioned latent prediction carries useful signal.

## Users

- Researchers who need a reproducible code-edit world-model baseline.
- Maintainers who need a local scorer for candidate code changes.
- Contributors who need clear data, model, evaluation, and release contracts.

## Goals

- Build a Python-only v0.1 pipeline from raw edit source to HDF5 transition data.
- Preserve the LeWM module shape: encoder, action encoder, predictor, projection
  heads, MSE next-latent loss, and SIGReg regularization.
- Encode code state as a deterministic context capsule, not a whole repository.
- Support natural-language text actions and deterministic abstract edit scripts.
- Evaluate on action-conditioned after-state retrieval and patch-surprise ranking.
- Ship CLI and Python APIs that can be used without reading training internals.
- Produce schema-versioned artifacts with manifests, lineage, and split proofs.
- Keep security and licensing boundaries explicit for public code data.

## Non-Goals

- Multi-language training in v0.1 or v1.0.
- Whole-repository encoding as the default state representation.
- Human-labeled edit taxonomies.
- LLM-as-judge evaluation.
- Training on test results as the primary signal.
- Patch generation as a claimed project capability.
- Multi-file reasoning as the default v0.1 unit.

## Milestones

### v0.1 Foundation

Deliverables:

- `codelewm.data` package with CommitPackFT-compatible loader, synthetic edit
  generator, parser filters, deduplication, repo-level splits, Parquet staging,
  HDF5 packing, and manifest writing.
- `codelewm.model` package with `CodeStateEncoder`, `TextActionEncoder`,
  `AbstractActionEncoder`, and a LeWM-compatible transition model wrapper.
- Tiny training config using one-step transitions, `history_size=1`,
  `num_preds=1`, latent dim `256`, and deterministic seeding.
- Retrieval evaluation with random, lexical, no-action, and shuffled-action
  baselines.
- `codelewm score` CLI that accepts before code, instruction, and candidate code.

The v0.1 synthetic edit generator is intentionally narrow. It emits only
parse-valid Python before/after rows for controlled local rewrites:

- rename a `value` function parameter to `result` when the target name is unused;
- add explicit `return None` to functions that otherwise return implicitly;
- replace `set([...])` over constant elements with a set literal.

Every synthetic row records `synthetic_transform_id`,
`synthetic_transform_version`, and a SHA-256 `source_digest`.

Pass gates:

- HDF5 smoke dataset contains at least 40k train transitions and 5k validation
  transitions.
- All examples parse before and after.
- Train/validation/test split is by repository or source identity.
- Smoke training runs without NaN, OOM, or embedding collapse.
- Synthetic hard-pool text-action `Recall@5 >= 0.40`.
- Shuffled-action retrieval is at least `2x` worse than text-action retrieval on
  the same candidate pool.

### v1.0 Research Artifact

Deliverables:

- Mixed real/synthetic dataset with 250k-350k train transitions, 20k validation
  transitions, and 20k test transitions, subject to licensing and source access.
- Abstract-action and patch-action ablations.
- Hard-negative retrieval benchmark, patch-surprise benchmark, and representation
  diagnostics.
- Candidate patch reranker CLI and Python API.
- Dataset card, model card, benchmark report, and release notes.

Pass gates:

- Held-out real-data text-action `Recall@5 >= 0.12` on hard-1k pools or a
  documented failure analysis that triggers the kill criteria.
- Text-action model beats no-action and shuffled-action models by at least `2x`.
- Patch-surprise pairwise `AUC >= 0.70` on real held-out data.
- Artifact manifests reproduce checksums, config hashes, and split membership.

## Open Questions

- AgentPack access: owner `maintainers`, target `2026-06-15`, resolution path
  documented in RFC-0002. If gated access is unavailable, v1.0 uses CommitPackFT,
  filtered CommitPack, self-mined permissive repos, and synthetic transforms.
- Frozen encoder baselines: owner `maintainers`, target `2026-07-01`, resolution
  path documented in RFC-0007. The first baseline set is lexical plus simple code
  embedding baselines; larger pretrained baselines are added only after dataset
  contracts are stable.
- Patch-action scope: owner `maintainers`, target `2026-07-15`, resolution path
  documented in RFC-0004. Patch actions are diagnostic and cannot be used as the
  headline inference action.

## Risk Markers

- RISK: Natural-language commit messages may not encode enough action signal.
  Resolution: compare text actions to abstract actions, no-action, and shuffled
  actions before claiming learned dynamics.
- RISK: Real commit transitions can mix unrelated edits. Resolution: enforce edit
  size caps, AST/CST parse checks, split discipline, and stratified filter reports.
- RISK: LeWM's simple objective may not transfer to code tokens. Resolution:
  monitor collapse diagnostics and add retrieval loss only under RFC-0005 gates.
