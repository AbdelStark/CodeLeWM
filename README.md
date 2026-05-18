# CodeLeWM

CodeLeWM is a specification-first project for learning latent dynamics over
Python code edits.

The core idea is to treat a code change like an offline world-model transition:

```text
code_state_before + edit_action -> latent(code_state_after)
```

The first useful artifact is not a patch generator. CodeLeWM is designed to be a
local scorer and reranker for candidate code changes: given the current code, an
edit instruction, and one or more candidate after-states, it estimates which
candidate best matches the learned edit transition.

## Status

This repository currently contains:

- the original LeWorldModel-derived implementation seed (`jepa.py`, `module.py`,
  `train.py`, `eval.py`, and Hydra configs);
- the canonical CodeLeWM specification corpus in `SPEC.md` and `docs/spec/`;
- accepted RFCs for the data, model, evaluation, harness, security, and release
  contracts in `docs/rfcs/`;
- the implementation issue tracker in `docs/roadmap/IMPLEMENTATION.md`.

The CodeLeWM-specific package, CLI, dataset pipeline, evaluation suite, and CI
gates are not implemented yet. They are decomposed into GitHub issues and should
be built against the spec, not by extending this README ad hoc.

## Goal

CodeLeWM aims to answer a narrow research and engineering question:

Can a compact JEPA-style latent transition model learn useful action-conditioned
structure from code edit trajectories?

If the answer is yes, the model should help with:

- retrieving similar historical edits;
- ranking candidate patches for a requested edit;
- detecting surprising or inconsistent candidate after-states;
- providing a reproducible baseline for code-edit world-model research.

The project deliberately avoids claiming that it generates code. Candidate edits
can come from deterministic codemods, search-and-replace tools, or external
editing systems. CodeLeWM scores the transition.

## Core Concepts

### Edit Transition

The atomic training example is:

```text
(state_before, action, state_after)
```

For v0.1, transitions are one-step Python edits. Multi-step file histories and
multi-language training are future work.

### CodeState

`CodeState` is a deterministic context capsule for the changed Python code. It
is not a whole repository dump.

The intended v0.1 state includes:

- path and module name;
- changed function, method, class, region, or small file;
- visible imports;
- enclosing class header when relevant;
- sibling signatures;
- local callee signatures when available;
- the primary changed code chunk;
- segment and changed-hunk masks.

The state is bounded by a token budget and uses structured truncation. The
signature and changed hunk are kept before lower-priority context.

### EditAction

Code edits do not have continuous robot-control actions, so CodeLeWM uses three
action views:

- `action_text`: natural-language instruction or commit message. This is the
  headline inference action.
- `action_abs`: deterministic abstract edit script derived from AST/CST changes.
  This supports ablations and structural diagnostics.
- `action_patch`: full diff-like view. This is leaky and is only a diagnostic
  upper bound, not the headline inference path.

### Latent Transition Model

The model preserves the LeWM module shape:

```text
CodeStateEncoder(state_before) -> z_before
ActionEncoder(action)          -> a
TransitionPredictor(z_before, a) -> z_pred_after
CodeStateEncoder(state_after)  -> z_after
```

Training uses next-latent prediction:

```text
loss = MSE(z_pred_after, z_after) + lambda_sigreg * SIGReg(...)
```

Collapse diagnostics are part of the contract. A run is not useful if embeddings
lose rank, variance, or neighborhood diversity.

### Transition Energy

Candidate scoring is based on transition energy:

```text
energy = || P(E(before), A(action)) - E(candidate_after) ||^2
```

Lower energy means the candidate after-state is closer to what the model
predicts for the requested edit.

## Pipeline

The intended v0.1 pipeline is:

```text
raw edit sources
  -> source loader
  -> parser and license filters
  -> changed-symbol extractor
  -> CodeState builder
  -> action extractor
  -> deduplication
  -> repository-level split assignment
  -> Parquet staging shards
  -> HDF5 transition packs
  -> training dataloader
  -> CodeLeWM checkpoint
  -> retrieval evaluation
  -> local score/rerank CLI
```

Every step must emit either a manifest entry or a structured error record.
Silent row drops are not allowed.

## Evaluation

The primary evaluation is action-conditioned after-state retrieval:

```text
given: state_before + action_text
rank:  true state_after among candidate after-states
```

Required metrics:

- Recall@1, Recall@5, Recall@10;
- MRR;
- median rank;
- hard-negative slice metrics.

Required baselines:

- random retrieval;
- lexical retrieval;
- no-action model;
- shuffled-action model;
- abstract-action ablation;
- patch-action diagnostic upper bound when available.

The secondary evaluation is patch surprise: true or passing after-states should
have lower transition energy than decoys.

## Planned CLI

The public API is specified but not implemented yet. The intended command shape
is:

```bash
codelewm dataset build --config config/data/commitpackft.yaml --out data/codelewm_v0_1
codelewm dataset pack --manifest data/codelewm_v0_1/manifest.json --out data/codelewm_v0_1/hdf5
codelewm train --config config/train/codelewm_tiny.yaml
codelewm eval retrieval --checkpoint runs/v0_1/checkpoint.pt --data data/codelewm_v0_1/hdf5/test.hdf5
codelewm score --before before.py --instruction instruction.txt --candidate after.py --checkpoint runs/v0_1/checkpoint.pt
codelewm rerank --before before.py --instruction instruction.txt --candidates patches/ --checkpoint runs/v0_1/checkpoint.pt
```

Do not treat these commands as available until the corresponding implementation
issues are closed.

## Milestones

### v0.1 Foundation

v0.1 proves the end-to-end local path:

- build a bounded Python transition dataset;
- produce schema-versioned shards and HDF5 packs;
- train a tiny one-step model without collapse;
- run retrieval evaluation with required baselines;
- expose a local scorer CLI.

### v1.0 Research Artifact

v1.0 adds the full research surface:

- mixed real/synthetic training data;
- abstract-action and patch-action ablations;
- hard-negative retrieval benchmark;
- patch-surprise benchmark;
- local transition index;
- dataset card, model card, benchmark report, and release checklist.

## What Is Out Of Scope

For the initial milestones, CodeLeWM is not:

- a general code generator;
- a whole-repository encoder;
- a multi-language training system;
- a human-labeled edit taxonomy project;
- an LLM-judged benchmark;
- a system that executes untrusted project code.

## Repository Map

```text
SPEC.md                         top-level specification index
docs/spec/                      canonical system specification
docs/rfcs/                      accepted design decisions
docs/roadmap/IMPLEMENTATION.md  GitHub issue tracker
codelewm/                       package boundary for new implementation work
codelewm/model/                 LeWM-derived JEPA and model module seed
codelewm/training/              current training helper seed
jepa.py                         compatibility wrapper for codelewm.model.jepa
module.py                       compatibility wrapper for codelewm.model.modules
train.py                        current LeWM-derived training entrypoint
eval.py                         current LeWM-derived evaluation entrypoint
config/                         current Hydra configs inherited from LeWM
```

The future project-specific implementation should live under `codelewm/` as
defined in `docs/spec/01-architecture.md`.

## Start Here

- Read `SPEC.md` for the canonical index.
- Read `docs/spec/00-overview.md` for goals, non-goals, and pass gates.
- Read `docs/spec/01-architecture.md` for subsystem boundaries.
- Read `docs/roadmap/IMPLEMENTATION.md` for the implementation issue plan.
- Pick work from the GitHub issues, not from unstated assumptions.

## Attribution

CodeLeWM starts from the LeWorldModel codebase and keeps its JEPA-style model
shape as the implementation seed. The original LeWorldModel work is:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
