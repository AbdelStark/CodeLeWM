# CodeLeWM

CodeLeWM is a Python research codebase for learning latent dynamics over code
edits.

The central transition is:

```text
CodeState_before + EditAction -> latent(CodeState_after)
```

The first useful product is not a code generator. CodeLeWM is built to become a
local scorer and reranker for candidate patches: given a before-state, an edit
instruction, and candidate after-states, it estimates which candidate best
matches the learned edit transition.

## Goal

CodeLeWM tests a narrow question:

Can a compact JEPA-style latent transition model learn action-conditioned
structure from Python edit trajectories?

If the answer is useful, the model should support:

- retrieval of similar historical edits;
- ranking candidate patches for a requested change;
- detecting surprising candidate after-states;
- controlled ablations over text, abstract, and patch-derived action views;
- reproducible baselines for code-edit world-model research.

CodeLeWM deliberately does not claim to generate code. Candidate edits can come
from codemods, search tools, manual patches, or external editing systems.
CodeLeWM scores the transition.

## Current Status

The repository is in pre-alpha implementation, but it is no longer only a spec.
The current package includes:

- `codelewm.data`: source adapters, parse/license filters, split and
  deduplication policy, deterministic synthetic transforms, `CodeState`
  extraction, normalization, masks, action extraction, dataset build artifacts,
  split HDF5/Parquet packing, and dataset manifests;
- `codelewm.model`: tensor contracts, text and abstract action tokenizers and
  encoders, a packed CodeState encoder, a torch transition model wrapper,
  pooled code-latent predictor adapter, transition energy, checkpoint
  compatibility manifests, MSE plus SIGReg objective, and gated retrieval loss;
- `codelewm.eval`: action-view report policy, retrieval metrics, required
  baseline reports, easy and hard candidate-pool reports, collapse diagnostics,
  evaluation gates, and kill-report artifacts;
- `codelewm.observability`: artifact manifests, structured JSONL log events,
  and redaction helpers for secrets, home paths, and long text snippets;
- `codelewm.harness`: package CLI entry point, local `codelewm dataset build`,
  `codelewm dataset pack`, `codelewm score`, and `codelewm rerank` commands, and structured
  dataset/score/rerank/error schemas;
- `codelewm.security`: license decision policy helpers, public artifact license
  gates, and non-execution parsing guards;
- `docs/spec/` and `docs/rfcs/`: the accepted system contracts.

The CodeLeWM-specific runtime is landing in stages. Manifest-backed training,
the CPU smoke training path, the package-native torch training executor,
retrieval reports, hard-negative sampling, baseline validation, local scoring,
safe candidate reranking, manifest verification, secret scanning, `uv`
dependency management, and pull-request CI are
implemented. A meaningful first training result is still pending: the remaining
work is the train/eval/index CLI workflow and reproducible first-results report tracked in
`docs/roadmap/FULL_COMPLETION.md`. Core harness commands can write local JSONL
logs with redaction via `--log-jsonl`. Root `train.py`, `eval.py`, and the
existing Hydra configs are inherited from the original LeWorldModel seed and are
kept for compatibility while the package runtime continues to replace them.

## Core Concepts

### Edit Transition

The atomic example is:

```text
(state_before, action, state_after)
```

For v0.1, transitions are one-step Python edits. Multi-file reasoning,
multi-step histories, and multi-language training are outside the initial
contract.

### CodeState

`CodeState` is a deterministic context capsule for the changed Python code. It is
not a whole-repository dump.

A state can include:

- path, module, symbol, and kind;
- visible imports;
- enclosing class context when relevant;
- sibling and local callee signatures;
- the primary changed code chunk;
- segment IDs and changed-hunk masks.

Normalization is structured. The pipeline drops lower-priority context before it
reduces primary code, and rows that still exceed budget fail instead of being
silently clipped.

### EditAction

Code edits do not have continuous robot-control actions, so CodeLeWM uses three
action views:

- `action_text`: normalized natural-language instruction or commit message; this
  is the headline inference path.
- `action_abs`: deterministic abstract edit script derived from code structure;
  this supports ablations and structural diagnostics.
- `action_patch`: diff-like diagnostic view; it is leaky and cannot be used for
  headline claims.

### Latent Transition Model

The model keeps the LeWM/JEPA shape but changes the domain from pixels to code:

```text
CodeStateEncoder(state_before)       -> z_before
ActionEncoder(action_text/action_abs) -> a
CodeLatentPredictor(z_before, a)      -> z_pred_after
CodeStateEncoder(state_after)         -> z_after
```

Training optimizes next-latent prediction with collapse control:

```text
loss = MSE(z_pred_after, z_after) + sigreg_weight * SIGReg(...)
```

An optional retrieval loss is available behind an explicit config gate. It is not
part of the default objective.

### Transition Energy

Candidate scoring is based on transition energy:

```text
energy = || P(E(before), A(action)) - E(candidate_after) ||^2
```

Lower energy means the candidate after-state is closer to the model prediction
for the requested edit.

### Collapse Diagnostics

Embedding collapse is a first-class failure mode. CodeLeWM tracks effective
rank, rank ratio, variance, pairwise cosine, norm statistics, and nearest-neighbor
entropy. Failed gates write kill reports instead of letting unusable checkpoints
look successful.

## Pipeline

The target v0.1 pipeline is:

```text
raw edit sources
  -> source adapter
  -> parser and license filters
  -> changed-symbol extractor
  -> CodeState builder
  -> action extractor
  -> split and deduplication
  -> transition JSONL artifact
  -> split Parquet staging shards
  -> split HDF5 transition packs
  -> manifest-backed training
  -> CodeLeWM checkpoint
  -> retrieval and surprise evaluation
  -> local score/rerank harness
```

Each stage must emit either a schema-versioned artifact, a manifest entry, or a
structured error record. Silent row drops are not allowed.

### Data Pipeline

The data layer accepts fixture data and CommitPackFT-style local JSONL shards.
Rows are filtered for Python-only parseable edits, license policy, size bounds,
message length, generated-file indicators, and edit-ratio limits.

Kept rows are converted into `CodeState` pairs, normalized, tokenized into stable
state/action arrays, assigned deterministic repository-level splits, deduplicated
to reduce leakage, and written by `codelewm dataset build` as a manifest-backed
transition JSONL artifact. `codelewm dataset pack` verifies the build manifest,
records lineage, stages split Parquet shards, and packs split HDF5 files for the
training runner when the data dependency group is installed. Dataset manifests
record artifact paths, row counts, split/source counts, feature flags, byte
sizes, license-gate metadata, and SHA-256 checksums.

### Model Pipeline

The model layer consumes fixed-shape transition batches:

- state token length: `1024`;
- text-action token length: `256`;
- abstract-action token length: `192`;
- latent dimension: `256`;
- v0.1 transition horizon: `history_size=1`, `num_preds=1`.

Text actions are the default inference action. Abstract actions are structural
ablations. Patch actions remain diagnostic upper bounds.

### Evaluation Pipeline

The primary evaluation is action-conditioned after-state retrieval:

```text
given: CodeState_before + action_text
rank:  true CodeState_after among candidate after-states
```

Required metrics are `Recall@1`, `Recall@5`, `Recall@10`, MRR, median rank, and
hard-negative slice metrics. Required baselines include random retrieval, lexical
retrieval, no-action, shuffled-action, abstract-action ablation, and
patch-action diagnostic upper bound when available.

The secondary evaluation is patch surprise: true or passing after-states should
have lower transition energy than decoys.

## Install

From a clean checkout:

```bash
uv sync --group dev
```

Optional dependency groups:

```bash
uv sync --group dev --group data      # h5py + pyarrow for dataset packing
uv sync --group dev --group train     # torch and training runtime adapters
uv sync --group dev --group eval      # optional evaluation helpers
uv sync --group dev --group docs      # documentation checks
uv sync --group dev --group release   # package build and release gates
```

The package extras mirror the same runtime boundaries for wheel consumers:

```bash
uv sync --extra data
uv sync --extra train
uv sync --extra eval
```

The package exposes a `codelewm` console script. Most command families are still
landing behind spec-tracked implementation issues, so the current CLI is a stable
entry point rather than the final user workflow.

## Validate

Lightweight validation:

```bash
uv run python -m unittest discover -s tests
```

With `pytest` installed:

```bash
uv run python -m pytest tests
```

Some tests skip when optional runtimes such as `torch`, `h5py`, or `pyarrow` are
not installed.

## Repository Map

```text
SPEC.md                         top-level specification index
AGENTS.md                       agent/contributor context and current gaps
docs/spec/                      canonical system specification
docs/rfcs/                      accepted design decisions
docs/roadmap/IMPLEMENTATION.md  implementation tracker
docs/roadmap/FULL_COMPLETION.md remaining first-results and release roadmap
docs/roadmap/NEXT_GOAL_PROMPT.md next autonomous implementation prompt
codelewm/data/                  source loading, filtering, CodeState, packing
codelewm/model/                 model contracts, actions, objective, checkpoints
codelewm/training/              configs, manifest runner, CPU smoke, torch executor
codelewm/eval/                  action policy, collapse gates, kill reports
codelewm/observability/         manifests, JSONL log events, redaction
codelewm/harness/               CLI, scorer, reranker, and output schemas
codelewm/security/              license policy and non-execution helpers
tests/                          unit and integration coverage
config/                         inherited LeWorldModel Hydra configs
train.py, eval.py               inherited LeWorldModel entry points
jepa.py, module.py, utils.py    compatibility wrappers
```

## Start Here

- Read `docs/usage/USAGE.md` for the install path and concrete CLI / Python examples.
- Read `AGENTS.md` for current implementation context and work rules.
- Read `SPEC.md` for the canonical index.
- Read `docs/roadmap/FULL_COMPLETION.md` for the ordered remaining backlog.
- Read `docs/spec/00-overview.md` for goals, non-goals, and pass gates.
- Read `docs/spec/01-architecture.md` for subsystem boundaries.
- Read `docs/spec/02-public-api.md` for the public CLI and Python contract.
- Read `docs/spec/03-data-model.md` for transition and artifact contracts.
- Read `docs/spec/07-testing-strategy.md` for validation expectations.

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
