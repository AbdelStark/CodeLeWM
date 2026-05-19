# Full Completion Roadmap

Last updated: 2026-05-19

This roadmap tracks the work from the current pre-alpha package to a meaningful
first training result, then to a publishable research artifact. GitHub issues
remain the source of truth for implementation state; this document explains the
order, acceptance boundary, and remaining risk.

The next implementation-run prompt lives in
`docs/roadmap/NEXT_GOAL_PROMPT.md`.

## Project Status

The repository is past pure specification. The package contains data contracts,
model components, training-run manifests, CPU smoke training, evaluation
contracts, scoring/reranking harness commands, observability, security gates,
CI, and release templates.

The repository now has a reproducible first-results smoke report. The landed
tiny path can build, pack, train the package-native torch model, emit retrieval
and surprise reports, build a transition index, verify manifests, run a secret
scan, and regenerate `docs/benchmark/FIRST_RESULTS.md` through
`scripts/first-results`.

The first report does not establish a research-quality learning result. On the
tiny fixture, text-action ties the required retrieval baselines and surprise
evaluation has only mutation decoy coverage. The remaining research work is to
scale to a bounded public-safe shard with enough held-out examples for meaningful
baseline and decoy comparisons.

Current landed CLI commands:

- `codelewm score`
- `codelewm rerank`
- `codelewm manifest verify`
- `codelewm secret-scan`
- `codelewm dataset build`
- `codelewm dataset pack`
- `codelewm train`
- `codelewm eval retrieval`
- `codelewm eval surprise`
- `codelewm index`

No additional first-results CLI command is currently missing. The remaining
work is scaled data acquisition, ablations, report/card population, and release
automation.

## Meaningful First Training Result

A first result is meaningful only when it is reproducible from a clean checkout
and produces evidence for the actual research question:

Can a compact JEPA-style latent transition model learn action-conditioned
structure from Python edit trajectories?

Minimum success criteria:

- A local raw edit fixture or bounded public shard builds into a verified
  `codelewm.dataset.v1` artifact.
- The dataset is packed into the training format with deterministic splits,
  leakage checks, license evidence, row counts, and SHA-256 checksums.
- `codelewm train` trains the package-native CodeLeWM transition model, writes a
  `codelewm.training_run.v1` manifest, writes trusted checkpoint manifests, and
  records finite metrics.
- Collapse diagnostics pass configured minimum gates or emit a kill report that
  blocks the result from being called successful.
- Retrieval evaluation reports random, lexical, no-action, shuffled-action,
  text-action, abstract-action, and patch-action diagnostic rows where
  applicable.
- Surprise evaluation scores true after-states against random, same-file,
  mutation, and action-cluster decoys.
- `codelewm index` builds a verified transition index, and the scorer/reranker
  can use its retrieval prior without changing the public schema.
- `docs/benchmark/FIRST_RESULTS.md` records exact commands, config hash, seed,
  source git SHA, metrics, caveats, and claim checklist.
- `codelewm manifest verify` and `codelewm secret-scan` pass over every
  first-results artifact selected for publication.

## Roadmap Phases

### Phase 1: Dependency And Runtime Foundation

Move the project to a locked `uv` workflow. Split dependencies into explicit
groups so lightweight CI, data packing, training, evaluation, docs, and release
jobs install only what they need. CI should use the same commands documented for
contributors.

Deliverables:

- `uv.lock`
- dependency groups for `dev`, `data`, `train`, `eval`, `docs`, and `release`
- CI using `uv sync` / `uv run`
- docs updated away from pip-only install guidance

### Phase 2: Dataset CLI

Expose the existing source adapters, filters, state/action extraction, split and
dedup policy, and pack helpers as real CLI commands.

Deliverables:

- `codelewm dataset build`
- `codelewm dataset pack`
- a tiny committed first-results dataset fixture
- manifest verification and contract tests for both commands

### Phase 3: Package-Native Training

Replace fixture-only training with a concrete executor that consumes CodeLeWM
transition batches and trains the package model modules through the
manifest-backed runner.

Deliverables:

- torch-backed transition dataset loader
- training executor for text and abstract action views
- SIGReg/collapse diagnostics integrated into the run report
- trusted checkpoint manifests
- resume compatibility tests
- `codelewm train`

### Phase 4: Evaluation And Indexing

Turn the evaluation and transition-index APIs into reproducible CLI workflows
that consume first-results artifacts.

Deliverables:

- `codelewm eval retrieval`
- `codelewm eval surprise`
- `codelewm index`
- scorer/reranker retrieval-prior integration
- headline baseline validation gates
- report and manifest verification tests

### Phase 5: First Results

Run the complete local loop and publish the first honest benchmark report in the
repo.

Deliverables:

- `scripts/first-results` or equivalent checked-in runner
- first-results config bundle
- generated artifact manifest inventory
- `docs/benchmark/FIRST_RESULTS.md`
- caveat section that separates smoke evidence from research evidence

Status: complete for smoke evidence through `scripts/first-results` and
`docs/benchmark/FIRST_RESULTS.md`; not complete for scaled research evidence.

### Phase 6: Scaled Research Artifact

Move beyond the tiny first-results fixture to a bounded, documented, public-safe
dataset and ablation suite.

Deliverables:

- public source acquisition and license-gate report
- scaled training configs
- ablations for text action, abstract action, no-action, shuffled-action,
  patch-action diagnostic, retrieval loss, and collapse settings
- calibration and scorer/reranker quality report
- dataset and model cards filled from real artifacts

### Phase 7: Publishing And Release

Package and publish the artifact without weakening the security and
observability contracts.

Deliverables:

- buildable wheel and sdist
- TestPyPI/PyPI release workflow or documented manual gate
- release evidence bundle
- dependency audit and supply-chain report
- final release checklist with all manifests verified

## Ordered Backlog

Issue numbers are assigned in GitHub. Keep this table in implementation order
and update it when issue scope changes.

| Order | Issue | Title | Milestone | Blocks |
| ----- | ----- | ----- | --------- | ------ |
| 1 | #109 | build: migrate dependency management and CI to uv | v0.2 First Results Foundation | all first-results runtime work |
| 2 | #110 | data: expose dataset build CLI | v0.2 First Results Foundation | dataset pack, training |
| 3 | #111 | data: expose dataset pack CLI and tiny first-results fixture | v0.2 First Results Foundation | training, eval |
| 4 | #112 | train: add package-native torch training executor | v0.2 First Results Foundation | train CLI, first result |
| 5 | #113 | train: expose codelewm train CLI | v0.2 First Results Foundation | eval, first result |
| 6 | #114 | eval: expose retrieval evaluation CLI | v0.2 First Results Foundation | benchmark report |
| 7 | #115 | eval: expose surprise evaluation CLI | v0.2 First Results Foundation | benchmark report |
| 8 | #116 | harness: expose index CLI and retrieval-prior scoring | v0.2 First Results Foundation | scorer evidence |
| 9 | #117 | results: add reproducible first-results runner and report | v0.2 First Results Foundation | scaled artifact |
| 10 | #118 | data: document and gate public source acquisition | v0.3 Scaled Research Artifact | scaled training |
| 11 | #119 | train: add scaled training configs and runbook | v0.3 Scaled Research Artifact | ablations |
| 12 | #120 | eval: add action-view ablation suite | v0.3 Scaled Research Artifact | public claims |
| 13 | #121 | eval: add scorer calibration and reranker quality report | v0.3 Scaled Research Artifact | release claims |
| 14 | #122 | docs: fill dataset and model cards from artifacts | v0.3 Scaled Research Artifact | release |
| 15 | #123 | release: add uv build and package publishing gates | v0.4 Publishing | release |
| 16 | #124 | release: add dependency audit and provenance evidence | v0.4 Publishing | release |
| 17 | #125 | docs: refresh public docs against first-results evidence | v0.4 Publishing | release |
| 18 | #126 | release: run final artifact freeze and checklist | v1.0 Public Research Release | v1.0 |

## Tracking Issue Mapping

Use the existing subsystem tracking issues instead of creating duplicate
trackers. The closed v0.1 foundation children remain in those bodies as completed
history; the open full-completion children are listed beneath them.

- #3 Edit transition dataset
- #7 Training runtime
- #8 Retrieval and surprise evaluation
- #9 Harness scorer and reranker
- #10 Observability and artifact lineage
- #11 Security and licensing boundaries
- #12 Public API and packaging
- #13 Release CI and governance

Close a tracking issue only when every child issue in its subsystem is complete
and the release checklist no longer lists a blocker for that subsystem.

## Residual Risks

- The current default dependency set pulls heavy inherited runtime packages into
  the base install. The `uv` migration must correct this before first-results
  work expands CI.
- The CPU smoke path validates runner contracts but does not prove the model can
  learn the CodeLeWM task.
- The first-results report is now artifact-backed, but it intentionally records
  a tiny-fixture baseline tie rather than a learning claim.
- Root legacy scripts can confuse contributors; public docs must keep the
  package-native path clearly marked as authoritative.
- First-result claims must stay narrow until scaled ablations show reliable
  action-conditioned improvement.
