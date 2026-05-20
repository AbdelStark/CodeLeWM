# Full Completion Roadmap

Last updated: 2026-05-20

This roadmap tracks the remaining path from the current scaled systems result to
a claim-eligible CodeLeWM research artifact. GitHub issues remain the source of
truth for implementation state; this document explains the order, acceptance
boundary, and remaining risk.

The next executable `/goal` prompt lives in
`docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md`.

## Project Status

The repository is past pure specification and past local smoke evidence. The
package contains data contracts, model components, manifest-backed training,
evaluation contracts, scoring/reranking harness commands, HF Jobs automation,
observability, security gates, CI, release templates, a local first-results
smoke loop, and two completed scaled HF Jobs runs.

Completed evidence:

- `scripts/first-results` regenerates the local smoke evidence in
  `docs/benchmark/FIRST_RESULTS.md`.
- HF Jobs run `codelewm-scaled-20260520-9699b53` completed on job
  `6a0d43c92dc5b1243da50bba`.
- Follow-up HF Jobs run `codelewm-action-use-20260520-6650183` completed on job
  `6a0d7a763aba298b21d147a9` with the no-action margin objective.
- Private HF repositories contain the dataset pack, model checkpoint, and run
  evidence under the documented run paths.
- The private artifacts were downloaded with `hf download`, verified locally,
  and rerun through retrieval, ablation, surprise, scorer-quality, score, and
  rerank checks.
- `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md` and the scaled dataset/model
  cards are the artifact-backed report for that run.
- `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md` and the action-use
  dataset/model cards are the artifact-backed report for the #154 follow-up
  run.

Current blocker:

- Both scaled runs are meaningful systems evidence, but neither is a positive
  action-conditioned model-quality result.
- The #154 action-use run beats random, shuffled-action, and lexical baselines,
  but no-action is stronger on headline retrieval: text-action Recall@1
  `0.363`, MRR `0.467875`; no-action Recall@1 `0.469`, MRR `0.549624`.
- The next project milestone is the second-stage action-use remediation tracked
  by #159, followed by public-docs and artifact-freeze release gates. The
  package build, manual publishing, dependency audit, and provenance gates are
  closed by #123 and #124. If #159 remains negative, the release path must be
  explicitly negative/diagnostic rather than a positive model-quality claim.

Current landed CLI commands:

- `codelewm dataset build`
- `codelewm dataset pack`
- `codelewm train`
- `codelewm eval retrieval`
- `codelewm eval ablation`
- `codelewm eval surprise`
- `codelewm eval scorer-quality`
- `codelewm index`
- `codelewm score`
- `codelewm rerank`
- `codelewm manifest verify`
- `codelewm secret-scan`

## Meaningful First Scaled Result

A first scaled result is meaningful only when it is reproducible from a clean
checkout and answers the actual research question:

Can a compact JEPA-style latent transition model learn action-conditioned
structure from Python edit trajectories?

Minimum success criteria:

- A bounded public-safe shard builds into a verified `codelewm.dataset.v1`
  artifact with source acquisition, license, split, leakage, row-count, and
  checksum evidence.
- The dataset is packed into deterministic train/validation/test artifacts.
- `codelewm train` trains the package-native CodeLeWM transition model, writes a
  `codelewm.training_run.v1` manifest, writes trusted checkpoint manifests, and
  records finite metrics.
- Collapse diagnostics pass configured gates or emit a kill report that blocks
  success claims.
- Retrieval evaluation reports random, lexical, no-action, shuffled-action,
  text-action, abstract-action when available, and patch-action diagnostic rows
  where applicable.
- Retrieval or ablation artifacts include explicit text-action versus no-action
  deltas and a machine-readable action-use claim gate.
- Surprise evaluation scores true after-states against random, same-file,
  mutation, and action-cluster decoys where source coverage allows.
- `codelewm index` builds a verified transition index, and the scorer/reranker
  can use its retrieval prior without changing the public schema.
- Benchmark docs, dataset cards, and model cards record exact commands, config
  paths, seed, source git SHA, run ID, job ID when applicable, metrics, caveats,
  and claim checklist.
- `codelewm manifest verify` and `codelewm secret-scan` pass over every artifact
  selected for publication.
- Published private artifacts can be downloaded with `hf download` and verified
  locally from a clean checkout.

For a positive public action-conditioning claim, text-action must beat the
no-action baseline on the agreed headline metrics. If it does not, the project
can still publish a negative/diagnostic artifact, but all release docs must keep
that claim boundary explicit.

## Completed Phases

### Phase 1: Dependency And Runtime Foundation

Status: complete.

Delivered:

- `uv` workflow and lockfile
- dependency groups for development and optional runtime paths
- CI using `uv`
- contributor and usage docs

### Phase 2: Dataset CLI

Status: complete.

Delivered:

- `codelewm dataset build`
- `codelewm dataset pack`
- committed tiny first-results fixture
- manifest verification and contract tests
- public CommitPackFT Python shard config and source-acquisition policy

### Phase 3: Package-Native Training

Status: complete for baseline objective and action-use remediation configs.

Delivered:

- torch-backed transition dataset loader
- training executor for text and abstract action views
- SIGReg/collapse diagnostics integrated into run reports
- trusted checkpoint manifests
- resume compatibility tests
- `codelewm train`
- scaled CPU/MPS/A10G configs
- no-action margin action-use objective
- primary A10G action-use and margin+retrieval fallback configs

Remaining: second-stage action-use remediation in #159 if the project still
wants a positive claim.

### Phase 4: Evaluation, Indexing, And Harness

Status: complete for baseline reports, claim gates, action-discriminative shard
diagnostics, and hard-negative metadata.

Delivered:

- `codelewm eval retrieval`
- `codelewm eval ablation`
- `codelewm eval surprise`
- `codelewm eval scorer-quality`
- `codelewm index`
- scorer/reranker retrieval-prior integration
- downloaded-artifact score/rerank smoke checks
- action-discriminative shard reports
- hard-negative sampler pools for same-before, near-before, same-file, and
  action-discriminative candidates

Remaining: #159 second-stage verification if the project still wants a positive
claim.

### Phase 5: First Results

Status: complete for smoke evidence.

Delivered:

- `scripts/first-results`
- first-results config bundle
- generated artifact manifest inventory
- `docs/benchmark/FIRST_RESULTS.md`
- caveat section separating smoke evidence from research evidence

### Phase 6: Scaled HF Artifact

Status: complete as systems evidence, blocked as positive action-conditioning
evidence.

Delivered:

- public source acquisition and license-gate report
- scaled training configs
- action-view ablation suite
- scorer/reranker quality report
- HF Jobs ml-intern automation
- private dataset/model/results publication
- downloaded-artifact verification
- scaled benchmark report and cards

Blocker: no-action baseline beats text-action on headline retrieval in both the
#138 baseline run and the #154 no-action margin follow-up run.

## Remaining Phases

### Phase 7: Action-Use Remediation

Goal: make action use measurable, train against the failure mode, and rerun the
scaled HF loop from private published artifacts.

Deliverables:

- no-action dominance diagnostics and machine-readable claim gates (#151,
  complete)
- action-discriminative shard diagnostics and hard negatives (#152, complete)
- action-use objective/intervention and scaled sweep configs (#153, complete)
- follow-up HF Jobs run launched, monitored, downloaded, verified, inferred, and
  evaluated through the `hf` CLI (#154, complete with a negative claim gate)
- second-stage action-use remediation sweep, likely using the margin+retrieval
  fallback config unless analysis shows a smaller correction is required (#159)

### Phase 8: Publishing And Release

Goal: package and publish only after the evidence boundary is clear.

Deliverables:

- wheel/sdist build and package publishing gates (#123, complete)
- dependency audit and provenance evidence (#124, complete)
- public docs refreshed against the scaled evidence and claim boundary (#125,
  complete)
- final artifact freeze, release checklist, and release notes (#126)

## Ordered Backlog

Keep this table in implementation order and update it when issue scope changes.

| Order | Issue | Title | Milestone | Blocks |
| ----- | ----- | ----- | --------- | ------ |
| 1 | #159 | run: execute second-stage action-use remediation sweep | Action-Use Remediation | positive claim path |
| 2 | #126 | release: run final artifact freeze and checklist | Public Research Release | v1.0 |

Completed backlog base:

- #109 through #122 closed the first-results and scaled-artifact implementation
  path.
- #137 added HF Jobs and ml-intern automation.
- #138 executed the first scaled HF Jobs run, private publication,
  downloaded-artifact verification, inference, and evals.
- #151 added no-action dominance diagnostics and machine-readable action-use
  claim gates.
- #152 added action-discriminative shard diagnostics and hard-negative metadata.
- #153 added the no-action margin objective and A10G action-use sweep configs.
- #154 executed the primary action-use HF Jobs follow-up run and verified the
  downloaded private artifacts; the run remained negative because no-action
  still beat text-action.
- #123 added wheel/sdist build, metadata, clean-install, typed marker, and
  manual publishing gates for the Python package.
- #124 added release dependency audit, provenance JSON, CI gates, and release
  checklist evidence for the package supply-chain path.
- #125 refreshed README, usage, public API, roadmap, and release docs against
  current artifact evidence and the negative action-use claim boundary.
- #150 tracks the remaining action-conditioned scaled-result and release
  readiness milestone.

## Tracking Issue Mapping

Use the existing subsystem tracking issues and the new completion tracker
instead of creating duplicate trackers.

- #3 Edit transition dataset: #152 complete; no active child.
- #7 Training runtime: #153 complete; no active child.
- #8 Retrieval and surprise evaluation: active child #159.
- #9 Harness scorer and reranker: active child #159.
- #10 Observability and artifact lineage: active children #159 and #126.
- #11 Security and licensing boundaries: active child #126.
- #12 Public API and packaging: no active child.
- #13 Release CI and governance: active children #150 and #126.
- #150 Action-conditioned scaled result and release readiness: active children
  #159 and release gate #126.

Close a tracking issue only when every child issue in its subsystem is complete
or explicitly superseded and the release checklist no longer lists a blocker for
that subsystem.

## Residual Risks

- The scaled result currently fails the core action-conditioning claim gate
  because no-action beats text-action.
- The current public shard may still not contain enough action-discriminative
  pressure for text actions to matter, even with targeted hard negatives.
- The no-action margin objective was proven insufficient by #154; the next
  claim-seeking run needs the #159 second-stage remediation.
- Scorer-quality evidence is still small and should remain a gate, not a broad
  calibration claim.
- Same-file and action-cluster surprise decoy counts are lower than random and
  mutation decoys on the first scaled shard.
- Root legacy scripts can confuse contributors; public docs now mark the
  package-native path as authoritative, but the final release should still avoid
  widening the compatibility surface.
- Release repositories must remain private until claim wording, provenance,
  package gates, manifest verification, and secret scans all pass.
