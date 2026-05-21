# Full Completion Roadmap

Last updated: 2026-05-21

This roadmap tracks the current completion boundary for CodeLeWM's first
meaningful scaled training and evaluation artifacts. GitHub issues remain the
source of truth for implementation state; this document explains what is
complete, what claims are blocked, and what a future positive-claim research
iteration would need.

The next research-planning `/goal` prompt lives in
`docs/roadmap/NEXT_GOAL_PROMPT.md`. The v0.2 HF/ml-intern prompt in
`docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md` is historical context for the
completed #172 run. The v0.2 research intervention spec lives in
`docs/roadmap/V0_2_ACTION_USE_RESEARCH_PLAN.md`. The post-v0.2 harness and
publication roadmap lives in `docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md`.

## Project Status

The repository is past pure specification and past local smoke evidence. The
package contains data contracts, model components, manifest-backed training,
evaluation contracts, scoring/reranking harness commands, HF Jobs automation,
observability, security gates, CI, release templates, a local first-results
smoke loop, and four completed scaled HF Jobs runs.

Completed evidence:

- `scripts/first-results` regenerates the local smoke evidence in
  `docs/benchmark/FIRST_RESULTS.md`.
- HF Jobs run `codelewm-scaled-20260520-9699b53` completed on job
  `6a0d43c92dc5b1243da50bba`.
- Follow-up HF Jobs run `codelewm-action-use-20260520-6650183` completed on job
  `6a0d7a763aba298b21d147a9` with the no-action margin objective.
- Public HF repositories contain the dataset pack, model checkpoint, and run
  evidence under the documented run paths.
- The artifacts were downloaded with `hf download`, verified locally,
  and rerun through retrieval, ablation, surprise, scorer-quality, score, and
  rerank checks.
- `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md` and the scaled dataset/model
  cards are the artifact-backed report for that run.
- `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md` and the action-use
  dataset/model cards are the artifact-backed report for the #154 follow-up
  run.
- The #159 second-stage remediation run
  `codelewm-action-use-retrieval-20260520-7895d18` completed as HF Jobs job
  `6a0da3a08229e585f969c3f7` from source
  `7895d185e165a917af0956a313d8948c04b33638` with
  `config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml`,
  `a10g-small`, and a `24h` timeout. It published artifacts to
  `abdelstark/codelewm-public-shard`,
  `abdelstark/codelewm-transition-model`, and `abdelstark/codelewm-runs`,
  then the artifacts were downloaded with `hf download` and verified locally.
- `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md` and the
  paired retrieval dataset/model cards are the artifact-backed report for #159.
- The v0.2 action-swap/inverse-action run
  `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` completed as HF Jobs job
  `6a0dea258229e585f969c808` from source
  `7c7cb0b8fe132e4819f05a77585c254267e77574` with
  `config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml`,
  `a10g-small`, and a `24h` timeout. It published artifacts to
  `abdelstark/codelewm-public-shard`,
  `abdelstark/codelewm-transition-model`, and `abdelstark/codelewm-runs`,
  then the artifacts were downloaded with `hf download` and verified locally.
- `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` and the paired
  v0.2 dataset/model cards are the artifact-backed report for #172.

Current blocker:

- All four scaled runs are meaningful systems evidence, but none is a positive
  action-conditioned model-quality result.
- The #154 action-use run beats random, shuffled-action, and lexical baselines,
  but no-action is stronger on headline retrieval: text-action Recall@1
  `0.363`, MRR `0.467875`; no-action Recall@1 `0.469`, MRR `0.549624`.
- The #159 margin+retrieval run improves text-action retrieval to Recall@1
  `0.597`, MRR `0.674500`, but no-action is still stronger: Recall@1 `0.650`,
  MRR `0.708037`. The claim gate is `claim_allowed=false`.
- The v0.2 action-swap/inverse-action run reaches text-action Recall@1
  `0.263`, MRR `0.370048`, while no-action reaches Recall@1 `0.441`,
  MRR `0.533105`. It also fails the exact-same-before and near-before
  action-contrast margins, the latent-probe representation gate, and the
  scaled downstream-reranking gate.
- The private diagnostic release-freeze gate is closed by #126 and recorded in
  `docs/release/RELEASE_FREEZE_2026-05-20.md`.
- The implementation milestone is complete as public negative/diagnostic
  evidence. Public positive action-conditioning, semantic latent-axis, and
  downstream coding-usefulness claims remain blocked. Future positive-claim
  work requires a new research hypothesis beyond the completed v0.2
  intervention.
- The v1.1 LLM + world-model harness and downstream benchmark milestone is
  complete as a claim-safe diagnostic workflow: #183, #184, and #185 tracked
  the streams, with child issues #186 through #194. Issue #206 adds the
  public BYOK/local-demo usability pass.
- The next open streams are live harness evidence (#207/#208), scaled
  downstream benchmark evidence (#209/#210/#211), and a future positive-model
  research hypothesis (#212, related to #178).

Current landed CLI commands:

- `codelewm dataset build`
- `codelewm dataset pack`
- `codelewm train`
- `codelewm eval retrieval`
- `codelewm eval ablation`
- `codelewm eval surprise`
- `codelewm eval scorer-quality`
- `codelewm eval downstream-pack`
- `codelewm eval downstream-rerank`
- `codelewm index`
- `codelewm score`
- `codelewm rerank`
- `codelewm llm-demo`
- `codelewm openrouter byok-register`
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
- Published artifacts can be downloaded with `hf download` and verified
  locally from a clean checkout.

For a positive public action-conditioning claim, text-action must beat the
no-action baseline on the agreed headline metrics. The current artifact set does
not meet that bar, so all release docs must keep the negative/diagnostic
boundary explicit.

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

Remaining for a positive claim: a new research issue with a stronger hypothesis
than the completed #159 margin+retrieval and v0.2 action-swap remediations.

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

Remaining for a positive claim: broader scorer calibration and a new
action-conditioned training/eval intervention.

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
- dataset/model/results publication
- downloaded-artifact verification
- scaled benchmark report and cards

Blocker: no-action baseline beats text-action on headline retrieval in the
#138 baseline run, the #154 no-action margin follow-up run, the #159
margin+retrieval remediation run, and the #172 v0.2 action-swap run.

## Remaining Phases

### Phase 7: Action-Use Remediation And v0.2 Intervention

Status: complete as negative/diagnostic evidence.

Goal: make action use measurable, train against the failure mode, and rerun the
scaled HF loop from published artifacts.

Deliverables:

- no-action dominance diagnostics and machine-readable claim gates (#151,
  complete)
- action-discriminative shard diagnostics and hard negatives (#152, complete)
- action-use objective/intervention and scaled sweep configs (#153, complete)
- follow-up HF Jobs run launched, monitored, downloaded, verified, inferred, and
  evaluated through the `hf` CLI (#154, complete with a negative claim gate)
- second-stage action-use remediation sweep using the margin+retrieval fallback
  config, completed as run `codelewm-action-use-retrieval-20260520-7895d18` on
  HF Jobs job `6a0da3a08229e585f969c3f7`; downloaded-artifact verification and
  local eval/inference checks passed, but the claim gate remained negative
  (#159)
- v0.2 action-contrast pools (#171), latent probe suite (#168), action-swap /
  inverse-action intervention config (#170), downstream reranking benchmark
  contract (#169), and public HF v0.2 sweep (#172), completed as run
  `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` on HF Jobs job
  `6a0dea258229e585f969c808`; downloaded-artifact verification passed, but the
  action-use, representation, and downstream gates remained negative

### Phase 8: Publishing And Release

Goal: package and publish only after the evidence boundary is clear.

Deliverables:

- wheel/sdist build and package publishing gates (#123, complete)
- dependency audit and provenance evidence (#124, complete)
- public docs refreshed against the scaled evidence and claim boundary (#125,
  complete)
- final artifact freeze, release checklist, and release notes (#126, complete
  as a private diagnostic freeze)

### Phase 9: LLM + World-Model Harness Demo

Status: complete as a claim-safe demo workflow. This still does not support a
coding-usefulness claim.

Goal: showcase the intended use case without claiming model improvement. An LLM
generates candidate patches through the OpenRouter Python SDK, CodeLeWM
scores/reranks the candidates, and a demo report records all baselines and
candidate errors.

Deliverables:

- OpenRouter candidate harness contract (#186)
- OpenRouter candidate generation adapter (#187)
- candidate-pack schema and safe patch capture (#188)
- end-to-end LLM + CodeLeWM demo report (#189)
- OpenRouter BYOK helper, local `uv run scripts/llm-world-model-demo` task,
  and public README polish (#206)
- learned torch checkpoint inference in the LLM demo scorer path (#220, open)

Claim boundary: this can prove the workflow, but not model usefulness.

### Phase 10: Downstream Candidate-Reranking Benchmark

Status: complete as a diagnostic fixture and claim-gated benchmark path. This
still does not support a coding-usefulness claim.

Goal: turn the harness into a falsifiable downstream benchmark with enough
labeled examples to support or block coding-usefulness claims.

Deliverables:

- downstream task schema, baselines, metrics, and claim gates (#190)
- public-safe labeled candidate reranking set (#191, complete)
- downstream reranking comparison and claim gate (#192, complete)
- scaled public-safe 100-example reranking set (#210, open)
- scaled downstream comparison and claim gate (#211, open)

Minimum success bar: at least 100 labeled examples and CodeLeWM improvement
over no-action and LLM-order baselines on the agreed headline metrics.

### Phase 11: Preliminary Results Publication

Status: complete as negative/diagnostic public evidence.

Goal: publish the current project state as verified infrastructure and a
negative/diagnostic result, not as a positive model-quality claim.

Deliverables:

- preliminary negative-results report (#193)
- public artifact index and announcement package (#194)

Publication artifacts:

- `docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md`
- `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-21.md`
- `docs/announcements/PRELIMINARY_RESULTS_2026-05-21.md`

## Ordered Backlog

Keep this table in implementation order and update it when issue scope changes.

| Order | Issue | Title | Milestone | Blocks |
| ----- | ----- | ----- | --------- | ------ |
| 1 | #186 | spec: lock OpenRouter LLM candidate harness contract | Harness Demo | complete |
| 2 | #193 | docs: publish preliminary negative-results report | Publication | complete |
| 3 | #194 | docs: prepare public artifact index and announcement package | Publication | complete |
| 4 | #187 | harness: add OpenRouter candidate generation adapter | Harness Demo | complete |
| 5 | #188 | harness: add candidate pack schema and safe patch capture | Harness Demo | complete |
| 6 | #189 | harness: build end-to-end LLM plus CodeLeWM demo report | Harness Demo | complete |
| 7 | #190 | benchmark: define downstream task schema and claim gates | Downstream Benchmark | complete |
| 8 | #191 | benchmark: build public-safe labeled candidate reranking set | Downstream Benchmark | complete |
| 9 | #192 | eval: run downstream reranking comparison and claim gate | Downstream Benchmark | complete |
| 10 | #206 | harness: add OpenRouter BYOK demo task and public README polish | Harness Demo | complete |
| 11 | #220 | harness: use learned world-model inference in LLM demo | Live Harness Evidence | #207 |
| 12 | #208 | run: execute live OpenRouter BYOK harness demo and publish diagnostic artifacts | Live Harness Evidence | #220 |
| 13 | #210 | data: build public-safe 100-example downstream reranking set | Scaled Downstream Benchmark | #209 |
| 14 | #211 | eval: run scaled downstream reranking comparison and claim gate | Scaled Downstream Benchmark | #210 |
| 15 | #178/#212 | evaluate CWM reuse and define next positive-model hypothesis | Research | open |

Completed backlog base:

- #109 through #122 closed the first-results and scaled-artifact implementation
  path.
- #137 added HF Jobs and ml-intern automation.
- #138 executed the first scaled HF Jobs run, artifact publication,
  downloaded-artifact verification, inference, and evals.
- #151 added no-action dominance diagnostics and machine-readable action-use
  claim gates.
- #152 added action-discriminative shard diagnostics and hard-negative metadata.
- #153 added the no-action margin objective and A10G action-use sweep configs.
- #154 executed the primary action-use HF Jobs follow-up run and verified the
  downloaded artifacts; the run remained negative because no-action
  still beat text-action.
- #159 executed run `codelewm-action-use-retrieval-20260520-7895d18` on HF Jobs
  job `6a0da3a08229e585f969c3f7`; artifacts were downloaded and
  verified locally, and the result remained negative because no-action still
  beat text-action.
- #167 tracked the completed v0.2 action-use and representation intervention.
- #171 built action-contrast benchmark pools.
- #168 added latent representation probes and axis diagnostics.
- #170 added v0.2 action-fusion and contrastive action-use objectives.
- #169 added the downstream reranking benchmark contract.
- #172 executed run `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` on HF
  Jobs job `6a0dea258229e585f969c808`; artifacts were downloaded and verified
  locally, and the result remained negative across action-use, representation,
  and downstream-readiness gates.
- #183 tracks the v1.1 LLM + world-model harness demo stream.
- #184 tracks the v1.1 downstream candidate-reranking benchmark stream.
- #185 tracks the preliminary results publication stream.
- #123 added wheel/sdist build, metadata, clean-install, typed marker, and
  manual publishing gates for the Python package.
- #124 added release dependency audit, provenance JSON, CI gates, and release
  checklist evidence for the package supply-chain path.
- #125 refreshed README, usage, public API, roadmap, and release docs against
  current artifact evidence and the negative action-use claim boundary.
- #126 froze the private diagnostic release artifact set, package/provenance
  evidence, and release checklist status without enabling public positive
  action-conditioning claims.
- #150 tracks the action-conditioned scaled-result and release readiness
  milestone; it can close with the explicit negative/diagnostic boundary.

## Tracking Issue Mapping

Use the existing subsystem tracking issues and the new completion tracker
instead of creating duplicate trackers.

- #3 Edit transition dataset: #152 complete; no active child.
- #7 Training runtime: #153 complete; no active child.
- #8 Retrieval and surprise evaluation: #159 complete; no active child.
- #9 Harness scorer and reranker: #159 complete; no active child.
- #10 Observability and artifact lineage: #159 complete; no active child.
- #11 Security and licensing boundaries: no active child.
- #12 Public API and packaging: no active child.
- #13 Release CI and governance: #150 closed with the negative/diagnostic
  boundary.
- #150 Action-conditioned scaled result and release readiness: #159 complete;
  closed with the negative/diagnostic boundary.
- #167 v0.2 action-use and representation research intervention: complete as
  negative/diagnostic evidence through #168 through #172.
- #183 LLM + world-model harness demo: complete; children #186 through #189.
- #184 Downstream candidate-reranking benchmark: complete; children #190
  through #192 are closed.
- #185 Preliminary results publication package: complete; children #193 and
  #194 are closed.
- #207 Live LLM plus world-model harness evidence: open; child #220 is the
  learned-scorer precondition for #208.
- #209 Scaled downstream reranking benchmark: open; children #210 and #211
  remain.
- #212 Next positive-model research hypothesis: open; related issue #178
  remains.

Close a tracking issue only when every child issue in its subsystem is complete
or explicitly superseded and the release checklist no longer lists a blocker for
that subsystem.

## Residual Risks

- The scaled results fail the core action-conditioning claim gate because
  no-action beats text-action, including in the v0.2 action-contrast sweep.
- The current public shard may still not contain enough action-discriminative
  pressure for text actions to matter, even with targeted hard negatives.
- The no-action margin objective was proven insufficient by #154, the
  margin+retrieval remediation was proven insufficient by #159, and the
  action-swap/inverse-action intervention was proven insufficient by #172.
- The v0.2 latent-probe gate does not support semantic latent-axis claims.
- The v0.2 downstream scorer-quality path remains blocked as scaled evidence
  because it has one labeled example instead of the required 100.
- Scorer-quality evidence is still small and should remain a gate, not a broad
  calibration claim.
- Same-file and action-cluster surprise decoy counts are lower than random and
  mutation decoys on the first scaled shard.
- Root legacy scripts can confuse contributors; public docs now mark the
  package-native path as authoritative, but the final release should still avoid
  widening the compatibility surface.
- HF repositories are public diagnostic artifact repositories; the current
  evidence boundary still does not support public positive action-conditioning
  claims.
