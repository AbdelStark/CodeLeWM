# Diagnostics-Driven Model Experiment

Last updated: 2026-06-07

Issue: #244. Tracker: #235. Status: specification complete; no training run
launched by this issue. The follow-up v0.9 epic #385 is complete through #392,
covering roadmap hygiene, cross-benchmark data/eval repair, guarded HF Jobs
execution, and the final diagnostic claim audit.

This document defines the next model-improvement experiment after the v1.4
visual observability work. It turns the `bugfix-edge-case` harness failure into
a falsifiable training plan instead of another blind sweep.

## Evidence Starting Point

The live `bugfix-edge-case` demo ran properly as a workflow:

- OpenRouter live candidate generation completed with Anthropic BYOK routing;
- 4/4 candidates were captured as valid, parseable patch artifacts;
- the learned package-native torch scorer loaded checkpoint `49965cd15fb4...`;
- manifest verification and secret scans passed;
- the demo claim gate stayed closed, as required.

The run is not positive model-quality evidence. CodeLeWM scores are transition
energies, so lower is better. The no-action score was `120.725449`, while the
best candidate score was `121.465622`; every generated patch scored worse than
doing nothing. The learned scorer also ranked `candidate_001` first among the
patches even though the failure analysis indicates it handled blank labels but
missed more complete normalization behavior that stronger candidates covered.

Current benchmark evidence points in the same direction:

- v0.2 text-action retrieval remains below no-action on headline metrics;
- action-contrast and latent-probe gates do not support semantic-axis claims;
- the downstream reranking path has fixture evidence, but not the scaled
  100-example benchmark required for coding-usefulness claims.

## Failure Diagnosis

The current scorer is learning a plausible next-state prior more than a
task-conditioned candidate-quality signal. The diagnostics suggest four failure
modes that the next experiment must distinguish:

- **No-action dominance:** before-state and local lexical priors remain strong
  enough that no-action can beat action-conditioned scoring.
- **Incomplete-candidate blindness:** a patch that makes a local change in the
  right region can receive a better energy than a semantically more complete
  patch.
- **Weak action pressure:** current action text does not force the latent
  transition to separate same-before, different-after candidates.
- **Unproven representation structure:** latent statistics and probe reports do
  not show stable semantic dimensions across seeds, splits, and controls.

## Intervention

Name: candidate-contrast action training.

Hypothesis: if training includes same-before, same-action candidate contrasts
where the positive after-state is paired with incomplete, no-op, over-broad, and
wrong-edit hard negatives, then CodeLeWM will assign lower energy to complete
task-solving candidates than to no-action and incomplete candidates on held-out
reranking tasks.

This is one intervention, not a bundle of unrelated changes:

1. Build a manifest-backed candidate-contrast training pack from public-safe
   transitions.
2. For each selected transition, keep the true after-state as positive and add
   bounded synthetic negative after-states:
   - no-op after-state;
   - partial local fix;
   - wrong-branch or wrong-symbol fix;
   - over-broad normalization;
   - shuffled-action same-before negative when available.
3. Train the existing transition model with a pairwise energy margin:
   `E(before, action, positive_after) + margin < E(before, action, negative_after)`.
4. Keep the current prediction, SIGReg, retrieval, and action-use losses as
   ablation-controlled components, but make the candidate-contrast margin the
   named experiment variable.

This intervention does not execute candidate code. Static analysis and optional
future sandbox checks can label candidate structure, but training inputs remain
untrusted text/artifact data.

## Required Data

Inputs:

- current public CodeLeWM transition shards;
- action-discriminative shard diagnostics from #152;
- hard-negative/action-contrast pools from the v0.2 work;
- meaningful harness scenario fixtures, especially `bugfix-edge-case`;
- optional live candidate packs only after secret-scan and license/source
  gates pass.

New artifact:

- `codelewm.data.candidate_contrast_pack.v1`

Minimum fields:

- `schema_version`;
- `source_transition_id`;
- `split`;
- `before_checksum`;
- `action_view`;
- `positive_after_checksum`;
- `negative_candidates[]` with `negative_kind`, checksum, parser status, static
  diagnostics, source provenance, and non-execution status;
- `license_decision`;
- `manifest_parent_artifacts`;
- `secret_scan`;
- `claim_boundary`.

Split policy:

- no source transition may appear in more than one split;
- generated negatives inherit the split of the source transition;
- live LLM negatives may be used only for train/dev unless a separate labeled
  benchmark split is created before generation.

## Config And HF Jobs Recipe

Planned implementation issue should add:

- data command: `codelewm dataset candidate-contrast-pack`;
- training config: `configs/training/v0_5_candidate_contrast_a10g.yaml`;
- HF launcher profile:
  `CODELEWM_HF_RUN_NAME=codelewm-v0-5-candidate-contrast-<date>-<sha>`;
- artifact card updates for dataset, model, and run repositories.

HF orchestration must use the `hf` CLI:

```bash
hf auth whoami
CODELEWM_HF_JOBS_DRY_RUN=1 uv run scripts/hf-launch-codelewm-job
hf jobs inspect <job-id>
hf jobs logs <job-id>
hf jobs stats <job-id>
hf download abdelstark/codelewm-runs <run-path> --local-dir <download-dir>
```

Local verification after download:

```bash
uv run codelewm manifest verify --manifest <run>/manifest.json --json
uv run codelewm secret-scan <run> --json
uv run codelewm eval retrieval ...
uv run codelewm eval ablation ...
uv run codelewm eval latent-probe ...
uv run codelewm eval latent-matrix ...
uv run codelewm eval downstream-rerank ...
uv run codelewm eval scorer-quality ...
```

## Metrics And Baselines

Training diagnostics:

- candidate-contrast margin satisfaction rate;
- positive energy distribution versus each negative kind;
- no-action delta distribution;
- action-use margin and retrieval-loss components;
- SIGReg/collapse diagnostics;
- effective rank, per-dimension variance, mean pairwise cosine, norm stats.

Representation metrics:

- `codelewm.eval.latent_probe_report.v1`;
- `codelewm.eval.latent_matrix_report.v1`;
- stable dimension associations across at least two seeds and held-out splits;
- semantic-axis claim gate remains closed unless axes beat lexical,
  metadata-only, random-latent, no-action, and shuffled-action controls.

Downstream reranking metrics:

- at least 100 labeled public-safe examples;
- pass@1/pass@k or equivalent task success labels;
- Recall@1, MRR, median rank, and true-rank distribution;
- valid-patch and static-check rates;
- scenario slices, including blank-label normalization.

Baselines:

- random order;
- lexical/static heuristic order;
- LLM original order;
- no-action score;
- shuffled-action score;
- transition-energy only;
- retrieval-prior only;
- any ensemble must report transition-only and retrieval-only ablations.

## Claim Gates

The experiment may support a positive model-quality claim only if all of these
are true:

- CodeLeWM improves over no-action and LLM-order baselines on the agreed
  headline downstream metrics;
- no-action dominance checks are false for text-action Recall@1 and MRR;
- candidate-contrast positives score better than each hard-negative class on
  held-out examples;
- latent probe and latent matrix gates do not show collapse and do not rely on
  unstable single-seed axes;
- checkpoint trust, artifact manifests, downloaded-artifact verification, and
  secret scans all pass;
- the claim-review report explicitly approves the wording.

If any gate fails, public claims remain limited to negative/diagnostic evidence.

## Expected Failure Modes

- The hard negatives are too synthetic and do not transfer to live LLM
  candidates.
- The model learns edit-size or formatting shortcuts instead of task semantics.
- Candidate-contrast improves held-out transition ranking but not downstream
  candidate reranking.
- No-action still wins because the before-state prior dominates.
- Latent probes improve without stable dimension-level semantics.
- Energy calibration shifts make rankings less interpretable across scenarios.

Each failure mode is publishable as diagnostic evidence if manifests, downloads,
verification, and scans pass.

## Blocked Claims If The Experiment Fails

If the run fails any claim gate, do not claim:

- CodeLeWM improves generated code;
- CodeLeWM has useful semantic latent axes;
- action conditioning is better than no-action;
- the harness scorer is reliable for candidate selection;
- TensorBoard, TUI, latent matrix, or checkpoint visualizations prove model
  quality.

Allowed language remains:

- the pipeline executed;
- artifacts were published and verified;
- diagnostics show where the current model fails;
- the result is negative or diagnostic evidence.

## Completed Follow-Up Issues

This issue only defines the experiment. The implementation has been split into
the completed v0.9 tracker and child issues:

- #385: v0.9 data/eval repair epic tracker;
- #386: reconcile stale trackers and roadmap queue state;
- #387: build the cross-benchmark pass/fail execution pack with stratified
  labels;
- #388: emit held-out `p_pass` ROC-AUC and calibration reports;
- #389: repair semantic-decoy alignment and coverage gates;
- #390: enforce probe-label coverage and representation gates;
- #391: execute guarded two-seed HF Jobs only after data/eval preflight;
- #392: run the final gate suite and publish the claim audit.
