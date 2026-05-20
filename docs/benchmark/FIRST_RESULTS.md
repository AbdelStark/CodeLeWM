# CodeLeWM First Results

- Report ID: `codelewm-first-results-2026-05-20`
- Schema version: `codelewm.first_results.v1`
- Evidence tier: smoke fixture, not scaled research evidence
- Source git SHA: `b7b259122a362103fdeefa2313c8b511a476bb37`
- Config bundle SHA-256: `6bf2b0b56dab663da0e42f29a5b0329aae3dc2213f5a14cd0db5d8a03d0af6f8`
- Runtime train config SHA-256: `a824192ddb13147219478695b6b4d8f24b069a007bcf715962aee459af03c8b0`
- Seed: dataset `7`, training `1337`, evaluation `0`
- Reproduction command: `uv run scripts/first-results --overwrite`

## Verdict

The complete local path now runs from a clean checkout: dataset build, pack, torch
training, retrieval evaluation, action-view ablation, surprise evaluation, transition-index build,
scorer/reranker quality reporting, manifest verification, report rendering, and secret scanning.

Text-action does not beat all required baselines on this fixture. The selected fixture has 1 held-out query and 1 retrieval candidate, so retrieval Recall@k is saturated.
This report is therefore useful as reproducibility evidence, not as evidence that
CodeLeWM has learned general action-conditioned code-edit structure.

## Reproduce

```bash
uv sync --group dev --group data --group train
uv run scripts/first-results --overwrite
uv run codelewm secret-scan .artifacts/first-results docs/benchmark/FIRST_RESULTS.md --json
```

The runner writes `.artifacts/first-results/manifest_inventory.json` with the
machine-readable command outputs and artifact IDs used by this report.

## Artifact Cards

- Dataset card: `docs/cards/codelewm-first-results-dataset-2026-05-19.md`.
- Model card: `docs/cards/codelewm-first-results-model-2026-05-19.md`.
- Both cards are filled from this smoke artifact set and must not be reused for scaled claims.

## Exact Commands

1. `uv run codelewm dataset build --config config/first_results/dataset_build.json --out .artifacts/first-results/build --json`
2. `uv run codelewm dataset pack --manifest .artifacts/first-results/build/manifest.json --out .artifacts/first-results/pack --json`
3. `uv run codelewm train --config .artifacts/first-results/configs/train_tiny.json --executor torch --device cpu --overwrite --json --log-jsonl .artifacts/first-results/logs/train.jsonl`
4. `uv run codelewm eval retrieval --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/retrieval --device cpu --seed 0 --overwrite --json --log-jsonl .artifacts/first-results/logs/retrieval.jsonl`
5. `uv run codelewm eval ablation --retrieval-artifact .artifacts/first-results/retrieval/manifest.json --training-artifact .artifacts/first-results/train/manifest.json --out .artifacts/first-results/ablation --overwrite --json --log-jsonl .artifacts/first-results/logs/ablation.jsonl`
6. `uv run codelewm eval surprise --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/surprise --device cpu --seed 0 --overwrite --json --log-jsonl .artifacts/first-results/logs/surprise.jsonl`
7. `uv run codelewm index --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/index --device cpu --overwrite --json --log-jsonl .artifacts/first-results/logs/index.jsonl`
8. `uv run codelewm eval scorer-quality --config config/first_results/scorer_quality.json --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --out .artifacts/first-results/scorer_quality --device cpu --index .artifacts/first-results/index --retrieval-prior-weight 1.0 --retrieval-prior-k 1 --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/index/manifest.json --overwrite --json --log-jsonl .artifacts/first-results/logs/scorer_quality.jsonl`
9. `uv run codelewm manifest verify --manifest .artifacts/first-results/build/manifest.json --json`
10. `uv run codelewm manifest verify --manifest .artifacts/first-results/pack/manifest.json --parent-manifest .artifacts/first-results/build/manifest.json --json`
11. `uv run codelewm manifest verify --manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json`
12. `uv run codelewm manifest verify --manifest .artifacts/first-results/retrieval/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json`
13. `uv run codelewm manifest verify --manifest .artifacts/first-results/ablation/manifest.json --parent-manifest .artifacts/first-results/retrieval/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --json`
14. `uv run codelewm manifest verify --manifest .artifacts/first-results/surprise/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json`
15. `uv run codelewm manifest verify --manifest .artifacts/first-results/index/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json`
16. `uv run codelewm manifest verify --manifest .artifacts/first-results/scorer_quality/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/index/manifest.json --json`
17. `uv run codelewm secret-scan .artifacts/first-results docs/benchmark/FIRST_RESULTS.md --json`

## Reproducibility Chain

| Artifact | Schema version | Manifest path | Artifact ID | Config SHA prefix |
| -------- | -------------- | ------------- | ----------- | ----------------- |
| dataset_build | codelewm.artifact_manifest.v1 | .artifacts/first-results/build/manifest.json | dataset-36dc92f0467bbb62 | a87cc5fbfcfe |
| dataset_pack | codelewm.artifact_manifest.v1 | .artifacts/first-results/pack/manifest.json | dataset-de5bf0e43f57e973 | 6ae8bed29201 |
| training_run | codelewm.artifact_manifest.v1 | .artifacts/first-results/train/manifest.json | training_run-b90df04419539c57 | a908d8aa3dc8 |
| retrieval_eval | codelewm.artifact_manifest.v1 | .artifacts/first-results/retrieval/manifest.json | eval_report-bbe01fc3b26d473e | 94dab553d9a3 |
| action_ablation | codelewm.artifact_manifest.v1 | .artifacts/first-results/ablation/manifest.json | eval_report-1e788a30c2fff3cd | 4afab675f686 |
| surprise_eval | codelewm.artifact_manifest.v1 | .artifacts/first-results/surprise/manifest.json | eval_report-e2de0929135acac6 | b02bfb81e70c |
| transition_index | codelewm.artifact_manifest.v1 | .artifacts/first-results/index/manifest.json | index-b640ef8750d7a1b3 | 100530ae6b92 |
| scorer_quality | codelewm.artifact_manifest.v1 | .artifacts/first-results/scorer_quality/manifest.json | score_report-7aab8b4778d2fa0b | 65e01cbb7e78 |
| Checkpoint | `codelewm.checkpoint.v1` | `checkpoint.pt` | `36010719687aa847` | `72822cb45ab8` |
| License gate | `codelewm.public_license_gate.v1` | `.artifacts/first-results/build/reports/license_gate_report.json` | `release_allowed=true` | n/a |

## Manifest Verification

| Artifact | Result | Files checked | Required parents | Command |
| -------- | ------ | ------------- | ---------------- | ------- |
| dataset_build | pass | 9 | none | uv run codelewm manifest verify --manifest .artifacts/first-results/build/manifest.json --json |
| dataset_pack | pass | 10 | dataset-36dc92f0467bbb62 | uv run codelewm manifest verify --manifest .artifacts/first-results/pack/manifest.json --parent-manifest .artifacts/first-results/build/manifest.json --json |
| training_run | pass | 6 | dataset-de5bf0e43f57e973 | uv run codelewm manifest verify --manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json |
| retrieval_eval | pass | 3 | training_run-b90df04419539c57, dataset-de5bf0e43f57e973 | uv run codelewm manifest verify --manifest .artifacts/first-results/retrieval/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json |
| action_ablation | pass | 1 | eval_report-bbe01fc3b26d473e, training_run-b90df04419539c57 | uv run codelewm manifest verify --manifest .artifacts/first-results/ablation/manifest.json --parent-manifest .artifacts/first-results/retrieval/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --json |
| surprise_eval | pass | 2 | training_run-b90df04419539c57, dataset-de5bf0e43f57e973 | uv run codelewm manifest verify --manifest .artifacts/first-results/surprise/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json |
| transition_index | pass | 3 | training_run-b90df04419539c57, dataset-de5bf0e43f57e973 | uv run codelewm manifest verify --manifest .artifacts/first-results/index/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json |
| scorer_quality | pass | 2 | training_run-b90df04419539c57, index-b640ef8750d7a1b3 | uv run codelewm manifest verify --manifest .artifacts/first-results/scorer_quality/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/index/manifest.json --json |

## Dataset And Training

- Packed rows: `3`; splits: `{"test": 0, "train": 2, "val": 1}`.
- License gate: release_allowed `true`, included rows `3`, excluded rows `1`, blocked rows `0`.
- Action-discriminative shard report: `codelewm.data.action_discriminative_shard_report.v1`; claim-ready `false`; held-out rows `1`.
- Available hard-negative pools: `diff_shape_controlled, edit_size_controlled`.
- Training executor: `torch` on `cpu` for `4` steps.
- Final loss: total `0.654402`, prediction MSE `0.507995`, SIGReg `1.62675`.
- Collapse diagnostics: effective rank `1.12557`, variance min `0.000130734`, nearest-neighbor entropy `1.56071`.

## Retrieval Evaluation

- Report schema: `codelewm.eval.retrieval_report.v1`.
- Headline text-action metrics: Recall@1 `1`, Recall@5 `1`, Recall@10 `1`, MRR `1`, median rank `1`.
- Candidate pool: `easy-1k`, entries `1`, excluded splits `train`.

| Baseline | Text Recall@1 | Baseline Recall@1 | Text MRR | Baseline MRR | Text beats baseline? |
| -------- | ------------- | ----------------- | -------- | ------------ | -------------------- |
| Random | 1 | 1 | 1 | 1 | no |
| Lexical | 1 | 1 | 1 | 1 | no |
| No-action | 1 | 1 | 1 | 1 | no |
| Shuffled-action | 1 | 1 | 1 | 1 | no |
| Patch-action diagnostic | n/a | n/a | n/a | n/a | not run for this headline smoke report |

### Action-Use Claim Gate

- Claim gate schema: `codelewm.eval.action_use_claim_gate.v1`.
- Positive action-conditioning claim allowed: `false`.
- Failure reasons: `baseline_not_beaten:random`, `baseline_not_beaten:lexical`, `no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action`, `baseline_not_beaten:shuffled_action`.

## Action-View Ablation

- Report schema: `codelewm.eval.action_ablation_report.v1`.
- Completed rows: `7`; blocked rows: `5`; failed rows: `0`.
- Ablation claim gate allowed: `false`.
- Missing abstract-action, retrieval-loss, patch-action diagnostic, and alternate SIGReg runs are explicit blocked rows rather than dropped rows.

| Row | Family | Status | Recall@1 | MRR | Reason |
| --- | ------ | ------ | -------- | --- | ------ |
| text_action | action_view | completed | 1 | 1 | available |
| abstract_action | action_view | blocked | n/a | n/a | no abstract-action checkpoint and retrieval report were supplied |
| patch_action_diagnostic | action_view | blocked | n/a | n/a | patch-action diagnostic upper-bound report was not supplied |
| random | baseline | completed | 1 | 1 | available |
| lexical | baseline | completed | 1 | 1 | available |
| no_action | baseline | completed | 1 | 1 | available |
| shuffled_action | baseline | completed | 1 | 1 | available |
| retrieval_loss_disabled | retrieval_loss | completed | 1 | 1 | available |
| retrieval_loss_enabled | retrieval_loss | blocked | n/a | n/a | paired retrieval-loss variant report was not supplied |
| collapse_sigreg_0.09 | collapse | completed | n/a | n/a | available |
| collapse_sigreg_0.05 | collapse | blocked | n/a | n/a | paired SIGReg/collapse setting report was not supplied |
| collapse_sigreg_0.15 | collapse | blocked | n/a | n/a | paired SIGReg/collapse setting report was not supplied |

## Patch-Surprise Evaluation

- Report schema: `codelewm.eval.surprise_report.v1`.
- Overall pairwise AUC: `0`.
- Mean true rank: `2`; median true rank: `2`; Recall@1 `0`.
- Examples scored: `1`; score direction: `lower_is_better`.

| Decoy category | Pairwise AUC | Decoy count | Caveat |
| -------------- | ------------ | ----------- | ------ |
| `random` | n/a | 0 | no other held-out after-states were available |
| `same_file` | n/a | 0 | no same-file held-out after-states were available |
| `mutation` | 0 | 1 | available |
| `action_cluster` | n/a | 0 | no same-action-cluster held-out after-states were available |

## Transition Index

- Index schema: `codelewm.transition_index.v1`.
- Count: `2` train-split vectors; dimension `256`; distance `l2`.
- Indexed splits: `train`.

## Scorer And Reranker Quality

- Report schema: `codelewm.harness.scorer_quality_report.v1`.
- Examples: `1`; candidates `4`; valid `2`; errors `2`.
- Ranking: Recall@1 `1`, MRR `1`, mean true rank `1`, median true rank `1`.
- Retrieval prior: weight `1.0`, k `1`; risk penalty `reserved; current scorer reports null and treats failures as error rows`.
- Failure counts: `{"invalid_syntax": 1, "patch_apply_failed": 1}`.
- Execution policy: `candidate code is parsed and diff-applied as text but never executed`.

| Slice | Candidates | Valid | Errors | Mean final score | Mean retrieval prior |
| ----- | ---------- | ----- | ------ | ---------------- | -------------------- |
| hard_negative | 1 | 1 | 0 | 119.769 | 118.701 |
| patch_failure | 1 | 0 | 1 | n/a | n/a |
| syntax_failure | 1 | 0 | 1 | n/a | n/a |
| true_after | 1 | 1 | 0 | 119.646 | 118.616 |

## Security Evidence

- Secret scan result: `pass`.
- Paths scanned: `46`.
- Findings: `0`.
- Published artifact policy: local fixture artifacts are full-text and pass the configured permissive-license gate.

## Claim Checklist

- [ ] Text-action beats random, lexical, no-action, and shuffled-action baselines on Recall@1 and MRR.
- [ ] Action-use claim gate allows a positive action-conditioning claim.
- [ ] Dataset shard has action-discriminative hard-negative coverage.
- [x] Headline retrieval uses `action_text`.
- [x] Action-view ablation records missing variants as blocked rows.
- [x] Scorer/reranker quality report records ranking metrics, calibration slices, failures, and caveats.
- [x] Hard-negative and candidate pools exclude `train` split rows.
- [ ] Patch-surprise covers all four decoy categories with non-zero decoy counts.
- [x] Every selected artifact manifest verifies with required parents.
- [x] Secret scan passes over the selected first-results artifact directory and this report.
- [x] License gate passed for the local fixture artifact.
- [ ] This report supports a scaled research claim about learned action-conditioned structure.

## Caveats

- Smoke evidence: this run proves the package-native workflow and artifact lineage,
  including trusted checkpoint loading and index-backed evaluation prerequisites.
- Research evidence: this fixture is too small for a learning claim. It has one
  held-out query, no random same-corpus retrieval competition beyond the true
  target, only mutation surprise decoys, and one scorer-quality example. Baseline ties,
  failed surprise rankings, and fixture-only reranker calibration must be read as
  blockers for any public model-quality claim.
- Next required work is a bounded public-safe shard with enough held-out examples
  to make random, lexical, no-action, shuffled-action, and surprise decoy
  comparisons meaningful.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| -------- | ---- | ------------- | ---- |
| Pending | First-results smoke review | Pending | Pending |
