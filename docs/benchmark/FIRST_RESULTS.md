# CodeLeWM First Results

- Report ID: `codelewm-first-results-2026-05-19`
- Schema version: `codelewm.first_results.v1`
- Evidence tier: smoke fixture, not scaled research evidence
- Source git SHA: `7ba35afb93b6bd8988501a193ea025cd6ac5b128`
- Config bundle SHA-256: `f2f2e36f531f883ecc701f0d1903c8b73e9034305837a5c6e9edc612221e5f5c`
- Runtime train config SHA-256: `a824192ddb13147219478695b6b4d8f24b069a007bcf715962aee459af03c8b0`
- Seed: dataset `7`, training `1337`, evaluation `0`
- Reproduction command: `uv run scripts/first-results --overwrite`

## Verdict

The complete local path now runs from a clean checkout: dataset build, pack, torch
training, retrieval evaluation, surprise evaluation, transition-index build,
manifest verification, report rendering, and secret scanning.

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

## Exact Commands

1. `uv run codelewm dataset build --config config/first_results/dataset_build.json --out .artifacts/first-results/build --json`
2. `uv run codelewm dataset pack --manifest .artifacts/first-results/build/manifest.json --out .artifacts/first-results/pack --json`
3. `uv run codelewm train --config .artifacts/first-results/configs/train_tiny.json --executor torch --device cpu --overwrite --json --log-jsonl .artifacts/first-results/logs/train.jsonl`
4. `uv run codelewm eval retrieval --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/retrieval --device cpu --seed 0 --overwrite --json --log-jsonl .artifacts/first-results/logs/retrieval.jsonl`
5. `uv run codelewm eval surprise --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/surprise --device cpu --seed 0 --overwrite --json --log-jsonl .artifacts/first-results/logs/surprise.jsonl`
6. `uv run codelewm index --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/index --device cpu --overwrite --json --log-jsonl .artifacts/first-results/logs/index.jsonl`
7. `uv run codelewm manifest verify --manifest .artifacts/first-results/build/manifest.json --json`
8. `uv run codelewm manifest verify --manifest .artifacts/first-results/pack/manifest.json --parent-manifest .artifacts/first-results/build/manifest.json --json`
9. `uv run codelewm manifest verify --manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json`
10. `uv run codelewm manifest verify --manifest .artifacts/first-results/retrieval/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json`
11. `uv run codelewm manifest verify --manifest .artifacts/first-results/surprise/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json`
12. `uv run codelewm manifest verify --manifest .artifacts/first-results/index/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json`
13. `uv run codelewm secret-scan .artifacts/first-results docs/benchmark/FIRST_RESULTS.md --json`

## Reproducibility Chain

| Artifact | Schema version | Manifest path | Artifact ID | Config SHA prefix |
| -------- | -------------- | ------------- | ----------- | ----------------- |
| dataset_build | codelewm.artifact_manifest.v1 | .artifacts/first-results/build/manifest.json | dataset-8a9eb207e80916a0 | a87cc5fbfcfe |
| dataset_pack | codelewm.artifact_manifest.v1 | .artifacts/first-results/pack/manifest.json | dataset-83212742cc9ccba8 | 6ae8bed29201 |
| training_run | codelewm.artifact_manifest.v1 | .artifacts/first-results/train/manifest.json | training_run-a37db3267f583c5e | a908d8aa3dc8 |
| retrieval_eval | codelewm.artifact_manifest.v1 | .artifacts/first-results/retrieval/manifest.json | eval_report-7763e41366bcd1ca | 94dab553d9a3 |
| surprise_eval | codelewm.artifact_manifest.v1 | .artifacts/first-results/surprise/manifest.json | eval_report-d28fe293b10d8385 | b02bfb81e70c |
| transition_index | codelewm.artifact_manifest.v1 | .artifacts/first-results/index/manifest.json | index-3ea18bd8cdb42854 | dd2d5d163954 |
| Checkpoint | `codelewm.checkpoint.v1` | `checkpoint.pt` | `89519896a6d378a3` | `72822cb45ab8` |
| License gate | `codelewm.public_license_gate.v1` | `.artifacts/first-results/build/reports/license_gate_report.json` | `release_allowed=true` | n/a |

## Manifest Verification

| Artifact | Result | Files checked | Required parents | Command |
| -------- | ------ | ------------- | ---------------- | ------- |
| dataset_build | pass | 7 | none | uv run codelewm manifest verify --manifest .artifacts/first-results/build/manifest.json --json |
| dataset_pack | pass | 9 | dataset-8a9eb207e80916a0 | uv run codelewm manifest verify --manifest .artifacts/first-results/pack/manifest.json --parent-manifest .artifacts/first-results/build/manifest.json --json |
| training_run | pass | 6 | dataset-83212742cc9ccba8 | uv run codelewm manifest verify --manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json |
| retrieval_eval | pass | 3 | training_run-a37db3267f583c5e, dataset-83212742cc9ccba8 | uv run codelewm manifest verify --manifest .artifacts/first-results/retrieval/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json |
| surprise_eval | pass | 2 | training_run-a37db3267f583c5e, dataset-83212742cc9ccba8 | uv run codelewm manifest verify --manifest .artifacts/first-results/surprise/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json |
| transition_index | pass | 3 | training_run-a37db3267f583c5e, dataset-83212742cc9ccba8 | uv run codelewm manifest verify --manifest .artifacts/first-results/index/manifest.json --parent-manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json |

## Dataset And Training

- Packed rows: `3`; splits: `{"test": 0, "train": 2, "val": 1}`.
- License gate: release_allowed `true`, included rows `3`, excluded rows `1`, blocked rows `0`.
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

## Security Evidence

- Secret scan result: `pass`.
- Paths scanned: `36`.
- Findings: `0`.
- Published artifact policy: local fixture artifacts are full-text and pass the configured permissive-license gate.

## Claim Checklist

- [ ] Text-action beats random, lexical, no-action, and shuffled-action baselines on Recall@1 and MRR.
- [x] Headline retrieval uses `action_text`.
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
  target, and only mutation surprise decoys. Baseline ties and failed surprise
  rankings must be read as blockers for any public model-quality claim.
- Next required work is a bounded public-safe shard with enough held-out examples
  to make random, lexical, no-action, shuffled-action, and surprise decoy
  comparisons meaningful.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| -------- | ---- | ------------- | ---- |
| Pending | First-results smoke review | Pending | Pending |
