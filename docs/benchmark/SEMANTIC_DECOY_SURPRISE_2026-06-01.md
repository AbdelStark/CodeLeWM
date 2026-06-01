# v0.6 Semantic Decoy Surprise Rerun

Date: 2026-06-01

Issue: #322

This reruns `codelewm eval execution-surprise` for both v0.6 training seeds
against the strengthened semantic decoy pack from #321. The report separates
numeric score gates from pair-count gates so a small decoy category cannot
support a broad semantic surprise claim by accident.

## Artifacts

| Seed | Manifest | Artifact ID |
| ---: | -------- | ----------- |
| 42 | `docs/benchmark/v0_6/seed-42/execution_surprise_semantic/manifest.json` | `eval_report-aeb0ae374582a8ec` |
| 1729 | `docs/benchmark/v0_6/seed-1729/execution_surprise_semantic/manifest.json` | `eval_report-5b20bd0e1da5928a` |

Both artifacts have three parents:

- the corresponding trusted v0.6 training run;
- `codelewm-execution-pack-20260528T102625Z`;
- semantic decoy pack `downstream_benchmark-f3e0dc65fbf18825`.

Command shape:

```bash
uv run codelewm eval execution-surprise \
  --checkpoint .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-<seed>/checkpoints/last.pt \
  --pack .artifacts/v0_6/execution-pack \
  --semantic-decoy-manifest docs/benchmark/v0_6/semantic_decoy_pack/manifest.json \
  --decoys mutation,same_problem_different_submission,same_code_different_input \
  --out docs/benchmark/v0_6/seed-<seed>/execution_surprise_semantic \
  --max-examples 1000 \
  --json
```

## Results

| Seed | Examples | Overall AUC | Mutation AUC / pairs | Same-code-different-input AUC / pairs | Same-problem-different-submission AUC / pairs | Pack count gate | Claim gate |
| ---: | -------: | ----------: | -------------------: | ------------------------------------: | ---------------------------------------------: | --------------- | ---------- |
| 42 | 236 | 1.0000 | 1.0000 / 236 | 1.0000 / 352 | 1.0000 / 6 | PASS, 358 pairs / 68 problems | CLOSED, `same_problem_different_submission` is 6 < 30 |
| 1729 | 236 | 1.0000 | 1.0000 / 236 | 1.0000 / 352 | 1.0000 / 6 | PASS, 358 pairs / 68 problems | CLOSED, `same_problem_different_submission` is 6 < 30 |

The model separates the available decoys in this diagnostic: all score gates
pass at AUC 1.0 for both seeds. The public semantic claim is still closed
because the narrow same-problem/different-submission category remains only six
scored pairs. The stronger same-code/different-input category contributes 352
same-problem semantic pairs and is useful evidence of input sensitivity, but it
does not replace the missing different-submission coverage.

## Validation

```bash
uv run codelewm manifest verify --manifest docs/benchmark/v0_6/seed-42/execution_surprise_semantic/manifest.json --parent-manifest .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42/manifest.json --parent-manifest .artifacts/v0_6/execution-pack/artifact_manifest.json --parent-manifest docs/benchmark/v0_6/semantic_decoy_pack/manifest.json --json
uv run codelewm manifest verify --manifest docs/benchmark/v0_6/seed-1729/execution_surprise_semantic/manifest.json --parent-manifest .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729/manifest.json --parent-manifest .artifacts/v0_6/execution-pack/artifact_manifest.json --parent-manifest docs/benchmark/v0_6/semantic_decoy_pack/manifest.json --json
uv run codelewm secret-scan --include-suffix .json --include-suffix .md docs/benchmark/v0_6/seed-42/execution_surprise_semantic docs/benchmark/v0_6/seed-1729/execution_surprise_semantic --json
```

All three checks pass with zero secret findings.
