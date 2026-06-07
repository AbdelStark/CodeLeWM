# CodeLeWM v0.9 Cross-Benchmark Execution Results (2026-06-07)

v0.9 repairs the v0.8 data/eval blockers before making any new model-quality
claim. The pass/fail execution pack is now cross-benchmark, includes HumanEval
and MBPP-Plus, carries held-out split coverage for `passed` and
`output_magnitude_bucket`, and the two guarded HF Jobs seeds completed with
structured job observability.

The result is still claim-closed for general downstream coding usefulness.
HumanEval WS-D reranking passes on both seeds, but MBPP-Plus is saturated:
CodeLeWM, no-action, and lexical all reach pass@1 `1.0000`, so the required
lift over controls is exactly zero. Broad semantic-decoy surprise is still
coverage-blocked against the available semantic-decoy pack, and representation
probe claims remain unsupported.

## Job Status

| Seed | HF job | Status | Run | Runtime image |
| --- | --- | --- | --- | --- |
| 42 | `6a241a07ece949d7b3dca989` | COMPLETED | `codelewm-v0-9-short-execution-20260606-69f798a-seed-42` | `ghcr.io/abdelstark/codelewm-runtime:v0.9@sha256:32fb888db51c0e31445ecabb9be4bb20993c0d13d9265866de265d0c1c4f36ab` |
| 1729 | `6a241a07368e0b5dc8064efb` | COMPLETED | `codelewm-v0-9-short-execution-20260606-69f798a-seed-1729` | `ghcr.io/abdelstark/codelewm-runtime:v0.9@sha256:32fb888db51c0e31445ecabb9be4bb20993c0d13d9265866de265d0c1c4f36ab` |

Both jobs wrote `CODELEWM_JOB_EVENT` streams, `reports/job_progress.jsonl`,
TensorBoard-compatible metadata, final 12,000-step checkpoints, and upload
completion events. `scripts/hf-job-event-status --from-file` reported
`complete=true` and `collapse_ok=true` for both downloaded progress logs.

## Reproducibility Chain

| Surface | Reference |
| --- | --- |
| Source SHA | `69f798a620fdf8e50c1773228428e80ffee1a6ef` |
| Training pack | `abdelstark/codelewm-execution-pack@v0.9.0-rc1`; artifact `codelewm-passfail-execution-pack-20260606T122240Z`; 2,188 records |
| Training config | `config/train/scaled/codelewm_execution_v0_9_short_a10g.yaml`; 12,000 steps; A10G; bf16; seeds `42`, `1729` |
| Training run artifacts | `abdelstark/codelewm-runs/codelewm-v0-9-short-execution-20260606-69f798a-seed-{42,1729}`; artifacts `training_run-992f7757f2780da4`, `training_run-91e9cf7c645379b3` |
| Eval artifacts | `docs/benchmark/v0_9/seed-{42,1729}/` |
| Dataset card | `docs/cards/codelewm-v0-9-execution-dataset-2026-06-07.md` |
| Model cards | `docs/cards/codelewm-v0-9-execution-model-seed-42-2026-06-07.md`, `docs/cards/codelewm-v0-9-execution-model-seed-1729-2026-06-07.md` |
| Public artifact index | `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-07.md` |

## Pack Repair

| Field | Value |
| --- | ---: |
| Records | 2,188 |
| HumanEval records | 1,882 |
| MBPP-Plus records | 306 |
| Train / val / test | 1,928 / 57 / 203 |
| `passed=true` / `passed=false` | 1,081 / 1,107 |
| `pos_weight` | 1.0240518038852915 |
| Sandbox timeout rejects | 26 |

The held-out split coverage gate passes. `output_magnitude_bucket` has val and
test labels in v0.9, fixing the v0.8 not-evaluable magnitude-probe blocker.

## Training Health

| Metric | Seed 42 | Seed 1729 |
| --- | ---: | ---: |
| Final loss_total | 0.113995 | 0.115219 |
| Final loss_p_pass_bce | 1.21375e-07 | 0.001716 |
| Final loss_prediction_mse | 0.009726 | 0.010695 |
| Final margin, no_action minus pred | +1.318410 | +1.277334 |
| Examples/sec | 65.3250 | 64.9533 |
| Steps/sec | 0.255176 | 0.253724 |
| z_pred effective-rank ratio | 0.388272 | 0.405622 |
| z_target effective-rank ratio | 0.398086 | 0.415045 |
| Checkpoint SHA-256 | `c783fa0dbe5da6bd072ff0b2f2753bdbac9fe684b49bf82e70ab6a2f69d513da` | `34ebb282b284580dd123c781ae77c93cc36bbffc4eeeee9f0bd4cdf8042001eb` |

Both seeds pass the non-collapse health check (`z_pred_effective_rank_ratio`
greater than the 0.20 gate). These are training-health results only.

## Gate Summary

| Gate | Verdict | Evidence |
| --- | --- | --- |
| Jobs complete | PASS | both HF Jobs show `COMPLETED` and final upload events |
| Artifact integrity | PASS | pack, runs, checkpoint inspections, and all 12 tracked eval manifests verify with parents |
| Secret scans | PASS | pack, run artifacts, checkpoint inspections, and `docs/benchmark/v0_9` scan clean |
| Collapse | PASS | z_pred effective-rank ratios `0.3883` / `0.4056` |
| Retrieval lift over controls | PASS | Recall@1 `0.2538` / `0.2731`, all baselines below `0.024` |
| Mutation surprise AUC | PASS | `1.0` / `1.0` |
| Broad semantic surprise | CLOSED | existing semantic-decoy pack has zero scorable same-problem/same-code pairs against the v0.9 pack |
| Probe-label coverage | PASS | `passed` and `output_magnitude_bucket` are both evaluable |
| Representation claim | CLOSED | probe results do not consistently beat controls across targets and seeds |
| Downstream HumanEval WS-D | PASS | both seeds pass lift-over-no-action CI gate |
| Downstream MBPP-Plus WS-D | CLOSED | CodeLeWM pass@1 `1.0000`, but no-action and lexical also `1.0000`; zero lift over no-action |
| Overall v0.9 downstream claim | CLOSED | cross-benchmark downstream lift and representation gates do not jointly pass |

## Retrieval

| Seed | Recall@1 | Recall@5 | Recall@10 | MRR | No-action Recall@1 | Shuffled-action Recall@1 | Lexical Recall@1 | Random Recall@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.2538 | 0.5115 | 0.6038 | 0.3804 | 0.0115 | 0.0192 | 0.0115 | 0.0000 |
| 1729 | 0.2731 | 0.5346 | 0.5923 | 0.3982 | 0.0192 | 0.0231 | 0.0115 | 0.0000 |

## Surprise

| Seed | Pairwise AUC overall | Mutation AUC | Recall@1 | Scored mutation pairs | Broad semantic gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 42 | 1.0000 | 1.0000 | 1.0000 | 260 | CLOSED: `same_problem_different_submission:0<30`; `same_code_different_input:0<100` |
| 1729 | 1.0000 | 1.0000 | 1.0000 | 260 | CLOSED: `same_problem_different_submission:0<30`; `same_code_different_input:0<100` |

The semantic-decoy pack itself has 358 pairs, but none align to scorable records
in the v0.9 cross-benchmark pass/fail pack. This is a typed coverage blocker,
not a positive semantic-surprise result.

## Latent Probes

| Seed | Target | z_pred_after test accuracy | Best control | Verdict |
| --- | --- | ---: | ---: | --- |
| 42 | `passed` | 0.6552 | 0.6897 (`no_action` / `z_before`) | CLOSED |
| 42 | `output_magnitude_bucket` | 0.5000 | 0.5000 (`z_after`) | CLOSED |
| 1729 | `passed` | 0.6158 | 0.7980 (`no_action` / `z_before`) | CLOSED |
| 1729 | `output_magnitude_bucket` | 0.8421 | 0.8421 (`z_after`) | CLOSED |

The v0.9 pack fixes label coverage, but it does not support a positive
representation claim. The strongest magnitude result is tied with a direct
`z_after` control, and the pass/fail target is weaker than no-action / before
state controls.

## Downstream WS-D Rerank

### HumanEval

| Seed | codelewm pass@1 | no_action | shuffled_action | lexical | llm_order | random | Lift over no_action | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.9787 | 0.8723 | 0.8936 | 0.6596 | 0.1489 | 0.2128 | +10.64 pt | [2.13, 21.28] |
| 1729 | 0.9787 | 0.8936 | 0.9149 | 0.6596 | 0.1489 | 0.2128 | +8.51 pt | [2.13, 17.02] |

HumanEval clears the local rerank gate on both seeds.

### MBPP-Plus

| Seed | codelewm pass@1 | no_action | shuffled_action | lexical | llm_order | random | Lift over no_action | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1765 | 0.2353 | +0.00 pt | [0.00, 0.00] |
| 1729 | 1.0000 | 1.0000 | 0.9412 | 1.0000 | 0.1765 | 0.2353 | +0.00 pt | [0.00, 0.00] |

MBPP-Plus is saturated under the current WS-D pack. This is better than the
v0.8 MBPP-Plus result for CodeLeWM, but it does not open the claim gate because
no-action and lexical controls are already perfect.

## Downstream Calibration

`codelewm eval p-pass-calibration` was run over the regenerated WS-D
`completion_scores.jsonl` rows for each seed. The rows now include
`benchmark_id` and `split=test`, so the calibration reports include real
HumanEval and MBPP-Plus slices.

The persisted downstream score rows do not expose a separate `p_pass` score key;
the report records `p_pass` as `status=missing` and selects the learned
`codelewm` score as the primary evaluable score.

| Seed | codelewm ROC-AUC | HumanEval ROC-AUC | MBPP-Plus ROC-AUC | No-action ROC-AUC | p_pass key | Artifact |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 42 | 0.9352 | 0.9196 | 0.9730 | 0.9482 | missing | `eval_report-203075dc79ada0d3` |
| 1729 | 0.9746 | 0.9717 | 0.9945 | 0.9731 | missing | `eval_report-9c92eb76a900e4be` |

These calibration reports are diagnostic. They do not override the downstream
claim gate.

## Eval Artifact IDs

| Seed | Retrieval | Surprise | Probe | HumanEval rerank | MBPP-Plus rerank | Calibration |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | `eval_report-a8c3610d40df7512` | `eval_report-3f298b8a90ac9cb6` | `eval_report-04b816b7a31f36d5` | `eval_report-0bc9a04d4a6bfa86` | `eval_report-7e9fa967ee6356af` | `eval_report-203075dc79ada0d3` |
| 1729 | `eval_report-0e49571dcb2fc373` | `eval_report-c59139ea35734802` | `eval_report-379017500d91ecf5` | `eval_report-3cd1cfeeb2fe2c09` | `eval_report-570bdbfeac5928ef` | `eval_report-9c92eb76a900e4be` |

## Verification Commands

Representative commands:

```bash
hf download abdelstark/codelewm-execution-pack \
  --repo-type dataset \
  --revision v0.9.0-rc1 \
  --local-dir .artifacts/v0_9/hf-pack-download

hf download abdelstark/codelewm-runs \
  --repo-type dataset \
  --include 'codelewm-v0-9-short-execution-20260606-69f798a-seed-42/**' \
  --include 'codelewm-v0-9-short-execution-20260606-69f798a-seed-1729/**' \
  --local-dir .artifacts/v0_9/hf-runs-download

uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_9/seed-42/execution_retrieval/manifest.json \
  --root docs/benchmark/v0_9/seed-42/execution_retrieval \
  --parent-manifest .artifacts/v0_9/hf-runs-download/codelewm-v0-9-short-execution-20260606-69f798a-seed-42/manifest.json \
  --parent-manifest .artifacts/v0_9/hf-pack-download/artifact_manifest.json \
  --json

uv run codelewm secret-scan docs/benchmark/v0_9 --json
```

Expected result: all manifest verifications return `ok=true`; secret scans
return `ok=true` with zero findings.

## Claim Boundary

Safe claim:

> CodeLeWM v0.9 completed two guarded A10G cross-benchmark correctness-aware
> execution training runs, fixed the v0.8 pass/fail and magnitude-label coverage
> blockers, kept non-collapse/retrieval/mutation-surprise diagnostics healthy,
> and passed HumanEval WS-D reranking on both seeds.

Do not claim:

- CodeLeWM v0.9 generally improves Python coding benchmarks.
- MBPP-Plus shows a CodeLeWM lift over no-action or lexical controls.
- The broad semantic-decoy surprise gate passed.
- The frozen latent representation robustly encodes `passed` or named output
  magnitude semantics.
- The downstream completion score rows expose a standalone `p_pass` score key.

The overall v0.9 downstream claim remains closed until a future artifact shows
cross-benchmark lift over controls and clears the representation and coverage
guards with the required seeds.
