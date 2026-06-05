# CodeLeWM v0.8 Correctness-Aware Execution Results (2026-06-05)

v0.8 adds the RFC-0015 WS-A correctness heads to the v0.7 transformer execution
substrate:

- `p_pass` BCE head for pass/fail correctness.
- output-value auxiliary head.
- the v0.7 transformer state encoder, retrieval term, action-swap contrastive
  term, and SIGReg stability objective.

The result is mixed. Both A10G training jobs completed and produced
non-collapsed checkpoints. HumanEval WS-D reranking passes on both seeds, but
the overall v0.8 downstream claim remains closed because MBPP-Plus does not
clear the bootstrap confidence gate, lexical is near-perfect on MBPP-Plus, and
the pack cannot evaluate the requested magnitude probe on the val split.

## Job Status

| Seed | HF job | Status | Run | Runtime image |
| --- | --- | --- | --- | --- |
| 42 | `6a2278d2e6aa50b87b9eba56` | COMPLETED | `codelewm-v0-8-short-execution-20260605-1b737e4-seed-42` | `ghcr.io/abdelstark/codelewm-runtime:v0.8@sha256:1dd5449879961eb78513c9a5a8bdbabbd19fb9d96e733acb38f1b69af621b35e` |
| 1729 | `6a227a6ce52fdd2a02ed9005` | COMPLETED | `codelewm-v0-8-short-execution-20260605-1b737e4-seed-1729` | `ghcr.io/abdelstark/codelewm-runtime:v0.8@sha256:1dd5449879961eb78513c9a5a8bdbabbd19fb9d96e733acb38f1b69af621b35e` |

Both replacement jobs wrote structured `CODELEWM_JOB_EVENT` streams and
`reports/job_progress.jsonl` artifacts. Post-run observability work in #380,
#381, #382, and #383 made future runs easier to monitor by adding step-level
progress, a status summarizer, transient status retry flags, and runtime
lifecycle events around pack download, command execution, upload, and teardown.
The refreshed runtime image published after #383 is
`ghcr.io/abdelstark/codelewm-runtime:v0.8@sha256:c399cb43dc5afcb96964802f061ea7ea4cff56aa66a4b1901edfa35d98056567`
(linux/amd64 manifest
`sha256:2d9a68c0287c57388da0accbf40fa48f74fb746ce0ecc3b82a5724cf1d0cf255`).

## Reproducibility Chain

| Surface | Reference |
| --- | --- |
| Training pack | `abdelstark/codelewm-execution-pack@v0.8.0-rc1`; artifact `codelewm-passfail-execution-pack-20260604T185716Z`; 1,882 HumanEval pass/fail execution records |
| Training config | `config/train/scaled/codelewm_execution_v0_8_short_a10g.yaml`; 12,000 steps; A10G; bf16; seeds `42`, `1729` |
| Training run artifacts | `abdelstark/codelewm-runs/codelewm-v0-8-short-execution-20260605-1b737e4-seed-{42,1729}`; artifacts `training_run-e2a757caf75cbcf2`, `training_run-951983cbf59f6fa6` |
| Published checkpoint mirror | `abdelstark/codelewm-transition-model/checkpoints/codelewm-v0-8-short-execution-20260605-1b737e4-seed-{42,1729}` |
| Eval artifacts | `docs/benchmark/v0_8/seed-{42,1729}/` |
| Dataset card | `docs/cards/codelewm-v0-8-execution-dataset-2026-06-05.md` |
| Model cards | `docs/cards/codelewm-v0-8-execution-model-seed-42-2026-06-05.md`, `docs/cards/codelewm-v0-8-execution-model-seed-1729-2026-06-05.md` |

## Training Health

| Metric | Seed 42 | Seed 1729 |
| --- | ---: | ---: |
| Initial loss_total | 5.9541 | 5.8362 |
| Final loss_total | 0.1159 | 0.1118 |
| Final loss_p_pass_bce | 0.000000013 | 0.000000038 |
| Final loss_prediction_mse | 0.00769 | 0.00745 |
| Final margin, no_action minus pred | +1.3264 | +1.3871 |
| z_pred effective-rank ratio | 0.3843 | 0.3928 |
| z_pred mean pairwise cosine | 0.00255 | 0.00295 |

Both seeds pass the non-collapse health check (`z_pred_effective_rank_ratio`
greater than the 0.20 gate) and learn a positive transition margin. These are
training-health results only; they do not prove downstream correctness.

## Gate Summary

| Gate | Verdict | Evidence |
| --- | --- | --- |
| Jobs complete | PASS | both HF Jobs show `COMPLETED` |
| Artifact integrity | PASS | pack, runs, model-repo downloads, checkpoint inspections, and all 10 eval manifests verify with parents |
| Secret scans | PASS | pack, run artifacts, published checkpoint mirrors, and eval tree scan clean |
| Collapse | PASS | z_pred effective-rank ratio `0.3843` / `0.3928` |
| Retrieval lift over no_action | PASS | recall@1 `0.2754` / `0.2924`; no_action recall@1 `0.0085` / `0.0169` |
| Surprise mutation AUC | PASS | `1.0` / `1.0` |
| Surprise broad semantic gate | CLOSED | no scorable same-problem or same-code semantic decoy pairs for this v0.8 pack |
| Pass/fail latent probe | NOT MET | seed 42 `z_pred_after` test accuracy `0.508`; seed 1729 `0.589`; controls are comparable or stronger |
| Magnitude probe | NOT EVALUABLE | val split has zero `output_magnitude_bucket` labels |
| HumanEval WS-D rerank | PASS | pass@1 `0.9787` / `0.9787`; CI lift over no_action excludes zero |
| MBPP-Plus WS-D rerank | NOT MET | pass@1 `0.3529` / `0.3529`; bootstrap CI over llm_order spans zero; lexical pass@1 `1.0` |
| Overall v0.8 downstream claim | CLOSED | second benchmark and representation guards do not clear |

## Retrieval

| Seed | Recall@1 | Recall@5 | Recall@10 | MRR | No-action Recall@1 | Shuffled-action Recall@1 | Lexical Recall@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.2754 | 0.5593 | 0.6568 | 0.4125 | 0.0085 | 0.0297 | 0.0127 |
| 1729 | 0.2924 | 0.5678 | 0.6441 | 0.4213 | 0.0169 | 0.0254 | 0.0127 |

The retrieval guard holds against all computed baselines, but absolute recall is
lower than v0.7. v0.8 trades part of the self-supervised retrieval strength for
supervised correctness-head training.

## Surprise

| Seed | Pairwise AUC overall | Mutation AUC | Recall@1 | Scored mutation pairs | Claim gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 42 | 1.0000 | 1.0000 | 1.0000 | 236 | CLOSED |
| 1729 | 1.0000 | 1.0000 | 1.0000 | 236 | CLOSED |

The score separates true outputs from mutation decoys perfectly on the evaluated
slice. The broad semantic-surprise claim stays closed because
`same_problem_different_submission` and `same_code_different_input` have zero
scorable pairs after aligning the v0.6 semantic decoy pack with the v0.8
HumanEval pass/fail pack.

## Latent Probes

| Seed | Target | z_pred_after test accuracy | Best control | Verdict |
| --- | --- | ---: | ---: | --- |
| 42 | `passed` | 0.5081 | 0.6216 (`shuffled_action`) | NOT MET |
| 1729 | `passed` | 0.5892 | 0.6486 (`no_action`) | NOT MET |
| 42 | `output_magnitude_bucket` | not evaluable | not evaluable | no val labels |
| 1729 | `output_magnitude_bucket` | not evaluable | not evaluable | no val labels |

This localizes the result: the learned scorer can rank HumanEval WS-D
completions, but the frozen latent probe does not expose a robust pass/fail
representation under the current nearest-centroid probe, and the magnitude
target is blocked by split coverage.

## Downstream WS-D Rerank

### HumanEval

| Seed | codelewm pass@1 | no_action | shuffled_action | lexical | llm_order | Lift over no_action | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.9787 | 0.8723 | 0.8723 | 0.6596 | 0.1489 | +10.64 pt | [2.13, 19.15] |
| 1729 | 0.9787 | 0.8723 | 0.8936 | 0.6596 | 0.1489 | +10.64 pt | [2.13, 19.15] |

HumanEval clears the local rerank gate on both seeds.

### MBPP-Plus

| Seed | codelewm pass@1 | no_action | shuffled_action | lexical | llm_order | Lift over no_action | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.3529 | 0.1176 | 0.1765 | 1.0000 | 0.1765 | +23.53 pt | [5.88, 47.06] |
| 1729 | 0.3529 | 0.2941 | 0.3529 | 1.0000 | 0.1765 | +5.88 pt | [-17.65, 29.41] |

MBPP-Plus does not clear the claim gate. The seed-42 lift over no_action is
positive, but the benchmark-level claim requires both seeds and clearance over
the configured controls. More importantly, lexical pass@1 is already `1.0` on
this 17-problem pack, so this slice does not support a CodeLeWM-over-baseline
coding-usefulness claim.

## Completion-Level ROC-AUC Diagnostic

The eval reports do not emit held-out training-pack `p_pass` ROC-AUC, so this
table is computed from the persisted WS-D `completion_scores.jsonl` rows. It is
a downstream completion-level diagnostic, not a replacement for a training-pack
validation AUC.

| Seed | Benchmark | codelewm AUC | no_action AUC | lexical AUC |
| --- | --- | ---: | ---: | ---: |
| 42 | HumanEval | 0.9622 | 0.9632 | 0.5280 |
| 1729 | HumanEval | 0.9699 | 0.9191 | 0.5280 |
| 42 | MBPP-Plus | 0.5772 | 0.5363 | 0.9824 |
| 1729 | MBPP-Plus | 0.6941 | 0.6851 | 0.9824 |

HumanEval shows a strong correctness scorer. MBPP-Plus is weak to marginal and
lexical dominates, which is consistent with the pass@1 gate.

## Verification Commands

Training and model artifacts were downloaded, manifest-verified with the v0.8
pack parent, checkpoint-inspected, and secret-scanned. The eval artifacts copied
under `docs/benchmark/v0_8/` were verified with the relevant training, pack,
semantic-decoy, and completion-label parents.

Representative commands:

```bash
hf jobs inspect 6a2278d2e6aa50b87b9eba56
hf jobs inspect 6a227a6ce52fdd2a02ed9005

uv run scripts/hf-job-event-status \
  --from-file .artifacts/hf-download/v0_8_20260605_seed_42/codelewm-v0-8-short-execution-20260605-1b737e4-seed-42/reports/job_progress.jsonl

uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_8/seed-42/execution_retrieval/manifest.json \
  --parent-manifest .artifacts/hf-download/v0_8_20260605_seed_42/codelewm-v0-8-short-execution-20260605-1b737e4-seed-42/manifest.json \
  --parent-manifest /tmp/codelewm-v0-8-pack/artifact_manifest.json \
  --json

uv run codelewm secret-scan docs/benchmark/v0_8 --json
```

## Claim Boundary

Safe claim:

> CodeLeWM v0.8 completed two A10G correctness-aware execution training runs,
> kept non-collapse/retrieval/surprise diagnostics healthy, and achieved a
> strong HumanEval WS-D rerank result on both seeds.

Do not claim:

- CodeLeWM v0.8 generally improves Python coding benchmarks.
- The v0.8 pass/fail latent is robustly decodable from frozen latents.
- The output-magnitude probe passed.
- The MBPP-Plus downstream gate passed.
- The broad semantic-decoy surprise gate passed.

The overall v0.8 claim remains diagnostic and benchmark-specific until a
follow-up benchmark or pack fixes the magnitude-label split coverage and shows
cross-benchmark rerank lift with confidence clearance.
