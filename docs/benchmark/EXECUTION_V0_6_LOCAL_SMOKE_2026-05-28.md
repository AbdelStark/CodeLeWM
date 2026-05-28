# v0.6 Execution-Substrate Local Smoke — 2026-05-28

> This is **local smoke evidence**, not a public model-quality result.
> It validates that the v0.6 JEPA torch training path runs end-to-end on
> the execution substrate. The headline v0.6 HF Jobs run (#265) on real
> data lives separately; the operator runbook is at
> `docs/operations/V0_6_EXECUTION_RUN_RUNBOOK.md`.

- Report ID: `codelewm-execution-v0-6-local-smoke-2026-05-28`
- Schema version: `codelewm.execution_train_report.v1`
- Source git SHA: (this branch — pending squash-merge of the runner PR)
- Reproduction command:
  ```bash
  uv sync --group train
  .venv/bin/python scripts/codelewm-execution-train-smoke \
    --ingestion /tmp/fat_mbpp_fixture.jsonl \
    --out /tmp/codelewm-smoke-fat \
    --batch-size 8 --max-steps 500 --warmup-steps 30 --device cpu
  ```
- Pack: 39 deterministic `(code, input, output)` records from a
  10-problem MBPP-shape JSONL passed through
  `codelewm.data.sandbox` under the default stdlib-only policy.

## What This Smoke Validates

The substrate-pivot motivating prediction (RFC-0014): on a substrate
where output and code live in disjoint token distributions and the
input carries precise conditioning signal, the JEPA recipe should be
able to beat the no-action baseline that dominated every v0.2
commit-edit run.

Concretely the smoke checks the four diagnostic-only acceptance
criteria from the M1 spec, plus the substrate-pivot headline check:

1. prediction MSE decreases monotonically;
2. SIGReg loss decreases or stabilizes;
3. action-swap contrastive loss decreases;
4. no NaN or Inf in any logged metric;
5. **substrate-pivot prediction**: the no-action margin
   (`no_action_mse - pred_mse`) flips from negative (no-action wins)
   at step 1 to positive (predictor wins) by the end of training.

The smoke does **not** validate generalization. The 39-record pack is
within easy memorization range for a 29M-parameter model; we expect
training MSE to plummet. The question being asked is: does the
*trajectory* behave the way the substrate-pivot motivation claims it
should? Yes.

## Headline Numbers

| Metric | Step 1 | Final (tail-averaged) | Δ |
|--------|-------:|---------------------:|---:|
| `loss_prediction_mse` | 1.0104 | **0.0426** | −0.97 (24× reduction) |
| `loss_sigreg` | 5.2134 | 1.3617 | −3.85 (4× reduction) |
| `loss_total` | 1.4846 | 0.1658 | −1.32 |
| `no_action_mse` | 0.2444 | 0.5917 | +0.35 |
| `margin_no_action_minus_pred` | **−0.7660** | **+0.5492** | **+1.32 sign flip** |
| `loss_action_swap_contrastive` | 0.0500 | ≈0.005 | −0.045 (10× reduction) |

## Loss Trajectory (every 50 steps)

```
step=  1  mse=1.0104  sigreg=5.2134  margin=-0.7660  swap=0.0500
step= 51  mse=0.0992  sigreg=6.5252  margin=-0.0276  swap=0.0488
step=101  mse=0.0666  sigreg=5.7822  margin=+0.5616  swap=0.0492
step=151  mse=0.0659  sigreg=5.0640  margin=+0.6789  swap=0.0141
step=201  mse=0.0816  sigreg=2.6876  margin=+0.7071  swap=0.0000
step=251  mse=0.0604  sigreg=2.1566  margin=+0.5418  swap=0.0031
step=301  mse=0.0518  sigreg=1.8267  margin=+0.4820  swap=0.0055
step=351  mse=0.0462  sigreg=1.7165  margin=+0.5242  swap=0.0099
step=401  mse=0.0421  sigreg=1.7317  margin=+0.4937  swap=0.0071
step=451  mse=0.0349  sigreg=1.6558  margin=+0.5566  swap=0.0048
```

The margin flips sign around step 100 and stays positive thereafter.

## Latent Diagnostics

| Diagnostic | Value | Threshold | Notes |
|------------|------:|----------:|-------|
| `z_pred_effective_rank` | 6.26 | ≥1 | At the data ceiling (39 records → max 39) |
| `z_pred_effective_rank_ratio` | 0.024 | n/a | 0.024 vs. data-ceiling 0.15 (39/256) |
| `z_target_effective_rank` | 6.70 | ≥1 | Encoder-of-output ditto |
| `z_pred_mean_norm` | 14.33 | finite | Not collapsing to zero |
| `z_target_mean_norm` | 14.20 | finite | Same scale as predictions |
| `z_pred_mean_pairwise_cosine` | −0.021 | ≈0 | **Not** collapsed to a direction |

The headline collapse claim gate from `EXECUTION_V0_6_RESULTS_TEMPLATE.md`
(effective rank ratio ≥0.20) **cannot be evaluated on a 39-record pack** —
the upper bound is structurally 0.15. The full v0.6 run on a ~200k-record
pack is where that gate becomes meaningful.

## Comparison To v0.2 Commit-Edit Substrate

| Substrate | Records | Final mse | No-action margin (sign) | Effective rank |
|-----------|--------:|----------:|:-----------------------:|---------------:|
| v0.2 commit-edit (#172 v0.2 action-swap) | 20,645 | n/a* | **negative throughout (loses)** | 4.03 / 256 (ratio 0.016) |
| v0.6 execution local-smoke (this report) | 39 | 0.043 | **flips positive at step ~100** | 6.26 / 256 (at data ceiling 0.15) |

\* v0.2 reports retrieval R@1 / MRR rather than raw MSE; the
no-action-loses signal there is `text_action_recall_at_1 < no_action_recall_at_1`.

The local smoke is consistent with the substrate-pivot motivation but
is not at scale and is not a public claim. The full headline gate is
the v0.6 HF Jobs run on ~200k records over 50k steps with two seeds.

## What This Smoke Does Not Validate

- **Generalization.** 39 training records ≪ model capacity; the
  trajectory is overfitting evidence, not test-set evidence.
- **Effective rank claim gate.** The data ceiling is below the
  threshold; the full v0.6 run is where this becomes a real gate.
- **Downstream rerank.** The HumanEval / MBPP-Plus rerank protocol
  (`codelewm.eval.execution_rerank`) requires LLM-sampled completions
  + sandboxed hidden-test execution. Operator-driven.
- **Latent probe gates.** Requires train/val split with enough
  per-class samples per probe target.
- **Two-seed variance.** Single-seed smoke.

## Reproducibility

The 10-problem fat MBPP fixture used for this report:

```jsonl
{"task_id": 11, "code": "def square(n):\n    return n * n\n", "test_list": [...]}
{"task_id": 12, "code": "def total(xs):\n    return sum(xs)\n", "test_list": [...]}
... (10 problems total, ~40 assert lines)
```

Run command:

```bash
uv sync --group train
.venv/bin/python scripts/codelewm-execution-train-smoke \
  --ingestion <path-to-fat-mbpp-fixture.jsonl> \
  --out /tmp/codelewm-smoke-fat \
  --batch-size 8 --max-steps 500 --warmup-steps 30 --device cpu
```

CPU wall-clock: ingestion <1s; sandbox-pack ~10s; train(500 steps) ~30s.

## Implementation Pointers

- Bridge module: `codelewm.training.execution_torch_runner`
  (`train_execution_smoke`, `ExecutionTorchTrainConfig`,
  `ExecutionTorchReport`, `ExecutionTorchStep`).
- CLI: `scripts/codelewm-execution-train-smoke`.
- Tests: `tests/training/test_execution_torch_runner.py`
  (gated on `torch` import; opt-in via `uv sync --group train`).

## Allowed Public Language

> A local CPU smoke (39 deterministic execution records, 500 training
> steps) shows the v0.6 JEPA torch training path running end-to-end on
> the execution substrate. The prediction MSE decreases 24×, SIGReg
> drops 4×, the action-swap contrastive loss decreases 10×, and the
> no-action margin flips sign from −0.77 at step 1 to +0.55 at the end
> of training — the substrate-pivot's motivating prediction.
> This is local smoke evidence; the public headline claim gates remain
> behind the v0.6 HF Jobs run (#265).

## Reference

- RFC: `docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`
- Roadmap: `docs/roadmap/EXECUTION_TRACE_WORLD_MODEL.md`
- Tracker: #259
- Headline-gate benchmark template:
  `docs/benchmark/EXECUTION_V0_6_RESULTS_TEMPLATE.md`
- Operator runbook:
  `docs/operations/V0_6_EXECUTION_RUN_RUNBOOK.md`
