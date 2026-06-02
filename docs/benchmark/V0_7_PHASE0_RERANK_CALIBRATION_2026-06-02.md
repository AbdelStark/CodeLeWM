# v0.7 Phase 0 — Rerank Calibration Probe (RFC-0015 WS-A1)

- Date: 2026-06-02
- Scope: RFC-0015 Phase 0; issues #337 (WS-A), #340 (WS-D), #341 (tracker)
- Compute: none (CPU-only analysis over committed v0.6 artifacts; no model inference, no GPU)

## Question

Before committing to v0.7 data + training runs: **is execution correctness
decodable from the per-completion features the v0.6 rerank eval already
produced?** If a cheap calibrator over those features recovers a usable
pass/fail signal, recalibration (WS-A1) is worthwhile on its own; if not, v0.7
must inject a real correctness signal (WS-A3 outcome head / WS-A2 label-aware
retrieval) and an unsaturated benchmark (WS-D).

## Method

`codelewm/eval/rerank_calibrator.py` loads the committed
`completion_scores.jsonl` artifacts (schema
`codelewm.eval.completion_score.v1`), fits a numpy L2 logistic-regression
calibrator on the available features (`codelewm, no_action, lexical,
shuffled_action, random, codelewm−no_action, −llm_order_rank`) against the
`passed` label using **problem-grouped 5-fold cross-validation** (no problem
leaks across folds), and reports completion-level decodability (ROC-AUC) plus
rerank pass@1 for each ranking over all problems and the unsaturated subset.

Reproduce:

```
uv run python -m codelewm.eval.rerank_calibrator \
  "docs/benchmark/v0_6/seed-*/downstream_rerank_full/*/reports/completion_scores.jsonl"
```

Pool: both seeds (42, 1729) × both benchmarks (HumanEval, MBPP-Plus), full
runs — 3,144 completions over 524 problems (6 candidates/problem pooled across
seeds), overall pass rate 0.93.

## Results

### 1. Correctness is only weakly decodable from the existing features

| Signal | ROC-AUC |
|---|---|
| Calibrator (CV, all features) | **0.548** |
| `codelewm` (best single feature) | 0.578 |
| `codelewm − no_action` | 0.547 |
| `lexical` | 0.535 |
| `random` | 0.527 |
| `no_action` | 0.504 |
| `shuffled_action` | 0.485 |

AUC ≈ 0.5 is chance. The v0.6 scores carry only a **faint** correctness signal;
a calibrator over all of them lands at 0.548. Recalibrating the existing
features is **not** a path to a usable correctness ranker.

### 2. The benchmark is structurally saturated

Of 524 problems, only **13 are rerankable** (a genuine pass/fail mix); 480 are
all-pass and 31 are all-fail. The maximum pass@1 lift a *perfect* reranker
could achieve is therefore **≤ 2.48 absolute points** — below the 3.0-pt
downstream gate. **No reranker, however good, can clear the gate on these
benchmarks.**

### 3. Rerank pass@1 (where it can matter)

| Ranking | all problems | unsaturated only (n=13) |
|---|---|---|
| lexical | 0.939 | **0.923** |
| no_action | 0.935 | 0.769 |
| calibrator | 0.935 | 0.769 |
| llm_order | 0.933 | 0.692 |
| codelewm | 0.929 | **0.538** |

On the rerankable subset the learned `codelewm` energy is the **worst** ranker
(0.538) — consistent with the documented flat/negative downstream result.
A lexical baseline is the strongest. The calibrator only ties no-action. All
calibrator-vs-baseline lift 95% CIs straddle zero.

## Conclusions and v0.7 implications

1. **WS-A1 (recalibrating existing scores) is insufficient.** Correctness is
   barely decodable (AUC ~0.55) from the dumped features; the learned energy is
   anti-helpful where reranking matters. The signal must come from elsewhere.
2. **WS-D (unsaturated benchmark) is a hard prerequisite, independent of the
   model.** With only 13/524 rerankable problems and ≤2.48-pt headroom, the
   downstream gate is unreachable on the current splits even with a perfect
   ranker. v0.7 must build harder/unsaturated splits before any rerank claim is
   measurable.
3. **Prioritize WS-A2/WS-A3 over WS-A1.** The correctness signal should be
   injected via a label-aware retrieval prior built on the learned encoder
   (which already achieves R@1 +0.61, unlike the weak dumped scores) and/or an
   execution-outcome head trained on sandbox `passed` labels — not by
   recalibrating the existing energy.

This is the intended Phase-0 outcome: a cheap, GPU-free decision that redirects
v0.7 effort away from recalibration and toward an unsaturated benchmark plus a
genuine correctness signal. The public claim boundary is unchanged.

## Related

- RFC-0015 (`docs/rfcs/RFC-0015-v0-7-execution-substrate-improvements.md`)
- `docs/benchmark/V0_6_RERANK_FULL_2026-06-01.md`,
  `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md`
- Tool: `codelewm/eval/rerank_calibrator.py`;
  tests: `tests/eval/test_rerank_calibrator.py`
