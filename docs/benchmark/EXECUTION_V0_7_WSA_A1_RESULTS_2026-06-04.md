# CodeLeWM WS-A / A1 Decodability Probe (2026-06-04)

WS-A asks: can the model's scores be made correctness-aware so it reranks?
A1 is the cheapest gate — fit a logistic calibrator on the per-completion
features already written by the rerank eval and ask whether correctness is
decodable *without* GPU. A1 must be answered before committing to A3 (a
trained `p_pass` head + a pass/fail data build + GPU).

The decisive A1 question (sharpened by WS-D, where a lexical baseline alone
hit pass@1 0.30): **do the model features add correctness signal beyond the
lexical control?** Answering it required a `--features` selector on the
calibrator (model-only vs lexical-only vs all).

## Result (pooled over both WS-D seeds: 47 problems, 564 completions)

Univariate ROC-AUC for predicting `passed`:

| feature | AUC |
|---------|-----|
| **codelewm (model energy)** | **0.500** |
| no_action | 0.502 |
| shuffled_action | 0.500 |
| codelewm − no_action | 0.495 |
| neg_llm_order_rank | 0.421 |
| lexical | 0.528 |
| random | 0.521 |

Grouped 5-fold calibrator CV-AUC by feature set:

| feature set | calibrator CV-AUC |
|-------------|-------------------|
| **model** (codelewm, no_action, shuffled_action, codelewm−no_action, neg_llm_order_rank) | **0.572** |
| **lexical** (lexical, random) | 0.477 |
| all | 0.558 |

Rerank pass@1 (all problems): calibrator 0.255, lexical 0.298, llm_order
0.149, no_action 0.149, codelewm 0.085.

## Verdict — A1 gate FAILS for the model; escalate to A3

- The model's energy features are at **exactly chance** (codelewm AUC 0.500,
  no_action 0.502, shuffled 0.500). They carry **no** correctness signal.
- The model-only calibrator (0.572) is marginal, below the ~0.62 ship bar,
  and is not driven by the energy (which is 0.500) — it is noise / weak
  exploitation of the rank feature.
- A calibrator over current features cannot beat the trivial lexical
  baseline in a way that reflects *model* understanding. **A1 and A2 (cheap,
  no-retrain fixes) are dead ends.**

The only remaining lever is **A3** — an execution-outcome head
`p_pass = σ(W·[z_code, action, z_pred_after])` trained with BCE on sandbox
`passed` labels — with one risk now made explicit: the *scalar* model
energies are at chance, so A3 will only succeed if the **full latent
vectors** encode correctness that the scalar energy collapses away. The
recommended A3 sequencing therefore trains the head on the **frozen** SSL
trunk first (head-only): if held-out `p_pass` AUC clears ~0.65 frozen, the
signal is in the latents; if it stays at chance, the SSL representation
itself lacks correctness information and WS-C (architecture) — not more A3
tuning — is implicated.

## Artifacts
- `docs/benchmark/wsa/a1/pooled_{model,lexical,all}.json` — full calibrator
  reports (decodability, per-feature AUC, rerank pass@1, bootstrap lifts).
- Reproduce: `python -m codelewm.eval.rerank_calibrator
  'results/wsd/seed-*/humaneval/reports/completion_scores.jsonl' --features model`.

## Next: A3 data dependency (the real cost)
The execution pack has no correctness labels by construction
(`execution_status` is crash-vs-ok, not pass-vs-fail), and the
completion-label artifacts drop the raw output (only a checksum is kept). A3
training data therefore needs a new `completion_label.v1 → pack` adapter that
re-executes the WS-D mutation completions to recover outputs, tokenizes,
assigns splits, and writes a `passed` label — then the head + BCE objective +
scorer fusion (all change points mapped) and a frozen-trunk second-stage GPU
run, evaluated on the WS-D benchmark.
