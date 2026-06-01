# Two-Substrate Paper Claim Audit

This audit binds the public claims in `two_substrate_paper.tex` to checked-in
reports or schema-versioned artifacts. It is the claim-gate review surface for
#306.

## Quantitative Claims

| Claim in draft | Value | Source |
| --- | ---: | --- |
| Substrate A #138 headline Recall@1 / MRR | 0.371 / 0.472984 | `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md` (`Retrieval`) |
| Substrate A #138 no-action Recall@1 / MRR | 0.459 / 0.546116 | `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md` (`Retrieval`) |
| Substrate A #154 headline Recall@1 / MRR | 0.363 / 0.467875 | `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md` (`Retrieval`) |
| Substrate A #154 no-action Recall@1 / MRR | 0.469 / 0.549624 | `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md` (`Retrieval`) |
| Substrate A #159 headline Recall@1 / MRR | 0.597 / 0.674500 | `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md` (`Retrieval`) |
| Substrate A #159 no-action Recall@1 / MRR | 0.650 / 0.708037 | `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md` (`Retrieval`) |
| Substrate A v0.2 effective-rank ratio | 0.015761 | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` (`Training`) |
| Substrate A v0.2 headline Recall@1 / MRR | 0.263 / 0.370048 | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` (`Headline Retrieval`) |
| Substrate A v0.2 no-action Recall@1 / MRR | 0.441 / 0.533105 | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` (`Headline Retrieval`) |
| Substrate A v0.2 mutation surprise AUC | 0.501 | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` (`Surprise`) |
| Substrate B v0.6 final prediction MSE | seed 42: 0.00064; seed 1729: 0.00060 | `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` (`Loss trajectory`) |
| Substrate B v0.6 final SIGReg | seed 42: 0.0364; seed 1729: 0.0348 | `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` (`Loss trajectory`) |
| Substrate B v0.6 final no-action margin | seed 42: +1.2308; seed 1729: +1.2434 | `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` (`Loss trajectory`) |
| Substrate B v0.6 effective-rank ratio | seed 42: 0.4669; seed 1729: 0.4768 | `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` (`Collapse And SIGReg Diagnostics`) |
| Substrate B v0.6 retrieval Recall@1 | seed 42: 0.6568; seed 1729: 0.6483 | `docs/benchmark/v0_6/seed-*/execution_retrieval/reports/retrieval_report.json` |
| Substrate B v0.6 retrieval MRR | seed 42: 0.7670; seed 1729: 0.7587 | `docs/benchmark/v0_6/seed-*/execution_retrieval/reports/retrieval_report.json` |
| Substrate B v0.6 retrieval lift over no-action | Recall@1 mean +0.6144; MRR mean +0.6588 | `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` (`Retrieval Evaluation`) |
| Substrate B v0.6 generated-decoy surprise AUC | 1.000 on mutation, same-code-different-input, same-problem-different-submission | `docs/benchmark/v0_6/seed-*/execution_surprise/reports/surprise_report.json` |
| Substrate B v0.6 same-problem decoy pair count | 6 pairs per seed | `docs/benchmark/v0_6/seed-*/execution_surprise/reports/execution_decoy_report.json` |
| Substrate B v0.6 latent probe target availability | only `output_type` evaluable | `docs/benchmark/v0_6/seed-*/execution_probe/reports/latent_probe_report.json` |
| Substrate B v0.6 output-type latent probe | seed 42: 0.4968; seed 1729: 0.5987 | `docs/benchmark/v0_6/seed-*/execution_probe/reports/latent_probe_report.json` |
| Substrate B v0.6 output-type lexical control | seed 42: 0.6624; seed 1729: 0.6178 | `docs/benchmark/v0_6/seed-*/execution_probe/reports/latent_probe_report.json` |
| Crash-prediction eval support | positives=0, negatives=236 on both seeds | `docs/benchmark/v0_6/seed-*/crash_prediction/reports/crash_prediction_report.json` |
| HumanEval / MBPP-Plus rerank status | not run; no live completion-label artifacts | `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` (`Downstream Rerank Gate`) |

## Artifact Claims

| Claim | Evidence |
| --- | --- |
| v0.6 execution pack id | `codelewm-execution-pack-20260528T102625Z`, `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` |
| v0.6 seed-42 training artifact id | `training_run-cb62408f881eff8c`, `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` and committed eval manifests |
| v0.6 seed-1729 training artifact id | `training_run-d0b59108447c9c4a`, `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` and committed eval manifests |
| #305 HF mirror | `abdelstark/codelewm-runs/runs/codelewm-v0-6-eval-pass-20260531`, HF commit `396a8fab5b86c16764bec0090e8af7518de41fbc` |
| BibTeX provenance | `docs/papers/two_substrate_references.bib` entries were fetched from `https://arxiv.org/bibtex/<arxiv-id>` during #306 drafting. |

## Comparability Boundary

The paper may say that the two substrate lines share the CodeLeWM
latent-transition model family and objective registry. It must not say the
comparison is a perfect single-variable ablation. Known run-level differences:

- v0.2 action-swap uses `action_fusion: gated_residual`,
  `max_steps=60000`, `lr=0.0001`, `weight_decay=0.001`,
  `action_swap_contrastive_weight=0.20`, and
  `inverse_action_reconstruction_weight=0.10`
  (`config/train/scaled/codelewm_scaled_v0_2_action_swap_inverse_gpu_a10g.yaml`).
- v0.6 execution uses the execution runner with conditional-transformer fusion,
  `max_steps=50000`, `lr=0.0003`, `weight_decay=0.1`,
  `action_swap_contrastive_weight=0.10`, and
  `inverse_action_reconstruction_weight=0.05`
  (`config/train/scaled/codelewm_execution_v0_6_a10g.yaml`,
  `codelewm/training/execution_runner.py`).
- The allowed claim is therefore a substrate-pivot comparison within one
  implementation and artifact discipline, not proof that substrate is the only
  changed variable.

## Claim Boundary

- Allowed: substrate-pivot comparison; negative v0.2 claim; partial-positive
  v0.6 substrate-shape, retrieval, and generated-decoy surprise claim.
- Not allowed: broad code-generation quality, semantic-axis naming, crash
  prediction utility, HumanEval / MBPP-Plus reranking utility, or superiority
  over pretrained code models.
