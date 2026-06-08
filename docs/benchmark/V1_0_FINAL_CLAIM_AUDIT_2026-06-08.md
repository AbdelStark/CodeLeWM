# CodeLeWM v1.0 Final Claim Audit (2026-06-08)

Issue: #406.

This audit consolidates the benchmark and demo evidence that should ground the
final v1.0 paper and release package. It is a paper-ready Markdown source for
the public claim boundary across v0.2, v0.6, v0.8, v0.9, and the final v1.0
paper demo.

## Verdict

Claim verdict: CLOSED for broad CodeLeWM coding-improvement claims.

The final release may claim a reproducible code-edit world-model research
harness, manifest-backed artifact publication, non-collapsed learned execution
checkpoints for diagnostic use, and one narrow HumanEval WS-D downstream
positive slice in the v0.9/v1.0 replay evidence. It must also say that the
aggregate downstream claim remains closed because MBPP-Plus is saturated against
no-action and lexical controls, v0.2 action-use retrieval loses to no-action,
semantic representation claims remain unsupported, and the final downstream
score rows do not expose a standalone `p_pass` score key.

## Source Artifacts

| Evidence stream | Source path | Role in this audit |
| --- | --- | --- |
| v0.2 action-swap/inverse-action run | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` | Negative action-use, latent-probe, surprise, and one-example scorer evidence. |
| v0.6 downstream rerank | `docs/benchmark/V0_6_RERANK_FULL_2026-06-01.md` | Historical downstream rerank table showing no stable CodeLeWM lift. |
| v0.8 execution-trace result | `docs/benchmark/EXECUTION_V0_8_RESULTS_2026-06-05.md` | First HumanEval WS-D positive slice plus MBPP/generalization and probe blockers. |
| v0.9 final result | `docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md` | Final two-seed gate suite and claim audit before the paper demo. |
| v1.0 paper demo artifact note | `docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md` | Committed replay artifact summary for the final demo package. |
| v1.0 paper demo machine report | `docs/benchmark/v1_0/paper_demo/reports/paper_demo_report.json` | Machine-readable source for the final downstream table below. |
| v1.0 paper demo claim gate | `docs/benchmark/v1_0/paper_demo/reports/paper_demo_claim_gate.json` | Aggregate demo claim gate, `claim_allowed=false`. |
| v0.9 p-pass calibration, seed 42 | `docs/benchmark/v0_9/seed-42/p_pass_calibration/downstream_completion/reports/p_pass_calibration_report.json` | Completion-level calibration diagnostic; `p_pass` score key missing. |
| v0.9 p-pass calibration, seed 1729 | `docs/benchmark/v0_9/seed-1729/p_pass_calibration/downstream_completion/reports/p_pass_calibration_report.json` | Completion-level calibration diagnostic; `p_pass` score key missing. |

## Milestone Evidence Summary

| Milestone | Coverage | Positive evidence | Closed or not-evaluable gates | Allowed public wording |
| --- | --- | --- | --- | --- |
| v0.2 action-swap/inverse-action | One A10G run; 1,000 retrieval examples; local latent, surprise, and scorer reruns. | Systems pipeline, manifests, checkpoint trust, secret scans, surprise random/action-cluster decoy separation. | Text-action Recall@1 `0.263` and MRR `0.370048` lose to no-action Recall@1 `0.441` and MRR `0.533105`; action-contrast pools fail; latent semantic axes unsupported; downstream scorer has one labeled example. | Negative action-use intervention and diagnostic artifact evidence only. |
| v0.6 execution-rerank | Two seeds over HumanEval and MBPP-Plus generated completions. | Non-collapsed execution substrate and useful diagnostic rerank machinery. | Downstream pass@1 lift is unstable or negative across seeds; no HumanEval/MBPP claim opens. | Execution-substrate diagnostic result, not coding-usefulness evidence. |
| v0.8 execution-trace repair | Two A10G seeds; HumanEval and MBPP-Plus WS-D packs. | HumanEval WS-D clears lift-over-no-action on both seeds; retrieval and mutation-surprise diagnostics pass. | MBPP-Plus does not clear controls; lexical is `1.0000` on MBPP-Plus; pass/fail probe does not beat controls; magnitude probe not evaluable. | Narrow HumanEval diagnostic slice plus mixed/negative aggregate conclusion. |
| v0.9 data/eval repair | Two guarded A10G seeds; regenerated retrieval, surprise, probe, p-pass calibration, and downstream rerank reports. | HumanEval WS-D clears on both seeds; retrieval lift over controls passes; mutation surprise AUC is `1.0000` on both seeds; probe label coverage is repaired. | MBPP-Plus has zero lift over no-action; broad semantic surprise has zero scorable semantic pairs; representation probes do not beat controls; standalone `p_pass` key is missing. | Narrow HumanEval WS-D positive slice and final claim-closed aggregate result. |
| v1.0 paper demo | Clean-checkout replay over checked-in v0.9 WS-D score rows; 4 slices, 128 problems, 768 completions. | Reproducible one-command paper-demo package, manifest-backed artifact lineage, and HumanEval WS-D replay rows with open slice gates. | Aggregate `claim_allowed=false` because both MBPP-Plus rows have CodeLeWM `1.0000`, no-action `1.0000`, lexical `1.0000`, and zero lift. | Paper demo for the mixed final conclusion, not a fresh checkpoint run or broad coding-improvement claim. |

## Downstream Rerank Tables

### Historical Downstream Evidence

| Version | Seed | Benchmark | CodeLeWM pass@1 | No-action pass@1 | LLM-order pass@1 | Lexical pass@1 | Lift vs no-action | 95% CI | Slice gate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0.6 | 42 | HumanEval | 0.9610 | 0.9481 | 0.9545 | 0.9545 | +1.30 pts | [0.00, 3.25] | closed |
| v0.6 | 42 | MBPP-Plus | 0.9162 | 0.9243 | 0.9243 | 0.9297 | -0.81 pts | [-1.89, 0.00] | closed |
| v0.6 | 1729 | HumanEval | 0.9481 | 0.9610 | 0.9545 | 0.9545 | -1.30 pts | [-3.25, 0.00] | closed |
| v0.6 | 1729 | MBPP-Plus | 0.9243 | 0.9243 | 0.9243 | 0.9297 | +0.00 pts | [-0.81, 0.81] | closed |
| v0.8 | 42 | HumanEval WS-D | 0.9787 | 0.8723 | 0.1489 | 0.6596 | +10.64 pts | [2.13, 19.15] | open |
| v0.8 | 42 | MBPP-Plus WS-D | 0.3529 | 0.1176 | 0.1765 | 1.0000 | +23.53 pts | [5.88, 47.06] | closed by controls |
| v0.8 | 1729 | HumanEval WS-D | 0.9787 | 0.8723 | 0.1489 | 0.6596 | +10.64 pts | [2.13, 19.15] | open |
| v0.8 | 1729 | MBPP-Plus WS-D | 0.3529 | 0.2941 | 0.1765 | 1.0000 | +5.88 pts | [-17.65, 29.41] | closed |
| v0.9 | 42 | HumanEval WS-D | 0.9787 | 0.8723 | 0.1489 | 0.6596 | +10.64 pts | [2.13, 21.28] | open |
| v0.9 | 42 | MBPP-Plus WS-D | 1.0000 | 1.0000 | 0.1765 | 1.0000 | +0.00 pts | [0.00, 0.00] | closed |
| v0.9 | 1729 | HumanEval WS-D | 0.9787 | 0.8936 | 0.1489 | 0.6596 | +8.51 pts | [2.13, 17.02] | open |
| v0.9 | 1729 | MBPP-Plus WS-D | 1.0000 | 1.0000 | 0.1765 | 1.0000 | +0.00 pts | [0.00, 0.00] | closed |

### Final Paper-Demo Replay Table

| Version | Seed | Benchmark | CodeLeWM pass@1 | No-action pass@1 | LLM-order pass@1 | Lexical pass@1 | Lift vs no-action | 95% CI | Slice gate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v1.0 demo replay | 42 | HumanEval WS-D | 0.9787 | 0.8723 | 0.1489 | 0.6596 | +10.64 pts | [2.13, 21.28] | open |
| v1.0 demo replay | 42 | MBPP-Plus WS-D | 1.0000 | 1.0000 | 0.1765 | 1.0000 | +0.00 pts | [0.00, 0.00] | closed |
| v1.0 demo replay | 1729 | HumanEval WS-D | 0.9787 | 0.8936 | 0.1489 | 0.6596 | +8.51 pts | [2.13, 17.02] | open |
| v1.0 demo replay | 1729 | MBPP-Plus WS-D | 1.0000 | 1.0000 | 0.1765 | 1.0000 | +0.00 pts | [0.00, 0.00] | closed |

Aggregate demo artifact: `demo_report-e6fc06c328eed245`.
Aggregate demo gate: `claim_allowed=false`.

## Diagnostic Tables

### Retrieval

| Evidence | Seed | CodeLeWM/text Recall@1 | No-action Recall@1 | Shuffled Recall@1 | Lexical Recall@1 | MRR | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v0.2 action-swap | n/a | 0.2630 | 0.4410 | 0.0010 | 0.0450 | 0.3700 | closed: no-action dominance |
| v0.8 execution | 42 | 0.2754 | 0.0085 | 0.0297 | 0.0127 | 0.4125 | diagnostic pass |
| v0.8 execution | 1729 | 0.2924 | 0.0169 | 0.0254 | 0.0127 | 0.4213 | diagnostic pass |
| v0.9 execution | 42 | 0.2538 | 0.0115 | 0.0192 | 0.0115 | 0.3804 | diagnostic pass |
| v0.9 execution | 1729 | 0.2731 | 0.0192 | 0.0231 | 0.0115 | 0.3982 | diagnostic pass |

Retrieval supports training-health and substrate-diagnostic claims for v0.8/v0.9.
It does not reopen the v0.2 action-conditioned quality claim because the v0.2
headline action-use gate failed against no-action.

### Surprise, Probes, And Calibration

| Evidence | Metric | Seed or slice | Value | Gate status |
| --- | --- | --- | ---: | --- |
| v0.2 | Surprise pairwise AUC overall | n/a | 0.732763 | diagnostic only |
| v0.2 | Latent semantic structure | n/a | unsupported | closed |
| v0.8 | Mutation surprise AUC | 42 / 1729 | 1.0000 / 1.0000 | diagnostic pass |
| v0.8 | Pass/fail latent probe `z_pred_after` accuracy | 42 / 1729 | 0.5081 / 0.5892 | not met; controls comparable or stronger |
| v0.8 | Magnitude probe | both seeds | not evaluable | no val labels |
| v0.8 | Completion-level CodeLeWM ROC-AUC | HumanEval / MBPP-Plus | 0.9622, 0.9699 / 0.5772, 0.6941 | diagnostic only |
| v0.9 | Mutation surprise AUC | 42 / 1729 | 1.0000 / 1.0000 | diagnostic pass |
| v0.9 | Broad semantic surprise | both seeds | 0 scorable semantic pairs | closed by coverage |
| v0.9 | Latent probe `passed` target | 42 / 1729 | 0.6552 / 0.6158 | closed; controls stronger |
| v0.9 | Latent probe `output_magnitude_bucket` target | 42 / 1729 | 0.5000 / 0.8421 | closed; tied with `z_after` control |
| v0.9 | CodeLeWM ROC-AUC | 42 / 1729 | 0.9352 / 0.9746 | diagnostic only |
| v0.9 | HumanEval ROC-AUC | 42 / 1729 | 0.9196 / 0.9717 | diagnostic only |
| v0.9 | MBPP-Plus ROC-AUC | 42 / 1729 | 0.9730 / 0.9945 | diagnostic only |
| v0.9 | `p_pass` score key in downstream rows | both seeds | missing | not recorded |

## Final Public Claim Audit

| Public claim | Verdict | Exact supporting or blocking evidence | Required wording |
| --- | --- | --- | --- |
| Reproducible code-edit world-model harness and artifact pipeline | ALLOWED | HF Jobs runs, public artifacts, manifest verification, secret scans, and v1.0 paper-demo artifact `demo_report-e6fc06c328eed245`. | CodeLeWM is a reproducible research harness with manifest-backed public artifacts. |
| Non-collapsed learned execution checkpoints for diagnostic use | ALLOWED | v0.8/v0.9 training-health gates, retrieval lift over controls, and mutation surprise AUC `1.0000` on both v0.9 seeds. | The learned checkpoints support diagnostic execution-substrate analysis. |
| HumanEval WS-D narrow positive slice | ALLOWED | v1.0 replay: CodeLeWM pass@1 `0.9787` on both seeds; no-action `0.8723` and `0.8936`; lift `+10.64 pts` and `+8.51 pts` with CIs excluding zero. | CodeLeWM strongly reranks the HumanEval WS-D slice in the checked-in v0.9 replay. |
| MBPP-Plus/general downstream improvement | BLOCKED | v1.0 replay: MBPP-Plus CodeLeWM `1.0000`, no-action `1.0000`, lexical `1.0000`, lift `+0.00 pts` on both seeds. | The aggregate downstream claim remains closed because MBPP-Plus is saturated. |
| Broad coding improvement | BLOCKED | Cross-benchmark gate is closed; MBPP-Plus shows zero lift; v0.6 had no stable lift; v0.8 MBPP was dominated by lexical. | Do not say CodeLeWM generally improves coding, generated patches, or candidate ranking. |
| Action-conditioned retrieval/model-quality claim | BLOCKED | v0.2 text-action Recall@1 `0.263` and MRR `0.370048` lose to no-action Recall@1 `0.441` and MRR `0.533105`. | The tested action-use interventions are negative. |
| Semantic latent representation axes | BLOCKED | v0.2 semantic structure unsupported; v0.8 pass/fail probe misses controls; v0.9 probes do not beat controls across targets and seeds. | Do not name or claim validated semantic latent dimensions. |
| Broad semantic surprise | BLOCKED | v0.8 and v0.9 semantic-decoy alignment yields zero scorable semantic pairs for the broad categories. | Mutation surprise is healthy; broad semantic surprise is not established. |
| `p_pass` score key in downstream rows | NOT RECORDED | v0.9 calibration reports record `p_pass` as missing and use CodeLeWM as the primary evaluable score. | Treat p-pass calibration as diagnostic and avoid claiming a standalone p-pass downstream score. |
| Live/public LLM patch utility | BLOCKED | Meaningful live demos are workflow diagnostics; candidate code remains untrusted; no scaled downstream live patch utility gate opens. | Demos prove capture/scoring/reporting workflow, not coding usefulness. |

## Approved Short Claim Boundary

CodeLeWM can be described as a reproducible code-edit world-model research
harness with manifest-backed public artifacts, negative action-use evidence,
healthy diagnostic execution-substrate signals, and a narrow HumanEval WS-D
reranking slice in the v0.9/v1.0 replay. The final public model-quality claim is
closed because MBPP-Plus shows zero lift over no-action, broad semantic and
representation gates remain closed, and the artifact set does not support a
general claim that CodeLeWM improves coding.

## Validation Notes

The table values above were checked against the source benchmark reports and the
committed v1.0 machine report. The required local validation for the #406 PR is:

```bash
uv run pytest tests/docs -q
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests
uv run codelewm secret-scan docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md README.md docs/roadmap/FULL_COMPLETION.md docs/roadmap/IMPLEMENTATION.md docs/roadmap/NEXT_GOAL_PROMPT.md tests/docs/test_v1_final_claim_audit.py --include-suffix .md --include-suffix .py --json
git diff --check
```
