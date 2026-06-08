# CodeLeWM Final Paper Claim Audit

This audit binds `docs/papers/codelewm_final_paper.tex` to the final v1.0
claim boundary. It supersedes the older v0.6-only
`docs/papers/two_substrate_claim_audit.md` for final release wording, while the
old audit remains historical evidence for the two-substrate draft.

## Source Reports

| Paper surface | Source artifact |
| --- | --- |
| v0.2 action-use negative result | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` |
| v0.8 execution repair | `docs/benchmark/EXECUTION_V0_8_RESULTS_2026-06-05.md` |
| v0.9 final gate suite | `docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md` |
| v1.0 paper-demo artifact note | `docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md` |
| v1.0 final benchmark/claim audit | `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md` |
| v1.0 demo machine report | `docs/benchmark/v1_0/paper_demo/reports/paper_demo_report.json` |
| final reproducibility checklist | `docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md` |

## Quantitative Claims

| Claim in paper | Value | Source |
| --- | ---: | --- |
| v0.2 text-action retrieval Recall@1 / MRR | 0.263 / 0.370048 | `V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` |
| v0.2 no-action retrieval Recall@1 / MRR | 0.441 / 0.533105 | `V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` |
| v0.9 seed-42 retrieval Recall@1 | 0.2538 | `EXECUTION_V0_9_RESULTS_2026-06-07.md` |
| v0.9 seed-1729 retrieval Recall@1 | 0.2731 | `EXECUTION_V0_9_RESULTS_2026-06-07.md` |
| v0.9 mutation surprise AUC | 1.0000 / 1.0000 | `EXECUTION_V0_9_RESULTS_2026-06-07.md` |
| v1.0 HumanEval WS-D CodeLeWM pass@1 | 0.9787 / 0.9787 | `paper_demo_report.json` |
| v1.0 HumanEval WS-D no-action pass@1 | 0.8723 / 0.8936 | `paper_demo_report.json` |
| v1.0 HumanEval WS-D lift over no-action | +10.64 / +8.51 pts | `paper_demo_report.json` |
| v1.0 HumanEval WS-D lift CI | [2.13, 21.28] / [2.13, 17.02] | `paper_demo_report.json` |
| v1.0 MBPP-Plus CodeLeWM/no-action/lexical pass@1 | 1.0000 / 1.0000 / 1.0000 | `paper_demo_report.json` |
| v1.0 MBPP-Plus lift over no-action | +0.00 pts on both seeds | `paper_demo_report.json` |
| v1.0 paper-demo artifact id | `demo_report-e6fc06c328eed245` | `manifest.json` |

## Structural Claims

The final paper is structured around three research questions, a PGFPlots
downstream replay figure, explicit lessons, a WS-D benchmark definition, and a
concrete outlook section. Those sections are permitted only insofar as they
preserve the claim boundary below:

- the research questions answer RQ1 negatively, RQ2 diagnostically, and RQ3 as
  a partial/narrow downstream result;
- WS-D is described as a deterministic mutation-distractor reranking benchmark
  over checked-in v0.9 labels and score rows, not as a live or freshly scored
  benchmark;
- the v1.0 replay coverage is limited to 128 problem-slice rows and 768
  candidate completions across the four seed/benchmark slices;
- the figure visualizes the v1.0 paper-demo replay and must show MBPP-Plus
  saturation by CodeLeWM, no-action, and lexical controls;
- the outlook lists future gates that would be required to open a stronger
  claim, not claims already achieved by the current artifact.

## Claim Boundary

Allowed:

- CodeLeWM is a reproducible code-edit world-model research harness.
- The artifact pipeline is manifest-backed, checksum-verifiable, and
  secret-scanned.
- v0.8/v0.9 execution substrates show healthy diagnostic retrieval and
  mutation-surprise signals.
- The final replay supports a narrow HumanEval WS-D reranking slice.

Blocked:

- broad coding improvement;
- MBPP-Plus or general downstream improvement;
- action-conditioned model-quality from v0.2;
- semantic latent-axis claims;
- broad semantic-surprise claims;
- standalone downstream `p_pass` scoring claims;
- live LLM patch utility claims.

The paper conclusion must keep the aggregate public model-quality claim closed.
