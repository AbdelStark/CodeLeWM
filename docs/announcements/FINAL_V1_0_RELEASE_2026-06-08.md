# CodeLeWM v1.0 Final Release Announcement Draft

Status: draft copy for the final v1.0 public release package. Do not publish
until the #408 PR is merged to `main`, hosted CI is green, and #401 is closed.

## Short Announcement

CodeLeWM v1.0 is now packaged as a reproducible code-edit world-model research
artifact.

The final release includes manifest-backed public runs, checked-in benchmark
reports, a deterministic paper-demo replay, a final claim audit, and an arXiv
source package. The result is intentionally claim-limited: CodeLeWM shows a
narrow HumanEval WS-D reranking slice in the v0.9/v1.0 replay, while the
aggregate downstream model-quality claim remains closed because MBPP-Plus is
saturated against no-action and lexical controls.

MBPP-Plus is saturated against no-action and lexical controls, so this release
does not open the aggregate downstream claim.

This is a mixed negative/diagnostic release, not a claim that CodeLeWM generally
improves coding.

## Blog Draft

CodeLeWM explores whether a latent transition model over code edits can score
candidate patches. The final v1.0 release packages the project as a reproducible
research artifact rather than a product claim.

The systems result is real: the repository now includes public-safe data
contracts, Hugging Face Jobs training runs, downloaded-artifact verification,
manifest checks, secret scans, evaluation reports, a deterministic downstream
paper-demo replay, and a paper-ready claim audit. The final artifact index is
`docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md`.

The model-quality result is mixed. The final replay preserves a narrow
HumanEval WS-D positive slice for two v0.9 execution checkpoints, but MBPP-Plus
is saturated: CodeLeWM, no-action, and lexical controls all reach pass@1
`1.0000`. Earlier action-use interventions also failed against no-action. The
release therefore closes the broad downstream claim and presents CodeLeWM as
negative/diagnostic evidence with one narrow positive slice.

Useful entry points:

- Final artifact index:
  `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md`
- Final claim audit:
  `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md`
- Paper-demo artifacts:
  `docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md`
- Final paper source:
  `docs/papers/codelewm_final_paper.tex`
- Reproducibility checklist:
  `docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md`

## Social Draft

CodeLeWM v1.0 is packaged as a reproducible code-edit world-model research
artifact.

Final claim boundary: manifest-backed public runs, negative action-use
evidence, and a narrow HumanEval WS-D reranking slice. No broad coding
improvement claim: MBPP-Plus stays saturated against controls.

## Do Not Say

- Do not say CodeLeWM generally improves coding.
- Do not say CodeLeWM is a useful generated-patch reranker across benchmarks.
- Do not say CodeLeWM beats HumanEval and MBPP-Plus controls in aggregate.
- Do not say CodeLeWM learns validated semantic latent dimensions.
- Do not say the paper-demo replay is a fresh checkpoint scoring run.
- Do not say the local or live LLM demos prove patch utility.

## Approved Claim Boundary

Approved:

> CodeLeWM v1.0 is a reproducible code-edit world-model research artifact with
> manifest-backed public runs, negative action-use evidence, and a narrow
> HumanEval WS-D reranking slice.

Required qualifier:

> The aggregate downstream model-quality claim remains closed because MBPP-Plus
> is saturated against no-action and lexical controls.

Blocked qualifier:

> CodeLeWM generally improves coding.
