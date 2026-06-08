# CodeLeWM v1.0 Final Release Card

- Release name: `codelewm-v1-0-final-public-package`
- Parent tracker: #401
- Release package issue: #408
- Final artifact index: `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md`
- Final reproducibility checklist:
  `docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md`
- Final announcement draft:
  `docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md`
- Card date: `2026-06-08`
- Release status: public v1.0 research artifact package; broad model-quality claim closed.

## Summary

This card summarizes the final public CodeLeWM release package. The release
combines the checked-in v0.9 execution artifacts, the v1.0 deterministic
paper-demo replay, the final paper source package, and conservative
announcement wording.

The release is evidence-backed but claim-limited. It may be described as a
reproducible code-edit world-model research artifact with manifest-backed
public runs, negative action-use evidence, and a narrow HumanEval WS-D reranking
slice. It must not be described as a model that generally improves coding.

## Release Surfaces

| Surface | Path |
| --- | --- |
| Final public artifact index | `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md` |
| Final claim audit | `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md` |
| Final paper-demo artifact note | `docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md` |
| Final paper-demo card | `docs/cards/codelewm-v1-0-paper-demo-2026-06-08.md` |
| v0.9 dataset card | `docs/cards/codelewm-v0-9-execution-dataset-2026-06-07.md` |
| v0.9 seed 42 model card | `docs/cards/codelewm-v0-9-execution-model-seed-42-2026-06-07.md` |
| v0.9 seed 1729 model card | `docs/cards/codelewm-v0-9-execution-model-seed-1729-2026-06-07.md` |
| Final paper TeX | `docs/papers/codelewm_final_paper.tex` |
| Final paper claim audit | `docs/papers/codelewm_final_claim_audit.md` |
| Final paper PDF | `docs/papers/codelewm_final_paper.pdf` |
| Final arXiv source tarball | `docs/papers/codelewm_final_arxiv_source.tar.gz` |
| Reproducibility checklist | `docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md` |
| Announcement draft | `docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md` |

## Evidence Boundary

| Claim | Status | Evidence |
| --- | --- | --- |
| Reproducible code-edit world-model harness | Allowed | `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md`; paper-demo manifest verification |
| Manifest-backed artifact publication | Allowed | v0.2/v0.8/v0.9 HF runs, v0.9 public artifact index, final v1.0 index |
| Narrow HumanEval WS-D reranking slice | Allowed | v1.0 replay HumanEval rows for seeds 42 and 1729 |
| Aggregate downstream coding improvement | Blocked | MBPP-Plus CodeLeWM, no-action, and lexical pass@1 are all `1.0000` |
| Action-conditioned model-quality claim | Blocked | v0.2 text-action retrieval loses to no-action |
| Semantic latent-axis claim | Blocked | v0.2, v0.8, and v0.9 probe gates do not clear |
| Live patch utility | Blocked | demos are workflow diagnostics and not scaled utility gates |

## Verification Summary

The release checklist pins the exact clean-checkout commands. The required local
checks for this release package are:

```bash
uv run pytest tests/docs -q
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests
uv run codelewm manifest verify \
  --manifest docs/benchmark/v1_0/paper_demo/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/mbpp_plus/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/mbpp_plus/manifest.json \
  --json
uv run codelewm secret-scan docs/benchmark/v1_0/paper_demo --include-suffix .json --include-suffix .md --include-suffix .html --json
git diff --check
```

## Publication Copy Boundary

Approved:

> CodeLeWM v1.0 is a reproducible code-edit world-model research artifact with
> negative action-use evidence and a narrow HumanEval WS-D reranking slice. Its
> aggregate downstream model-quality claim remains closed.

Blocked:

- CodeLeWM generally improves coding.
- CodeLeWM has proven useful for generated patch ranking.
- CodeLeWM learns validated semantic latent dimensions.
- CodeLeWM beats the agreed baselines across HumanEval and MBPP-Plus.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Release shepherd | @AbdelStark | 2026-06-08 |
