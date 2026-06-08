# CodeLeWM v1.0 Reproducibility Checklist 2026-06-08

This checklist is the clean-checkout release reproduction path for the final
v1.0 paper/demo package. It does not launch new HF Jobs and does not execute
candidate patches outside the existing sandboxed data-build artifacts.

## Scope

| Surface | Canonical path |
| --- | --- |
| Final public artifact index | `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md` |
| Final claim audit | `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md` |
| Paper-demo artifact note | `docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md` |
| Paper-demo artifact directory | `docs/benchmark/v1_0/paper_demo/` |
| Paper TeX source | `docs/papers/codelewm_final_paper.tex` |
| Paper claim audit | `docs/papers/codelewm_final_claim_audit.md` |
| Paper PDF | `docs/papers/codelewm_final_paper.pdf` |
| arXiv source tarball | `docs/papers/codelewm_final_arxiv_source.tar.gz` |
| Final release card | `docs/cards/codelewm-v1-0-final-release-2026-06-08.md` |
| Paper-demo card | `docs/cards/codelewm-v1-0-paper-demo-2026-06-08.md` |
| Announcement draft | `docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md` |

## 1. Prepare Environment

```bash
uv sync --group dev --group data --group train
```

Expected result: uv installs the project and development/test dependencies from
the checked-in lockfile.

## 2. Regenerate The Final Paper Demo

```bash
uv run scripts/paper-demo \
  --out docs/benchmark/v1_0/paper_demo \
  --overwrite \
  --json
```

Expected result:

- `schema_version=codelewm.harness.paper_demo_run.v1`;
- `score_source=replay_existing_scores`;
- `slice_count=4`;
- `problem_count=128`;
- `completion_count=768`;
- aggregate `claim_allowed=false`.

## 3. Verify Paper-Demo Manifest Lineage

```bash
uv run codelewm manifest verify \
  --manifest docs/benchmark/v1_0/paper_demo/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/mbpp_plus/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/mbpp_plus/manifest.json \
  --json
```

Expected result: `ok=true`, six files checked, and the four parent rerank
artifacts verified.

## 4. Secret-Scan Public Demo Artifacts

```bash
uv run codelewm secret-scan docs/benchmark/v1_0/paper_demo \
  --include-suffix .json \
  --include-suffix .md \
  --include-suffix .html \
  --json
```

Expected result: `ok=true` with zero findings.

## 5. Rebuild The Final Paper Package

```bash
scripts/build-codelewm-final-paper
```

Expected result:

- `docs/papers/codelewm_final_paper.pdf` exists;
- `docs/papers/codelewm_final_arxiv_source.tar.gz` exists;
- the paper preserves the claim boundary in
  `docs/papers/codelewm_final_claim_audit.md`.

Minor TeX layout warnings are acceptable only when the script exits 0 and the
PDF is produced.

## 6. Secret-Scan Paper And Release Docs

```bash
uv run codelewm secret-scan \
  docs/papers/codelewm_final_paper.tex \
  docs/papers/codelewm_final_claim_audit.md \
  docs/papers/ARXIV_SUBMISSION.md \
  docs/papers/codelewm_final_paper.pdf \
  docs/papers/codelewm_final_arxiv_source.tar.gz \
  docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md \
  docs/cards/codelewm-v1-0-final-release-2026-06-08.md \
  docs/cards/codelewm-v1-0-paper-demo-2026-06-08.md \
  docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md \
  docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md \
  --include-suffix .tex \
  --include-suffix .md \
  --include-suffix .pdf \
  --include-suffix .gz \
  --json
```

Expected result: `ok=true` with zero findings.

## 7. Validate Docs And Python Importability

```bash
uv run pytest tests/docs -q
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests
git diff --check
```

Expected result: docs tests pass, compileall exits 0, and `git diff --check`
has no output.

## 8. Confirm Final Claim Boundary

The release is ready only if all of the following statements remain true:

- README links `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md`.
- README links this checklist and
  `docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md`.
- `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md` says broad
  coding-improvement claims are closed.
- `docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md` says not to claim
  general coding improvement.
- `docs/roadmap/NEXT_GOAL_PROMPT.md` is a completion record, not an active
  child-issue prompt.

## 9. Hosted CI And Issue Closure

After local validation:

1. Push the #408 branch and open a PR that closes only #408.
2. Wait for hosted CI to pass on the PR head.
3. Merge #408 to `main`.
4. Confirm #408 is closed.
5. Update #401 so the #408 workstream and final global checks are checked.
6. Close #401 only after `main` contains the #408 merge commit and CI is green.

## Final Public Wording

Allowed:

> CodeLeWM v1.0 is a reproducible code-edit world-model research artifact with
> manifest-backed public evidence, negative action-use results, and a narrow
> HumanEval WS-D reranking slice.

Blocked:

> CodeLeWM generally improves coding.

> CodeLeWM is a useful generated-patch reranker across benchmarks.

> CodeLeWM learns validated semantic latent axes.
