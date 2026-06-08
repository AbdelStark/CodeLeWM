# CodeLeWM v1.0 Final Public Artifact Index 2026-06-08

This index maps the final CodeLeWM v1.0 public release surfaces for #401/#408.
It is a release-package index, not a general coding-usefulness claim.

Current claim boundary:

- CodeLeWM is a reproducible code-edit world-model research harness with
  manifest-backed public artifacts.
- The tested action-use interventions remain negative.
- The v0.9/v1.0 replay supports a narrow HumanEval WS-D positive diagnostic
  slice across two seeds.
- The aggregate downstream claim remains closed because MBPP-Plus is saturated
  against no-action and lexical controls.
- Broad coding-improvement, live patch-utility, semantic latent-axis, and
  general benchmark-improvement claims remain blocked.

## Public Repositories

| Surface | Repository | Repo type | Revision / path |
| --- | --- | --- | --- |
| Historical public code-edit shard | `abdelstark/codelewm-public-shard` | dataset | documented by the v0.2 and earlier cards |
| Historical transition-model artifacts | `abdelstark/codelewm-transition-model` | model | documented by the v0.2 and earlier cards |
| v0.9 execution pack | `abdelstark/codelewm-execution-pack` | dataset | `v0.9.0-rc1` |
| Training and result runs | `abdelstark/codelewm-runs` | dataset | v0.2, v0.8, and v0.9 run paths |
| Final paper/demo/release docs | `github.com/AbdelStark/CodeLeWM` | git | `docs/benchmark/`, `docs/cards/`, `docs/papers/`, `docs/release/`, `docs/announcements/` |

## Final Indexed Artifacts

| Artifact | Public location | Local evidence | Card / report | Claim posture |
| --- | --- | --- | --- | --- |
| v0.2 action-swap/inverse-action run | `abdelstark/codelewm-runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b` | downloaded and locally verified HF Jobs artifacts; manifest and secret-scan gates passed | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`; `docs/cards/codelewm-v0-2-action-swap-dataset-2026-05-20.md`; `docs/cards/codelewm-v0-2-action-swap-model-2026-05-20.md` | negative action-use and representation evidence |
| v0.6 execution-rerank evidence | Code repo and public run artifacts | two-seed downstream rerank table and historical arXiv package | `docs/benchmark/V0_6_RERANK_FULL_2026-06-01.md`; `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-31.md`; `docs/papers/two_substrate_paper.tex` | historical substrate diagnostic, not final claim |
| v0.8 execution-trace result | `abdelstark/codelewm-runs` and checked-in eval artifacts | HumanEval WS-D positive slice with MBPP/probe blockers | `docs/benchmark/EXECUTION_V0_8_RESULTS_2026-06-05.md`; `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-05.md`; v0.8 dataset/model cards | mixed diagnostic result |
| v0.9 execution pass/fail pack | `abdelstark/codelewm-execution-pack@v0.9.0-rc1` | `codelewm-passfail-execution-pack-20260606T122240Z`; 2,188 records; manifest verify `ok=true`; secret scan `ok=true` | `docs/cards/codelewm-v0-9-execution-dataset-2026-06-07.md` | final training/eval substrate |
| v0.9 seed 42 model run | `abdelstark/codelewm-runs/codelewm-v0-9-short-execution-20260606-69f798a-seed-42` | `training_run-992f7757f2780da4`; checkpoint SHA `c783fa0dbe5da6bd072ff0b2f2753bdbac9fe684b49bf82e70ab6a2f69d513da` | `docs/cards/codelewm-v0-9-execution-model-seed-42-2026-06-07.md` | HumanEval WS-D positive; aggregate claim closed |
| v0.9 seed 1729 model run | `abdelstark/codelewm-runs/codelewm-v0-9-short-execution-20260606-69f798a-seed-1729` | `training_run-91e9cf7c645379b3`; checkpoint SHA `34ebb282b284580dd123c781ae77c93cc36bbffc4eeeee9f0bd4cdf8042001eb` | `docs/cards/codelewm-v0-9-execution-model-seed-1729-2026-06-07.md` | HumanEval WS-D positive; aggregate claim closed |
| v0.9 checked-in eval reports | Code repo | seed 42 and seed 1729 retrieval, surprise, latent-probe, rerank, and calibration reports under `docs/benchmark/v0_9/` | `docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md`; `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-07.md` | final benchmark evidence before v1.0 replay |
| v1.0 paper-demo replay | Code repo | `demo_report-e6fc06c328eed245`; `docs/benchmark/v1_0/paper_demo/reports/paper_demo_report.json`; `slice_count=4`; `problem_count=128`; `completion_count=768`; aggregate `claim_allowed=false` | `docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md`; `docs/cards/codelewm-v1-0-paper-demo-2026-06-08.md`; `docs/benchmark/v1_0/paper_demo/` | final deterministic demo package |
| v1.0 final claim audit | Code repo | paper-ready consolidated table over v0.2, v0.6, v0.8, v0.9, and v1.0 replay | `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md` | broad claim closed; narrow HumanEval allowed |
| v1.0 final paper package | Code repo | TeX source, claim audit, PDF, and arXiv source tarball | `docs/papers/codelewm_final_paper.tex`; `docs/papers/codelewm_final_claim_audit.md`; `docs/papers/codelewm_final_paper.pdf`; `docs/papers/codelewm_final_arxiv_source.tar.gz`; `docs/papers/ARXIV_SUBMISSION.md` | paper package for final mixed conclusion |
| v1.0 release package | Code repo | final public index, release card, reproducibility checklist, and announcement draft | `docs/cards/codelewm-v1-0-final-release-2026-06-08.md`; `docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md`; `docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md` | public release packaging and communications boundary |

## Final Paper-Demo Verification

```bash
uv run scripts/paper-demo \
  --out docs/benchmark/v1_0/paper_demo \
  --overwrite \
  --json

uv run codelewm manifest verify \
  --manifest docs/benchmark/v1_0/paper_demo/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/mbpp_plus/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/mbpp_plus/manifest.json \
  --json

uv run codelewm secret-scan docs/benchmark/v1_0/paper_demo \
  --include-suffix .json \
  --include-suffix .md \
  --include-suffix .html \
  --json
```

Expected result: paper-demo aggregate `claim_allowed=false`, manifest
verification `ok=true`, and secret scan `ok=true` with zero findings.

## Final Paper Package Verification

```bash
scripts/build-codelewm-final-paper

uv run codelewm secret-scan \
  docs/papers/codelewm_final_paper.tex \
  docs/papers/codelewm_final_claim_audit.md \
  docs/papers/ARXIV_SUBMISSION.md \
  docs/papers/codelewm_final_paper.pdf \
  docs/papers/codelewm_final_arxiv_source.tar.gz \
  --include-suffix .tex \
  --include-suffix .md \
  --include-suffix .pdf \
  --include-suffix .gz \
  --json
```

Expected result: the build writes the final PDF and arXiv source tarball, and
secret scan returns `ok=true` with zero findings.

## Release Documentation Verification

```bash
uv run pytest tests/docs -q
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests
uv run codelewm secret-scan \
  README.md \
  docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md \
  docs/cards/codelewm-v1-0-final-release-2026-06-08.md \
  docs/cards/codelewm-v1-0-paper-demo-2026-06-08.md \
  docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md \
  docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md \
  --include-suffix .md \
  --json
git diff --check
```

Expected result: docs tests pass, compileall exits 0, secret scan returns
`ok=true`, and `git diff --check` has no output.

## Approved Publication Copy

Safe short copy:

> CodeLeWM v1.0 is a reproducible code-edit world-model research artifact with
> manifest-backed public runs, negative action-use evidence, and a narrow
> HumanEval WS-D reranking slice. The final aggregate claim remains closed
> because MBPP-Plus is saturated against no-action and lexical controls, so the
> release does not claim broad coding improvement.

Do not publish wording that says CodeLeWM generally improves coding, generated
patches, or candidate ranking without a future manifest-backed gate that clears
the agreed cross-benchmark baselines.
