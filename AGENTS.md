# CodeLeWM Agent Context

This repository is a spec-driven research artifact for learning latent
transition models over Python code edits. Treat the spec corpus, RFCs,
GitHub issues, and roadmap docs as the contract before making code changes.

## Current State

As of 2026-05-19, CodeLeWM has a real package surface but does not yet have a
meaningful first training result.

Implemented foundations:

- `codelewm.data`: source adapters, filtering, license decisions, split and
  dedup policy, CodeState extraction, action extraction, staging and pack
  helpers, and dataset manifests.
- `codelewm.model`: transition interfaces, action encoders, predictor modules,
  transition energy, objective helpers, retrieval-loss gate, and checkpoint
  compatibility manifests.
- `codelewm.training`: manifest-backed training runner, resume compatibility,
  default configs, a CPU smoke executor, and a package-native torch executor
  over packed CodeLeWM transition batches.
- `codelewm.eval`: retrieval metrics, hard-negative pools, required baselines,
  action-view policy, collapse diagnostics, and patch-surprise reports.
- `codelewm.harness`: package CLI entry point with landed `score`, `rerank`,
  `train`, `eval retrieval`, `eval surprise`, `manifest verify`, and
  `secret-scan` commands.
- `codelewm.observability` and `codelewm.security`: artifact manifests,
  structured logs, redaction, public license gates, checkpoint trust checks,
  non-execution guards, and secret scanning.

Missing for first meaningful results:

- `codelewm index` CLI flow and retrieval-prior scorer/reranker integration.
- A reproducible first-results report with dataset, training, checkpoint,
  retrieval, surprise, index, license, and secret-scan evidence.
- Release and publishing automation that can build, verify, and publish package
  artifacts without weakening optional-runtime boundaries.

Root `train.py`, root `eval.py`, and the Hydra configs are inherited from the
original image/LeWM seed. They are compatibility artifacts, not the source of
truth for CodeLeWM's code-edit training path.

## Required Reading

Before editing, read:

- `SPEC.md`
- The relevant file under `docs/spec/`
- The relevant RFC under `docs/rfcs/`
- `docs/roadmap/FULL_COMPLETION.md`
- `docs/roadmap/IMPLEMENTATION.md`
- `CONTRIBUTING.md`

If security, manifests, checkpoints, logs, licensing, candidate code, configs,
or reports are touched, also read `docs/spec/06-security.md` and
`docs/spec/05-observability.md`.

## Work Rules

- One GitHub issue per branch and PR.
- Keep public docs direct and evidence-backed. Do not add unverifiable claims.
- Preserve schema versions unless the issue explicitly changes a public schema.
- Candidate code, source data, configs, and checkpoints are untrusted input.
- Do not import, execute, or test-run candidate code while parsing or scoring it.
- Keep optional training/data dependencies host-owned and explicitly grouped.
- Prefer explicit typed failures over silent row drops, silent clipping, or best
  effort artifact writes.
- Every published artifact must be schema-versioned, finite, JSON-native where
  applicable, checksum-verifiable, and secret-scanned.

## Implementation Order

Use GitHub issues as the authoritative queue. The intended order starts with:

1. #109 Migrate dependency management and CI to `uv`.
2. #110 Add dataset build CLI.
3. #111 Add dataset pack CLI and a tiny committed fixture dataset.
4. #112 Wire a concrete package-native training executor.
5. #113 Expose `codelewm train`.
6. #114 and #115 expose retrieval and surprise evaluation CLI commands.
7. #116 Build transition indexes and connect retrieval priors to scoring.
8. #117 Generate the first-results report from reproducible commands.
9. #118 through #122 scale the dataset, training, ablations, reports, and cards.
10. #123 through #126 harden publishing, release automation, docs, and final
   artifact freeze.

## Validation

Current lightweight validation after the `uv` migration:

```bash
uv sync --group dev
uv run pytest tests/
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests
uv run codelewm --help
```

First-results work is not complete until artifact verification, secret scanning,
retrieval baselines, surprise metrics, and the benchmark report all pass from a
clean checkout.
