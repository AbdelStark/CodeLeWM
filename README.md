# CodeLeWM

<p align="center">
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/AbdelStark/CodeLeWM/pr.yml?branch=main&style=for-the-badge&label=CI">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.14-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/github/license/AbdelStark/CodeLeWM?style=for-the-badge">
  <img alt="Hugging Face artifacts" src="https://img.shields.io/badge/Hugging%20Face-public%20artifacts-FFD21E?style=for-the-badge">
  <img alt="Claim boundary" src="https://img.shields.io/badge/claims-negative%20diagnostic-111827?style=for-the-badge">
</p>

CodeLeWM is a Python ML research harness for learning latent transition models
over code edits.

It is not a code generator. It is a scorer and reranker for candidate patches:
given a before-state, an edit instruction, and candidate after-states or diffs,
CodeLeWM estimates which candidate best matches the learned transition.

```text
CodeState_before + EditAction -> latent(CodeState_after)
```

## Current Result

The systems path works end to end:

- public-safe Python edit datasets;
- manifest-backed training on Hugging Face Jobs;
- public dataset/model/run artifacts on Hugging Face;
- downloaded-artifact verification with checksums and secret scans;
- retrieval, action ablation, surprise, latent-probe, latent-matrix,
  scorer-quality, score,
  rerank, downstream-pack, downstream-rerank, and LLM-demo reports.

The first scientific result is negative. The tested action-conditioned variants
do not beat the no-action baseline on headline retrieval, and the v0.2
representation/downstream gates remain closed. This repository is publishable as
infrastructure and negative evidence, not as a claim that CodeLeWM improves
coding.

## Quickstart

```bash
uv sync --group dev --group data --group train
uv run scripts/first-results --overwrite
uv run codelewm secret-scan .artifacts/first-results docs/benchmark/FIRST_RESULTS.md --json
```

This rebuilds the local smoke artifact set and regenerates
`docs/benchmark/FIRST_RESULTS.md`. It proves the package-native dataset, pack,
train, eval, index, scorer-quality, manifest, and secret-scan loop on tiny
fixtures. It does not prove model quality.

## LLM + World-Model Demo

Run the deterministic fixture demo:

```bash
uv sync --group dev --group data --group train --group llm
uv run scripts/llm-world-model-demo
```

The task loads `.env` if present, stays in `CODELEWM_LLM_DRY_RUN=1` by default,
materializes the `bugfix-edge-case` scenario, generates candidate diffs through
the OpenRouter adapter fixture path, writes `codelewm.llm_candidate_pack.v1`,
runs `codelewm llm-demo` with a trusted package-native torch checkpoint and
`--require-learned-scorer`, verifies manifests, secret-scans publishable
outputs, and writes a visual report at
`.artifacts/llm-world-model-demo/run/demo.html`. If the local first-results
checkpoint is missing, the script regenerates it before scoring. The default
output is a terminal walkthrough of scenario selection, candidate generation,
learned world-model inference, artifact gates, and claim status.

Expected success signal:

```text
CodeLeWM LLM + World-Model Demo
mode: fixture dry-run | scorer: codelewm.torch_transition_scorer.v1 | success: true
[ok] 4/6 World-model inference
[ok] 5/6 Artifact gates
html report: .artifacts/llm-world-model-demo/run/demo.html
```

For non-interactive JSON output, use `uv run scripts/llm-world-model-demo --json`
or `CODELEWM_LLM_DEMO_OUTPUT=json`.

Select another built-in scenario with `--scenario <id>` or
`CODELEWM_LLM_DEMO_SCENARIO=<id>`. List available scenarios with
`uv run scripts/llm-world-model-demo --list-scenarios`.

Run the v0.6 execution-rerank tour with a downloaded seed-42 checkpoint:

```bash
CODELEWM_LLM_DRY_RUN=0 CODELEWM_LLM_MAX_CANDIDATES=2 \
  uv run scripts/llm-world-model-demo \
  --scenario execution-rerank-mbpp \
  --checkpoint .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42/checkpoints/last.pt \
  --tour 5 \
  --html .artifacts/v0-6-execution-rerank-tour-live.html
```

The tour samples live OpenRouter candidates for five public-safe synthetic
MBPP-style tasks, labels them only through `codelewm.data.sandbox`, scores them
with the v0.6 execution-substrate checkpoint, writes
`codelewm.harness.execution_rerank_tour.v1` plus the unchanged
`codelewm.harness.execution_rerank_view_model.v1`, and keeps the claim gate
closed below the scaled 100-example downstream benchmark. A committed
asciicast lives in `docs/demo/execution_rerank_tour_2026-05-31.cast`.

Live OpenRouter mode is explicit:

```bash
cp .env.example .env
# Fill OPENROUTER_API_KEY locally. Keep .env untracked.
CODELEWM_LLM_DRY_RUN=0 uv run scripts/llm-world-model-demo
```

### Anthropic BYOK Through OpenRouter

CodeLeWM supports OpenRouter BYOK for Anthropic keys without silently switching
to a direct Anthropic client.

```bash
# .env, kept local
OPENROUTER_API_KEY=<openrouter-api-key>
OPENROUTER_MANAGEMENT_KEY=<openrouter-management-key>
ANTHROPIC_API_KEY=<anthropic-provider-key>
CODELEWM_LLM_DRY_RUN=0
CODELEWM_OPENROUTER_BYOK=1
CODELEWM_OPENROUTER_BYOK_PROVIDER=anthropic
CODELEWM_OPENROUTER_BYOK_KEY_ENV=ANTHROPIC_API_KEY
CODELEWM_OPENROUTER_BYOK_MANAGEMENT_KEY_ENV=OPENROUTER_MANAGEMENT_KEY
CODELEWM_OPENROUTER_BYOK_REQUIRE=1
CODELEWM_OPENROUTER_BYOK_REGISTER=1
CODELEWM_OPENROUTER_BYOK_DRY_RUN=0
```

`CODELEWM_OPENROUTER_BYOK_REGISTER=1` intentionally creates an encrypted
Anthropic BYOK credential in the OpenRouter workspace via OpenRouter's BYOK API.
Keep `CODELEWM_OPENROUTER_BYOK_DRY_RUN=1` to validate the registration contract
without sending the provider key. Registration uses the OpenRouter management
key named by `CODELEWM_OPENROUTER_BYOK_MANAGEMENT_KEY_ENV`; normal chat requests
still authenticate with `OPENROUTER_API_KEY`. If the BYOK credential already
exists in the OpenRouter dashboard, set `CODELEWM_OPENROUTER_BYOK_REGISTER=0`
and keep `CODELEWM_OPENROUTER_BYOK=1`. CodeLeWM records redacted BYOK routing
metadata and never writes provider keys to reports.

For Anthropic BYOK, start with
`CODELEWM_LLM_PROVIDER_OPTIONS_JSON='{"sort":"price"}'`. Add `zdr: true` only
when OpenRouter shows a matching Zero Data Retention endpoint for the pinned
provider route; otherwise OpenRouter rejects the request before generation.

Dry-run the registration contract without sending secrets:

```bash
uv run codelewm openrouter byok-register \
  --provider anthropic \
  --key-env ANTHROPIC_API_KEY \
  --management-key-env OPENROUTER_MANAGEMENT_KEY \
  --name "CodeLeWM Anthropic BYOK" \
  --allowed-model anthropic/claude-4.5-sonnet \
  --dry-run \
  --json
```

## Evidence

| Evidence | Result | Report |
| --- | --- | --- |
| First local smoke loop | systems smoke only | `docs/benchmark/FIRST_RESULTS.md` |
| Scaled HF systems run | negative vs no-action | `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md` |
| Action-use margin run | negative vs no-action | `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md` |
| Margin + retrieval run | improved but still negative | `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md` |
| v0.2 action-swap run | negative across action-use, latent-probe, downstream gates | `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md` |
| Public summary | negative/diagnostic boundary | `docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md` |
| Public artifact index | HF dataset/model/run paths | `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-21.md` |
| Downstream fixture gate | one example, claim-blocked | `docs/benchmark/DOWNSTREAM_RERANKING_BENCHMARK.md` |

Public Hugging Face repositories:

- `abdelstark/codelewm-public-shard`
- `abdelstark/codelewm-transition-model`
- `abdelstark/codelewm-runs`

## Command Surface

```bash
uv run codelewm dataset build --help
uv run codelewm dataset pack --help
uv run codelewm train --help
uv run codelewm eval retrieval --help
uv run codelewm eval latent-probe --help
uv run codelewm eval latent-matrix --help
uv run codelewm eval surprise --help
uv run codelewm eval scorer-quality --help
uv run codelewm eval downstream-pack --help
uv run codelewm eval downstream-rerank --help
uv run codelewm score --help
uv run codelewm rerank --help
uv run codelewm llm-demo --help
uv run codelewm openrouter byok-register --help
uv run codelewm manifest verify --help
uv run codelewm secret-scan --help
```

Full usage guide: `docs/usage/USAGE.md`.

## Architecture

```text
raw edit sources
  -> source adapters, license gates, split/dedup policy
  -> CodeState_before, EditAction, CodeState_after
  -> packed transition batches
  -> JEPA-style latent transition training
  -> checkpoint + transition index
  -> score/rerank, retrieval, surprise, downstream, and LLM-demo reports
```

Core packages:

- `codelewm.data`: source loading, filtering, CodeState extraction, packing;
- `codelewm.model`: action encoders, predictor modules, objective helpers;
- `codelewm.training`: manifest-backed CPU smoke and torch training runners;
- `codelewm.eval`: retrieval, surprise, latent probes, downstream claim gates;
- `codelewm.harness`: scorer, reranker, OpenRouter adapter, LLM demo, CLI;
- `codelewm.observability`: artifact manifests, logs, redaction;
- `codelewm.security`: non-execution parsing, license policy, secret scans.

Root `train.py`, `eval.py`, and Hydra configs are compatibility artifacts from
the original LeWorldModel seed. The package CLI is the CodeLeWM path.

## Install

```bash
uv sync --group dev
uv sync --group dev --group data
uv sync --group dev --group train
uv sync --group dev --group eval
uv sync --group dev --group llm
uv sync --group dev --group release
```

The package extras mirror the same boundaries:

```bash
uv sync --extra data
uv sync --extra train
uv sync --extra eval
uv sync --extra llm
```

## Validate

```bash
uv run pytest tests/ -q
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests
uv lock --check
git diff --check
```

For release work, also run the package build, dependency audit, provenance, and
wheel-install gates described in `docs/release/DEPENDENCY_PROVENANCE.md`.

## Roadmap

The completed v1.1 boundary is a claim-safe diagnostic workflow:

- LLM + world-model demo complete through #186-#189;
- preliminary publication package complete through #193-#194;
- downstream reranking fixture and claim gate complete through #190-#192;
- BYOK/demo/readme polish complete through #206.
- meaningful scenario selection complete through #226.
- visual model observability and TUI roadmap locked through #236.
- optional TensorBoard-compatible training export complete through #237.
- checkpoint tensor/layer inspection complete through #238.
- latent representation matrix diagnostics complete through #239.
- run timeline and monitoring reports complete through #240.
- optional Textual TUI for demo inspection complete through #241.
- shared visual view model for JSON/rich/HTML demo parity complete through #242.
- model/latent/tensor diagnostic links in demo reports complete through #243.
- diagnostics-driven candidate-contrast experiment plan complete through #244.
- public visual observability artifact set complete through #245.
- v0.6 execution-rerank LLM showcase complete through #307, with a committed
  asciicast and static HTML report path.

Open next streams:

- scaled downstream reranking benchmark: #209/#210/#211;
- next positive-model research hypothesis: #212, with CWM comparison in #178.

Public wording cannot say CodeLeWM improves candidate patch ranking until a
scaled downstream gate passes on at least 100 labeled examples.

Live planning:

- `docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md`
- `docs/roadmap/MODEL_OBSERVABILITY_TUI_ROADMAP.md`
- `docs/roadmap/DIAGNOSTICS_DRIVEN_MODEL_EXPERIMENT.md`
- `docs/benchmark/VISUAL_OBSERVABILITY_ARTIFACTS_2026-05-21.md`
- `docs/roadmap/FULL_COMPLETION.md`
- `docs/roadmap/IMPLEMENTATION.md`
- `docs/roadmap/NEXT_GOAL_PROMPT.md`

## Claim Boundary

You can cite CodeLeWM today as:

- a public, reproducible code-edit world-model research harness;
- a verified Hugging Face Jobs and artifact-publication pipeline;
- a negative result for tested action-use interventions;
- a fixture-proven LLM-candidate reranking workflow.

Do not cite it today as:

- a model that improves coding;
- a model with validated semantic latent dimensions;
- a checkpoint that beats no-action on action-conditioned retrieval;
- a downstream patch-ranking system with proven usefulness.

## Attribution

CodeLeWM starts from the LeWorldModel codebase and keeps its JEPA-style model
shape as the implementation seed:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
