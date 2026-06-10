# CodeLeWM

<p align="center">
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/AbdelStark/CodeLeWM/pr.yml?branch=main&style=for-the-badge&label=CI">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.14-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge">
  <img alt="Hugging Face artifacts" src="https://img.shields.io/badge/Hugging%20Face-public%20artifacts-FFD21E?style=for-the-badge">
  <img alt="Claim boundary" src="https://img.shields.io/badge/claims-negative%20diagnostic-111827?style=for-the-badge">
  <a href="https://zenodo.org/records/20630120">
    <img alt="Paper DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20630120-2D9CDB?style=for-the-badge&logo=zenodo&logoColor=white">
  </a>
  <a href="docs/papers/codelewm_final_paper.pdf">
    <img alt="Companion paper PDF" src="https://img.shields.io/badge/paper-PDF-B31B1B?style=for-the-badge">
  </a>
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
  rerank, downstream-pack, downstream-rerank, LLM-demo, execution-rerank, and
  p-pass calibration reports.

The final evidence boundary is mixed and claim-limited. CodeLeWM has a
reproducible code-edit world-model harness, verified public artifact
infrastructure, negative action-conditioned v0.2 evidence, and a narrow v0.9
HumanEval WS-D downstream reranking win across two seeds. The overall public
claim remains closed because MBPP-Plus has zero lift over no-action and broader
semantic, representation, and coding-usefulness gates do not clear. This
repository is publishable as a reproducible negative/diagnostic study with a
narrow positive HumanEval slice, not as a broad claim that CodeLeWM improves
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
HTML report and asciicast live in `docs/demo/`.

## Final Paper Demo

Assemble the deterministic v1.0 downstream paper-demo artifact set:

```bash
uv run scripts/paper-demo --out .artifacts/paper-demo --overwrite
```

This is a clean-checkout replay over the checked-in v0.9 WS-D score rows, not a
fresh checkpoint scoring run and not a live OpenRouter demo. It writes
`reports/paper_demo_report.json`, `reports/paper_demo_claim_gate.json`,
`reports/paper_demo_table.md`, `reports/run_timeline.json`, `demo.html`,
`reports/secret_scan_report.json`, and `manifest.json`. The aggregate claim
gate remains closed because MBPP-Plus is saturated against no-action.
The committed artifact set is documented in
`docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md` and lives under
`docs/benchmark/v1_0/paper_demo/`.

## Final v1.0 Release Package

The final public release package is:

- artifact index: `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md`;
- final claim audit: `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md`;
- release card: `docs/cards/codelewm-v1-0-final-release-2026-06-08.md`;
- paper-demo card: `docs/cards/codelewm-v1-0-paper-demo-2026-06-08.md`;
- reproducibility checklist:
  `docs/release/V1_0_REPRODUCIBILITY_CHECKLIST_2026-06-08.md`;
- announcement draft:
  `docs/announcements/FINAL_V1_0_RELEASE_2026-06-08.md`;
- final paper package: `docs/papers/codelewm_final_paper.tex`,
  `docs/papers/codelewm_final_paper.pdf`, and
  `docs/papers/codelewm_final_arxiv_source.tar.gz`.

The final package preserves the same public boundary as the paper: reproducible
research artifact, negative action-use evidence, narrow HumanEval WS-D slice,
and no broad coding-improvement claim.

Release tracker status: #406 consolidated benchmark tables and the final claim
audit, #407 rewrote the paper around the final downstream evidence, and
#408 published the final artifact index and release package.

## v0.6 Publication Landing

The v0.6 public artifact map is:

- artifact index: `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-31.md`;
- dataset card: `docs/cards/codelewm-v0-6-execution-dataset-2026-05-31.md`;
- model cards:
  `docs/cards/codelewm-v0-6-execution-model-seed-42-2026-05-31.md` and
  `docs/cards/codelewm-v0-6-execution-model-seed-1729-2026-05-31.md`;
- blog-style announcement draft:
  `docs/blog/2026-05-31-codelewm-v0-6-substrate-pivot.md`;
- demo: `docs/demo/execution_rerank_tour_2026-05-31.html`;
- arXiv package: `docs/papers/ARXIV_SUBMISSION.md`.

This v0.6 landing is historical evidence. Its remaining arXiv/HF publication
follow-through is superseded by the final #401 paper/demo release queue, which
uses the v0.9 claim audit and final downstream demo package as the public
release boundary.

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
