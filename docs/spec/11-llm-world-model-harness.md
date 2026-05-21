# LLM + World-Model Harness

This spec defines the post-v0.2 showcase and evaluation surface: an LLM
generates candidate patches, CodeLeWM scores and reranks the candidates, and a
report records whether the world-model score added value over the baselines.

The harness is a research instrument. It is allowed to be useful as a demo
before the model wins a benchmark, but public docs must keep the claim boundary
clear: current CodeLeWM checkpoints do not yet support a positive
action-conditioned quality, latent-axis, or downstream coding-usefulness claim.

## Streams

The post-v0.2 work is split into three streams:

1. LLM + world-model harness demo: issue #183, with child issues #186 through
   #189.
2. Downstream candidate-reranking benchmark: issue #184, with child issues #190
   through #192.
3. Preliminary results publication package: issue #185, with child issues #193
   and #194.

Each stream must land as one issue per branch and PR. The demo stream may run
before the benchmark stream proves usefulness, but it must emit claim-safe
reports.

## OpenRouter Contract

The public LLM integration path uses the OpenRouter Python SDK. The SDK is beta,
so runtime work must pin the dependency version and keep the adapter thin.

Required environment variables:

| Variable | Purpose |
| --- | --- |
| `CODELEWM_LLM_PROVIDER` | Must be `openrouter` for the first public adapter. |
| `OPENROUTER_API_KEY` | OpenRouter API key. Never print, commit, or summarize. |
| `CODELEWM_LLM_MODEL` | Model slug, defaulting to an Anthropic model through OpenRouter. |
| `CODELEWM_LLM_MAX_CANDIDATES` | Number of candidates requested from the LLM. |
| `CODELEWM_LLM_TIMEOUT_SECONDS` | Per-request timeout. |
| `CODELEWM_LLM_TEMPERATURE` | Generation temperature. |
| `CODELEWM_LLM_DRY_RUN` | Fixture mode that never calls the network. |

The OpenRouter adapter must not read a raw `ANTHROPIC_API_KEY`. To use
Anthropic models through the public adapter, set `CODELEWM_LLM_MODEL` to an
Anthropic OpenRouter slug and authenticate with `OPENROUTER_API_KEY`. If a raw
Anthropic provider key is required, configure it as OpenRouter BYOK outside the
repo or open a separate direct-Anthropic adapter issue. Do not silently mix the
two auth modes.

## Candidate Pack

LLM outputs are untrusted inputs and must be stored before scoring.

Schema: `codelewm.llm_candidate_pack.v1`.

Required fields:

- `schema_version`;
- `task_id`;
- `task_prompt_hash`;
- `context_hash`;
- `generator.provider`;
- `generator.model`;
- `generator.sdk`;
- `generator.sdk_version`;
- `generation_config`;
- `candidates[]`;
- `created_at`;
- `artifact_manifest`.

Each candidate records:

- stable `candidate_id`;
- raw patch text or after-state path;
- parser status;
- structured generation errors;
- token/count metadata when available;
- candidate content checksum;
- redaction and secret-scan status.

The candidate pack must never store API keys. Prompts, completions, patches, and
reports must pass `codelewm secret-scan` before publication.

## Demo Report

Schema: `codelewm.harness.demo_report.v1`.

The demo report composes:

- task prompt and context provenance;
- candidate pack manifest;
- CodeLeWM checkpoint manifest;
- optional transition index manifest;
- LLM original order;
- CodeLeWM score/rerank order;
- lexical, random, and no-action baselines when available;
- candidate parser and patch-application errors;
- optional static/test check metadata;
- claim gate.

The claim gate defaults to `allowed=false` unless a benchmark report, not a demo
report, proves the configured downstream success criteria.

## Downstream Benchmark

Schema: `codelewm.downstream_rerank_benchmark.v1`.
Report schema: `codelewm.downstream_rerank_report.v1`.

Each example must contain:

- task prompt;
- before-state;
- candidate patches or after-states;
- labels or check outcomes;
- candidate source metadata;
- split assignment;
- source/license policy;
- leak checks.

Minimum scaled evidence:

- at least 100 labeled examples;
- explicit train/validation/test or evaluation-only policy;
- random, lexical, no-action, LLM-order, CodeLeWM energy, retrieval-prior, and
  supported ensemble baselines;
- pass@1, pass@k, MRR, valid-patch rate, static/test check-pass rate, and
  calibration slices;
- bootstrap or confidence intervals when sample count supports them.

Usefulness claims remain blocked unless CodeLeWM improves over both no-action
and LLM-order baselines on the agreed headline metrics.

## Security Boundary

The default harness does not execute candidate code. Candidate parsing and patch
application run as static transformations over isolated files. If a future issue
adds test execution, it must be opt-in, run in a disposable checkout, and record
the command allowlist, timeout, environment, and failure mode in the report.

Remote LLM calls can receive repository context. The context builder must be
explicit about included files, truncation, ignored paths, and secret scanning.
No `.env`, token, credential, or private artifact path may be included.

## Publication Boundary

The preliminary public story is:

- validated: dataset/training/evaluation/HF publication/downloaded-artifact
  pipeline;
- invalidated: the tested action-use interventions as positive model-quality
  claims;
- unsupported: named semantic latent axes and scaled downstream usefulness;
- next: LLM-generated candidate reranking as the falsifiable downstream test.

Publication docs may show the harness workflow and public artifacts. They must
not imply that CodeLeWM improves coding until the downstream benchmark gate
passes.
