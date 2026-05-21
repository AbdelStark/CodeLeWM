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

The public usability pass for BYOK and the local demo task is #206. The first
live artifact direction (#207/#208) was superseded after the terminal demo
proved the toy comment task was too weak for a public showcase. The current
open follow-up streams are the meaningful harness demo (#224 through #231),
visual model observability and TUI harness (#235 through #245), scaled
downstream benchmarking (#209/#210/#211), and the next positive-model research
hypothesis (#212, with CWM comparison in #178).

Each stream must land as one issue per branch and PR. The demo stream may run
before the benchmark stream proves usefulness, but it must emit claim-safe
reports.

## OpenRouter Contract

The public LLM integration path uses the OpenRouter Python SDK. The SDK is beta,
so runtime work must pin the dependency version and keep the adapter thin.

Dependency policy for #187:

- add OpenRouter as an optional LLM/runtime dependency, not a base install
  dependency;
- pin the first supported SDK as `openrouter==0.9.1`, the latest release
  observed when this contract was written on 2026-05-21;
- record `openrouter` package version, Python version, and adapter version in
  every live candidate pack;
- require a small adapter-compatibility test before changing the pin.

Required environment variables:

| Variable | Purpose |
| --- | --- |
| `CODELEWM_LLM_PROVIDER` | Must be `openrouter` for the first public adapter. |
| `OPENROUTER_API_KEY` | OpenRouter API key. Never print, commit, or summarize. |
| `OPENROUTER_MANAGEMENT_KEY` | OpenRouter management key for administrative BYOK registration only. Never print, commit, or summarize. |
| `CODELEWM_LLM_MODEL` | Model slug, defaulting to an Anthropic model through OpenRouter. |
| `CODELEWM_LLM_MAX_CANDIDATES` | Number of candidates requested from the LLM. |
| `CODELEWM_LLM_TIMEOUT_SECONDS` | Per-request timeout. |
| `CODELEWM_LLM_TEMPERATURE` | Generation temperature. |
| `CODELEWM_LLM_DRY_RUN` | Fixture mode that never calls the network. |

Optional environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_HTTP_REFERER` | Optional OpenRouter attribution header. |
| `OPENROUTER_APP_TITLE` | Optional OpenRouter attribution title. |
| `CODELEWM_LLM_PROVIDER_OPTIONS_JSON` | JSON object forwarded as OpenRouter provider routing options. |
| `CODELEWM_LLM_RETRY_LIMIT` | Bounded retry count for retryable provider failures. |

When `CODELEWM_LLM_PROVIDER_OPTIONS_JSON` includes `{"zdr": true}`, OpenRouter
routes only to endpoints with a Zero Data Retention policy. BYOK provider-only
routes can fail before generation when no matching ZDR endpoint exists; the
adapter must surface the provider error and remediation instead of returning a
generic provider failure.

OpenRouter BYOK variables:

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Optional raw Anthropic provider key used only by explicit BYOK registration. Never print, commit, or summarize. |
| `CODELEWM_OPENROUTER_BYOK` | Enables redacted Anthropic BYOK routing metadata for OpenRouter requests. |
| `CODELEWM_OPENROUTER_BYOK_PROVIDER` | Provider slug for the first BYOK path; currently `anthropic`. |
| `CODELEWM_OPENROUTER_BYOK_KEY_ENV` | Environment variable name that holds the raw provider key. |
| `CODELEWM_OPENROUTER_BYOK_MANAGEMENT_KEY_ENV` | Environment variable name that holds the OpenRouter management key for registration. |
| `CODELEWM_OPENROUTER_BYOK_REQUIRE` | Routes requests to the BYOK provider only and disables fallback when true. |
| `CODELEWM_OPENROUTER_BYOK_REGISTER` | Runs the BYOK registration helper before a live demo run when true. |
| `CODELEWM_OPENROUTER_BYOK_DRY_RUN` | Validates BYOK registration without sending provider secrets; set to `0` only for real registration. |
| `CODELEWM_OPENROUTER_BYOK_NAME` | Human-readable OpenRouter BYOK credential name. |
| `CODELEWM_OPENROUTER_BYOK_ALLOWED_MODELS` | Comma-separated BYOK model allowlist. |
| `CODELEWM_OPENROUTER_BYOK_WORKSPACE_ID` | Optional OpenRouter workspace UUID. |
| `CODELEWM_OPENROUTER_BYOK_IS_FALLBACK` | Registers the provider key as fallback capacity when true. |

The public chat path still authenticates with `OPENROUTER_API_KEY`. A raw
`ANTHROPIC_API_KEY` may be read only by the explicit BYOK registration helper
or when `CODELEWM_OPENROUTER_BYOK_REGISTER=1`; it must never be serialized into
candidate packs, logs, reports, manifests, or docs. Request metadata records
only redacted BYOK state such as `enabled`, provider slug, key env name,
management-key env name, allowlist, and whether a workspace id was set.

BYOK registration emits `codelewm.openrouter_byok_register.v1`. Non-dry-run
registration uses the OpenRouter management key named by
`CODELEWM_OPENROUTER_BYOK_MANAGEMENT_KEY_ENV`, sends the raw provider key only
to OpenRouter's BYOK API, and returns a redacted summary. If the credential
already exists in OpenRouter, set `CODELEWM_OPENROUTER_BYOK_REGISTER=0` and
keep `CODELEWM_OPENROUTER_BYOK=1`.

`OPENROUTER_DEBUG` must be treated as unsafe for publishable runs because SDK
debug logging may include request or response content. Live publishable runs
must either unset it or record that debug logging was disabled in the candidate
pack metadata.

## Request Contract

Schema: `codelewm.openrouter_candidate_request.v1`.

The adapter receives a request object rather than ad hoc CLI strings. Required
fields:

- `schema_version`;
- `task_id`;
- `instruction`;
- `context_bundle`;
- `prompt_template_id`;
- `model`;
- `max_candidates`;
- `timeout_seconds`;
- `temperature`;
- `provider_options`;
- `dry_run`;
- `output_policy`.

`context_bundle` must record included file paths, ignored paths, truncation
policy, byte counts, content checksums, and secret-scan status before any live
remote call. `.env`, token-bearing files, local checkpoint files, private HF
download roots, and ignored artifact directories are excluded by default.

`provider_options` is passed through to OpenRouter only after JSON validation.
Allowed keys for v0.3 are `order`, `only`, `sort`, `allow_fallbacks`,
`require_parameters`, and `zdr`. Unknown keys must fail validation instead of
being silently forwarded.

`output_policy` must request unified diffs unless the issue explicitly enables
after-state files. The prompt must instruct the model to return exactly
`max_candidates` candidates when possible, with a stable candidate identifier
and no prose outside the candidate blocks.

## Candidate Pack

LLM outputs are untrusted inputs and must be stored before scoring.

Schema: `codelewm.llm_candidate_pack.v1`.

Required fields:

- `schema_version`;
- `task_id`;
- `prompt`;
- `context_hash`;
- `generator.provider`;
- `generator.model`;
- `generator.sdk`;
- `generator.sdk_version`;
- `generator.adapter_version`;
- `provider_routing`;
- `generation_config`;
- `candidates[]`;
- `errors[]`;
- `created_at`;
- `artifact_manifest`.

The `prompt` object records `template_id`, `rendered_sha256`,
`redacted_prompt_path`, `prompt_preview`, and `secret_scan`. Raw prompt text may
be stored only in a local artifact path that passes secret scanning. Published
candidate packs may include the redacted prompt path or preview, not unscanned
raw prompt content.

The `provider_routing` object records requested model slug, requested provider
ordering, fallback policy, zero-data-retention request flag when configured, and
provider response metadata available from the SDK. If the SDK does not expose
the final provider, the value must be `null` with a warning rather than inferred.

Each candidate records:

- stable `candidate_id`;
- raw patch text or after-state path;
- normalized patch checksum;
- parser status;
- dry-run patch-application status;
- structured generation errors;
- provider finish reason when available;
- token/count metadata when available;
- candidate content checksum;
- redaction and secret-scan status.

The candidate pack must never store API keys. Prompts, completions, patches, and
reports must pass `codelewm secret-scan` before publication.

Materialized candidate-pack artifacts use artifact kind `candidate_pack`. They
write `candidate_pack.json`, a redacted prompt file, candidate patch files, and
`manifest.json`. Invalid or oversized candidates are kept as structured
candidate errors and remain rankable after valid candidates; they must not abort
the whole pack.

## Dry-Run And Failure Behavior

`CODELEWM_LLM_DRY_RUN=1` remains the default fixture path. Dry-run mode must not
import the OpenRouter SDK, read `OPENROUTER_API_KEY`, or make network calls. It
loads deterministic fixture responses and still writes the same request,
candidate-pack, manifest, and secret-scan surfaces as live mode.

Live mode failure behavior:

- missing `OPENROUTER_API_KEY`: typed configuration error, no fallback to another
  provider key;
- invalid model slug: typed provider-request error recorded in the candidate
  pack and command summary;
- timeout: typed timeout error with elapsed time and retry count;
- rate limit or transient provider failure: retry only up to
  `CODELEWM_LLM_RETRY_LIMIT`;
- malformed LLM output: candidate-level parse errors, not whole-run success;
- fewer than requested candidates: report the count and block demo success;
- zero valid candidates: demo report `success=false` and claim gate
  `allowed=false`.

No failure path may print token values, raw `.env` contents, full unredacted
prompts, or full candidate content in logs.

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

When the demo is configured with a package-native torch checkpoint, the
world-model scorer must deserialize the trusted checkpoint after the checkpoint
manifest gate and report `codelewm.torch_transition_scorer.v1`. Fixture hashing
scorers are allowed only for explicit fixture tests or local smoke paths that do
not claim learned-model inference.

Demo success requires:

- at least two generated candidates;
- at least one valid parseable/applicable candidate;
- successful CodeLeWM score/rerank execution;
- manifest verification over the candidate pack and demo report;
- secret scan with zero findings over publishable artifacts.

The demo artifact must also include `demo.html`, a self-contained visual report
that makes dry-run versus live mode, candidate patches, CodeLeWM ranking,
baseline orders, and the claim gate visible without reading JSON by hand.
The local `scripts/llm-world-model-demo` entry point should default to a visual
terminal walkthrough over the same JSON artifacts. Raw JSON command summaries
remain available through explicit non-interactive mode.

The #235 visual model observability and TUI follow-up extends the same report
state with model/checkpoint inspection links, latent matrix diagnostics, run
timeline artifacts, and optional Textual TUI rendering. The TUI uses the same
underlying JSON artifacts as terminal and HTML modes; interactive UI support
does not replace non-interactive JSON output or require Textual for normal CLI
execution.

## Meaningful Demo Scenarios

The v1.3 harness demo must move beyond comment-preserving toy edits. The
default scenario should be a small public-safe code task that is meaningful to
inspect as code, such as a bug fix, edge-case handling change, API behavior
adjustment, or behavior-preserving refactor. Documentation/comment-only edits
are allowed only for scenarios explicitly about documentation.

Scenario metadata must be manifestable and include:

- stable scenario id and title;
- task instruction;
- included context paths and exclusion policy;
- before-state files or fixture root;
- expected static constraints, such as touched files or symbols;
- optional check command id when the scenario supports sandboxed checks;
- prompt template id;
- publication and claim-boundary notes.

The demo script must support selecting a scenario through CLI and environment
configuration while preserving deterministic dry-run fixtures. Dry-run mode must
not require OpenRouter, network access, provider keys, or sandbox execution.

Candidate generation for meaningful scenarios should ask for diverse,
task-solving unified diffs. The prompt should reject comment-only, whitespace
only, and no-op candidates unless the scenario explicitly permits them.

Candidate packs and demo reports should include static patch analysis for each
candidate: changed files, hunk counts, parse/apply status, touched symbols when
available, AST-level edit hints when available, and risk flags. Static analysis
must follow the same non-execution boundary as scoring. Invalid candidates are
kept as structured errors and remain visible in the report.

Terminal and HTML reports should expose enough of the meaningful task to be
understandable without opening raw JSON: scenario title, task instruction
summary, candidate summaries, compact diff previews or changed-symbol
summaries, CodeLeWM score/rank, score direction, candidate-minus-no-action
delta interpretation, scorer backend/checkpoint metadata, optional check
metadata, artifact gates, and claim gate. Current CodeLeWM transition scores
are lower-is-better energies, so a positive candidate-minus-no-action delta
means the no-action baseline scored better.
The non-interactive renderer contract is
`codelewm.harness.visual_view_model.v1`: an ANSI-free JSON view model that
normalizes summary state, generator metadata, candidate rank rows, compact diff
statistics, diagnostic slots, artifact gates, warnings, and the claim gate for
terminal, HTML, raw JSON, and optional Textual TUI rendering. The optional TUI
also exposes a deterministic `codelewm.harness.demo_tui_snapshot.v1` snapshot
for headless tests.

Demo failure is not a model failure. The report must distinguish provider
errors, malformed candidate outputs, invalid candidate patches, score/rerank
errors, and claim-gate failures.

The claim gate defaults to `allowed=false` unless a benchmark report, not a demo
report, proves the configured downstream success criteria. A demo may be
published as workflow evidence only.

## Downstream Benchmark

Schema: `codelewm.downstream_rerank_benchmark.v1`.
Report schema: `codelewm.downstream_rerank_report.v1`.
Pack config schema: `codelewm.downstream_rerank_benchmark_config.v1`.
Readiness schema: `codelewm.downstream_benchmark_readiness.v1`.
Eval run schema: `codelewm.downstream_rerank_eval_run.v1`.

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

The first public-safe fixture pack is built with:

```bash
uv run codelewm eval downstream-pack \
  --config config/benchmark/downstream_rerank_fixture.json \
  --out .artifacts/downstream-rerank-fixture \
  --json
```

It is a claim-blocked smoke artifact. It records source/license policy, split
leakage checks, checksums, and a secret-scan report, but it has one labeled task
and therefore cannot pass the 100-example gate.

The downstream comparison is run with:

```bash
uv run codelewm eval downstream-rerank \
  --benchmark-manifest .artifacts/downstream-rerank-fixture/manifest.json \
  --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt \
  --out .artifacts/downstream-rerank-report \
  --json
```

It consumes the benchmark manifest and optional candidate-pack manifests,
compares the required baselines, reports confidence intervals when the sample
count permits, records baseline availability status, and writes the same claim
gate used for public wording. Retrieval-prior baselines are marked blocked
unless an index produces finite retrieval-prior scores.

## Security Boundary

The default harness does not execute candidate code. Candidate parsing and patch
application run as static transformations over isolated files. If a future issue
adds test execution, it must be opt-in, run in a disposable checkout, and record
the command allowlist, timeout, environment, and failure mode in the report.
The v1.3 sandbox-check issue (#230) is the only planned execution exception.
It must keep execution disabled by default and must reject arbitrary user
commands; scenarios may refer only to allowlisted command ids.

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
