# RFC-0013: LLM + World-Model Harness And Publication

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-21
- Target milestone: v1.1

## Summary

After the completed negative v0.2 action-use sweep, CodeLeWM's next credible
step is not another blind training run. The next step is a harness that lets an
LLM generate candidate patches and lets CodeLeWM score/rerank those candidates
under a reproducible downstream evaluation contract.

The harness demo is allowed to be public before it proves model usefulness. The
benchmark and publication streams decide what claims are allowed.

## Motivation

The current evidence validates a serious systems path: dataset construction,
HF Jobs training, artifact publication, clean downloads, manifest verification,
and local reruns all work. The current evidence also invalidates the tested
positive action-use hypotheses. Text-action still loses to no-action on the
headline retrieval metrics, latent probes do not support semantic-axis claims,
and downstream reranking has only a one-example smoke path.

A downstream candidate-reranking harness tests the most relevant product
question:

Can a code-edit world model add value when an LLM already generated plausible
candidate patches?

## Goals

- Use the OpenRouter Python SDK as the public LLM adapter so users can choose
  model slugs across providers.
- Generate candidate patches or after-states from task prompts and bounded
  repository context.
- Store LLM outputs as untrusted, manifest-backed candidate packs.
- Score and rerank candidates with CodeLeWM without executing candidate code.
- Emit a demo report that compares LLM order, CodeLeWM order, no-action,
  lexical, random, and retrieval-prior baselines where available.
- Define a downstream benchmark that can validate or falsify coding-usefulness
  claims.
- Publish preliminary results as negative/diagnostic evidence with strict claim
  boundaries.

## Non-Goals

- Claim that the current v0.2 checkpoint improves coding.
- Treat a demo report as a benchmark result.
- Execute candidate code by default.
- Send secrets, `.env` files, private artifacts, or token-bearing context to an
  LLM provider.
- Add a broad agent framework before the candidate-pack and reranking contracts
  are stable.

## Design

The first public flow is:

```text
task + bounded context
  -> OpenRouter candidate generator
  -> codelewm.llm_candidate_pack.v1
  -> CodeLeWM score/rerank
  -> optional static/check metadata
  -> codelewm.harness.demo_report.v1
```

The benchmark flow extends it:

```text
benchmark tasks + labels
  -> candidate packs
  -> baseline and CodeLeWM reranking
  -> codelewm.downstream_rerank_report.v1
  -> claim gate
```

The publication flow summarizes:

- what the HF runs validated;
- what the HF runs invalidated;
- which claims remain blocked;
- what the LLM + world-model harness will test next.

## OpenRouter Adapter

The public adapter uses `openrouter.OpenRouter` with `OPENROUTER_API_KEY`.
The default model slug should target an Anthropic model through OpenRouter, for
example `anthropic/claude-4.5-sonnet` or a documented newer slug when the issue
is implemented.

The runtime issue must add OpenRouter as an optional dependency pinned to
`openrouter==0.9.1`, the latest release observed when #186 locked this contract
on 2026-05-21. Because the SDK is beta, changing that pin requires updating this
RFC, the spec, and a fixture compatibility test in the same PR.

The adapter must support:

- dry-run fixture mode;
- model slug selection;
- max candidate count;
- timeout;
- temperature;
- provider options;
- structured retries;
- redacted logs.
- explicit Anthropic BYOK routing and registration through OpenRouter when
  requested by local environment variables.

The OpenRouter SDK is beta and may introduce breaking changes before a major
version bump. Runtime implementation must pin the SDK version and record it in
candidate-pack artifacts.

Direct provider chat adapters remain out of scope. If local experiments use an
Anthropic provider key, it must be configured through the explicit OpenRouter
BYOK registration helper. Chat requests still authenticate to OpenRouter with
`OPENROUTER_API_KEY`; BYOK registration uses an OpenRouter management key such
as `OPENROUTER_MANAGEMENT_KEY`. Candidate-pack metadata must record only
redacted BYOK state and env-var names, never raw keys.

OpenRouter debug logging is out of scope for publishable runs. The adapter may
support local debug mode later, but any enabled debug mode must be rejected by
publication gates unless request and response content are separately redacted
and secret-scanned.

## Artifact Schemas

New schemas:

- `codelewm.openrouter_candidate_request.v1`
- `codelewm.openrouter_byok_register.v1`
- `codelewm.llm_candidate_pack.v1`
- `codelewm.harness.demo_report.v1`
- `codelewm.downstream_rerank_benchmark.v1`
- `codelewm.downstream_rerank_report.v1`

All runtime artifacts must be JSON-native, schema-versioned, manifest-backed, checksum
verifiable, and secret-scanned before publication.

The candidate pack must record prompt metadata, model slug, provider routing
request and response metadata, patch text or after-state path, parser status,
dry-run patch-application status, generation errors, token/count metadata when
available, and redaction/secret-scan status. API keys and raw provider secrets
must never appear in the pack.

The demo report must define success and failure separately from scientific
claims. Demo success proves only that candidate generation, capture, scoring,
reranking, manifests, and scans completed. Coding-usefulness claims require the
downstream benchmark gate.

## Milestones

Stream A, LLM + world-model harness demo:

- #183 tracks the stream.
- #186 locks the OpenRouter harness contract.
- #187 adds the OpenRouter candidate generator.
- #188 adds candidate-pack schema and safe patch capture.
- #189 builds the end-to-end demo report.
- #206 adds explicit OpenRouter BYOK registration, the local
  `uv run scripts/llm-world-model-demo` task, and public README polish.
- #207/#208 track one live, claim-safe OpenRouter BYOK harness artifact.

Stream B, downstream candidate-reranking benchmark:

- #184 tracks the stream.
- #190 defines benchmark schema and claim gates.
- #191 builds the public-safe labeled candidate set and claim-blocked fixture
  benchmark pack.
- #192 runs the downstream comparison and claim gate.
- #209/#210/#211 track the scaled 100-example benchmark and claim gate.

Stream C, preliminary results publication:

- #185 tracks the stream.
- #193 publishes the preliminary negative-results report.
- #194 prepares the artifact index and announcement package.

Stream D, next model hypothesis:

- #212 tracks the next positive-model research hypothesis.
- #178 remains the CWM comparison spike that can feed that hypothesis only if it
  yields reusable baselines, datasets, or evaluation criteria.

## Testing Strategy

- Contract tests for schema versions and JSON output.
- Fixture-mode tests that require no network access.
- Secret redaction tests for prompts, completions, and logs.
- Non-execution tests for malicious candidate patches.
- CLI/API tests for missing key, timeout, invalid model slug, invalid patch,
  and oversized output.
- End-to-end fixture demo test from candidate generation through reranking
  report.
- Benchmark report tests that block claims below 100 labeled examples.

## Rollout

1. Land docs/spec issue #186.
2. Land fixture-first runtime slices #187 and #188.
3. Land demo report #189.
4. Land benchmark contract #190 before collecting or publishing a labeled set.
5. Land evaluation issue #192 against the #191 benchmark pack.
6. Land publication issues #193 and #194 once every public claim links to
   artifact-backed evidence.

## References

- `docs/spec/11-llm-world-model-harness.md`
- `docs/spec/02-public-api.md`
- `docs/spec/06-security.md`
- `docs/rfcs/RFC-0008-agent-harness-scorer-reranker.md`
- `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`
- `docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md`
