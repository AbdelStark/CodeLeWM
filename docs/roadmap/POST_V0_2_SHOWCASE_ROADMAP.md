# Post-v0.2 Showcase Roadmap

Last updated: 2026-05-21

This roadmap turns the completed negative v0.2 result into the next public
milestone: a claim-safe LLM + world-model harness, a real downstream reranking
benchmark, and a preliminary publication package. As of the BYOK/demo polish
pass, it also defines the next three open streams: live harness evidence,
scaled downstream benchmarking, and the next positive-model research
hypothesis.

The current evidence boundary is explicit:

- validated: CodeLeWM can build public-safe datasets, train on HF Jobs, publish
  artifacts, download them, verify manifests, rerun evals, and produce cards;
- invalidated: the tested action-use interventions as positive model-quality
  claims;
- unsupported: named semantic latent axes and scaled downstream
  coding-usefulness claims;
- next: test whether CodeLeWM adds value when an LLM supplies candidate
  patches.

## Stream A: LLM + World-Model Harness Demo

Tracker: #183. Status: complete after #189 and the BYOK/readme/demo polish in
#206.

Purpose: show the intended product use case without overstating the model. An
LLM generates candidate patches through the OpenRouter Python SDK. CodeLeWM
scores/reranks those candidates. The report shows the model score, baselines,
candidate errors, and optional checks.

Issues:

| Order | Issue | Title | Status |
| --- | --- | --- | --- |
| A1 | #186 | spec: lock OpenRouter LLM candidate harness contract | Closed |
| A2 | #187 | harness: add OpenRouter candidate generation adapter | Closed |
| A3 | #188 | harness: add candidate pack schema and safe patch capture | Closed |
| A4 | #189 | harness: build end-to-end LLM plus CodeLeWM demo report | Closed |
| A5 | #206 | harness: add OpenRouter BYOK demo task and public README polish | Closed after merge |

Success for the stream:

- deterministic fixture demo runs without network access;
- live demo can use `OPENROUTER_API_KEY` and an Anthropic OpenRouter model slug;
- Anthropic BYOK mode is explicit, redacted, and optional through OpenRouter;
- `uv run scripts/llm-world-model-demo` exercises the full local fixture demo,
  manifest verification, and secret scan;
- the runtime adapter pins `openrouter==0.9.1` until a compatibility PR updates
  the contract;
- candidate packs and demo reports are manifest-backed and secret-scanned;
- public docs clearly state that the demo is not a model-quality proof.

## Stream D: Live Harness Evidence

Tracker: #207. Status: Open.

Purpose: turn the fixture-proven harness into one live, publishable diagnostic
artifact set without changing the claim boundary.

Issues:

| Order | Issue | Title | Status |
| --- | --- | --- | --- |
| D1 | #208 | run: execute live OpenRouter BYOK harness demo and publish diagnostic artifacts | Open |
| D2 | #220 | harness: use learned world-model inference in LLM demo | Open |

Success for the stream:

- `CODELEWM_LLM_DRY_RUN=0 uv run scripts/llm-world-model-demo` runs with
  `OPENROUTER_API_KEY` and explicit BYOK settings where desired;
- demo reports show `codelewm.torch_transition_scorer.v1` from a trusted
  package-native checkpoint, not the deterministic hashing fixture scorer;
- candidate-pack and demo-report artifacts pass manifest verification and
  `codelewm secret-scan`;
- public docs record the live run as a workflow artifact, not evidence that
  CodeLeWM improves generated code.

## Stream B: Downstream Candidate-Reranking Benchmark

Tracker: #184 for the initial fixture. New scaled tracker: #209. Status: Open
for scaled work.

Purpose: decide whether the harness is useful beyond a demo. The benchmark must
compare CodeLeWM against LLM order, random, lexical, no-action, retrieval-prior,
and supported ensemble baselines.

Issues:

| Order | Issue | Title | Status |
| --- | --- | --- | --- |
| B1 | #190 | benchmark: define downstream task schema and claim gates | Closed |
| B2 | #191 | benchmark: build public-safe labeled candidate reranking set | Closed |
| B3 | #192 | eval: run downstream reranking comparison and claim gate | Closed |
| B4 | #210 | data: build public-safe 100-example downstream reranking set | Open |
| B5 | #211 | eval: run scaled downstream reranking comparison and claim gate | Open |

Minimum scaled benchmark gate:

- at least 100 labeled examples;
- pass@1, pass@k, MRR, valid-patch rate, static/test check-pass rate;
- claim gate `allowed=false` unless CodeLeWM improves over no-action and
  LLM-order baselines on the agreed metrics.

## Stream E: Next Positive-Model Research Hypothesis

Tracker: #212. Status: Open.

Purpose: prevent another blind training sweep. Any future positive model-quality
claim needs a falsifiable hypothesis, an explicit config/data intervention, and
the same download, eval, secret-scan, checkpoint-trust, and claim-review gates
used by the v0.2 HF runs.

Related issue:

| Order | Issue | Title | Status |
| --- | --- | --- | --- |
| E1 | #178 | Spike: Evaluate Meta AI's CWM for comparison and benchmark reuse | Open |

Success for the stream:

- a new research issue states the hypothesis, expected failure modes, compute
  plan, baselines, and public claim gate before training launches;
- CWM comparison work is used only if it produces reusable baselines, datasets,
  or evaluation criteria;
- public docs stay negative/diagnostic until a new gate passes.

## Stream C: Preliminary Results Publication

Tracker: #185.

Status: complete as of #194.

Purpose: publish the project in a way that is useful and honest. The right story
is an infrastructure and negative-result milestone, plus a clear next
downstream test.

Issues:

| Order | Issue | Title | Status |
| --- | --- | --- | --- |
| C1 | #193 | docs: publish preliminary negative-results report | Closed |
| C2 | #194 | docs: prepare public artifact index and announcement package | Closed |

Publication boundary:

- allowed: public HF artifact links, reports, cards, run IDs, metrics, negative
  action-use result, and the harness roadmap;
- blocked: positive action-conditioned quality, semantic latent-axis, or
  coding-usefulness claims.

Publication artifacts:

- `docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md`
- `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-21.md`
- `docs/announcements/PRELIMINARY_RESULTS_2026-05-21.md`

## Implementation Order

Recommended order:

1. #186
2. #193
3. #194
4. #187
5. #188
6. #189
7. #190
8. #191
9. #192
10. #206
11. #220
12. #208
13. #210
14. #211
15. #178 / #212

Rationale: lock the contract first, publish the current result honestly, build
the demo, prove one live workflow artifact, and only then spend effort on
scaled downstream labels or a new model hypothesis.

## `/goal` Prompt

```text
/goal Continue CodeLeWM from the completed negative v0.2 evidence boundary.
The #186 through #194 stream is complete, and #206 added the public BYOK/local
demo/readme polish. Select one open stream before making changes:
#220/#208 for live OpenRouter BYOK harness evidence, #210/#211 for scaled
downstream reranking, or #178/#212 for the next positive-model research
hypothesis.

Ground in AGENTS.md, SPEC.md, docs/spec/11-llm-world-model-harness.md,
docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md,
docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md,
docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md,
docs/benchmark/DOWNSTREAM_RERANKING_BENCHMARK.md,
docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md.

The harness and downstream benchmark stream is complete through #206. The
public LLM adapter uses the OpenRouter Python SDK with OPENROUTER_API_KEY and
model slugs such as anthropic/claude-4.5-sonnet. Anthropic BYOK is explicit:
only `codelewm openrouter byok-register` or
`CODELEWM_OPENROUTER_BYOK_REGISTER=1` may read the raw provider key. BYOK
registration requires an OpenRouter management key such as
`OPENROUTER_MANAGEMENT_KEY`; normal chat requests still use `OPENROUTER_API_KEY`.
No reports may serialize raw keys.

Keep the claim boundary explicit: the current v0.2 checkpoint and downstream
fixture report are public negative/diagnostic evidence. The harness demo can
show workflow value, but it must not claim CodeLeWM improves coding until a
future scaled downstream benchmark gate passes from manifest-backed artifacts.

Run the strongest local validation for any future issue, commit, push, open a
PR, wait for checks, merge when clean, return to main, and pull latest main.
```
