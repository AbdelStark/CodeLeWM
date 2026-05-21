# Post-v0.2 Showcase Roadmap

Last updated: 2026-05-21

This roadmap turns the completed negative v0.2 result into the next public
milestone: a claim-safe LLM + world-model harness, a real downstream reranking
benchmark, and a preliminary publication package.

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

Tracker: #183. Status: complete after #189.

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

Success for the stream:

- deterministic fixture demo runs without network access;
- live demo can use `OPENROUTER_API_KEY` and an Anthropic OpenRouter model slug;
- the runtime adapter pins `openrouter==0.9.1` until a compatibility PR updates
  the contract;
- candidate packs and demo reports are manifest-backed and secret-scanned;
- public docs clearly state that the demo is not a model-quality proof.

## Stream B: Downstream Candidate-Reranking Benchmark

Tracker: #184. Status: Closed.

Purpose: decide whether the harness is useful beyond a demo. The benchmark must
compare CodeLeWM against LLM order, random, lexical, no-action, retrieval-prior,
and supported ensemble baselines.

Issues:

| Order | Issue | Title | Status |
| --- | --- | --- | --- |
| B1 | #190 | benchmark: define downstream task schema and claim gates | Closed |
| B2 | #191 | benchmark: build public-safe labeled candidate reranking set | Closed |
| B3 | #192 | eval: run downstream reranking comparison and claim gate | Closed |

Minimum scaled benchmark gate:

- at least 100 labeled examples;
- pass@1, pass@k, MRR, valid-patch rate, static/test check-pass rate;
- claim gate `allowed=false` unless CodeLeWM improves over no-action and
  LLM-order baselines on the agreed metrics.

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

Rationale: lock the contract first, publish the current result honestly, then
build the demo and only then spend effort on scaled downstream labels.

## `/goal` Prompt

```text
/goal Continue CodeLeWM from the completed negative v0.2 evidence boundary.
The #186 through #194 stream is complete. Start a new issue for any future
positive-claim research hypothesis.

Ground in AGENTS.md, SPEC.md, docs/spec/11-llm-world-model-harness.md,
docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md,
docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md,
docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md,
docs/benchmark/DOWNSTREAM_RERANKING_BENCHMARK.md,
docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md.

The harness and downstream benchmark stream is complete through #192. The
public LLM adapter uses the OpenRouter Python SDK with OPENROUTER_API_KEY and
model slugs such as anthropic/claude-4.5-sonnet. Do not silently read raw
provider keys in the OpenRouter adapter.

Keep the claim boundary explicit: the current v0.2 checkpoint and downstream
fixture report are public negative/diagnostic evidence. The harness demo can
show workflow value, but it must not claim CodeLeWM improves coding until a
future scaled downstream benchmark gate passes from manifest-backed artifacts.

Run the strongest local validation for any future issue, commit, push, open a
PR, wait for checks, merge when clean, return to main, and pull latest main.
```
