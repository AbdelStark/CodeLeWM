# Next Goal Prompt

Use this prompt only if opening a new post-#192 research issue. The v0.2 HF
execution prompt in `docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md` is historical
context for the completed negative v0.2 sweep. The #186 through #194 harness,
publication, and downstream benchmark stream is complete.

```text
/goal Continue CodeLeWM from the completed negative v0.2 evidence boundary.
Open or select a new issue with a falsifiable post-#192 hypothesis before
making code changes. Work one issue per branch and PR.

Ground in AGENTS.md, SPEC.md, docs/spec/11-llm-world-model-harness.md,
docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md,
docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md,
docs/roadmap/FULL_COMPLETION.md, docs/roadmap/IMPLEMENTATION.md,
docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md,
docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md,
docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md, CONTRIBUTING.md,
and the new issue.

Do not relaunch #159 or #172. The current public artifact set is valid
negative/diagnostic evidence: v0.2 text-action reached Recall@1 0.263 and MRR
0.370048, while no-action reached Recall@1 0.441 and MRR 0.533105. Latent
probes and downstream gates also failed.

The OpenRouter LLM candidate harness contract, adapter, candidate-pack capture,
fixture demo, downstream schema/claim-gate contract, public-safe downstream
benchmark pack, and downstream rerank report are complete through #192. The public LLM adapter uses the
OpenRouter Python SDK with OPENROUTER_API_KEY and model slugs such as
anthropic/claude-4.5-sonnet. Do not silently read raw provider keys in the
OpenRouter adapter. If direct Anthropic API key support is required, open a
separate adapter issue or configure provider keys as OpenRouter BYOK outside the
repo.

No active completion issue remains for the current public evidence boundary.

For benchmark work, keep fixture/dry-run mode available so local validation does
not require network or paid LLM calls. Any live OpenRouter mode must redact
secrets, record SDK/model/provider metadata, and write manifest-backed candidate
packs.

Public docs must stay artifact-backed. The harness demo can show workflow
value, but it must not claim CodeLeWM improves coding until the downstream
benchmark gate passes on at least 100 labeled examples from manifest-backed
artifacts. The current completed boundary is explicitly negative/diagnostic.

After each issue, run the strongest relevant local validation, commit, push,
open a PR, wait for available checks, merge when clean, return to main, pull
latest main, and continue.
```
