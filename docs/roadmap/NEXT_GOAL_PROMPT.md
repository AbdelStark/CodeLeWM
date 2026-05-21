# Next Goal Prompt

Use this prompt only if opening a new post-#206 research issue. The v0.2 HF
execution prompt in `docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md` is historical
context for the completed negative v0.2 sweep. The #186 through #194 harness,
publication, and downstream benchmark stream is complete, and #206 completed
the BYOK/local-demo/readme usability pass. Issues #220 and #222 made the local
demo learned-scorer-backed and terminal-first. Issues #207/#208 are closed as
superseded by the meaningful harness demo stream.

```text
/goal Continue CodeLeWM from the completed negative v0.2 evidence boundary.
Open or select a new issue from the updated backlog before making code changes:
#226 through #231 for the meaningful LLM + world-model harness demo, #210/#211
for the scaled downstream reranking benchmark, or #178/#212 for the next
positive-model research hypothesis. Work one issue per branch and PR.

Ground in AGENTS.md, SPEC.md, docs/spec/11-llm-world-model-harness.md,
docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md,
docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md,
docs/roadmap/MEANINGFUL_HARNESS_DEMO.md,
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
fixture demo, BYOK registration helper, local `uv run scripts/llm-world-model-demo`
task, downstream schema/claim-gate contract, public-safe downstream benchmark
pack, and downstream rerank report are complete through #206. Issue #220 makes
the local demo use a trusted learned torch checkpoint scorer instead of the
deterministic hashing fixture scorer; #222 makes the local demo terminal-first
by default while preserving explicit raw JSON mode. The comment-style live
artifact path #207/#208 is closed as superseded. The next harness queue is
#224: #226 scenario fixtures and selection, #227 task-solving prompts, #228
static patch analysis, #229 scorer traces and compact diff previews, #230
opt-in sandbox checks, and #231 live meaningful diagnostic artifacts. The
public LLM adapter uses the OpenRouter Python SDK with OPENROUTER_API_KEY and
model slugs
such as anthropic/claude-4.5-sonnet. Anthropic BYOK is explicit: only
`codelewm openrouter byok-register` or
`CODELEWM_OPENROUTER_BYOK_REGISTER=1` may read `ANTHROPIC_API_KEY`. BYOK
registration requires an OpenRouter management key such as
`OPENROUTER_MANAGEMENT_KEY`; normal chat requests still use
`OPENROUTER_API_KEY`. No reports may serialize raw provider keys.

No active completion issue remains for the current public evidence boundary.
The open streams are #224/#226-#231, #209/#210/#211, and #212.

For benchmark work, keep fixture/dry-run mode available so local validation does
not require network or paid LLM calls. Any live OpenRouter mode must redact
secrets, record SDK/model/provider metadata, and write manifest-backed candidate
packs.

Public docs must stay artifact-backed. The harness demo can show workflow
value, but it must not claim CodeLeWM improves coding until the downstream
benchmark gate passes on at least 100 labeled examples from manifest-backed
artifacts. The current completed boundary is explicitly negative/diagnostic.
For meaningful demo work, the default scenario should be a public-safe bug fix,
edge-case handling task, API behavior change, or behavior-preserving refactor,
not a comment/no-op edit. Candidate code remains untrusted; sandbox checks are
disabled unless #230's explicit allowlist/timeout/disposable-checkout contract
is implemented and selected.

After each issue, run the strongest relevant local validation, commit, push,
open a PR, wait for available checks, merge when clean, return to main, pull
latest main, and continue.
```
