# Next Goal Prompt

Use this prompt for the active v0.9 data/eval repair epic (#385). The v0.2 HF
execution prompt in `docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md` is historical
context for the completed negative v0.2 sweep. The #186 through #194 harness,
publication, and downstream benchmark stream is complete, #206 completed the
BYOK/local-demo/readme usability pass, and #220/#222 made the local demo
learned-scorer-backed and terminal-first. The v0.8 execution-trace result is
published as diagnostic evidence; the next active queue is #386 through #392.

```text
/goal Continue CodeLeWM from the completed v0.8 execution-trace evidence
boundary. Use the active v0.9 epic tracker #385 and work one child issue per
branch and PR in dependency order: #386 roadmap/tracker hygiene, #387
cross-benchmark pass/fail execution data, #388 held-out p_pass AUC and
calibration reports, #389 semantic-decoy coverage repair, #390 probe-label and
representation gates, #391 guarded two-seed HF Jobs run, and #392 final
gate-suite report and claim audit.

Ground in AGENTS.md, SPEC.md, docs/spec/05-observability.md,
docs/spec/06-security.md, docs/spec/11-llm-world-model-harness.md,
docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md,
docs/rfcs/RFC-0015-v0-7-execution-substrate-improvements.md,
docs/benchmark/EXECUTION_V0_8_RESULTS_2026-06-05.md,
docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-05.md,
docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md,
docs/roadmap/MEANINGFUL_HARNESS_DEMO.md,
docs/roadmap/MODEL_OBSERVABILITY_TUI_ROADMAP.md,
docs/roadmap/DIAGNOSTICS_DRIVEN_MODEL_EXPERIMENT.md,
docs/benchmark/VISUAL_OBSERVABILITY_ARTIFACTS_2026-05-21.md,
docs/roadmap/FULL_COMPLETION.md, docs/roadmap/IMPLEMENTATION.md,
docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md,
docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md,
docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md, CONTRIBUTING.md,
and the new issue.

Do not relaunch #159, #172, or the completed v0.8 jobs. The current public
artifact set is valid negative/diagnostic evidence. v0.8 completed two A10G
correctness-aware execution runs and passed HumanEval WS-D reranking on both
seeds, but MBPP-Plus, latent-probe, magnitude-label, and broad semantic-decoy
gates keep the overall cross-benchmark claim closed.

The OpenRouter LLM candidate harness contract, adapter, candidate-pack capture,
fixture demo, BYOK registration helper, local `uv run scripts/llm-world-model-demo`
task, downstream schema/claim-gate contract, public-safe downstream benchmark
pack, and downstream rerank report are complete through #206. Issue #220 makes
the local demo use a trusted learned torch checkpoint scorer instead of the
deterministic hashing fixture scorer; #222 makes the local demo terminal-first
by default while preserving explicit raw JSON mode. The comment-style live
artifact path #207/#208 is closed as superseded. The next harness queue is
#224: #226 scenario fixtures and selection is complete, #227 task-solving prompts, #228
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

The #235 visual model observability and TUI tracker is complete through #245.
#237 closed optional TensorBoard-compatible model-generation traces for
training/checkpoint runs; #238 closed trusted checkpoint tensor/layer
inspection; #239 closed manifest-backed latent representation matrix
diagnostics; #240 closed manifest-backed run timelines; #242 closed the shared
JSON/rich/HTML visual view model; #241 closed optional Textual TUI mode and
deterministic TUI snapshots; #243 closed manifest-backed demo diagnostic links.
Issue #244 closed the diagnostics-driven candidate-contrast action training
plan. Issue #245 published the public visual observability artifact set. The
latest live meaningful demo worked end to end, but it also
showed the current scorer can rank an incomplete patch above semantically
stronger candidates; treat that as a diagnostic failure mode to inspect, not as
a positive model result.

Live issue state as of 2026-06-06: #178, #209, #210, #211, #212, #224, and
#235 are closed. The older meaningful-demo queue #224/#227-#231 is complete and
is not the current v0.9 epic path. The stale v0.7 tracker/issues #337 through
#341 are superseded by the completed v0.8 evidence and the v0.9 tracker #385.

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

For visual observability/TUI work, keep TensorBoard-compatible export and
Textual dependencies optional. #237 implemented the observability group and
`codelewm train --tensorboard`; #238 implemented
`codelewm model inspect-checkpoint`; base imports, fixture tests, JSON reports,
and non-interactive rich terminal output must keep working without optional groups.
Model, tensor, latent, and TUI artifacts remain diagnostic until scaled
representation and downstream benchmark gates pass.

For v0.9, do the cheap data/eval preflight before any GPU launch. #391 must not
start until #387 through #390 have produced manifest-backed data, held-out
correctness metrics, semantic-decoy coverage, and probe-label coverage reports
or typed blockers. #392 may publish a positive claim only if the final
cross-benchmark gate suite clears; otherwise publish the result as diagnostic
evidence with the exact closed gates.

After each issue, run the strongest relevant local validation, commit, push,
open a PR, wait for available checks, merge when clean, return to main, pull
latest main, and continue.
```
