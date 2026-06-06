# Meaningful Harness Demo Roadmap

Last updated: 2026-06-06

Tracker: #224. Status: closed.

Related follow-up tracker: #235 for visual model observability, latent matrix
diagnostics, TensorBoard/checkpoint inspection, and optional Textual TUI work;
that follow-up tracker is also closed.

This roadmap records the completed v1.3 harness-demo upgrade. The demo proves the
system path: OpenRouter candidate generation, candidate-pack capture, learned
CodeLeWM scoring, manifest verification, secret scanning, terminal output, and
HTML reporting. Its weakness is the task itself: adding or preserving comments
does not make a compelling code-world-model demo.

The v1.3 target is a public-safe code-edit task that is meaningful to inspect:
a bug fix, edge-case handling change, API behavior adjustment, or
behavior-preserving refactor. The demo still remains diagnostic workflow
evidence. It does not create a positive coding-usefulness claim.

## Claim Boundary

Allowed:

- show an LLM generating multiple candidate patches for a realistic small code
  task;
- show CodeLeWM scoring and reranking those candidates;
- show no-action deltas, static patch summaries, optional check metadata, and
  artifact gates;
- publish manifest-backed live artifacts after secret scans.

Blocked:

- claiming CodeLeWM improves coding from a demo report;
- claiming semantic latent axes from candidate rankings;
- executing candidate code by default;
- publishing prompts, completions, reports, or check logs before secret scans.

Coding-usefulness claims still require the scaled downstream benchmark gate:
at least 100 labeled examples and CodeLeWM improvement over both no-action and
LLM-order baselines on the agreed headline metrics.

## Demo Shape

```text
scenario fixture
  -> bounded context bundle
  -> OpenRouter candidate generation
  -> candidate pack with static analysis
  -> learned CodeLeWM transition scoring
  -> optional sandbox check metadata
  -> terminal + HTML report
  -> manifest verify + secret scan
  -> claim gate
```

The default scenario should be easy to understand from a terminal report. A
good first target is a tiny single-file bug fix with an edge case, because it
lets the report show meaningful candidate differences without needing a large
repository context.

## Issue Backlog

| Order | Issue | Slice | Acceptance summary |
| --- | --- | --- | --- |
| 0 | #225 | Roadmap and agent context | Closed: specs, roadmap, tracker, prompt, and AGENTS.md point at this stream. |
| 1 | #226 | Scenarios and selector | Closed: demo supports scenario ids and defaults to `bugfix-edge-case`. |
| 2 | #227 | Task-solving prompts | Closed: prompt template asks for diverse unified diffs that solve the scenario. |
| 3 | #228 | Static patch analysis | Closed: candidate packs and reports summarize changed files, hunks, symbols, parse/apply status, and risk flags. |
| 4 | #229 | Scorer traces and previews | Closed: terminal and HTML reports show scenario summary, compact diff previews, score/no-action deltas, and scorer metadata. |
| 5 | #230 | Opt-in sandbox checks | Closed: scenario-owned allowlisted checks can run in disposable environments with timeout, redaction, manifests, and secret scans. |
| 6 | #231 | Live artifact publication | Closed: a live OpenRouter/BYOK meaningful demo artifact set is verified and published as diagnostic evidence. |

## Scenario Contract

Scenario metadata should be JSON-native and include:

- `scenario_id`;
- `title`;
- `instruction`;
- `context_paths`;
- `before_files` or fixture root;
- `prompt_template_id`;
- expected static constraints, such as changed file or symbol names;
- optional `check_command_id`;
- publication caveats.

The script should support:

```bash
uv run --group data --group train --group llm scripts/llm-world-model-demo --scenario bugfix-edge-case
CODELEWM_LLM_DEMO_SCENARIO=bugfix-edge-case uv run --group data --group train --group llm scripts/llm-world-model-demo
```

Raw machine-readable mode remains:

```bash
uv run scripts/llm-world-model-demo --json
```

## Security Requirements

Candidate code, scenario context, LLM completions, prompts, and check outputs
are untrusted. The default path must continue to parse and transform text only.

Sandbox checks are allowed only after #230 and only when explicitly enabled.
They must:

- use scenario-owned allowlisted command ids, not arbitrary command strings;
- run in a disposable checkout or temporary directory;
- scrub environment variables;
- enforce timeout and output limits;
- record a schema-versioned check report;
- secret-scan every publishable output.

## Success Signal

The stream completed when #231 published one live artifact set whose terminal
and HTML reports let a reader answer:

- What code task was attempted?
- What did each candidate change?
- Which candidate did CodeLeWM score highest?
- How did that compare with no-action and LLM order?
- Did static checks and optional sandbox checks pass?
- Which artifacts and manifests prove the run?
- Why does the claim gate remain closed?

The broader visual observability work in #235 reused this scenario-driven path's
candidate packs, static summaries, scorer traces, and claim gates rather than
creating a second incompatible demo surface.
