# Model Observability And TUI Roadmap

Last updated: 2026-05-21

Tracker: #235.

This roadmap defines the v1.4 visual model observability and TUI harness
stream. It follows the meaningful harness demo work in #224 and answers the
main gap exposed by the first live `bugfix-edge-case` demo: the workflow ran
end to end, but the learned scorer ranked an incomplete candidate above more
semantically complete patches. The next step is to make the model, latent space,
candidate structure, and run state inspectable enough to debug that behavior.

The stream adds TensorBoard-compatible model-generation traces, checkpoint
inspection reports, latent representation matrix diagnostics, richer run
timelines, and an optional Textual TUI. It keeps the current non-interactive
surfaces: JSON reports, rich terminal output, HTML reports, manifest
verification, secret scans, and closed claim gates.

## Claim Boundary

Allowed:

- show training scalars, tensor summaries, checkpoint metadata, and layer
  statistics as diagnostics;
- show latent dimension matrices, variance/effective-rank metrics, probe
  associations, and stability gates;
- show a Textual TUI, rich terminal report, HTML report, and JSON artifacts over
  the same run state;
- publish visual diagnostic artifacts after manifest verification and secret
  scans.

Blocked:

- claiming semantic latent axes from a single run, plot, or heatmap;
- claiming CodeLeWM improves generated code from a demo report;
- treating a TensorBoard trace or TUI as benchmark evidence;
- executing candidate code by default;
- adding Textual, TensorBoard, or visualization dependencies to the base
  install.

Positive coding-usefulness claims still require the scaled downstream benchmark
gate: CodeLeWM must improve over no-action and LLM-order baselines on at least
100 labeled examples from manifest-backed artifacts.

## Current Evidence

The latest live meaningful demo evidence is diagnostic:

- live OpenRouter generation worked with explicit Anthropic BYOK routing;
- the selected scenario was `bugfix-edge-case`;
- 4/4 candidates were parseable and patch-applicable;
- the trusted package-native torch scorer loaded checkpoint
  `49965cd15fb4...`;
- manifests and secret scans passed;
- the demo claim gate stayed closed.

The ranking itself is not positive evidence. Candidate 001 handled blank labels
but did not collapse repeated whitespace; candidates 002, 003, and 004 better
matched the scenario. The learned scorer still ranked candidate 001 first. This
is the failure mode v1.4 must make visible and actionable.

## Desired User Experience

```text
CodeLeWM visual harness
  -> select scenario or load existing run
  -> generate or load candidate pack
  -> inspect candidate diffs and static summaries
  -> inspect model/checkpoint/tensor summaries
  -> inspect latent matrix and probe gates
  -> inspect score traces and no-action deltas
  -> inspect run timeline, logs, manifests, and scans
  -> export JSON, HTML, terminal, and TUI views
  -> preserve claim gate
```

Interactive mode should use Textual. Non-interactive mode remains the default
automation surface and must keep:

- raw JSON output for scripts and CI;
- rich terminal output for local runs;
- self-contained HTML for shareable inspection;
- manifest-backed reports for reproducibility.

## Issue Backlog

| Order | Issue | Slice | Acceptance summary |
| --- | --- | --- | --- |
| 0 | #236 | Roadmap and tracker lock | Docs, AGENTS.md, tracker, issue backlog, and next prompt point at #235. |
| 1 | #237 | TensorBoard export | Closed: optional TensorBoard-compatible event logs for training/checkpoint scalars and bounded summaries. |
| 2 | #238 | Checkpoint inspection | Closed: schema-versioned model/layer/tensor report with trust gates and manifests. |
| 3 | #239 | Latent matrix diagnostics | Closed: dimension matrix, finite stats, effective rank, probe associations, bounded heatmap previews, and semantic-axis claim gates. |
| 4 | #240 | Run timeline and monitoring | Closed: structured run timeline artifacts and richer redacted monitoring logs. |
| 5 | #242 | Non-interactive report parity | Shared view model for JSON, rich terminal, and HTML outputs before TUI rendering. |
| 6 | #241 | Textual TUI | Optional interactive TUI that loads fixture/live reports without affecting base CLI imports. |
| 7 | #243 | Diagnostics in demo reports | Demo reports link checkpoint, latent, timeline, and tensor artifacts consistently. |
| 8 | #244 | Diagnostics-driven model experiment | Define the next falsifiable model improvement from observed ranking/latent failures. |
| 9 | #245 | Visual artifact publication | Publish one manifest-backed visual observability artifact set as diagnostic evidence. |

## Artifact Contracts

Planned artifact schemas:

- `codelewm.training.tensorboard_export.v1`: event-log metadata, scalar tags,
  histogram tags, event-file paths, checksums, and parent training/checkpoint
  artifacts.
- `codelewm.model_checkpoint_inspection.v1`: implemented module tree,
  parameter counts, tensor shapes, dtypes, finite-value checks, norms, summary
  histograms, config, checkpoint manifest, and trust-gate status.
- `codelewm.eval.latent_matrix_report.v1`: implemented latent matrix shape, dimension
  count, sample count, split/source coverage, per-dimension statistics,
  covariance/correlation summaries, effective rank, probe associations, and
  semantic-axis claim gates.
- `codelewm.run_timeline.v1`: implemented ordered steps, timestamps, durations, commands,
  artifact ids, warnings, typed failures, and redaction status.
- `codelewm.harness.visual_view_model.v1`: normalized view data consumed by
  JSON, rich terminal, HTML, and Textual surfaces.

All artifacts must be JSON-native where applicable, finite, schema-versioned,
manifest-backed, checksum-verifiable, and secret-scanned before publication.

## Dependency Policy

Visualization dependencies are optional:

- TensorBoard-compatible training export is behind the optional observability
  dependency group.
- Textual belongs behind a future TUI/runtime dependency group.
- Base install, fixture tests, JSON reports, and normal CLI imports must work
  without either dependency.

Dependency errors must be typed and include remediation such as:

```bash
uv sync --group dev --group observability
uv sync --group dev --group tui
```

The observability group is now landed for TensorBoard-compatible export. The
TUI group remains planned until the Textual implementation issue lands.

## TUI Contract

The Textual TUI is a viewer and orchestrator, not a new execution authority. It
may:

- load existing demo reports and candidate packs;
- run deterministic fixture demo paths;
- follow progress from structured timeline/log artifacts;
- open local artifact summaries;
- display candidate rankings, model metadata, latent diagnostics, warnings, and
  claim gates.

It must not:

- import or execute candidate code;
- require provider keys for fixture mode;
- require Textual for JSON/rich/HTML output;
- print secrets or unscanned raw provider responses.

## Model Improvement Contract

The model-improvement issue (#244) must start from diagnostics, not from a blind
training run. It should use:

- candidate static analysis and semantic completeness failures;
- scorer and no-action deltas;
- checkpoint tensor and layer reports;
- latent matrix diagnostics and probe gates;
- downstream reranking metrics and baselines.

Before any new scaled run, it must define:

- falsifiable hypothesis;
- data/config intervention;
- expected failure modes;
- metrics and baselines;
- HF Jobs recipe;
- artifact download and verification gates;
- public wording if the run fails.

## Validation Gates

Every implementation issue should run the strongest relevant subset:

```bash
uv run pytest tests/
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests
uv run codelewm --help
```

Additional expected gates:

- TensorBoard/export issues: fixture event-log metadata tests, no live server.
- Checkpoint inspection: trusted tiny checkpoint fixture, manifest verify.
- Latent matrix: closed with finite-statistics tests, shape checks, blocked
  claim gates, CLI fixture coverage, and manifest-backed output.
- TUI: no-Textual import tests plus Textual fixture tests under optional group.
- Publication issue: manifest verify and secret scan over every publishable
  artifact.

## Success Signal

The stream is complete when #245 publishes one artifact set that lets a reader
answer:

- What scenario and candidate patches were evaluated?
- Which model/checkpoint/scorer produced the scores?
- What did the checkpoint tensor/layer inspection show?
- What did the latent dimension matrix and probe gates show?
- What did CodeLeWM rank highest, and how did that compare with no-action and
  LLM order?
- What run timeline, logs, manifests, and scans prove the path?
- Why does the claim gate remain closed?

## `/goal` Prompt

```text
/goal Continue CodeLeWM from the completed #236 roadmap lock. Work one GitHub
issue per branch and PR. Use #235 as the tracker for the v1.4 visual model
observability and TUI harness stream.

Ground in AGENTS.md, SPEC.md, docs/spec/05-observability.md,
docs/spec/06-security.md, docs/spec/11-llm-world-model-harness.md,
docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md,
docs/roadmap/MODEL_OBSERVABILITY_TUI_ROADMAP.md,
docs/roadmap/MEANINGFUL_HARNESS_DEMO.md,
docs/roadmap/IMPLEMENTATION.md, docs/roadmap/FULL_COMPLETION.md,
docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md, and the selected issue.

The current state is diagnostic: the live bugfix-edge-case harness run worked
end to end, but the learned scorer ranked an incomplete candidate above more
semantically complete patches. Do not claim semantic latent axes, coding
usefulness, or action-conditioned quality from demo artifacts.

Recommended order: finish or account for the v1.3 meaningful-demo prerequisites
(#227-#231) when the selected issue depends on them, then continue with #242,
#241, #243, #244, and #245 under #235. Issues #237, #238, #239, and #240 are
closed and provide the optional TensorBoard-compatible export, trusted
checkpoint tensor/layer inspection, latent-matrix diagnostic surfaces, and
run-timeline artifacts.

Keep visualization dependencies optional. TensorBoard-compatible exports and
Textual TUI support must not affect base imports, normal JSON output, fixture
tests, or non-interactive CLI usage. Candidate code remains untrusted and is
not executed by default. All publishable artifacts must be manifest-backed,
checksum-verifiable, schema-versioned, and secret-scanned.
```
