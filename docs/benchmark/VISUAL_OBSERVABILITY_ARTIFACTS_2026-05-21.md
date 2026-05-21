# Visual Observability Artifact Set

Date: 2026-05-21

Issue: #245. Tracker: #235.

Run id: `codelewm-visual-observability-20260521-6a8ac81`

HF artifact path:
`https://huggingface.co/datasets/abdelstark/codelewm-runs/tree/main/visual-observability/codelewm-visual-observability-20260521-6a8ac81`

Source SHA: `6a8ac81e27eaa3aa7bfc9b60eefe58496edec99b`

## Summary

This is the v1.4 visual observability harness artifact set. It demonstrates the
model/checkpoint, latent, TensorBoard, run-timeline, terminal, HTML, JSON, and
TUI surfaces together on one deterministic fixture path and one live
OpenRouter/BYOK path.

This is diagnostic workflow evidence only. It does not support a claim that
CodeLeWM improves generated code, has useful semantic latent axes, or beats
no-action and LLM-order baselines.

## Published Artifacts

The public bundle was uploaded to `abdelstark/codelewm-runs` under:

```text
visual-observability/codelewm-visual-observability-20260521-6a8ac81/
```

Key files:

- `ARTIFACT_INDEX.json`
- `README.md`
- `CHECKSUMS.sha256`
- `train/manifest.json`
- `train/reports/tensorboard_export.json`
- `checkpoint_inspection/manifest.json`
- `checkpoint_inspection/reports/model_checkpoint_inspection.json`
- `latent_matrix/manifest.json`
- `latent_matrix/reports/latent_matrix_report.json`
- `demo_fixture/manifest.json`
- `demo_fixture/reports/llm_world_model_demo_report.json`
- `demo_fixture/reports/visual_view_model.json`
- `demo_fixture/reports/run_timeline.json`
- `demo_fixture/demo.html`
- `demo_fixture_terminal.txt`
- `demo_fixture_tui_snapshot.json`
- `demo_live/manifest.json`
- `demo_live/reports/llm_world_model_demo_report.json`
- `demo_live/reports/visual_view_model.json`
- `demo_live/reports/run_timeline.json`
- `demo_live/demo.html`
- `demo_live_terminal.txt`
- `demo_live_tui_snapshot.json`
- `verification/secret_scan_after_index.json`

## What Ran

Deterministic fixture path:

- built and packed the first-results fixture dataset;
- ran a 4-step package-native torch training job on CPU;
- emitted TensorBoard-compatible event metadata;
- inspected the trusted checkpoint tensors and layers;
- generated latent matrix diagnostics;
- ran a dry-run LLM demo with linked checkpoint, latent, TensorBoard, and
  timeline diagnostics;
- emitted terminal, HTML, JSON, and deterministic TUI snapshot surfaces.

Live path:

- ran OpenRouter/BYOK candidate generation with debug logging unset;
- generated 4 parseable candidates;
- scored them with the learned torch transition scorer;
- linked the same checkpoint, latent, TensorBoard, and timeline diagnostics;
- emitted terminal, HTML, JSON, and deterministic TUI snapshot surfaces.

## Key Diagnostics

Checkpoint inspection:

- schema: `codelewm.model_checkpoint_inspection.v1`
- parameters: `29275392`
- tensor count: `147`
- all tensors finite: `true`

TensorBoard export:

- schema: `codelewm.training.tensorboard_export.v1`
- event files: `1`
- scalar tags include loss, collapse, action diagnostics, and latent summaries;
- histogram tags include selected latent and model-parameter summaries.

Latent matrix:

- schema: `codelewm.eval.latent_matrix_report.v1`
- rows inspected: `3`
- semantic-axis claim gate: closed
- blocked claims: semantic latent axes, action-conditioned quality, downstream
  coding usefulness.

## Demo Results

Scores are transition energies, so lower is better.

Fixture demo:

- candidates: `4`
- best candidate: `candidate_004.patch`
- best score: `121.55994415283203`
- no-action score: `121.22270202636719`
- claim gate: closed

Live demo:

- candidates: `4`
- best candidate: `candidate_001.patch`
- best score: `120.6988525390625`
- no-action score: `121.22270202636719`
- claim gate: closed

The live demo is useful workflow evidence because candidate generation, learned
scoring, diagnostics, manifests, terminal/HTML/JSON reports, and TUI snapshots
all worked. It is still not benchmark evidence. The claim gate remains closed
because a demo report is not a downstream benchmark.

## Verification

Local gates before publication:

```bash
uv run codelewm manifest verify --manifest <demo_fixture>/manifest.json ...
uv run codelewm manifest verify --manifest <demo_live>/manifest.json ...
uv run codelewm secret-scan .artifacts/visual-observability-v1-4/codelewm-visual-observability-20260521-6a8ac81 --json
```

The final secret scan reported:

```json
{"ok": true, "findings_count": 0, "paths_scanned": 69}
```

Post-upload check:

- downloaded `ARTIFACT_INDEX.json`, `README.md`, `CHECKSUMS.sha256`, and the
  live demo report/view files from HF;
- compared the downloaded index and live report to local copies with `cmp`;
- confirmed `claim_gate.allowed=false` and `secret_scan_ok=true` in the
  downloaded `ARTIFACT_INDEX.json`.

## Claim Boundary

Allowed:

- cite this as a public visual observability artifact set;
- show terminal, HTML, JSON, TensorBoard metadata, checkpoint inspection,
  latent matrix, run timeline, and TUI snapshot surfaces;
- say the live and fixture workflows executed and were published after manifest
  and secret-scan gates.

Blocked:

- CodeLeWM improves generated code;
- CodeLeWM has useful semantic latent axes;
- action conditioning is better than no-action;
- the harness scorer is reliable for candidate selection;
- the current checkpoint is useful for downstream coding tasks.

Those claims still require the scaled downstream and representation gates.
