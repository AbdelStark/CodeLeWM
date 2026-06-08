# Next Goal Prompt

This is the active prompt for the final v1.0 paper/demo release tracker #401.
It supersedes the historical v0.6 follow-through issues #293, #306, #308, and
#309, and it replaces the completed v0.9 data/eval repair prompt for #385.

```text
/goal Complete CodeLeWM's final v1.0 paper/demo release tracker #401 end to
end, landing one issue branch and PR per child issue, waiting for green CI, and
merging each PR to main before starting the next child.

Authoritative issue order:

1. #402 - v1.0 hygiene: reconcile stale v0.6 issues and final queue state.
2. #403 - v1.0 demo: define fixed downstream learned-world-model paper-demo
   contract.
3. #404 - v1.0 demo: implement one-command downstream learned-world-model paper
   demo.
4. #405 - v1.0 run: publish final downstream paper-demo artifacts.
5. #406 - v1.0 results: consolidate benchmark tables and final claim audit.
6. #407 - v1.0 paper: rewrite CodeLeWM paper around final downstream evidence.
7. #408 - v1.0 release: publish final artifact index, cards, README,
   reproducibility checklist, and announcement package.

Ground in AGENTS.md, SPEC.md, docs/spec/05-observability.md,
docs/spec/06-security.md, docs/spec/09-release-and-versioning.md,
docs/spec/11-llm-world-model-harness.md,
docs/rfcs/RFC-0013-llm-world-model-harness-and-publication.md,
docs/rfcs/RFC-0015-v0-7-execution-substrate-improvements.md,
docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md,
docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-07.md,
docs/roadmap/FULL_COMPLETION.md, docs/roadmap/IMPLEMENTATION.md,
docs/roadmap/POST_V0_2_SHOWCASE_ROADMAP.md,
docs/roadmap/MEANINGFUL_HARNESS_DEMO.md,
docs/roadmap/MODEL_OBSERVABILITY_TUI_ROADMAP.md,
docs/roadmap/DIAGNOSTICS_DRIVEN_MODEL_EXPERIMENT.md,
docs/benchmark/VISUAL_OBSERVABILITY_ARTIFACTS_2026-05-21.md,
docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md, and
CONTRIBUTING.md.

Do not relaunch completed #159, #172, v0.8, or v0.9 HF Jobs. Use the checked-in
v0.9 artifacts as the current benchmark evidence unless a child issue explicitly
requires a new fixed paper-demo artifact run.

The final public claim boundary is strict. CodeLeWM has a reproducible
code-edit world-model harness, verified HF Jobs and artifact-publication
infrastructure, negative action-conditioned v0.2 evidence, and a narrow
HumanEval WS-D downstream positive slice in v0.9. The overall public claim stays
closed because MBPP-Plus shows zero lift over no-action and broader semantic,
representation, and general coding-usefulness gates remain closed. Do not claim
CodeLeWM generally improves coding.

Candidate code, generated patches, configs, checkpoints, reports, and provider
outputs are untrusted inputs. Do not execute candidate code unless a child issue
uses the existing explicit sandbox allowlist/timeout/disposable-checkout
contract. Every publishable artifact must be schema-versioned where applicable,
manifest-backed, checksum-verifiable, and secret-scanned.

For each child issue:

- inspect live GitHub issue and PR state before editing;
- read the relevant specs, RFCs, roadmap docs, and current benchmark artifacts;
- keep public docs evidence-backed and claim-limited;
- run the strongest relevant validation, including docs tests, focused tests,
  compileall when code changes, manifest verification and secret scans when
  artifacts are touched, and git diff --check;
- commit, push, open a PR that closes only the current child issue, wait for
  hosted CI, merge to main, return to main, pull latest main, then continue.

Issue #408 is the final release package and depends on #402 through #407 being
complete. Close #401 only after every child issue is closed, the final release
docs are merged to main, and CI is green.
```
