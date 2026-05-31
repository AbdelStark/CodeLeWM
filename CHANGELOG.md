# Changelog

All notable changes to CodeLeWM are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once `1.0.0` ships. Schema-versioned public surfaces (CLI flags, JSON
report schemas, manifest schemas, error contracts) are listed
explicitly so consumers can pin against them.

The deprecation policy is documented in `CONTRIBUTING.md` and in
`docs/spec/09-release-and-versioning.md`. A removal lands at the
earliest one minor release after the deprecation notice.

## [Unreleased]

### Added

- v0.6 execution-rerank LLM showcase for #307:
  `scripts/llm-world-model-demo --scenario execution-rerank-mbpp --tour 5`
  now runs a five-problem public-safe synthetic MBPP-style tour. The tour
  samples OpenRouter candidates, labels them through `codelewm.data.sandbox`,
  scores candidate code and hidden inputs with
  `codelewm.execution_torch_transition_scorer.v1`, writes
  `codelewm.harness.execution_rerank_tour.v1`, preserves the existing
  `codelewm.harness.execution_rerank_view_model.v1` renderer contract, exports
  a self-contained HTML report, and includes a committed asciicast at
  `docs/demo/execution_rerank_tour_2026-05-31.cast`. The live 2026-05-31 tour
  remains workflow evidence only: claim gates stay closed below the scaled
  100-example downstream benchmark.
- Two-substrate paper draft and arXiv package for #306:
  `docs/papers/two_substrate_paper.tex`, fetched arXiv BibTeX references,
  `docs/papers/two_substrate_claim_audit.md`,
  `docs/papers/ARXIV_SUBMISSION.md`, and
  `scripts/build-two-substrate-paper`. The draft turns the v0.2 negative
  evidence and v0.6 partial-positive evidence into a full manuscript,
  embeds collapse and margin figures, and keeps every quantitative claim
  tied to checked-in benchmark reports or schema-versioned eval artifacts.
- v0.6 downloaded-artifact eval pass for #305: committed per-seed
  eval artifacts under `docs/benchmark/v0_6/` for seed 42 and seed
  1729 across `execution-retrieval`, `execution-surprise`,
  `execution-probe`, and `crash-prediction`. The v0.6 results report
  now includes concrete cross-seed retrieval, surprise, latent-probe,
  and crash-prediction tables. Retrieval passes the no-action gates
  on both seeds (Recall@1 lift +0.6186 / +0.6102; MRR lift +0.6628 /
  +0.6547), generated-decoy surprise AUC is 1.0 on all configured
  decoy categories, latent probes remain claim-blocked because lexical
  controls beat the latent view, crash prediction is not evaluable
  because the val/test slice has zero positives, and HumanEval /
  MBPP-Plus rerank remains blocked until live completion-label
  artifacts exist. The eval-report tree is mirrored to
  `abdelstark/codelewm-runs/runs/codelewm-v0-6-eval-pass-20260531`
  at HF commit `396a8fab5b86c16764bec0090e8af7518de41fbc`.
- v0.6 downstream-rerank completion sampler:
  `scripts/sample-execution-rerank-completions` now emits
  manifest-backed HumanEval / MBPP-Plus completion-label JSONL artifacts
  with schema `codelewm.eval.completion_label.v1`. The sampler supports
  deterministic offline dry-runs for CI and explicit live OpenRouter
  sampling with `--live`, labels candidate code only through
  `codelewm.data.sandbox` under the v0.6 stdlib-only policy, writes a
  `codelewm.eval.completion_sampling_report.v1` report, and records
  secret-scan evidence before manifesting the artifact (#304).
- v0.6 JSONL execution-pack eval CLIs: `codelewm eval
  execution-retrieval`, `codelewm eval execution-surprise`,
  `codelewm eval execution-probe`, and `codelewm eval
  crash-prediction` now consume `codelewm.execution_train_checkpoint.v1`
  checkpoints plus execution pack directories and write
  schema-versioned, manifest-backed reports for retrieval, surprise,
  latent probes, and crash prediction (#302).
- v0.6 execution-substrate end-to-end results report:
  `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` documents both
  the first complete v0.6 HF Jobs run for tracker #289 and the #305
  downloaded-artifact eval pass. Two seeds (42, 1729) trained to 50k
  steps each on `a10g-small`. Headline substrate-pivot prediction
  confirmed across both seeds: prediction MSE drops 1500× (0.95 →
  6e-4), SIGReg drops 1200× (44 → 0.036), no-action margin flips from
  −0.77 to +1.24, effective-rank ratio of predicted latents reaches
  0.47 (2.3× the 0.20 collapse gate), and execution-pack retrieval
  now beats no-action by +61.4 Recall@1 points / +65.9 MRR points on
  average across seeds. The allowed public framing is partial
  positive: internal substrate and execution-pack gates pass, broader
  downstream utility remains unsupported. Artifacts are published at
  `abdelstark/codelewm-runs/runs/codelewm-v0-6-execution-20260530-af1a114-seed-{42,1729}`.
- Two-substrate paper outline updated for the v0.6 first end-to-end
  run and the #305 eval pass: abstract, §6.1 (retrieval), §6.2
  (collapse and surprise), §6.3 (latent probes), §6.4 (downstream
  rerank status), §6.5 (crash prediction), and §11 now commit to the
  partial-positive framing with concrete numbers and explicit blocked
  claim surfaces.

- v0.6 runtime entrypoint: post-training artifact upload to the
  configured `hf_jobs.artifact_repo_id` dataset repo. The HF Jobs
  container's `/tmp` is ephemeral, so without an explicit upload
  step the checkpoints + `metrics.jsonl` + reports written by the
  runner disappear when the job exits. The entrypoint now runs the
  operator's command (no longer via `exec`), captures its exit
  code, and on success uploads `$CODELEWM_RUN_OUTPUT_DIR` to
  `$CODELEWM_UPLOAD_REPO_ID:$CODELEWM_UPLOAD_PATH_IN_REPO` via
  `hf upload`. Crashed runs are kept out of the dataset. The
  launcher (`codelewm.training.execution_launch_plan`) wires the
  three new env vars and adds `--out <run_dir>` + `--json` to the
  command vector so the runner writes to a known path and the
  training_manifest is captured in stdout. Two new launcher tests
  (`test_command_wires_artifact_upload`) and two new entrypoint
  static checks (`CODELEWM_UPLOAD_REPO_ID`, `hf upload`) cover
  every wiring point.

### Fixed

- `codelewm manifest verify --parent-manifest` now accepts the
  historical v0.6 `execution_pack:<artifact_id>` parent reference when
  the provided parent manifest has the raw execution-pack artifact id.
  The compatibility alias lets immutable seed-42 and seed-1729 run
  manifests verify cleanly after #303 fixed future runner output.
- HumanEval source ingestion now strips trailing indentation placeholders
  from `prompt` before appending `canonical_solution`, preventing
  fixture and dry-run canonical completions from producing duplicated
  indentation and sandbox `IndentationError` failures (#304).
- v0.6 runner: `_train_one_step` now passes `action_emb` and
  `action_reconstruction` to `compute_transition_objective` when
  the v0.6 config sets a non-zero
  `inverse_action_reconstruction_weight` (the default since #289).
  Without the fix the production runner aborted on the first batch
  with `ValueError: enable_inverse_action_reconstruction requires
  action_emb and action_reconstruction`. The smoke runner path was
  unaffected because its `ExecutionTorchTrainConfig` defaults the
  inverse-reconstruction weight to 0. New test in
  `tests/training/test_execution_runner.py` covers the
  inverse-reconstruction branch end-to-end on the MBPP fixture.
- v0.6 runtime container: `UV_CACHE_DIR` and `HF_HOME` default to
  `/tmp/uv-cache` and `/tmp/huggingface` respectively. HF Jobs runs
  the container as a UID without write access to the build-time
  HOME, so writes to `/root/.cache/uv` and `~/.cache/huggingface`
  fail with `Permission denied`. Pinning the cache dirs to `/tmp`
  fixes both `uv` invocations and the entrypoint's `hf download`.

### Changed

- v0.6 launch plan: the `LaunchPlan.command` vector now prepends
  `/usr/local/bin/codelewm-runtime-entrypoint` and invokes
  `codelewm train` directly (replacing the previous `uv run codelewm
  train`). Two reasons:
  (1) HF Jobs strips the image's `ENTRYPOINT` when a COMMAND is
  supplied, so the entrypoint script that pre-downloads the
  execution pack must be invoked explicitly;
  (2) `codelewm` is installed system-wide at image build time, so
  `uv run` adds nothing useful and its cache-dir creation fails on
  HF Jobs' read-only HOME. The command also passes
  `--secrets HF_TOKEN` so the entrypoint can authenticate against
  the Hub. Two new launcher tests cover both invariants. The legacy
  command vector (`uv run codelewm train …`) is no longer emitted.
- v0.6 launch plan: the runtime image reference is now configurable
  via `hf_jobs.runtime_image` in
  `codelewm.execution_train_config.v1`. The v0.6 config now points at
  `ghcr.io/abdelstark/codelewm-runtime:v0.6` (the GHCR registry the
  CodeLeWM maintainers publish to). When the field is absent the
  launcher falls back to `DEFAULT_RUNTIME_IMAGE` (also pointed at
  GHCR), preserving backwards compatibility with any in-flight
  consumers of older configs. Tests in
  `tests/training/test_execution_launch_plan.py` cover both branches.

### Added

- v0.6 execution-substrate production training runner:
  `codelewm.training.execution_runner` exposes `train_execution_run`
  and `ExecutionTrainRunResult`. The runner consumes the
  `codelewm.execution_train_config.v1` YAML, resolves the pack
  (`pack_local_dir` kwarg, `CODELEWM_EXECUTION_PACK_LOCAL_DIR` env, or
  `huggingface_hub.snapshot_download` fallback), and writes a full
  artifact set: artifact `manifest.json`, `training_manifest.json`,
  `metrics.jsonl`, step-tagged checkpoints plus `last.pt` / `best.pt`
  pointers with `.manifest.json` sidecars, a per-step
  `reports/execution_train_run_report.json`, and a
  `reports/collapse_diagnostics.jsonl` row emitted every
  `trainer.collapse_diagnostics_every_n_steps` (#293). The runner
  honours `trainer.keep_last_n_checkpoints`,
  `trainer.keep_best_by_metric`, and `trainer.tensorboard_enabled`.
  No EMA target encoder — SIGReg alone is the anti-collapse term so
  the substrate-pivot comparison with v0.2 stays controlled. New
  schemas: `codelewm.execution_train_run_report.v1` and
  `codelewm.execution_train_collapse_diagnostics.v1`. Pairs with the
  runtime container entrypoint (#292) that pre-populates
  `CODELEWM_EXECUTION_PACK_LOCAL_DIR` so the runner short-circuits
  its HF download inside HF Jobs.
- v0.6 execution-substrate training config schema:
  `codelewm.training.execution_train_config` exposes
  `ExecutionTrainConfig`, `load_execution_train_config`, and
  `peek_train_config_schema_version`. The config is the operator-facing
  shape generated by `scripts/hf-launch-execution-run` and the existing
  v0.6 launcher (`codelewm.training.execution_launch_plan`); the new
  parser keeps the two in lockstep so a config that validates against
  one validates against the other. Schema marker:
  `codelewm.execution_train_config.v1`.
- `codelewm train` dispatches on `schema_version`: a
  `codelewm.execution_train_config.v1` config is routed to the
  execution runner with the new `--seed` and `--pack-local-dir` flags;
  any other schema falls through to the legacy HDF5 path. The
  HF Jobs invocation
  `uv run codelewm train --config config/train/scaled/codelewm_execution_v0_6_a10g.yaml --seed 42`
  now does the right thing end-to-end (#293).
- v0.6 execution-substrate torch training bridge:
  `codelewm.training.execution_torch_runner` adds
  `train_execution_smoke()`, `ExecutionTorchTrainConfig`,
  `ExecutionTorchReport`, `ExecutionTorchStep`,
  `ExecutionTorchRunnerError`. The bridge reuses the existing
  `TorchCodeTransitionModel` and `compute_transition_objective` end
  to end and wires them to `ExecutionPackBatch`. Output tokens are
  padded up to `STATE_SEQUENCE_LENGTH` so the same state encoder
  produces `z_output` with shared weights. Action-swap contrastive
  is built via intra-batch input rolling. New CLI:
  `scripts/codelewm-execution-train-smoke` (ingest → sandbox-pack →
  train → JSON report with a built-in smoke gate). New schemas:
  `codelewm.execution_train_report.v1`,
  `codelewm.execution_train_step.v1`,
  `codelewm.execution_train_smoke.v1`. Local smoke evidence in
  `docs/benchmark/EXECUTION_V0_6_LOCAL_SMOKE_2026-05-28.md`:
  prediction MSE 1.01 → 0.04 (24×), SIGReg 5.21 → 1.36 (4×),
  no-action margin −0.77 → +0.55 (the substrate-pivot's headline
  prediction). Tests under `tests/training/` skip cleanly without
  torch installed and run end-to-end under `uv sync --group train`.
- v0.6 benchmark report template and two-substrate paper outline:
  `docs/benchmark/EXECUTION_V0_6_RESULTS_TEMPLATE.md` mirrors
  `docs/benchmark/REPORT_TEMPLATE.md` scoped to the
  execution-substrate pivot, with explicit claim-gate tables for
  retrieval, collapse, surprise, latent probes, downstream
  rerank, and crash prediction.
  `docs/papers/two_substrate_outline.md` defines the publishable
  comparison between Substrate A (commit-edit, v0.2) and Substrate B
  (execution-trace, v0.6), structured so it ships under either
  outcome (gates pass or gates fail). `docs/PROJECT_EXPLAINER.md`
  links the substrate-pivot stack. (#272)
- Execution-substrate rerank visual view model:
  `codelewm.harness.execution_rerank_view_model` adds
  `build_execution_rerank_view_model()`, `ExecutionRerankViewModel`,
  `CompletionPanelEntry`, `ExecutionRerankViewModelError`. The view
  model is the JSON-shape contract between the rerank report (#268)
  and the HTML / terminal / TUI renderers. It exposes a headline
  panel (pass@1 lift, bootstrap CI, claim status), per-completion
  panels (predicted output latent norm, test results, scores per
  baseline, ranks under each baseline), and a notes block that
  surfaces blocked-claim warnings. New schema:
  `codelewm.harness.execution_rerank_view_model.v1`. (#271)
- Execution-substrate rerank LLM demo scenario:
  `execution-rerank-mbpp` is registered alongside `bugfix-edge-case`
  in `codelewm.harness.demo_scenarios`. The scenario ships a stub
  `compute_square` function with example input `[3]` and expected
  output `9`; the prompt template id is
  `codelewm.openrouter.demo_scenario.execution_rerank.v1`. New
  export: `codelewm.harness.EXECUTION_RERANK_SCENARIO_ID`. The
  scenario's publication notes explicitly scope sandbox use to the
  operator-reviewed example input. (#270)
- Crash-prediction binary classification eval:
  `codelewm.eval.crash_prediction` adds `evaluate_crash_prediction()`,
  `CrashSample`, `CrashPredictionReport`, `MethodMetrics`,
  `CrashPredictionError`, `LATENT_METHODS`, `NON_LATENT_METHODS`.
  The eval computes per-method accuracy, AUC-ROC, AUC-PR, F1 (at 0.5
  threshold), per-exception-class AUC, and per-source-dataset AUC.
  The scoped claim is allowed when the best latent-based method beats
  the best non-latent method by ≥0.05 absolute AUC. New schema:
  `codelewm.eval.crash_prediction_report.v1`. (#269)
- Execution-substrate rerank evaluation:
  `codelewm.eval.execution_rerank` adds `rerank_completions()`,
  `CompletionLabel`, `ScoredCompletion`, `ExecutionRerankReport`,
  `BaselineSummary`, `ExecutionRerankError`, `load_completion_labels()`,
  `EXECUTION_RERANK_BASELINES`. The protocol computes pass@1 under
  random / lexical / `llm_order` / `no_action` / `shuffled_action` /
  `codelewm` baselines, reports the CodeLeWM lift over LLM original
  order, and emits a bootstrap 95% CI on the lift. The claim gate
  fires only when both `lift >= min_lift_for_claim` (default 3.0
  absolute points) and the lower CI bound exceeds zero. The module
  is model-agnostic: callers supply scored completions and ground-
  truth pass/fail labels. New schema:
  `codelewm.eval.execution_rerank_report.v1`. (#268)
- Execution-substrate surprise-eval decoy generators:
  `codelewm.eval.execution_surprise_decoys` adds two new categories
  that test program semantics rather than surface code similarity:
  `same_problem_different_submission` (different submission for the
  same problem and input; outputs must differ) and
  `same_code_different_input` (same submission, different input;
  outputs must differ). Each generator returns a list of `DecoyPair`
  and a `DecoyGenerationReport` with `pair_count`,
  `eligible_query_count`, and per-reason skip tallies. New schema:
  `codelewm.eval.execution_surprise_decoy.v1`. (#267)
- Execution-substrate latent probe target label extractors:
  `codelewm.eval.execution_probe_targets` exposes `label_record`,
  `extract_labels`, `LabelExtraction`, `EXECUTION_PROBE_TARGETS`
  tuple, `ExecutionProbeTargetError`, and a new schema marker
  `codelewm.eval.execution_probe_target.v1`. Six targets are
  supported: `output_type`, `will_raise`, `output_magnitude_bucket`,
  `output_length_bucket`, `arithmetic_vs_string_vs_collection`, and
  `judge_verdict`. Records outside a target's domain return `None`
  and are excluded from the probe's eval — matching the
  applicable-only policy already used by the commit-edit probe
  runner. (#266)
- v0.6 execution-substrate HF Jobs launcher: a config-driven launch
  plan generator (`codelewm.training.execution_launch_plan` exports
  `load_v0_6_config`, `build_launch_plans`, `LaunchPlan`,
  `ExecutionLaunchPlanError`, `EXECUTION_LAUNCH_PLAN_SCHEMA_VERSION`),
  an operator-facing dry-run-by-default script
  `scripts/hf-launch-execution-run` that prints one plan per
  configured seed, the v0.6 training config at
  `config/train/scaled/codelewm_execution_v0_6_a10g.yaml` carrying
  loader/trainer/optimizer/objective/seeds/hf_jobs/claim_gates/
  claim_boundary sections, and the operator runbook
  `docs/operations/V0_6_EXECUTION_RUN_RUNBOOK.md`. New schemas:
  `codelewm.execution_train_config.v1`,
  `codelewm.execution_launch_plan.v1`. Live runs are operator
  triggered; the launcher itself never contacts HF. (#265)
- Execution-substrate smoke pipeline:
  `codelewm.training.execution_pack_loader` reads `pack.jsonl` (from
  #262), pads/truncates each tokenized field to configured
  `code_sequence_length` / `action_sequence_length` /
  `output_sequence_length`, and yields `ExecutionPackBatch` instances
  with NumPy `int32` token arrays and `bool` attention masks. Stable
  `OUTPUT_TYPE_VOCAB` indexes the per-batch `output_type_index`
  consumed by the upcoming probe target (#266). Diagnostics counter
  tracks truncation rates, output_type/output_kind/execution_status
  histograms, and per-split counts. New schemas:
  `codelewm.execution_pack_batch.v1`,
  `codelewm.execution_smoke_report.v1`,
  `codelewm.execution_smoke_config.v1`. New script:
  `scripts/smoke-execution-train` builds a tiny MBPP pack and runs
  the loader end-to-end, surfacing tokenization breakage before the
  v0.6 HF Jobs run (#265). New config:
  `config/train/scaled/codelewm_execution_smoke_cpu.yaml`. (#264)
- Execution-pack publish workflow: pre-publish gate
  (`codelewm.data.execution_pack.run_pre_publish_gate`) that verifies
  manifest schema, pack.jsonl checksum, claim-boundary embedding and
  fingerprint, permissive-license-only policy, and attribution
  completeness; dataset-card renderer
  (`render_dataset_card`, `context_from_manifest`,
  `DatasetCardContext`); `PrePublishGateError`, `PrePublishReport`
  classes; new script `scripts/hf-publish-execution-pack` with
  dry-run-by-default safety; dataset-card template at
  `docs/cards/dataset_card.execution_pack.v1.md`. New schema:
  `codelewm.execution_pack_publish_plan.v1`. (#263)
- Execution-substrate pack builder:
  `codelewm.data.execution_pack` with `build_execution_pack()`,
  `PackedExecutionRecord`, `ExecutionPackManifest`,
  `ExecutionPackResult`, and `ExecutionPackBuilderError`. The builder
  reads ingestion JSONLs (from `codelewm dataset ingest`), drives the
  sandbox per `(code, input)` case, drops records that fail the
  determinism / policy / OOM / timeout / output-truncation gates,
  tokenizes the surviving code / input repr / output repr, partitions
  by `source_problem_id`, and writes `pack.jsonl`, `manifest.json`,
  `attribution.json`, `sandbox_audit_summary.json`, and a copy of the
  execution-substrate claim boundary. Held-out ingestion records
  (MBPP-Plus, HumanEval) are tallied but not packed. New CLI:
  `codelewm dataset execution-pack --ingestion <jsonl> --output <dir>`
  with sandbox policy, split, balance, and target-record flags. New
  schemas: `codelewm.execution_pack_manifest.v1`,
  `codelewm.execution_pack_record.v1`. (#262)
- Source adapters for the execution-substrate ingestion path:
  `codelewm.data.execution_sources` with `SourceSubmission`,
  `InputCase`, `ExecutionSourceAdapter` protocol, and concrete adapters
  for CodeNet, MBPP, MBPP-Plus, APPS, and HumanEval. MBPP-Plus and
  HumanEval are flagged `held_out_for_eval=True` so the pack builder
  (#262) refuses to put them in train/val splits. New CLI:
  `codelewm dataset ingest --source <name> --input <jsonl> --output
  <out.jsonl>`. New schema: `codelewm.execution_source_record.v1`. (#261)
- Sandboxed deterministic Python executor for execution-substrate data
  prep: `codelewm.data.sandbox` with `run_one`, `SandboxPolicy`,
  `SandboxResult`, `SandboxExitCode`, `SandboxPolicyError`,
  `SandboxRunnerError`, and the `codelewm.sandbox_result.v1` schema. The
  child process enforces a stdlib-only import allowlist, denies network
  and subprocess primitives, audits filesystem writes outside the
  scratch directory, applies `RLIMIT_AS` and `RLIMIT_CPU` rlimits on
  POSIX, and performs a determinism re-run. The runner enforces the
  wall-clock budget. New CLI: `codelewm dataset execute`. New harness
  error types: `input_missing`, `invalid_arguments`,
  `sandbox_runner_error`. The sandbox is data-prep only; a structural
  test in `tests/security/test_sandbox_import_boundary.py` blocks
  training, model, eval, observability, scorer, index, and quality
  paths from importing it. (#260)
- Execution-substrate governance scaffolding: RFC-0014
  (`docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`), the
  substrate roadmap (`docs/roadmap/EXECUTION_TRACE_WORLD_MODEL.md`), the
  claim boundary at
  `codelewm/security/claim_boundaries/execution_substrate.v1.md`, the
  `codelewm.security.claim_boundaries` loader module, and the operations
  doc `docs/operations/sandbox_policy.md`. The new module exports
  `load_claim_boundary`, `claim_boundary_fingerprint`,
  `available_claim_boundaries`, and `ClaimBoundaryError`. (#273)
- Reproducible first-results workflow through `scripts/first-results`, the
  `config/first_results/` bundle, `codelewm.results`, and
  `docs/benchmark/FIRST_RESULTS.md`.
- Hugging Face Jobs/ml-intern training automation through `.env.example`,
  `scripts/hf-launch-codelewm-job`, `scripts/hf-run-codelewm-pipeline`,
  `scripts/hf-publish-codelewm-artifacts`,
  `docs/operations/HF_ML_INTERN_TRAINING.md`, and
  `docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md`.
- Scaled CPU, MPS, and HF A10G training configs under `config/train/scaled/`,
  plus `scripts/validate-training-configs` and
  `docs/training/SCALED_TRAINING_RUNBOOK.md`.
- Action-view ablation reports through `codelewm eval ablation` and
  `codelewm.eval.action_ablation_report.v1`.
- Release package gates for wheel/sdist builds, metadata checks, clean wheel
  install smoke, dependency audit, and `codelewm.release_provenance.v1`
  provenance reports.
- A filled #126 release-freeze report at
  `docs/release/RELEASE_FREEZE_2026-05-20.md` for the private diagnostic
  action-use artifact set.
- Initial governance documents: `CONTRIBUTING.md`, `SECURITY.md`,
  `CHANGELOG.md`, and a pull-request template at
  `.github/PULL_REQUEST_TEMPLATE.md`.
- OpenRouter Anthropic BYOK helper through
  `codelewm openrouter byok-register`, redacted BYOK request metadata, and the
  local `uv run scripts/llm-world-model-demo` task for the LLM + world-model
  fixture demo.
- Visual `demo.html` output for LLM + world-model demo artifacts, including
  generation mode, candidate patches, ranking bars, baseline orders, and the
  claim gate.

### Changed

- Public README, usage, API, roadmap, and release docs now separate smoke
  evidence, scaled systems evidence, and negative action-use evidence.
- Architecture and roadmap docs now describe the implemented package-native
  runtime and the #256 production-cleanup tracker instead of stale bootstrap
  language.

### Deprecated

- Nothing yet.

### Removed

- Nothing yet.

### Fixed

- Collapse diagnostics no longer emit NumPy covariance runtime warnings for
  ordinary finite embeddings.
- The run-timeline context manager no longer trips static dead-code checks for
  intentionally unused exception metadata.

### Security

- Nothing yet.

## Schema Reference

The following schema versions are exposed by the package. A consumer
pinning against `codelewm` should also pin against the schema versions
their workflow depends on.

| Surface                       | Schema version                                  |
| ----------------------------- | ----------------------------------------------- |
| Dataset manifest              | `codelewm.dataset.v1`                           |
| Source acquisition report     | `codelewm.source_acquisition.v1`                |
| Action-discriminative shard report | `codelewm.data.action_discriminative_shard_report.v1` |
| Artifact manifest             | `codelewm.artifact_manifest.v1`                 |
| Manifest verification report  | `codelewm.manifest_verify.v1`                   |
| Checkpoint manifest           | `codelewm.checkpoint.v1`                        |
| Training config               | `codelewm.train_config.v1`                      |
| Training config validation    | `codelewm.train_config_validation.v1`           |
| Training-run manifest         | `codelewm.training_run.v1`                      |
| Training metrics              | `codelewm.training_metrics.v1`                  |
| CPU smoke checkpoint          | `codelewm.cpu_smoke_checkpoint.v1`              |
| CPU smoke report              | `codelewm.cpu_smoke_report.v1`                  |
| First-results inventory       | `codelewm.first_results.v1`                     |
| Index build report            | `codelewm.index_build.v1`                       |
| Retrieval eval run            | `codelewm.eval.retrieval_run.v1`                |
| Retrieval metrics             | `codelewm.eval.retrieval_metrics.v1`            |
| Retrieval report              | `codelewm.eval.retrieval_report.v1`             |
| Candidate pool                | `codelewm.eval.candidate_pool.v1`               |
| Action-contrast pool report   | `codelewm.eval.action_contrast_pool_report.v1`  |
| Latent probe eval run         | `codelewm.eval.latent_probe_run.v1`             |
| Latent probe report           | `codelewm.eval.latent_probe_report.v1`          |
| Hard-negative sample          | `codelewm.eval.hard_negative_sample.v1`         |
| Hard-negative sampler report  | `codelewm.eval.hard_negative_sampler_report.v1` |
| Patch-surprise eval run       | `codelewm.eval.surprise_run.v1`                 |
| Patch-surprise report         | `codelewm.eval.surprise_report.v1`              |
| Action-view policy            | `codelewm.eval.action_view_policy.v1`           |
| Action ablation report        | `codelewm.eval.action_ablation_report.v1`       |
| Action ablation run           | `codelewm.eval.action_ablation_run.v1`          |
| Scorer quality config         | `codelewm.harness.scorer_quality_config.v1`     |
| Scorer quality report         | `codelewm.harness.scorer_quality_report.v1`     |
| Scorer quality run            | `codelewm.harness.scorer_quality_run.v1`        |
| LLM candidate pack            | `codelewm.llm_candidate_pack.v1`                |
| OpenRouter BYOK registration  | `codelewm.openrouter_byok_register.v1`          |
| LLM demo report               | `codelewm.harness.demo_report.v1`               |
| LLM demo run                  | `codelewm.harness.demo_run.v1`                  |
| Downstream rerank benchmark   | `codelewm.downstream_rerank_benchmark.v1`       |
| Downstream rerank config      | `codelewm.downstream_rerank_benchmark_config.v1` |
| Downstream benchmark pack run | `codelewm.downstream_benchmark_pack_run.v1`     |
| Downstream benchmark readiness | `codelewm.downstream_benchmark_readiness.v1`    |
| Downstream source license policy | `codelewm.downstream_source_license_policy.v1` |
| Downstream split leakage report | `codelewm.downstream_split_leakage_report.v1`   |
| Downstream rerank eval run    | `codelewm.downstream_rerank_eval_run.v1`        |
| Downstream rerank report      | `codelewm.downstream_rerank_report.v1`          |
| Downstream rerank claim gate  | `codelewm.downstream_rerank_claim_gate.v1`      |
| Collapse report               | `codelewm.eval.collapse_report.v1`              |
| Kill-switch report            | `codelewm.eval.kill_report.v1`                  |
| Public license gate           | `codelewm.public_license_gate.v1`               |
| Secret-scan report            | `codelewm.secret_scan.v1`                       |
| Score result                  | `codelewm.score.v1`                             |
| Rerank result                 | `codelewm.rerank.v1`                            |
| Harness error report          | `codelewm.error.v1`                             |
| Transition index              | `codelewm.transition_index.v1`                  |
| Structured log event          | `codelewm.log_event.v1`                         |
| Release provenance report     | `codelewm.release_provenance.v1`                |
| Transition record             | `codelewm.transition.v1`                        |

A schema version bump (for example `codelewm.score.v1` to
`codelewm.score.v2`) is treated as a breaking change for that surface
and lands behind an explicit migration entry in this changelog.
