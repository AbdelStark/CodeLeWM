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
