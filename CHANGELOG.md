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

- Reproducible first-results workflow through `scripts/first-results`, the
  `config/first_results/` bundle, `codelewm.results`, and
  `docs/benchmark/FIRST_RESULTS.md`.
- Initial governance documents: `CONTRIBUTING.md`, `SECURITY.md`,
  `CHANGELOG.md`, and a pull-request template at
  `.github/PULL_REQUEST_TEMPLATE.md`.

### Changed

- Nothing yet.

### Deprecated

- Nothing yet.

### Removed

- Nothing yet.

### Fixed

- Nothing yet.

### Security

- Nothing yet.

## Schema Reference

The following schema versions are exposed by the package. A consumer
pinning against `codelewm` should also pin against the schema versions
their workflow depends on.

| Surface                       | Schema version                                  |
| ----------------------------- | ----------------------------------------------- |
| Dataset manifest              | `codelewm.dataset.v1`                           |
| Artifact manifest             | `codelewm.artifact_manifest.v1`                 |
| Manifest verification report  | `codelewm.manifest_verify.v1`                   |
| Checkpoint manifest           | `codelewm.checkpoint.v1`                        |
| Training config               | `codelewm.train_config.v1`                      |
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
| Hard-negative sample          | `codelewm.eval.hard_negative_sample.v1`         |
| Hard-negative sampler report  | `codelewm.eval.hard_negative_sampler_report.v1` |
| Patch-surprise eval run       | `codelewm.eval.surprise_run.v1`                 |
| Patch-surprise report         | `codelewm.eval.surprise_report.v1`              |
| Action-view policy            | `codelewm.eval.action_view_policy.v1`           |
| Collapse report               | `codelewm.eval.collapse_report.v1`              |
| Kill-switch report            | `codelewm.eval.kill_report.v1`                  |
| Public license gate           | `codelewm.public_license_gate.v1`               |
| Secret-scan report            | `codelewm.secret_scan.v1`                       |
| Score result                  | `codelewm.score.v1`                             |
| Rerank result                 | `codelewm.rerank.v1`                            |
| Harness error report          | `codelewm.error.v1`                             |
| Transition index              | `codelewm.transition_index.v1`                  |
| Structured log event          | `codelewm.log_event.v1`                         |
| Transition record             | `codelewm.transition.v1`                        |

A schema version bump (for example `codelewm.score.v1` to
`codelewm.score.v2`) is treated as a breaking change for that surface
and lands behind an explicit migration entry in this changelog.
