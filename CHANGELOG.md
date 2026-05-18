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

| Surface                  | Schema version                       |
| ------------------------ | ------------------------------------ |
| Dataset manifest         | `codelewm.dataset.v1`                |
| Artifact manifest        | `codelewm.artifact_manifest.v1`      |
| Checkpoint manifest      | `codelewm.checkpoint.v1`             |
| Training-run manifest    | `codelewm.training_run.v1`           |
| Retrieval report         | `codelewm.eval.retrieval_report.v1`  |
| Candidate pool           | `codelewm.eval.candidate_pool.v1`    |
| Public license gate      | `codelewm.public_license_gate.v1`    |
| Score result             | `codelewm.score.v1`                  |
| Rerank result            | `codelewm.rerank.v1`                 |
| Harness error report     | `codelewm.error.v1`                  |
| Structured log event     | `codelewm.log_event.v1`              |
| Transition record        | `codelewm.transition.v1`             |

A schema version bump (for example `codelewm.score.v1` to
`codelewm.score.v2`) is treated as a breaking change for that surface
and lands behind an explicit migration entry in this changelog.
