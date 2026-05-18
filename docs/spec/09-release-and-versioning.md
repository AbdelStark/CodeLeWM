# Release And Versioning

## Versioning

The package follows semantic versioning after the first public release.

- `0.x`: API can change, but schema migrations must be documented.
- `1.x`: stable CLI and JSON contracts.
- Major version: breaking CLI, JSON, artifact, or public Python API changes.
- Minor version: new compatible functionality.
- Patch version: bug fixes and documentation corrections.

## Schema Versioning

Dataset schemas, manifests, score outputs, and evaluation reports carry explicit
schema versions. A schema version can be loaded only by compatible code or by an
explicit migration command.

## Deprecation

Deprecations require:

- warning in one minor release before removal for stable APIs;
- changelog entry;
- migration note;
- tests for the old path until removal.

## Release Artifacts

A public release includes:

- source distribution and wheel;
- signed or checksummed release artifacts;
- changelog;
- model card when a checkpoint is published;
- dataset card when a dataset is published;
- benchmark report;
- reproducibility manifest.

## Contributor Workflow

Every implementation pull request must link to:

- one spec section;
- one RFC where applicable;
- one GitHub issue.

Every pull request must include:

- summary;
- validation commands;
- artifact impact;
- risk or caveat if present.

## License Discipline

The repository is currently MIT-licensed. Contributions that add data, model
weights, or third-party code must declare license compatibility before merge.
