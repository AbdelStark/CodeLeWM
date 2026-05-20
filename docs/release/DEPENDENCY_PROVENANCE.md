# Dependency Audit And Provenance Gate

This gate covers package dependencies, built Python distributions, and the
release evidence that ties them to one source commit. It complements
`docs/release/PACKAGE_PUBLISHING.md`; it does not publish artifacts or change
Hugging Face repository visibility.

## Status

Pull-request CI audits the installed base, development, and release dependency
environment with `pip-audit`, then writes a schema-versioned provenance report
for the built wheel, source distribution, lockfile, audit report, and release
gate docs.

The report schema is `codelewm.release_provenance.v1`.

## Dependency Audit

Run from a clean checkout after syncing the release dependency group:

```bash
uv sync --group dev --group release
mkdir -p .artifacts/release/dependency-audit
uv run pip-audit \
  --format json \
  --output .artifacts/release/dependency-audit/pip-audit.json
```

`pip-audit` exits non-zero when it finds a vulnerability. That blocks release
unless the release checklist records a signed waiver with the advisory ID,
affected package/version, reason, mitigation, and reviewer sign-off.

The CI audit scope is the installed base package plus the `dev` and `release`
dependency groups. If a release candidate includes optional data, train, or eval
runtime artifacts, run and attach a separate audit for that environment before
#126 signs off the release.

## Provenance Report

Generate provenance after building the package artifacts:

```bash
uv build --sdist --wheel --out-dir .artifacts/package-gate/dist --clear
uv run scripts/release-provenance \
  --dist .artifacts/package-gate/dist \
  --audit-report .artifacts/release/dependency-audit/pip-audit.json \
  --include docs/release/PACKAGE_PUBLISHING.md \
  --include docs/release/DEPENDENCY_PROVENANCE.md \
  --out .artifacts/release/provenance/provenance.json \
  --require-clean-tracked-tree \
  --json
```

The provenance report records:

- source git SHA;
- tracked dirty-state evidence;
- Python and platform metadata;
- `uv.lock` checksum;
- built wheel and source distribution checksums;
- dependency audit report checksum;
- additional release evidence files.

The report intentionally records repository-relative paths only. Absolute local
paths are rejected so the file is safe to publish or attach to release notes.

## Residual Risks

- CI does not generate an SBOM in #124. The release gate uses `pip-audit` JSON
  plus CodeLeWM provenance JSON as the auditable supply-chain evidence.
- Optional train/data/eval environments can add heavyweight dependencies that
  are not installed in the package-build job. Release candidates that publish
  artifacts requiring those environments must attach an additional audit report.
- A clean dependency audit does not prove source integrity for remote datasets,
  model checkpoints, or Hugging Face-hosted artifacts. Those remain covered by
  manifest verification, checkpoint trust, license gates, and secret scanning.
