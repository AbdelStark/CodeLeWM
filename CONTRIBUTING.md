# Contributing to CodeLeWM

CodeLeWM is a spec-driven research artifact. Every change should be
traceable back to a spec section, an RFC where applicable, and a GitHub
issue. This document describes the workflow, the validation gates, and
the deprecation policy contributors are expected to honor.

## Working Tree Layout

```
codelewm/        Python package: data, model, training, eval, harness,
                 observability, security, training utilities.
docs/spec/       Canonical specification (`00-` through `10-`).
docs/rfcs/       Accepted RFCs that derive into implementation issues.
docs/roadmap/    Implementation tracker.
tests/           Test suite, mirrored to the package layout.
.github/         Pull-request template and CI workflows.
```

## Required Reading Before Editing

Open the spec section or RFC you are implementing and review:

- `docs/spec/00-overview.md` for the system framing;
- the spec section listed on the issue (typically one file in `docs/spec/`);
- the accepted RFC referenced by the issue (`docs/rfcs/RFC-NNNN-*.md`);
- `docs/spec/06-security.md` and `docs/spec/04-error-model.md` whenever
  your change touches user input, checkpoints, configs, logs, or reports.

Brief PRs that read tools rather than the specification tend to bounce.

## Branch and Commit Conventions

- One issue per branch. One PR per branch.
- Branch names are short, hyphenated, no tool branding.
- Commit subjects use the area prefix already used in `git log`
  (`harness:`, `eval:`, `data:`, `security:`, `observability:`,
  `train:`, `docs:`, `ci:`, etc.) followed by an imperative summary.
- Commit bodies explain motivation, the structural change, and the
  validation evidence. Reference the issue number in the body.

## Pull Request Requirements

A PR is reviewable when its description has all of:

- a one-paragraph statement of the problem;
- the structural change, with file paths;
- the exact validation commands that were run, copied from the
  terminal;
- the artifact impact (manifests, schemas, reports, public APIs);
- caveats or follow-ups.

Every PR must also state:

- the GitHub issue it closes (`Closes #N`);
- the spec section it implements;
- the RFC it derives from, when applicable;
- which public surface is added, changed, or removed.

The PR template at `.github/PULL_REQUEST_TEMPLATE.md` enforces these
fields. Do not delete fields you are not using; mark them `N/A` so
reviewers know the question was considered.

## Local Validation

Run the strongest validation that applies to your change before opening
the PR:

```bash
uv sync --group dev
uv run python -m pytest tests/                  # full unit + integration suite
uv run python -m pytest tests/<area>            # area-focused run
uv run python -m compileall codelewm tests      # syntactic regression check
```

Schema, observability, security, harness, and CLI tests live under
`tests/<area>` and mirror the package layout. CI runs the same
commands; differences between local and CI runs are bugs in the
workflow file.

Install optional runtime groups only when the touched surface needs
them:

```bash
uv sync --group dev --group data      # dataset packing: h5py, pyarrow
uv sync --group dev --group train     # torch/Lightning training runtime
uv sync --group dev --group eval      # optional evaluation helpers
uv sync --group dev --group docs      # documentation checks
uv sync --group dev --group release   # package build and release gates
```

Release and packaging changes must also prove the built artifacts:

```bash
uv build --sdist --wheel --out-dir .artifacts/package-gate/dist --clear
uv run twine check .artifacts/package-gate/dist/*
mkdir -p .artifacts/release/dependency-audit
uv run pip-audit --format json --output .artifacts/release/dependency-audit/pip-audit.json
uv run scripts/release-provenance \
  --dist .artifacts/package-gate/dist \
  --audit-report .artifacts/release/dependency-audit/pip-audit.json \
  --include docs/release/PACKAGE_PUBLISHING.md \
  --include docs/release/DEPENDENCY_PROVENANCE.md \
  --out .artifacts/release/provenance/provenance.json \
  --json
uv venv .artifacts/package-gate/venv --python 3.13
uv pip install --python .artifacts/package-gate/venv/bin/python .artifacts/package-gate/dist/codelewm-*.whl
.artifacts/package-gate/venv/bin/codelewm --help
```

## Deprecation Policy

Public CLI flags, JSON schemas, error contracts, and public Python APIs
are deprecated, never silently removed.

- Land the deprecation in one minor release before removal.
- Add a `CHANGELOG.md` entry under "Deprecated" pointing at the
  replacement.
- Keep tests for the deprecated path until the removal release.
- Add a migration note in the relevant spec section.

Experimental APIs live under `codelewm.experimental` and carry no
stability promise. Mark them clearly in their docstrings.

## Security

If a change touches secret handling, checkpoint loading, config
validation, or non-execution boundaries, mark the PR as
`type:security` and explicitly state the trust-boundary impact in the
PR description. Read `docs/spec/06-security.md` first.

To report a vulnerability privately, follow `SECURITY.md`.
