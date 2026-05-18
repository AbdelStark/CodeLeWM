# RFC-0012: Release, CI, And Governance

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

CodeLeWM requires pull-request CI, release gates, changelog discipline, and issue
traceability before implementation work scales. Every feature PR must link to a
spec section, RFC, and GitHub issue.

## Motivation

The backlog requires a spec-driven implementation issue set. Without CI and
release governance, the issue set will not converge into a reproducible research
artifact.

## Goals

- Add CI for linting, unit tests, integration smoke, docs checks, and schema
  validation.
- Add issue labels and milestones aligned to the spec.
- Require PRs to cite spec/RFC references.
- Add changelog and release checklist.
- Define deprecation policy before public APIs expand.

## Non-Goals

- Heavy release bureaucracy before v0.1.
- Mandatory GPU CI for every pull request.
- Maintainer-only development workflow.

## Proposed Design

CI gates:

```yaml
pull_request:
  - lint
  - unit tests
  - integration smoke
  - docs link check
  - manifest schema validation
  - secret scan
push main:
  - same gates
```

Repository files:

```text
.github/workflows/ci.yml
CONTRIBUTING.md
SECURITY.md
CHANGELOG.md
docs/roadmap/IMPLEMENTATION.md
```

Issue traceability:

- implementation issues use `type:*`, `area:*`, `priority:*`, `effort:*`, and
  `spec:rfc-NNNN` labels;
- tracking issues group child issues by subsystem;
- PR template requires linked issue, spec, RFC, and validation commands.

Release gate:

```text
tests pass
docs links pass
manifest verifier passes
benchmark reports exist for claimed metrics
model/dataset cards match manifests
security and license reports pass
changelog updated
```

Failure modes:

- PR without spec link: blocked by review checklist;
- release without benchmark report for a claim: blocked;
- missing issue traceability: blocked.

## Alternatives Considered

- Issue-free implementation: rejected because the backlog requires complete
  decomposition and traceability.
- Manual release notes only: rejected because artifact manifests must back public
  claims.
- GPU CI mandatory: deferred because it would block contributors before CPU
  smoke tests are stable.

## Drawbacks

- CI setup work delays first model code.
- Traceability labels need maintenance.
- CPU smoke tests do not replace full GPU validation.

## Migration / Rollout

1. Create label taxonomy and milestones.
2. File spec-derived issues and tracking issues.
3. Add CI and contributor docs.
4. Add release checklist.
5. Add GPU validation workflow only after v0.1 CPU path is stable.

## Testing Strategy

- Validate workflow syntax.
- Run unit and integration tests locally.
- Test PR template links manually through issue/PR references.
- Add docs link checker.
- Add manifest verifier fixture.

## Open Questions

- Owner: maintainers. Target: 2026-06-30. Which GPU validation path is affordable
  for pre-release checks? Resolution: keep GPU validation manual until v0.1 CPU
  smoke and one single-GPU training run are reproducible.

## References

- `docs/spec/07-testing-strategy.md#ci-policy`
- `docs/spec/09-release-and-versioning.md`
- `docs/roadmap/IMPLEMENTATION.md`
