# RFC-0009: Observability And Artifact Lineage

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

Every build, train, eval, index, and score command emits structured logs and
schema-versioned manifests. Claims must be traceable to artifact IDs, config
hashes, and checksums.

## Motivation

Dataset leakage, noisy filters, collapse, and benchmark drift can invalidate the
project. Observability is therefore part of correctness, not an optional
operator feature.

## Goals

- Emit JSONL logs for machine-readable validation.
- Write manifests for every artifact directory.
- Record parent artifact IDs.
- Include collapse and evaluation reports in release artifacts.
- Redact secrets and private paths.

## Non-Goals

- External telemetry by default.
- Remote logging requirement.
- Storing full source text in logs.

## Proposed Design

Manifest schema:

```python
@dataclass(frozen=True)
class ManifestFile:
    path: str
    sha256: str
    bytes: int

@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    artifact_id: str
    artifact_kind: Literal[
        "candidate_pack",
        "dataset",
        "demo_report",
        "downstream_benchmark",
        "checkpoint",
        "training_run",
        "index",
        "eval_report",
        "score_report",
    ]
    created_at: str
    source_git_sha: str
    command: tuple[str, ...]
    config_sha256: str
    parent_artifacts: tuple[str, ...]
    files: tuple[ManifestFile, ...]
```

Event schema:

```python
@dataclass(frozen=True)
class LogEvent:
    schema_version: str
    run_id: str
    event: str
    level: str
    step: str
    fields: Mapping[str, Any]
```

Required reports:

- dataset build report;
- training run report;
- collapse report;
- retrieval report;
- surprise report;
- score report for harness commands.

Failure modes:

- manifest checksum mismatch: artifact refuses to load;
- missing parent artifact: warning for local dev, failure for release gate;
- unredacted secret pattern: CI failure.

## Alternatives Considered

- Console logs only: rejected because release gates need structured evidence.
- One global manifest: rejected because artifacts are built and reused
  independently.
- Raw source snippets in logs: rejected for privacy and licensing reasons.

## Drawbacks

- More files in every artifact directory.
- Strict checksum validation can slow local development.
- Redaction can hide useful debug details unless opt-in debug flags are used.

## Migration / Rollout

1. Add manifest dataclasses and validation.
2. Add dataset and checkpoint manifests.
3. Add eval and score reports.
4. Add CI schema validation and secret scanning.

## Testing Strategy

- Manifest round-trip unit tests.
- Checksum mismatch test.
- Parent artifact lineage test.
- Redaction tests for path and secret patterns.
- CI fixture that validates all report schemas.

## Open Questions

- Owner: maintainers. Target: 2026-06-30. Should artifact IDs use content hashes
  only or include run timestamps? Resolution: use content hash plus short run ID
  in v0.1 and revisit if cache collisions or readability issues appear.

## References

- `docs/spec/05-observability.md`
- `docs/spec/02-public-api.md#artifact-contracts`
- `docs/spec/07-testing-strategy.md#ci-policy`
