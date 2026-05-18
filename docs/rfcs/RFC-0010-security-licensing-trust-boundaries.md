# RFC-0010: Security, Licensing, And Trust Boundaries

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

CodeLeWM treats source data, local repositories, configs, and candidate patches
as untrusted input. The system parses and scores text but does not execute
untrusted code. Public artifacts must honor source licensing metadata.

## Motivation

The project operates on public and local source code, some of which may be
malicious, private, generated, or license-restricted. A local scorer must also be
safe to run inside a developer checkout.

## Goals

- Enforce non-execution during data build and scoring.
- Validate source license policies before public dataset or model release.
- Redact secrets from logs and reports.
- Prefer safe checkpoint formats where possible.
- Make trust boundaries visible in docs and errors.

## Non-Goals

- Sandboxed test execution in v0.1.
- Legal advice about every upstream dataset.
- Remote processing of local code.

## Proposed Design

Trust model:

```text
trusted:   installed CodeLeWM package, pinned dependencies, validated manifests
untrusted: source rows, local code, patches, user instructions, external configs
```

Controls:

- parse source text without importing modules;
- use timeouts for parser/tokenizer work;
- reject overlong rows before tokenization;
- load checkpoints only through validated paths;
- scan configs and logs for secret-like values;
- require source license policy before public artifact release.

License policy:

```python
@dataclass(frozen=True)
class LicenseDecision:
    allowed: bool
    reason: str
    source: SourceKind
    license: str | None
    artifact_policy: Literal["exclude", "metadata_only", "embeddings", "full_text"]
```

Failure modes:

- unknown license in public build: row excluded;
- config requests untrusted code execution: `SecurityBoundaryError`;
- serialized object checkpoint without trusted manifest: refused by default;
- private path in public report: release gate failure.

## Alternatives Considered

- Execute tests during scoring: rejected for v0.1 because it crosses the
  non-execution boundary.
- Ignore licensing until release: rejected because source metadata affects
  dataset design and model-card claims.
- Raw pickle checkpoint loading by default: rejected unless accompanied by a
  trusted manifest and local opt-in.

## Drawbacks

- Non-execution prevents some correctness checks.
- Strict licensing filters reduce dataset size.
- Safe checkpoint handling may require compatibility adapters.

## Migration / Rollout

1. Add non-execution tests.
2. Add license decision records.
3. Add redaction utilities.
4. Add checkpoint manifest checks.
5. Add release gate for public artifact policy.

## Testing Strategy

- Malicious fixture that would execute on import; verify it is never imported.
- Secret redaction tests.
- Unknown-license exclusion tests.
- Unsafe checkpoint refusal test.
- Public report scan test.

## Open Questions

- Owner: maintainers. Target: 2026-07-15. What is the strictest public model-card
  wording for models trained on mixed-license source? Resolution: publish only
  after license policy report is reviewed and the dataset card states source
  constraints.

## References

- `docs/spec/06-security.md`
- `docs/spec/04-error-model.md#failure-modes`
- `docs/rfcs/RFC-0002-edit-transition-dataset.md`
