# Security

## Trust Boundaries

Inputs are untrusted:

- dataset rows;
- local repository code;
- patch candidates;
- command-line instructions;
- configs loaded from outside the repository.

Trusted code is the installed CodeLeWM package and its pinned runtime
dependencies. Trusted artifacts are those with valid manifests and checksums.

## Non-Execution Policy

CodeLeWM parses and transforms source text. It must not import, evaluate, run
tests for, or execute untrusted project code during dataset construction,
training, scoring, indexing, or evaluation.

## Secrets Handling

- Secrets are read only from the environment or the user's configured credential
  helpers.
- Secrets are never written to manifests, reports, logs, configs, or checkpoints.
- Debug logs redact values matching token, key, password, credential, and secret
  patterns.

## Dataset Licensing

Each source adapter declares:

```python
@dataclass(frozen=True)
class SourceLicensePolicy:
    source: SourceKind
    allowed_licenses: tuple[str, ...]
    require_license_field: bool
    redistribution_allowed: bool
    derived_artifact_policy: Literal["metadata_only", "embeddings", "full_text"]
```

Rows without acceptable license metadata are excluded from public dataset
artifacts unless a source-level license grants use.

## Public Artifact Policy

- Public dataset cards must state source mix, license policy, row counts, and
  known exclusions.
- Public model cards must state whether training used full source text,
  synthetic transforms, gated sources, or private local repositories.
- Public examples must not include private code or generated secrets.

## Threat Model

Threats:

- malicious source code designed to trigger parser or tokenizer bugs;
- poisoned transitions that encode secrets or private code;
- license contamination;
- split leakage that invalidates benchmark claims;
- checkpoint loading of untrusted serialized Python objects.

Controls:

- text-only parsing;
- tokenizer length limits;
- manifest checksums;
- safe checkpoint formats where possible;
- schema validation before loading;
- explicit allowlists for source licenses;
- no untrusted code execution.
