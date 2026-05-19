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

All parser-facing dataset and harness code routes Python validation through
`codelewm.security.parse_python_source_text`, which is limited to text parsing
with `ast.parse`. It returns syntax trees for inspection and transformation; it
does not compile modules, import project packages, call user functions, run test
commands, or load user-supplied Python plugins.

Configs loaded from outside the package are also part of the trust boundary.
Validation rejects explicit execution requests such as `run_tests`,
`execute_user_code`, `eval_user_code`, `import_user_modules`, and
`test_command`. CodeLeWM may support sandboxed execution later, but that must be
a separate opt-in subsystem with its own isolation, manifest, and logging
contract.

## Secrets Handling

- Secrets are read only from the environment or the user's configured credential
  helpers.
- Secrets are never written to manifests, reports, logs, configs, or checkpoints.
- Debug logs redact values matching token, key, password, credential, and secret
  patterns.

`codelewm.security.scan_paths` walks one or more files or directories and
returns a `codelewm.secret_scan.v1` report listing every match by path, line,
pattern name, and SHA-256-truncated digest. The scanner intentionally never
echoes a raw secret value into its output: every match is reported only as
`[REDACTED_SECRET sha256=... length=...]`. Release gates and CI run the scanner
over generated reports, log directories, and any other shareable artifact. The
scanner's pattern set is kept aligned with the redactor in
`codelewm.observability.logging` so that anything the scanner can detect is
also stripped from structured logs by default.

## Checkpoint Trust

Checkpoints are loaded only behind the trust boundary:

- Every checkpoint must ship with a `codelewm.checkpoint.v1` manifest at
  `<checkpoint>.manifest.json` (the default path) or at an explicitly passed
  manifest location.
- The manifest is validated for schema, file existence, and SHA-256 match
  before any code attempts to deserialize the checkpoint.
- `codelewm.security.require_trusted_checkpoint` is the single function any
  loader must call before reading a checkpoint file.
- `codelewm.harness.load_scorer` calls the gate by default. Callers can pass
  `allow_unsafe=True` (Python API) or `--allow-unsafe-checkpoint` (CLI) to
  bypass the gate in a trusted local environment; the override exists for
  fixture-only workflows and is the only documented escape hatch.
- Tampered or missing-manifest loads are refused with a structured
  `checkpoint_error` report; no partial state is created on the loader's
  behalf.

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

The default public full-text artifact policy allows only clearly permissive
licenses:

```python
PERMISSIVE_PUBLIC_LICENSES = (
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc0-1.0",
    "isc",
    "mit",
    "unlicense",
)
```

Missing, unknown, copyleft, or otherwise non-allowlisted license values produce
`LicenseDecision(allowed=False, artifact_policy="exclude")`. The decision is
attached to filter drop records so public dataset reports can account for every
license exclusion.

Public artifact gates use `schema_version=codelewm.public_license_gate.v1`:

```python
@dataclass(frozen=True)
class PublicLicenseGateReport:
    schema_version: str
    artifact_policy: Literal["exclude", "metadata_only", "embeddings", "full_text"]
    included_rows: int
    excluded_rows: int
    blocked_rows: int
    release_allowed: bool
    included_licenses: Mapping[str, int]
    excluded_licenses: Mapping[str, int]
    included_sources: Mapping[str, int]
    excluded_sources: Mapping[str, int]
    excluded_reasons: Mapping[str, int]
```

Public full-text artifacts fail the gate when any included row has a denied
decision or an artifact policy other than `full_text`. Excluded rows are allowed
only when their decision is counted in the gate report. Dataset manifests include
an included-row license summary and can embed the full gate report under
`metadata.license_gate_report`.

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

- text-only parsing through the shared non-execution parser;
- rejection of configs that request untrusted code execution;
- tokenizer length limits;
- manifest checksums;
- safe checkpoint formats where possible;
- schema validation before loading;
- explicit allowlists for source licenses;
- no untrusted code execution.
