# Error Model

## Exception Types

```python
class CodeLeWMError(Exception): ...
class ConfigError(CodeLeWMError): ...
class SourceUnavailableError(CodeLeWMError): ...
class LicensePolicyError(CodeLeWMError): ...
class ParseError(CodeLeWMError): ...
class FilteredRecord(CodeLeWMError): ...
class SchemaError(CodeLeWMError): ...
class SplitLeakageError(CodeLeWMError): ...
class CheckpointCompatibilityError(CodeLeWMError): ...
class EvaluationGateError(CodeLeWMError): ...
class SecurityBoundaryError(CodeLeWMError): ...
```

Every exception exposed through the CLI serializes to:

```python
@dataclass(frozen=True)
class ErrorReport:
    schema_version: str
    error_type: str
    message: str
    remediation: str
    record_id: str | None
    artifact: str | None
    caused_by: str | None
```

Harness CLI error records use `schema_version=codelewm.error.v1`. Current
`error_type` values are:

- `missing_file`;
- `invalid_syntax`;
- `patch_apply_failed`;
- `checkpoint_error`;
- `scoring_error`.

`codelewm score --json` writes an `ErrorReport` JSON object and exits with code
`2` for invalid input. Error messages and causes must not include long raw source
snippets.

## Failure Modes

### Source unavailable

Response: fail the source job, record `SourceUnavailableError`, and continue only
if the config declares a fallback source. Silent source substitution is forbidden.

### License denied

Response: drop the row, record `LicensePolicyError`, and report counts by source
and license. Training jobs fail if the denied rate exceeds the configured budget.

### Parse failure

Response: drop row with `ParseError`, including parser name, path, and failure
category. Parser tracebacks are logged only at debug level.

### Split leakage

Response: fail the dataset build with `SplitLeakageError`. Leakage is a hard gate.

### Embedding collapse

Response: fail the training gate with `EvaluationGateError` when effective rank,
per-dimension variance, or nearest-neighbor entropy crosses kill thresholds.

### Checkpoint/schema mismatch

Response: refuse to load with `CheckpointCompatibilityError` unless an explicit
migration command exists and writes a new manifest.

### User code execution

Response: CodeLeWM must parse and transform text. It must not import, execute, or
run untrusted project code during dataset construction or scoring.

## Recovery Rules

- Recoverable row-level failures are emitted as structured filter records.
- Artifact-level failures abort the command and leave incomplete outputs under a
  `.partial` suffix.
- Commands never overwrite completed artifacts unless `--overwrite` is passed.
- `--overwrite` first verifies that the target is a CodeLeWM artifact directory.

## Redaction

Error reports redact:

- environment variables;
- authentication tokens;
- absolute home-directory paths unless `--debug-paths` is passed;
- source text snippets longer than 20 lines.
