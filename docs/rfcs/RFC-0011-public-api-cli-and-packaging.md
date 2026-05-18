# RFC-0011: Public API, CLI, And Packaging

- Status: Accepted
- Authors: CodeLeWM maintainers
- Created: 2026-05-18
- Target milestone: v0.1

## Summary

CodeLeWM becomes an installable Python package with a stable CLI entrypoint,
typed public dataclasses, JSON output schemas, and contributor-facing validation
commands.

## Motivation

The current checkout lacks packaging metadata and exposes root scripts inherited
from the LeWM seed. Contributors need a clear package boundary and users need
commands that do not require knowing internal script names.

## Goals

- Add `pyproject.toml` with package metadata and console script.
- Move project code under `codelewm/`.
- Keep root scripts as compatibility wrappers where useful.
- Define stable JSON outputs for dataset, train, eval, score, and rerank.
- Provide public Python scorer API.

## Non-Goals

- Stable `1.0` API before the v1.0 research artifact.
- Plugin ecosystem.
- Web service.

## Proposed Design

Package structure:

```text
codelewm/
  data/
  model/
  eval/
  harness/
  schemas/
  utils/
```

Entrypoint:

```toml
[project.scripts]
codelewm = "codelewm.harness.cli:main"
```

CLI command groups:

- `dataset build`;
- `dataset pack`;
- `train`;
- `eval retrieval`;
- `eval surprise`;
- `index`;
- `score`;
- `rerank`;
- `manifest verify`.

Public API:

```python
def load_scorer(checkpoint: Path, *, device: str = "auto") -> CodeLeWMScorer: ...

class CodeLeWMScorer:
    def score_files(self, before: Path, instruction: str, candidate: Path) -> ScoreResult: ...
    def rerank_patches(self, before: Path, instruction: str, candidates: Sequence[Path]) -> RerankResult: ...
```

JSON compatibility:

- every JSON output contains `schema_version`;
- fields can be added in `0.x`;
- fields cannot be silently renamed without changelog and migration note.

Failure modes:

- missing optional dependency: command returns config error with install hint;
- unsupported schema: command refuses to run;
- invalid CLI combination: exit code `2`.

## Alternatives Considered

- Keep script-only repo: rejected because it prevents stable issue decomposition
  and public API testing.
- Notebook-first workflow: rejected because reproducibility and CI gates require
  command-line contracts.
- Multiple CLI binaries: rejected because one command group is easier to
  document and version.

## Drawbacks

- Packaging adds maintenance burden.
- Compatibility wrappers can hide deprecated paths if not managed.
- JSON schema discipline slows rapid CLI experimentation.

## Migration / Rollout

1. Add `pyproject.toml` and package skeleton.
2. Move new implementation under `codelewm/`.
3. Add console script.
4. Add wrappers or deprecation warnings for root scripts.
5. Add API docs and JSON schema tests.

## Testing Strategy

- `python -m build` or equivalent package build test.
- CLI help snapshot tests.
- JSON schema validation tests.
- Public scorer API fixture test.
- Deprecation warning test for compatibility wrappers.

## Open Questions

- Owner: maintainers. Target: 2026-06-30. Should root `train.py` and `eval.py`
  stay as wrappers through v1.0? Resolution: keep wrappers in v0.1, decide after
  package CLI is stable.

## References

- `docs/spec/02-public-api.md`
- `docs/spec/09-release-and-versioning.md`
- `docs/rfcs/RFC-0008-agent-harness-scorer-reranker.md`
