# Public API

## CLI

The package exposes one console script:

```toml
[project.scripts]
codelewm = "codelewm.harness.cli:main"
```

Commands:

```bash
codelewm dataset build --config config/data/commitpackft.yaml --out data/codelewm_v0_1
codelewm dataset pack --manifest data/codelewm_v0_1/manifest.json --out data/codelewm_v0_1/hdf5
codelewm train --config config/train/codelewm_tiny.yaml
codelewm eval retrieval --checkpoint runs/v0_1/checkpoint.pt --data data/codelewm_v0_1/hdf5/test.hdf5
codelewm eval surprise --checkpoint runs/v0_1/checkpoint.pt --data data/codelewm_v0_1/hdf5/test.hdf5
codelewm index --checkpoint runs/v0_1/checkpoint.pt --data data/codelewm_v0_1/hdf5/train.hdf5 --out indexes/v0_1
codelewm score --before before.py --instruction instruction.txt --candidate after.py --checkpoint runs/v0_1/checkpoint.pt
codelewm rerank --before before.py --instruction instruction.txt --candidates patches/ --checkpoint runs/v0_1/checkpoint.pt
```

All commands support:

```bash
--json
--seed <int>
--device cpu|cuda|mps|auto
--log-level debug|info|warning|error
--artifact-dir <path>
```

## Python API

```python
from pathlib import Path
from codelewm.harness import load_scorer

scorer = load_scorer(Path("runs/v0_1/checkpoint.pt"), device="cuda")
result = scorer.score_files(
    before=Path("before.py"),
    instruction="add timeout handling to the retry loop",
    candidate=Path("after.py"),
)
```

`ScoreResult` schema:

```python
@dataclass(frozen=True)
class ScoreResult:
    schema_version: str
    transition_energy: float
    retrieval_prior: float | None
    risk_penalty: float | None
    final_score: float
    model_id: str
    checkpoint_sha256: str
    input_digest: str
    warnings: tuple[str, ...]
```

## Artifact Contracts

Every generated artifact directory contains:

```text
manifest.json
MANIFEST.sha256
config.yaml
reports/
```

`manifest.json` includes:

```python
@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    artifact_kind: Literal["dataset", "checkpoint", "index", "eval_report"]
    created_at: str
    source_git_sha: str
    command: list[str]
    config_sha256: str
    files: tuple[ManifestFile, ...]
    parent_artifacts: tuple[str, ...]
```

## Compatibility Policy

- Public CLI flags cannot be removed within a stable major version.
- JSON output fields can be added but not renamed or removed within a stable
  major version.
- Dataset schema changes require a new `schema_version`.
- Experimental APIs live under `codelewm.experimental` and carry no stability
  promise.

## Error Surface

The CLI exits with:

- `0`: success.
- `2`: invalid user input or config.
- `3`: source data unavailable.
- `4`: parse/filter contract failure.
- `5`: model or checkpoint incompatibility.
- `6`: evaluation gate failure.
- `70`: unexpected internal error.
