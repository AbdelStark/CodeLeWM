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
from codelewm.model import AbstractActionEncoderConfig, TextActionEncoderConfig

scorer = load_scorer(Path("runs/v0_1/checkpoint.pt"), device="cuda")
result = scorer.score_files(
    before=Path("before.py"),
    instruction="add timeout handling to the retry loop",
    candidate=Path("after.py"),
)
```

Public model configuration helpers include `TextActionEncoderConfig` for the
headline text action path and `AbstractActionEncoderConfig` for structural
ablation runs. Both project action encodings to the v0.1 latent dimension.

Training is exposed through the package runner:

```python
from codelewm.training import load_train_config, train

cfg = load_train_config("config/train/codelewm_tiny.yaml")
manifest = train(cfg, executor=executor)
```

The runner owns config validation, parent dataset manifest validation, output
layout, metrics files, checkpoint hashes, and training-run manifests. Concrete
CPU/GPU model execution is supplied by an executor implementation.

Retrieval evaluation exposes the metric and report contract independently of the
model runtime:

```python
from codelewm.eval import (
    HardNegativeSamplerConfig,
    build_baseline_metrics,
    build_easy_candidate_pool,
    build_hard_candidate_pool,
    build_retrieval_report,
    lexical_baseline_ranks,
    rank_targets,
    random_baseline_ranks,
    validate_required_headline_baselines,
)

pool = build_easy_candidate_pool(rows, max_size=1000, seed=0)
hard_pool, hard_negative_sample = build_hard_candidate_pool(
    query,
    rows,
    config=HardNegativeSamplerConfig(max_negatives=1000),
)
ranks = rank_targets(score_rows, candidate_ids_by_query, target_ids)
baselines = build_baseline_metrics({
    "random": random_baseline_ranks(candidate_ids_by_query, target_ids),
    "lexical": lexical_baseline_ranks(query_texts, candidate_texts, candidate_ids_by_query, target_ids),
    "no_action": no_action_ranks,
    "shuffled_action": shuffled_action_ranks,
})
report = validate_required_headline_baselines(
    build_retrieval_report(ranks, candidate_pool=pool, baselines=baselines)
)
```

Reports use `schema_version=codelewm.eval.retrieval_report.v1`; candidate pools
use `schema_version=codelewm.eval.candidate_pool.v1` and must exclude `train`
split rows.

`ScoreResult` schema:

```python
@dataclass(frozen=True)
class ScoreResult:
    schema_version: str
    candidate: str
    transition_energy: float
    retrieval_prior: float | None
    risk_penalty: float | None
    final_score: float
    model_id: str
    checkpoint_sha256: str
    input_digest: str
    warnings: tuple[str, ...]
```

`load_scorer` verifies the checkpoint path and records its SHA-256 before
scoring. The initial runtime-light backend is deterministic and intended for API
and fixture validation; model-backed checkpoint execution can replace the backend
without changing `ScoreResult`.

JSON schema helper functions are available for automation:

```python
from codelewm.harness import (
    error_report_json_schema,
    rerank_result_json_schema,
    score_result_json_schema,
)
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
    artifact_id: str
    artifact_kind: Literal[
        "dataset",
        "checkpoint",
        "training_run",
        "index",
        "eval_report",
        "score_report",
    ]
    created_at: str
    source_git_sha: str
    command: list[str]
    config_sha256: str
    parent_artifacts: tuple[str, ...]
    files: tuple[ManifestFile, ...]
    metadata: Mapping[str, Any]
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
