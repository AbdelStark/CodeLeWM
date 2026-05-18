# Observability

## Logging

All commands emit structured JSONL logs when `--json` is enabled:

```python
@dataclass(frozen=True)
class LogEvent:
    schema_version: str
    event: str
    level: Literal["debug", "info", "warning", "error"]
    run_id: str
    artifact_id: str | None
    step: str
    message: str
    fields: Mapping[str, Any]
```

Human-readable logs are allowed by default, but every release gate consumes JSONL
or report JSON.

## Metrics

Dataset metrics:

- source row count;
- parse success rate;
- filter counts by reason;
- dedup rejection rate;
- split counts by repo and source;
- token length percentiles;
- edit size percentiles;
- license counts.

Training metrics:

- `loss/total`;
- `loss/prediction_mse`;
- `loss/sigreg`;
- `loss/sigreg_weighted`;
- optional `loss/retrieval`;
- optional `loss/retrieval_weighted`;
- embedding effective rank;
- embedding effective rank ratio;
- per-dimension variance min/median/max;
- mean pairwise cosine;
- embedding norm mean/std;
- nearest-neighbor entropy;
- gradient norm;
- throughput examples per second;
- peak memory.

Evaluation metrics:

- `Recall@1`, `Recall@5`, `Recall@10`;
- MRR;
- median rank;
- patch-surprise AUC;
- baseline deltas;
- per-source and per-edit-size slices.

Retrieval reports use `schema_version=codelewm.eval.retrieval_report.v1` and
carry the same top-level metrics as the RFC:

```python
@dataclass(frozen=True)
class RetrievalReport:
    schema_version: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    median_rank: float
    candidate_pool: CandidatePool | None
    baselines: Mapping[str, RetrievalMetrics]
    slices: Mapping[str, RetrievalMetrics]
    metadata: Mapping[str, Any]
```

Candidate pools use `schema_version=codelewm.eval.candidate_pool.v1`, store
held-out transition IDs, and reject any `train` split rows. The v0.1 easy pool is
a deterministic random sample of up to 1,000 validation/test after-states.

## Artifact Lineage

Every dataset, checkpoint, index, and report includes:

- source git SHA;
- command argv;
- config hash;
- parent artifact IDs;
- file checksums;
- schema version;
- created timestamp in UTC;
- host platform and Python version;
- dependency lock hash when available.

Shared artifact manifests use `schema_version=codelewm.artifact_manifest.v1`.
The manifest is JSON-native and validates before any release gate consumes it:

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
        "dataset",
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
    metadata: Mapping[str, Any]
```

Manifest file paths are relative to the artifact root and cannot contain `..` or
absolute paths. Checksums are lowercase SHA-256 digests over file bytes.
`parent_artifacts` records upstream manifest IDs, such as the dataset manifest
used by a training run or the checkpoint manifest used by an evaluation report.

## Redaction Rules

Logs and reports must not include:

- secrets;
- raw environment variable values;
- full source code snippets by default;
- absolute user home paths by default;
- private repository names unless the input source is explicitly marked public.

## Kill Reports

Training and evaluation gates write a kill report when they fail. The report must
name the failed threshold, observed value, command, config hash, and suggested
next RFC-governed action.

Collapse kill reports use `schema_version=codelewm.eval.kill_report.v1` and
include:

- `reason=embedding_collapse`;
- `collapse_report` with effective-rank, variance, cosine, norm, and
  nearest-neighbor entropy metrics;
- one entry per failed threshold;
- command and config hash when available.
