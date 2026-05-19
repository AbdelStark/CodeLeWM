# Observability

## Logging

Commands that support local machine-readable logging expose `--log-jsonl <path>`
and append one JSON object per line. JSON command output remains reserved for
the command result itself.

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

Structured log events use `schema_version=codelewm.log_event.v1`.
Human-readable logs are allowed by default, but every release gate consumes JSONL
or report JSON. Current harness commands emit start, completion, and request
error events when `--log-jsonl` is provided.

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
Hard-negative sampler reports use
`schema_version=codelewm.eval.hard_negative_sampler_report.v1` and include
selected negative IDs, target/train rejection counts, and aggregate composition
counts for source, edit-size bucket, action-cluster, similarity, and fallback
selection.
Headline retrieval reports are invalid unless the `baselines` mapping includes
`random`, `lexical`, `no_action`, and `shuffled_action` metrics.

Action-view ablation reports use
`schema_version=codelewm.eval.action_ablation_report.v1`. They store one row per
expected action-view, baseline, retrieval-loss, or collapse-setting variant.
Rows are never silently dropped: unavailable variants are recorded as `blocked`
with a reason, and patch-action rows must be tagged as diagnostic upper bounds.

Patch-surprise reports use
`schema_version=codelewm.eval.surprise_report.v1`. They store transition-energy
scores for the true after-state and decoys, aggregate pairwise AUC overall and
by category, record per-example true ranks, and include category caveats in
metadata when the held-out data cannot support random, same-file, mutation, or
action-cluster decoys.

Transition indexes use `schema_version=codelewm.transition_index.v1` in
`index.json`, store train-split `state_after` vectors in `vectors.npy`, and
store one JSONL metadata record per vector in `entries.jsonl`. The index artifact
manifest records both the training-run artifact and packed-dataset artifact as
parents.

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

The shared log redaction layer:

- redacts fields whose keys contain token, key, password, credential, or secret
  markers;
- redacts common secret-looking values such as API tokens;
- replaces the current user's home directory prefix with `~`;
- replaces long text payloads with a digest-bearing placeholder instead of
  storing raw source snippets.

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
