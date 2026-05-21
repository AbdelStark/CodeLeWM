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
- optional `loss/action_use_margin`;
- optional `loss/action_use_margin_weighted`;
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
counts for same-before different-after, near-before different-after, same-file,
source, edit-size bucket, action-cluster, similarity, action-discriminative, and
fallback selection.
Action-contrast pool reports use
`schema_version=codelewm.eval.action_contrast_pool_report.v1` and are written by
retrieval evaluation as `reports/action_contrast_pool_report.json`. They record
exact-same-before, near-before, same-file, action-cluster, edit-shape, mutation,
and random control pools where available; every pool stores held-out transition
IDs only, split-membership proofs, unavailable-pool reasons, leakage counts, and
whether same-before multi-action examples make the no-action prior insufficient.
Latent probe reports use
`schema_version=codelewm.eval.latent_probe_report.v1` and are written by
`codelewm eval latent-probe` as `reports/latent_probe_report.json`. They record
train/validation/test label counts for edit class, AST node kind, symbol kind,
edit-size bucket, action cluster, and source family; probe metrics for
`z_before`, `z_after`, and `z_pred_after`; majority, lexical, metadata-only,
random-latent, no-action, and shuffled-action controls; bootstrap confidence
intervals; and per-dimension association diagnostics. Dimension-level semantic
claims remain blocked unless stable axes are demonstrated across seeds and
splits.
Latent matrix reports use
`schema_version=codelewm.eval.latent_matrix_report.v1` and are written by
`codelewm eval latent-matrix` as `reports/latent_matrix_report.json`. They
record latent shape, dimension count, sample count, split/source coverage,
finite-value status, per-dimension mean/std/variance/norm summaries, effective
rank, effective-rank ratio, mean pairwise cosine, covariance/correlation
summaries, bounded heatmap-ready covariance/correlation previews, inline
dimension-label association diagnostics, optional latent-probe report links,
and semantic-axis claim gates. Raw latent vectors are not serialized by
default. Dimension-level semantic claims remain blocked unless axes are stable
across seeds and splits and beat declared controls.
Action-discriminative shard reports use
`schema_version=codelewm.data.action_discriminative_shard_report.v1` and are
manifested by dataset build and pack artifacts as
`reports/action_discriminative_shard_report.json`.
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

Scorer/reranker quality reports use
`schema_version=codelewm.harness.scorer_quality_report.v1` and
`artifact_kind=score_report`. They store the labeled candidate rows used for
reranking, aggregate Recall@1/MRR/true-rank summaries, transition-energy,
retrieval-prior, and final-score distributions, calibration slices by candidate
kind, parse/patch failure counts, scoring-policy parameters, and the
non-execution policy. Any training or index manifests passed through
`--parent-manifest` are verified before the report is written and recorded in
the artifact manifest's `parent_artifacts`.

Transition indexes use `schema_version=codelewm.transition_index.v1` in
`index.json`, store train-split `state_after` vectors in `vectors.npy`, and
store one JSONL metadata record per vector in `entries.jsonl`. The index artifact
manifest records both the training-run artifact and packed-dataset artifact as
parents.

## Planned Visual Observability

The #235 visual model observability and TUI stream extends this contract with
diagnostic artifacts for model generation, checkpoint inspection, latent
geometry, and run monitoring. These artifacts are planned surfaces until their
implementation issues land; they must not replace the existing JSON reports,
JSONL logs, or artifact manifests.

Planned schemas:

- `codelewm.training.tensorboard_export.v1`: implemented
  TensorBoard-compatible event-log metadata for training/checkpoint scalars,
  bounded model-parameter histograms, and bounded latent summaries.
- `codelewm.model_checkpoint_inspection.v1`: implemented model/layer/tensor
  structure, parameter counts, tensor shapes, dtype/device metadata,
  finite-value checks, norms, checkpoint-manifest provenance, compatibility
  metadata, and bounded summary histograms.
- `codelewm.eval.latent_matrix_report.v1`: implemented latent dimension count,
  per-dimension statistics, effective rank, covariance/correlation summaries,
  probe associations, and semantic-axis claim gates.
- `codelewm.run_timeline.v1`: implemented ordered run steps, timestamps,
  durations, command ids, artifact ids, warnings, and typed failures.
- `codelewm.harness.visual_view_model.v1`: normalized data consumed by JSON,
  rich terminal, HTML, and Textual TUI views.

TensorBoard-compatible output and Textual rendering remain optional runtime
surfaces. Base package imports, fixture tests, JSON reports, and non-interactive
CLI usage must work without either dependency group. TensorBoard export is
enabled through `codelewm train --tensorboard`, requires the optional
observability dependency group, writes event files plus
`reports/tensorboard_export.json`, and includes both in the training artifact
manifest with checksums. Checkpoint inspection is enabled through
`codelewm model inspect-checkpoint`, verifies the checkpoint trust gate before
loading unless an explicit unsafe local override is selected, writes
`reports/model_checkpoint_inspection.json`, and includes the report in an
artifact manifest with checksums. Latent matrix diagnostics are enabled through
`codelewm eval latent-matrix`, verify dataset, training-run, and checkpoint
manifests before loading model weights, write
`reports/latent_matrix_report.json`, and include the report in an artifact
manifest with checksums. Run timeline reports are emitted today by
`codelewm llm-demo` and `codelewm eval latent-matrix` as
`reports/run_timeline.json`; they preserve append-only JSONL logging as the
operational stream while adding a manifest-backed summary for visual reports
and future TUI panels. Visual reports are diagnostic only and cannot
support positive semantic-latent-axis or coding-usefulness claims without the
relevant benchmark gates.

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
