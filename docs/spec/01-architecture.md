# Architecture

## System Shape

CodeLeWM has four bounded subsystems:

- `codelewm.data`: builds transition datasets.
- `codelewm.model`: encodes states/actions and predicts latent transitions.
- `codelewm.eval`: measures retrieval, surprise, collapse, and baselines.
- `codelewm.harness`: indexes historical transitions and scores candidate edits.

The existing root files `jepa.py`, `module.py`, `train.py`, and `eval.py` are the
LeWM-derived implementation seed. New code must move project-specific behavior
under `codelewm/` while preserving compatibility with Hydra and Lightning-style
training entry points.

## Data Flow

```text
raw sources
  -> source loader
  -> parser and license filters
  -> changed-symbol extractor
  -> CodeState builder
  -> action extractor
  -> deduplicator
  -> split assigner
  -> Parquet shards
  -> HDF5 packs
  -> training dataloader
  -> model checkpoint
  -> retrieval index
  -> scorer/reranker CLI
```

Every arrow emits either a manifest entry or a structured error record. Silent
row drops are forbidden.

## Module Boundaries

### `codelewm.data`

Public functions:

```python
def load_source(spec: SourceSpec) -> Iterable[RawEditRecord]: ...
def build_transition(raw: RawEditRecord, cfg: BuildConfig) -> TransitionRecord: ...
def assign_split(record: TransitionRecord, policy: SplitPolicy) -> SplitName: ...
def write_parquet(records: Iterable[TransitionRecord], out: Path) -> ShardManifest: ...
def pack_hdf5(shards: Sequence[Path], out: Path, schema: SchemaVersion) -> HDF5Manifest: ...
```

This package owns parsing, normalization, filtering, deduplication, split policy,
and artifact manifests. It does not own model tokenization beyond producing
stable tokenizable fields and optional pre-tokenized arrays.

### `codelewm.model`

Public classes:

```python
class CodeStateEncoder(nn.Module): ...
class TextActionEncoder(nn.Module): ...
class AbstractActionEncoder(nn.Module): ...
class CodeTransitionModel(nn.Module): ...
```

The model package owns tensor contracts, device/dtype behavior, and latent
prediction. It does not read raw datasets or decide split membership.

### `codelewm.eval`

Public functions:

```python
def evaluate_retrieval(model: CodeTransitionModel, dataset: EvalDataset, cfg: RetrievalEvalConfig) -> RetrievalReport: ...
def evaluate_surprise(model: CodeTransitionModel, dataset: EvalDataset, cfg: SurpriseEvalConfig) -> SurpriseReport: ...
def compute_collapse_metrics(embeddings: Tensor) -> CollapseReport: ...
```

Evaluation owns negative sampling, metrics, baselines, and reports. It must not
train models or mutate datasets.

### `codelewm.harness`

Public functions:

```python
def build_index(checkpoint: Path, transitions: Path, out: Path) -> IndexManifest: ...
def score_candidate(model: CodeTransitionModel, before: CodeState, action: EditAction, candidate: CodeState) -> ScoreResult: ...
def rerank_candidates(model: CodeTransitionModel, request: RerankRequest) -> RerankResult: ...
```

The harness owns user-facing scoring and retrieval. It does not generate patches
for v0.1; it accepts candidates from callers.

## Load-Bearing Invariants

- INV-ARCH-001: Training and evaluation consume the same schema-versioned
  `TransitionRecord` contract.
- INV-ARCH-002: A row's split is assigned before tokenization and cannot be
  changed by downstream packing.
- INV-ARCH-003: The headline inference path uses `action_text`, not `action_patch`.
- INV-ARCH-004: `state_after` is never available to action encoders in the
  inference path.
- INV-ARCH-005: All model outputs include latent dimension, dtype, device, and
  schema version in their report metadata.

## Existing Repository Maturity

The current checkout is a minimal LeWM codebase: root model/training scripts,
Hydra configs for image-control environments, and no packaging metadata. The
project-specific CodeLeWM package, data pipeline, evaluation pipeline, CLI, tests,
CI, and docs are not implemented yet. This spec therefore defines a bootstrap
path rather than documenting an already complete system.
