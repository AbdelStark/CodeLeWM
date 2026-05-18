# Architecture

## System Shape

CodeLeWM has four bounded subsystems:

- `codelewm.data`: builds transition datasets.
- `codelewm.model`: encodes states/actions and predicts latent transitions.
- `codelewm.eval`: measures retrieval, surprise, collapse, and baselines.
- `codelewm.harness`: indexes historical transitions and scores candidate edits.

The LeWM-derived implementation seed now lives under package boundaries:

- `codelewm.model.jepa` contains the JEPA wrapper.
- `codelewm.model.modules` contains SIGReg, the autoregressive predictor, and
  supporting network blocks.
- `codelewm.training.utils` contains current training helpers.

Root `jepa.py`, `module.py`, and `utils.py` remain compatibility wrappers for
existing Hydra configs and scripts. Root `train.py` and `eval.py` remain
LeWM-derived entry points until the CodeLeWM training and evaluation runners are
implemented.

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
class CodeStateBatch: ...
class ActionBatch: ...
class TransitionBatch: ...
class CodeStateEncoder(nn.Module): ...
class TextActionEncoder(nn.Module): ...
class AbstractActionEncoder(nn.Module): ...
class CodeTransitionModel(nn.Module): ...
```

The model package owns tensor contracts, device/dtype behavior, transition
energy, and latent prediction. The initial interface contracts are importable
without the optional ML runtime so package and CLI checks can run in lightweight
environments; concrete encoders and predictors remain ML-runtime-backed. The
model package does not read raw datasets or decide split membership.

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
