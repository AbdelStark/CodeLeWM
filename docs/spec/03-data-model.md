# Data Model

## Core Types

```python
SchemaVersion = Literal["codelewm.transition.v1"]
SplitName = Literal["train", "val", "test"]
ActionView = Literal["text", "abstract", "patch"]
SourceKind = Literal["commitpackft", "commitpack", "agentpack", "synthetic", "local_repo"]
AdapterKind = SourceKind | Literal["fixture"]
```

`AdapterKind` names source adapters. The `fixture` adapter is for deterministic
tests and local smoke data; emitted `RawEditRecord.source` values still use the
canonical `SourceKind` set.

The `commitpackft` adapter accepts local `.jsonl` and `.jsonl.gz` shards, or a
directory containing those shards. It streams rows without materializing the full
source and maps CommitPackFT fields into `RawEditRecord` as follows:

| CommitPackFT field | RawEditRecord field |
| --- | --- |
| `repos` | `repo` |
| `commit` | `commit` |
| `old_file` | `path_before` |
| `new_file` | `path_after` |
| `old_contents` | `before` |
| `new_contents` | `after` |
| `message` | `message` |
| `license` | `license` |

Rows must declare `lang`/`language` as `Python` unless the source spec overrides
the expected language. The adapter stores `subject`, shard path, line number, and
language in record metadata.

```python
@dataclass(frozen=True)
class RawEditRecord:
    source: SourceKind
    repo: str
    commit: str
    path_before: str
    path_after: str
    before: str
    after: str
    message: str
    license: str | None
    timestamp: str | None
    metadata: Mapping[str, Any]
```

```python
@dataclass(frozen=True)
class CodeState:
    language: Literal["python"]
    path: str
    module: str
    symbol: str | None
    kind: Literal["function", "method", "class", "region", "small_file"]
    imports: str
    enclosing_class: str | None
    sibling_signatures: tuple[str, ...]
    callee_signatures: tuple[str, ...]
    primary: str
    token_count: int
    changed_hunk_mask: tuple[bool, ...]
    fallback_reason: str | None
```

`CodeState` normalization emits the pack format from RFC-0003 after:

- dedenting primary code and normalizing whitespace through Python AST unparse
  when the primary block parses;
- replacing over-budget string and numeric literals with typed placeholders;
- removing docstrings by default;
- preserving identifiers in the main view.

Truncation is structured, not arbitrary tail clipping. The normalizer first drops
lower-priority sibling signatures, then callee signatures, then import context.
Only after those sections are removed may it reduce `primary` to signature or
decorator lines, changed-hunk vicinity, returns, raises, exception handlers, and
context-manager lines. Rows that still exceed the token budget fail instead of
being silently clipped.

```python
@dataclass(frozen=True)
class EditAction:
    text: str
    abstract: tuple[str, ...]
    patch: str | None
    patch_is_leaky: bool
    metadata: Mapping[str, Any]
```

`action_text` is the normalized row message or synthetic transform template,
truncated to the configured action-text length. `action_abs` contains structural
operation tokens only and must not include full inserted after-code lines.
`action_patch` is optional diagnostic data; when present, `patch_is_leaky=true`
and it cannot be used for headline inference.

```python
@dataclass(frozen=True)
class TransitionRecord:
    schema_version: SchemaVersion
    transition_id: str
    source: SourceKind
    repo: str
    commit: str
    path: str
    state_before: CodeState
    state_after: CodeState
    action: EditAction
    split: SplitName
    filter_flags: tuple[str, ...]
    dedup_keys: tuple[str, ...]
    license: str | None
```

## HDF5 Layout

```text
/state_before/input_ids           int32 [N, 1024]
/state_before/attention_mask      bool  [N, 1024]
/state_before/segment_ids         int16 [N, 1024]
/state_before/changed_hunk_mask   bool  [N, 1024]
/state_after/input_ids            int32 [N, 1024]
/state_after/attention_mask       bool  [N, 1024]
/state_after/segment_ids          int16 [N, 1024]
/action_text/input_ids            int32 [N, 256]
/action_text/attention_mask       bool  [N, 256]
/action_abs/input_ids             int32 [N, 192]
/action_abs/attention_mask        bool  [N, 192]
/action_patch/input_ids           int32 [N, 512]
/action_patch/attention_mask      bool  [N, 512]
/metadata/repo                    utf8  [N]
/metadata/path                    utf8  [N]
/metadata/commit                  utf8  [N]
/metadata/source                  int8  [N]
/metadata/split                   int8  [N]
/metadata/edit_size               int32 [N]
/metadata/token_count_before      int32 [N]
/metadata/token_count_after       int32 [N]
```

`action_patch` can be omitted from v0.1 packs. If omitted, the manifest must set
`features.action_patch=false`.

State `segment_ids` use stable section codes:

| Segment | ID |
| --- | ---: |
| path/module/symbol markers | 1 |
| imports | 2 |
| class/kind/enclosing-class markers | 3 |
| sibling signatures | 4 |
| callee signatures | 5 |
| primary code | 6 |

`changed_hunk_mask` is aligned to the tokenized normalized state and is true only
for primary-code tokens that originate from changed primary lines retained after
structured truncation.

Packed HDF5 files store `schema_version`, `features.action_patch`, and
`row_count` as root attributes. Dataset manifests are JSON objects with:

- `schema_version`;
- `row_count`;
- `features`;
- artifact entries with `path`, `kind`, `rows`, `sha256`, and `bytes`;
- `split_counts`;
- `source_counts`.

Manifest checksums are SHA-256 over artifact bytes. Parquet staging shards are
written before HDF5 packing and are listed alongside the HDF5 artifact.

## Filtering Rules

Keep rows only when:

- language is Python;
- before and after parse successfully;
- old and new paths end in `.py`;
- before and after are non-empty;
- changed lines are `1..150`;
- changed files are at most `5` for multi-file source records;
- primary state is at most `1024` state tokens after structured truncation;
- edit distance ratio is in `[0.02, 0.60]`;
- action text has `8..512` characters;
- license is permissive or the source license permits the intended use.

Drop rows with:

- revert, work-in-progress, vendor, generated, migration, lockfile, snapshot, or
  dependency-bump indicators;
- whitespace-only changes;
- comment/docstring-only changes unless the synthetic task is explicitly a
  documentation edit;
- huge literal/table changes;
- syntax-invalid after states.

Filter passes must emit machine-readable drop records. At minimum each dropped
row records a stable row identifier, a reason code, a human-readable message,
and structured details. The initial reason-code set is:

```python
DropReasonCode = Literal[
    "parse_error",
    "non_python_path",
    "empty_state",
    "whitespace_only_change",
    "edit_size",
    "edit_ratio",
    "message_length",
    "generated_file",
    "license_denied",
]
```

Filter reports include `total_before`, `total_after`, `total_dropped`, and
`drop_reasons` counts. Silent row drops are invalid.

## Split Policy

Primary split key:

```python
split_key = normalized_repo_name if repo else source_identity
```

Policy:

- `train`: 80% of split keys.
- `val`: 10% of split keys.
- `test`: 10% of split keys.
- Synthetic transforms inherit the split of their source file before generation.
- Rows cannot move split after packing.
- Split assignment is deterministic from `sha256(seed + split_key)` and happens
  before tokenization.

## Deduplication

Dedup keys:

- exact normalized `(before, action_text, after)` SHA-256;
- exact normalized `(before, after)` SHA-256;
- near-duplicate 64-bit SimHash for `state_before`;
- diff-shape hash for operation histogram and size bucket.

Validation and test rows are rejected if their near-duplicate distance to train
rows is below the configured Hamming threshold. The v0.1 default threshold is
`3`.

Split/dedup reports include `total_before`, `total_after`, `total_dropped`,
per-split kept counts, and drop-reason counts. Drop reasons are:

```python
DedupDropReasonCode = Literal["exact_duplicate", "train_leakage"]
```

## Retrieval Candidate Metadata

Retrieval candidate pools reference packed transition rows by `transition_id`.
The hard-negative sampler consumes the following fields from each candidate:

- `transition_id`;
- `split`;
- `source`;
- `repo`;
- `path`;
- `edit_size`;
- optional `metadata.action_cluster`, `metadata.weak_action_cluster`, or
  `metadata.action_abs_cluster`;
- optional `metadata.similarity`, `metadata.similarity_to_query`,
  `metadata.state_similarity`, or `metadata.lexical_similarity`.

Candidate pools must reject `train` split rows. Hard-negative reports must record
how many selected negatives match source, edit-size bucket, action cluster,
similarity metadata, and fallback selection.

Lexical baselines operate on caller-supplied query and candidate text. Reports
must not embed raw source code by default; if candidate text is persisted, it
must already be an approved redacted or fixture-safe representation.

## Invariants

- INV-DATA-001: `state_after` must not appear in `action_text` or `action_abs`.
- INV-DATA-002: `action_patch` is tagged as leaky and excluded from headline
  inference evaluation.
- INV-DATA-003: Every row has one split and one source.
- INV-DATA-004: Every dropped row has a machine-readable reason.
- INV-DATA-005: Manifests include row counts before filters, after filters, after
  deduplication, and per split.
