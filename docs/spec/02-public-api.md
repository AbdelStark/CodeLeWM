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
codelewm eval retrieval --checkpoint runs/v0_1/checkpoints/checkpoint.pt --data data/codelewm_v0_1/hdf5 --out reports/v0_1/retrieval
codelewm eval surprise --checkpoint runs/v0_1/checkpoints/checkpoint.pt --data data/codelewm_v0_1/hdf5 --out reports/v0_1/surprise
codelewm index --checkpoint runs/v0_1/checkpoint.pt --data data/codelewm_v0_1/hdf5/train.hdf5 --out indexes/v0_1
codelewm score --before before.py --instruction instruction.txt --candidate after.py --checkpoint runs/v0_1/checkpoint.pt
codelewm rerank --before before.py --instruction instruction.txt --candidates patches/ --checkpoint runs/v0_1/checkpoint.pt
codelewm secret-scan runs/v0_1/ logs/
codelewm manifest verify --manifest runs/v0_1/manifest.json
```

`codelewm dataset build` is the public raw-shard to transition-artifact
path. It accepts JSON configs in the base environment and YAML configs when
`omegaconf` is installed:

```bash
codelewm dataset build \
  --config tests/fixtures/dataset_build/config.json \
  --out data/codelewm_fixture \
  --json
```

The command emits:

- `manifest.json`: `codelewm.artifact_manifest.v1`, verified with
  `codelewm manifest verify --manifest <out>/manifest.json --json`;
- `dataset_manifest.json`: `codelewm.transition.v1` dataset manifest with row
  counts, split counts, source counts, feature flags, checksums, and
  license-gate metadata;
- `transitions.jsonl`: fixed-schema transition rows for the follow-on pack
  command;
- `reports/filter_report.json`, `reports/license_gate_report.json`,
  `reports/split_dedup_report.json`, and `reports/row_counts.json`.

Invalid configs exit 2 with `error_type=config_error`; unavailable sources exit
3 with `error_type=source_unavailable`; malformed rows or empty kept outputs
exit 4 with `error_type=dataset_build_error` or `empty_dataset`.

`codelewm dataset pack` is the public transition-artifact to training-artifact
path:

```bash
codelewm dataset pack \
  --manifest data/codelewm_fixture/manifest.json \
  --out data/codelewm_fixture_packed \
  --json
```

It requires the data dependency group (`h5py` and `pyarrow`), verifies the input
artifact and dataset manifests, records the build artifact as the parent, and
writes:

- split HDF5 files at `hdf5/train.hdf5`, `hdf5/val.hdf5`, and
  `hdf5/test.hdf5`;
- split Parquet staging shards under `parquet/{train,val,test}/`;
- `dataset_manifest.json` with pack artifact checksums and inherited
  license-gate metadata;
- `reports/pack_report.json`;
- `manifest.json` with `parent_artifacts=[<build artifact id>]`.

Verify lineage with:

```bash
codelewm manifest verify \
  --manifest data/codelewm_fixture_packed/manifest.json \
  --parent-manifest data/codelewm_fixture/manifest.json \
  --json
```

Missing `h5py` or `pyarrow` exits 2 with
`error_type=optional_dependency_missing`. Input manifest, checksum, and row-count
mismatches exit 4 with `error_type=dataset_build_error`.

`codelewm train` is the public packed-dataset to training-run path:

```bash
codelewm train \
  --config tests/fixtures/tiny_train.json \
  --out .artifacts/tiny-train \
  --executor torch \
  --device cpu \
  --json
```

It loads `codelewm.train_config.v1`, optionally overrides `output.run_dir` with
`--out`, verifies the packed dataset artifact manifest, selects `torch` or
`cpu-smoke` with `--executor`, and writes:

- `manifest.json`: `codelewm.artifact_manifest.v1` for the training run;
- `training_manifest.json`: `codelewm.training_run.v1`;
- `metrics.jsonl`: `codelewm.training_metrics.v1`;
- `reports/metrics_report.json`;
- `reports/torch_training_report.json` when `--executor torch` is used;
- `checkpoints/checkpoint.pt` and
  `checkpoints/checkpoint.pt.manifest.json` for torch runs.

`--resume-from <training_manifest.json>` validates the parent training run,
artifact manifest, and paired checkpoint manifest before loading the checkpoint.
Resume incompatibility exits 5 with `error_type=checkpoint_error`. Missing
train/data runtime dependencies exit 2 with
`error_type=optional_dependency_missing`. `--log-jsonl` appends
`codelewm.log_event.v1` start, completion, and error events without replacing
JSON stdout.

`codelewm eval retrieval` is the public training-run plus packed-dataset to
retrieval-report path:

```bash
codelewm eval retrieval \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --data .artifacts/tiny-pack \
  --out .artifacts/tiny-retrieval \
  --json
```

It verifies the packed dataset artifact, infers and verifies the parent
training-run artifact manifest from the checkpoint directory, validates the
paired checkpoint manifest before loading torch weights, and writes:

- `manifest.json`: `codelewm.artifact_manifest.v1` for the eval report;
- `config.json`: normalized retrieval evaluation config;
- `reports/retrieval_report.json`: `codelewm.eval.retrieval_report.v1`;
- `reports/hard_negative_sampler_report.json`:
  `codelewm.eval.hard_negative_sampler_report.v1`.

The command emits `codelewm.eval.retrieval_run.v1` on JSON stdout. Reports
include `Recall@1`, `Recall@5`, `Recall@10`, MRR, median rank, candidate
counts, required random/lexical/no-action/shuffled-action baselines,
hard-negative slices, and action-view policy metadata. Headline reports require
the text action view; patch actions remain diagnostic upper bounds and are
rejected for headline reports. Evaluation gate failures exit 6 with
`error_type=evaluation_gate_error`.

`codelewm eval surprise` is the public training-run plus packed-dataset to
patch-surprise-report path:

```bash
codelewm eval surprise \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --data .artifacts/tiny-pack \
  --out .artifacts/tiny-surprise \
  --json
```

It verifies the packed dataset artifact, infers and verifies the parent
training-run artifact manifest from the checkpoint directory, validates the
paired checkpoint manifest before loading torch weights, and writes:

- `manifest.json`: `codelewm.artifact_manifest.v1` for the eval report;
- `config.json`: normalized surprise evaluation config;
- `reports/surprise_report.json`: `codelewm.eval.surprise_report.v1`.

The command emits `codelewm.eval.surprise_run.v1` on JSON stdout. Reports score
the true after-state against random, same-file, mutation, and action-cluster
decoys where the held-out data supports them. The report includes pairwise AUC,
true ranks, per-category AUC/count slices, and explicit caveats for unavailable
decoy categories. Scores are squared transition energies, so lower is better.
Evaluation gate failures exit 6 with `error_type=evaluation_gate_error`.

`manifest verify` validates that every file declared in an artifact manifest
exists, matches its recorded byte size and SHA-256, and that any required parent
artifacts are passed in with `--parent-manifest`. The verifier exits with code 2
on any mismatch and emits a `codelewm.manifest_verify.v1` JSON report with
`--json`. Release gates call this verifier.

`codelewm secret-scan` accepts one or more files or directories and emits a
`codelewm.secret_scan.v1` JSON report listing every secret-pattern match by
path, line, pattern name, and redacted digest. The scanner returns exit code 2
when any match is found. Use `--include-suffix` to extend the default suffix
set or `--no-recursive` to scan only the top level. See
`docs/spec/06-security.md#secrets-handling` for the scanner contract.

`score` and `rerank` refuse to load a checkpoint without a paired manifest by
default; pass `--allow-unsafe-checkpoint` to opt out in a trusted local
environment. See `docs/spec/06-security.md#checkpoint-trust`.

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
from codelewm.training import load_train_config, train, train_torch

cfg = load_train_config("config/train/codelewm_tiny.yaml")
manifest = train(cfg, executor=executor)
torch_manifest = train_torch(cfg, device="cpu")
```

The runner owns config validation, parent dataset manifest validation, output
layout, metrics files, checkpoint hashes, and training-run manifests. Concrete
CPU/GPU model execution is supplied by an executor implementation. The
package-native torch executor consumes packed HDF5 transition artifacts, trains
the CodeLeWM state/action/predictor stack, writes
`reports/torch_training_report.json`, and emits a paired
`codelewm.checkpoint.v1` manifest beside `checkpoints/checkpoint.pt`. The public
`codelewm train` command calls the same runner and executor contracts.

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

The package-native model-backed runner is also exposed for local automation:

```python
from codelewm.eval import run_retrieval_evaluation

result = run_retrieval_evaluation(
    checkpoint=".artifacts/tiny-train/checkpoints/checkpoint.pt",
    data=".artifacts/tiny-pack",
    out=".artifacts/tiny-retrieval",
)
```

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

`codelewm rerank` accepts either one candidate path or a directory of candidates.
Candidate files are interpreted as complete after-state Python files unless the
suffix is `.patch` or `.diff`, in which case the candidate is applied as a
single-file unified diff against the `--before` file in memory. The command never
modifies the user's working tree.

Valid candidates are sorted by ascending `final_score`. Candidates that fail
syntax validation or dry-run patch application are represented as `ErrorReport`
items and appear after valid `ScoreResult` items.

`RerankResult` schema:

```python
@dataclass(frozen=True)
class RerankResult:
    schema_version: str
    results: tuple[ScoreResult | ErrorReport, ...]
    warnings: tuple[str, ...]
```

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
