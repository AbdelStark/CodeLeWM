# CodeLeWM Usage Guide

This guide documents the install path, the public CLI subcommands
that are landed today, the Python API for scoring candidate
after-states, and where every artifact lives on disk. The guide
is deliberately artifact-backed: every command points at the
manifest schema it consumes or produces, and every Python example is
short enough to copy into a notebook.

The canonical surface is described in
`docs/spec/02-public-api.md`. This guide is the **how-to** companion;
when the two disagree the spec wins.

## Install

```bash
uv sync --group dev
```

Optional groups:

```bash
uv sync --group dev --group data      # h5py + pyarrow for dataset packing
uv sync --group dev --group train     # torch and training runtime adapters
uv sync --group dev --group eval      # optional evaluation helpers
uv sync --group dev --group docs      # documentation checks
uv sync --group dev --group release   # package build and release gates
```

Verify the install:

```bash
uv run codelewm --version
uv run python -m pytest tests/
```

Some tests skip when optional runtimes such as `torch`, `h5py`, or
`pyarrow` are absent.

## Command Surface

The console script exposes one entry point, `codelewm`, with the
following subcommands. Commands marked **landed** are runnable
today; commands marked **planned** are part of the spec and exist
either as a Python API only or as a placeholder.

| Command | Status | Public schema produced |
| ------- | ------ | ---------------------- |
| `codelewm score` | landed | `codelewm.score.v1`, `codelewm.error.v1` |
| `codelewm rerank` | landed | `codelewm.rerank.v1`, `codelewm.error.v1` |
| `codelewm manifest verify` | landed | `codelewm.manifest_verify.v1`, `codelewm.error.v1` |
| `codelewm secret-scan` | landed | `codelewm.secret_scan.v1`, `codelewm.error.v1` |
| `codelewm dataset build` | landed | `codelewm.dataset_build_report.v1`, `codelewm.artifact_manifest.v1`, `codelewm.transition.v1`, `codelewm.error.v1` |
| `codelewm dataset pack` | landed | `codelewm.dataset_pack_report.v1`, `codelewm.artifact_manifest.v1`, `codelewm.transition.v1`, `codelewm.error.v1` |
| `codelewm train` | landed | `codelewm.training_run.v1`, `codelewm.torch_training_report.v1`, `codelewm.checkpoint.v1`, `codelewm.error.v1` |
| `codelewm eval retrieval` | landed | `codelewm.eval.retrieval_run.v1`, `codelewm.eval.retrieval_report.v1`, `codelewm.artifact_manifest.v1`, `codelewm.error.v1` |
| `codelewm eval surprise` | landed | `codelewm.eval.surprise_run.v1`, `codelewm.eval.surprise_report.v1`, `codelewm.artifact_manifest.v1`, `codelewm.error.v1` |
| `codelewm index` | planned | `codelewm.transition_index.v1` |

Run `codelewm <command> --help` for the current flag set. JSON
output is opt-in via `--json` on every landed command.

### `codelewm score`

Score a single candidate after-state against a before-state and
instruction:

```bash
codelewm score \
  --before tests/fixtures/before.py \
  --instruction "increment value" \
  --candidate tests/fixtures/after.py \
  --checkpoint runs/v0_1/checkpoint.pt \
  --json
```

Sample JSON output:

```json
{
  "schema_version": "codelewm.score.v1",
  "candidate": "tests/fixtures/after.py",
  "transition_energy": 0.42,
  "retrieval_prior": null,
  "risk_penalty": null,
  "final_score": 0.42,
  "model_id": "codelewm-hashing-fixture",
  "checkpoint_sha256": "....",
  "input_digest": "....",
  "warnings": ["lightweight scorer backend ..."]
}
```

Failure modes — every one of these returns exit code 2 with a
`codelewm.error.v1` JSON payload (the `--json` flag is opt-in but
recommended for automation):

- candidate is not valid Python (`error_type=invalid_syntax`);
- candidate path does not exist (`error_type=missing_file`);
- checkpoint cannot be loaded (`error_type=checkpoint_error`);
- a patch candidate cannot apply cleanly (`error_type=patch_apply_failed`).

Optional flags:

- `--device {cpu,cuda,mps,auto}` (default `auto`);
- `--log-jsonl <path>` appends structured `codelewm.log_event.v1`
  events to a local JSONL file. The redactor in
  `codelewm.observability.logging` removes secret patterns, home
  paths, and overly long source snippets before writing.

### `codelewm rerank`

Rerank candidate after-states or candidate patches:

```bash
codelewm rerank \
  --before tests/fixtures/before.py \
  --instruction "increment value" \
  --candidates tests/fixtures/candidates/ \
  --checkpoint runs/v0_1/checkpoint.pt \
  --json
```

`--candidates` accepts a file or a directory. Files ending in
`.patch` / `.diff` are applied to `--before` in memory via dry-run
unified-diff application; other files are interpreted as complete
after-state Python files. The user's working tree is never
modified.

Valid candidates are sorted by ascending `final_score`. Candidates
that fail syntax validation or dry-run patch application are
emitted as `codelewm.error.v1` records after every valid
`codelewm.score.v1` record.

### `codelewm dataset build`

Build a local transition dataset artifact from fixture records or
CommitPackFT-style JSONL shards:

```bash
codelewm dataset build \
  --config tests/fixtures/dataset_build/config.json \
  --out data/codelewm_fixture \
  --json
```

The config schema is `codelewm.dataset_build_config.v1`. JSON
configs work with the base development environment; YAML configs
are accepted when `omegaconf` is installed, for example through the
training dependency group.

The command writes:

```
<out>/
  manifest.json              codelewm.artifact_manifest.v1
  dataset_manifest.json      codelewm.transition.v1
  config.json                normalized build config
  transitions.jsonl          packed transition rows
  reports/
    filter_report.json
    license_gate_report.json
    split_dedup_report.json
    row_counts.json
```

`manifest.json` is the verifier target:

```bash
codelewm manifest verify --manifest data/codelewm_fixture/manifest.json --json
```

The build applies parse, Python-path, generated-file, size,
message, edit-ratio, license, deterministic split, and dedup
policies before writing transitions. Rejected rows are recorded in
the filter and split/dedup reports; zero kept transitions fail with
`codelewm.error.v1`.

### `codelewm dataset pack`

Pack a built transition artifact into split HDF5 files and Parquet staging
shards for training:

```bash
codelewm dataset pack \
  --manifest data/codelewm_fixture/manifest.json \
  --out data/codelewm_fixture_packed \
  --json
```

This command requires the data dependency group:

```bash
uv sync --group dev --group data
```

The pack command verifies the input artifact manifest and the input dataset
manifest before writing output. The packed output includes:

```
<out>/
  manifest.json              codelewm.artifact_manifest.v1
  dataset_manifest.json      codelewm.transition.v1
  config.json                normalized pack config
  hdf5/
    train.hdf5
    val.hdf5
    test.hdf5
  parquet/
    train/
    val/
    test/
  reports/
    pack_report.json
```

Verify lineage by passing the build manifest as the parent:

```bash
codelewm manifest verify \
  --manifest data/codelewm_fixture_packed/manifest.json \
  --parent-manifest data/codelewm_fixture/manifest.json \
  --json
```

If `h5py` or `pyarrow` is unavailable, the command exits with
`codelewm.error.v1` and `error_type=optional_dependency_missing`.

### `codelewm train`

Run manifest-backed training over a packed dataset artifact:

```bash
codelewm train \
  --config tests/fixtures/tiny_train.json \
  --out .artifacts/tiny-train \
  --executor torch \
  --device cpu \
  --json
```

The default executor is `torch`, which trains the package-native CodeLeWM
state/action/predictor stack and writes:

```
<out>/
  manifest.json                         codelewm.artifact_manifest.v1
  training_manifest.json                codelewm.training_run.v1
  config.json                           normalized train config
  metrics.jsonl                         codelewm.training_metrics.v1
  checkpoints/
    checkpoint.pt
    checkpoint.pt.manifest.json         codelewm.checkpoint.v1
  reports/
    metrics_report.json
    torch_training_report.json          codelewm.torch_training_report.v1
```

The command verifies `data.manifest`, records the packed dataset artifact as the
parent, and refuses incompatible resumes before any new output is written:

```bash
codelewm train \
  --config tests/fixtures/tiny_train.json \
  --out runs/resumed \
  --resume-from runs/previous/training_manifest.json \
  --json
```

Use `--executor cpu-smoke` only for the dependency-light NumPy smoke path; it
keeps validating runner contracts but is not a model-quality claim and does not
replace the torch executor for first-results training. Missing train/data
runtime packages return `error_type=optional_dependency_missing`; incompatible
resume checkpoints return `error_type=checkpoint_error`.

### `codelewm eval retrieval`

Run model-backed retrieval evaluation over a packed dataset artifact:

```bash
codelewm eval retrieval \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --data .artifacts/tiny-pack \
  --out .artifacts/tiny-retrieval \
  --json
```

The command requires the data and train dependency groups. It verifies the
packed dataset artifact, infers and verifies the parent training-run artifact
from the checkpoint location, validates the paired checkpoint manifest before
loading torch weights, and writes:

```
<out>/
  manifest.json                              codelewm.artifact_manifest.v1
  config.json                                normalized retrieval config
  reports/
    retrieval_report.json                    codelewm.eval.retrieval_report.v1
    hard_negative_sampler_report.json        codelewm.eval.hard_negative_sampler_report.v1
```

The JSON stdout summary uses `codelewm.eval.retrieval_run.v1`. The report
includes `Recall@1`, `Recall@5`, `Recall@10`, MRR, median rank, easy and hard
candidate counts, random/lexical/no-action/shuffled-action baselines,
hard-negative slices, and action-view policy metadata. Headline reports require
`action_text`; patch-action reports remain diagnostic only.

Verify lineage by passing both parent manifests:

```bash
codelewm manifest verify \
  --manifest .artifacts/tiny-retrieval/manifest.json \
  --parent-manifest .artifacts/tiny-train/manifest.json \
  --parent-manifest .artifacts/tiny-pack/manifest.json \
  --json
```

Retrieval gate failures return `error_type=evaluation_gate_error`. Missing
runtime packages return `error_type=optional_dependency_missing`; missing or
tampered checkpoint manifests return `error_type=checkpoint_error`.

### `codelewm eval surprise`

Run model-backed patch-surprise evaluation over a packed dataset artifact:

```bash
codelewm eval surprise \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --data .artifacts/tiny-pack \
  --out .artifacts/tiny-surprise \
  --json
```

The command requires the data and train dependency groups. It verifies the
packed dataset artifact, infers and verifies the parent training-run artifact
from the checkpoint location, validates the paired checkpoint manifest before
loading torch weights, and writes:

```
<out>/
  manifest.json                              codelewm.artifact_manifest.v1
  config.json                                normalized surprise config
  reports/
    surprise_report.json                     codelewm.eval.surprise_report.v1
```

The JSON stdout summary uses `codelewm.eval.surprise_run.v1`. The report
includes pairwise AUC, per-example true ranks, per-category AUC/count slices,
and explicit caveats for missing random, same-file, mutation, or action-cluster
decoy categories. Scores are squared transition energies, so lower is better.

Verify lineage by passing both parent manifests:

```bash
codelewm manifest verify \
  --manifest .artifacts/tiny-surprise/manifest.json \
  --parent-manifest .artifacts/tiny-train/manifest.json \
  --parent-manifest .artifacts/tiny-pack/manifest.json \
  --json
```

Surprise gate failures return `error_type=evaluation_gate_error`. Missing
runtime packages return `error_type=optional_dependency_missing`; missing or
tampered checkpoint manifests return `error_type=checkpoint_error`.

### Planned commands (spec contract)

The following command appears in the spec and has manifests defined but is not
yet wired up as a full CLI subcommand.

```bash
codelewm index --checkpoint <ckpt> --data <hdf5> --out <dir>
```

## Python API

### Train the package-native model

```python
from codelewm.training import load_train_config, train_torch

cfg = load_train_config("config/train/codelewm_tiny.yaml")
manifest = train_torch(cfg, device="cpu")
print(manifest.final_metrics["loss/total"])
```

`train_torch` consumes split HDF5 files produced by `codelewm dataset pack`,
uses `action_text` by default, refuses patch-action training configs, writes
`training_manifest.json`, `metrics.jsonl`, `reports/torch_training_report.json`,
`checkpoints/checkpoint.pt`, and `checkpoints/checkpoint.pt.manifest.json`, and
records the packed dataset artifact as the parent manifest. Install it with:

```bash
uv sync --group dev --group data --group train
```

### Score one candidate

```python
from pathlib import Path
from codelewm.harness import load_scorer

scorer = load_scorer(
    Path("runs/v0_1/checkpoint.pt"),
    device="cuda",
)
result = scorer.score_files(
    before=Path("before.py"),
    instruction="add timeout handling to the retry loop",
    candidate=Path("after.py"),
)
print(result.to_dict())
```

`load_scorer` verifies the checkpoint path, records its SHA-256
checksum, and returns a `CodeLeWMScorer`. The initial
runtime-light backend is deterministic and intended for API and
fixture validation; a model-backed backend can replace it without
changing `ScoreResult`.

### Rerank a directory of candidates

```python
from pathlib import Path
from codelewm.harness import load_scorer

result = load_scorer(Path("runs/v0_1/checkpoint.pt")).rerank_files(
    before=Path("before.py"),
    instruction="add timeout handling to the retry loop",
    candidates=Path("candidates/"),
)
for item in result.results:
    if hasattr(item, "final_score"):
        print(item.candidate, item.final_score)
    else:
        print(item.artifact, item.error_type, item.message)
```

### Build a retrieval report

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
```

See `codelewm.eval` for the full API. Reports use
`schema_version=codelewm.eval.retrieval_report.v1`; candidate pools
use `schema_version=codelewm.eval.candidate_pool.v1` and must
exclude `train`-split rows. Every headline retrieval report must
include random, lexical, no-action, and shuffled-action baselines.

### Run a patch-surprise evaluation

```python
from codelewm.eval import (
    SurpriseExampleInput,
    build_decoys,
    build_surprise_report,
    score_surprise_example,
)

example = SurpriseExampleInput(
    transition_id="t-0001",
    repo="example/repo",
    path="pkg/mod.py",
    action_cluster="rename",
    true_after="def value():\n    return 2\n",
)
decoys = build_decoys(example, corpus=(example,), seed=0)
result = score_surprise_example(
    example,
    decoys=decoys,
    score_fn=lambda ex, candidate: 0.0 if candidate == ex.true_after else 1.0,
)
report = build_surprise_report([result], decoy_seed=0)
```

Reports use `codelewm.eval.surprise_report.v1`.

### Verify a manifest

The manifest verifier is exposed as a Python API and is reachable
via the artifact-manifest schema directly:

```python
from codelewm.observability import (
    read_artifact_manifest,
    validate_artifact_checksums,
)

manifest = read_artifact_manifest("runs/v0_1/manifest.json")
validate_artifact_checksums(manifest, root="runs/v0_1")
```

The same verifier is exposed through the landed CLI:

```bash
codelewm manifest verify --manifest runs/v0_1/manifest.json --json
```

### Scan for secrets in local reports

```python
from codelewm.security import scan_paths

report = scan_paths(["runs/v0_1/", "logs/"])
if not report.ok:
    for finding in report.findings:
        print(finding.path, finding.line, finding.pattern, finding.redacted)
```

Findings carry only `[REDACTED_SECRET sha256=... length=...]` for
the matched value — the scanner never re-publishes the secret it
flagged. The schema is `codelewm.secret_scan.v1`.

## Artifact Locations

Every CodeLeWM artifact directory contains the same minimum set of
files:

```
<artifact_dir>/
  manifest.json           codelewm.artifact_manifest.v1
  config.yaml             pinned config for the run
  reports/                schema-versioned JSON reports
```

The training run additionally writes:

```
<run_dir>/
  manifest.json             codelewm.artifact_manifest.v1
  training_manifest.json    codelewm.training_run.v1
  config.json               cfg.to_dict() for the run
  metrics.jsonl             codelewm.training_metrics.v1 events
  checkpoints/
    checkpoint.state
    checkpoint.state.manifest.json  codelewm.checkpoint.v1
  reports/
    metrics_report.json     codelewm.training_metrics.v1 summary
    collapse.json           codelewm.collapse_report.v1
```

See `docs/spec/05-observability.md` and
`docs/spec/02-public-api.md#artifact-contracts` for the canonical
field list.

## Report and Manifest Schemas

| Surface | Schema version |
| ------- | -------------- |
| Dataset build report | `codelewm.dataset_build_report.v1` |
| Dataset pack report | `codelewm.dataset_pack_report.v1` |
| Dataset manifest | `codelewm.transition.v1` |
| Artifact manifest | `codelewm.artifact_manifest.v1` |
| Manifest verifier report | `codelewm.manifest_verify.v1` |
| Checkpoint manifest | `codelewm.checkpoint.v1` |
| Training run manifest | `codelewm.training_run.v1` |
| Training metrics | `codelewm.training_metrics.v1` |
| Retrieval eval run | `codelewm.eval.retrieval_run.v1` |
| Retrieval report | `codelewm.eval.retrieval_report.v1` |
| Surprise eval run | `codelewm.eval.surprise_run.v1` |
| Surprise report | `codelewm.eval.surprise_report.v1` |
| Candidate pool | `codelewm.eval.candidate_pool.v1` |
| Public license gate | `codelewm.public_license_gate.v1` |
| Transition index | `codelewm.transition_index.v1` |
| Score result | `codelewm.score.v1` |
| Rerank result | `codelewm.rerank.v1` |
| Harness error report | `codelewm.error.v1` |
| Structured log event | `codelewm.log_event.v1` |
| Secret scan report | `codelewm.secret_scan.v1` |

`CHANGELOG.md` carries the same table and is the authoritative
location for schema-version history once a release ships.

## Trust Boundaries

CodeLeWM treats source data, configs, and candidate patches as
untrusted. The CLI and Python API are designed so that:

- candidate code is never imported, evaluated, or test-executed
  (`docs/spec/06-security.md#non-execution-policy`);
- secret-pattern leakage is rejected by
  `codelewm.security.scan_paths`;
- checkpoint loading goes through
  `codelewm.security.require_trusted_checkpoint` (the
  `--allow-unsafe-checkpoint` flag is the documented escape hatch
  for fixture-only workflows);
- public dataset artifacts pass the license gate before
  publication.

Read `docs/spec/06-security.md` before writing code that loads
checkpoints, parses user configs, or publishes reports.

## Next Steps

- `docs/spec/02-public-api.md` — canonical CLI and Python contract.
- `docs/spec/04-error-model.md` — exit codes, error types, and how
  errors surface in JSON.
- `docs/spec/05-observability.md` — structured logging and
  artifact-manifest contract.
- `docs/spec/07-testing-strategy.md` — required commands the CI
  workflow runs.
- `docs/spec/09-release-and-versioning.md` — versioning, checkpoint
  compatibility, deprecation policy, and the release artifact list.
- `docs/roadmap/IMPLEMENTATION.md` — open implementation issues
  grouped by milestone.
