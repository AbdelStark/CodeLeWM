# Public API

## CLI

The package exposes one console script:

```toml
[project.scripts]
codelewm = "codelewm.harness.cli:main"
```

Commands:

```bash
codelewm dataset build --config config/first_results/dataset_build.json --out .artifacts/first-results/build --json
codelewm dataset pack --manifest .artifacts/first-results/build/manifest.json --out .artifacts/first-results/pack --json
codelewm train --config config/first_results/train_tiny.json --out .artifacts/first-results/train --executor torch --device cpu --json
codelewm eval retrieval --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/retrieval --device cpu --json
codelewm eval latent-probe --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/latent_probe --device cpu --json
codelewm eval latent-matrix --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/latent_matrix --device cpu --json
codelewm eval ablation --retrieval-artifact .artifacts/first-results/retrieval/manifest.json --training-artifact .artifacts/first-results/train/manifest.json --out .artifacts/first-results/ablation --json
codelewm eval surprise --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/surprise --device cpu --json
codelewm index --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --data .artifacts/first-results/pack --out .artifacts/first-results/index --device cpu --json
codelewm eval scorer-quality --config config/first_results/scorer_quality.json --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --out .artifacts/first-results/scorer_quality --index .artifacts/first-results/index --retrieval-prior-weight 1.0 --json
codelewm score --before tests/fixtures/codestate/class_method_before.py --instruction "rewrite the accumulator update explicitly" --candidate config/first_results/scorer_quality_candidates/true_after.py --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --json
codelewm rerank --before tests/fixtures/codestate/class_method_before.py --instruction "rewrite the accumulator update explicitly" --candidates config/first_results/scorer_quality_candidates --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --json
codelewm llm-demo --before tests/fixtures/codestate/class_method_before.py --instruction "rewrite the accumulator update explicitly" --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt --out .artifacts/llm-demo --allow-unsafe-checkpoint --json
codelewm openrouter byok-register --provider anthropic --key-env ANTHROPIC_API_KEY --management-key-env OPENROUTER_MANAGEMENT_KEY --allowed-model anthropic/claude-4.5-sonnet --dry-run --json
codelewm secret-scan .artifacts/first-results docs/benchmark/FIRST_RESULTS.md --json
codelewm manifest verify --manifest .artifacts/first-results/train/manifest.json --parent-manifest .artifacts/first-results/pack/manifest.json --json
```

The reproducible local orchestration command is
`uv run scripts/first-results --overwrite`; it is the shortest way to exercise
the full package-native smoke path. Scaled evidence is documented in
`docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md` and
`docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`. Both scaled runs are
negative for positive action-conditioning claims because text-action does not
beat no-action on headline retrieval.

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
  source-acquisition and license-gate metadata;
- `transitions.jsonl`: fixed-schema transition rows for the follow-on pack
  command;
- `reports/filter_report.json`, `reports/license_gate_report.json`,
  `reports/source_acquisition_report.json`,
  `reports/split_dedup_report.json`, `reports/row_counts.json`, and
  `reports/action_discriminative_shard_report.json`.

The action-discriminative shard report uses
`schema_version=codelewm.data.action_discriminative_shard_report.v1`. It records
action-signature coverage, edit-size buckets, same-file/near-before hard
negative opportunities, duplicate before-state pressure, action metadata
coverage, and a shard-level readiness gate for positive action-use claims.

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
- `reports/action_discriminative_shard_report.json`;
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
  --tensorboard \
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
- `reports/tensorboard_export.json`:
  `codelewm.training.tensorboard_export.v1` when `--tensorboard` is enabled;
- `tensorboard/events.out.tfevents.*` when `--tensorboard` is enabled;
- `checkpoints/checkpoint.pt` and
  `checkpoints/checkpoint.pt.manifest.json` for torch runs.

`--resume-from <training_manifest.json>` validates the parent training run,
artifact manifest, and paired checkpoint manifest before loading the checkpoint.
Resume incompatibility exits 5 with `error_type=checkpoint_error`. Missing
train/data runtime dependencies exit 2 with
`error_type=optional_dependency_missing`. `--tensorboard` requires the optional
observability dependency group, records bounded scalar and histogram diagnostics
in the training artifact manifest, and does not replace JSONL metrics or JSON
reports. `--log-jsonl` appends
`codelewm.log_event.v1` start, completion, and error events without replacing
JSON stdout.

`codelewm model inspect-checkpoint` is the public trusted-checkpoint to model
inspection artifact path:

```bash
codelewm model inspect-checkpoint \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --out .artifacts/tiny-checkpoint-inspection \
  --parent-manifest .artifacts/tiny-train/manifest.json \
  --json
```

It verifies the paired `codelewm.checkpoint.v1` manifest before deserializing
the checkpoint unless `--allow-unsafe-checkpoint` is explicitly passed. It
writes `reports/model_checkpoint_inspection.json` with
`schema_version=codelewm.model_checkpoint_inspection.v1` and a
`codelewm.artifact_manifest.v1` manifest over the report. The report contains
module/layer names, parameter counts, tensor shapes, dtype/device metadata,
finite-value checks, scalar statistics, norm summaries, selected bounded
histograms, checkpoint-manifest provenance, compatibility metadata, and a
diagnostic-only claim gate. It must not serialize raw tensor arrays, optimizer
state, secrets, or unredacted private paths.

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
  `codelewm.eval.hard_negative_sampler_report.v1`;
- `reports/action_contrast_pool_report.json`:
  `codelewm.eval.action_contrast_pool_report.v1`.

The command emits `codelewm.eval.retrieval_run.v1` on JSON stdout. Reports
include `Recall@1`, `Recall@5`, `Recall@10`, MRR, median rank, candidate
counts, required random/lexical/no-action/shuffled-action baselines,
hard-negative slices, and action-view policy metadata. Headline reports require
the text action view; patch actions remain diagnostic upper bounds and are
rejected for headline reports. Evaluation gate failures exit 6 with
`error_type=evaluation_gate_error`.

`codelewm eval latent-probe` is the public frozen-latent probe path:

```bash
codelewm eval latent-probe \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --data .artifacts/tiny-pack \
  --out .artifacts/tiny-latent-probe \
  --json
```

It verifies the same dataset, training-run, and checkpoint manifests as
retrieval evaluation, then writes `reports/latent_probe_report.json` with
schema `codelewm.eval.latent_probe_report.v1`. The report probes `z_before`,
`z_after`, and `z_pred_after` over train/validation/test labels for edit class,
AST node kind, symbol kind, edit-size bucket, action cluster, and source family.
It includes majority, lexical, metadata-only, random-latent, no-action, and
shuffled-action controls, bootstrap confidence intervals, and per-dimension
association diagnostics. Positive semantic-axis names are blocked unless future
runs demonstrate stability across seeds and splits.

`codelewm eval latent-matrix` is the public latent-geometry diagnostic path:

```bash
codelewm eval latent-matrix \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --data .artifacts/tiny-pack \
  --out .artifacts/tiny-latent-matrix \
  --latent-probe-report .artifacts/tiny-latent-probe/reports/latent_probe_report.json \
  --json
```

It verifies the same dataset, training-run, and checkpoint manifests as
retrieval and latent-probe evaluation, then writes
`reports/latent_matrix_report.json` with schema
`codelewm.eval.latent_matrix_report.v1` and `reports/run_timeline.json` with
schema `codelewm.run_timeline.v1`. The report covers `z_before`,
`z_after`, and `z_pred_after` latent matrices, including row/dimension counts,
split and source coverage, finite-value checks, per-dimension statistics,
effective rank, norm summaries, mean pairwise cosine, bounded covariance and
correlation previews suitable for heatmaps, inline dimension-label association
diagnostics, and optional links to `codelewm.eval.latent_probe_report.v1`
controls. It never serializes raw latent vectors by default. Semantic-axis,
action-conditioned-quality, and downstream-coding-usefulness claim gates remain
closed unless future multi-seed/split evidence and downstream benchmarks pass.

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

`codelewm eval ablation` consolidates a retrieval artifact and training artifact
into an action-view ablation report:

```bash
codelewm eval ablation \
  --retrieval-artifact .artifacts/tiny-retrieval/manifest.json \
  --training-artifact .artifacts/tiny-train/manifest.json \
  --out .artifacts/tiny-ablation \
  --json
```

It verifies both parent artifacts and writes:

- `manifest.json`: `codelewm.artifact_manifest.v1` for the ablation artifact;
- `reports/action_view_ablation_report.json`:
  `codelewm.eval.action_ablation_report.v1`.

The command emits `codelewm.eval.action_ablation_run.v1` on JSON stdout. The
report records completed text-action and required-baseline rows, completed
collapse/retrieval-loss rows when artifact evidence exists, and explicit
`blocked` rows for missing abstract-action, retrieval-loss, alternate SIGReg,
or patch-action diagnostic reports. Patch-action rows must use diagnostic scope
and `diagnostic_upper_bound=true`.

`codelewm index` is the public training-run plus packed-dataset to transition
index path:

```bash
codelewm index \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --data .artifacts/tiny-pack \
  --out .artifacts/tiny-index \
  --json
```

It verifies the packed dataset artifact, infers and verifies the parent
training-run artifact manifest from the checkpoint directory, validates the
paired checkpoint manifest before loading torch weights, and writes:

- `manifest.json`: `codelewm.artifact_manifest.v1` for the index artifact;
- `index.json`: `codelewm.transition_index.v1`;
- `entries.jsonl`: train-split transition metadata rows;
- `vectors.npy`: train-split `state_after` latent vectors.

The command emits `codelewm.index_build.v1` on JSON stdout. Index artifacts use
only the `train` split, so scorer/reranker retrieval priors do not retrieve
held-out evaluation rows. Index validation failures exit 6 with
`error_type=evaluation_gate_error`; malformed requests exit 2 with
`error_type=config_error`.

`codelewm eval scorer-quality` builds the release-facing score/rerank quality
artifact:

```bash
codelewm eval scorer-quality \
  --config config/first_results/scorer_quality.json \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --out .artifacts/tiny-scorer-quality \
  --index .artifacts/tiny-index \
  --retrieval-prior-weight 1.0 \
  --retrieval-prior-k 10 \
  --parent-manifest .artifacts/tiny-train/manifest.json \
  --parent-manifest .artifacts/tiny-index/manifest.json \
  --json
```

The config schema is `codelewm.harness.scorer_quality_config.v1`. The command
validates the checkpoint trust gate, verifies any repeated `--parent-manifest`
arguments, parses after-state candidates, dry-run-applies patch candidates as
text, and never executes candidate code. It writes:

- `manifest.json`: `codelewm.artifact_manifest.v1` with
  `artifact_kind=score_report`;
- `config.json`: normalized scorer quality config;
- `reports/scorer_quality_report.json`:
  `codelewm.harness.scorer_quality_report.v1`.

The command emits `codelewm.harness.scorer_quality_run.v1` on JSON stdout. The
report includes ranking metrics, score distributions, calibration slices by
candidate kind, parse/patch failure counts, retrieval-prior settings, and the
current risk-penalty caveat. Lower `final_score` remains better.

`codelewm eval downstream-pack` builds a self-contained downstream reranking
benchmark pack:

```bash
codelewm eval downstream-pack \
  --config config/benchmark/downstream_rerank_fixture.json \
  --out .artifacts/downstream-rerank-fixture \
  --json
```

The config schema is `codelewm.downstream_rerank_benchmark_config.v1`. The
command copies public-safe before-state and candidate files into the artifact,
emits `codelewm.downstream_rerank_benchmark.v1`, source/license policy, split
leakage, readiness, and secret-scan reports, and never imports or executes
candidate code. The checked-in fixture has one labeled task and is explicitly
blocked by the 100-example readiness gate.

`codelewm eval downstream-rerank` consumes the benchmark pack and writes the
downstream comparison:

```bash
codelewm eval downstream-rerank \
  --benchmark-manifest .artifacts/downstream-rerank-fixture/manifest.json \
  --checkpoint .artifacts/tiny-train/checkpoints/checkpoint.pt \
  --out .artifacts/downstream-rerank-report \
  --json
```

The command emits `codelewm.downstream_rerank_eval_run.v1` on stdout and writes
`codelewm.downstream_rerank_report.v1` with required baselines, slices,
baseline availability status, confidence intervals when the sample count
permits, and the downstream claim gate. Retrieval-prior baselines are reported
as blocked unless an index produces finite retrieval-prior scores.

`codelewm llm-demo` runs the LLM plus world-model showcase path:

```bash
codelewm llm-demo \
  --before tests/fixtures/codestate/class_method_before.py \
  --instruction "rewrite the accumulator update explicitly" \
  --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt \
  --out .artifacts/llm-demo \
  --allow-unsafe-checkpoint \
  --json
```

It builds an OpenRouter request from environment variables, captures generated
candidate patches as `codelewm.llm_candidate_pack.v1`, writes a manifest-backed
candidate-pack artifact, scores/reranks candidates without executing them, and
writes `codelewm.harness.demo_report.v1` plus `codelewm.run_timeline.v1` under
`reports/run_timeline.json`. The command emits
`codelewm.harness.demo_run.v1` on stdout and writes a self-contained
`demo.html` visual report into the demo artifact. Demo score payloads include
`score_direction=lower_is_better`; candidate-minus-no-action deltas are better
when negative and worse when positive. Fixture mode remains the default with
`CODELEWM_LLM_DRY_RUN=1`; live mode requires `OPENROUTER_API_KEY`. Use
`uv run scripts/llm-world-model-demo` for the end-to-end local task that creates
a tiny input file, ensures a first-results checkpoint exists, runs the demo,
verifies manifests, secret-scans the output, and prints the visual report path.

`codelewm openrouter byok-register` creates or dry-runs an OpenRouter BYOK
provider credential from local environment secrets:

```bash
codelewm openrouter byok-register \
  --provider anthropic \
  --key-env ANTHROPIC_API_KEY \
  --management-key-env OPENROUTER_MANAGEMENT_KEY \
  --allowed-model anthropic/claude-4.5-sonnet \
  --dry-run \
  --json
```

The command emits `codelewm.openrouter_byok_register.v1`. Non-dry-run mode
requires the OpenRouter management key named by `--management-key-env` plus the
raw provider key named by `--key-env`. The raw provider key is sent only to
OpenRouter's BYOK API and is never printed or serialized in the returned JSON.
Normal chat requests still use `OPENROUTER_API_KEY`; management keys are only
for administrative registration. When `CODELEWM_OPENROUTER_BYOK=1`, request
metadata records redacted BYOK routing state and, with
`CODELEWM_OPENROUTER_BYOK_REQUIRE=1`, restricts OpenRouter provider routing to
the configured provider.

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
environment. Both commands accept `--index <dir>` plus
`--retrieval-prior-weight <float>` and `--retrieval-prior-k <int>`. The prior is
a nearest-neighbor distance penalty over the local transition index. Lower
`final_score` remains better. With the default weight of `0.0`, the prior is
reported but does not alter ordering. See
`docs/spec/06-security.md#checkpoint-trust`.

Root `train.py`, root `eval.py`, and inherited Hydra configs remain
compatibility surfaces for the LeWorldModel seed. They are not the public CodeLeWM artifact path. Public docs, tests, and release gates use the package
commands above plus `scripts/first-results` and `scripts/hf-*`.

Common flags are command-specific. Use `codelewm <command> --help` for the exact
surface. Landed automation paths use these conventions:

```bash
--json
--overwrite
--device cpu|cuda|mps|auto
--log-jsonl <path>
```

## Python API

```python
from pathlib import Path
from codelewm.harness import load_scorer
from codelewm.model import AbstractActionEncoderConfig, TextActionEncoderConfig

scorer = load_scorer(Path(".artifacts/first-results/train/checkpoints/checkpoint.pt"), device="cpu")
result = scorer.score_files(
    before=Path("tests/fixtures/codestate/class_method_before.py"),
    instruction="add timeout handling to the retry loop",
    candidate=Path("config/first_results/scorer_quality_candidates/true_after.py"),
)
```

Public model configuration helpers include `TextActionEncoderConfig` for the
headline text action path and `AbstractActionEncoderConfig` for structural
ablation runs. Both project action encodings to the v0.1 latent dimension.

Training is exposed through the package runner:

```python
from codelewm.training import load_train_config, train_torch

cfg = load_train_config("config/first_results/train_tiny.json")
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
scoring. It can also load a local `codelewm.transition_index.v1` directory and
populate `retrieval_prior` without changing the score schema. The initial
runtime-light backend is deterministic and intended for API and fixture
validation; model-backed checkpoint execution can replace the backend without
changing `ScoreResult`.

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
config.json
reports/
```

`manifest.json` includes:

```python
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
