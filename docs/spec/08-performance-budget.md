# Performance Budget

## Data Build

Targets on a local workstation:

- parse and filter at least `10` Python files per second per worker for normal
  source files under 2k lines;
- write Parquet shards of `10k-50k` rows;
- keep peak memory under `4 GiB` for v0.1 builds;
- support resumable source processing by shard.

Rows that exceed parser timeouts or token budgets are dropped with structured
reasons rather than slowing the full build.

## Model

v0.1 tiny model:

- latent dim `256`;
- state length `1024`;
- action text length `256`;
- abstract action length `192`;
- parameter target `10M-20M`;
- CPU one-batch smoke supported;
- GPU training target fits in `24 GiB` with batch size `64`.

v1.0 small model:

- parameter target `20M-35M`;
- bf16 mixed precision;
- effective batch size `128`;
- main run target `60k-100k` steps.

## Training Config Defaults

`config/train/codelewm_tiny.yaml` is the local smoke config. It uses:

- `history_size=1`;
- `num_preds=1`;
- `embed_dim=256`;
- `action_view=text`;
- CPU accelerator;
- batch size `4`;
- `16` max steps;
- `float32` precision;
- retrieval loss disabled;
- action-use margin disabled.

`config/train/codelewm_small.yaml` is the initial single-device training config.
It uses:

- `history_size=1`;
- `num_preds=1`;
- `embed_dim=256`;
- `action_view=text`;
- accelerator `auto` with `devices=1`;
- batch size `64`;
- `10000` max steps;
- `bf16-mixed` precision;
- retrieval loss disabled;
- action-use margin disabled.

Neither default config may reference the inherited image-control datasets or
pixel/proprioception loader keys.

Scaled research profiles live under `config/train/scaled/`:

- `codelewm_scaled_cpu.yaml`: CPU rehearsal profile, seed `240119`, batch size
  `8`, `2048` steps, `float32`, expected 6-18h on 8-12 CPU cores, 8-12 GiB RAM,
  1-3 GiB artifacts.
- `codelewm_scaled_mps.yaml`: Apple Silicon development profile, seed `240119`,
  batch size `32`, `10000` steps, `float32`, expected 4-12h on M2/M3 Max class
  hardware, 16-32 GiB unified memory, 1-4 GiB artifacts.
- `codelewm_scaled_gpu_a10g.yaml`: Hugging Face Jobs baseline profile, seed
  `240119`, batch size `64`, `60000` steps, `bf16-mixed`, expected 12-24h on
  `a10g-small`, <=24 GiB device memory, 2-8 GiB artifacts.
- `codelewm_scaled_action_use_margin_gpu_a10g.yaml`: Hugging Face Jobs primary
  action-use follow-up profile, seed `240119`, batch size `64`, `60000` steps,
  `bf16-mixed`, no-action margin enabled with weight `0.25` and margin `0.02`,
  expected 12-24h on `a10g-small`, <=24 GiB device memory, 2-8 GiB artifacts.
- `codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml`: Hugging Face
  Jobs fallback profile with the same A10G budget plus retrieval loss enabled at
  weight `0.05`.

All scaled configs keep `action_view=text` as the headline path. Patch-action
remains diagnostic-only and cannot be used by training configs.

## Indexing And Scoring

Index targets:

- build embeddings for `250k` transitions in one resumable job;
- store vectors in a local index with manifest checksums;
- retrieve top `100` nearest transitions in under `500 ms` on a developer laptop
  for v0.1 scale.

Scoring targets:

- `codelewm score` returns one candidate result in under `2 s` on CPU for fixture
  inputs and under `300 ms` on GPU after model load;
- `codelewm rerank` handles `100` candidates by batching model calls.

## Profiling

Performance reports include:

- examples/sec;
- tokens/sec;
- peak resident memory;
- peak device memory;
- data loading time percentage;
- model forward time percentage;
- index lookup time.

Regression threshold:

- any `>20%` slowdown on fixture performance tests requires a changelog note and
  justification.
