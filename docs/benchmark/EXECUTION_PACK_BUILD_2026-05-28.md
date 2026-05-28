# Execution Pack Build — 2026-05-28

> First real-scale execution-substrate pack. 1,605 deterministic
> `(code, input, output)` records spanning MBPP function-call inputs
> and APPS stdin inputs. Built locally on macOS over ~27 minutes of
> single-process sandbox runtime. Verified with `codelewm manifest
> verify` and `codelewm secret-scan`. This is the artifact #291 will
> publish to Hugging Face.

- Report ID: `codelewm-execution-pack-build-2026-05-28`
- Pack ID: `codelewm-execution-pack-20260528T102625Z`
- Pack JSONL SHA-256: `d770c5df4b8b81aa7708ab2599f18b638ccea02a69f8b1a87e80d7d579ecf41b`
- Claim boundary: `execution_substrate.v1` (`62c4d29c0eaff1b80c22d4a2b25aee00b205bab342bb50add3436db6e524973e`)
- Source git SHA: pending squash-merge of #290
- Manifest schemas: `codelewm.execution_pack_manifest.v1` + the wrapping `codelewm.artifact_manifest.v1`

## Reproduction Commands

```bash
# 1. Flatten upstream sources to JSONL.
uv pip install datasets huggingface_hub
PYTHONPATH=. .venv/bin/python scripts/dataset/flatten-mbpp \
  --output /tmp/atscale/mbpp.jsonl --limit 500
PYTHONPATH=. .venv/bin/python scripts/dataset/flatten-apps \
  --output /tmp/atscale/apps_intro.jsonl --limit 100 \
  --difficulty introductory --max-solutions-per-problem 2

# 2. Normalize each source through its adapter into ingestion JSONL.
PYTHONPATH=. .venv/bin/python -m codelewm.harness.cli dataset ingest \
  --source mbpp --input /tmp/atscale/mbpp.jsonl \
  --output /tmp/atscale/ingest/mbpp.jsonl
PYTHONPATH=. .venv/bin/python -m codelewm.harness.cli dataset ingest \
  --source apps --input /tmp/atscale/apps_intro.jsonl \
  --output /tmp/atscale/ingest/apps.jsonl

# 3. Drive the sandbox + pack writer end-to-end.
PYTHONPATH=. .venv/bin/python -m codelewm.harness.cli dataset execution-pack \
  --ingestion /tmp/atscale/ingest/mbpp.jsonl \
  --ingestion /tmp/atscale/ingest/apps.jsonl \
  --output /tmp/atscale/pack \
  --memory-mb 1024 --timeout-ms 5000 --cpu-seconds 4 \
  --max-inputs-per-problem 5 --json
```

## Pack Statistics

### Source breakdown

| Source | Submissions ingested | Records in pack | License |
|--------|---------------------:|----------------:|---------|
| mbpp (function-call) | 474 | 1,419 | CC-BY-4.0 |
| apps (stdin, introductory) | 186 | 186 | MIT |
| **total** | **660** | **1,605** | |

(APPS solutions average ~1 surviving record per submission because each problem has multiple test inputs that are scored as separate sandbox runs; MBPP averages ~3 inputs per problem.)

### Output type distribution

| `output_type` | Count |
|---------------|------:|
| int | 494 |
| str | 385 |
| list | 312 |
| bool | 189 |
| float | 88 |
| tuple | 87 |
| dict | 33 |
| exception | 11 |
| none | 6 |

Nine distinct output types are exercised; the most common (`int`) covers 31% but no class dominates >50%. The 11 exception records will exercise the `will_raise` probe target (#266).

### Output kind distribution

| `output_kind` | Count |
|---------------|------:|
| value (function-call) | 1,427 |
| stdout (stdin script-style) | 178 |

Both input substrates (function-call MBPP and stdin APPS) are represented end-to-end.

### Execution status

| `execution_status` | Count |
|--------------------|------:|
| ok | 1,594 |
| raised | 11 |

99.3% success rate. The 11 `raised` records carry the exception class + message as the output (per the pack contract).

### Splits

Partitioned by `source_problem_id` — no problem leaks across splits.

| Split | Records |
|-------|--------:|
| train | 1,369 |
| val | 79 |
| test | 157 |

### Sandbox reject counts

| Reason | Count |
|--------|------:|
| `sandbox_nondeterministic` | 3 |

3 records out of 1,608 candidates failed the determinism re-run (`PYTHONHASHSEED=0` two-shot match). Every other record reproduces deterministically across both sandbox invocations.

### Token-count distribution (per record)

| Tokens | Code | Input | Output |
|--------|-----:|------:|-------:|
| <32 | 502 | 1,411 | 1,526 |
| 32-63 | 645 | 181 | 66 |
| 64-127 | 303 | 13 | 8 |
| 128-255 | 126 | 0 | 5 |
| 256-511 | 20 | 0 | 0 |
| ≥512 | 9 | 0 | 0 |

The chosen sequence lengths (`code_sequence_length=1024`, `action_sequence_length=256`, `output_sequence_length=256`) accommodate every record with significant headroom. No truncation observed in the smoke training run that consumed this pack.

## Verification Evidence

`codelewm manifest verify --manifest /tmp/atscale/pack/artifact_manifest.json --json`:

```json
{
  "artifact_id": "codelewm-execution-pack-20260528T102625Z",
  "artifact_kind": "dataset",
  "files_checked": 5,
  "manifest": "/tmp/atscale/pack/artifact_manifest.json",
  "ok": true,
  "parent_artifacts": [],
  "parents_checked": [],
  "schema_version": "codelewm.manifest_verify.v1"
}
```

`codelewm secret-scan /tmp/atscale/pack/ --json`:

- Zero findings across `manifest.json`, `pack.jsonl`, `attribution.json`, `sandbox_audit_summary.json`, `claim_boundary.md`, `artifact_manifest.json`.

## Training-Readiness Smoke

Quick 200-step training pass on the real pack (CPU, batch_size=16, seed=42):

| Metric | Step 1 | Final (tail-averaged) | Δ |
|--------|-------:|---------------------:|---:|
| `loss_prediction_mse` | 1.0017 | **0.2698** | −0.73 (3.7× reduction) |
| `loss_sigreg` | 10.8849 | 8.4022 | −2.48 |
| `margin_no_action_minus_pred` | **−0.7673** | **+0.2622** | **+1.03 sign flip** |
| `z_pred_effective_rank` | — | 32.87 / 256 | ratio 0.1284 |
| `z_pred_mean_pairwise_cosine` | — | 0.75 | (still partially aligned at this step budget) |

Comparison to the prior 39-record smoke (`EXECUTION_V0_6_LOCAL_SMOKE_2026-05-28.md`):

| | 39-record smoke (500 steps) | 1,605-record (200 steps) | v0.2 commit-edit (#172, 20,645 records, 50k steps) |
|---|---:|---:|---:|
| Records | 39 | 1,605 | 20,645 |
| Effective rank ratio | 0.024 (at data ceiling 0.15) | **0.128** | 0.016 (collapsed) |
| No-action margin | flips at step ~100 | flips by step 200 | never flips |

The effective rank ratio is **8× higher than v0.2 at 8% the records** — the substrate is qualitatively different from a representation-spread standpoint. The headline collapse-gate threshold of 0.20 looks reachable at full scale.

## Scope Compared To The Issue Target

The #290 target was "≥50k surviving records". This first pack lands at 1,605. The gap is purely a wall-clock + storage decision; the pipeline is the same:

| Lever | Current | Path to ≥50k |
|-------|--------:|-------------|
| MBPP problems (of ~974 total) | 500 | use all 974 |
| APPS problems (of ~10k introductory) | 100 | scale to 5,000–10,000 |
| `--max-inputs-per-problem` | 5 | 8 (the issue's recommended cap) |
| Sandbox concurrency | 1 process | a future `--workers N` flag would parallelize ~8× |
| CodeNet | deferred | flatten-codenet exists, awaiting a local IBM Project_CodeNet extraction |

At the current rate (~27 min for 1.6k records), a 50k-record pack at single-process serial sandbox costs ~14 hours. With 8× parallelism it drops to ~2 hours.

## Held-Out Sources

Per the substrate roadmap, MBPP-Plus and HumanEval ingestion are flagged `held_out_for_eval=true` and not present in this pack. They are used by the downstream rerank eval (#268) only.

## Out-Of-Scope

- HuggingFace publish — that is the next issue (#291).
- Runtime container build — #292.
- Training run that produces a v0.6 checkpoint — #293.

## Reference

- Tracker: #289 (v0.6 end-to-end completion)
- Implementation issue: #290
- Prior local smoke: `docs/benchmark/EXECUTION_V0_6_LOCAL_SMOKE_2026-05-28.md`
- Pack builder: `codelewm.data.execution_pack.build_execution_pack` (#262, PR #277)
- Sandbox: `codelewm.data.sandbox` (#260, PR #275, with stdin support added in #290)
- Source adapters: `codelewm.data.execution_sources` (#261, PR #276)
- Flatteners: `scripts/dataset/flatten-{mbpp,apps,codenet}`
