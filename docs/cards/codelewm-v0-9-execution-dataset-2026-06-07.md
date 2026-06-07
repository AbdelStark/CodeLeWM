# CodeLeWM v0.9 Execution Pass/Fail Pack Dataset Card

- Dataset name: `codelewm-v0-9-execution-passfail-pack`
- HF repo: `abdelstark/codelewm-execution-pack`
- Revision: `v0.9.0-rc1`
- Pack ID: `codelewm-passfail-execution-pack-20260606T122240Z`
- Pack JSONL SHA-256:
  `a2f994404c00c9129f4265c631e1cfa34d53b39e3b5d5c87959ca9497f1fdbaa`
- Manifest schema: `codelewm.execution_pack_manifest.v1`
- Per-record schema: `codelewm.execution_pack_record.v2`
- Claim boundary: `execution_substrate.v1`
- Card date: `2026-06-07`

## Summary

This pack is the v0.9 cross-benchmark correctness-aware execution substrate. It
combines HumanEval and MBPP-Plus WS-D completion-label data, adds pass/fail
labels for supervised correctness training, and preserves held-out split
coverage for `passed` and `output_magnitude_bucket` probes.

The execution-pack data artifact is the deterministic output of running
licensed public Python submissions in an isolated sandbox under a stdlib-only
policy at data-build time. The artifact contains no executable payload; it
contains tokenized code, tokenized inputs, tokenized outputs, and metadata.
Training and inference never execute code.

## Links

| Surface | Link |
| --- | --- |
| Dataset repo | `https://huggingface.co/datasets/abdelstark/codelewm-execution-pack/tree/v0.9.0-rc1` |
| Results report | `docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md` |
| Public artifact index | `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-07.md` |
| Seed 42 model card | `docs/cards/codelewm-v0-9-execution-model-seed-42-2026-06-07.md` |
| Seed 1729 model card | `docs/cards/codelewm-v0-9-execution-model-seed-1729-2026-06-07.md` |

## Provenance And License

| Source | Records | Notes |
| --- | ---: | --- |
| HumanEval WS-D completion labels | 1,882 | MIT source, deterministic mutation completions |
| MBPP-Plus WS-D completion labels | 306 | public benchmark source, deterministic mutation completions |

Generation used deterministic mutation-distractor WS-D surfaces rather than an
LLM provider.

## Pack Statistics

| Field | Value |
| --- | ---: |
| Records | 2,188 |
| Train / val / test | 1,928 / 57 / 203 |
| `passed=true` train / val / test | 960 / 18 / 103 |
| `passed=false` train / val / test | 968 / 39 / 100 |
| HumanEval / MBPP-Plus records | 1,882 / 306 |
| Sandbox timeout rejects | 26 |
| `pos_weight` | 1.0240518038852915 |

Output magnitude labels are present in held-out splits:

| Split | Counts |
| --- | --- |
| Train | `large=47`, `medium=161`, `negative=67`, `small=404`, `zero=86` |
| Val | `large=6`, `medium=17`, `small=9`, `zero=10` |
| Test | `medium=18`, `small=13`, `zero=7` |

## Readiness Gates

| Gate | Status |
| --- | --- |
| Pass/fail classes present | PASS |
| Problem leakage absent | PASS |
| Held-out label coverage for `output_magnitude_bucket` | PASS |
| Secret scan | PASS |

## Sandbox Policy

| Field | Value |
| --- | --- |
| Policy schema | `codelewm.sandbox_policy.v1` |
| Import allowlist | `stdlib_only` |
| Timeout | 5,000 ms |
| Memory | 512 MB |
| CPU seconds | 10 |
| Network | denied |
| Subprocess | denied |
| Determinism check | enabled |
| `PYTHONHASHSEED` | 0 |

## Verification

```bash
hf download abdelstark/codelewm-execution-pack \
  --repo-type dataset \
  --revision v0.9.0-rc1 \
  --local-dir .artifacts/v0_9/hf-pack-download
uv run codelewm manifest verify \
  --manifest .artifacts/v0_9/hf-pack-download/artifact_manifest.json \
  --json
uv run codelewm secret-scan .artifacts/v0_9/hf-pack-download --json
```

Expected result: manifest verification `ok=true`; secret scan `ok=true` with
zero findings.

## Claim Boundary

This pack supports v0.9 correctness-aware training and cross-benchmark
diagnostics. It does not by itself support claims about generated-code utility,
non-Python code, third-party dependencies, filesystem/network behavior, named
semantic latent axes, or general HumanEval / MBPP-Plus benchmark improvement.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Dataset curator | @AbdelStark | 2026-06-07 |
