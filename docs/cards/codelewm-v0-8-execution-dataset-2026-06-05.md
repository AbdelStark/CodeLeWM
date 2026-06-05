# CodeLeWM v0.8 Execution Pass/Fail Pack Dataset Card

- Dataset name: `codelewm-v0-8-execution-passfail-pack`
- HF repo: `abdelstark/codelewm-execution-pack`
- Revision: `v0.8.0-rc1`
- Pack ID: `codelewm-passfail-execution-pack-20260604T185716Z`
- Pack JSONL SHA-256:
  `b0fbe377a9539484ad316dbd5702a03e0147ddaf9fe3183ee710ec628c83194f`
- Manifest schema: `codelewm.execution_pack_manifest.v1`
- Per-record schema: `codelewm.execution_pack_record.v2`
- Claim boundary: `execution_substrate.v1`
- Claim boundary fingerprint:
  `62c4d29c0eaff1b80c22d4a2b25aee00b205bab342bb50add3436db6e524973e`
- Card date: `2026-06-05`

## Summary

This pack is the v0.8 correctness-aware execution substrate. It is derived from
the deterministic WS-D HumanEval completion-label pack and adds per-record
`passed` labels for the supervised `p_pass` head. It is a training/evaluation
data artifact, not an executable code bundle.

The execution-pack data artifact is the deterministic output of running
licensed public Python submissions in an isolated sandbox under a stdlib-only
policy at data-build time. The artifact contains no executable payload; it
contains tokenized code, tokenized inputs, tokenized outputs, and metadata.
Training and inference never execute code. The sandbox is reused only in the
dedicated downstream-evaluation scenario (`execution-rerank`) to label
completion correctness against hidden tests, and only on inputs the operator has
reviewed.

## Links

| Surface | Link |
| --- | --- |
| Dataset repo | `https://huggingface.co/datasets/abdelstark/codelewm-execution-pack/tree/v0.8.0-rc1` |
| Results report | `docs/benchmark/EXECUTION_V0_8_RESULTS_2026-06-05.md` |
| Public artifact index | `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-05.md` |
| Seed 42 model card | `docs/cards/codelewm-v0-8-execution-model-seed-42-2026-06-05.md` |
| Seed 1729 model card | `docs/cards/codelewm-v0-8-execution-model-seed-1729-2026-06-05.md` |

## Provenance And License

| Source | Records | License |
| --- | ---: | --- |
| HumanEval WS-D completion labels | 1,882 | MIT |

The source path recorded in the pack config is `data/raw/humaneval.jsonl`.
Generation used the existing deterministic mutation-distractor WS-D surface
rather than an LLM provider.

## Pack Statistics

| Field | Value |
| --- | ---: |
| Records | 1,882 |
| Train / val / test | 1,646 / 51 / 185 |
| `passed=true` train / val / test | 853 / 31 / 99 |
| `passed=false` train / val / test | 793 / 20 / 86 |
| Execution status `ok` / `raised` | 1,732 / 150 |
| Held-out eval excluded | 0 |
| Sandbox timeout rejects | 26 |

Output type distribution: `int=599`, `bool=512`, `list=354`, `str=159`,
`exception=150`, `float=59`, `tuple=49`.

The pack carries `output_magnitude_bucket` labels for some rows, but the val
split has zero magnitude-labeled rows. That makes the v0.8 magnitude probe not
evaluable and is a known coverage gap.

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
  --revision v0.8.0-rc1 \
  --local-dir /tmp/codelewm-v0-8-pack
uv run codelewm manifest verify \
  --manifest /tmp/codelewm-v0-8-pack/artifact_manifest.json \
  --json
uv run codelewm secret-scan /tmp/codelewm-v0-8-pack --json
```

Expected result: manifest verification `ok=true`; secret scan `ok=true` with
zero findings.

## Claim Boundary

This pack supports the v0.8 correctness-head training and the HumanEval WS-D
diagnostic. It does not by itself support claims about MBPP-Plus, non-Python
code, third-party dependencies, filesystem/network behavior, named semantic
latent axes, or general generated-code utility.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Dataset curator | @AbdelStark | 2026-06-05 |
