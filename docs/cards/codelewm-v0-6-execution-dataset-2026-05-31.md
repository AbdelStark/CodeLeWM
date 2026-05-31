# CodeLeWM v0.6 Execution Pack Dataset Card

- Dataset name: `codelewm-execution-pack`
- HF repo: `abdelstark/codelewm-execution-pack`
- Revision: `v0.6.0`
- Pack ID: `codelewm-execution-pack-20260528T102625Z`
- Pack JSONL SHA-256:
  `d770c5df4b8b81aa7708ab2599f18b638ccea02a69f8b1a87e80d7d579ecf41b`
- Manifest schema: `codelewm.execution_pack_manifest.v1`
- Per-record schema: `codelewm.execution_pack_record.v1`
- Claim boundary: `execution_substrate.v1`
- Claim boundary fingerprint:
  `62c4d29c0eaff1b80c22d4a2b25aee00b205bab342bb50add3436db6e524973e`
- Card date: `2026-05-31`

## Summary

The v0.6 execution pack is the substrate-pivot dataset for CodeLeWM's
execution-trace world model. Each record is a deterministic Python
`(code, input, output)` trace produced by the project sandbox at pack-build
time. This is not a code-edit dataset; it is the Substrate B comparison against
the v0.2 commit-edit substrate.

## Links

| Surface | Link |
| --- | --- |
| Dataset repo | `https://huggingface.co/datasets/abdelstark/codelewm-execution-pack/tree/v0.6.0` |
| Results report | `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` |
| Public artifact index | `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-31.md` |
| Seed 42 model card | `docs/cards/codelewm-v0-6-execution-model-seed-42-2026-05-31.md` |
| Seed 1729 model card | `docs/cards/codelewm-v0-6-execution-model-seed-1729-2026-05-31.md` |
| Paper draft | `docs/papers/two_substrate_paper.tex` |
| arXiv status | `docs/papers/ARXIV_SUBMISSION.md` (`pending operator upload`) |

## Provenance And License

| Source | Records | License |
| --- | ---: | --- |
| MBPP-style source records | 1,419 | CC-BY-4.0 |
| APPS source records | 186 | MIT |

All sources are public and permissively usable under the pack's publication
policy. Attribution metadata is published in the HF repo as `attribution.json`.

## Pack Statistics

| Field | Value |
| --- | ---: |
| Records | 1,605 |
| Train / val / test | 1,369 / 79 / 157 |
| Execution status `ok` / `raised` | 1,594 / 11 |
| Held-out eval excluded | 0 |
| Max inputs per problem | 5 |
| Sandbox nondeterministic rejects | 3 |

Output type distribution: `int=494`, `str=385`, `list=312`, `bool=189`,
`float=88`, `tuple=87`, `dict=33`, `exception=11`, `none=6`.

## Sandbox Policy

| Field | Value |
| --- | --- |
| Policy schema | `codelewm.sandbox_policy.v1` |
| Import allowlist | `stdlib_only` |
| Timeout | 5,000 ms |
| Memory | 1,024 MB |
| CPU seconds | 4 |
| Network | denied |
| Subprocess | denied |
| Determinism check | enabled |
| `PYTHONHASHSEED` | 0 |

## Verification

Round-trip evidence is recorded in
`docs/benchmark/EXECUTION_PACK_PUBLISH_2026-05-28.md` and summarized in
`docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-31.md`.

```bash
hf download abdelstark/codelewm-execution-pack \
  --repo-type dataset --revision v0.6.0 \
  --local-dir /tmp/codelewm-execution-pack-v0-6-0
uv run codelewm manifest verify --manifest /tmp/codelewm-execution-pack-v0-6-0/artifact_manifest.json --json
uv run codelewm secret-scan /tmp/codelewm-execution-pack-v0-6-0 --json
```

Expected result: manifest verification `ok=true`; secret scan `ok=true` with
zero findings.

## Claim Boundary

The dataset supports the narrow execution-substrate research comparison. It
does not, by itself, support claims that CodeLeWM improves generated code,
solves HumanEval / MBPP-Plus tasks, or exposes named semantic latent axes.

## Citation

```bibtex
@misc{codelewm_execution_pack_v06,
  title = {CodeLeWM Execution Pack v0.6: Deterministic Python execution traces for latent world-model training},
  author = {CodeLeWM maintainers},
  year = {2026},
  howpublished = {Hugging Face Datasets},
  url = {https://huggingface.co/datasets/abdelstark/codelewm-execution-pack}
}
```
