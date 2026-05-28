# Dataset Card Template: CodeLeWM Execution Pack v1

> This template is the canonical layout of the dataset card published
> with the execution-substrate pack. The publish script
> (`scripts/hf-publish-execution-pack`) renders a concrete card from
> the manifest at upload time; this file documents the structure and
> the required language.

- Dataset name: `codelewm-execution-pack`
- Dataset manifest schema: `codelewm.execution_pack_manifest.v1`
- Per-record schema: `codelewm.execution_pack_record.v1`
- Claim boundary: `execution_substrate.v1`
- Substrate: execution-trace world model (RFC-0014)

## Summary

The CodeLeWM execution pack is the **substrate-pivot dataset** for the
v0.6 line. Each line of `pack.jsonl` is a `(code, input, output)`
triple where the output was captured by running a licensed public
Python submission inside a sandboxed deterministic executor at
data-build time. The pack is the training substrate for the
execution-trace world model and the eval substrate for the latent
probe targets that are specific to execution semantics.

This pack is **not** a code-edit dataset. Compare with
`codelewm-public-shard` (commit-edit substrate) when interpreting
results.

## Provenance

| Source | Records | License | Attribution URL |
| ------ | ------- | ------- | --------------- |

(Filled in by the publish script from the manifest.)

## Schema Versions

| Surface | Schema version |
| ------- | -------------- |
| Manifest | `codelewm.execution_pack_manifest.v1` |
| Per-record | `codelewm.execution_pack_record.v1` |
| Claim boundary | `execution_substrate.v1` |

## Sandbox Policy

The policy applied at data-build time is recorded verbatim in
`manifest.json` under `sandbox_policy`. Default policy:

| Field | Value |
| ----- | ----- |
| `import_allowlist` | `stdlib_only` |
| `timeout_ms` | `5000` |
| `memory_mb` | `256` (rlimit_data) |
| `cpu_seconds` | `10` |
| `deny_network` | `true` |
| `deny_subprocess` | `true` |
| `deny_filesystem_writes_outside_scratch` | `true` |
| `determinism_check` | `true` |
| `output_truncation_bytes` | `4096` |
| `stdout_truncation_bytes` | `4096` |
| `python_hash_seed` | `0` |

See `docs/operations/sandbox_policy.md` for the policy threat model.

## Determinism Guarantees

Every packed record was produced by **two** sandbox runs of the same
`(code, input)` pair with identical `PYTHONHASHSEED`. Records whose
two runs disagreed are dropped and tallied as
`nondeterministic` in `sandbox_audit_summary.json`.

## Split Policy

Records are partitioned by `source_problem_id`. No problem leaks
across train / val / test. MBPP-Plus and HumanEval ingestion rows
are flagged `held_out_for_eval=true` at the adapter (#261) and the
pack builder (#262) refuses to include them.

## Tokenization

| Field | Value |
| ----- | ----- |
| Tokenizer | `codelewm.tokenizer.blake2b_hash` |
| Version | `v1` |
| Vocabulary | stable 31-bit `blake2b` hash per token |

The tokenizer is identical to the one used by the commit-edit pack;
this keeps the encoder weights interchangeable across substrates.

## Pack Statistics

(Filled in by the publish script from the manifest.)

- record_count
- split_counts
- output_type_distribution
- output_kind_distribution
- execution_status_distribution
- sandbox_reject_counts
- held_out_eval_excluded_count

## Claim Boundary

This pack is governed by `execution_substrate.v1`. The verbatim text
is included as `claim_boundary.md` and the SHA-256 fingerprint is
recorded in `manifest.claim_boundary.fingerprint`. **Both the
dataset card and the model card MUST include the required-language
paragraph verbatim** from the claim boundary file.

## How To Verify

```bash
hf download <repo-id> --revision <revision> --local-dir <download-dir>
uv run codelewm manifest verify --manifest <download-dir>/manifest.json --json
uv run codelewm secret-scan <download-dir> --json
```

## Limitations

- Python-only. The sandbox refuses non-Python code at ingestion.
- Stdlib-only. Third-party imports are denied by the import hook.
- Deterministic-only. Programs that depend on `time`, `random` without
  a fixed seed, network, environment, or filesystem state are
  filtered.
- The substrate-pivot tracker (#259) defines the headline claim
  gates this dataset is built to support; see the run manifest for
  the v0.6 line for the actual per-gate evidence.

## Citation

```bibtex
@misc{codelewm_execution_pack_v1,
  title = {CodeLeWM Execution Pack v1: Deterministic Python execution traces for latent world-model training},
  author = {CodeLeWM maintainers},
  year = {2026},
  howpublished = {Hugging Face Datasets},
  url = {https://huggingface.co/datasets/abdelstark/codelewm-execution-pack}
}
```

Per-source citation lines are inserted by the publish script from
the manifest's `parent_artifacts` and `source_breakdown`.
