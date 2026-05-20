# CodeLeWM v0.2 Action-Swap Dataset Card

- Dataset name: `codelewm-public-shard-commitpackft-python-v0-2-action-swap`
- Run ID: `codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b`
- Dataset artifact id: `dataset-daecac9f9965c563`
- Build artifact id: `dataset-d67b1cd46dc05bea`
- Dataset repo: `abdelstark/codelewm-public-shard`
- Dataset repo path: `runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/pack`
- Results repo path: `abdelstark/codelewm-runs/runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b`
- Source git SHA: `7c7cb0b8fe132e4819f05a77585c254267e77574`
- Benchmark report: `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`
- Evidence tier: scaled public HF artifact evidence
- Card date: `2026-05-20`

## Summary

This dataset card describes the public-safe CommitPackFT Python shard used for
the v0.2 action-swap/inverse-action HF Jobs run. The shard was built through the
source-acquisition and public-license gates, packed into train/validation/test
artifacts, published to Hugging Face, downloaded with `hf download`, and
verified locally from the downloaded manifest chain.

The dataset supports action-contrast diagnostics, including exact-same-before
and near-before pools. The paired v0.2 model still fails the action-use claim
gate, so this dataset is evidence for a negative diagnostic run, not proof of a
positive action-conditioned model.

## Source Mix

| Source | Rows loaded | Rows included by license gate | Rows excluded by license gate |
| --- | ---: | ---: | ---: |
| `commitpackft` | 56,025 | 23,015 | 33,010 |

Packed transitions after parse, size, edit-ratio, generated-file, license, and
dedup filters: `20,645`.

## Split Counts

| Split | Rows |
| --- | ---: |
| train | 18,019 |
| val | 1,291 |
| test | 1,335 |

## Action-Contrast Coverage

| Field | Value |
| --- | ---: |
| Query count | 1,000 |
| `exact_same_before` candidates | 6 |
| `near_before` candidates | 186 |
| `same_file` candidates | 218 |
| `action_cluster` candidates | 281 |
| `edit_shape` candidates | 14,619 |
| `mutation` candidates | 0 |
| `random` candidates | 16,000 |
| Selected train rows | 0 |
| Leakage detected | false |
| Exact-same-before query count | 6 |
| Same-before multi-action query count | 4 |
| Synthetic controlled same-before query count | 0 |

Synthetic controlled transforms are covered by the #171 fixture tests. This
public scaled shard did not add synthetic controlled rows to the HF dataset
artifact.

## License Policy

| Field | Value |
| --- | --- |
| Artifact policy | `full_text` |
| Gate result | pass |
| `release_allowed` | `true` |
| Included rows | 23,015 |
| Excluded rows | 33,010 |
| Blocked rows | 0 |

Included licenses: `mit=12340`, `bsd-3-clause=4704`, `apache-2.0=3994`,
`bsd-2-clause=1448`, `isc=273`, `unlicense=177`, `cc0-1.0=79`.

## Schema Versions

| Surface | Schema version |
| --- | --- |
| Dataset artifact manifest | `codelewm.artifact_manifest.v1` |
| Source acquisition report | `codelewm.source_acquisition.v1` |
| Public license gate | `codelewm.public_license_gate.v1` |
| Dataset pack report | `codelewm.dataset_pack_report.v1` |
| Action-contrast pool report | `codelewm.eval.action_contrast_pool_report.v1` |

## Verification

Downloaded local path:
`.artifacts/hf-download/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/dataset/runs/codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b/pack`.

Verified by:

```bash
CODELEWM_HF_RUN_ID=codelewm-v0-2-action-swap-rerun-20260520-7c7cb0b \
  uv run scripts/hf-verify-codelewm-run --json
```

Result: dataset manifest verification `ok=true`, parent
`dataset-d67b1cd46dc05bea`. Secret scan over the downloaded artifact root
returned `ok=true` with zero findings.

## Intended Use

- Reproduce the v0.2 action-swap/inverse-action scaled training and evaluation
  artifact chain.
- Evaluate whether action-contrast pools reduce no-action dominance.
- Support negative diagnostic analysis of action-conditioned code-edit world
  models.

## Limitations

- The paired v0.2 model does not beat no-action on headline or v0.2
  action-contrast retrieval gates.
- The source is limited to the Python slice of CommitPackFT.
- `mutation` had zero candidates in the scaled action-contrast pool report.
- Public repository visibility does not imply a positive model-quality claim.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Curator | @AbdelStark | 2026-05-20 |
