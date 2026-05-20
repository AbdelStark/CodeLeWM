# CodeLeWM Action-Use Dataset Card

- Dataset name: `codelewm-public-shard-commitpackft-python-action-use-v0-3`
- Run ID: `codelewm-action-use-20260520-6650183`
- Dataset artifact id: `dataset-67895f8dc3e217c4`
- Build artifact id: `dataset-9750a00ae69ee5e1`
- Dataset repo: `abdelstark/codelewm-public-shard`
- Dataset repo path: `runs/codelewm-action-use-20260520-6650183/pack`
- Results repo path: `abdelstark/codelewm-runs/runs/codelewm-action-use-20260520-6650183`
- Source git SHA: `6650183`
- Benchmark report: `docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md`
- Evidence tier: scaled private HF artifact evidence
- Card date: `2026-05-20`

## Summary

This dataset card describes the public-safe shard used for the #154 action-use
HF Jobs follow-up run. The shard is built from the Python slice of
`bigcode/commitpackft`, filtered through the repository's source-acquisition
and license gates, packed into train/validation/test artifacts, published
privately to Hugging Face, downloaded with `hf download`, and verified locally
from the downloaded manifest chain.

The shard passes the action-discriminative readiness gate, but the paired model
still does not beat the no-action retrieval baseline. Treat this dataset as
valid evidence for the negative #154 run, not proof of positive
action-conditioned model quality.

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

## Action-Discriminative Coverage

| Field | Value |
| --- | ---: |
| Claim-readiness gate | true |
| Action text nonempty ratio | 1.0 |
| Action abstract nonempty ratio | 0.980673 |
| Held-out rows | 2,626 |
| Same-file or near-before pairs | 2,576 |
| Action-cluster pairs | 15,447 |
| Edit-size bucket pairs | 86,486,992 |
| Unique action signatures | 19,064 |

Available hard-negative pools: `action_cluster`, `diff_shape_controlled`,
`edit_size_controlled`, `same_before_different_after`, and `same_file`.
Unavailable hard-negative pools: `near_before_different_after`.

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
| Action-discriminative shard report | `codelewm.data.action_discriminative_shard_report.v1` |

## Verification

Downloaded local path:
`.artifacts/hf-download/codelewm-action-use-20260520-6650183/dataset/runs/codelewm-action-use-20260520-6650183/pack`.

Verified command:

```bash
uv run codelewm manifest verify \
  --manifest .artifacts/hf-download/codelewm-action-use-20260520-6650183/dataset/runs/codelewm-action-use-20260520-6650183/pack/manifest.json \
  --parent-manifest .artifacts/hf-download/codelewm-action-use-20260520-6650183/results/runs/codelewm-action-use-20260520-6650183/build/manifest.json \
  --json
```

Result: `ok=true`, parent `dataset-9750a00ae69ee5e1`. Secret scan over the
downloaded artifact root returned `ok=true` with zero findings.

## Intended Use

- Reproduce the #154 action-use scaled training and evaluation artifact chain.
- Evaluate whether no-action margin training improves action-conditioned
  retrieval over no-action baselines.
- Build hard-negative retrieval and surprise diagnostics over public-safe
  code-transition rows.

## Limitations

- This shard did not produce a positive action-conditioning result with the
  paired #154 model.
- The source is limited to the Python slice of CommitPackFT.
- `near_before_different_after` was unavailable because the pair scan was
  truncated.
- Repositories remain private until release gates and claim wording are
  reviewed.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Curator | @AbdelStark | 2026-05-20 |
