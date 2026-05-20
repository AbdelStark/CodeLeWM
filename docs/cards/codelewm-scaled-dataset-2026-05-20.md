# CodeLeWM Scaled Dataset Card

- Dataset name: `codelewm-public-shard-commitpackft-python-v0-3`
- Run ID: `codelewm-scaled-20260520-9699b53`
- Dataset artifact id: `dataset-ef8ad3f4f48dea9e`
- Build artifact id: `dataset-5a1d4677b02c75f2`
- Dataset repo: `abdelstark/codelewm-public-shard`
- Dataset repo path: `runs/codelewm-scaled-20260520-9699b53/pack`
- Results repo path: `abdelstark/codelewm-runs/runs/codelewm-scaled-20260520-9699b53`
- Source git SHA: `9699b5309e43a3278f272663ef60cda23040d92a`
- Benchmark report: `docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md`
- Evidence tier: scaled public HF artifact evidence
- Card date: `2026-05-20`

## Summary

This dataset card describes the first scaled CodeLeWM public-safe shard used for
the HF Jobs run `codelewm-scaled-20260520-9699b53`. The shard is built from the
Python split of `bigcode/commitpackft`, filtered through the repository's
source-acquisition and license gates, packed into train/validation/test HDF5
artifacts, published to Hugging Face, downloaded with `hf download`,
and verified locally from the downloaded manifest chain.

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

Excluded licenses include copyleft, missing, and unsupported licenses recorded
in `reports/license_gate_report.json`; those rows are not part of the published
full-text artifact.

## Filtering Notes

Main post-source filtering drop reasons:

| Reason | Rows |
| --- | ---: |
| empty_state | 13,502 |
| edit_ratio | 13,012 |
| parse_error | 3,378 |
| license_denied | 2,583 |
| whitespace_only_change | 373 |
| message_length | 139 |
| generated_file | 22 |
| edit_size | 1 |

## Schema Versions

| Surface | Schema version |
| --- | --- |
| Dataset artifact manifest | `codelewm.artifact_manifest.v1` |
| Source acquisition report | `codelewm.source_acquisition.v1` |
| Public license gate | `codelewm.public_license_gate.v1` |
| Dataset pack report | `codelewm.dataset_pack_report.v1` |

## Verification

Downloaded local path:
`.artifacts/hf-download/codelewm-scaled-20260520-9699b53/dataset/runs/codelewm-scaled-20260520-9699b53/pack`.

Verified command:

```bash
uv run codelewm manifest verify \
  --manifest .artifacts/hf-download/codelewm-scaled-20260520-9699b53/dataset/runs/codelewm-scaled-20260520-9699b53/pack/manifest.json \
  --parent-manifest .artifacts/hf-download/codelewm-scaled-20260520-9699b53/results/runs/codelewm-scaled-20260520-9699b53/build/manifest.json \
  --json
```

Result: `ok=true`, files checked `29`, parent `dataset-5a1d4677b02c75f2`.

Secret scan over the downloaded dataset repo artifact returned `ok=true` with
zero findings.

## Intended Use

- Reproduce the scaled CodeLeWM training and evaluation artifact chain.
- Evaluate one-step text-action latent transition modeling for code edits.
- Build retrieval and surprise benchmarks over public-safe code-transition rows.

## Limitations

- This is a first public-safe shard, not a complete multilingual or multi-domain
  code-edit corpus.
- The source is limited to the Python slice of CommitPackFT.
- The paired scaled model did not beat the no-action baseline on headline
  retrieval, so this dataset should not be presented as proving action-conditioned
  model quality by itself.
- Repositories are public diagnostic artifact repositories; positive
  model-quality claims remain blocked by the no-action baseline result.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Curator | @AbdelStark | 2026-05-20 |
