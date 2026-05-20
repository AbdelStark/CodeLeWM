# CodeLeWM First-Results Dataset Card

- Dataset name: `codelewm-first-results-fixture`
- Dataset artifact id: `dataset-0bfb8da3ed5a70e5`
- Dataset manifest path: `.artifacts/first-results/pack/dataset_manifest.json`
- Artifact manifest path: `.artifacts/first-results/pack/manifest.json`
- Build artifact id: `dataset-d98a48587399bc7a`
- Source git SHA: `ce687cbd501fd3b113f7286e19a4ffc018f66949`
- Source acquisition report: `.artifacts/first-results/build/reports/source_acquisition_report.json`
- License gate report: `.artifacts/first-results/build/reports/license_gate_report.json`
- Benchmark report: `docs/benchmark/FIRST_RESULTS.md`
- Curator: `@AbdelStark`
- Evidence tier: smoke fixture, not scaled release evidence
- Card date: `2026-05-19`

## Summary

This card describes the tiny public-safe fixture dataset used by the
`scripts/first-results` smoke workflow. The dataset supports CodeLeWM transition
packing, CPU training, retrieval evaluation, ablation reporting, surprise
evaluation, transition-index construction, and scorer/reranker quality plumbing.
It is suitable for reproducibility and release-gate testing only; it is not a
training corpus for a model-quality claim.

## Source Mix

| Source | Rows loaded | Rows included | Rows excluded | Reason for exclusion |
| ------ | ----------- | ------------- | ------------- | -------------------- |
| `local_repo` | 4 | 3 | 1 | One fixture row is filtered as `non_python_path`; included rows pass the permissive license gate. |

The table matches the source acquisition and license gate evidence for
`.artifacts/first-results/build`: `included_sources.local_repo=3` and
`excluded_sources.local_repo=1`.

## Schema Versions

| Surface | Schema version |
| ------- | -------------- |
| Dataset manifest | `codelewm.transition.v1` |
| Artifact manifest | `codelewm.artifact_manifest.v1` |
| Source acquisition report | `codelewm.source_acquisition.v1` |
| Public license gate | `codelewm.public_license_gate.v1` |
| Dataset pack report | `codelewm.dataset_pack_report.v1` |
| Action-discriminative shard report | `codelewm.data.action_discriminative_shard_report.v1` |

## Row Counts

| Split | Count | Notes |
| ----- | ----- | ----- |
| train | 2 | Used for the tiny training run and transition index. |
| val | 1 | Used as the held-out query for retrieval and surprise smoke checks. |
| test | 0 | No test rows in this fixture. |

Total packed rows: `3`.

## Feature Flags

| Feature | Present | Description |
| ------- | ------- | ----------- |
| `action_text` | yes | Text action is the headline training and retrieval action view. |
| `action_patch` | no | Patch action is not included in the packed fixture features. |
| `action_abs` | no | Abstract-action rows are blocked in the ablation report. |
| `synthetic` | no | The fixture contains no synthetic transform rows. |

## License Policy

| Field | Value |
| ----- | ----- |
| Public artifact policy | `full_text` |
| Allowed licenses | `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `cc0-1.0`, `isc`, `mit`, `unlicense` |
| Gate result | pass |
| Included rows | 3 |
| Excluded rows | 1 |
| Blocked rows | 0 |
| Included licenses | `apache-2.0`: 1, `mit`: 2 |
| Excluded licenses | `mit`: 1 |

`release_allowed=true` for this local fixture artifact. That does not imply the
future scaled corpus is release-ready.

## Curation Procedure

The build uses `config/first_results/dataset_build.json` with deterministic seed
`7` and the fixture source
`tests/fixtures/dataset_build/records.jsonl`. Rows pass through source loading,
path filtering, parseability checks, size/edit-ratio filters, license policy,
deduplication, and deterministic split assignment before packing. The source
acquisition report records that relative fixture paths may be published and that
raw private or home-relative paths are not published.

## Synthetic Transforms

No synthetic rows.

## Known Limitations

- The dataset contains only 3 packed rows and one held-out validation query.
- Retrieval Recall@k is saturated because the held-out candidate pool has one entry.
- The fixture has no test split and no scaled source diversity.
- Patch-action and abstract-action evidence are absent and represented as blocked
  rows in the action-view ablation report.
- This card supports smoke reproducibility, not a public dataset release claim.

## Reproduction

| Step | Command |
| ---- | ------- |
| Build and pack | `uv run scripts/first-results --overwrite` |
| Verify pack | `uv run codelewm manifest verify --manifest .artifacts/first-results/pack/manifest.json --parent-manifest .artifacts/first-results/build/manifest.json --json` |
| Secret scan | `uv run codelewm secret-scan .artifacts/first-results docs/benchmark/FIRST_RESULTS.md --json` |

## Caveats

The card is intentionally filled from first-results smoke artifacts. Replace it
with a scaled artifact-backed dataset card before changing repository visibility
or making a dataset-quality claim.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| -------- | ---- | ------------- | ---- |
| AbdelStark | Curator | @AbdelStark | 2026-05-19 |
