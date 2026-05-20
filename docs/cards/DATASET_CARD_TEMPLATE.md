# CodeLeWM Dataset Card Template

> Copy this file into `docs/cards/<dataset>-<release>.md` and fill in
> every section. The release checklist rejects cards that omit
> required sections, leave manifest IDs blank, or make claims
> without backing evidence.

- Dataset name: `<short slug>`
- Dataset manifest id: `<artifact_id from codelewm.dataset.v1>`
- Dataset manifest path: `<docs-relative path>`
- Schema version: `codelewm.dataset.v1`
- Source git SHA: `<40-char SHA matching manifest.source_git_sha>`
- Source acquisition report: `<path to codelewm.source_acquisition.v1 file>`
- License gate report: `<path to codelewm.public_license_gate.v1 file>`
- Author / curator: `<github-handle>`
- Release tag: `<v0.x.y | v1.x.y>`

## Summary

> One paragraph: what the dataset is, what task it supports, why
> this curation exists. No marketing copy.

## Source Mix

| Source | Rows included | Rows excluded | Reason for exclusion |
| ------ | ------------- | ------------- | -------------------- |
| | | | |

Match against the license gate report's `included_sources` and
`excluded_sources` maps. Discrepancies must be explained in the
caveats section.

## Schema Versions

| Surface | Schema version |
| ------- | -------------- |
| Dataset manifest | `codelewm.dataset.v1` |
| Artifact manifest | `codelewm.artifact_manifest.v1` |
| Source acquisition report | `codelewm.source_acquisition.v1` |
| Public license gate | `codelewm.public_license_gate.v1` |
| Action-discriminative shard report | `codelewm.data.action_discriminative_shard_report.v1` |
| Transition record | `codelewm.transition.v1` |

## Row Counts

| Split | Count | Notes |
| ----- | ----- | ----- |
| train | | |
| val | | |
| test | | |

Match against `dataset_manifest.split_counts`.

## Feature Flags

List every feature the dataset advertises via
`manifest.features` (action_text, action_abs, action_patch, masks,
synthetic, ...).

| Feature | Present | Description |
| ------- | ------- | ----------- |
| | | |

## License Policy

| Field | Value |
| ----- | ----- |
| Public artifact policy | `<exclude | metadata_only | embeddings | full_text>` |
| Allowed licenses | `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `cc0-1.0`, `isc`, `mit`, `unlicense` |
| Gate result | `<pass | fail>` |
| Included rows | `<int>` |
| Excluded rows | `<int>` |
| Blocked rows | `<int>` |

The full license gate report must be linked at the top of the
card. Released datasets without `release_allowed=true` are blocked.

## Curation Procedure

> Describe every filter the rows passed through, in order:
> parseability, license, message length, size bounds, generated-file
> detection, edit-ratio limits, deduplication, split assignment.

Each filter must cite the manifest field, drop record, or report
that backs it. The source acquisition report must identify the
approved source adapter, configured source count, source mix, path
redaction policy, and license-gate result used to fill this card.

## Synthetic Transforms

> If the dataset includes synthetic rows
> (`feature.synthetic=true`), describe the transforms applied, the
> seed, and the share of synthetic rows in each split. Otherwise
> write "no synthetic rows".

## Known Limitations

> Honest bullets: missing source coverage, language coverage gaps,
> documented bias in repository selection, missing licenses, et
> cetera. This section is meant to make the dataset's gaps obvious
> to a downstream consumer; a card without this section is not
> publishable.

## Reproduction

| Step | Command |
| ---- | ------- |
| Build | `codelewm dataset build --config <yaml> --out <dir>` |
| Pack | `codelewm dataset pack --manifest <dir>/manifest.json --out <dir>/hdf5` |
| Verify | `codelewm manifest verify --manifest <dir>/manifest.json` |

The build and pack commands above are the documented surface (see
`docs/spec/02-public-api.md`); the verifier is mandatory before
publication.

## Caveats

> Anything a reader needs to know that doesn't fit the categories
> above. License surprises, dataset-distribution shift compared to
> previous releases, dropped sources, etc.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| -------- | ---- | ------------- | ---- |
| | | | |

At least one curator sign-off is required.
