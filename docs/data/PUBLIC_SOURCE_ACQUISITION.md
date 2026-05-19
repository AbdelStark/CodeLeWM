# Public Source Acquisition

CodeLeWM can train on public code only after source acquisition, license
filtering, and path redaction are recorded as artifacts. This document is the
operator contract for #118.

## Approved Inputs

The current public-safe inputs are:

- `fixture`: checked-in JSON/JSONL fixture records used for deterministic smoke
  evidence.
- `commitpackft`: local CommitPackFT-style `.jsonl` or `.jsonl.gz` shards with
  Python rows and per-row license metadata.

Other source kinds remain unsupported for public artifacts until an adapter,
license policy, and card wording are added.

## License Policy

Public full-text artifacts allow only:

- `apache-2.0`
- `bsd-2-clause`
- `bsd-3-clause`
- `cc0-1.0`
- `isc`
- `mit`
- `unlicense`

Rows with missing, unknown, copyleft, or non-allowlisted licenses are excluded.
The exclusion must appear in both `reports/license_gate_report.json` and
`reports/source_acquisition_report.json`.

## Path Policy

Shareable artifacts must not publish private filesystem paths. Dataset builds
therefore write a sanitized `config.json`, sanitized artifact-manifest command
metadata, and a `codelewm.source_acquisition.v1` report:

```text
<out>/
  config.json
  manifest.json
  dataset_manifest.json
  reports/
    source_acquisition_report.json
    license_gate_report.json
```

Relative checked-in paths may appear as-is. Absolute and home-relative paths are
replaced by a redacted token with a SHA-256 digest so the operator can reconcile
the original source locally without publishing it.

## Validation Command

Use a clean output directory:

```bash
uv run codelewm dataset build \
  --config config/first_results/dataset_build.json \
  --out .artifacts/source-acquisition-smoke \
  --json

uv run codelewm secret-scan .artifacts/source-acquisition-smoke --json

uv run python -m pytest tests/data tests/security tests/manifest
```

For a candidate scaled shard, replace the config path with the checked-in
public-shard config from #119. The build is publishable only when:

- `dataset_manifest.metadata.license_gate_report.release_allowed=true`;
- `dataset_manifest.metadata.source_acquisition_report.release_allowed=true`;
- `source_acquisition_report.public_path_policy.raw_private_paths_published=false`;
- source mix in the dataset card matches
  `source_acquisition_report.dataset_card_fields.source_mix`;
- secret scan returns `ok=true` for the candidate artifact directory.
