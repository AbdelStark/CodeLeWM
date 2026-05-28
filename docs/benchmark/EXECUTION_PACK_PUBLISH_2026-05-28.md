# Execution Pack Publish Evidence — 2026-05-28

> The at-scale execution pack from `EXECUTION_PACK_BUILD_2026-05-28.md`
> is published live to Hugging Face. This is the publication-evidence
> document for #291.

## Live Artifact

- **Repo**: [`abdelstark/codelewm-execution-pack`](https://huggingface.co/datasets/abdelstark/codelewm-execution-pack)
- **Revision tag**: `v0.6.0`
- **Visibility**: **public**
- **Owner**: `abdelstark` (verified via `hf auth whoami`)
- **Commit**: [`776725fb6bc47f8297f506d2fb87218d1768f02a`](https://huggingface.co/datasets/abdelstark/codelewm-execution-pack/commit/776725fb6bc47f8297f506d2fb87218d1768f02a)
- **Local source pack ID**: `codelewm-execution-pack-20260528T102625Z`
- **Local pack.jsonl SHA-256**: `d770c5df4b8b81aa7708ab2599f18b638ccea02a69f8b1a87e80d7d579ecf41b`
- **Claim boundary fingerprint**: `62c4d29c0eaff1b80c22d4a2b25aee00b205bab342bb50add3436db6e524973e` (`execution_substrate.v1`)

## Pre-Publish Gate

Every check in `codelewm.data.execution_pack.run_pre_publish_gate`
passed (`gate.allowed=true`):

| Check | Status |
|-------|:------:|
| `manifest_schema_version_supported` | ✅ |
| `pack_jsonl_present` | ✅ |
| `attribution_json_present` | ✅ |
| `sandbox_audit_summary_present` | ✅ |
| `claim_boundary_embedded` | ✅ |
| `pack_jsonl_checksum_matches` | ✅ |
| `claim_boundary_fingerprint_matches` | ✅ |
| `all_licenses_permissive` | ✅ |
| `every_source_has_attribution` | ✅ |
| `record_count_nonzero` | ✅ |

License breakdown (from manifest): `CC-BY-4.0: 1419, MIT: 186` — both
permissive. Source attribution URLs recorded for `mbpp` (HuggingFace
canonical) and `apps` (codeparrot/apps).

## Publish Command (verbatim)

```bash
export HF_TOKEN=...  # sourced from .env
PYTHONPATH=. python scripts/hf-publish-execution-pack \
  --pack-dir /tmp/atscale/pack \
  --repo-id abdelstark/codelewm-execution-pack \
  --revision v0.6.0 \
  --public \
  --no-dry-run
```

Output:

```
published /tmp/atscale/pack -> abdelstark/codelewm-execution-pack@v0.6.0
commit: https://huggingface.co/datasets/abdelstark/codelewm-execution-pack/commit/776725fb6bc47f8297f506d2fb87218d1768f02a
```

## Round-Trip Verification

```bash
hf download abdelstark/codelewm-execution-pack \
  --repo-type dataset --revision v0.6.0 \
  --local-dir /tmp/codelewm-execution-pack-v0-6-0
codelewm manifest verify --manifest .../artifact_manifest.json --json
codelewm secret-scan .../ --json
```

Results:

- `Fetching 8 files: 100% in ~2s` — pack JSONL + manifest + sidecars
  + auto-generated README.md from the publish script.
- `codelewm manifest verify` → `ok: true`, 5 files checked.
- `codelewm secret-scan` → `ok: true`, zero findings.
- The downloaded `pack.jsonl` matches the local SHA-256 exactly
  (`d770c5df4b8b81aa...`).

## What's In The Public Repo

```
README.md                  # auto-rendered dataset card from the publish script
artifact_manifest.json     # codelewm.artifact_manifest.v1, kind=dataset
manifest.json              # codelewm.execution_pack_manifest.v1
attribution.json           # {mbpp: <url>, apps: <url>}
claim_boundary.md          # verbatim execution_substrate.v1
sandbox_audit_summary.json # {sandbox_nondeterministic: 3}
pack.jsonl                 # 1,605 records (~3.0 MB)
```

## Notes

- The publish script's first attempt failed with `RevisionNotFoundError`
  because `HfApi.upload_folder(revision=X)` only accepts a branch that
  already exists. The script now uploads to `main` first then
  `create_tag(..., tag=v0.6.0, exist_ok=True)`. This is the idiomatic
  HF Hub pattern for "tag a release on upload."
- The auto-generated README.md is the rendered dataset card from
  `codelewm.data.execution_pack.render_dataset_card` plus the required-
  language paragraph from the claim boundary. It does not yet have
  YAML metadata (license / task_categories tags); HF warns about this
  but the upload succeeds. Future work can add the `language: en`,
  `license: cc-by-4.0`, `tags: world-model, jepa, code-execution`
  YAML block.
- The dataset card template at
  `docs/cards/dataset_card.execution_pack.v1.md` is the canonical
  layout; this publish's card is its first concrete rendering.

## What Just Got Unblocked

- #293 (train): the launcher's `CODELEWM_EXECUTION_PACK_REPO_ID=abdelstark/codelewm-execution-pack`
  + `CODELEWM_EXECUTION_PACK_REVISION=v0.6.0` env-var references now
  resolve.
- The benchmark template's
  `docs/benchmark/EXECUTION_V0_6_RESULTS_TEMPLATE.md` "Reproducibility
  Chain" row for "Execution pack" can be filled in with the live
  revision URL.

## Reference

- Pack-build report: `docs/benchmark/EXECUTION_PACK_BUILD_2026-05-28.md`
- Tracker: #289
- Implementation issue: #291
- Publish script: `scripts/hf-publish-execution-pack`
- Pre-publish gate: `codelewm.data.execution_pack.run_pre_publish_gate`
- Dataset card template: `docs/cards/dataset_card.execution_pack.v1.md`
