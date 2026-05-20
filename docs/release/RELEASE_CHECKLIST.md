# CodeLeWM Release Checklist

> Every release tag must pass this checklist. Reviewers tick the
> boxes against artifact manifests, benchmark reports, security
> evidence, and license evidence — claims without backing manifests
> are blocked.

Filled release-freeze instances:

- `docs/release/RELEASE_FREEZE_2026-05-20.md` records the #126 private
  diagnostic artifact freeze. The later #159 remediation run is also
  negative/diagnostic. HF artifact repositories may be public, but public
  positive action-conditioning claims remain blocked until a future artifact
  passes the action-use claim gate.

- Release tag: `<v0.x.y | v1.x.y>`
- Release date (UTC): `<YYYY-MM-DD>`
- Source git SHA: `<40-char SHA matching every manifest.source_git_sha>`
- Release shepherd: `<github-handle>`
- Codeowner sign-off: `<github-handle>`
- Linked release notes: `<PR or release URL>`
- Schema-version snapshot: see `CHANGELOG.md` for the canonical
  table of public schema versions shipped with this release.

## Pre-Flight

- [ ] **Branch is up to date with `main`.** Evidence: `git log
      origin/main..HEAD` is empty for the release branch.
- [ ] **CI is green on the release commit.** Evidence: GitHub Actions
      run URL for `.github/workflows/pr.yml`.
- [ ] **CHANGELOG.md has a dated section for this release.**
      Evidence: a non-empty `## [<tag>] - YYYY-MM-DD` section with at
      least one of Added / Changed / Deprecated / Removed / Fixed /
      Security populated; the "Unreleased" section is empty.
- [ ] **Implementation tracker is up to date.** Evidence:
      `docs/roadmap/IMPLEMENTATION.md` reflects current
      open/closed state and the contract test passes.
- [ ] **Package publishing gate is current.** Evidence:
      `docs/release/PACKAGE_PUBLISHING.md` matches the release
      workflow and package-build CI job.
- [ ] **Dependency audit and provenance gate is current.**
      Evidence: `docs/release/DEPENDENCY_PROVENANCE.md` matches the
      release workflow and package-build CI job.

## Package Artifacts

- [ ] **Wheel and source distribution build from a clean checkout.**
      Evidence: `uv build --sdist --wheel --out-dir <dist> --clear`
      writes exactly one `.whl` and one `.tar.gz`.
- [ ] **Package metadata renders.** Evidence:
      `uv run twine check <dist>/*` passes for the wheel and sdist.
- [ ] **Built wheel installs in a clean environment.** Evidence:
      `uv venv <venv>` plus `uv pip install --python <venv>/bin/python
      <dist>/codelewm-*.whl` succeeds.
- [ ] **Installed console script works.** Evidence:
      `<venv>/bin/codelewm --help` prints the expected command surface.
- [ ] **Publishing remains manually gated.** Evidence: no CI job uploads to
      TestPyPI/PyPI; a maintainer runs `uv run twine upload` only after this
      checklist, #126, and the public visibility gate are complete.

## Tests

- [ ] **Full test suite passes locally.** Evidence:
      `uv run python -m pytest tests/` output, no failures.
- [ ] **CPU smoke training path runs.** Evidence:
      `uv run python -m pytest tests/integration/test_cpu_train_smoke.py`
      output.
- [ ] **Public CLI contract tests pass.** Evidence:
      `uv run python -m pytest tests/api/test_cli_contract.py` output.

## Manifests

Every artifact published with this release must verify. List every
artifact below and pin its manifest id. The expected schema versions
are listed in the second column so reviewers can spot drift at a
glance.

| Artifact | Expected schema | Manifest path | Artifact ID | Verifies? |
| -------- | --------------- | ------------- | ----------- | --------- |
| Dataset | `codelewm.dataset.v1` | | | |
| Training run | `codelewm.training_run.v1` | | | |
| Checkpoint | `codelewm.checkpoint.v1` | | | |
| Index | `codelewm.transition_index.v1` | | | |
| Retrieval report | `codelewm.eval.retrieval_report.v1` | | | |
| Action ablation report | `codelewm.eval.action_ablation_report.v1` | | | |
| Surprise report | `codelewm.eval.surprise_report.v1` | | | |
| Scorer quality report | `codelewm.harness.scorer_quality_report.v1` | | | |
| License gate | `codelewm.public_license_gate.v1` | | | |

- [ ] **Every manifest above verifies cleanly.** Evidence:
      `codelewm manifest verify --manifest <path>` returns exit 0
      for every row (issue #52 ships the verifier CLI; for the
      manifests this release relies on, the Python helpers
      `codelewm.observability.read_artifact_manifest` plus
      `validate_artifact_checksums` are equivalent).
- [ ] **Parent-artifact lineage is complete.** Evidence: training
      run's `parent_artifacts` includes the dataset artifact id;
      checkpoint manifest references the training run; index
      manifest (if present) references the training run.

## Benchmark Evidence

- [ ] **Benchmark report is filled in from
      `docs/benchmark/REPORT_TEMPLATE.md`.** Evidence: link to the
      filled-in report under `docs/benchmark/<release>-<date>.md`.
- [ ] **Headline retrieval reports include all four required
      baselines.** Evidence: random, lexical, no-action, and
      shuffled-action rows in the benchmark report's required
      baselines table.
- [ ] **Patch-surprise report covers all four decoy categories with
      non-zero decoy counts where possible.** Evidence:
      `pairwise_auc_by_category` covers random, same_file,
      mutation, action_cluster.
- [ ] **Headline retrieval uses `action_text`.** Evidence:
      benchmark report's "Action view" slice or
      `action_view_policy=headline_text_only` in the retrieval
      report.
- [ ] **Action-use claim gate is explicit.** Evidence:
      retrieval or ablation report records text-action versus
      no-action deltas and a machine-readable claim gate. Positive
      action-conditioning claims are allowed only when the gate
      passes; otherwise release notes must frame the artifact as
      negative/diagnostic.
- [ ] **Action-view ablation report accounts for missing runs.**
      Evidence: `codelewm.eval.action_ablation_report.v1` includes
      completed baseline rows and explicit `blocked` rows for
      missing abstract-action, retrieval-loss, collapse-setting, and
      patch-action diagnostic variants.
- [ ] **Scorer/reranker quality report accounts for candidate
      failures and calibration slices.** Evidence:
      `codelewm.harness.scorer_quality_report.v1` includes ranking
      metrics, score distributions, retrieval-prior settings,
      parse/patch failure counts, and the non-execution policy.
- [ ] **Patch-action results are tagged diagnostic.** Evidence:
      `action_view_policy=diagnostic_only` flag on any
      patch-action results.

## Security Evidence

- [ ] **Release dependency audit passes.** Evidence:
      `uv run pip-audit --format json --output <audit>/pip-audit.json`
      exits 0 for the release environment; any non-zero result has a
      signed waiver with advisory ID, affected package/version,
      mitigation, and reviewer sign-off.
- [ ] **Release provenance report is attached.** Evidence:
      `<provenance>/provenance.json` validates as
      `codelewm.release_provenance.v1` and records source SHA,
      tracked dirty-state evidence, `uv.lock`, built wheel/sdist,
      dependency audit report, and release evidence file checksums.
- [ ] **Secret scan over all published artifacts returns clean.**
      Evidence: `codelewm secret-scan <release_dir>` exit 0.
- [ ] **No raw user code or private paths in logs.** Evidence:
      `codelewm.observability.logging.redact_text` and
      `redact_value` are unchanged; spot-check published JSONL
      logs for `[REDACTED_...]` markers where applicable.
- [ ] **Checkpoint trust gate is exercised by the release flow.**
      Evidence: the checkpoint shipped with this release has a
      paired `<checkpoint>.manifest.json`; loading it without
      `--allow-unsafe-checkpoint` succeeds and loading without a
      manifest refuses with `error_type=checkpoint_error`.
- [ ] **No `--allow-unsafe-checkpoint` invocation in release
      automation.** Evidence: grep of release scripts / CI.

## License Evidence

- [ ] **Public license gate report is attached to the dataset
      manifest.** Evidence: the dataset manifest's
      `metadata.license_gate_report` is present and
      `release_allowed=true`.
- [ ] **Public source acquisition report is attached to the dataset
      manifest.** Evidence: the dataset manifest's
      `metadata.source_acquisition_report.schema_version` is
      `codelewm.source_acquisition.v1` and
      `public_path_policy.raw_private_paths_published=false`.
- [ ] **Allowed source mix matches the dataset card.** Evidence:
      `source_acquisition_report.dataset_card_fields.source_mix`
      and `included_sources` from the license gate report match the
      dataset card's "Source Mix" table.
- [ ] **Every non-allowlisted license is accounted for.** Evidence:
      `excluded_licenses` map in the license gate report covers
      every license that appeared in the raw corpus but was
      filtered out.

## Cards And Documentation

- [ ] **Dataset card filled in from
      `docs/cards/DATASET_CARD_TEMPLATE.md`.** Evidence: link to
      `docs/cards/<dataset>-<release>.md`.
- [ ] **Model card filled in from
      `docs/cards/MODEL_CARD_TEMPLATE.md`.** Evidence: link to
      `docs/cards/<model>-<release>.md`.
- [ ] **Public API docs reflect the release.** Evidence:
      `docs/usage/USAGE.md` and `docs/spec/02-public-api.md`
      describe the surface shipping in this tag.
- [ ] **Public docs preserve the evidence boundary.** Evidence:
      README, usage docs, benchmark report, and cards distinguish
      local smoke evidence, scaled systems evidence, and negative
      action-use evidence; positive action-conditioning language is
      absent unless the claim gate passes.
- [ ] **Deprecation notices listed.** Evidence: any public surface
      removed by this release appeared in `CHANGELOG.md` under
      "Deprecated" in the previous minor release; migration notes
      are linked.

## Reproducibility

- [ ] **All training and evaluation commands are documented in the
      benchmark report.** Evidence: "Run reproduction command"
      field on the benchmark report.
- [ ] **Seed and config hash are recorded.** Evidence:
      `codelewm.training_run.v1.config_sha256` and `seed` fields on
      the run manifest.
- [ ] **Random-baseline retrieval was rebuilt with the same data
      contract.** Evidence: timestamp on the retrieval report
      newer than the dataset manifest.

## Communications

- [ ] **Release notes describe scope, breaking changes, and
      deprecations.** Evidence: PR body or release-notes draft.
- [ ] **No marketing copy in technical artifacts.** Evidence: dataset
      card and model card avoid unverifiable claims; every claim
      points at a manifest field or report metric.
- [ ] **Public communications do not include claims that aren't on
      the benchmark report's claim checklist.** Evidence: the
      release announcement only mentions claims that are ticked in
      the benchmark report.

## Sign-off

| Role | GitHub handle | Date |
| ---- | ------------- | ---- |
| Release shepherd | | |
| Codeowner | | |
| Security reviewer | | |

A release without all three sign-offs is blocked.
