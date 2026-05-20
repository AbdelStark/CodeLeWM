# CodeLeWM Release Freeze 2026-05-20

- Issue: #126
- Freeze type: private diagnostic artifact freeze
- Release tag: not cut
- Package version: `0.0.0`
- Selected model/data source git SHA:
  `6650183eda75ecced9f826d5f2875548808c39f0`
- Package/provenance source git SHA: generated from the #126 release branch
  and PR CI rather than pinned in this self-referential report
- Release date: 2026-05-20
- Release status: blocked for public positive action-conditioning claims
- Selected scaled artifact run: `codelewm-action-use-20260520-6650183`
- HF job ID: `6a0d7a763aba298b21d147a9`

## Verdict

The release gates are frozen for a private diagnostic artifact set. The package
build, dependency audit, provenance, manifest lineage, license, secret-scan,
checkpoint-trust, docs, and CI-style local validation gates pass.

The model-quality gate does not pass. Text-action still loses to no-action on
the agreed headline retrieval metrics, so the Hugging Face repositories remain
private and public positive action-conditioning claims remain disabled.

After this freeze, #159 completed the second-stage margin+retrieval remediation
run and remained negative. The remaining project gap, if a positive public claim
is still desired, is a new research iteration beyond the frozen diagnostic
artifact set.

## Post-Freeze #159 Boundary

The #159 run `codelewm-action-use-retrieval-20260520-7895d18` completed on HF
Jobs job `6a0da3a08229e585f969c3f7` from source
`7895d185e165a917af0956a313d8948c04b33638`. It published private artifacts to
the same Hugging Face repositories, downloaded them with `hf download`, and
passed manifest verification, local retrieval, ablation, surprise,
scorer-quality, score, rerank, and secret-scan checks.

The run improved text-action retrieval, but did not pass the claim gate:

| Metric | Text action | No action | Gate |
| --- | ---: | ---: | --- |
| Recall@1 | 0.597 | 0.650 | fail |
| Recall@5 | 0.770 | 0.774 | fail |
| Recall@10 | 0.813 | 0.816 | fail |
| MRR | 0.674500 | 0.708037 | fail |

The action-use claim gate remains `claim_allowed=false` with:

```text
no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action
```

See `docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md`,
`docs/cards/codelewm-action-use-retrieval-dataset-2026-05-20.md`, and
`docs/cards/codelewm-action-use-retrieval-model-2026-05-20.md`.

## Selected Artifacts

| Surface | Repository | Path | Visibility |
| --- | --- | --- | --- |
| Dataset pack | `abdelstark/codelewm-public-shard` | `runs/codelewm-action-use-20260520-6650183/pack` | private |
| Model checkpoint | `abdelstark/codelewm-transition-model` | `checkpoints/codelewm-action-use-20260520-6650183` | private |
| Run evidence | `abdelstark/codelewm-runs` | `runs/codelewm-action-use-20260520-6650183` | private |

Downloaded local root:
`.artifacts/hf-download/codelewm-action-use-20260520-6650183`.

## Package Gate

| Gate | Result | Evidence |
| --- | --- | --- |
| Build wheel and sdist | pass | `uv build --sdist --wheel --out-dir .artifacts/release-freeze-20260520/dist --clear` |
| Metadata render | pass | `uv run twine check .artifacts/release-freeze-20260520/dist/*` |
| Clean wheel install | pass | `uv pip install --python .artifacts/release-freeze-20260520/package-venv/bin/python .artifacts/release-freeze-20260520/dist/codelewm-*.whl` |
| Installed CLI | pass | `.artifacts/release-freeze-20260520/package-venv/bin/codelewm --help` |
| Installed version | pass | `0.0.0` |
| Manual publication gate | pass | no CI workflow uploads to TestPyPI or PyPI |

Built distributions:

| File | SHA-256 | Bytes |
| --- | --- | ---: |
| `.artifacts/release-freeze-20260520/dist/codelewm-0.0.0-py3-none-any.whl` | `d143ed3c225a08d7efaa186c40810dae156cc215295f8e52b89471780a1515b1` | 203620 |
| `.artifacts/release-freeze-20260520/dist/codelewm-0.0.0.tar.gz` | `18c3361f13b74106028860be1259cbd4a713a101b3cb368dc8daefc6ac7e596a` | 182380 |

## Provenance And Dependency Audit

| Gate | Result | Evidence |
| --- | --- | --- |
| HF auth identity | pass | `hf auth whoami` returned user `abdelstark` |
| Dependency audit | pass | `uv run pip-audit --format json --output .artifacts/release-freeze-20260520/dependency-audit/pip-audit.json` |
| Vulnerability count | pass | 0 known vulnerabilities across 47 audited dependencies |
| Provenance schema | pass | `codelewm.release_provenance.v1` |
| Tracked tree state | pass | `tracked_git_dirty=false` |
| Provenance report | pass | `.artifacts/release-freeze-20260520/provenance/provenance.json`; PR CI regenerates this gate on the final branch commit |

The provenance report records `uv.lock`, the built wheel and sdist, the
dependency audit report, release docs, the action-use benchmark report, and the
action-use dataset/model cards.

## Manifest Freeze

All selected manifests verify with `uv run codelewm manifest verify`.

| Artifact | Manifest path | Artifact ID | Result |
| --- | --- | --- | --- |
| Build | `.artifacts/hf-download/codelewm-action-use-20260520-6650183/results/runs/codelewm-action-use-20260520-6650183/build/manifest.json` | `dataset-9750a00ae69ee5e1` | pass |
| Dataset pack | `.artifacts/hf-download/codelewm-action-use-20260520-6650183/dataset/runs/codelewm-action-use-20260520-6650183/pack/manifest.json` | `dataset-67895f8dc3e217c4` | pass |
| Training run | `.artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/manifest.json` | `training_run-ce98fe8768af2143` | pass |
| Retrieval report | `.artifacts/hf-download/codelewm-action-use-20260520-6650183/local-checks/retrieval/manifest.json` | `eval_report-fb416999d92f049a` | pass |
| Action ablation report | `.artifacts/hf-download/codelewm-action-use-20260520-6650183/local-checks/ablation/manifest.json` | `eval_report-db0b1cc04fbdbf54` | pass |
| Surprise report | `.artifacts/hf-download/codelewm-action-use-20260520-6650183/local-checks/surprise/manifest.json` | `eval_report-527127ad385d6326` | pass |
| Transition index | `.artifacts/hf-download/codelewm-action-use-20260520-6650183/results/runs/codelewm-action-use-20260520-6650183/index/manifest.json` | `index-79cfc212a0f6a0fd` | pass |
| Scorer quality report | `.artifacts/hf-download/codelewm-action-use-20260520-6650183/local-checks/scorer_quality/manifest.json` | `score_report-5b95171b5725531f` | pass |

The model-repo training manifest is the authoritative checkpoint-bearing
training artifact for this freeze. The results-repo copy of `train/manifest.json`
still excludes the checkpoint payload and is not selected as a release artifact.

## Benchmark Evidence

| Metric | Text action | No action | Gate |
| --- | ---: | ---: | --- |
| Recall@1 | 0.363 | 0.469 | fail |
| Recall@5 | 0.589 | 0.640 | fail |
| Recall@10 | 0.673 | 0.700 | fail |
| MRR | 0.467875 | 0.549624 | fail |

The action-use claim gate is `claim_allowed=false` with:

```text
no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action
```

Other evidence remains useful but diagnostic:

| Surface | Result |
| --- | --- |
| Text action vs random, shuffled-action, lexical | pass |
| Action-discriminative shard readiness | pass |
| Surprise pairwise AUC | 0.746553 |
| Surprise Recall@1 | 0.495 |
| Scorer quality Recall@1 | 0.0 |
| Scorer quality MRR | 0.5 |
| Score/rerank smoke from downloaded checkpoint | pass, diagnostic only |

## Security And License Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| License gate | pass | `release_allowed=true`, `blocked_rows=0` |
| Secret scan | pass | `uv run codelewm secret-scan .artifacts/hf-download/codelewm-action-use-20260520-6650183 docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md docs/cards/codelewm-action-use-dataset-2026-05-20.md docs/cards/codelewm-action-use-model-2026-05-20.md docs/release --json` |
| Secret findings | pass | `0` |
| Raw local path/token grep | pass | no matches for local user path, HF token variable, or token-like pattern in selected logs/docs |
| Checkpoint trust accepts paired manifest | pass | `codelewm score` from the downloaded checkpoint and index returned `codelewm.score.v1` |
| Checkpoint trust refuses missing manifest | pass | missing-manifest score smoke exited nonzero with `codelewm.error.v1` / `checkpoint_error` |
| Unsafe checkpoint flag in release automation | pass | no `--allow-unsafe-checkpoint` use in `scripts` or `.github`; the only release-doc occurrence is the checklist prohibition |

## Local Validation

```bash
uv sync --group dev --group release
uv build --sdist --wheel --out-dir .artifacts/release-freeze-20260520/dist --clear
uv run twine check .artifacts/release-freeze-20260520/dist/*
uv run pip-audit --format json --output .artifacts/release-freeze-20260520/dependency-audit/pip-audit.json
uv venv .artifacts/release-freeze-20260520/package-venv --python 3.13
uv pip install --python .artifacts/release-freeze-20260520/package-venv/bin/python .artifacts/release-freeze-20260520/dist/codelewm-*.whl
.artifacts/release-freeze-20260520/package-venv/bin/codelewm --help
uv run scripts/release-provenance --dist .artifacts/release-freeze-20260520/dist --audit-report .artifacts/release-freeze-20260520/dependency-audit/pip-audit.json --include docs/release/PACKAGE_PUBLISHING.md --include docs/release/DEPENDENCY_PROVENANCE.md --include docs/release/RELEASE_CHECKLIST.md --include docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md --include docs/cards/codelewm-action-use-dataset-2026-05-20.md --include docs/cards/codelewm-action-use-model-2026-05-20.md --out .artifacts/release-freeze-20260520/provenance/provenance.json --require-clean-tracked-tree --json
uv sync --group dev --group data --group train --group eval
uv run pytest tests/ -q
uv run python -m compileall -q -x 'tests/fixtures/codestate/invalid_(before|after)\.py$' codelewm tests scripts
uv lock --check
uv run codelewm --help
uv run scripts/validate-training-configs
```

Observed test result:

```text
474 passed, 4 skipped, 1 warning, 591 subtests passed
```

The warning is PyTorch's nested-tensor prototype warning in
`tests/eval/test_retrieval_cli.py`.

## Release Notes Draft

CodeLeWM now has a package-native first-results runtime, private HF Jobs
publication and downloaded-artifact verification path, package build gates,
dependency audit, release provenance, dataset/model cards, and a frozen private
diagnostic scaled artifact set.

This is not a public positive model-quality release. The selected action-use
checkpoint and the later #159 margin+retrieval checkpoint are useful evidence
for the system and for the no-action dominance failure mode, but no-action
remains stronger than text-action on Recall@1 and MRR. Public repositories and
public positive claims stay blocked unless a future research iteration produces
a passing action-use claim gate.

## Sign-Off Status

| Role | Status |
| --- | --- |
| Release shepherd | blocked for public tag |
| Codeowner | blocked for public tag |
| Security reviewer | blocked for public tag |

The diagnostic freeze is complete. A public tag, public HF visibility flip, and
positive action-conditioned claim remain blocked by the negative #159 result.
