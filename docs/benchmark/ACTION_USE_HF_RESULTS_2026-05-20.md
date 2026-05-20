# CodeLeWM Action-Use HF Results 2026-05-20

- Report ID: `codelewm-action-use-hf-results-2026-05-20`
- Run ID: `codelewm-action-use-20260520-6650183`
- Job ID: `6a0d7a763aba298b21d147a9`
- Job URL: `https://huggingface.co/jobs/abdelstark/6a0d7a763aba298b21d147a9`
- Source git SHA: `6650183`
- Evidence tier: scaled HF Jobs artifact evidence, private repositories
- Hardware: HF Jobs `a10g-small`, timeout `24h`
- Dataset card: `docs/cards/codelewm-action-use-dataset-2026-05-20.md`
- Model card: `docs/cards/codelewm-action-use-model-2026-05-20.md`

## Verdict

The #154 action-use follow-up run completed on Hugging Face Jobs, published
private dataset/model/results artifacts, downloaded them with `hf download`, and
verified the downloaded artifacts locally.

This is a valid negative action-use result. The #153 no-action margin objective
did not make text-action beat the no-action baseline. Text-action still beats
random, lexical, and shuffled-action baselines, but no-action remains stronger
on the agreed headline metrics. Public positive action-conditioning claims
remain blocked.

## Published Private Artifacts

| Surface | Repository | Path |
| --- | --- | --- |
| Dataset pack | `abdelstark/codelewm-public-shard` | `runs/codelewm-action-use-20260520-6650183/pack` |
| Model checkpoint | `abdelstark/codelewm-transition-model` | `checkpoints/codelewm-action-use-20260520-6650183` |
| Run evidence | `abdelstark/codelewm-runs` | `runs/codelewm-action-use-20260520-6650183` |

Downloaded local root:
`.artifacts/hf-download/codelewm-action-use-20260520-6650183`.

## Artifact Chain

| Artifact | Path | Artifact ID |
| --- | --- | --- |
| Build | `results/runs/codelewm-action-use-20260520-6650183/build/manifest.json` | `dataset-9750a00ae69ee5e1` |
| Pack | `results/runs/codelewm-action-use-20260520-6650183/pack/manifest.json` | `dataset-67895f8dc3e217c4` |
| Dataset repo pack | `dataset/runs/codelewm-action-use-20260520-6650183/pack/manifest.json` | `dataset-67895f8dc3e217c4` |
| Training run | `model/checkpoints/codelewm-action-use-20260520-6650183/manifest.json` | `training_run-ce98fe8768af2143` |
| Retrieval | `results/runs/codelewm-action-use-20260520-6650183/retrieval/manifest.json` | `eval_report-3c6e7bc3ec557ae7` |
| Action ablation | `results/runs/codelewm-action-use-20260520-6650183/ablation/manifest.json` | `eval_report-30df13f5d61cfc81` |
| Surprise | `results/runs/codelewm-action-use-20260520-6650183/surprise/manifest.json` | `eval_report-ed842c3b84f9769d` |
| Index | `results/runs/codelewm-action-use-20260520-6650183/index/manifest.json` | `index-79cfc212a0f6a0fd` |
| Scorer quality | `results/runs/codelewm-action-use-20260520-6650183/scorer_quality/manifest.json` | `score_report-66d00b4f4eb6b65e` |

The model-repo training manifest is the authoritative checkpoint manifest for
verification. The results-repo `train/manifest.json` copy references
`checkpoints/checkpoint.pt`, but the results repository intentionally excludes
the checkpoint payload; verifying that copy fails with a missing checkpoint
file. Keep this as a publisher cleanup item for the final artifact freeze.

## Dataset

| Field | Value |
| --- | --- |
| Source | `bigcode/commitpackft`, `data/python/data.jsonl`, revision `main` |
| Raw rows loaded | 56,025 |
| License-included rows before filters | 23,015 |
| License-excluded rows | 33,010 |
| Packed transitions | 20,645 |
| Train / val / test | 18,019 / 1,291 / 1,335 |
| License gate | `release_allowed=true`, `blocked_rows=0` |

## Action-Discriminative Shard Coverage

| Field | Value |
| --- | ---: |
| `claim_readiness.positive_action_use_claim_ready` | true |
| `action_text_nonempty_ratio` | 1.0 |
| `action_abs_nonempty_ratio` | 0.980673 |
| Held-out rows | 2,626 |
| Same-file or near-before pairs | 2,576 |
| Action-cluster pairs | 15,447 |
| Edit-size bucket pairs | 86,486,992 |
| Unique action signatures | 19,064 |

Available hard-negative pools: `action_cluster`, `diff_shape_controlled`,
`edit_size_controlled`, `same_before_different_after`, and `same_file`.
`near_before_different_after` is unavailable because the scan was truncated.

## Training

| Field | Value |
| --- | --- |
| Config | `config/train/scaled/codelewm_scaled_action_use_margin_gpu_a10g.yaml` |
| Executor | `torch` |
| Device | `cuda` |
| Precision | `bf16-mixed` |
| Torch | `2.12.0+cu130` |
| Steps | 60,000 |
| Checkpoint SHA-256 | `1e361498c722893c9754abcc9c2efa4499a615590572b77c7f0de939e789ac66` |
| Objective | MSE + SIGReg + no-action margin |
| Action-use margin weight | 0.25 |
| Action-use margin | 0.02 |
| Retrieval loss | disabled |
| Loss total | 0.099949 |
| Validation loss total | 0.398456 |
| Prediction MSE | 0.005575 |
| Action-use margin loss | 0.006401 |
| Validation action-use margin loss | 0.055218 |
| Collapse effective rank | 5.913854 |
| Collapse effective-rank ratio | 0.023101 |
| Examples per second | 1088.744 |

## Retrieval

Remote headline retrieval, 1,000 held-out queries with 1,000 candidates:

| Variant | Recall@1 | Recall@5 | Recall@10 | MRR | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text action | 0.363 | 0.589 | 0.673 | 0.467875 | 3 |
| Random | 0.001 | 0.004 | 0.008 | 0.007118 | 502 |
| Shuffled action | 0.001 | 0.004 | 0.010 | 0.007474 | 501.5 |
| Lexical | 0.045 | 0.130 | 0.190 | 0.093745 | 152 |
| No action | 0.469 | 0.640 | 0.700 | 0.549624 | 2 |

Downloaded local CPU rerun reproduced Recall@1 `0.363`, Recall@5 `0.589`,
Recall@10 `0.673`, median rank `3`, and MRR `0.467875` within floating-point
noise.

## Action-Use Claim Gate

The #151 gate evaluates this run as `claim_allowed=false`.

| Baseline | Recall@1 delta | Recall@5 delta | Recall@10 delta | MRR delta | Median-rank improvement | Text beats baseline? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random | 0.362 | 0.585 | 0.665 | 0.460758 | 499 | yes |
| Shuffled action | 0.362 | 0.585 | 0.663 | 0.460401 | 498.5 | yes |
| Lexical | 0.318 | 0.459 | 0.483 | 0.374130 | 149 | yes |
| No action | -0.106 | -0.051 | -0.027 | -0.081749 | -1 | no |

Failure reason:

```text
no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action
```

## Action-View Ablation

| Field | Value |
| --- | ---: |
| Completed rows | 7 |
| Blocked rows | 5 |
| Failed rows | 0 |

Blocked rows explicitly cover unavailable abstract-action, patch-action
diagnostic, retrieval-loss-enabled, and alternate SIGReg variants.

## Surprise

| Metric | Value |
| --- | ---: |
| Example count | 1,000 |
| Pairwise AUC overall | 0.746553 |
| Recall@1 | 0.495 |
| Mean true rank | 1.533 |
| Median true rank | 2 |

| Decoy category | Count | Pairwise AUC |
| --- | ---: | ---: |
| Random | 1,000 | 0.967 |
| Mutation | 1,000 | 0.524 |
| Same file | 63 | 0.650794 |
| Action cluster | 40 | 0.950 |

Downloaded local CPU rerun reproduced these surprise values.

## Scorer And Reranker

The scorer-quality report has one labeled example, four candidates, two valid
candidates, and two expected error candidates.

| Field | Value |
| --- | ---: |
| Examples | 1 |
| Candidates | 4 |
| Valid candidates | 2 |
| Error candidates | 2 |
| Recall@1 | 0.0 |
| MRR | 0.5 |
| Mean true rank | 2.0 |

Failure counts are `invalid_syntax=1` and `patch_apply_failed=1`.
`codelewm score` and `codelewm rerank` ran from the downloaded checkpoint and
transition index with retrieval-prior weight `1.0` and k `10`. The fixture
true-after candidate scored `25.630047`; rerank placed the hard negative before
the true-after candidate, so this remains smoke evidence, not scorer
calibration evidence.

## Verification Commands

```bash
hf auth whoami
CODELEWM_HF_JOBS_DRY_RUN=1 uv run scripts/hf-launch-codelewm-job
hf download bigcode/commitpackft data/python/data.jsonl --repo-type dataset --local-dir .artifacts/hf-sources/commitpackft --dry-run
CODELEWM_HF_JOBS_DRY_RUN=0 CODELEWM_HF_PUBLISH_DRY_RUN=0 CODELEWM_HF_REF=6650183 CODELEWM_HF_RUN_ID=codelewm-action-use-20260520-6650183 uv run scripts/hf-launch-codelewm-job
hf jobs inspect 6a0d7a763aba298b21d147a9
hf jobs logs 6a0d7a763aba298b21d147a9
hf jobs stats 6a0d7a763aba298b21d147a9
hf download abdelstark/codelewm-runs --repo-type dataset --include 'runs/codelewm-action-use-20260520-6650183/**' --local-dir .artifacts/hf-download/codelewm-action-use-20260520-6650183/results
hf download abdelstark/codelewm-transition-model --repo-type model --include 'checkpoints/codelewm-action-use-20260520-6650183/**' --local-dir .artifacts/hf-download/codelewm-action-use-20260520-6650183/model
hf download abdelstark/codelewm-public-shard --repo-type dataset --include 'runs/codelewm-action-use-20260520-6650183/pack/**' --local-dir .artifacts/hf-download/codelewm-action-use-20260520-6650183/dataset
uv run codelewm eval retrieval --checkpoint .artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/checkpoints/checkpoint.pt --data .artifacts/hf-download/codelewm-action-use-20260520-6650183/dataset/runs/codelewm-action-use-20260520-6650183/pack --out .artifacts/hf-download/codelewm-action-use-20260520-6650183/local-checks/retrieval --device cpu --seed 0 --overwrite --json
uv run codelewm eval ablation --retrieval-artifact .artifacts/hf-download/codelewm-action-use-20260520-6650183/local-checks/retrieval/manifest.json --training-artifact .artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/manifest.json --out .artifacts/hf-download/codelewm-action-use-20260520-6650183/local-checks/ablation --overwrite --json
uv run codelewm eval surprise --checkpoint .artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/checkpoints/checkpoint.pt --data .artifacts/hf-download/codelewm-action-use-20260520-6650183/dataset/runs/codelewm-action-use-20260520-6650183/pack --out .artifacts/hf-download/codelewm-action-use-20260520-6650183/local-checks/surprise --device cpu --seed 0 --overwrite --json
uv run codelewm eval scorer-quality --config config/first_results/scorer_quality.json --checkpoint .artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/checkpoints/checkpoint.pt --out .artifacts/hf-download/codelewm-action-use-20260520-6650183/local-checks/scorer_quality --device cpu --index .artifacts/hf-download/codelewm-action-use-20260520-6650183/results/runs/codelewm-action-use-20260520-6650183/index --retrieval-prior-weight 1.0 --retrieval-prior-k 10 --parent-manifest .artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/manifest.json --parent-manifest .artifacts/hf-download/codelewm-action-use-20260520-6650183/results/runs/codelewm-action-use-20260520-6650183/index/manifest.json --overwrite --json
uv run codelewm score --before tests/fixtures/codestate/class_method_before.py --instruction 'rewrite the accumulator update explicitly' --candidate config/first_results/scorer_quality_candidates/true_after.py --checkpoint .artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/checkpoints/checkpoint.pt --device cpu --index .artifacts/hf-download/codelewm-action-use-20260520-6650183/results/runs/codelewm-action-use-20260520-6650183/index --retrieval-prior-weight 1.0 --retrieval-prior-k 10 --json
uv run codelewm rerank --before tests/fixtures/codestate/class_method_before.py --instruction 'rewrite the accumulator update explicitly' --candidates config/first_results/scorer_quality_candidates --checkpoint .artifacts/hf-download/codelewm-action-use-20260520-6650183/model/checkpoints/codelewm-action-use-20260520-6650183/checkpoints/checkpoint.pt --device cpu --index .artifacts/hf-download/codelewm-action-use-20260520-6650183/results/runs/codelewm-action-use-20260520-6650183/index --retrieval-prior-weight 1.0 --retrieval-prior-k 10 --json
uv run codelewm secret-scan .artifacts/hf-download/codelewm-action-use-20260520-6650183 --json
```

## Claim Checklist

- [x] Follow-up scaled HF Jobs action-use run completed on the merged source SHA.
- [x] Private dataset, model, and result artifacts were published to HF.
- [x] Published artifacts were downloaded with `hf download`.
- [x] Retrieval, ablation, surprise, scorer-quality, score, and rerank checks ran from downloaded artifacts.
- [x] License gate, checkpoint trust, manifest verification, and secret scan passed for the authoritative artifacts.
- [x] Action-discriminative shard readiness passed.
- [x] Text action beats random, shuffled-action, and lexical baselines.
- [ ] Text action beats the no-action baseline.
- [ ] This report supports a public positive action-conditioning model-quality claim.

## Next Gap

Issue #159 executed the second-stage margin+retrieval remediation sweep in
`codelewm-action-use-retrieval-20260520-7895d18`. That run improved text-action
retrieval but still lost to no-action, so the current project boundary remains
negative/diagnostic.
