# CodeLeWM Action-Use Retrieval HF Results 2026-05-20

- Report ID: `codelewm-action-use-retrieval-hf-results-2026-05-20`
- Run ID: `codelewm-action-use-retrieval-20260520-7895d18`
- Job ID: `6a0da3a08229e585f969c3f7`
- Job URL: `https://huggingface.co/jobs/abdelstark/6a0da3a08229e585f969c3f7`
- Source git SHA: `7895d185e165a917af0956a313d8948c04b33638`
- Evidence tier: scaled HF Jobs artifact evidence, private repositories
- Hardware: HF Jobs `a10g-small`, timeout `24h`
- Dataset card: `docs/cards/codelewm-action-use-retrieval-dataset-2026-05-20.md`
- Model card: `docs/cards/codelewm-action-use-retrieval-model-2026-05-20.md`

## Verdict

The #159 margin+retrieval remediation run completed on Hugging Face Jobs,
published private dataset/model/results artifacts, downloaded them with
`hf download`, and verified the downloaded artifacts locally.

This is a valid negative/diagnostic action-use result. Adding the retrieval loss
improved text-action retrieval substantially over #154, but it still did not
beat the no-action baseline on the agreed headline metrics. Public positive
action-conditioning claims remain blocked.

## Published Private Artifacts

| Surface | Repository | Path |
| --- | --- | --- |
| Dataset pack | `abdelstark/codelewm-public-shard` | `runs/codelewm-action-use-retrieval-20260520-7895d18/pack` |
| Model checkpoint | `abdelstark/codelewm-transition-model` | `checkpoints/codelewm-action-use-retrieval-20260520-7895d18` |
| Run evidence | `abdelstark/codelewm-runs` | `runs/codelewm-action-use-retrieval-20260520-7895d18` |

Downloaded local root:
`.artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18`.

## Artifact Chain

| Artifact | Path | Artifact ID |
| --- | --- | --- |
| Build | `results/runs/codelewm-action-use-retrieval-20260520-7895d18/build/manifest.json` | `dataset-1dff4ef2c6b1ee5e` |
| Pack | `dataset/runs/codelewm-action-use-retrieval-20260520-7895d18/pack/manifest.json` | `dataset-5695087296ce4a97` |
| Training run | `model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/manifest.json` | `training_run-924cd056375f11ea` |
| Retrieval | `results/runs/codelewm-action-use-retrieval-20260520-7895d18/retrieval/manifest.json` | `eval_report-59f0da4f4d34c7a8` |
| Action ablation | `results/runs/codelewm-action-use-retrieval-20260520-7895d18/ablation/manifest.json` | `eval_report-6e6e169931009062` |
| Surprise | `results/runs/codelewm-action-use-retrieval-20260520-7895d18/surprise/manifest.json` | `eval_report-41e7a04cb2df600b` |
| Index | `results/runs/codelewm-action-use-retrieval-20260520-7895d18/index/manifest.json` | `index-533e8220768f129a` |
| Scorer quality | `results/runs/codelewm-action-use-retrieval-20260520-7895d18/scorer_quality/manifest.json` | `score_report-54542feb7a29aec1` |

Local CPU verification regenerated retrieval, ablation, surprise, and
scorer-quality manifests under
`.artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/local-checks`.

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

Available hard-negative pools: `action_cluster`, `diff_shape_controlled`,
`edit_size_controlled`, `same_before_different_after`, and `same_file`.
`near_before_different_after` remains unavailable because the scan was
truncated.

## Training

| Field | Value |
| --- | --- |
| Config | `config/train/scaled/codelewm_scaled_action_use_margin_retrieval_gpu_a10g.yaml` |
| Executor | `torch` |
| Device | `cuda` |
| Precision | `bf16-mixed` |
| Torch | `2.12.0+cu130` |
| Seed | 240119 |
| Steps | 60,000 |
| Checkpoint SHA-256 | `0cb4daf1500495579f5c59cc9fd8aa39f5f70e88f55c0c121320d023b43ddeda` |
| Objective | MSE + SIGReg + no-action margin + retrieval loss |
| Action-use margin weight | 0.25 |
| Action-use margin | 0.02 |
| Retrieval loss weight | 0.05 |
| Retrieval temperature | 0.1 |
| Loss total | 0.100775 |
| Validation loss total | 0.412853 |
| Prediction MSE | 0.006972 |
| Action-use margin loss | 0.005475 |
| Retrieval loss | 0.208046 |
| Validation retrieval loss | 0.915677 |
| Collapse effective rank | 10.542142 |
| Collapse effective-rank ratio | 0.041180 |
| Examples per second | 1064.506 |

## Retrieval

Remote headline retrieval, 1,000 held-out queries with 1,000 candidates:

| Variant | Recall@1 | Recall@5 | Recall@10 | MRR | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text action | 0.597 | 0.770 | 0.813 | 0.674500 | 1 |
| Random | 0.001 | 0.004 | 0.008 | 0.007118 | 502 |
| Shuffled action | 0.000 | 0.003 | 0.010 | 0.006567 | 506.5 |
| Lexical | 0.045 | 0.130 | 0.190 | 0.093745 | 152 |
| No action | 0.650 | 0.774 | 0.816 | 0.708037 | 1 |

Downloaded local CPU rerun reproduced the claim decision with Recall@1 `0.597`,
Recall@5 `0.769`, Recall@10 `0.813`, median rank `1`, and MRR `0.674472`.

## Action-Use Claim Gate

The #151 gate evaluates this run as `claim_allowed=false`.

| Baseline | Recall@1 delta | Recall@5 delta | Recall@10 delta | MRR delta | Median-rank improvement | Text beats baseline? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random | 0.596 | 0.766 | 0.805 | 0.667382 | 501 | yes |
| Shuffled action | 0.597 | 0.767 | 0.803 | 0.667934 | 505.5 | yes |
| Lexical | 0.552 | 0.640 | 0.623 | 0.580755 | 151 | yes |
| No action | -0.053 | -0.004 | -0.003 | -0.033537 | 0 | no |

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
diagnostic, retrieval-loss-disabled, and alternate SIGReg variants.

## Surprise

| Metric | Remote | Local CPU rerun |
| --- | ---: | ---: |
| Example count | 1,000 | 1,000 |
| Pairwise AUC overall | 0.757965 | 0.758440 |
| Recall@1 | 0.511 | 0.512 |
| Mean true rank | 1.509 | 1.508 |
| Median true rank | 1 | 1 |

| Decoy category | Count | Remote pairwise AUC |
| --- | ---: | ---: |
| Random | 1,000 | 0.993 |
| Mutation | 1,000 | 0.526 |
| Same file | 63 | 0.571429 |
| Action cluster | 40 | 0.975 |

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
transition index with retrieval-prior weight `1.0` and k `10`. The public CLI
still reports the deterministic lightweight scorer backend plus retrieval prior;
it is an inference and harness smoke check, not a calibrated neural scorer.

The fixture true-after candidate scored `56.275929`; rerank placed the hard
negative before the true-after candidate.

## Verification Commands

```bash
hf auth whoami
hf jobs ps
hf jobs inspect 6a0da3a08229e585f969c3f7
hf jobs logs 6a0da3a08229e585f969c3f7
hf jobs stats 6a0da3a08229e585f969c3f7
hf download abdelstark/codelewm-runs --repo-type dataset --include 'runs/codelewm-action-use-retrieval-20260520-7895d18/**' --local-dir .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/results
hf download abdelstark/codelewm-transition-model --repo-type model --include 'checkpoints/codelewm-action-use-retrieval-20260520-7895d18/**' --local-dir .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model
hf download abdelstark/codelewm-public-shard --repo-type dataset --include 'runs/codelewm-action-use-retrieval-20260520-7895d18/pack/**' --local-dir .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/dataset
uv run codelewm manifest verify --manifest .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/manifest.json --parent-manifest .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/dataset/runs/codelewm-action-use-retrieval-20260520-7895d18/pack/manifest.json --json
uv run codelewm eval retrieval --checkpoint .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/checkpoints/checkpoint.pt --data .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/dataset/runs/codelewm-action-use-retrieval-20260520-7895d18/pack --out .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/local-checks/retrieval --device cpu --seed 0 --overwrite --json
uv run codelewm eval ablation --retrieval-artifact .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/local-checks/retrieval/manifest.json --training-artifact .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/manifest.json --out .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/local-checks/ablation --overwrite --json
uv run codelewm eval surprise --checkpoint .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/checkpoints/checkpoint.pt --data .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/dataset/runs/codelewm-action-use-retrieval-20260520-7895d18/pack --out .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/local-checks/surprise --device cpu --seed 0 --overwrite --json
uv run codelewm eval scorer-quality --config config/first_results/scorer_quality.json --checkpoint .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/checkpoints/checkpoint.pt --out .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/local-checks/scorer_quality --device cpu --index .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/results/runs/codelewm-action-use-retrieval-20260520-7895d18/index --retrieval-prior-weight 1.0 --retrieval-prior-k 10 --parent-manifest .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/manifest.json --parent-manifest .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/results/runs/codelewm-action-use-retrieval-20260520-7895d18/index/manifest.json --overwrite --json
uv run codelewm score --before tests/fixtures/codestate/class_method_before.py --instruction 'rewrite the accumulator update explicitly' --candidate config/first_results/scorer_quality_candidates/true_after.py --checkpoint .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/checkpoints/checkpoint.pt --device cpu --index .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/results/runs/codelewm-action-use-retrieval-20260520-7895d18/index --retrieval-prior-weight 1.0 --retrieval-prior-k 10 --json
uv run codelewm rerank --before tests/fixtures/codestate/class_method_before.py --instruction 'rewrite the accumulator update explicitly' --candidates config/first_results/scorer_quality_candidates --checkpoint .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/model/checkpoints/codelewm-action-use-retrieval-20260520-7895d18/checkpoints/checkpoint.pt --device cpu --index .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18/results/runs/codelewm-action-use-retrieval-20260520-7895d18/index --retrieval-prior-weight 1.0 --retrieval-prior-k 10 --json
uv run codelewm secret-scan .artifacts/hf-download/codelewm-action-use-retrieval-20260520-7895d18 --json
```

`hf jobs stats` was attempted as part of the HF CLI lifecycle, but the local HF
CLI stats stream repeatedly hung and had to be bounded with `timeout`; job
status, logs, download, and artifact verification supplied the usable evidence.

## Claim Checklist

- [x] Second-stage scaled HF Jobs action-use run completed on the recorded source SHA.
- [x] Private dataset, model, and result artifacts were published to HF.
- [x] Published artifacts were downloaded with `hf download`.
- [x] Retrieval, ablation, surprise, scorer-quality, score, and rerank checks ran from downloaded artifacts.
- [x] License gate, checkpoint trust, manifest verification, and secret scan passed for the authoritative artifacts.
- [x] Action-discriminative shard readiness passed.
- [x] Text action beats random, shuffled-action, and lexical baselines.
- [ ] Text action beats the no-action baseline.
- [ ] This report supports a public positive action-conditioning model-quality claim.

## Completion Boundary

Issue #159 closes as a completed negative/diagnostic remediation run. The
meaningful first scaled training and evaluation path is complete, but the
positive action-conditioned model-quality claim is not supported. Further work
should be scoped as a new research iteration, not as release cleanup.
