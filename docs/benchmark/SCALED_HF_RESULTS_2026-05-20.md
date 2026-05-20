# CodeLeWM Scaled HF Results 2026-05-20

- Report ID: `codelewm-scaled-hf-results-2026-05-20`
- Run ID: `codelewm-scaled-20260520-9699b53`
- Job ID: `6a0d43c92dc5b1243da50bba`
- Job URL: `https://huggingface.co/jobs/abdelstark/6a0d43c92dc5b1243da50bba`
- Source git SHA: `9699b5309e43a3278f272663ef60cda23040d92a`
- Evidence tier: scaled HF Jobs artifact evidence, private repositories
- Hardware: HF Jobs `a10g-small`, timeout `24h`
- Dataset card: `docs/cards/codelewm-scaled-dataset-2026-05-20.md`
- Model card: `docs/cards/codelewm-scaled-model-2026-05-20.md`

## Verdict

The scaled HF pipeline completed end to end after PR #148 bounded transition
index embedding batches. Dataset, model, and run evidence artifacts were
published privately to Hugging Face, downloaded with `hf download`, verified
locally, and rerun through retrieval, ablation, surprise, scorer-quality,
score, and rerank checks from the downloaded checkpoint/artifacts.

This is a meaningful scaled systems result, but it is not a positive
action-conditioning quality claim. The text-action model beats random,
shuffled-action, and lexical baselines, but the no-action baseline is stronger
on the headline retrieval metric. Public model-quality claims remain blocked
until a follow-up run beats the required no-action baseline or the benchmark is
reframed with a justified claim boundary.

## Published Private Artifacts

| Surface | Repository | Path |
| --- | --- | --- |
| Dataset pack | `abdelstark/codelewm-public-shard` | `runs/codelewm-scaled-20260520-9699b53/pack` |
| Model checkpoint | `abdelstark/codelewm-transition-model` | `checkpoints/codelewm-scaled-20260520-9699b53` |
| Run evidence | `abdelstark/codelewm-runs` | `runs/codelewm-scaled-20260520-9699b53` |

Downloaded local roots:

- `.artifacts/hf-download/codelewm-scaled-20260520-9699b53/dataset`
- `.artifacts/hf-download/codelewm-scaled-20260520-9699b53/model`
- `.artifacts/hf-download/codelewm-scaled-20260520-9699b53/results`

## Artifact Chain

| Artifact | Path | Artifact ID |
| --- | --- | --- |
| Build | `results/runs/codelewm-scaled-20260520-9699b53/build/manifest.json` | `dataset-5a1d4677b02c75f2` |
| Pack | `results/runs/codelewm-scaled-20260520-9699b53/pack/manifest.json` | `dataset-ef8ad3f4f48dea9e` |
| Dataset repo pack | `dataset/runs/codelewm-scaled-20260520-9699b53/pack/manifest.json` | `dataset-ef8ad3f4f48dea9e` |
| Training run | `model/checkpoints/codelewm-scaled-20260520-9699b53/manifest.json` | `training_run-d9074199c0d58911` |
| Retrieval | `results/runs/codelewm-scaled-20260520-9699b53/retrieval/manifest.json` | `eval_report-448c4fbecb0d693b` |
| Action ablation | `results/runs/codelewm-scaled-20260520-9699b53/ablation/manifest.json` | `eval_report-26708794ed1b855f` |
| Surprise | `results/runs/codelewm-scaled-20260520-9699b53/surprise/manifest.json` | `eval_report-a6b4978ebae9c6e2` |
| Index | `results/runs/codelewm-scaled-20260520-9699b53/index/manifest.json` | `index-f68711abc64e0f52` |
| Scorer quality | `results/runs/codelewm-scaled-20260520-9699b53/scorer_quality/manifest.json` | `score_report-23c368a43df78678` |

All listed manifests verified locally with `uv run codelewm manifest verify`
using their required parents. The downloaded local-check manifests for rerun
retrieval, ablation, surprise, and scorer-quality also verified.

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

This first scaled artifact predates
`codelewm.data.action_discriminative_shard_report.v1`, so it does not contain a
manifested action-discriminative shard report. That absence is now an explicit
data-evidence blocker: the follow-up #154 run must use the #152/#153 code path
to regenerate the public shard build and pack artifacts and publish
`reports/action_discriminative_shard_report.json` before any positive
action-use claim can be considered.

Required follow-up fields:

- `claim_readiness.positive_action_use_claim_ready`
- `hard_negative_pools.same_before_different_after.pair_count`
- `hard_negative_pools.near_before_different_after.pair_count`
- `hard_negative_pools.same_file.pair_count`
- `hard_negative_pools.action_cluster.pair_count`
- `unavailable_hard_negative_pools`

## Training

| Field | Value |
| --- | --- |
| Config | `config/train/scaled/codelewm_scaled_gpu_a10g.yaml` |
| Executor | `torch` |
| Device | `cuda` |
| Precision | `bf16-mixed` |
| Torch | `2.12.0+cu130` |
| Steps | 60,000 |
| Checkpoint SHA-256 | `09bf8d3880ec272a858dd9b19f2b29622a66a5ebbef6dbd1f8e4ebeb8b6392b8` |
| Loss total | 0.113565 |
| Validation loss total | 0.355089 |
| Prediction MSE | 0.005167 |
| Collapse effective rank | 5.437365 |
| Collapse effective-rank ratio | 0.021240 |
| Examples per second | 1103.204 |

## Retrieval

Remote headline retrieval, 1,000 held-out queries with 1,000 candidates:

| Variant | Recall@1 | Recall@5 | Recall@10 | MRR | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text action | 0.371 | 0.586 | 0.672 | 0.472984 | 3 |
| Random | 0.001 | 0.004 | 0.008 | 0.007118 | 502 |
| Shuffled action | 0.001 | 0.006 | 0.011 | 0.007518 | 510 |
| Lexical | 0.045 | 0.130 | 0.190 | 0.093745 | 152 |
| No action | 0.459 | 0.641 | 0.712 | 0.546116 | 2 |

Downloaded local rerun on CPU reproduced the same headline Recall@1 `0.371`,
Recall@5 `0.586`, Recall@10 `0.672`, and median rank `3`; MRR was
`0.473175`.

## Action-Use Claim Gate

The #151 gate evaluates this run as `claim_allowed=false`. The blocker is
specifically no-action dominance: text-action is not strictly above no-action on
Recall@1 or MRR.

| Baseline | Recall@1 delta | Recall@5 delta | Recall@10 delta | MRR delta | Median-rank improvement | Text beats baseline? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random | 0.370 | 0.582 | 0.664 | 0.465866 | 499 | yes |
| Shuffled action | 0.370 | 0.580 | 0.661 | 0.465466 | 507 | yes |
| Lexical | 0.326 | 0.456 | 0.482 | 0.379239 | 149 | yes |
| No action | -0.088 | -0.055 | -0.040 | -0.073132 | -1 | no |

Machine-readable gate fields expected in follow-up retrieval artifacts:

```json
{
  "schema_version": "codelewm.eval.action_use_claim_gate.v1",
  "claim_allowed": false,
  "failure_reasons": [
    "no_action_dominance:text_action_recall_at_1_or_mrr_not_strictly_above_no_action"
  ]
}
```

## Action-View Ablation

| Field | Value |
| --- | --- |
| Completed rows | 7 |
| Blocked rows | 5 |
| Failed rows | 0 |

Blocked rows explicitly cover unavailable abstract-action, patch-action
diagnostic, retrieval-loss-enabled, and alternate SIGReg variants.

## Surprise

| Metric | Value |
| --- | ---: |
| Example count | 1,000 |
| Pairwise AUC overall | 0.755587 |
| Recall@1 | 0.514 |
| Mean true rank | 1.514 |
| Median true rank | 1 |

| Decoy category | Count | Pairwise AUC |
| --- | ---: | ---: |
| Random | 1,000 | 0.975 |
| Mutation | 1,000 | 0.533 |
| Same file | 63 | 0.666667 |
| Action cluster | 40 | 0.975 |

Downloaded local rerun reproduced these surprise values exactly.

## Scorer And Reranker

The published scorer-quality report and downloaded local rerun both record:

| Field | Value |
| --- | ---: |
| Examples | 1 |
| Candidates | 4 |
| Valid candidates | 2 |
| Error candidates | 2 |
| Recall@1 | 1.0 |
| MRR | 1.0 |
| Mean true rank | 1.0 |

Failure counts are `invalid_syntax=1` and `patch_apply_failed=1`. The scorer
and reranker smoke commands also ran from the downloaded checkpoint and index,
with retrieval-prior weight `1.0`, k `10`, and final score `12.753259` for the
fixture candidate.

## Verification Commands

```bash
hf download abdelstark/codelewm-runs --repo-type dataset --local-dir .artifacts/hf-download/codelewm-scaled-20260520-9699b53/results
hf download abdelstark/codelewm-transition-model --repo-type model --include 'checkpoints/codelewm-scaled-20260520-9699b53/**' --local-dir .artifacts/hf-download/codelewm-scaled-20260520-9699b53/model
hf download abdelstark/codelewm-public-shard --repo-type dataset --include 'runs/codelewm-scaled-20260520-9699b53/pack/**' --local-dir .artifacts/hf-download/codelewm-scaled-20260520-9699b53/dataset
uv run codelewm manifest verify --manifest .artifacts/hf-download/codelewm-scaled-20260520-9699b53/model/checkpoints/codelewm-scaled-20260520-9699b53/manifest.json --parent-manifest .artifacts/hf-download/codelewm-scaled-20260520-9699b53/results/runs/codelewm-scaled-20260520-9699b53/pack/manifest.json --json
uv run codelewm eval retrieval --checkpoint .artifacts/hf-download/codelewm-scaled-20260520-9699b53/model/checkpoints/codelewm-scaled-20260520-9699b53/checkpoints/checkpoint.pt --data .artifacts/hf-download/codelewm-scaled-20260520-9699b53/dataset/runs/codelewm-scaled-20260520-9699b53/pack --out .artifacts/hf-download/codelewm-scaled-20260520-9699b53/local-checks/retrieval --device cpu --seed 0 --overwrite --json
uv run codelewm eval surprise --checkpoint .artifacts/hf-download/codelewm-scaled-20260520-9699b53/model/checkpoints/codelewm-scaled-20260520-9699b53/checkpoints/checkpoint.pt --data .artifacts/hf-download/codelewm-scaled-20260520-9699b53/dataset/runs/codelewm-scaled-20260520-9699b53/pack --out .artifacts/hf-download/codelewm-scaled-20260520-9699b53/local-checks/surprise --device cpu --seed 0 --overwrite --json
uv run codelewm secret-scan .artifacts/hf-download/codelewm-scaled-20260520-9699b53/results --json
```

## Claim Checklist

- [x] Scaled HF Jobs run completed on the merged source SHA.
- [x] Private dataset, model, and result artifacts published to HF.
- [x] Published artifacts downloaded with `hf download`.
- [x] Manifest chain verified locally from downloaded artifacts.
- [x] Retrieval, ablation, surprise, scorer-quality, score, and rerank checks ran from downloaded artifacts.
- [x] License and secret-scan gates passed.
- [x] Text action beats random, shuffled-action, and lexical baselines.
- [ ] Text action beats the no-action baseline.
- [ ] This report supports a public positive action-conditioning model-quality claim.

## Release Blocker

Do not make the model or dataset public as a positive CodeLeWM quality claim
from this run alone. The no-action baseline remains stronger than text action
on headline retrieval, so the next training/eval iteration must address
action-use evidence before public release claims are widened.
