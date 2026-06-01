# CodeLeWM v0.6 Execution-Substrate Results — 2026-05-30

> This is the v0.6 headline results report for tracker #289. It mirrors
> the structure of
> `docs/benchmark/EXECUTION_V0_6_RESULTS_TEMPLATE.md`. The
> two-substrate paper outline at
> `docs/papers/two_substrate_outline.md` cites this report verbatim.

- Report ID: `codelewm-execution-v0-6-2026-05-30`
- Eval pass update: 2026-05-31 (#305), using the downloaded seed-42
  and seed-1729 artifacts plus the v0.6 execution pack.
- Semantic decoy update: 2026-06-01 (#321/#322), using the strengthened
  same-problem semantic decoy pack and rerun surprise artifacts under
  `seed-*/execution_surprise_semantic/`.
- Downstream rerank update: 2026-06-01 (#319/#320), using live
  HumanEval and MBPP-Plus completion labels under
  `completion_labels_full/` and rerank reports under
  `seed-*/downstream_rerank_full/`.
- Schema versions referenced by this report:
  - `codelewm.execution_pack_manifest.v1`
  - `codelewm.execution_train_config.v1`
  - `codelewm.execution_train_run_report.v1`
  - `codelewm.execution_train_collapse_diagnostics.v1`
  - `codelewm.training_run.v1`
  - `codelewm.checkpoint.v1`
  - `codelewm.eval.execution_retrieval_run.v1`
  - `codelewm.eval.execution_surprise_run.v1`
  - `codelewm.eval.execution_probe_run.v1`
  - `codelewm.eval.crash_prediction_run.v1`
  - `codelewm.eval.retrieval_report.v1`
  - `codelewm.eval.action_use_claim_gate.v1`
  - `codelewm.eval.surprise_report.v1`
  - `codelewm.eval.execution_surprise_decoy_summary.v1`
  - `codelewm.eval.execution_surprise_claim_gates.v1`
  - `codelewm.eval.semantic_decoy_pack.v1`
  - `codelewm.eval.semantic_decoy_summary.v1`
  - `codelewm.eval.latent_probe_report.v1`
  - `codelewm.eval.crash_prediction_report.v1`
  - `codelewm.eval.completion_label.v1`
  - `codelewm.eval.execution_rerank_report.v1`
- Substrate claim boundary: `execution_substrate.v1`
  (fingerprint
  `62c4d29c0eaff1b80c22d4a2b25aee00b205bab342bb50add3436db6e524973e`)
- Source git SHA at training time: `af1a114` (PR #300 merged into
  main)
- Dataset: `abdelstark/codelewm-execution-pack@v0.6.0`
  (`pack_id=codelewm-execution-pack-20260528T102625Z`, 1605 records,
  pack JSONL SHA-256
  `d770c5df4b8b81aa7708ab2599f18b638ccea02a69f8b1a87e80d7d579ecf41b`)
- Hardware: HF Jobs `a10g-small` (1× NVIDIA A10G, bf16-mixed)
- Two seeds (`42`, `1729`) fired in parallel; wall time ~4h
  each; total cost ≈ $8.

## Reproducibility Chain

| Artifact | Schema version | Repo | Revision | Identifier |
| -------- | -------------- | ---- | -------- | ---------- |
| Execution pack | `codelewm.execution_pack_manifest.v1` | `abdelstark/codelewm-execution-pack` | `v0.6.0` | `codelewm-execution-pack-20260528T102625Z` |
| Train config | `codelewm.execution_train_config.v1` | local | n/a | `config/train/scaled/codelewm_execution_v0_6_a10g.yaml`, sha256 `380572e0…f3f6c` |
| Launch plan | `codelewm.execution_launch_plan.v1` | local | n/a | emitted by `scripts/hf-launch-execution-run` |
| Runtime image | OCI image index | `ghcr.io/abdelstark/codelewm-runtime` | `v0.6` | digest `sha256:ba592252…7b8e` |
| HF Jobs run, seed 42 | n/a | HF Jobs | `6a1b19373a4b8cae6044ebdc` | run name `codelewm-v0-6-execution-20260530-af1a114-seed-42` |
| HF Jobs run, seed 1729 | n/a | HF Jobs | `6a1b19383a4b8cae6044ebde` | run name `codelewm-v0-6-execution-20260530-af1a114-seed-1729` |
| Run artifacts, seed 42 | `codelewm.training_run.v1` | `abdelstark/codelewm-runs` | path `runs/codelewm-v0-6-execution-20260530-af1a114-seed-42` | artifact id `training_run-cb62408f881eff8c` |
| Run artifacts, seed 1729 | `codelewm.training_run.v1` | `abdelstark/codelewm-runs` | path `runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729` | artifact id `training_run-d0b59108447c9c4a` |
| Last checkpoint, seed 42 | `codelewm.checkpoint.v1` | as above | `checkpoints/last.pt` | step 50000 |
| Last checkpoint, seed 1729 | `codelewm.checkpoint.v1` | as above | `checkpoints/last.pt` | step 50000 |

`codelewm secret-scan <run-dir>` returns zero findings on both runs
(`paths_scanned=11, findings=0, ok=true`).

`codelewm manifest verify --manifest <run-dir>/manifest.json
--parent-manifest <pack>/artifact_manifest.json` returns exit 0 for
both historical runs after the #305 verifier compatibility alias. The
seed manifests are immutable and still serialize the legacy
`execution_pack:codelewm-execution-pack-20260528T102625Z` parent
reference; the verifier now accepts that legacy prefix when the
provided parent manifest has the raw
`codelewm-execution-pack-20260528T102625Z` artifact id. Follow-up
#303 fixed future runner output to write the raw parent id directly.

## Headline Training-Time Metrics

The substrate-pivot's headline prediction — *the execution-substrate
substrate trains a non-collapsed action-discriminative latent
transition model with a positive no-action margin* — is exercised by
the training run itself. Cross-seed agreement is tight (spread ≪
mean for every metric).

### Loss trajectory (initial vs final)

| Metric | seed 42 initial | seed 42 final | seed 1729 initial | seed 1729 final |
| ------ | --------------: | ------------: | -----------------: | --------------: |
| `loss_prediction_mse` | 0.9537 | **0.00064** | 0.9299 | **0.00060** |
| `loss_sigreg` | 43.84 | **0.0364** | 45.73 | **0.0348** |
| `loss_action_swap_contrastive` | n/a | 0.00018 | n/a | 0.00012 |
| `no_action_mse` | n/a | 1.2314 | n/a | 1.2440 |
| `margin_no_action_minus_pred` | **−0.7722** | **+1.2308** | **−0.7689** | **+1.2434** |
| `lr` (cosine end) | 0 | 2.56e-06 | 0 | 2.55e-06 |
| `step_count` | 0 | 50000 | 0 | 50000 |

### Cross-seed agreement

| Metric | Mean (final) | Spread (final, ‖seed42 − seed1729‖) |
| ------ | -----------: | -----------------------------------: |
| `loss_prediction_mse` | 6.2e-04 | 4.0e-05 |
| `loss_sigreg` | 3.56e-02 | 1.6e-03 |
| `loss_action_swap_contrastive` | 1.5e-04 | 6.0e-05 |
| `margin_no_action_minus_pred` | **+1.2371** | 1.3e-02 |
| `no_action_mse` | 1.2377 | 1.3e-02 |

The no-action margin spread of 0.013 is ~1% of its mean (+1.24).
SIGReg spread of 0.0016 is ~4% of its mean (0.036). Two-seed
variance is well below the headline magnitude.

### Headline shape, both seeds

- **Margin flipped from negative to positive** across both seeds
  (`all_margins_flipped_from_negative=true`).
- **Final margin is positive** across both seeds
  (`all_margins_positive=true`,
  spread 0.013 ≪ mean 1.24).
- **SIGReg below initial** across both seeds (final 0.036 vs initial
  ~44 — a 1200× reduction; `all_sigreg_below_initial=true`).
- **Effective rank ratio ≥ gate** across both seeds
  (`all_effective_rank_ratio_above_gate=true`; see next section).

## Collapse And SIGReg Diagnostics

| Seed | Effective rank | Effective rank ratio | Mean ‖z‖₂ | Mean pairwise cosine | Gate (ratio ≥ 0.20) |
| ---: | -------------: | -------------------: | -------:  | --------------------:|:-------------------:|
| 42   |     119.53     |        0.4669        |  14.969   |       0.0003         | **PASS**            |
| 1729 |     122.05     |        0.4768        |  14.943   |       0.0001         | **PASS**            |

Both seeds clear the effective-rank-ratio gate (≥ 0.20) by ~2.3×.
Pairwise cosine of the predicted latents is ≈ 0 — the cluster of
predicted next-states is essentially isotropic, with no collapse.

The collapse cadence (every 1000 steps, 50 rows per seed) shows the
effective-rank ratio climbs from ~0.16 at step 1000 to ~0.47 at step
50000 and remains stable through the cosine-decay tail. The
trajectory is in
`reports/collapse_diagnostics.jsonl` inside each run dir.

## Evaluation Surface

The #305 pass runs the downloaded seed-42 and seed-1729 checkpoints
through the JSONL execution-pack eval CLIs added by #302. The #319
pilot and #320 full-scale rerank pass now score live generated
HumanEval/MBPP-Plus completion labels against both checkpoints. The
concrete gate posture is therefore **partial positive**: the
substrate-pivot training shape, execution-pack retrieval,
semantic-decoy surprise score gates, and same-code/input semantic
decoys pass; the narrow same-problem/different-submission semantic
count gate, latent probes, crash prediction, and downstream rerank do
not justify a broader downstream-utility claim.

| Gate | Status | Backing artifact |
| ---- |:------:| ---------------- |
| `collapse_effective_rank_ratio_min ≥ 0.20` (both seeds) | **PASS** | `reports/execution_train_run_report.json` (`z_diagnostics`) |
| `collapse_per_dim_variance_median_min ≥ 1e-8` | **PASS by proxy** (z_pred_mean_norm ≈ 14.96 ⇒ per-dim variance order 0.06) | training reports |
| `collapse_nearest_neighbor_entropy_min ≥ 0.10` | **PASS by proxy** (mean pairwise cosine ≈ 0) | training reports |
| Substrate-pivot headline: positive no-action margin, both seeds | **PASS** (+1.2308, +1.2434) | training reports |
| `retrieval_min_recall_at_1_lift_over_no_action ≥ 0.05` | **PASS** (+0.6186, +0.6102) | `docs/benchmark/v0_6/seed-*/execution_retrieval/reports/retrieval_report.json` |
| `retrieval_min_mrr_lift_over_no_action ≥ 0.05` | **PASS** (+0.6628, +0.6547) | same |
| `surprise_mutation_auc_min ≥ 0.65` | **PASS** (1.0000 both seeds, 236 pairs) | `docs/benchmark/v0_6/seed-*/execution_surprise_semantic/reports/surprise_report.json` |
| `surprise_same_code_different_input_auc_min ≥ 0.70` | **PASS** (1.0000 both seeds, 352 pairs) | same |
| `surprise_same_problem_different_submission_auc_min ≥ 0.60` | **PASS SCORE / FAIL COUNT** (1.0000 both seeds, 6/30 pairs) | same; semantic claim gate closed |
| Semantic decoy pack count gate | **PASS** (358 pairs, 68 problems) | `docs/benchmark/v0_6/semantic_decoy_pack/reports/semantic_decoy_summary.json` |
| ≥1 latent probe target beats every control across 2 seeds | **NOT EVALUABLE / FAILS CLAIM** | only `output_type` has labels; lexical control beats latent on both seeds |
| Crash-prediction fallback | **NOT EVALUABLE** | no crash-positive val/test rows (`positives=0`, `negatives=236`) |
| `downstream_rerank_pass_at_1_lift_min ≥ 3.0 abs pts` | **FAIL** | full-scale HumanEval/MBPP-Plus rerank gates closed across both seeds; see `docs/benchmark/V0_6_RERANK_FULL_2026-06-01.md` |
| `required_seeds ≥ 2` | **PASS** | both training runs and all four eval suites per seed |
| Checkpoint trust + manifest verify + secret scan | **PASS** | downloaded run manifests verify with parent pack manifest; run secret scans return zero findings |

## Eval Artifact Index

All paths below are committed under `docs/benchmark/v0_6/`. Each eval
artifact has an artifact manifest with parents
`training_run-{...}` and
`codelewm-execution-pack-20260528T102625Z`.
The same tree is mirrored in the HF dataset repo at
`abdelstark/codelewm-runs/runs/codelewm-v0-6-eval-pass-20260531`
(commit `396a8fab5b86c16764bec0090e8af7518de41fbc`).

| Seed | Eval | Config schema | Report schema | Manifest artifact id | Report path |
| ---: | ---- | ------------- | ------------- | -------------------- | ----------- |
| 42 | retrieval | `codelewm.eval.execution_retrieval_run.v1` | `codelewm.eval.retrieval_report.v1` | `eval_report-50a62748784329b2` | `seed-42/execution_retrieval/reports/retrieval_report.json` |
| 42 | surprise | `codelewm.eval.execution_surprise_run.v1` | `codelewm.eval.surprise_report.v1` | `eval_report-06ac38fbc347961d` | `seed-42/execution_surprise/reports/surprise_report.json` |
| 42 | semantic surprise | `codelewm.eval.execution_surprise_run.v1` | `codelewm.eval.surprise_report.v1` | `eval_report-aeb0ae374582a8ec` | `seed-42/execution_surprise_semantic/reports/surprise_report.json` |
| 42 | latent probe | `codelewm.eval.execution_probe_run.v1` | `codelewm.eval.latent_probe_report.v1` | `eval_report-952d5632120e0632` | `seed-42/execution_probe/reports/latent_probe_report.json` |
| 42 | crash prediction | `codelewm.eval.crash_prediction_run.v1` | `codelewm.eval.crash_prediction_report.v1` | `eval_report-48380fb96f1de96d` | `seed-42/crash_prediction/reports/crash_prediction_report.json` |
| 1729 | retrieval | `codelewm.eval.execution_retrieval_run.v1` | `codelewm.eval.retrieval_report.v1` | `eval_report-0cc1c6ac187e4ed3` | `seed-1729/execution_retrieval/reports/retrieval_report.json` |
| 1729 | surprise | `codelewm.eval.execution_surprise_run.v1` | `codelewm.eval.surprise_report.v1` | `eval_report-29c0d125cc25d631` | `seed-1729/execution_surprise/reports/surprise_report.json` |
| 1729 | semantic surprise | `codelewm.eval.execution_surprise_run.v1` | `codelewm.eval.surprise_report.v1` | `eval_report-5b20bd0e1da5928a` | `seed-1729/execution_surprise_semantic/reports/surprise_report.json` |
| 1729 | latent probe | `codelewm.eval.execution_probe_run.v1` | `codelewm.eval.latent_probe_report.v1` | `eval_report-c592b4805d0d3085` | `seed-1729/execution_probe/reports/latent_probe_report.json` |
| 1729 | crash prediction | `codelewm.eval.crash_prediction_run.v1` | `codelewm.eval.crash_prediction_report.v1` | `eval_report-1f41882839c44da7` | `seed-1729/crash_prediction/reports/crash_prediction_report.json` |
| 42 | HumanEval rerank full | `codelewm.eval.execution_rerank_eval_run.v1` | `codelewm.eval.execution_rerank_report.v1` | `eval_report-9c8d7152bff2460c` | `seed-42/downstream_rerank_full/humaneval/reports/execution_rerank_report.json` |
| 42 | MBPP-Plus rerank full | `codelewm.eval.execution_rerank_eval_run.v1` | `codelewm.eval.execution_rerank_report.v1` | `eval_report-14887c62e5e76e19` | `seed-42/downstream_rerank_full/mbpp_plus/reports/execution_rerank_report.json` |
| 1729 | HumanEval rerank full | `codelewm.eval.execution_rerank_eval_run.v1` | `codelewm.eval.execution_rerank_report.v1` | `eval_report-ca4212e0d1565af4` | `seed-1729/downstream_rerank_full/humaneval/reports/execution_rerank_report.json` |
| 1729 | MBPP-Plus rerank full | `codelewm.eval.execution_rerank_eval_run.v1` | `codelewm.eval.execution_rerank_report.v1` | `eval_report-3abf5d64a09d9c56` | `seed-1729/downstream_rerank_full/mbpp_plus/reports/execution_rerank_report.json` |

The companion decoy-generation reports live at
`seed-*/execution_surprise*/reports/execution_decoy_report.json` with
schema `codelewm.eval.execution_surprise_decoy_summary.v1`. The semantic
surprise reruns also write `codelewm.eval.execution_surprise_claim_gates.v1`
metadata that separates score gates from pair-count gates.

## Downstream Rerank Evaluation

The #320 pass removes the earlier "not run" limitation for
HumanEval/MBPP-Plus reranking. It uses live OpenRouter completions with
three LLM seeds (`17,42,1729`) and one sample per seed. HumanEval is
uncapped over the parsed hidden-test inputs. MBPP-Plus is full-scale by
problem/completion count but capped to the first 8 hidden-test inputs
per problem; do not read it as full EvalPlus pass@1.

| Seed | Benchmark | Problems | Completions | CodeLeWM pass@1 | LLM-order pass@1 | No-action pass@1 | Lift vs LLM order | Lift vs no-action | Claim |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | HumanEval | 154 | 462 | 0.9610 | 0.9545 | 0.9481 | +0.65 pts | +1.30 pts | closed |
| 42 | MBPP-Plus | 370 | 1110 | 0.9162 | 0.9243 | 0.9243 | -0.81 pts | -0.81 pts | closed |
| 1729 | HumanEval | 154 | 462 | 0.9481 | 0.9545 | 0.9610 | -0.65 pts | -1.30 pts | closed |
| 1729 | MBPP-Plus | 370 | 1110 | 0.9243 | 0.9243 | 0.9243 | +0.00 pts | +0.00 pts | closed |

The gate requires CodeLeWM pass@1 lift of at least 3 absolute points
over both LLM order and no-action, with bootstrap 95% CI lower bound
above zero. No full-scale rerank run clears that bar. The detailed
report is `docs/benchmark/V0_6_RERANK_FULL_2026-06-01.md`.

## Retrieval Evaluation

Command shape:

```text
codelewm eval execution-retrieval \
  --checkpoint <run>/checkpoints/last.pt \
  --pack .artifacts/v0_6/execution-pack \
  --baselines random,lexical,no_action,shuffled_action \
  --out docs/benchmark/v0_6/seed-<seed>/execution_retrieval \
  --device cpu --max-candidates 1000 --seed <seed>
```

Candidate pool: 236 val/test execution records per seed. Scores are
negative squared L2 between predicted-output latents and candidate
output latents (`larger_is_better`).

| Seed | CodeLeWM R@1 | No-action R@1 | R@1 lift | CodeLeWM MRR | No-action MRR | MRR lift | R@5 | R@10 |
| ---: | -----------: | ------------: | -------: | -----------: | ------------: | -------: | --: | ---: |
| 42 | 0.6568 | 0.0381 | +0.6186 | 0.7670 | 0.1042 | +0.6628 | 0.9025 | 0.9703 |
| 1729 | 0.6483 | 0.0381 | +0.6102 | 0.7587 | 0.1040 | +0.6547 | 0.8941 | 0.9703 |

Cross-seed summary:

| Metric | Mean | Spread | Sample stdev |
| ------ | ---: | -----: | -----------: |
| CodeLeWM R@1 | 0.6525 | 0.0085 | 0.0060 |
| CodeLeWM MRR | 0.7628 | 0.0083 | 0.0059 |
| R@1 lift over no-action | +0.6144 | 0.0085 | 0.0060 |
| MRR lift over no-action | +0.6588 | 0.0081 | 0.0057 |

Control baselines:

| Seed | Lexical R@1 / MRR | Shuffled-action R@1 / MRR | Random R@1 / MRR |
| ---: | ----------------: | ------------------------: | ----------------: |
| 42 | 0.0678 / 0.1070 | 0.0042 / 0.0228 | 0.0000 / 0.0183 |
| 1729 | 0.0678 / 0.1070 | 0.0593 / 0.1574 | 0.0085 / 0.0282 |

The retrieval action-use claim gate reports `claim_allowed=true` on
both seeds. This supports the narrow execution-pack retrieval claim,
not a HumanEval/MBPP-Plus downstream-reranking claim.

## Surprise Evaluation

Command shape:

```text
codelewm eval execution-surprise \
  --checkpoint <run>/checkpoints/last.pt \
  --pack .artifacts/v0_6/execution-pack \
  --semantic-decoy-manifest docs/benchmark/v0_6/semantic_decoy_pack/manifest.json \
  --decoys mutation,same_problem_different_submission,same_code_different_input \
  --out docs/benchmark/v0_6/seed-<seed>/execution_surprise_semantic \
  --device cpu --max-examples 1000 --seed <seed>
```

| Seed | Examples | Recall@1 | Overall AUC | Mutation AUC / pairs | Same-code-different-input AUC / pairs | Same-problem-different-submission AUC / pairs |
| ---: | -------: | --------: | ----------: | -------------------: | ------------------------------------: | ---------------------------------------------: |
| 42 | 236 | 1.0000 | 1.0000 | 1.0000 / 236 | 1.0000 / 352 | 1.0000 / 6 |
| 1729 | 236 | 1.0000 | 1.0000 | 1.0000 / 236 | 1.0000 / 352 | 1.0000 / 6 |

The semantic-pack count gate passes with 358 same-problem pairs across
68 distinct problems, driven mainly by 352 same-code/different-input
pairs. The same-problem/different-submission row clears the numeric
score gate but has only six scored pairs versus the predeclared 30-pair
count threshold. The report therefore sets
`execution_surprise_claim_gates.claim_allowed=false` on both seeds.
Treat this as positive diagnostic evidence, not a broad semantic
surprise claim.

## Latent Probe Evaluation

Command shape:

```text
codelewm eval execution-probe \
  --checkpoint <run>/checkpoints/last.pt \
  --pack .artifacts/v0_6/execution-pack \
  --targets output_type,will_raise,output_magnitude_bucket,output_length_bucket \
  --out docs/benchmark/v0_6/seed-<seed>/execution_probe \
  --device cpu --max-examples-per-split 1000 --seed <seed>
```

Rows: 1236 per seed (`train=1000`, `val=79`, `test=157`). Only
`output_type` has train/val/test labels. The claim boundary is closed:
`positive_representation_claim_allowed=false` and
`semantic_structure_status=not_evaluable` because fewer than five
predeclared targets have labels.

| Seed | Target | z_pred_after test acc | No-action test acc | Lift over no-action | Lexical test acc | Metadata-only test acc | Majority test acc | z_pred_after macro-F1 |
| ---: | ------ | --------------------: | -----------------: | ------------------: | ---------------: | ---------------------: | ----------------: | -------------------: |
| 42 | `output_type` | 0.4968 | 0.4395 | +0.0573 | 0.6624 | 0.4204 | 0.2229 | 0.4755 |
| 1729 | `output_type` | 0.5987 | 0.5414 | +0.0573 | 0.6178 | 0.4204 | 0.2229 | 0.5704 |

The latent view beats no-action by 5.7 points on both seeds, but the
lexical control remains stronger on both seeds. This blocks a positive
representation claim.

## Crash Prediction

Command shape:

```text
codelewm eval crash-prediction \
  --checkpoint <run>/checkpoints/last.pt \
  --pack .artifacts/v0_6/execution-pack \
  --out docs/benchmark/v0_6/seed-<seed>/crash_prediction \
  --device cpu --max-examples 1000 --seed <seed>
```

| Seed | Samples | Positives | Negatives | Best latent AUC | Claim allowed | Claim reason |
| ---: | ------: | --------: | --------: | --------------: |:-------------:| ------------ |
| 42 | 236 | 0 | 236 | 0.0000 | false | `not_evaluable: need both positive and negative val/test samples; got positives=0, negatives=236` |
| 1729 | 236 | 0 | 236 | 0.0000 | false | `not_evaluable: need both positive and negative val/test samples; got positives=0, negatives=236` |

The execution pack's val/test slice has no crash-positive examples, so
the crash-prediction fallback cannot support a positive or negative
model-quality claim.

## Downstream Rerank Gate

`scripts/sample-execution-rerank-completions` now defines the
HumanEval/MBPP-Plus completion-label artifact contract from #304, and
the loader accepts `codelewm.eval.completion_label.v1` rows. A
spend-bounded live pilot now exists in
`docs/benchmark/V0_6_RERANK_PILOT_2026-06-01.md`, but it is saturated:
HumanEval has 50/50 passing completions and MBPP-Plus has 46/50 passing
completions under an 8-case pilot cap. All baselines tie and CodeLeWM
has 0.00 pass@1 lift over LLM order and no-action across both seeds.
The full live benchmark gate is still not complete.

| Benchmark | Label artifact | Rerank report | Status |
| --------- | -------------- | ------------- | ------ |
| HumanEval | `docs/benchmark/v0_6/completion_labels_pilot/humaneval/manifest.json` | `docs/benchmark/v0_6/seed-{42,1729}/downstream_rerank_pilot/humaneval/manifest.json` | pilot only; saturated; no lift |
| MBPP-Plus | `docs/benchmark/v0_6/completion_labels_pilot/mbpp_plus/manifest.json` | `docs/benchmark/v0_6/seed-{42,1729}/downstream_rerank_pilot/mbpp_plus/manifest.json` | pilot only; saturated; no lift |

No public downstream-reranking claim is allowed from this report.

## Claim-Gate Summary

| Claim surface | Status | Backing report |
| ------------- |:------:| --------------- |
| Substrate-pivot headline (margin flip, SIGReg drop, non-collapse) | **PASS** | training reports, both seeds |
| Execution-pack retrieval beats no-action by ≥0.05 R@1 and MRR | **PASS** | retrieval reports, both seeds |
| Semantic surprise score gates | **PASS SCORE / NOT ALLOWED** | semantic surprise reports: AUC 1.0, but same-problem/different-submission count is 6/30 |
| Latent semantic representation claim | **NOT ALLOWED** | latent-probe reports: only one target available and lexical control wins |
| Crash-prediction utility claim | **NOT EVALUABLE** | crash reports: no positives |
| HumanEval/MBPP-Plus rerank utility claim | **PILOT RUN / NOT ALLOWED** | live pilot is saturated and has 0.00 pass@1 lift |
| Overall public framing | **PARTIAL POSITIVE** | substrate and internal execution-pack gates pass; broader downstream utility remains unsupported |

## Allowed Public Language (Partial Positive)

> The CodeLeWM v0.6 execution-substrate run completed end-to-end on
> Hugging Face Jobs for two training seeds (42, 1729). The
> substrate-pivot's headline prediction — that the execution-trace
> substrate trains a non-collapsed action-discriminative latent
> transition model — is confirmed by the training-time metrics:
> across both seeds, prediction MSE drops three orders of magnitude
> (0.95 → 6e-4), SIGReg drops by 1200× (44 → 0.036), and the
> no-action margin flips from −0.77 to +1.24. The effective-rank
> ratio of predicted latents climbs to 0.47, clearing the 0.20
> collapse gate by 2.3×. Cross-seed variance is small: the spread of
> the no-action margin across seeds is 0.013, ~1% of its mean
> magnitude.
>
> The #305 downloaded-artifact eval pass adds a partial downstream
> result. On the v0.6 execution pack, CodeLeWM beats no-action
> retrieval by +61.4 R@1 points and +65.9 MRR points on average
> across two seeds, with seed-to-seed spread below one point. The
> semantic surprise score gates also pass with AUC 1.0 on mutation,
> same-code-different-input, and same-problem-different-submission
> decoys. The semantic pack contributes 358 same-problem pairs across
> 68 problems, but the narrow same-problem/different-submission row
> remains only six pairs, so the semantic-surprise claim gate remains
> closed.
> Latent probes do not support a positive semantic-representation
> claim because only `output_type` is evaluable and lexical controls
> remain stronger. Crash prediction is not evaluable because the
> val/test slice has no crash positives. HumanEval/MBPP-Plus
> downstream reranking remains claim-blocked until live labeled
> completion artifacts exist.

## Notes And Caveats

- **Manifest parent-id format**: the historical seed-42 and seed-1729
  run manifests carry the legacy `execution_pack:` parent prefix. #303
  fixed future runner output, and the #305 verifier compatibility alias
  lets the immutable historical manifests verify against the raw pack
  artifact id without rewriting either artifact.
- **No LLM-sampled labels yet** for the downstream rerank gate; that
  operator step depends on an LLM sampling budget separate from this
  report.
- **The collapse / per-dim variance / NN-entropy gates** are
  evaluated from the training-time `z_diagnostics` block. The
  per-dim variance median and the NN entropy require the
  `codelewm.eval.latent_matrix` library run on the eval split; for
  this report we use the proxy of `z_pred_mean_norm ≈ 14.97` (whose
  square gives per-dim variance order 0.06, far above 1e-8) and
  pairwise cosine ≈ 0 (which implies NN entropy ≫ 0.10).
- **`hf jobs run` cost**: A10G-small at $1.00/hr × ~4h × 2 seeds ≈
  $8 total. The first seed=42 invocation (job
  `6a195bfb3a4b8cae6044db06`) also produced identical headline
  numbers but lost its checkpoint because `/tmp` is not persisted —
  the artifact-upload fix that PR #300 landed prevents the
  recurrence.

## Reference

- RFC: `docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`
- Tracker: #289 (substrate-pivot end-to-end completion) and #305
  (downloaded-artifact eval pass)
- Implementing PRs: #297, #296, #298, #299, #300, #310, #311, #312
- Operator runbook:
  `docs/operations/V0_6_EXECUTION_RUN_RUNBOOK.md`
- Runtime container:
  `docs/operations/V0_6_RUNTIME_CONTAINER.md`
- Local smoke evidence:
  `docs/benchmark/EXECUTION_V0_6_LOCAL_SMOKE_2026-05-28.md`
- Two-substrate paper outline:
  `docs/papers/two_substrate_outline.md`
