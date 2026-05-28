# CodeLeWM v0.6 Execution-Substrate Results — Template

> Copy this file into
> `docs/benchmark/EXECUTION_V0_6_RESULTS_<YYYY-MM-DD>.md` and fill in
> every section after the v0.6 HF Jobs run (#265) completes and the
> headline evaluations (#266-#269) are run against the downloaded
> artifacts. The release gate rejects a report that leaves required
> tables blank or that makes a positive claim without all gates green.
>
> This is the benchmark-report sibling of
> `docs/benchmark/REPORT_TEMPLATE.md` scoped to the execution-substrate
> pivot. The two-substrate paper outline at
> `docs/papers/two_substrate_outline.md` cites this report verbatim.

- Report ID: `codelewm-execution-v0-6-<date>`
- Schema versions referenced by this report:
  - `codelewm.execution_pack_manifest.v1`
  - `codelewm.execution_train_config.v1`
  - `codelewm.execution_launch_plan.v1`
  - `codelewm.eval.retrieval_report.v1`
  - `codelewm.eval.surprise_report.v1`
  - `codelewm.eval.execution_rerank_report.v1`
  - `codelewm.eval.crash_prediction_report.v1`
  - `codelewm.eval.execution_probe_target.v1`
  - `codelewm.checkpoint.v1`
- Substrate claim boundary: `execution_substrate.v1` (record SHA-256
  here from the dataset card)
- Source git SHA: `<40-char SHA>`
- Dataset: `abdelstark/codelewm-execution-pack@v0.6.0` (record
  manifest path + checksum)
- Run reproduction commands: copy the launch plans from
  `scripts/hf-launch-execution-run --json` verbatim.

## Reproducibility Chain

| Artifact | Schema version | Repo | Revision | Local manifest path |
| -------- | -------------- | ---- | -------- | ------------------- |
| Execution pack | `codelewm.execution_pack_manifest.v1` | `abdelstark/codelewm-execution-pack` | `v0.6.0` | |
| Train config | `codelewm.execution_train_config.v1` | local | n/a | `config/train/scaled/codelewm_execution_v0_6_a10g.yaml` |
| Launch plan | `codelewm.execution_launch_plan.v1` | local | n/a | |
| Checkpoint, seed 42 | `codelewm.checkpoint.v1` | `abdelstark/codelewm-transition-model` | `v0.6.0-seed-42` | |
| Checkpoint, seed 1729 | `codelewm.checkpoint.v1` | `abdelstark/codelewm-transition-model` | `v0.6.0-seed-1729` | |
| Run artifacts, seed 42 | `codelewm.training_run.v1` | `abdelstark/codelewm-runs` | run-name | |
| Run artifacts, seed 1729 | `codelewm.training_run.v1` | `abdelstark/codelewm-runs` | run-name | |
| Retrieval report, seed 42 | `codelewm.eval.retrieval_report.v1` | local | | |
| Retrieval report, seed 1729 | `codelewm.eval.retrieval_report.v1` | local | | |
| Surprise report | `codelewm.eval.surprise_report.v1` | local | | |
| Latent probe report | `codelewm.eval.latent_probe_report.v1` | local | | |
| Rerank report (HumanEval) | `codelewm.eval.execution_rerank_report.v1` | local | | |
| Rerank report (MBPP-Plus) | `codelewm.eval.execution_rerank_report.v1` | local | | |
| Crash prediction report | `codelewm.eval.crash_prediction_report.v1` | local | | |

Every manifest above must pass
`codelewm manifest verify --manifest <path> --json` before this report
is accepted. The `codelewm secret-scan <download-dir>` step must
return zero findings.

## Headline Retrieval (Claim Gate)

| Run | Seed | Text-action Recall@1 | No-action Recall@1 | Δ | Text-action MRR | No-action MRR | Δ | Gate status |
|-----|-----:|---------------------:|-------------------:|--:|---------------:|--------------:|--:|:-----------:|
| v0.6 execution | 42   |  |  |  |  |  |  |  |
| v0.6 execution | 1729 |  |  |  |  |  |  |  |

Gate requires:

- text-action Recall@1 - no-action Recall@1 ≥ +0.05 absolute on test
  split for **both** seeds;
- text-action MRR - no-action MRR ≥ +0.05 absolute on test split for
  both seeds.

## Collapse And SIGReg Diagnostics

| Seed | Effective rank | Effective rank ratio | Per-dim variance median | NN entropy | Gate |
|-----:|---------------:|---------------------:|------------------------:|-----------:|:----:|
| 42   |  |  |  |  |  |
| 1729 |  |  |  |  |  |

Gate requires effective rank ratio ≥ 0.20, per-dim variance median ≥
1e-8, NN entropy ≥ 0.10 across both seeds.

## Surprise Evaluation

| Decoy category | AUC, seed 42 | AUC, seed 1729 | Gate threshold | Status |
| -------------- | -----------: | --------------: | --------------: | :----: |
| `random` |  |  | n/a | informational |
| `mutation` |  |  | ≥ 0.65 |  |
| `same_problem_different_submission` |  |  | ≥ 0.60 |  |
| `same_code_different_input` |  |  | ≥ 0.70 |  |

## Latent Probe Matrix

| Target | Best latent / acc | Best control / acc | Beats every control across 2 seeds? |
| ------ | ----------------- | ------------------ | :---------------------------------: |
| `output_type` |  |  |  |
| `will_raise` |  |  |  |
| `output_magnitude_bucket` |  |  |  |
| `output_length_bucket` |  |  |  |
| `arithmetic_vs_string_vs_collection` |  |  |  |
| `judge_verdict` |  |  |  |

Gate requires **at least one** target where the best latent beats
every control across both seeds.

## Downstream Reranking (HumanEval / MBPP-Plus)

| Benchmark | Baseline | Pass@1 | CodeLeWM lift | Bootstrap 95% CI | Claim status |
| --------- | -------- | -----: | ------------: | :--------------: | :----------: |
| HumanEval | `llm_order` |  | n/a | n/a | baseline |
| HumanEval | `codelewm` |  |  |  |  |
| MBPP-Plus | `llm_order` |  | n/a | n/a | baseline |
| MBPP-Plus | `codelewm` |  |  |  |  |

Gate requires lift ≥ 3.0 absolute points on at least one benchmark
with bootstrap 95% CI excluding zero across ≥3 LLM sampling seeds.

## Crash Prediction (Scoped Fallback)

| Method | Accuracy | AUC-ROC | AUC-PR | F1 |
| ------ | -------: | ------: | -----: | -: |
| `linear_code` |  |  |  |  |
| `linear_code_input` |  |  |  |  |
| `linear_predicted_output` |  |  |  |  |
| `lexical` |  |  |  |  |
| `static` |  |  |  |  |
| `random` |  |  |  |  |

Scoped claim requires the best latent method AUC ≥ best non-latent
AUC + 0.05 across both seeds.

## Claim-Gate Summary

| Gate | Status | Backing report |
| ---- | :----: | --------------- |
| Retrieval ≥0.05 over no-action across 2 seeds |  |  |
| Collapse gates satisfied across 2 seeds |  |  |
| ≥1 latent probe target beats every control across 2 seeds |  |  |
| Surprise mutation AUC ≥0.65 |  |  |
| Surprise `same_problem_different_submission` AUC ≥0.60 |  |  |
| Surprise `same_code_different_input` AUC ≥0.70 |  |  |
| Rerank pass@1 lift ≥3pts with CI excluding zero on ≥1 benchmark |  |  |
| Checkpoint trust + manifest verify + secret scan pass |  |  |

If **every** row is green, the headline positive claim is allowed.
If any row is red, public language is restricted to negative or
diagnostic evidence.

## Allowed Public Language (If All Gates Pass)

> The CodeLeWM v0.6 execution-substrate run satisfies the
> retrieval, collapse, latent-probe, surprise, and downstream-rerank
> claim gates documented in
> `docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`.
> Reranking N candidate completions per problem with the v0.6 latent
> transition model lifts pass@1 over the LLM's own sampling order by
> _X_ absolute points (95% CI _A_..._B_) on _benchmark_, across _N_
> LLM sampling seeds and 2 training seeds.

## Allowed Public Language (If Any Gate Fails)

> The CodeLeWM v0.6 execution-substrate run completed the full
> training, evaluation, and artifact publication pipeline. _Failed
> gates_ remain closed; the artifact set is diagnostic evidence
> only. The two-substrate comparison
> (`docs/papers/two_substrate_outline.md`) cites this report as
> negative evidence for _claim shape_.

## Notes And Caveats

- Reference LLM (and version), sampling seeds, hardware, runtime,
  cost: filled in by the operator at report time.
- Any partial-positive shapes (e.g., "crash prediction beats baseline
  but rerank lift below threshold") must be scoped here with the
  exact language from RFC-0014 §Expected Failure Modes.

## Reference

- RFC: `docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`
- Tracker: #259
- Roadmap: `docs/roadmap/EXECUTION_TRACE_WORLD_MODEL.md`
- Operator runbook:
  `docs/operations/V0_6_EXECUTION_RUN_RUNBOOK.md`
- Two-substrate paper outline: `docs/papers/two_substrate_outline.md`
