# CodeLeWM v0.6 Live Rerank Pilot 2026-06-01

This is the #319 spend-bounded HumanEval / MBPP-Plus live rerank pilot. It is
not full downstream generated-code utility evidence and does not open a public
coding-agent usefulness claim.

## Protocol

- Sources were materialized locally with
  `uv run scripts/dataset/materialize-rerank-raw-sources --out-dir data/raw --overwrite --json`.
- Public upstream sources: `openai/openai_humaneval` test split and
  `evalplus/mbppplus` test split.
- Raw row counts: 164 HumanEval rows and 378 MBPP-Plus rows.
- Strict adapter parse counts: 154 HumanEval submissions with 1059 cases, and
  370 MBPP-Plus submissions with 39704 cases.
- Live sampler: `openrouter:anthropic/claude-haiku-4-5`, seeds `17,42`, one
  completion per seed, 25 problems per benchmark, 50 completions per benchmark.
- Pilot label budget: `--max-cases-per-problem 8` and
  `--short-circuit-failures`.
- HumanEval used the sandbox determinism rerun. MBPP-Plus disabled the
  determinism rerun to avoid doubling EvalPlus sandbox cost.
- Rerank scoring used both v0.6 learned checkpoint seeds with
  `--require-learned-scorer`.

## Artifacts

| Artifact | Path | Artifact id | Notes |
| --- | --- | --- | --- |
| HumanEval live labels | `docs/benchmark/v0_6/completion_labels_pilot/humaneval/manifest.json` | `downstream_benchmark-7d549a6ec13ab791` | 25 problems, 50 completions, 50 passed |
| MBPP-Plus live labels | `docs/benchmark/v0_6/completion_labels_pilot/mbpp_plus/manifest.json` | `downstream_benchmark-db47a1daa39e2d3e` | 25 problems, 50 completions, 46 passed |
| Seed 42 HumanEval rerank | `docs/benchmark/v0_6/seed-42/downstream_rerank_pilot/humaneval/manifest.json` | `eval_report-7222346bfae136af` | claim gate closed |
| Seed 42 MBPP-Plus rerank | `docs/benchmark/v0_6/seed-42/downstream_rerank_pilot/mbpp_plus/manifest.json` | `eval_report-c9c7f3dce3c7b877` | claim gate closed |
| Seed 1729 HumanEval rerank | `docs/benchmark/v0_6/seed-1729/downstream_rerank_pilot/humaneval/manifest.json` | `eval_report-9e80aaf3a656f995` | claim gate closed |
| Seed 1729 MBPP-Plus rerank | `docs/benchmark/v0_6/seed-1729/downstream_rerank_pilot/mbpp_plus/manifest.json` | `eval_report-36662148dc03d44c` | claim gate closed |

## Results

All baselines tie because the pilot label set is saturated: HumanEval has no
failing completion and MBPP-Plus has only four failing completions across 50
samples. CodeLeWM has zero pass@1 lift over LLM order and no-action on both
seeds.

| Seed | Benchmark | Problems | Completions | Pass rate | LLM order pass@1 | No-action pass@1 | CodeLeWM pass@1 | CodeLeWM lift | Claim |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | HumanEval | 25 | 50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 pts | closed |
| 42 | MBPP-Plus | 25 | 50 | 0.92 | 0.92 | 0.92 | 0.92 | 0.00 pts | closed |
| 1729 | HumanEval | 25 | 50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 pts | closed |
| 1729 | MBPP-Plus | 25 | 50 | 0.92 | 0.92 | 0.92 | 0.92 | 0.00 pts | closed |

The bootstrap lift CIs are `[0.0, 0.0]` for every pilot report. The claim gate
reason in each report is:

```text
llm_order_lift=0.00pts ci=(0.00,0.00); requires lift>=3.0 and ci_lo>0; no_action_lift=0.00pts ci=(0.00,0.00); requires lift>=3.0 and ci_lo>0
```

## Interpretation

The pilot validates the completion-label artifact contract, live OpenRouter
sampling path, sandbox labeling path, learned-checkpoint scoring path, parent
manifest verification, and public secret-scan gate. It does not validate a
downstream utility claim because the candidate set is too easy and the reranker
has no room to improve pass@1.

The #320 full-scale run should proceed only with a harder predeclared protocol:

- use the full benchmark problem set;
- prefer uncapped tests for HumanEval and as many MBPP-Plus tests as runtime
  permits;
- increase candidate diversity with more LLM sampling seeds and/or higher
  temperature;
- record any case cap as a limitation, not as EvalPlus pass@1;
- keep both v0.6 checkpoint seeds and the no-action baseline mandatory.

## Validation

```bash
uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_6/completion_labels_pilot/humaneval/manifest.json \
  --json
uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_6/completion_labels_pilot/mbpp_plus/manifest.json \
  --json
for seed in 42 1729; do
  for bench in humaneval mbpp_plus; do
    uv run codelewm manifest verify \
      --manifest docs/benchmark/v0_6/seed-${seed}/downstream_rerank_pilot/${bench}/manifest.json \
      --parent-manifest docs/benchmark/v0_6/completion_labels_pilot/${bench}/manifest.json \
      --json
  done
done
uv run codelewm secret-scan \
  docs/benchmark/v0_6/completion_labels_pilot \
  docs/benchmark/v0_6/seed-42/downstream_rerank_pilot \
  docs/benchmark/v0_6/seed-1729/downstream_rerank_pilot \
  --json
```
