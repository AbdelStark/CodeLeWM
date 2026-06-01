# CodeLeWM v0.6 Full-Scale Downstream Rerank Evidence - 2026-06-01

This report closes the #320 downstream-rerank gap for the v0.6
execution-substrate line. It evaluates generated HumanEval and MBPP-Plus
completion labels against both public v0.6 checkpoints. It is a negative /
diagnostic result, not a downstream coding-utility claim.

## Scope

- HumanEval: 154 parsed problems, 462 live completions, 3 LLM seeds
  (`17,42,1729`), one sample per seed, uncapped hidden-test inputs, sandbox
  determinism check enabled.
- MBPP-Plus: 370 parsed problems, 1110 live completions, 3 LLM seeds
  (`17,42,1729`), one sample per seed, first 8 hidden-test inputs per problem,
  sandbox determinism check disabled.
- LLM: `openrouter:anthropic/claude-haiku-4-5`.
- Rerank checkpoints:
  `.artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42/checkpoints/last.pt`
  and
  `.artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729/checkpoints/last.pt`.
- Claim gate: CodeLeWM pass@1 lift must be at least 3 absolute points over both
  LLM order and no-action, with bootstrap 95% CI lower bound greater than 0.

The MBPP-Plus labels are full-scale by problem and completion count, but not
full EvalPlus pass@1 because hidden-test cases are capped to 8 per problem. The
aborted 32-case MBPP-Plus attempt was still sandboxing after roughly four
hours with no partial artifact; the 8-case cap is the tractable evidence point
recorded here.

## Completion Label Artifacts

| Benchmark | Artifact id | Problems | Completions | Passed | Valid candidates | Test pass rate | Valid rate | Case cap | Determinism |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HumanEval | `downstream_benchmark-62060bb55ca7c890` | 154 | 462 | 439 | 458 | 0.9502 | 0.9913 | none | enabled |
| MBPP-Plus | `downstream_benchmark-5bff755577cc17ba` | 370 | 1110 | 1023 | 1086 | 0.9216 | 0.9784 | 8 | disabled |

Both completion-label manifests verify, and `codelewm secret-scan` returns
`ok=true` with zero findings across the completion-label and rerank trees.

## Rerank Results

| Seed | Benchmark | Eval artifact | CodeLeWM pass@1 | LLM-order pass@1 | No-action pass@1 | Lexical pass@1 | CodeLeWM MRR | Lift vs LLM order | Bootstrap CI | Lift vs no-action | Bootstrap CI | Claim |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| 42 | HumanEval | `eval_report-9c8d7152bff2460c` | 0.9610 | 0.9545 | 0.9481 | 0.9545 | 0.9632 | +0.65 pts | [0.00, 1.95] | +1.30 pts | [0.00, 3.25] | closed |
| 42 | MBPP-Plus | `eval_report-14887c62e5e76e19` | 0.9162 | 0.9243 | 0.9243 | 0.9297 | 0.9221 | -0.81 pts | [-1.89, 0.27] | -0.81 pts | [-1.89, 0.00] | closed |
| 1729 | HumanEval | `eval_report-ca4212e0d1565af4` | 0.9481 | 0.9545 | 0.9610 | 0.9545 | 0.9567 | -0.65 pts | [-1.95, 0.00] | -1.30 pts | [-3.25, 0.00] | closed |
| 1729 | MBPP-Plus | `eval_report-3abf5d64a09d9c56` | 0.9243 | 0.9243 | 0.9243 | 0.9297 | 0.9266 | +0.00 pts | [-1.08, 1.08] | +0.00 pts | [-0.81, 0.81] | closed |

The generated completions are highly saturated: HumanEval pass@k is 0.9675 and
MBPP-Plus capped pass@k is 0.9297. CodeLeWM does not produce stable lift over
LLM order, no-action, or lexical ordering on this artifact set.

## Verification

Commands run locally:

```bash
uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_6/completion_labels_full/humaneval/manifest.json \
  --json
uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_6/completion_labels_full/mbpp_plus/manifest.json \
  --json
uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_6/seed-42/downstream_rerank_full/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_6/completion_labels_full/humaneval/manifest.json \
  --json
uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_6/seed-42/downstream_rerank_full/mbpp_plus/manifest.json \
  --parent-manifest docs/benchmark/v0_6/completion_labels_full/mbpp_plus/manifest.json \
  --json
uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_6/seed-1729/downstream_rerank_full/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_6/completion_labels_full/humaneval/manifest.json \
  --json
uv run codelewm manifest verify \
  --manifest docs/benchmark/v0_6/seed-1729/downstream_rerank_full/mbpp_plus/manifest.json \
  --parent-manifest docs/benchmark/v0_6/completion_labels_full/mbpp_plus/manifest.json \
  --json
uv run codelewm secret-scan \
  docs/benchmark/v0_6/completion_labels_full \
  docs/benchmark/v0_6/seed-42/downstream_rerank_full \
  docs/benchmark/v0_6/seed-1729/downstream_rerank_full \
  --json
```

All manifest checks returned `ok=true`; the secret scan returned `ok=true` with
zero findings.

## Claim Boundary

This evidence removes the previous "not run" limitation for HumanEval and
MBPP-Plus reranking, but it does not open a positive downstream-utility claim.
The safe public boundary remains:

> CodeLeWM v0.6 learns a non-collapsed, action-conditioned execution-trace
> substrate with strong execution-pack retrieval and semantic-decoy diagnostics;
> HumanEval / MBPP-Plus generated-code reranking remains negative or
> inconclusive on the current full-scale artifact set.

