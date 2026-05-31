# Demo Artifacts

This directory contains small, public-safe demo recordings and pointers. Demo
artifacts are workflow evidence only; they do not support coding-usefulness
claims.

## v0.6 Execution-Rerank Tour

- Asciicast: `execution_rerank_tour_2026-05-31.cast`
- Static HTML: `execution_rerank_tour_2026-05-31.html`
- Scenario: `execution-rerank-mbpp`
- Checkpoint source: `abdelstark/codelewm-runs/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42`
- Result: five synthetic MBPP-style tasks, two live OpenRouter candidates per
  task, sandbox labels, v0.6 execution-substrate scores, and a closed claim
  gate.

Recreate the HTML report locally:

```bash
CODELEWM_LLM_DRY_RUN=0 CODELEWM_LLM_MAX_CANDIDATES=2 \
  uv run scripts/llm-world-model-demo \
  --scenario execution-rerank-mbpp \
  --checkpoint .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42/checkpoints/last.pt \
  --tour 5 \
  --html .artifacts/v0-6-execution-rerank-tour-live.html
```
