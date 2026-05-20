# Downstream Reranking Benchmark

Last updated: 2026-05-20

Issue: #169. Parent tracker: #167.

## Status

Current status: **diagnostic only**.

The landed benchmark path is `codelewm eval scorer-quality`, which writes
`codelewm.harness.scorer_quality_report.v1`. The report now separates:

- `component_metrics.final_score`: the exact policy used by `codelewm rerank`;
- `component_metrics.transition_energy_only`: model transition energy without
  the retrieval prior;
- `component_metrics.retrieval_prior_only`: retrieval-prior-only ranking when a
  transition index is supplied.
- `baseline_controls`: random, lexical, no-action, #159 replay, and
  retrieval-prior-only controls, with blocked reasons when a control is not
  evaluable from the current artifact set.

The current checked-in fixture has one labeled reranking example, not the 100
examples required for a scaled downstream usefulness claim. Therefore
`benchmark_readiness.scaled_evaluation_ready=false` and
`benchmark_readiness.downstream_claim_allowed=false` are expected for fixture
runs.

## Minimum Scaled Contract

A scaled downstream claim needs:

- at least `100` labeled reranking examples;
- candidates for `true_after`, `hard_negative`, `syntax_failure`,
  `patch_failure`, and plausible wrong edits where available;
- non-execution parsing and patch-application checks;
- top-1, Recall@5, Recall@10, MRR, mean/median true rank, and failure counts;
- controls for random, lexical, no-action, #159 replay, retrieval-prior-only,
  transition-energy-only, and final combined score;
- runs from downloaded Hugging Face artifacts, not a job working directory.

## Fixture Command

```bash
uv run codelewm eval scorer-quality \
  --config config/first_results/scorer_quality.json \
  --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt \
  --out .artifacts/first-results/scorer_quality \
  --allow-unsafe-checkpoint \
  --overwrite \
  --json
```

The fixture command validates the benchmark plumbing and failure accounting. It
does not support a downstream coding usefulness claim.

## Downloaded HF Artifact Command

For #172 or any future scaled run, use only downloaded artifacts:

```bash
hf download "$CODELEWM_HF_RESULTS_REPO_ID" \
  --repo-type dataset \
  --include "runs/<run-id>/**" \
  --local-dir .artifacts/hf-download/<run-id>/results

hf download "$CODELEWM_HF_MODEL_REPO_ID" \
  --repo-type model \
  --include "checkpoints/<run-id>/**" \
  --local-dir .artifacts/hf-download/<run-id>/model

uv run codelewm score \
  --checkpoint .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --before tests/fixtures/codestate/class_method_before.py \
  --instruction "rewrite the accumulator update explicitly" \
  --candidate config/first_results/scorer_quality_candidates/true_after.py \
  --index .artifacts/hf-download/<run-id>/results/runs/<run-id>/index \
  --retrieval-prior-weight 1.0 \
  --json

uv run codelewm rerank \
  --checkpoint .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --before tests/fixtures/codestate/class_method_before.py \
  --instruction "rewrite the accumulator update explicitly" \
  --candidates config/first_results/scorer_quality_candidates \
  --index .artifacts/hf-download/<run-id>/results/runs/<run-id>/index \
  --retrieval-prior-weight 1.0 \
  --json

uv run codelewm eval scorer-quality \
  --config <scaled-reranking-config.json> \
  --checkpoint .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/checkpoints/checkpoint.pt \
  --out .artifacts/hf-download/<run-id>/local-checks/downstream_reranking \
  --index .artifacts/hf-download/<run-id>/results/runs/<run-id>/index \
  --retrieval-prior-weight 1.0 \
  --retrieval-prior-k 10 \
  --parent-manifest .artifacts/hf-download/<run-id>/model/checkpoints/<run-id>/manifest.json \
  --parent-manifest .artifacts/hf-download/<run-id>/results/runs/<run-id>/index/manifest.json \
  --overwrite \
  --json
```

## Current Verdict

Downstream reranking usefulness is **not supported yet**. The repo now has the
report fields needed to evaluate it, but the available fixture tier is below the
100-example gate and the current scorer backend remains diagnostic unless a
future downloaded HF run passes the benchmark readiness and component-metric
gates.
