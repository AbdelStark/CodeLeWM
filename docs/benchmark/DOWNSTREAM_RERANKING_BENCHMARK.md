# Downstream Reranking Benchmark

Last updated: 2026-06-08

Issue: #192. Parent tracker: #184.

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

Issue #191 adds a public-safe benchmark pack builder:

```bash
uv run codelewm eval downstream-pack \
  --config config/benchmark/downstream_rerank_fixture.json \
  --out .artifacts/downstream-rerank-fixture \
  --overwrite \
  --json
```

The fixture pack writes `codelewm.downstream_rerank_benchmark.v1`,
`codelewm.downstream_benchmark_readiness.v1`,
`codelewm.downstream_source_license_policy.v1`,
`codelewm.downstream_split_leakage_report.v1`,
`codelewm.secret_scan.v1`, and `codelewm.artifact_manifest.v1` artifacts. The
source policy is project-owned synthetic fixtures under the repository license.
It is public-safe but remains smoke evidence because it has one labeled task.

Issue #192 adds the downstream comparison and claim gate:

```bash
uv run codelewm eval downstream-rerank \
  --benchmark-manifest .artifacts/downstream-rerank-fixture/manifest.json \
  --checkpoint .artifacts/first-results/train/checkpoints/checkpoint.pt \
  --out .artifacts/downstream-rerank-report \
  --allow-unsafe-checkpoint \
  --overwrite \
  --json
```

The report schema is `codelewm.downstream_rerank_report.v1`; the stdout run
schema is `codelewm.downstream_rerank_eval_run.v1`. The report compares LLM
order, random, lexical, no-action, CodeLeWM energy, retrieval prior, and score
ensemble baselines, records baseline availability status, then writes
`codelewm.downstream_rerank_claim_gate.v1`. The retrieval-prior baseline is
marked blocked when no `--index` produces finite retrieval-prior scores.
For the fixture, the expected claim gate remains closed because
`example_count=1`.

## Benchmark Payload Schema

Benchmark schema: `codelewm.downstream_rerank_benchmark.v1`.

Each benchmark payload records:

- `benchmark_id`;
- `tasks[]`;
- `required_baselines`;
- `required_metrics`;
- `min_labeled_examples`;
- `provenance`.

Each task records:

- `task_id`;
- `task_type`;
- `prompt`;
- `before_path`;
- `candidates[]`;
- task-level provenance, including source dataset and candidate-pack lineage.

Each candidate records:

- stable `candidate_id`;
- original `llm_rank`;
- human or check-derived label: `pass`, `fail`, or `unknown`;
- `patch_path` or `after_state_path`;
- static-check status: `pass`, `fail`, `not_run`, or `not_applicable`;
- test-check status: `pass`, `fail`, `not_run`, or `not_applicable`;
- candidate source metadata;
- provenance, including LLM model, candidate-pack artifact, and checksum data.

Report schema: `codelewm.downstream_rerank_report.v1`.

Claim gate schema: `codelewm.downstream_rerank_claim_gate.v1`.

## Minimum Scaled Contract

A scaled downstream claim needs:

- at least `100` labeled reranking examples;
- candidates for `true_after`, `hard_negative`, `syntax_failure`,
  `patch_failure`, and plausible wrong edits where available;
- non-execution parsing and patch-application checks;
- pass@1, pass@k, MRR, valid-patch rate, static/test check-pass rate,
  mean/median true rank, and failure counts;
- slices by task type, candidate source, and failure type;
- required baselines: LLM order, random, lexical, no-action, CodeLeWM,
  retrieval prior, and score ensemble;
- runs from downloaded Hugging Face artifacts, not a job working directory.

## Hard Anti-Saturation Follow-Up

RFC-0016 defines the next benchmark profile:
`anti_saturation_semantic_v1`. This profile exists because the v1.0 paper-demo
replay showed the core blocker: MBPP-Plus WS-D is saturated, so CodeLeWM cannot
show added value over no-action or lexical controls on that slice.

The harder benchmark must write
`codelewm.downstream_anti_saturation_report.v1` before CodeLeWM scoring. A
headline slice is eligible only when:

- locked test `problem_count >= 100`;
- candidate pools contain `6` to `12` candidates;
- no-action pass@1 is below `0.85`;
- lexical pass@1 is below `0.85`;
- LLM-order pass@1 is below `0.90`;
- at least `70%` of problems contain two or more failing hard-negative classes;
- source/license, split-leakage, manifest, checkpoint-trust, and secret-scan
  gates pass.

Required hard-negative classes include no-action baits, partial fixes,
wrong-symbol or wrong-branch fixes, over-broad fixes, deterministic semantic
mutants, and LLM-generated valid candidates. Parser/apply failures are retained
for accounting but cannot open a positive claim.

The hard benchmark can support a stronger claim only if CodeLeWM beats
no-action, lexical, and LLM-order on pass@1 and MRR with bootstrap confidence
intervals over all three lifts excluding zero. If no-action or lexical are
near-perfect, that slice is marked saturated and remains diagnostic.

## Claim Gate

The downstream claim gate defaults to `allowed=false`.

It may flip to `allowed=true` only when all of the following are true:

- `example_count >= 100`;
- CodeLeWM is strictly above LLM order on pass@1 and MRR;
- CodeLeWM is strictly above no-action on pass@1 and MRR;
- valid-patch and check-pass rates are reported for the evaluated set;
- slices do not show that the improvement comes only from invalid, unchecked, or
  unsupported candidate categories.

This explicitly falsifies the downstream usefulness hypothesis if CodeLeWM does
not improve over LLM order or no-action after the 100-example gate is met.

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
