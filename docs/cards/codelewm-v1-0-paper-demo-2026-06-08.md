# CodeLeWM v1.0 Paper Demo Card

- Demo name: `codelewm-v1-0-paper-demo-replay`
- Artifact ID: `demo_report-e6fc06c328eed245`
- Manifest schema: `codelewm.artifact_manifest.v1`
- Demo report schema: `codelewm.harness.paper_demo_report.v1`
- Claim gate schema: `codelewm.harness.paper_demo_claim_gate.v1`
- Output directory: `docs/benchmark/v1_0/paper_demo/`
- Card date: `2026-06-08`
- Release status: final v1.0 deterministic replay artifact; aggregate claim
  closed.

## Summary

This demo is the final deterministic downstream replay used by the v1.0 paper
and release package. It replays checked-in v0.9 WS-D score rows for two learned
execution checkpoints and emits the final paper table, HTML report, timeline,
claim gate, manifest, and secret-scan report.

It is not a fresh checkpoint scoring run, not a live OpenRouter demo, and not a
coding-utility benchmark. Candidate code remains untrusted and is not imported,
executed, or test-run by the replay command.

## Inputs

| Input | Path |
| --- | --- |
| Seed 42 HumanEval rerank manifest | `docs/benchmark/v0_9/seed-42/rerank/humaneval/manifest.json` |
| Seed 42 MBPP-Plus rerank manifest | `docs/benchmark/v0_9/seed-42/rerank/mbpp_plus/manifest.json` |
| Seed 1729 HumanEval rerank manifest | `docs/benchmark/v0_9/seed-1729/rerank/humaneval/manifest.json` |
| Seed 1729 MBPP-Plus rerank manifest | `docs/benchmark/v0_9/seed-1729/rerank/mbpp_plus/manifest.json` |
| Seed 42 model card | `docs/cards/codelewm-v0-9-execution-model-seed-42-2026-06-07.md` |
| Seed 1729 model card | `docs/cards/codelewm-v0-9-execution-model-seed-1729-2026-06-07.md` |
| Dataset card | `docs/cards/codelewm-v0-9-execution-dataset-2026-06-07.md` |

## Outputs

| Output | Path |
| --- | --- |
| Manifest | `docs/benchmark/v1_0/paper_demo/manifest.json` |
| Machine report | `docs/benchmark/v1_0/paper_demo/reports/paper_demo_report.json` |
| Claim gate | `docs/benchmark/v1_0/paper_demo/reports/paper_demo_claim_gate.json` |
| Paper table | `docs/benchmark/v1_0/paper_demo/reports/paper_demo_table.md` |
| Timeline | `docs/benchmark/v1_0/paper_demo/reports/run_timeline.json` |
| HTML report | `docs/benchmark/v1_0/paper_demo/demo.html` |
| Secret-scan report | `docs/benchmark/v1_0/paper_demo/reports/secret_scan_report.json` |
| Artifact note | `docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md` |
| Final artifact index | `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-08.md` |

## Slice Outcomes

| Seed | Benchmark | CodeLeWM pass@1 | No-action pass@1 | LLM-order pass@1 | Lexical pass@1 | Slice gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 42 | HumanEval WS-D | 0.9787 | 0.8723 | 0.1489 | 0.6596 | open |
| 42 | MBPP-Plus WS-D | 1.0000 | 1.0000 | 0.1765 | 1.0000 | closed |
| 1729 | HumanEval WS-D | 0.9787 | 0.8936 | 0.1489 | 0.6596 | open |
| 1729 | MBPP-Plus WS-D | 1.0000 | 1.0000 | 0.1765 | 1.0000 | closed |

Aggregate claim gate: `claim_allowed=false`.

## Verification

```bash
uv run scripts/paper-demo --out docs/benchmark/v1_0/paper_demo --overwrite --json

uv run codelewm manifest verify \
  --manifest docs/benchmark/v1_0/paper_demo/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/mbpp_plus/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/mbpp_plus/manifest.json \
  --json

uv run codelewm secret-scan docs/benchmark/v1_0/paper_demo \
  --include-suffix .json \
  --include-suffix .md \
  --include-suffix .html \
  --json
```

Expected result: manifest verification `ok=true`; secret scan `ok=true` with
zero findings.

## Claim Boundary

This demo supports the final mixed CodeLeWM conclusion: a narrow HumanEval WS-D
diagnostic slice is positive, while the aggregate downstream claim remains
closed because MBPP-Plus is saturated against no-action and lexical controls.
It does not support a broad claim that CodeLeWM generally improves coding,
generated patches, or candidate ranking.

## Sign-off

| Reviewer | Role | GitHub handle | Date |
| --- | --- | --- | --- |
| AbdelStark | Release shepherd | @AbdelStark | 2026-06-08 |
