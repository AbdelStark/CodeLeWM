# CodeLeWM v1.0 Paper Demo Artifacts (2026-06-08)

This is the final downstream paper-demo artifact set for #405. It is a
clean-checkout replay over the checked-in v0.9 WS-D score rows, not a fresh
checkpoint scoring run and not a live OpenRouter demo.

## Artifact Set

| Surface | Path |
| --- | --- |
| Artifact directory | `docs/benchmark/v1_0/paper_demo/` |
| Manifest | `docs/benchmark/v1_0/paper_demo/manifest.json` |
| Machine report | `docs/benchmark/v1_0/paper_demo/reports/paper_demo_report.json` |
| Claim gate | `docs/benchmark/v1_0/paper_demo/reports/paper_demo_claim_gate.json` |
| Paper table | `docs/benchmark/v1_0/paper_demo/reports/paper_demo_table.md` |
| Timeline | `docs/benchmark/v1_0/paper_demo/reports/run_timeline.json` |
| HTML report | `docs/benchmark/v1_0/paper_demo/demo.html` |
| Secret-scan report | `docs/benchmark/v1_0/paper_demo/reports/secret_scan_report.json` |

Manifest artifact id: `demo_report-e6fc06c328eed245`.

Manifest source git SHA: `b75316bd987b2aafa016020eeb108cb189b23d3c`.

## Reproduction

```bash
uv run scripts/paper-demo --out docs/benchmark/v1_0/paper_demo --overwrite --json
```

The command emitted `schema_version=codelewm.harness.paper_demo_run.v1`,
`score_source=replay_existing_scores`, `slice_count=4`,
`problem_count=128`, `completion_count=768`, and `claim_allowed=false`.

## Verification

```bash
uv run codelewm manifest verify \
  --manifest docs/benchmark/v1_0/paper_demo/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-42/rerank/mbpp_plus/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/humaneval/manifest.json \
  --parent-manifest docs/benchmark/v0_9/seed-1729/rerank/mbpp_plus/manifest.json \
  --json
```

Result: `ok=true`, `files_checked=6`, and four parent rerank report artifacts
checked:

- `eval_report-0bc9a04d4a6bfa86`;
- `eval_report-7e9fa967ee6356af`;
- `eval_report-3cd1cfeeb2fe2c09`;
- `eval_report-570bdbfeac5928ef`.

```bash
uv run codelewm secret-scan docs/benchmark/v1_0/paper_demo \
  --include-suffix .json \
  --include-suffix .md \
  --include-suffix .html \
  --json
```

Result: `ok=true`, zero findings, seven files scanned.

## Learned Checkpoint Lineage

| Seed | Run artifact | Training artifact | Checkpoint SHA-256 |
| --- | --- | --- | --- |
| 42 | `abdelstark/codelewm-runs/codelewm-v0-9-short-execution-20260606-69f798a-seed-42` | `training_run-992f7757f2780da4` | `c783fa0dbe5da6bd072ff0b2f2753bdbac9fe684b49bf82e70ab6a2f69d513da` |
| 1729 | `abdelstark/codelewm-runs/codelewm-v0-9-short-execution-20260606-69f798a-seed-1729` | `training_run-91e9cf7c645379b3` | `34ebb282b284580dd123c781ae77c93cc36bbffc4eeeee9f0bd4cdf8042001eb` |

The paper-demo artifact records this lineage but does not re-load checkpoints
in replay mode. Candidate code remains untrusted and is not imported, executed,
or re-tested by the paper-demo command.

## Slice Outcomes

| Seed | Benchmark | CodeLeWM pass@1 | No-action pass@1 | LLM-order pass@1 | Claim gate |
| --- | --- | ---: | ---: | ---: | --- |
| 42 | HumanEval WS-D | 97.9% | 87.2% | 14.9% | open |
| 42 | MBPP-Plus WS-D | 100.0% | 100.0% | 17.6% | closed |
| 1729 | HumanEval WS-D | 97.9% | 89.4% | 14.9% | open |
| 1729 | MBPP-Plus WS-D | 100.0% | 100.0% | 17.6% | closed |

## Claim Boundary

The aggregate downstream claim gate is closed. Approved public wording:

> On the v0.9 WS-D replay, CodeLeWM strongly reranks HumanEval slices but the
> aggregate downstream claim remains closed because MBPP-Plus is saturated
> against the no-action baseline.

This artifact can support a narrow HumanEval WS-D positive diagnostic slice and
a mixed/negative aggregate conclusion. It does not support a broad claim that
CodeLeWM generally improves coding or generated patches.
