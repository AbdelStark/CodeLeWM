# CodeLeWM v0.9 Public Artifact Index 2026-06-07

This index maps the public v0.9 cross-benchmark execution artifacts. It is a
diagnostic artifact index, not a general coding-usefulness claim.

Current claim boundary:

- the v0.9 pass/fail execution pack covers HumanEval and MBPP-Plus;
- the two v0.9 A10G jobs completed and uploaded verified run artifacts;
- HumanEval WS-D reranking passes on both seeds;
- MBPP-Plus WS-D is saturated, with CodeLeWM, no-action, and lexical all at
  pass@1 `1.0000`;
- broad semantic-decoy surprise and representation gates remain closed.

## Public Repositories

| Surface | Repository | Repo type | Revision / path |
| --- | --- | --- | --- |
| v0.9 training pack | `abdelstark/codelewm-execution-pack` | dataset | `v0.9.0-rc1` |
| Training runs | `abdelstark/codelewm-runs` | dataset | `codelewm-v0-9-short-execution-20260606-69f798a-seed-{42,1729}` |
| Eval reports | `github.com/AbdelStark/CodeLeWM` | git | `docs/benchmark/v0_9/seed-{42,1729}/` |

## Indexed Artifacts

| Artifact | Public location | Local evidence | Card / report | Claim posture |
| --- | --- | --- | --- | --- |
| v0.9 pass/fail execution pack | `abdelstark/codelewm-execution-pack@v0.9.0-rc1` | `codelewm-passfail-execution-pack-20260606T122240Z`; 2,188 records; manifest verify `ok=true`; secret scan `ok=true` | `docs/cards/codelewm-v0-9-execution-dataset-2026-06-07.md` | Cross-benchmark training/eval substrate |
| Seed 42 training run | `abdelstark/codelewm-runs/codelewm-v0-9-short-execution-20260606-69f798a-seed-42` | `training_run-992f7757f2780da4`; checkpoint SHA `c783fa0dbe5da6bd072ff0b2f2753bdbac9fe684b49bf82e70ab6a2f69d513da`; checkpoint inspection `eval_report-54b696c9cd038493`; manifest verify `ok=true`; secret scan `ok=true` | `docs/cards/codelewm-v0-9-execution-model-seed-42-2026-06-07.md` | HumanEval rerank positive; overall claim closed |
| Seed 1729 training run | `abdelstark/codelewm-runs/codelewm-v0-9-short-execution-20260606-69f798a-seed-1729` | `training_run-91e9cf7c645379b3`; checkpoint SHA `34ebb282b284580dd123c781ae77c93cc36bbffc4eeeee9f0bd4cdf8042001eb`; checkpoint inspection `eval_report-5c9de2e6f492809c`; manifest verify `ok=true`; secret scan `ok=true` | `docs/cards/codelewm-v0-9-execution-model-seed-1729-2026-06-07.md` | HumanEval rerank positive; overall claim closed |
| Seed 42 eval reports | Code repo | retrieval `eval_report-a8c3610d40df7512`; surprise `eval_report-3f298b8a90ac9cb6`; probe `eval_report-04b816b7a31f36d5`; HumanEval rerank `eval_report-0bc9a04d4a6bfa86`; MBPP-Plus rerank `eval_report-7e9fa967ee6356af`; calibration `eval_report-203075dc79ada0d3`; all manifests verify with parents | `docs/benchmark/v0_9/seed-42/` | HumanEval rerank positive; MBPP/semantic/probe gates closed |
| Seed 1729 eval reports | Code repo | retrieval `eval_report-0e49571dcb2fc373`; surprise `eval_report-c59139ea35734802`; probe `eval_report-379017500d91ecf5`; HumanEval rerank `eval_report-3cd1cfeeb2fe2c09`; MBPP-Plus rerank `eval_report-570bdbfeac5928ef`; calibration `eval_report-9c92eb76a900e4be`; all manifests verify with parents | `docs/benchmark/v0_9/seed-1729/` | HumanEval rerank positive; MBPP/semantic/probe gates closed |
| Results report | Code repo | verified eval table and claim-boundary audit | `docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md` | benchmark-specific diagnostic result |

## Verification Commands

```bash
hf download abdelstark/codelewm-execution-pack \
  --repo-type dataset \
  --revision v0.9.0-rc1 \
  --local-dir .artifacts/v0_9/hf-pack-download

uv run codelewm manifest verify \
  --manifest .artifacts/v0_9/hf-pack-download/artifact_manifest.json \
  --json

uv run codelewm secret-scan .artifacts/v0_9/hf-pack-download --json
uv run codelewm secret-scan docs/benchmark/v0_9 --json
```

Run and eval child manifests require their parent manifests when verifying. The
results report lists representative parent-aware commands.

## Publication Copy Boundary

Safe copy:

> CodeLeWM v0.9 repairs the v0.8 cross-benchmark data/eval blockers, completes
> two guarded A10G execution-training runs, keeps core diagnostics healthy, and
> passes HumanEval WS-D reranking on both seeds. The overall downstream claim
> remains closed because MBPP-Plus is saturated with zero lift over no-action
> and lexical controls, and representation and broad semantic gates do not
> clear.

Avoid stronger claims unless a follow-up artifact reruns the closed gates and
passes them with the required seeds.
