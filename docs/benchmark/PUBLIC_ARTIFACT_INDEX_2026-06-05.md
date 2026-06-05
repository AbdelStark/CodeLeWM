# CodeLeWM v0.8 Public Artifact Index 2026-06-05

This index maps the public v0.8 correctness-aware execution artifacts. It is a
diagnostic artifact index, not a general coding-usefulness claim.

Current claim boundary:

- the two v0.8 A10G jobs completed and uploaded verified run artifacts;
- the model checkpoints are public under both the run-artifact repo and the
  transition-model checkpoint mirror;
- HumanEval WS-D reranking passes on both seeds;
- MBPP-Plus WS-D reranking, pass/fail latent probing, magnitude probing, and
  broad semantic-decoy surprise do not support a general positive claim.

## Public Repositories

| Surface | Repository | Repo type | Revision / path |
| --- | --- | --- | --- |
| v0.8 training pack | `abdelstark/codelewm-execution-pack` | dataset | `v0.8.0-rc1` |
| Training runs | `abdelstark/codelewm-runs` | dataset | `codelewm-v0-8-short-execution-20260605-1b737e4-seed-{42,1729}` |
| Checkpoint mirror | `abdelstark/codelewm-transition-model` | model | `checkpoints/codelewm-v0-8-short-execution-20260605-1b737e4-seed-{42,1729}` |
| Eval reports | `github.com/AbdelStark/CodeLeWM` | git | `docs/benchmark/v0_8/seed-{42,1729}/` |

## Indexed Artifacts

| Artifact | Public location | Local evidence | Card / report | Claim posture |
| --- | --- | --- | --- | --- |
| v0.8 pass/fail execution pack | `abdelstark/codelewm-execution-pack@v0.8.0-rc1` | `codelewm-passfail-execution-pack-20260604T185716Z`; 1,882 records; manifest verify `ok=true`; secret scan `ok=true` | `docs/cards/codelewm-v0-8-execution-dataset-2026-06-05.md` | HumanEval-only correctness training substrate |
| Seed 42 training run | `abdelstark/codelewm-runs/codelewm-v0-8-short-execution-20260605-1b737e4-seed-42` | `training_run-e2a757caf75cbcf2`; checkpoint SHA `03707bc09d1e60d74bdd94d649f4179632ea57fc06b6f0640d05083664aa7136`; manifest verify `ok=true`; secret scan `ok=true` | `docs/cards/codelewm-v0-8-execution-model-seed-42-2026-06-05.md` | HumanEval rerank positive; overall claim closed |
| Seed 1729 training run | `abdelstark/codelewm-runs/codelewm-v0-8-short-execution-20260605-1b737e4-seed-1729` | `training_run-951983cbf59f6fa6`; checkpoint SHA `dafc3bf8022d1f3a9560b63f63c620873a2f2b3922e4f773a350e3f88c15ecfe`; manifest verify `ok=true`; secret scan `ok=true` | `docs/cards/codelewm-v0-8-execution-model-seed-1729-2026-06-05.md` | HumanEval rerank positive; overall claim closed |
| Seed 42 eval reports | Code repo | retrieval `eval_report-5eea8084a2ba3b6b`; surprise `eval_report-a174318a68d39f78`; probe `eval_report-c36b5955682a345f`; HumanEval rerank `eval_report-c6b95bc1d7074b86`; MBPP-Plus rerank `eval_report-0e9ca50bd99101c9`; all manifests verify with parents | `docs/benchmark/v0_8/seed-42/` | HumanEval rerank positive; probe/magnitude/MBPP gates closed |
| Seed 1729 eval reports | Code repo | retrieval `eval_report-28a06070d6d0364b`; surprise `eval_report-298fe49fe1efa80c`; probe `eval_report-da9e8fcb14e4dc14`; HumanEval rerank `eval_report-f31174db07b944a5`; MBPP-Plus rerank `eval_report-1ec2e56ac88ab11b`; all manifests verify with parents | `docs/benchmark/v0_8/seed-1729/` | HumanEval rerank positive; probe/magnitude/MBPP gates closed |
| Results report | Code repo | verified eval table and claim-boundary audit | `docs/benchmark/EXECUTION_V0_8_RESULTS_2026-06-05.md` | benchmark-specific diagnostic result |

## Verification Commands

```bash
hf download abdelstark/codelewm-execution-pack \
  --repo-type dataset \
  --revision v0.8.0-rc1 \
  --local-dir /tmp/codelewm-v0-8-pack

uv run codelewm manifest verify \
  --manifest /tmp/codelewm-v0-8-pack/artifact_manifest.json \
  --json

uv run codelewm secret-scan /tmp/codelewm-v0-8-pack --json

uv run codelewm secret-scan docs/benchmark/v0_8 --json
```

Run and eval child manifests require their parent manifests when verifying.
The results report lists representative parent-aware commands.

## Publication Copy Boundary

Safe copy:

> CodeLeWM v0.8 turns the v0.7 execution substrate into a correctness-aware
> diagnostic scorer. It completes two A10G runs, keeps non-collapse and
> execution-pack diagnostics healthy, and passes HumanEval WS-D reranking on
> both seeds. Cross-benchmark downstream utility remains unsupported because
> MBPP-Plus and latent-probe gates do not clear.

Avoid stronger claims unless a follow-up artifact reruns the closed gates and
passes them with the required seeds.
