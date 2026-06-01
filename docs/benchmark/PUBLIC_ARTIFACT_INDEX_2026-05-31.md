# CodeLeWM v0.6 Public Artifact Index 2026-05-31

This index is the public landing map for the v0.6 substrate-pivot line. It is
not a coding-usefulness claim.

Current claim boundary:

- v0.6 is a partial-positive research result for the execution-trace substrate;
- training collapse gates, execution-pack retrieval, and semantic-decoy score
  diagnostics pass across two seeds;
- broad semantic surprise, latent-probe semantic-axis, crash-prediction, and
  HumanEval / MBPP-Plus downstream rerank utility claims remain unsupported;
- the 2026-06-01 HumanEval / MBPP-Plus rerank pilot validates live plumbing but
  is saturated and has zero pass@1 lift;
- final announcement and HF README refresh still wait for the #306 arXiv URL.

## Public Repositories

| Surface | Repository | Repo type | Revision / path |
| --- | --- | --- | --- |
| Execution pack | `abdelstark/codelewm-execution-pack` | dataset | `v0.6.0` |
| Training runs | `abdelstark/codelewm-runs` | dataset | `runs/codelewm-v0-6-execution-20260530-af1a114-seed-{42,1729}` |
| Checkpoint files | `abdelstark/codelewm-runs` | dataset | `runs/codelewm-v0-6-execution-20260530-af1a114-seed-{42,1729}/checkpoints/last.pt` |
| Eval pass mirror | `abdelstark/codelewm-runs` | dataset | `runs/codelewm-v0-6-eval-pass-20260531` |
| Code and demo | `github.com/AbdelStark/CodeLeWM` | git | `main` |

The canonical v0.6 checkpoint surface is the public `codelewm-runs` dataset
repo. Earlier CodeLeWM checkpoints remain available from
`abdelstark/codelewm-transition-model`, but this index does not document v0.6
model-repo tags because the resolving public v0.6 files are the run-artifact
checkpoint paths above.

## Indexed Artifacts

| Artifact | Public location | Local evidence | Card / report | Claim posture |
| --- | --- | --- | --- | --- |
| Execution pack | `abdelstark/codelewm-execution-pack@v0.6.0` | `manifest verify ok=true`; `secret-scan ok=true`; pack SHA `d770c5df...cf41b` | `docs/cards/codelewm-v0-6-execution-dataset-2026-05-31.md` | dataset substrate only |
| Seed 42 checkpoint and training run | `abdelstark/codelewm-runs/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42` | `training_run-cb62408f881eff8c`; checkpoint `checkpoints/last.pt`; `manifest verify ok=true`; `secret-scan ok=true` | `docs/cards/codelewm-v0-6-execution-model-seed-42-2026-05-31.md` | execution-pack retrieval and semantic-decoy score diagnostics pass; broad downstream utility unsupported |
| Seed 1729 checkpoint and training run | `abdelstark/codelewm-runs/runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729` | `training_run-d0b59108447c9c4a`; checkpoint `checkpoints/last.pt`; `manifest verify ok=true`; `secret-scan ok=true` | `docs/cards/codelewm-v0-6-execution-model-seed-1729-2026-05-31.md` | execution-pack retrieval and semantic-decoy score diagnostics pass; broad downstream utility unsupported |
| Seed 42 eval reports | `abdelstark/codelewm-runs/runs/codelewm-v0-6-eval-pass-20260531` | retrieval `eval_report-50a62748784329b2`; surprise `eval_report-06ac38fbc347961d`; probe `eval_report-952d5632120e0632`; crash `eval_report-48380fb96f1de96d`; all manifests verify with parents | `docs/benchmark/v0_6/seed-42/` | retrieval/surprise pass; probe/crash blocked |
| Seed 1729 eval reports | `abdelstark/codelewm-runs/runs/codelewm-v0-6-eval-pass-20260531` | retrieval `eval_report-0cc1c6ac187e4ed3`; surprise `eval_report-29c0d125cc25d631`; probe `eval_report-c592b4805d0d3085`; crash `eval_report-1f41882839c44da7`; all manifests verify with parents | `docs/benchmark/v0_6/seed-1729/` | retrieval/surprise pass; probe/crash blocked |
| Semantic decoy pack | Code repo | `downstream_benchmark-f3e0dc65fbf18825`; 358 pairs across 68 problems; secret scan `ok=true` | `docs/benchmark/SEMANTIC_DECOY_PACK_2026-06-01.md` | count source for semantic rerun; not a model claim |
| Semantic surprise reruns | Code repo | seed 42 `eval_report-aeb0ae374582a8ec`; seed 1729 `eval_report-5b20bd0e1da5928a`; manifests verify with training, pack, and semantic-pack parents | `docs/benchmark/SEMANTIC_DECOY_SURPRISE_2026-06-01.md` | score gates pass; broad semantic claim closed at 6/30 same-problem/different-submission pairs |
| HumanEval / MBPP-Plus rerank pilot | Code repo | labels `downstream_benchmark-7d549a6ec13ab791`, `downstream_benchmark-db47a1daa39e2d3e`; rerank reports `eval_report-7222346bfae136af`, `eval_report-c9c7f3dce3c7b877`, `eval_report-9e80aaf3a656f995`, `eval_report-36662148dc03d44c`; secret scan `ok=true` | `docs/benchmark/V0_6_RERANK_PILOT_2026-06-01.md` | live pipeline validated; saturated; no downstream utility claim |
| Results report | Code repo | `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md` | same | partial-positive summary |
| Paper package | Code repo | `docs/papers/two_substrate_paper.pdf`; `docs/papers/two_substrate_arxiv_source.tar.gz` | `docs/papers/ARXIV_SUBMISSION.md` | arXiv URL pending operator upload |
| Demo tour | Code repo | `docs/demo/execution_rerank_tour_2026-05-31.cast`; `docs/demo/execution_rerank_tour_2026-05-31.html`; secret scan `ok=true` | `docs/demo/README.md` | workflow evidence only |
| Blog-style announcement | Code repo | `docs/blog/2026-05-31-codelewm-v0-6-substrate-pivot.md` | same | draft until arXiv URL is assigned |

## Verification Commands

Execution pack:

```bash
uv run codelewm manifest verify \
  --manifest .artifacts/v0_6/execution-pack/artifact_manifest.json \
  --json
uv run codelewm secret-scan .artifacts/v0_6/execution-pack --json
```

Training runs:

```bash
hf download abdelstark/codelewm-runs \
  --repo-type dataset \
  --include 'runs/codelewm-v0-6-execution-20260530-af1a114-seed-42/**' \
  --local-dir /tmp/codelewm-v0-6-seed-42
hf download abdelstark/codelewm-runs \
  --repo-type dataset \
  --include 'runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729/**' \
  --local-dir /tmp/codelewm-v0-6-seed-1729
uv run codelewm manifest verify \
  --manifest .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42/manifest.json \
  --parent-manifest .artifacts/v0_6/execution-pack/artifact_manifest.json \
  --json
uv run codelewm manifest verify \
  --manifest .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729/manifest.json \
  --parent-manifest .artifacts/v0_6/execution-pack/artifact_manifest.json \
  --json
uv run codelewm secret-scan .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-42 --json
uv run codelewm secret-scan .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-1729 --json
```

Eval reports:

```bash
for seed in 42 1729; do
  for suite in execution_retrieval execution_surprise execution_probe crash_prediction; do
    uv run codelewm manifest verify \
      --manifest docs/benchmark/v0_6/seed-${seed}/${suite}/manifest.json \
      --parent-manifest .artifacts/v0_6/runs/codelewm-v0-6-execution-20260530-af1a114-seed-${seed}/manifest.json \
      --parent-manifest .artifacts/v0_6/execution-pack/artifact_manifest.json \
      --json
  done
done
uv run codelewm secret-scan docs/benchmark/v0_6 --json
```

Demo:

```bash
uv run codelewm secret-scan docs/demo --json
uv run codelewm secret-scan docs/demo --include-suffix .cast --json
uv run codelewm secret-scan docs/demo --include-suffix .html --json
```

## Publication Copy Boundary

Use this index to navigate artifacts. Do not say that CodeLeWM improves coding
agents, passes HumanEval / MBPP-Plus, or exposes semantic latent axes. The safe
public sentence is:

> CodeLeWM v0.6 shows that a JEPA-style latent world model can learn a
> non-collapsed, action-conditioned execution-trace substrate that passes
> execution-pack retrieval and semantic-decoy surprise score diagnostics across
> two seeds; broad semantic surprise and downstream generated-code utility
> remain open claims.
