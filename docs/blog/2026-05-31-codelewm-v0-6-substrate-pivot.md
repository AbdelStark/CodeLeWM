# CodeLeWM v0.6: A Two-Substrate Test For Code World Models

Status: repo draft. Final publication waits for the arXiv URL from #306.

CodeLeWM asks a narrow question: can a compact JEPA-style latent transition
model learn useful dynamics over code-related states without decoding tokens?
The first answer was negative. The v0.2 commit-edit substrate produced a
working pipeline, but no-action beat action-conditioned scoring and the
representation gates stayed closed.

The v0.6 result changes the substrate, not the architecture. Instead of
predicting a latent after-state for a code edit, v0.6 predicts the latent output
of a deterministic Python execution trace: `(code, input, output)`.

## Why Change The Substrate?

The v0.2 substrate mixed three hard problems at once: edit semantics, patch
style, and natural-language action conditioning. That made negative results
hard to interpret. Was JEPA-style predictive learning failing on code, or was
the state/action substrate too noisy?

v0.6 removes one confound. Execution traces provide a cleaner transition: code
and input condition the output. The model still uses the same package-native
transition model family and objective registry, but the data now has a sharper
notion of "next state."

## Substrate A: Negative Commit-Edit Evidence

The v0.2 action-swap/inverse-action run remained diagnostic:

- text-action Recall@1 `0.263`, MRR `0.370048`;
- no-action Recall@1 `0.441`, MRR `0.533105`;
- latent-probe representation gates stayed closed;
- downstream scorer-quality evidence was fixture-scale only.

That is a useful result: the first substrate did not justify public positive
model-quality claims.

## Substrate B: v0.6 Execution-Trace Evidence

Across seed 42 and seed 1729, the v0.6 execution substrate passes the internal
shape gates:

- prediction MSE drops from about `0.94` to about `0.00062`;
- SIGReg drops from about `45` to about `0.036`;
- the no-action margin flips from roughly `-0.77` to `+1.24`;
- predicted latent effective-rank ratio reaches about `0.47`, above the `0.20`
  collapse gate.

The downloaded eval pass then tests the checkpoints on the execution pack:

| Seed | CodeLeWM Recall@1 | No-action Recall@1 | CodeLeWM MRR | No-action MRR |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.6568 | 0.0381 | 0.7670 | 0.1042 |
| 1729 | 0.6483 | 0.0381 | 0.7587 | 0.1040 |

The #322 semantic-decoy surprise rerun also passes the score gates with
Recall@1 `1.0000` and pairwise AUC `1.0000` on both seeds. Its count gates keep
the broader semantic claim closed: the semantic pack has 358 same-problem pairs
across 68 problems, but the narrow same-problem-different-submission slice is
still only 6/30 scored pairs.

## What This Does And Does Not Imply

This is not a claim that CodeLeWM improves coding agents. The HumanEval /
MBPP-Plus downstream-rerank benchmark is still not complete at the required
100-example scale. The live demo tour is a workflow showcase, and its claim
gate is closed.

The stronger statement is more precise: changing from noisy commit edits to
deterministic execution traces turns the same broad world-model recipe from a
negative result into a substrate-level partial positive. That is evidence that
state/action substrate design is load-bearing for code world models.

## Artifacts

- Public artifact index:
  `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-31.md`
- v0.6 results report:
  `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md`
- Semantic-decoy surprise rerun:
  `docs/benchmark/SEMANTIC_DECOY_SURPRISE_2026-06-01.md`
- Dataset card:
  `docs/cards/codelewm-v0-6-execution-dataset-2026-05-31.md`
- Model cards:
  `docs/cards/codelewm-v0-6-execution-model-seed-42-2026-05-31.md` and
  `docs/cards/codelewm-v0-6-execution-model-seed-1729-2026-05-31.md`
- Demo:
  `docs/demo/execution_rerank_tour_2026-05-31.html`
- Paper draft and arXiv package:
  `docs/papers/two_substrate_paper.tex`,
  `docs/papers/two_substrate_paper.pdf`, and
  `docs/papers/ARXIV_SUBMISSION.md`

## Safe Citation Wording

Until the arXiv URL lands, cite the repository artifacts directly:

> CodeLeWM v0.6 is a two-substrate diagnostic result: v0.2 commit-edit
> transition learning is negative, while v0.6 deterministic execution traces
> pass execution-pack retrieval and semantic-decoy score diagnostics across
> two seeds. Broad semantic surprise and downstream generated-code utility
> remain unsupported.
