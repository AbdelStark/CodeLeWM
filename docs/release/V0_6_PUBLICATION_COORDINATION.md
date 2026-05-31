# v0.6 Publication Coordination

Status: repo-side release package staged; final public announcement blocked on
operator arXiv upload.

## Landing Files

| Surface | Path |
| --- | --- |
| Public artifact index | `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-31.md` |
| Blog-style announcement draft | `docs/blog/2026-05-31-codelewm-v0-6-substrate-pivot.md` |
| Dataset card | `docs/cards/codelewm-v0-6-execution-dataset-2026-05-31.md` |
| Seed 42 model card | `docs/cards/codelewm-v0-6-execution-model-seed-42-2026-05-31.md` |
| Seed 1729 model card | `docs/cards/codelewm-v0-6-execution-model-seed-1729-2026-05-31.md` |
| Demo HTML | `docs/demo/execution_rerank_tour_2026-05-31.html` |
| Demo asciicast | `docs/demo/execution_rerank_tour_2026-05-31.cast` |
| Paper package | `docs/papers/ARXIV_SUBMISSION.md` |

## Finalization Checklist

- [ ] Operator uploads `docs/papers/two_substrate_arxiv_source.tar.gz` through
  the arXiv author account.
- [ ] Commit the assigned arXiv URL into `docs/papers/ARXIV_SUBMISSION.md`,
  `docs/PROJECT_EXPLAINER.md`, `README.md`, the model cards, dataset card,
  artifact index, and blog post.
- [ ] Upload the model-card READMEs to
  `abdelstark/codelewm-transition-model@v0.6.0-seed-42` and
  `abdelstark/codelewm-transition-model@v0.6.0-seed-1729`.
- [ ] Refresh the HF dataset README for
  `abdelstark/codelewm-execution-pack@v0.6.0` with links to the arXiv URL,
  model cards, public artifact index, and demo.
- [ ] Publish or link the blog post from the chosen public surface.
- [ ] Re-run manifest verification and secret scans listed in
  `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-31.md`.

## Claim Boundary

The final announcement must preserve the partial-positive wording:
execution-pack retrieval and generated-decoy surprise pass across two seeds;
latent semantic axes, crash prediction, and scaled downstream coding-usefulness
claims remain unsupported.
