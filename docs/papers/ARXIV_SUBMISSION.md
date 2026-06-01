# Two-Substrate Paper arXiv Submission Package

Status: source package prepared; operator upload pending.

The #306 draft lives in:

- Source: `docs/papers/two_substrate_paper.tex`
- References: `docs/papers/two_substrate_references.bib`
- Claim audit: `docs/papers/two_substrate_claim_audit.md`
- Built PDF: `docs/papers/two_substrate_paper.pdf`
- arXiv source bundle: `docs/papers/two_substrate_arxiv_source.tar.gz`

Regenerate the PDF and source bundle from a clean checkout:

```bash
scripts/build-two-substrate-paper
```

Current build evidence:

- Builder: `tectonic`
- Output PDF: `docs/papers/two_substrate_paper.pdf`
- Source bundle: `docs/papers/two_substrate_arxiv_source.tar.gz`

Artifact pointers to include in the arXiv metadata or comments field:

- Code: `https://github.com/AbdelStark/CodeLeWM`
- v0.2 report: `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`
- v0.6 report: `docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md`
- v0.6 semantic-decoy surprise rerun:
  `docs/benchmark/SEMANTIC_DECOY_SURPRISE_2026-06-01.md`
- v0.6 eval artifacts: `docs/benchmark/v0_6/`
- v0.6 eval HF mirror:
  `abdelstark/codelewm-runs/runs/codelewm-v0-6-eval-pass-20260531`
  at commit `396a8fab5b86c16764bec0090e8af7518de41fbc`
- v0.6 execution pack: `abdelstark/codelewm-execution-pack@v0.6.0`
- v0.6 public artifact index:
  `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-31.md`
- Release coordination checklist:
  `docs/release/V0_6_PUBLICATION_COORDINATION.md`

Operator upload checklist:

1. Build with `scripts/build-two-substrate-paper`.
2. Upload `docs/papers/two_substrate_arxiv_source.tar.gz` through the
   arXiv author account.
3. Confirm that arXiv's TeX build matches
   `docs/papers/two_substrate_paper.pdf`.
4. Commit the assigned arXiv URL back into this file,
   `docs/PROJECT_EXPLAINER.md`, and the relevant roadmap.

arXiv URL: pending operator upload.
