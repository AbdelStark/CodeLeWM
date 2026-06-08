# CodeLeWM Final Paper arXiv Submission Package

Status: final paper source prepared by #407; operator upload pending.

The active final v1.0 paper package lives in:

- Source: `docs/papers/codelewm_final_paper.tex`
- References: `docs/papers/two_substrate_references.bib`
- Claim audit: `docs/papers/codelewm_final_claim_audit.md`
- Build script: `scripts/build-codelewm-final-paper`
- Built PDF: `docs/papers/codelewm_final_paper.pdf`
- arXiv source bundle: `docs/papers/codelewm_final_arxiv_source.tar.gz`

Regenerate the final PDF and source bundle from a clean checkout:

```bash
scripts/build-codelewm-final-paper
```

Current build evidence:

- Builder: `tectonic`
- Output PDF: `docs/papers/codelewm_final_paper.pdf`
- Source bundle: `docs/papers/codelewm_final_arxiv_source.tar.gz`

Historical v0.6 two-substrate paper package:

- Source: `docs/papers/two_substrate_paper.tex`
- Claim audit: `docs/papers/two_substrate_claim_audit.md`
- Build script: `scripts/build-two-substrate-paper`
- PDF: `docs/papers/two_substrate_paper.pdf`
- Source bundle: `docs/papers/two_substrate_arxiv_source.tar.gz`

That historical package is superseded for final v1.0 release wording. It
remains useful as the v0.6 substrate-pivot record but should not be uploaded as
the final CodeLeWM paper.

Artifact pointers to include in the arXiv metadata or comments field:

- Code: `https://github.com/AbdelStark/CodeLeWM`
- v0.2 report: `docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md`
- v0.8 report: `docs/benchmark/EXECUTION_V0_8_RESULTS_2026-06-05.md`
- v0.9 final gate-suite report:
  `docs/benchmark/EXECUTION_V0_9_RESULTS_2026-06-07.md`
- v0.9 public artifact index:
  `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-06-07.md`
- v1.0 final paper-demo artifacts:
  `docs/benchmark/PAPER_DEMO_V1_0_ARTIFACTS_2026-06-08.md`
- v1.0 final claim audit:
  `docs/benchmark/V1_0_FINAL_CLAIM_AUDIT_2026-06-08.md`
- v1.0 paper-demo manifest:
  `docs/benchmark/v1_0/paper_demo/manifest.json`
- v1.0 paper-demo artifact id:
  `demo_report-e6fc06c328eed245`

Operator upload checklist:

1. Build with `scripts/build-codelewm-final-paper`.
2. Upload `docs/papers/codelewm_final_arxiv_source.tar.gz` through the
   arXiv author account.
3. Confirm that arXiv's TeX build matches
   `docs/papers/codelewm_final_paper.pdf`.
4. Commit the assigned arXiv URL back into this file,
   `docs/PROJECT_EXPLAINER.md`, and the relevant roadmap.

arXiv URL: pending operator upload.
