# v0.6 Semantic Decoy Pack

Date: 2026-06-01

Issue: #321

This is a count-gate artifact for the strengthened same-problem semantic
decoy pack. It does not evaluate either v0.6 checkpoint. The #322 rerun at
`docs/benchmark/SEMANTIC_DECOY_SURPRISE_2026-06-01.md` consumes this pack and
keeps the model-level semantic surprise claim blocked because the narrow
same-problem/different-submission category remains small-n.

## Artifact

Manifest: `docs/benchmark/v0_6/semantic_decoy_pack/manifest.json`

Artifact ID: `downstream_benchmark-f3e0dc65fbf18825`

Command:

```bash
uv run codelewm eval semantic-decoy-pack \
  --pack .artifacts/v0_6/execution-pack \
  --out docs/benchmark/v0_6/semantic_decoy_pack \
  --max-pairs-per-query 3 \
  --min-pairs-for-claim 100 \
  --min-distinct-problems-for-claim 30 \
  --json
```

Expected output schema:

- `codelewm.eval.semantic_decoy_pack_run.v1` on stdout.
- `codelewm.eval.semantic_decoy_pack.v1` in artifact-manifest metadata.
- `codelewm.eval.semantic_decoy_pair.v1` for pair rows.
- `codelewm.eval.semantic_decoy_summary.v1` for the summary report.

## Count Gate

| Metric | Value | Gate | Result |
| --- | ---: | ---: | --- |
| Same-problem semantic pairs | 358 | >= 100 | PASS |
| Distinct problems | 68 | >= 30 | PASS |
| Same-problem/different-submission pairs | 6 | diagnostic | small-n |
| Same-code/different-input pairs | 352 | diagnostic | available |

The strengthened pack treats both categories as same-problem semantic
adversarial decoys, but keeps their control labels separate:

- `same_problem_different_submission`: same problem, different submission,
  differing output. This remains the narrowest category and still has only
  six pairs in the current public v0.6 pack.
- `same_code_different_input`: same problem and same submission, different
  input, differing output. This is the stronger input-sensitivity semantic
  category that raises the pack-level count gate above threshold.

## Boundary

This artifact only says that the decoy source is now large enough to support
a meaningful rerun. It does not say that CodeLeWM separates these decoys. The
per-seed #322 reruns separate score gates from pair-count gates and keep the
broad semantic surprise claim closed while recording the positive diagnostic
AUCs.
