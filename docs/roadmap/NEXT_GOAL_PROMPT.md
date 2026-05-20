# Next Goal Prompt

Use this prompt for the next full-completion run. The full headless Hugging Face
and ml-intern prompt is maintained in
`docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md`.

```text
/goal Complete CodeLeWM's remaining path to a claim-eligible first scaled
training result and release-ready artifact set.

Start from the current main branch. Ground in AGENTS.md, SPEC.md,
docs/roadmap/FULL_COMPLETION.md, docs/roadmap/IMPLEMENTATION.md,
docs/operations/HF_ML_INTERN_TRAINING.md, docs/training/SCALED_TRAINING_RUNBOOK.md,
CONTRIBUTING.md, the relevant docs/spec files, and the relevant docs/rfcs files.

Do not redo closed HF infrastructure work. Issues #109 through #122, #137, and
#138 are complete. The first scaled HF run proved the systems path but failed
the positive action-conditioned quality gate because text-action lost to the
no-action baseline.

Work sequentially, one issue per branch and PR:

1. #151 eval: add no-action dominance diagnostics and claim gates.
2. #152 data: add action-discriminative shard diagnostics and hard negatives.
3. #153 train: add action-use objective and scaled sweep configs.
4. #154 run: execute follow-up HF Jobs action-use training and verify artifacts.
5. #123 release: add uv build and package publishing gates.
6. #124 release: add dependency audit and provenance evidence.
7. #125 docs: refresh public docs against first-results evidence.
8. #126 release: run final artifact freeze and checklist.

For HF work, orchestrate the remote job lifecycle with the hf CLI: hf auth
whoami, hf jobs run, hf jobs ps, hf jobs inspect <job-id>, hf jobs logs
<job-id>, hf jobs stats <job-id>, and hf download. Keep Hugging Face
repositories private until the claim gate, release gates, secret scans, manifest
verification, and checkpoint-trust checks all pass.

After each issue, run the strongest relevant local validation, commit, push,
open a PR, wait for available checks, merge when clean, return to main, pull
latest main, and continue. Public docs must stay artifact-backed and must not
claim action-conditioned quality until text-action beats no-action on the agreed
headline metrics or the release is explicitly framed as negative/diagnostic.
```

Launch the full HF/ml-intern recipe with:

```bash
ml-intern --max-iterations -1 "$(cat docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md)"
```
