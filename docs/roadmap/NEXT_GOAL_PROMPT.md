# Next Goal Prompt

Use this prompt for the next full-completion run. The full headless Hugging Face
and ml-intern prompt is maintained in
`docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md`.

```text
/goal Complete CodeLeWM's first scaled-result project boundary. Treat the
current artifact set as a public negative/diagnostic result unless new evidence
actually passes the action-use claim gate.

Start from the current main branch. Ground in AGENTS.md, SPEC.md,
docs/roadmap/FULL_COMPLETION.md, docs/roadmap/IMPLEMENTATION.md,
docs/operations/HF_ML_INTERN_TRAINING.md, docs/training/SCALED_TRAINING_RUNBOOK.md,
docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md,
docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md,
docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md, CONTRIBUTING.md,
the relevant docs/spec files, and the relevant docs/rfcs files.

Do not redo closed infrastructure work. Issues #109 through #126, #137,
#138, #151 through #154, and #159 are complete. The first scaled HF run proved
the systems path. The #154 action-use margin run completed and verified
downloaded artifacts, but failed the positive claim gate. The #159
margin+retrieval run also completed and verified downloaded artifacts;
it improved text-action retrieval to Recall@1 `0.597` and MRR `0.674500`, but
still lost to no-action Recall@1 `0.650` and MRR `0.708037`.

Work sequentially, one issue per branch and PR:

1. Execute the v0.2 action-use research intervention from #167 and
   docs/roadmap/V0_2_ACTION_USE_RESEARCH_PLAN.md. Start with the child issue
   that owns the concrete implementation slice.
2. Publish HF artifacts publicly after source/license, manifest, secret-scan,
   and checkpoint-trust gates pass. Do not turn public visibility into a
   positive model-quality claim unless the benchmark gate passes.

For any future HF work, orchestrate the remote job lifecycle with the hf CLI:
hf auth whoami, hf jobs run, hf jobs ps, hf jobs inspect <job-id>, hf jobs logs
<job-id>, hf jobs stats <job-id>, and hf download. Hugging Face artifacts may
be published publicly after source/license, secret-scan, manifest verification,
and checkpoint-trust checks pass.

After each issue, run the strongest relevant local validation, commit, push,
open a PR, wait for available checks, merge when clean, return to main, pull
latest main, and continue. Public docs must stay artifact-backed and must not
claim action-conditioned quality until text-action beats no-action on the
agreed headline metrics. The current completed boundary is explicitly
negative/diagnostic.
```

Launch the full HF/ml-intern recipe with:

```bash
ml-intern --max-iterations -1 "$(cat docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md)"
```
