# Next Goal Prompt

Use this prompt for the next research-planning run. The v0.2 HF execution prompt
in `docs/roadmap/HF_ML_INTERN_GOAL_PROMPT.md` is now historical context for the
completed negative v0.2 sweep.

```text
/goal Plan the next CodeLeWM research intervention after the completed v0.2
negative result. Treat the current artifact set as public negative/diagnostic
evidence.

Start from the current main branch. Ground in AGENTS.md, SPEC.md,
docs/roadmap/FULL_COMPLETION.md, docs/roadmap/IMPLEMENTATION.md,
docs/roadmap/V0_2_ACTION_USE_RESEARCH_PLAN.md,
docs/operations/HF_ML_INTERN_TRAINING.md, docs/training/SCALED_TRAINING_RUNBOOK.md,
docs/benchmark/SCALED_HF_RESULTS_2026-05-20.md,
docs/benchmark/ACTION_USE_HF_RESULTS_2026-05-20.md,
docs/benchmark/ACTION_USE_RETRIEVAL_HF_RESULTS_2026-05-20.md,
docs/benchmark/V0_2_ACTION_SWAP_HF_RESULTS_2026-05-20.md, CONTRIBUTING.md, the
relevant docs/spec files, and the relevant docs/rfcs files.

Do not redo closed infrastructure work. Issues #109 through #126, #137, #138,
#151 through #154, #159, and #168 through #172 are complete. The first scaled
HF run proved the systems path. The #154 and #159 action-use runs failed the
positive claim gate. The #172 v0.2 action-swap/inverse-action run also
completed and verified downloaded artifacts; it reached text-action Recall@1
`0.263` and MRR `0.370048`, but still lost to no-action Recall@1 `0.441` and
MRR `0.533105`. Its representation and downstream gates also failed.

Work sequentially, one issue per branch and PR:

1. Audit the v0.2 negative evidence and propose one new falsifiable research
   hypothesis that directly addresses no-action dominance or weak action
   supervision.
2. Turn that hypothesis into a new GitHub issue with acceptance criteria,
   metrics, gates, and an HF artifact plan.
3. Do not launch compute until the new issue specifies what result would count
   as success or falsification.

For any future HF work, orchestrate the remote job lifecycle with the hf CLI:
hf auth whoami, hf jobs run, hf jobs ps, hf jobs inspect <job-id>, hf jobs logs
<job-id>, hf jobs stats <job-id>, and hf download. Hugging Face artifacts may
be published publicly after source/license, secret-scan, manifest verification,
and checkpoint-trust checks pass.

After each issue, run the strongest relevant local validation, commit, push,
open a PR, wait for available checks, merge when clean, return to main, pull
latest main, and continue. Public docs must stay artifact-backed and must not
claim action-conditioned quality until text-action beats no-action on the
agreed headline and action-contrast metrics. The current completed boundary is
explicitly negative/diagnostic.
```
