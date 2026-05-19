# Next Goal Prompt

Use this prompt for the next implementation run.

```text
Continue CodeLeWM from the current main branch and execute the full-completion
roadmap one issue at a time.

Start with #118: data: document and gate public source acquisition.

Before editing, read AGENTS.md, SPEC.md, docs/roadmap/FULL_COMPLETION.md,
docs/roadmap/IMPLEMENTATION.md, CONTRIBUTING.md, the relevant docs/spec file,
and the relevant docs/rfcs file.

Work sequentially: one issue, one branch, one PR. The dependency, dataset,
training, evaluation, index, and first-results smoke foundations are now landed;
do not add unsupported claims while moving toward a scaled public-safe shard.

For each issue:
1. Re-read the issue body and linked spec/RFC.
2. Inspect the current code and tests before editing.
3. Implement the smallest complete change that satisfies the acceptance
   criteria.
4. Add or update focused tests and docs for public behavior.
5. Run the strongest relevant validation locally.
6. Commit, push, open one PR, and include problem, solution, validation,
   caveats, and `Closes #<issue>`.
7. Return to main after the PR is merged, pull latest main, and continue with the
   next issue in docs/roadmap/FULL_COMPLETION.md.

The next milestone is scaled research evidence: a bounded, documented,
public-safe dataset path whose manifests, license gates, and source acquisition
reports can support non-trivial retrieval baselines and surprise decoys.
```
