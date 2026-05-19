# Next Goal Prompt

Use this prompt for the next implementation run.

```text
Continue CodeLeWM from the current main branch and execute the full-completion
roadmap one issue at a time.

Start with #109: build: migrate dependency management and CI to uv.

Before editing, read AGENTS.md, SPEC.md, docs/roadmap/FULL_COMPLETION.md,
docs/roadmap/IMPLEMENTATION.md, CONTRIBUTING.md, the relevant docs/spec file,
and the relevant docs/rfcs file.

Work sequentially: one issue, one branch, one PR. Do not skip #109, because the
uv dependency and CI contract is the foundation for the dataset, training,
evaluation, index, publishing, and release issues. Keep public docs direct and
evidence-backed. Do not add unsupported claims.

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

First-results is complete only when #109 through #117 are closed and a clean
checkout can build the dataset, pack it, train the package-native model, run
retrieval and surprise evaluation, build the index, verify manifests, secret
scan selected artifacts, and regenerate docs/benchmark/FIRST_RESULTS.md from
documented commands.
```
