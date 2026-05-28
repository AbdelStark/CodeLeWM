# Execution-Trace World Model (Pipeline A)

Last updated: 2026-05-28

Issue: TBD. Tracker: TBD. Status: specification draft; no training run launched
by this issue.

This document defines a substrate pivot for the CodeLeWM transition model. The
JEPA latent-transition architecture, training paradigm, objective terms, and
eval gates are kept intact. The *world* being modeled changes from
commit-message-conditioned code edits to **input-conditioned program
execution**.

The thesis becomes: a JEPA latent-transition model can learn meaningful
abstractions of Python semantics if and only if the training substrate carries
strong, deterministic conditioning signal. Code edits do not. Program execution
does.

## Evidence Starting Point

Four scaled HF Jobs runs on CommitPackFT have failed the headline claim gate
with the same failure reason (`no_action_dominance`):

| Run | Text-action Recall@1 | No-action Recall@1 | Delta |
|-----|---------------------:|-------------------:|------:|
| #138 scaled systems | 0.371 | 0.459 | -0.088 |
| #154 action-use margin | 0.363 | 0.469 | -0.106 |
| #159 margin+retrieval | 0.597 | 0.650 | -0.053 |
| #172 v0.2 action-swap | 0.263 | 0.441 | -0.178 |

The v0.2 run reports `effective_rank_ratio=0.015761` against a collapse
threshold of `0.20`, and mutation-decoy surprise AUC of `0.501` (chance). Every
latent probe target (`edit_class`, `ast_node_kind`, `symbol_kind`,
`edit_size_bucket`, `action_cluster`) is beaten by lexical or metadata-only
controls.

The `bugfix-edge-case` live demo run scored every generated patch *worse* than
the no-action baseline (best candidate `121.466` versus no-action `120.725`).

## Substrate Diagnosis

The CommitPackFT substrate has three compounding signal problems that the
current architecture cannot overcome by tuning objectives or adding hard
negatives:

- **`after ≈ before` is a strong Bayes-optimal prior.** Most commits are small
  edits; copying `z_before` into `z_after` is correct on most examples. The
  predictor collapses to identity. SIGReg cannot push back against the
  population gradient.
- **Commit messages are weak action signals.** Strings like `fix bug`, `wip`,
  `update` carry near-zero conditioning information. The TextActionEncoder has
  nothing distinctive to learn.
- **Edits are multi-purpose.** A single commit may rename, refactor, and fix a
  typo. There is no canonical transformation for the predictor to model.

The candidate-contrast intervention (`DIAGNOSTICS_DRIVEN_MODEL_EXPERIMENT.md`)
attacks the third symptom (hard negatives) but does not change the first two.
It is worth running, but it is unlikely to flip the no-action dominance gate.

## Intervention

Name: input-conditioned execution world model.

Hypothesis: if training transitions are defined as
`(code, input) → output` triples from deterministic Python execution, then
CodeLeWM will satisfy headline retrieval gates against the no-action baseline,
because no-action is a structurally wrong prediction in this substrate (outputs
do not live in the code-token distribution).

This is a substrate change, not an architecture change. The schema-level
mapping is a relabeling:

| Slot | Code-edit use | Execution use |
|------|---------------|---------------|
| `CodeStateEncoder` | `code_before` | `code` |
| `TextActionEncoder` | commit message | **`repr(input)` serialized value(s)** |
| `CodeLatentPredictor` | predicts `z_after` | predicts `z_output` |
| Target encoder (EMA) | embeds `code_after` | embeds `repr(output)` |
| Pack layout `(before, action, after)` | unchanged | unchanged |
| MSE + SIGReg + action-swap + inverse-action losses | unchanged | unchanged |

The four existing objective terms are arguably designed for this setting:

- **MSE prediction.** Predicting `z_output` from `(z_code, z_input)` is a true
  latent dynamics task; there is no identity shortcut.
- **SIGReg.** Output diversity in the natural data distribution provides a
  population-level anti-collapse signal that aligns with SIGReg rather than
  fighting it.
- **Action-swap contrastive.** Same `code`, different `input` → different
  `output` is automatic, abundant, hard-negative supervision. The action encoder
  must learn the input value structure or it cannot satisfy the margin.
- **Inverse-action reconstruction.** Recovering `input` from `(code, output)`
  is symbolic execution in latent space. This is the strongest possible
  "latent encodes program semantics" signal we can train against.

This intervention does not change the model graph, the objective registry, the
training executor, the manifest schema, the eval harness, or the scorer/reranker
surfaces. It changes the data pipeline and the pack contents only.

## Required Data

Inputs:

- **CodeNet (IBM, public)** — ~14M submissions across competitive-programming
  problems with judge verdicts (`Accepted`, `Wrong Answer`, `Runtime Error`).
  Python subset, MIT/Apache-licensed problems only.
- **MBPP / MBPP-Plus** — sanitized Python problems with hidden tests and
  reference solutions.
- **HumanEval / HumanEval-X (Python)** — held out for downstream eval; not used
  for training.
- **APPS** — large competitive-programming corpus with hidden test cases.
- Optional later: **LiveCodeBench**, **CodeContests**, **CodeForces2k**.

License gating reuses the existing `codelewm.security.license_policy` machinery;
all source datasets above are gated MIT/Apache/CC-BY only.

New artifact: `codelewm.data.execution_pack.v1`

Minimum fields per record:

- `schema_version`;
- `source_problem_id` and `source_submission_id`;
- `source_dataset` (`codenet|mbpp|apps|…`);
- `split` (`train|val|test`);
- `code_checksum` and `code_tokens`;
- `input_repr_checksum` and `input_repr_tokens`;
- `output_repr_checksum` and `output_repr_tokens`;
- `output_kind` (`value|stdout|exception|timeout`);
- `output_type` (`int|float|str|list|dict|bool|none|exception_class|...`);
- `execution_status` (`ok|raised|timeout|nondeterministic`);
- `wall_time_ms` and `peak_rss_kb` (from sandbox);
- `determinism_check` — same code+input executed twice, outputs must match;
- `license_decision`;
- `manifest_parent_artifacts`;
- `secret_scan`;
- `claim_boundary`.

Determinism gate: any record with `determinism_check=false`, `output_kind=timeout`,
or detected nondeterministic sources (`time`, `random` without seed, `socket`,
`os.environ`, `input()`) is dropped from the pack. The pack is a record of
deterministic, reproducible state transitions.

Execution sandbox requirements:

- isolated process, no network, no filesystem write outside scratch dir;
- CPU timeout (default 5s), memory cap (default 256MB);
- stdlib-only first cut; no third-party imports;
- canonical `repr()` of return value; truncate to 4KB; over-cap records dropped;
- reuse `codelewm.security.non_execution_guards` semantics — training inputs
  remain untrusted serialized data; the sandbox is a one-shot data builder, not
  a training-time component.

Split policy:

- partition by `source_problem_id`, not by submission;
- no problem appears in more than one split;
- HumanEval and MBPP-Plus held entirely out of train/val.

Target scale for the first run:

- ~200k–500k deterministic (code, input, output) records, Python only;
- ≥20 distinct inputs per code where available (for action-swap negatives);
- balanced across `output_type` to avoid collapse to `int`-only predictions.

## Config And HF Jobs Recipe

Planned implementation issue should add:

- data command: `codelewm dataset execution-pack`;
- sandbox runner: `codelewm dataset execute --policy stdlib-only`;
- training config: `configs/training/v0_6_execution_a10g.yaml`;
- HF launcher profile:
  `CODELEWM_HF_RUN_NAME=codelewm-v0-6-execution-<date>-<sha>`;
- artifact card updates for dataset, model, and run repositories;
- new claim boundary template explicitly scoped to "deterministic Python
  execution over stdlib-only programs."

HF orchestration uses the existing `hf` CLI workflow:

```bash
hf auth whoami
CODELEWM_HF_JOBS_DRY_RUN=1 uv run scripts/hf-launch-codelewm-job
hf jobs inspect <job-id>
hf jobs logs <job-id>
hf jobs stats <job-id>
hf download abdelstark/codelewm-runs <run-path> --local-dir <download-dir>
```

Local verification after download:

```bash
uv run codelewm manifest verify --manifest <run>/manifest.json --json
uv run codelewm secret-scan <run> --json
uv run codelewm eval retrieval ...
uv run codelewm eval ablation ...
uv run codelewm eval latent-probe ...
uv run codelewm eval latent-matrix ...
uv run codelewm eval surprise ...
uv run codelewm eval downstream-rerank ...
uv run codelewm eval scorer-quality ...
```

## Architecture Notes (What Does Not Change vs. What May)

Does not change for the first run:

- single shared encoder for code and stringified output (reuse
  `CodeStateEncoder`);
- `TextActionEncoder` reused for `repr(input)` tokenization;
- `ARPredictor` (6-layer Transformer) reused;
- EMA target encoder reused;
- objective registry, SIGReg weight (~0.09), action-swap, inverse-action losses
  reused with the same default weights as v0.2;
- training loop, manifest runner, TensorBoard export, resume logic reused.

Considered for v0.6.1 (deferred until first run reports):

- a dedicated output-encoder head when `output_type` is structured (list, dict);
- a typed output predictor that predicts `(output_type, output_latent)`
  jointly, so the model can be evaluated on type-prediction accuracy as a
  cheap downstream probe;
- multi-step trace prediction: feed the ARPredictor a sequence of statements
  with intermediate state snapshots, predict `z_{t+1}` from `z_t` and the
  next statement (V-JEPA-style temporal masking).

## Metrics And Baselines

Training diagnostics (reused, just point at the new packs):

- prediction MSE on output latent;
- SIGReg / collapse: effective rank, per-dimension variance median, nearest
  neighbor entropy, mean pairwise cosine, norm stats;
- action-swap margin satisfaction rate (same code, different input);
- inverse-action reconstruction loss;
- output-type distribution per batch (collapse-to-int detector).

Representation metrics:

- `codelewm.eval.latent_probe_report.v1` with new targets:
  - `output_type` (multi-class);
  - `will_raise` (binary);
  - `output_magnitude_bucket` (multi-class on numeric outputs);
  - `output_length_bucket` (multi-class on sequence outputs);
  - `arithmetic_vs_string_vs_collection` (multi-class).
- `codelewm.eval.latent_matrix_report.v1` for stability across seeds.
- Surprise eval with three decoy categories:
  - random output from the dataset (easy);
  - output of a *different submission* on the *same problem and same input*
    (medium — captures algorithmic correctness);
  - output of the same code on a *different input* (hard — captures
    input-sensitivity).

Downstream reranking metrics (the headline claim):

- **HumanEval pass@k reranking.** Sample N=10 completions per problem from a
  reference LLM (Claude Haiku 4.5 or equivalent), score each with CodeLeWM
  conditioned on the problem's example input, rerank, report pass@1.
- Baselines: random order, lexical similarity to prompt, LLM original order,
  no-action score.
- **MBPP-Plus pass@k reranking** under the same protocol.
- **Crash prediction** (binary classification on `will_raise` from a
  held-out submission set with judge verdicts).

Required scale:

- ≥1000 HumanEval/MBPP-Plus rerank examples (full benchmark);
- ≥2 training seeds for variance bounds;
- ≥3 LLM sampling seeds per problem.

## Claim Gates

The experiment may support a positive model-quality claim only if all of these
are true:

- **Headline retrieval gate flips.** Text-action Recall@1 and MRR on the
  execution pack must exceed no-action by ≥0.05 absolute on both metrics, on
  test split, across ≥2 seeds.
- **Collapse gate satisfied.** Effective rank ratio ≥ 0.20; per-dim variance
  median ≥ 1e-8; nearest neighbor entropy ≥ 0.10.
- **At least one latent probe target beats every control.** Specifically
  `output_type` or `will_raise` must beat lexical, metadata-only,
  random-latent, no-action, and shuffled-action baselines, across ≥2 seeds.
- **Downstream rerank lift.** CodeLeWM-reranked pass@1 on HumanEval (or
  MBPP-Plus) must exceed LLM-original-order pass@1 by ≥3 absolute points,
  across ≥3 LLM sampling seeds, with bootstrap 95% CI excluding zero.
- **Surprise eval.** Mutation-decoy AUC ≥ 0.65 (vs. 0.501 baseline on v0.2);
  same-problem-different-submission AUC ≥ 0.60.
- Checkpoint trust, artifact manifests, downloaded-artifact verification, and
  secret scans all pass.
- Claim-review report explicitly approves the wording.

If any gate fails, public claims remain limited to negative/diagnostic
evidence — same protocol as `DIAGNOSTICS_DRIVEN_MODEL_EXPERIMENT.md`.

## Expected Failure Modes

- **Output tokenization is too lossy.** Stringifying lists or dicts collapses
  structural information; the predictor cannot match. Mitigation: typed output
  head in v0.6.1.
- **Distribution skew toward `int` outputs.** Most competitive-programming
  problems return ints; the model collapses to int-prediction. Mitigation:
  balanced pack sampling per `output_type`.
- **Single-input problems dominate.** Action-swap contrastive needs ≥2
  distinct inputs per code; if most problems have only one example input, the
  loss term degenerates. Mitigation: filter for problems with ≥3 inputs in
  the test corpus; synthetically generate inputs from fuzzing where safe.
- **Determinism filter drops most data.** If too many submissions use
  `random`/`time`/`input()`, the pack is too small. Mitigation: relax to
  seeded determinism (run twice with same `PYTHONHASHSEED`), or accept stdin
  as part of the input.
- **HumanEval rerank lift exists but is below the 3-point gate.** The lift is
  real but the gate is too tight. Mitigation: a 1-point lift with tight CI is
  still publishable as a partial positive, but not under the headline claim
  language.
- **Crash prediction works, output prediction does not.** Latent has learned
  control-flow but not value semantics. Publishable as a scoped positive.
- **Latent predicts type but not value.** Probe gate passes on `output_type`,
  fails on `output_magnitude`. Publishable as a scoped positive ("CodeLeWM
  learns types, not values"). This is the most likely partial-positive shape.

Each failure mode is publishable as diagnostic evidence if manifests,
downloads, verification, and scans pass.

## Blocked Claims If The Experiment Fails

If the run fails any claim gate, do not claim:

- CodeLeWM understands program semantics;
- CodeLeWM has useful semantic latent axes;
- CodeLeWM improves code generation;
- the JEPA architecture is suited to code (only that *this substrate* did not
  work);
- output-conditioned action signal is sufficient.

Allowed language remains:

- the pipeline executed;
- the execution-pack artifact was published and verified;
- diagnostics show where the model fails on deterministic Python execution;
- the result is negative or diagnostic evidence; or
- partial-positive results scoped explicitly (e.g., "crash prediction beats
  baseline; output value prediction does not").

## Why This Is The Right Pivot

This experiment is designed so the failure modes of the current pipeline
cannot recur for structural reasons:

| Current failure | Why this pipeline does not have it |
|---|---|
| No-action dominance | Output and code live in disjoint distributions; copying `z_code` is structurally wrong |
| Commit-message noise | Inputs are typed deterministic values |
| Multi-purpose edits | Each record is single-purpose: this code, this input → this output |
| Collapse from `after ≈ before` | Outputs differ across inputs by construction; population gradient pushes against collapse |
| Probes beaten by lexical baselines | Output prediction is not solvable by surface lexical features; semantic structure is required |
| Surprise mutation AUC ≈ 0.5 | Mutated programs produce different outputs on the same input by definition |

The architecture, training loop, eval harness, manifest gates, security
controls, HF Jobs pipeline, scorer, reranker, and ~70% of the existing code
transfer untouched. The pivot is a data pipeline change, not an architecture
change.

## Paper Framing

This experiment unlocks a clean two-substrate narrative:

> The same JEPA latent-transition recipe fails on commit-message-conditioned
> code edits — where the no-action prior dominates and the latent collapses to
> ≤5 effective dimensions — and succeeds (or fails in informative ways) on
> input-conditioned program execution, where output diversity provides
> sufficient population gradient for the predictor to learn meaningful latent
> structure. The execution-conditioned latents transfer to HumanEval pass@1
> reranking with measurable lift over original LLM sampling order.

If gates pass, this is a positive ML paper. If they fail, the comparison
between the two substrates is itself a publishable diagnostic contribution on
*what kinds of code data carry sufficient signal for JEPA-style world models*.
Either outcome ships.

## Implementation Issues To Open Next

This issue only defines the experiment. A future implementation should split
into separate issues:

- **data-build:** add `codelewm dataset execute` sandbox + `codelewm dataset
  execution-pack` pack builder; license gates; determinism filter;
- **data-publish:** publish `codelewm-execution-pack-v1` HF dataset card with
  manifest, checksums, license attribution, claim boundary;
- **train:** add `configs/training/v0_6_execution_a10g.yaml`; reuse existing
  objective registry; verify SIGReg/action-swap defaults transfer;
- **eval-probes:** add `output_type`, `will_raise`, `output_magnitude_bucket`
  probe targets to `codelewm.eval.latent_probe`;
- **eval-rerank:** add `codelewm eval rerank-humaneval` and
  `codelewm eval rerank-mbpp-plus` commands with LLM-sampling adapter (reuse
  OpenRouter/BYOK plumbing from the harness demo);
- **run:** execute HF Jobs v0.6 execution experiment, verify artifacts, write
  benchmark report;
- **docs:** add `docs/benchmark/EXECUTION_V0_6_RESULTS_<date>.md` with full
  claim-gate table;
- **harness:** add an `execution-rerank` demo scenario alongside
  `bugfix-edge-case` so the LLM demo can showcase the new substrate end-to-end.
