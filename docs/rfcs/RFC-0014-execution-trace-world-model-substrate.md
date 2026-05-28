# RFC-0014: Execution-Trace World Model Substrate

- Status: Draft
- Authors: CodeLeWM maintainers
- Created: 2026-05-28
- Target milestone: v0.6

## Summary

CodeLeWM's first substrate — commit-message-conditioned code edits over
CommitPackFT Python — has produced four scaled HF Jobs runs that all fail the
headline action-conditioning retrieval gate with the same failure reason
(`no_action_dominance`). The v0.2 run reports severe latent collapse
(effective rank ratio `0.015761` against a `0.20` threshold) and chance-level
mutation-decoy surprise AUC.

This RFC proposes that the next research step is **not** another objective or
hard-negative intervention on the same substrate, and **not** an architecture
change. The next step is a substrate pivot: the same JEPA latent-transition
recipe applied to `(code, input) → output` triples from deterministic Python
execution.

The architecture, training paradigm, objective registry, eval harness,
manifests, security boundary, scorer/reranker, and HF Jobs pipeline are
reused. Only the data pipeline, sandboxed data builder, claim-boundary
scope, and downstream evaluation surface change.

## Motivation

The CommitPackFT substrate has three compounding signal problems that no
objective tuning can overcome:

- `after ≈ before` is a Bayes-optimal prior on most commits, so the
  predictor collapses to identity;
- commit messages carry near-zero action signal;
- single commits are multi-purpose, so the predictor cannot model a
  canonical transformation.

`(code, input) → output` execution triples invert all three:

- the output distribution is disjoint from the code-token distribution, so
  copying `z_code` is structurally wrong;
- the input is a typed deterministic value, providing precise action
  signal;
- each record is single-purpose and judge-verifiable.

The four objective terms already implemented (MSE, SIGReg, action-swap
contrastive, inverse-action reconstruction) are arguably designed for this
setting and were under-rewarded on the previous substrate.

## Goals

- Reuse the JEPA latent-transition architecture, training loop, and
  objective registry without modification.
- Reuse the HDF5 transition-pack layout
  (`before_tokens`, `action_tokens`, `after_tokens`) by relabeling
  contents.
- Build a sandboxed deterministic Python executor that runs licensed public
  submissions at data-build time and records `(code, input, output)`
  triples.
- Introduce `codelewm.data.execution_pack.v1`.
- Publish the pack to Hugging Face with a manifest-verified dataset card.
- Add latent probe targets specific to execution semantics
  (`output_type`, `will_raise`, `output_magnitude_bucket`,
  `output_length_bucket`, `arithmetic_vs_string_vs_collection`,
  `judge_verdict`).
- Add surprise eval decoy categories that test program semantics rather
  than surface code similarity.
- Add a headline downstream task: HumanEval / MBPP-Plus pass@k reranking
  with the existing scorer/reranker, where CodeLeWM is conditioned on the
  problem's example input.
- Add a scoped fallback task: crash prediction.
- Add an `execution-rerank` LLM demo scenario alongside `bugfix-edge-case`.
- Maintain claim-gate discipline: no positive claim without bootstrap CI
  excluding zero and ≥2 seeds.

## Non-Goals

- Claim that the v0.2 commit-edit checkpoint or the upcoming v0.6
  execution checkpoint generalizes to non-Python code.
- Execute code in training or inference paths. The sandbox is a
  data-prep-only component.
- Replace the CommitPackFT substrate; the v0.2 work remains the public
  negative-evidence record and is comparable head-to-head with the new
  substrate.
- Introduce third-party imports into the sandbox policy. Initial policy is
  stdlib-only.

## Design

The substrate change is a relabeling of the existing transition schema:

| Slot | Code-edit use (v0.2) | Execution use (v0.6) |
|------|-----------------------|----------------------|
| `CodeStateEncoder` | `code_before` | `code` |
| `TextActionEncoder` | commit message | `repr(input)` |
| `CodeLatentPredictor` | predicts `z_after` | predicts `z_output` |
| Target encoder (EMA) | embeds `code_after` | embeds `repr(output)` |
| Pack layout `(before, action, after)` | unchanged | unchanged |
| MSE + SIGReg + action-swap + inverse-action losses | unchanged | unchanged |

### Sandbox

A new module `codelewm.data.sandbox` runs untrusted Python in an isolated
subprocess with:

- a stdlib-only import allowlist (via an import hook installed in the
  subprocess bootstrap, keyed on `sys.stdlib_module_names`);
- a network deny-list and a filesystem-write audit hook
  (`sys.addaudithook`);
- CPU and memory rlimits (`RLIMIT_CPU`, `RLIMIT_AS`) on POSIX;
- a hard process timeout with kill on overrun;
- a determinism re-run that requires identical outputs across two
  executions with the same `PYTHONHASHSEED`.

The sandbox is **not** invoked at training, scoring, or evaluation time.
Its single job is to produce a manifested data artifact. The only second
use of the sandbox is the dedicated `execution-rerank` downstream
evaluation that runs hidden tests against LLM-sampled completions; that
use is operator-reviewed and bound to a specific evaluation scenario.

### Data pipeline

- `codelewm dataset ingest --source <codenet|mbpp|mbpp_plus|apps|humaneval>`
  produces uniform intermediate JSONL.
- `codelewm dataset execute --policy stdlib-only` runs the sandbox per
  `(code, input)` and produces `(code, input, output, metadata)` records.
- `codelewm dataset execution-pack` packs determinism-gated records into
  HDF5 with a manifest, attribution sidecar, license summary, and claim
  boundary.

Split policy partitions by `source_problem_id`. HumanEval and MBPP-Plus
are held out entirely for downstream evaluation.

### Training

- `configs/training/v0_6_execution_a10g.yaml` reuses the existing torch
  executor.
- Two seeds (42 and 1729) for variance bounds.
- Existing diagnostics, manifests, HF Jobs launcher, and artifact publish
  flow are unchanged.

### Evaluation

- Existing retrieval, ablation, surprise, latent-probe, latent-matrix, and
  scorer-quality reports are reused with new substrate-appropriate
  targets.
- New downstream: `codelewm eval rerank-humaneval` and
  `codelewm eval rerank-mbpp-plus`. LLM sampling reuses the OpenRouter
  adapter and BYOK plumbing.
- New scoped task: `codelewm eval crash-prediction`.

### Claim gates

A positive headline claim requires:

- text-action Recall@1 and MRR exceed no-action by ≥0.05 absolute on test
  split across ≥2 seeds;
- collapse gate satisfied (effective rank ratio ≥0.20; per-dim variance
  median ≥1e-8; nearest neighbor entropy ≥0.10);
- at least one latent probe target beats every control across ≥2 seeds;
- HumanEval or MBPP-Plus pass@1 lift over LLM-original-order ≥3 absolute
  points with bootstrap 95% CI excluding zero across ≥3 LLM sampling
  seeds;
- mutation-decoy surprise AUC ≥0.65;
  same-problem-different-submission AUC ≥0.60.

If any gate fails, public claims remain limited to negative or diagnostic
evidence. Partial-positive shapes (e.g., "the latent encodes output type
but not output value") are publishable when scoped.

## Open Questions

- Tokenization efficiency on `repr(output)`. The current tokenizer was
  trained on code; serialized outputs may tokenize inefficiently. Decision
  point at the end of the smoke training run (#264).
- Whether a dedicated output-encoder head is needed for structured outputs
  (lists, dicts). Deferred to v0.6.1 unless the smoke run mandates it.
- Whether to extend the sandbox policy beyond stdlib. Deferred until v0.6
  ships.

## Implementation

This RFC derives the implementation issues tracked under #259. The
roadmap doc lives at `docs/roadmap/EXECUTION_TRACE_WORLD_MODEL.md`.

Implementation issues, in dependency order:

- #260 — sandbox executor;
- #261 — source adapters;
- #262 — execution-pack builder;
- #263 — HF dataset publication;
- #264 — smoke training run;
- #265 — v0.6 HF Jobs run;
- #266 — new latent probe targets;
- #267 — surprise eval decoy extensions;
- #268 — HumanEval / MBPP-Plus rerank command;
- #269 — crash-prediction eval;
- #270 — `execution-rerank` demo scenario;
- #271 — demo report extensions;
- #272 — benchmark report and paper outline;
- #273 — claim-boundary template and non-execution guard update (this RFC's
  governance pre-work).

## Acceptance

This RFC is accepted when the substrate pivot is approved as the next
research direction and at least the first implementation issue (#260) is
opened against it.

## Related

- `docs/roadmap/EXECUTION_TRACE_WORLD_MODEL.md`
- `docs/roadmap/DIAGNOSTICS_DRIVEN_MODEL_EXPERIMENT.md` (parallel
  intervention on the v0.2 substrate)
- `docs/spec/06-security.md` (non-execution policy; sandbox is the named
  follow-on subsystem the policy anticipates)
- `docs/spec/11-llm-world-model-harness.md`
- RFC-0010 (security/licensing/trust boundaries)
- RFC-0013 (LLM + world-model harness)
