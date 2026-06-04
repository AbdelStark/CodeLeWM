# CodeLeWM v0.7 WS-D Unsaturated-Rerank Results (2026-06-04)

WS-D set out to remove the benchmark saturation that pinned v0.7 downstream
rerank lift near zero (only 3.9% of HumanEval problems had a pass/fail mix at
temp 0.2, so reranking could not move pass@1). It did remove it — and in
doing so produced a decisive, honest negative about the **model**.

## The WS-D benchmark

Built with the deterministic mutation-distractor pipeline (`#356`/`#357`/
`#358`/`#359`, `scripts/build-wsd-rerank-pack`): each problem is one
known-correct reference solution plus `pool_size-1` plausible failing
distractors derived by single-point AST mutation of the reference, labeled in
the sandbox.

| property | value |
|----------|-------|
| source | HumanEval references (`data/raw/humaneval.jsonl`) |
| kept problems | 47 |
| pool size | 6 (1 passing reference + 5 failing distractors) |
| pass/fail mix rate | **1.00** (vs 0.039 for the frontier-LLM pack) |
| test pass rate | **0.167** (= 1/pool_size; vs ~0.95 saturated) |
| generation | deterministic mutation, no LLM, no provider budget |
| baselines | leak-free (candidates shuffled; real prompt written for lexical) |

Random pass@1 is therefore ~1/6 ≈ 0.167 — genuine, well-defined headroom.

## Rerank result (v0.7 short checkpoints, 2 seeds)

pass@1 over the 47-problem pack (`codelewm eval rerank-humaneval`, 2000
bootstrap samples):

| baseline | seed 42 | seed 1729 |
|----------|---------|-----------|
| **codelewm** | **0.064** | **0.170** |
| random | 0.213 | 0.213 |
| llm_order (shuffled) | 0.149 | 0.149 |
| no_action | 0.106 | 0.128 |
| shuffled_action | 0.085 | 0.149 |
| lexical (prompt overlap) | 0.660 | 0.660 |

codelewm lift over llm_order: −8.51 pt (seed 42) / +2.13 pt (seed 1729);
over no_action: −4.26 / +4.26. Both seeds: `claim_allowed=False`, bootstrap
CIs span 0.

## What this means

1. **WS-D worked.** The benchmark is genuinely unsaturated (mix rate 1.0,
   random ≈ 1/6), so reranking now *can* move pass@1. The flat v0.7 rerank on
   the LLM pack was not only saturation.

2. **The v0.7 model reranks at ~chance.** codelewm pass@1 sits at or below
   random and shows no significant lift over llm_order / no_action. Its
   self-supervised execution-energy does **not** encode correctness finely
   enough to pick the correct solution from single-point mutants.

3. **A trivial lexical baseline (0.66) crushes it.** On a mutation benchmark
   the correct code echoes the prompt's constants and structure better than
   the mutants do, so prompt-overlap is a strong correctness signal — one the
   world-model energy fails to match, let alone beat.

## Conclusion → WS-A is the next lever

WS-D converts the earlier ambiguous "rerank is flat" into a sharp diagnosis:
even with full headroom, the SSL-only v0.7 model carries no usable correctness
signal for reranking. More data (WS-B) or a different self-supervised recipe
will not fix this directly. The indicated fix is **WS-A — an explicit
execution-outcome head** (`p_pass = σ(W·[z_code, action, z_pred_after])`
trained with BCE on the sandbox `passed` labels, RFC-0015 §WS-A3), run as a
supervised second stage on the SSL trunk so retrieval/surprise are preserved.
WS-D is now the benchmark that will measure whether WS-A delivers the lift.

### Caveats

- 47 HumanEval problems, single benchmark; MBPP-Plus WS-D not yet built.
- Mutation distractors are single-point near-misses; they make lexical a
  strong baseline, which is a property of this benchmark design (a
  complementary "semantically-diverse distractor" pack would stress execution
  understanding where lexical cannot help).
- An earlier build had two baseline leaks (reference emitted first; missing
  prompts file → spurious lexical 1.0); both fixed in #359. The codelewm
  result was unaffected (continuous energy, no ties) and is unchanged here.
