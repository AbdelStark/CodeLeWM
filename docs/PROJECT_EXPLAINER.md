# CodeLeWM: A Latent World Model for Python Code Edits

*An end-to-end technical explainer for ML researchers and engineers.*

> **Substrate-pivot status (2026-05-31).** Sections 1-8 below describe
> the v0.2 commit-edit substrate. Issues #259-#273 add a parallel
> v0.6 execution-trace substrate that keeps the JEPA architecture
> and objective registry verbatim, swapping only the data pipeline.
> The substrate-pivot rationale, motivation, and claim gates live at
> RFC-0014
> (`docs/rfcs/RFC-0014-execution-trace-world-model-substrate.md`),
> the substrate roadmap
> (`docs/roadmap/EXECUTION_TRACE_WORLD_MODEL.md`), the operator
> runbook (`docs/operations/V0_6_EXECUTION_RUN_RUNBOOK.md`), the
> benchmark template
> (`docs/benchmark/EXECUTION_V0_6_RESULTS_TEMPLATE.md`), the
> artifact-backed v0.6 results report
> (`docs/benchmark/EXECUTION_V0_6_RESULTS_2026-05-30.md`), the
> committed per-seed eval reports under `docs/benchmark/v0_6/`, and
> the two-substrate paper outline
> (`docs/papers/two_substrate_outline.md`). The current framing is
> partial positive: v0.6 passes the substrate-shape, execution-pack
> retrieval, and generated-decoy surprise gates; latent-probe,
> crash-prediction, and HumanEval / MBPP-Plus rerank utility claims
> remain unsupported.

---

## 0. TL;DR

CodeLeWM is a **Joint-Embedding Predictive Architecture (JEPA)** applied to a
domain it has not been applied to before: the discrete, structured world of
**source code edits**. Where the original LeWorldModel learns latent dynamics
of pixel observations under continuous robot actions, CodeLeWM keeps the same
encoder/predictor/SIGReg skeleton and changes the *world* to be Python code and
the *actions* to be edit instructions.

The atomic example is a one-step transition:

```text
(CodeState_before, EditAction, CodeState_after)
```

The model learns to predict the latent of `CodeState_after` from
`(CodeState_before, EditAction)` without ever decoding back into tokens. The
downstream artifact is **not a code generator** — it is a **scorer/reranker**
for candidate patches.

The narrow scientific question the project is set up to answer:

> Can a compact JEPA-style latent transition model learn action-conditioned
> structure over Python edit trajectories that is informative enough to
> retrieve true after-states from hard negatives, beat lexical and
> action-corrupted baselines, and rank candidate patches?

A *positive* answer validates the JEPA paradigm as a representation learner for
discrete structured domains beyond perception. A *negative* answer, captured by
explicit kill criteria and collapse diagnostics, falsifies it for this regime
and forces a different objective.

---

## 1. Why This Project Exists

### 1.1 The unexamined gap in JEPA

JEPA / LeCun-style world models claim that **predicting future latents** —
rather than future pixels, tokens, or trajectories — is a sufficient signal for
learning useful representations of a domain. The strongest evidence for that
claim lives in perception: I-JEPA, V-JEPA, LeWorldModel, MC-JEPA. All operate
on dense continuous observations with continuous or near-continuous actions.

Three properties of those domains are quietly load-bearing:

1. **Continuity** — neighboring states have near-equal latents.
2. **Dense self-supervision** — every pixel/patch contributes signal.
3. **Smooth action manifold** — small action changes produce small state changes.

Source code violates all three.

- Two valid Python files that differ by one token can be semantically opposite
  (`if x:` vs `if not x:`).
- Tokens are sparse, discrete, and many are stylistic noise.
- Edits are not continuous — an edit instruction is a natural-language
  utterance or a discrete operation.

So the JEPA assumption — *predict the next latent, regularize against collapse,
get useful representations* — has never been stress-tested in a domain that
breaks its perception-friendly assumptions. CodeLeWM is that stress test.

### 1.2 Why a scorer, not a generator

The agent ecosystem already has fluent patch *generators* (LLM coding agents,
codemods, search-replace tools, refactor IDEs). What it lacks is a cheap,
**local, trust-checkable** way to ask "of these N candidate patches for this
requested edit, which one is most consistent with how this kind of edit usually
happens?"

Framing the project as a scorer:

- **Isolates the scientific claim.** Retrieval/scoring metrics are mechanical;
  generation quality is not.
- **Avoids the "did the LLM solve it" confound.** Candidate quality is
  controlled by the caller.
- **Permits non-execution.** The harness never imports or runs candidate code.
- **Has a real product surface.** Patch reranking is a concrete use case for
  CI bots, refactor tools, and agent loops.

### 1.3 What "useful representation" means here, operationally

We do not measure usefulness in the abstract. We bind the question to four
falsifiable measurements:

| Question | Metric |
| --- | --- |
| Can the latent rank the true after-state above hard negatives? | `Recall@k`, MRR on hard-1k |
| Is the action signal real, not stylistic leakage? | text-action vs shuffled-action retrieval ratio |
| Does the representation collapse? | `effective_rank_ratio`, per-dim variance, pairwise cosine, NN entropy |
| Can the model distinguish a true after-state from plausible decoys? | patch-surprise AUC |

If any of these fail, the JEPA-for-code hypothesis fails *for this
configuration*, and the kill-report output makes that visible rather than
silent.

---

## 2. The Paradigm: JEPA Transferred to Code

### 2.1 The core equation

```text
            E(·)                 A(·)
            ───>                 ───>
  state_before ────┐
                   │
                   ▼
                z_before          a
                   │              │
                   └──────┬───────┘
                          ▼
                    P(z_before, a)
                          │
                          ▼
                       z_pred_after
                          │
                          │   compare
                          ▼
                       z_after  ◄── E(state_after)
```

Symbols:

- `E` is the **CodeStateEncoder** — shared between before and after.
- `A` is the **ActionEncoder** (text or abstract).
- `P` is the **CodeLatentPredictor**.
- All vectors live in `R^256`.

Training objective:

```text
loss = MSE(P(E(s_b), A(act)), E(s_a))
     + λ_sig · SIGReg(stack(E(s_b), E(s_a), P(E(s_b), A(act))))
     [+ λ_ret · in_batch_retrieval_xent     # off by default, gated]
```

`SIGReg` (Sinkhorn-style information-geometry regularizer inherited from
LeWorldModel) controls **embedding collapse** — the failure mode where
`E(·)` learns the constant function and trivially minimizes MSE.

### 2.2 What JEPA buys us over the obvious alternatives

| Approach | Why we rejected it |
| --- | --- |
| Token-level autoregressive `s_b, act → s_a` | Defines the project as a code generator, not a representation learner. Loses the JEPA claim. |
| Supervised intent classifier over 8 fixed edit types | Cannot express realistic edits; doesn't test latent dynamics. |
| Contrastive-only (SimCLR-style) on `(s_b, s_a)` pairs | Removes the action-conditioned prediction contract — the actual claim. |
| Two-tower retrieval with no predictor | Cannot answer "what *should* the after-state look like under this action"; collapses to similarity. |
| Reward-from-tests | Mixes representation learning with execution; out of scope and a known confound. |

The JEPA path is the unique choice that:

- Predicts a target *latent* (not tokens), so it admits compact 256-d
  representations.
- Is action-conditioned, so success requires the model to use `act`.
- Has well-understood collapse failure modes with established diagnostics.
- Inherits a working implementation seed from LeWorldModel.

### 2.3 The state-after stop-gradient question (open)

A key open question (RFC-0005): should the gradient flow into `E(s_after)`?

```text
loss_pred = MSE(z_pred_after, z_after)                # gradient into both
loss_pred = MSE(z_pred_after, z_after.detach())       # target-network style
```

JEPA literature is mixed. For v0.1 the smoke runs the **non-detached** variant
under SIGReg; the detached variant is a paired ablation in the fixture
collapse-report harness. The decision is empirical and bound to a target date.

---

## 3. Representing the World: `CodeState`

A code edit happens inside a repository, but a whole repository is the wrong
unit of observation: too long, too noisy, too leaky across train/test.
`CodeState` is the **bounded, deterministic context capsule** that plays the
role of "image" in this project.

### 3.1 Pack format

```text
<LANG python>
<PATH package/module.py>
<SYMBOL package.module.Class.method>
<KIND method>
<IMPORTS>
  from typing import Iterable
  from .errors import RetryError
<ENCLOSING_CLASS>
  class HTTPClient:
      base_url: str
      timeout: float
<SIBLING_SIGNATURES>
  def __init__(self, base_url: str, timeout: float = 5.0) -> None: ...
  def get(self, path: str) -> Response: ...
<CALLEE_SIGNATURES>
  def _retry(fn, attempts: int) -> Any: ...
<PRIMARY>
  def post(self, path: str, body: dict) -> Response:
      for attempt in range(self.attempts):
          try:
              return self._send("POST", path, body)
          except RetryError:
              continue
      raise RetryError(path)
```

Sections are not free-text — they are stable, segment-ID-tagged regions:

| Segment | seg_id |
| --- | ---: |
| path/module/symbol markers | 1 |
| imports | 2 |
| class/kind/enclosing-class markers | 3 |
| sibling signatures | 4 |
| callee signatures | 5 |
| primary code | 6 |

### 3.2 Structured truncation, never tail clipping

Code is not text — clipping a function at token 1024 destroys the very signature
or `return` we need to score the edit. Truncation follows a **fixed priority
order**:

```text
Drop order (highest priority first = dropped LAST):

  6. primary code — signature, decorators
  5. primary code — changed-hunk neighborhood
  4. primary code — return/raise/exception handlers
  3. primary code — long literals (replace with typed placeholders)
  2. primary code — docstrings/comments (unless doc-edit)
  1. callee signatures (lower-priority first)
  0. sibling signatures (lower-priority first)
```

If a row still doesn't fit after all reductions, **it fails** rather than being
silently clipped. This is a deliberate choice: silent clipping would
contaminate evaluation with rows whose context is structurally incomplete.

### 3.3 Tokenized layout

Each `CodeState` becomes four aligned int/bool tensors of length 1024:

```text
input_ids        : int32  [1024]   subword token ids
attention_mask   : bool   [1024]   1 = real token
segment_ids      : int16  [1024]   1..6 per the table above
changed_hunk_mask: bool   [1024]   1 = token belongs to a changed primary line
```

`changed_hunk_mask` is the most subtle: it gives the encoder an explicit hint
about *where the edit happened*, which becomes the bridge between state and
action.

---

## 4. Representing the Action: Three Views

Code edits have no natural continuous control vector. CodeLeWM uses **three
views** with strict roles:

```text
┌──────────────────┬────────────────────────────────────────────────────┬───────────────────────────┐
│ View             │ Source                                             │ Role                      │
├──────────────────┼────────────────────────────────────────────────────┼───────────────────────────┤
│ action_text      │ commit message, instruction, synthetic template    │ HEADLINE inference        │
│ action_abstract  │ AST/CST diff → operation/node/path/size tokens     │ Structural ablation       │
│ action_patch     │ unified diff / changed spans                       │ DIAGNOSTIC ONLY (leaky)   │
└──────────────────┴────────────────────────────────────────────────────┴───────────────────────────┘
```

### 4.1 Text actions (headline)

The realistic inference path. A user (or agent) says:

> "add timeout handling to the retry loop"

This is what scoring and reranking consume in production. It is also the
noisiest signal — many commit messages are uninformative (`fix bug`, `update
deps`). That noise is part of the test: if the model still ranks correctly
under noisy text, the representation is doing real work.

### 4.2 Abstract actions (ablation)

A deterministic, structural encoding derived from AST diff:

```text
OP_UPDATE NODE_Return PATH_DEPTH_4 OLD_Call NEW_CallWithKeyword SIZE_SMALL
OP_INSERT NODE_ExceptHandler PATH_DEPTH_2 SIZE_MEDIUM
OP_DELETE NODE_Assign PATH_DEPTH_3 SIZE_SMALL
```

**Critical constraint:** abstract actions must **not contain inserted
after-code text**. They describe the *shape* of the edit, not its content.
This view exists to ask: "is the predictor learning anything beyond raw text
overlap?"

### 4.3 Patch actions (forbidden for headline)

Unified-diff style. Includes the literal after-text. Used only to compute a
*diagnostic upper bound*: how well could we score if the answer were partly in
the action? Any headline report that uses `action_patch` is **rejected by the
evaluation policy**.

### 4.4 Encoder shapes

```text
TextActionEncoder:     transformer, 4 layers, d=256, h=8, max_len=256  → R^256
AbstractActionEncoder: transformer, 3 layers, d=256, h=8, max_len=192  → R^256
PatchActionEncoder:    optional,    max_len=512                        → R^256
```

All three project to the same 256-d action manifold so the predictor `P` does
not care which view it received.

---

## 5. The Model in Detail

### 5.1 Layout

```text
                ┌────────────────────────────────────────┐
                │              CodeTransitionModel        │
                │                                         │
   state_before │   ┌──────────────────────────┐          │
   ───────────► │   │   CodeStateEncoder (E)   │ ──► z_b  │
                │   └──────────────────────────┘          │
                │                                         │
   action       │   ┌──────────────────────────┐          │
   ───────────► │   │ Text / Abstract Action   │ ──► a    │
                │   │   Encoder (A)            │          │
                │   └──────────────────────────┘          │
                │                                         │
                │   ┌──────────────────────────┐          │
                │   │  CodeLatentPredictor (P) │          │
                │   │     P(z_b, a)            │ ──► z_p  │
                │   └──────────────────────────┘          │
                │                                         │
                │   ┌──────────────────────────┐          │
                │   │   Projection heads       │          │
                │   │   projector, pred_proj   │          │
                │   └──────────────────────────┘          │
                │                                         │
   state_after  │   ┌──────────────────────────┐          │
   ───────────► │   │   CodeStateEncoder (E)   │ ──► z_a  │
                │   │   (weights shared with   │          │
                │   │    above)                │          │
                │   └──────────────────────────┘          │
                └────────────────────────────────────────┘
                            │
                            ▼
                  transition_energy = ||z_p - z_a||²
```

### 5.2 Tensor contracts

```text
state input_ids        :  int64  [B, 1024]
state attention_mask   :  bool   [B, 1024]
state segment_ids      :  int64  [B, 1024]
state changed_hunk_mask:  bool   [B, 1024]

text action input_ids  :  int64  [B, 256]
text action mask       :  bool   [B, 256]

abstract input_ids     :  int64  [B, 192]
abstract mask          :  bool   [B, 192]

z_before, z_after, z_p :  float  [B, 256]
action_emb             :  float  [B, 256]
```

`history_size=1, num_preds=1` for v0.1. Multi-step trajectories are deliberately
out of scope until single-step retrieval works.

### 5.3 Why pooled (not sequence-to-sequence) latents

The predictor sees a single pooled `[B, 256]` rather than the per-token encoder
output. This matters:

- It forces `E` to do real summarization work — the predictor cannot "cheat"
  by attending to specific after-positions.
- It keeps `z` small enough to be a usable retrieval key.
- It mirrors LeWM's pooled observation latent, preserving the implementation
  seed.

The cost is real: long-range structural cues compete with local signature
information in the same 256-d bottleneck. This is one place we expect to learn
something about JEPA-in-code regardless of the headline result.

### 5.4 Transition energy

```text
energy(s_b, act, s_a) = || P(E(s_b), A(act)) − E(s_a) ||²
```

This single scalar is the **only** consumer-facing model output. Everything
downstream — retrieval ranking, scoring, reranking — is derived from it.

---

## 6. Dataset Engineering

The dataset is where most JEPA-style projects quietly die. Code makes it
worse: leakage is everywhere (forks, near-duplicates, vendored libraries,
generated files), and a single contaminated row can invalidate a Recall
metric.

### 6.1 End-to-end build pipeline

```text
                  ┌──────────────────────────────────────────────┐
                  │             raw edit sources                  │
                  │  CommitPackFT shards │ permissive Python repos│
                  │  synthetic transforms│ (later) AgentPack      │
                  └──────────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │   source adapter → RawEditRecord       │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ parser filter   │ license filter        │
                  │   (Python AST)  │   (permissive only)   │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │       changed-symbol extractor          │
                  │   (function/method/class/region)        │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │       CodeState builder                 │
                  │   (pack format, structured truncation)  │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │       action extractor                  │
                  │  text │ abstract (AST diff) │ patch     │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │  dedup  →  near-dup SimHash, diff-shape │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │   split assigner   (repo-level, hashed) │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ transitions.jsonl + filter/dedup reports│
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ Parquet staging shards (train/val/test) │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ HDF5 packs   train.hdf5 / val.hdf5 /    │
                  │              test.hdf5                  │
                  └────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ dataset_manifest.json + checksums       │
                  │ + license-gate report                   │
                  └────────────────────────────────────────┘
```

Every arrow either emits a manifest entry or a structured drop record. **Silent
row drops are forbidden** — they would make leakage and bias invisible.

### 6.2 Filters

Keep only when **all** of:

- language is Python; old and new path end `.py`
- `ast.parse(before)` and `ast.parse(after)` both succeed
- before and after non-empty
- changed lines ∈ [1, 150]
- multi-file commit: ≤ 5 changed files
- primary state ≤ 1024 state tokens *after* structured truncation
- edit distance ratio ∈ [0.02, 0.60]
- action text length ∈ [8, 512] characters
- license is permissive (MIT, Apache-2.0, BSD-{2,3}-Clause, …)

Drop when **any** of:

- revert / WIP / vendored / generated / migration / lockfile / dependency-bump
  signal
- whitespace-only
- comment/docstring-only (unless it's a documentation edit)
- huge literal/table changes
- syntax-invalid after-state

Every drop produces a `(row_id, reason_code, message, details)` record.

### 6.3 Split policy (the leakage you cannot afford)

```text
split_key = normalized_repo_name (or source_identity if no repo)
split     = bucket(sha256(seed || split_key))     # 80/10/10
```

Repo-level splits — *not* row-level — because:

- two functions in the same repo share style, helpers, and contributors
- forks are common; row-level random splits put forks of the same commit on
  both sides
- a single popular library appearing in train and test inflates Recall

Synthetic transforms inherit their source file's split *before* generation, so
synthetic-augmented training never leaks across the split boundary.

### 6.4 Deduplication

Four keys, each one a separate kind of leak:

```text
exact_norm    : sha256(normalized before || action_text || after)
exact_pair    : sha256(normalized before || after)
simhash       : 64-bit SimHash over state_before, hamming threshold = 3
diff_shape    : (operation histogram bucket, size bucket)
```

Validation and test rows are rejected if `simhash` Hamming distance to any
training row is ≤ 3.

### 6.5 Synthetic transforms (the v0.1 controlled experiment)

Three intentionally tiny rewrites used to bootstrap the pipeline and provide
a *known-difficult-but-learnable* signal:

| Transform | Description |
| --- | --- |
| `RENAME_VALUE_TO_RESULT` | Rename function parameter `value → result` if `result` is unused. |
| `ADD_EXPLICIT_RETURN_NONE` | Add `return None` to functions that fall off implicitly. |
| `SET_LITERAL_FOR_SET_OF_CONSTANTS` | Replace `set([1, 2, 3])` with `{1, 2, 3}`. |

Each synthetic row stores `synthetic_transform_id`, `version`, and `source_digest`.
These transforms exist because:

- they parse before and after
- they have unambiguous text actions
- they let us assert a pass gate: `Recall@5 ≥ 0.40` on the synthetic hard pool.
  If the model cannot do *that*, nothing else matters.

### 6.6 HDF5 layout (what training reads)

```text
/state_before/input_ids         int32 [N, 1024]
/state_before/attention_mask    bool  [N, 1024]
/state_before/segment_ids       int16 [N, 1024]
/state_before/changed_hunk_mask bool  [N, 1024]
/state_after/input_ids          int32 [N, 1024]
/state_after/attention_mask     bool  [N, 1024]
/state_after/segment_ids        int16 [N, 1024]
/action_text/input_ids          int32 [N, 256]
/action_text/attention_mask     bool  [N, 256]
/action_abs/input_ids           int32 [N, 192]
/action_abs/attention_mask      bool  [N, 192]
/action_patch/input_ids         int32 [N, 512]    (optional)
/action_patch/attention_mask    bool  [N, 512]
/metadata/repo, path, commit, source, split, edit_size,
          token_count_before, token_count_after
```

Root attributes: `schema_version`, `features.action_patch`, `row_count`.

### 6.7 Dataset scale targets

```text
v0.1 smoke   : ≥ 40k train, ≥ 5k val, repo-level split, parse-valid
v1.0 final   : 250–350k train, 20k val, 20k test, mixed real+synthetic
```

---

## 7. The Training Pipeline

### 7.1 Config-driven, manifest-backed

A run is fully described by a `TrainConfig` YAML/JSON. Example:

```yaml
seed: 1337
data:
  train: data/codelewm_v0_1/hdf5/train.hdf5
  val:   data/codelewm_v0_1/hdf5/val.hdf5
wm:
  history_size: 1
  num_preds: 1
  embed_dim: 256
  action_view: text          # text | abstract | patch (patch = ablation only)
trainer:
  max_steps: 10000
  accelerator: auto
  devices: 1
  precision: bf16-mixed
loss:
  sigreg_weight: 0.09
  retrieval_weight: 0.0      # off by default; gated by RFC-0005
```

The runner:

1. Loads and validates the config.
2. Verifies the parent dataset artifact manifest (checksums + lineage).
3. Selects executor: `cpu-smoke` or package-native `torch`.
4. Trains.
5. Writes:
   - `manifest.json` (artifact manifest, schema `codelewm.artifact_manifest.v1`)
   - `training_manifest.json` (schema `codelewm.training_run.v1`)
   - `metrics.jsonl` (schema `codelewm.training_metrics.v1`)
   - `checkpoints/checkpoint.pt` + paired `checkpoint.pt.manifest.json`
   - `reports/metrics_report.json`, `reports/torch_training_report.json`

### 7.2 Determinism & reproducibility

```text
seed Python, NumPy, PyTorch          ✓
seed DataLoader generators           ✓
log nondeterministic backends used   ✓
record git SHA, config hash, seed    ✓ (in training_manifest.json)
record dataset manifest IDs           ✓ (in parent_artifacts[])
```

Resume only loads checkpoints whose manifest:

- matches schema version
- matches model config hash
- matches parent dataset artifact ID

Otherwise the runner raises `CheckpointCompatibilityError`.

### 7.3 Inside a training step

```text
for batch in dataloader:                           # B = batch_size
    z_b   = E(batch.state_before)                  # [B, 256]
    z_a   = E(batch.state_after)                   # [B, 256]
    a     = A(batch.action_text)                   # [B, 256]
    z_p   = P(z_b, a)                              # [B, 256]

    loss_pred = mse(z_p, z_a)
    loss_sig  = sigreg(stack(z_b, z_a, z_p))
    loss      = loss_pred + λ_sig · loss_sig

    if cfg.loss.retrieval_weight > 0:              # gated
        S = cos(z_p[:, None, :], z_a[None, :, :]) / τ
        loss_ret = ce(S, arange(B))
        loss = loss + λ_ret · loss_ret

    loss.backward()
    optimizer.step()
```

### 7.4 Validation loop and collapse diagnostics

Every validation interval the runner computes a `CollapseReport`:

```text
CollapseReport:
  effective_rank             # of E embeddings (entropy of normalized singular values)
  effective_rank_ratio       # / max possible rank
  per_dim_variance_min       # smallest var across dims
  per_dim_variance_median
  pairwise_cosine_mean       # in-batch mean(cos(z_i, z_j))
  embedding_norm_mean
  nearest_neighbor_entropy   # NN distribution entropy in eval pool
```

Kill thresholds (run is halted and a `KillReport` written):

```text
effective_rank_ratio < 0.20
median per-dim variance ≈ 0 for two consecutive eval windows
NN entropy collapses below the fixture baseline
NaN/inf in loss, embeddings, or grads
```

These are intentionally conservative. The system prefers a stopped run with a
kill report over a "successful" run with a useless representation — that is
the entire point of building collapse diagnostics in.

### 7.5 Hyperparameter sweep policy

```text
sigreg_weight    : {0.05, 0.09, 0.15}
retrieval_weight : 0.00 for base; 0.05 only after base diagnostics pass
state-after grad : {flow, detach}   (RFC-0005 open question)
```

### 7.6 End-to-end CLI

```bash
# 1. Build the transition dataset (JSONL + reports + manifest)
codelewm dataset build \
  --config config/data/commitpackft.yaml \
  --out data/codelewm_v0_1

# 2. Pack into Parquet + HDF5
codelewm dataset pack \
  --manifest data/codelewm_v0_1/manifest.json \
  --out data/codelewm_v0_1

# 3. Train
codelewm train \
  --config config/train/codelewm_tiny.yaml \
  --out runs/v0_1 \
  --executor torch \
  --device cuda

# 4. Verify lineage of every artifact written
codelewm manifest verify --manifest runs/v0_1/manifest.json
```

---

## 8. The Evaluation Protocol

Evaluation is the falsification surface. It is built before training is built,
because the entire project is meaningless if we cannot tell a real result from
a hallucinated one.

### 8.1 Primary task — action-conditioned after-state retrieval

```text
query     : (state_before, action_text)
target    : true state_after
candidates: target ∪ hard_negatives
score(c)  : -|| P(E(state_before), A(action_text)) − E(c) ||²    (higher = better)
```

Reported per pool:

```text
Recall@1, Recall@5, Recall@10, MRR, median_rank
```

### 8.2 Candidate pools

```text
easy-1k     :  random held-out after-states
hard-1k     :  same source + same edit-size bucket + same weak action cluster
hard-10k    :  v1.0 extension
repo-heldout:  only unseen repos
```

`hard-1k` is the metric that actually matters. `easy-1k` is mostly sanity.

### 8.3 Required baselines (no headline report without all five)

```text
random              : random ranking → upper bound on chance
lexical (BM25/TFIDF): textual overlap of action + before vs candidate after
no-action           : model trained/evaluated with action_emb = 0
shuffled-action     : evaluate with action_emb from a different row in batch
abstract-action     : same model architecture, action_view = abstract
```

Diagnostic upper bound (optional, never headline):

```text
patch-action        : forbidden for headline by ActionViewReportPolicy
```

### 8.4 The pass/fail gates

v0.1:

```text
synthetic hard-pool text-action Recall@5 ≥ 0.40
shuffled-action retrieval ≥ 2× worse than text-action on the same pool
no NaN/OOM/collapse during smoke training
```

v1.0:

```text
held-out real-data text-action Recall@5 ≥ 0.12 on hard-1k
                  OR a documented kill report
text-action beats no-action and shuffled-action by ≥ 2×
patch-surprise pairwise AUC ≥ 0.70
all artifacts pass manifest verify + secret-scan
```

### 8.5 Secondary task — patch surprise

For each `(s_b, act, s_a_true)`, build decoys:

```text
random_after        :  some other commit's after-state
same_file_wrong     :  a different version of the same file
same_cluster_wrong  :  same abstract-action cluster, different row
mutated             :  small AST mutation of s_a_true
syntax_bug_inject   :  fixture with a known semantic bug
```

Compute energy on each, report:

- pairwise AUC: P(energy(true) < energy(decoy))
- mean rank of true among decoys

Patch surprise is the "is the world model actually a *world* model?" test.
Retrieval tests pose `s_a_true` against random other-domain answers; patch
surprise poses it against *plausible* alternatives.

### 8.6 Eval CLI

```bash
codelewm eval retrieval \
  --checkpoint runs/v0_1/checkpoints/checkpoint.pt \
  --data       runs/v0_1_pack \
  --out        reports/v0_1/retrieval

codelewm eval surprise \
  --checkpoint runs/v0_1/checkpoints/checkpoint.pt \
  --data       runs/v0_1_pack \
  --out        reports/v0_1/surprise
```

Both commands write artifact-manifest-backed reports that include:

- the metric numbers
- the relevant candidate or decoy policy
- required retrieval baselines or surprise category caveats
- action-view metadata for the checkpoint being evaluated
- input checksums, model id, checkpoint sha256, dataset manifest id

---

## 9. Downstream Tasks and Harness Integration

The eval gates prove the representation has signal. The harness turns that
signal into something a coding agent or CI bot can call.

### 9.1 Two products: `score` and `rerank`

```text
codelewm score
    INPUT:   before.py, instruction, ONE candidate after.py
    OUTPUT:  one ScoreResult{ transition_energy, final_score, … }

codelewm rerank
    INPUT:   before.py, instruction, MANY candidates (dir of after-files or .patch)
    OUTPUT:  RerankResult{ results: sorted [ScoreResult | ErrorReport] }
```

### 9.2 What scoring does, step by step

```text
load_scorer(checkpoint):
   1. require paired checkpoint.pt.manifest.json (or --allow-unsafe-checkpoint)
   2. verify sha256(checkpoint.pt) matches manifest
   3. load model into eval mode, fixed device/dtype
   4. record (model_id, checkpoint_sha256) for ScoreResult lineage

score(before, instruction, candidate):
   parse-check candidate                    # non-execution: NEVER import/run
   build CodeState(before), CodeState(candidate)
   tokenize instruction → action_text batch

   z_b      = E(CodeState(before))
   a        = A(action_text)
   z_p      = P(z_b, a)
   z_c      = E(CodeState(candidate))

   transition_energy = || z_p − z_c ||²
   final_score       = transition_energy
                     + α · retrieval_prior        # default α = 0
                     + β · risk_penalty           # v0.1: β = 0
   return ScoreResult{...}
```

`α` defaults to zero, so scoring remains pure transition energy unless the user
explicitly passes `--index` and `--retrieval-prior-weight`. `β` remains zero in
v0.1 until risk metrics are validated.

### 9.3 ScoreResult

```python
@dataclass(frozen=True)
class ScoreResult:
    schema_version: str
    candidate: str
    transition_energy: float
    retrieval_prior: float | None
    risk_penalty: float | None
    final_score: float
    model_id: str
    checkpoint_sha256: str
    input_digest: str
    warnings: tuple[str, ...]
```

Every field is a load-bearing piece of provenance: `model_id` +
`checkpoint_sha256` + `input_digest` together let any downstream consumer
verify a score came from the model it claims to come from, evaluating exactly
the inputs claimed.

### 9.4 The rerank flow

```text
                 ┌────────────────────────────────────────┐
                 │           RerankRequest                 │
                 │   before, action_text, candidates[]     │
                 └────────────────────────────────────────┘
                                  │
              ┌───────────────────┴──────────────────┐
              ▼                                      ▼
    ┌────────────────────┐                ┌──────────────────────┐
    │ candidate is .py   │                │ candidate is .patch  │
    │ → load as after    │                │ → dry-run apply to   │
    │                    │                │   before IN MEMORY   │
    │                    │                │ → result is after    │
    └────────────────────┘                └──────────────────────┘
              │                                      │
              └───────────────────┬──────────────────┘
                                  ▼
                  ┌──────────────────────────────┐
                  │   parse-check the after-state │
                  │   fail → ErrorReport          │
                  └──────────────────────────────┘
                                  │
                                  ▼
                  ┌──────────────────────────────┐
                  │   score(before, instr, after) │
                  └──────────────────────────────┘
                                  │
                                  ▼
                  ┌──────────────────────────────┐
                  │  sort valid asc final_score   │
                  │  append ErrorReports at end   │
                  └──────────────────────────────┘
```

Non-execution invariants the harness must uphold:

- **Never `import` or `exec` candidate code.** Parse only.
- **Never modify the user's working tree.** All patch application is
  in-memory.
- **Never load a checkpoint without its paired manifest** unless the user
  explicitly opted in with `--allow-unsafe-checkpoint`.

### 9.5 Integration in a coding agent loop

```text
                 ┌──────────────────────────────────────┐
                 │  Agent / IDE / CI receives request    │
                 │  "modify X to do Y"                  │
                 └──────────────────────────────────────┘
                                  │
              ┌───────────────────┴──────────────────────┐
              ▼                  ▼                       ▼
    ┌──────────────┐   ┌──────────────────┐    ┌──────────────────┐
    │ LLM coding   │   │ Codemod /        │    │ Search-and-      │
    │ agent emits  │   │ refactor tool    │    │ replace patches  │
    │ N patches    │   │ emits N patches  │    │                  │
    └──────────────┘   └──────────────────┘    └──────────────────┘
              │                  │                       │
              └──────────────────┼───────────────────────┘
                                 ▼
                ┌─────────────────────────────────────┐
                │  candidate set { c_1, …, c_N }     │
                └─────────────────────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────┐
                │  codelewm rerank                    │
                │   --before before.py                │
                │   --instruction "..."               │
                │   --candidates patches/             │
                └─────────────────────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────┐
                │  ranked [ScoreResult]               │
                └─────────────────────────────────────┘
                                 │
                ┌────────────────┼─────────────────────┐
                ▼                ▼                     ▼
        ┌──────────────┐ ┌──────────────┐   ┌──────────────────┐
        │ Agent picks  │ │ CI gates on  │   │ Human reviewer   │
        │ top-1 patch  │ │ top-1 energy │   │ reads ranked list│
        └──────────────┘ └──────────────┘   └──────────────────┘
```

The agent / tool generates; CodeLeWM **scores**. The split is the entire
product thesis.

### 9.6 Index-backed retrieval prior

`codelewm index` builds a local index over `z_after` for train-split
transitions. At score time, the harness can:

```text
neighbors        = index.knn(candidate_after_proxy, k=K)
retrieval_prior  = mean_distance(candidate_after_proxy, neighbors)
final_score      = transition_energy + α · retrieval_prior
```

This gives the harness a "have I seen after-states like this before?" signal.
Lower `final_score` remains better. The default α is `0.0`, which reports the
prior without changing ordering; non-zero weights are explicit CLI choices.

---

## 10. End-to-End Flow Diagrams

### 10.1 Train → Eval → Score, one picture

```text
   raw sources
       │
       ▼
   ┌─────────────────────────────────────────────────────┐
   │ codelewm dataset build                              │
   │   → transitions.jsonl + manifest.json + reports/    │
   └─────────────────────────────────────────────────────┘
       │
       ▼
   ┌─────────────────────────────────────────────────────┐
   │ codelewm dataset pack                               │
   │   → hdf5/{train,val,test}.hdf5                      │
   │   → parquet/{train,val,test}/                       │
   │   → dataset_manifest.json (parent=build manifest)   │
   └─────────────────────────────────────────────────────┘
       │
       ▼
   ┌─────────────────────────────────────────────────────┐
   │ codelewm train                                      │
   │   → checkpoints/checkpoint.pt (+ .manifest.json)    │
   │   → training_manifest.json + metrics.jsonl          │
   │   → KillReport on collapse / NaN / OOM              │
   └─────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────────┐
       ▼                                                  ▼
   ┌────────────────────────────┐         ┌───────────────────────────┐
   │ codelewm eval retrieval    │         │ codelewm eval surprise    │
   │   Recall@k, MRR, baselines │         │   pairwise AUC vs decoys  │
   │   hard-1k sampler report   │         │                           │
   └────────────────────────────┘         └───────────────────────────┘
                                  │
                                  ▼ (only after gates pass)
   ┌─────────────────────────────────────────────────────┐
   │ codelewm index                                      │
   │   → ANN over training z_after                       │
   └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────┐
   │ codelewm score / codelewm rerank                    │
   │   → ScoreResult[] with full lineage                 │
   └─────────────────────────────────────────────────────┘
```

### 10.2 Inference path (`score` / `rerank`)

```text
   before.py   instruction      candidate after-state(s)
       │           │                     │
       ▼           ▼                     ▼
   ┌────────┐  ┌────────┐           ┌────────┐
   │ build  │  │ tokenize│           │ build  │  (per candidate)
   │ Code-  │  │ action │            │ Code-  │
   │ State  │  └────────┘           │ State  │
   └────────┘       │                └────────┘
        │           │                     │
        ▼           ▼                     ▼
       E(·)       A(·)                  E(·)
        │           │                     │
        ▼           ▼                     ▼
       z_b ───────► P(z_b, a)            z_c
                       │                  │
                       ▼                  │
                     z_p ─────── − ───────┘
                                 │
                                 ▼
                       || z_p − z_c ||²
                                 │
                                 ▼
                       transition_energy
                                 │
                  (+ α · retrieval_prior, default α=0)
                  (+ β · risk_penalty,    v0.1: β=0)
                                 │
                                 ▼
                            final_score
```

---

## 11. Assumptions, Risks, Falsifiers

Every project has assumptions; serious projects state them and pre-commit to
how they will fail.

### 11.1 Assumptions

1. **Compositional structure exists in code edits.** Most edits decompose
   into operations whose distribution is learnable from a modest corpus
   (10^5 transitions).
2. **A 1024-token capsule captures enough context.** For function/method
   edits, local context + signatures dominate the signal.
3. **256-d latents are large enough.** Edit diversity is high but the
   *informative* axes are few.
4. **Natural-language commit messages carry usable action signal.** Even
   noisy, terse messages constrain the after-state distribution.
5. **SIGReg is sufficient to prevent collapse.** No EMA / target-network /
   stop-gradient is needed for v0.1.
6. **Repo-level splits eliminate the dominant leakage path.** Combined with
   SimHash dedup, generalization measurements are credible.

Each assumption maps to a measurable diagnostic.

### 11.2 Risks (R) and resolution paths

```text
R-1   Commit-message text is too noisy to learn from.
      → compare text vs abstract vs no-action vs shuffled-action.
      → if text doesn't beat shuffled by 2×, claim fails.

R-2   Mixed-purpose commits contaminate transitions.
      → edit-size caps, AST parse checks, edit-ratio bounds, stratified filter
        report.

R-3   LeWM's MSE+SIGReg may not transfer to discrete code tokens.
      → collapse diagnostics every val interval.
      → kill report on threshold breach.
      → retrieval-loss gate is the explicit recovery path.

R-4   The 256-d bottleneck is too tight.
      → if hard-1k Recall@5 plateaus < 0.40 on synthetic, scale dim before
        scaling data.

R-5   Hard negatives are too easy or too hard.
      → hard-negative sampler emits its own report (cluster overlap rates).
      → easy and hard reports both required; ratio examined.

R-6   Candidate code is adversarial (malicious patches).
      → non-execution invariant: parse only, never import / exec.
      → checkpoint trust requires paired manifest sha256.
      → secret-scan on all artifacts before release.

R-7   Train/test leakage via near-duplicates.
      → 64-bit SimHash, Hamming ≤ 3 → reject.
      → diff-shape dedup.
      → repo-level split.
```

### 11.3 The kill criteria (explicit falsifiers)

The project pre-commits to declaring failure when:

```text
KILL-A   v0.1 synthetic hard-pool text-action Recall@5 < 0.40
KILL-B   shuffled-action retrieval ≥ 0.5× text-action  (action not used)
KILL-C   effective_rank_ratio < 0.20 sustained two windows
KILL-D   v1.0 real text-action Recall@5 < 0.12 on hard-1k
KILL-E   patch-surprise AUC < 0.70 on real held-out
```

A failure here is not a bug to be hidden — it is a `KillReport` artifact in
the run directory, with the diagnostic numbers that produced it.

---

## 12. What This Project Validates or Invalidates

### 12.1 If the metrics pass

We will have shown:

- JEPA's "predict the next latent" objective generalizes to a domain that
  violates the perception-friendly assumptions of continuity and dense
  signal — *if* the state representation is engineered carefully (capsule,
  changed-hunk mask, structured truncation) and *if* collapse is actively
  policed.
- Action-conditioned latent prediction extracts signal from natural-language
  commit messages that lexical baselines miss.
- A 256-d code-edit representation is dense enough to drive useful
  reranking.
- The architecture seed from LeWorldModel is transferable, modulo state and
  action encoders.

### 12.2 If the metrics fail

The kill reports localize the failure:

- **Collapse-driven failure** (KILL-C) ⇒ MSE+SIGReg is insufficient for
  code; next experiment is EMA teacher or stronger SIGReg parameterization.
- **Action-insensitive failure** (KILL-B) ⇒ the predictor is ignoring the
  action; next experiment changes the action injection (FiLM, cross-attn).
- **Generalization failure on real data only** (KILL-D) ⇒ synthetic
  transforms learn codemod artifacts; the dataset mix must change.
- **Surprise failure** (KILL-E) ⇒ the model has memorized retrieval anchors
  but does not represent edit dynamics; next experiment must change either
  the predictor (richer P) or the training distribution (more diverse
  edits).

Either outcome is publishable. The structural value of the project is that
both are *legible*: every artifact is manifest-backed, every score is
lineage-tagged, every drop has a reason code, and every gate has a number.

---

## 13. Appendix: Where to Find Things

```text
SPEC.md                              top-level index
docs/spec/00-overview.md             thesis, goals, non-goals, gates
docs/spec/01-architecture.md         subsystem boundaries + invariants
docs/spec/02-public-api.md           CLI + Python API contracts
docs/spec/03-data-model.md           TransitionRecord, CodeState, HDF5 layout
docs/spec/04-error-model.md          typed errors and exit codes
docs/spec/05-observability.md        logs, metrics, manifests, redaction
docs/spec/06-security.md             licensing, trust, non-execution, secrets
docs/spec/07-testing-strategy.md     validation pyramid + release gates
docs/spec/08-performance-budget.md   training/index/inference budgets
docs/spec/09-release-and-versioning.md
docs/spec/10-glossary.md             canonical terms

docs/rfcs/RFC-0001  LeWM-compatible code transition model
docs/rfcs/RFC-0002  Edit transition dataset
docs/rfcs/RFC-0003  CodeState schema and normalization
docs/rfcs/RFC-0004  Action views and encoders
docs/rfcs/RFC-0005  Objective and collapse diagnostics
docs/rfcs/RFC-0006  Training runtime and configs
docs/rfcs/RFC-0007  Retrieval and surprise evaluation
docs/rfcs/RFC-0008  Harness scorer and reranker
docs/rfcs/RFC-0009  Observability and artifact lineage
docs/rfcs/RFC-0010  Security, licensing, trust boundaries
docs/rfcs/RFC-0011  Public API, CLI, packaging
docs/rfcs/RFC-0012  Release, CI, governance

codelewm/data/         loading, filtering, CodeState, dedup, splits, packing
codelewm/model/        tensor contracts, encoders, predictor, energy, objective
codelewm/training/     configs, runner, CPU smoke, package-native torch executor
codelewm/eval/         retrieval, surprise, collapse, baselines, candidate pools
codelewm/harness/      CLI, scorer, reranker, output schemas
codelewm/observability/ artifact manifests, JSONL log events, redaction
codelewm/security/     license policy, checkpoint trust, non-execution guards
```

---

*Status: pre-alpha. The package surface is real and tested; the first
end-to-end meaningful training result is the next milestone tracked in
`docs/roadmap/FULL_COMPLETION.md`.*
