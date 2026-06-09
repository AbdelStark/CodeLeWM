# HumanEval WS-D Shuffled-Action Control Analysis (2026-06-09)

This note adds the lift of CodeLeWM over the **shuffled-action** control on the
two open HumanEval WS-D slices, including bootstrap confidence intervals that are
not serialized in the released v0.9 rerank reports. It exists because
shuffled-action is the strongest non-trivial control on exactly the slices the
final paper headlines, and the released artifacts compute a bootstrap CI only for
the no-action and LLM-order comparisons
(`codelewm/eval/execution_rerank.py`).

## Method

The numbers below are recomputed **from the checked-in per-completion score rows**
`docs/benchmark/v0_9/seed-{42,1729}/rerank/humaneval/reports/completion_scores.jsonl`
using the repository's own ranking and bootstrap procedure
(`_rank_batch` + `_bootstrap_lift_ci` in `codelewm/eval/execution_rerank.py`):
top-1 per problem is the highest-scoring candidate under each baseline
(ties broken by `completion_id`); the lift CI is a paired problem-level percentile
bootstrap with `seed=17`, `2000` resamples, `confidence_level=0.95`. No new
scoring run, checkpoint load, or candidate-code execution is performed.

## Validation

Recomputing the **no-action** lift CI with this procedure reproduces the value
serialized in the released reports exactly, which validates the recomputation:

| Seed | Recomputed no-action lift CI | Released `bootstrap_lift_over_no_action_ci` |
| ---: | --- | --- |
| 42 | [2.13, 21.28] | [2.13, 21.28] |
| 1729 | [2.13, 17.02] | [2.13, 17.02] |

## Result

| Seed | CodeLeWM | Shuffled-action | Lift vs shuffled | Problems | 95% bootstrap CI |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 46/47 | 42/47 (0.8936) | +8.51 pts | 4 / 47 | [0.00, 19.15] |
| 1729 | 46/47 | 43/47 (0.9149) | +6.38 pts | 3 / 47 | [0.00, 14.89] |

Shuffled-action pass@1 (0.8936 / 0.9149) is strictly above no-action
(0.8723 / 0.8936) on both open slices, so it is the binding control. CodeLeWM
beats it by 4 and 3 problems, but the 95% bootstrap interval for the
shuffled-action lift **includes zero on both seeds**. The HumanEval WS-D slice
therefore clears its pre-registered no-action gate (lift `>= 3.0` pts, CI excludes
zero) but is *not* robust against the strongest control. This is consistent with,
and a sharper statement of, the paper's central thesis that apparent reranking
wins must be read against the strongest available control.

## Reproduction

```bash
uv run python - <<'PY'
import json, random
from collections import defaultdict

def load(p):
    by=defaultdict(list)
    for l in open(p):
        if l.strip():
            r=json.loads(l); by[r["problem_id"]].append(r)
    return by

def top1_pass(batch, b):
    if b == "llm_order":
        s=sorted(batch, key=lambda c:(c["llm_order_rank"], c["completion_id"]))
    else:
        s=sorted(batch, key=lambda c:(-c["scores"].get(b, float("-inf")), c["completion_id"]))
    return bool(s[0]["passed"])

def boot_ci(ppp, a, b, seed=17, samples=2000, cl=0.95):
    pids=sorted(ppp); n=len(pids); rng=random.Random(seed); lifts=[]
    for _ in range(samples):
        bs=rng.choices(pids, k=n)
        ac=sum(ppp[p][a] for p in bs); bc=sum(ppp[p][b] for p in bs)
        lifts.append((ac-bc)/n*100.0)
    lifts.sort(); a_=(1-cl)/2
    return lifts[int(samples*a_)], lifts[min(samples-1, int(samples*(1-a_)))]

for seed in (42, 1729):
    by=load(f"docs/benchmark/v0_9/seed-{seed}/rerank/humaneval/reports/completion_scores.jsonl")
    ppp={pid:{b:top1_pass(batch,b) for b in ("codelewm","no_action","shuffled_action")}
         for pid,batch in by.items()}
    print(seed, "vs no_action", boot_ci(ppp,"codelewm","no_action"),
                "vs shuffled", boot_ci(ppp,"codelewm","shuffled_action"))
PY
```

Expected output: seed 42 `vs no_action (2.13, 21.28) vs shuffled (0.0, 19.15)`;
seed 1729 `vs no_action (2.13, 17.02) vs shuffled (0.0, 14.89)`.
