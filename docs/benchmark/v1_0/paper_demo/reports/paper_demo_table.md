# CodeLeWM v1.0 Paper Demo Table

| Seed | Benchmark | CodeLeWM pass@1 | No-action pass@1 | LLM-order pass@1 | Lift vs no-action | Lift vs LLM-order | Claim gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | HumanEval WS-D | 97.9% | 87.2% | 14.9% | +10.6 pts | +83.0 pts | open |
| 42 | MBPP-Plus WS-D | 100.0% | 100.0% | 17.6% | +0.0 pts | +82.4 pts | closed |
| 1729 | HumanEval WS-D | 97.9% | 89.4% | 14.9% | +8.5 pts | +83.0 pts | open |
| 1729 | MBPP-Plus WS-D | 100.0% | 100.0% | 17.6% | +0.0 pts | +82.4 pts | closed |

Aggregate claim gate: closed.

Approved wording:

On the v0.9 WS-D replay, CodeLeWM strongly reranks HumanEval slices but the aggregate downstream claim remains closed because MBPP-Plus is saturated against the no-action baseline.
