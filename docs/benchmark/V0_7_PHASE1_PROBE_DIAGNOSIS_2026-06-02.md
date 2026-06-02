# v0.7 Phase 1 — Representation-Probe Diagnosis (RFC-0015 WS-B/WS-C)

- Date: 2026-06-02
- Scope: RFC-0015 Phase 1; issues #338 (WS-B), #339 (WS-C), #341 (tracker)
- Compute: none beyond CPU; ran `codelewm eval execution-probe` against the local
  v0.6 execution checkpoint (`seed-42/last.pt`, trusted) over the local v0.6 pack.

## Why the probe gate is closed — corrected, ground-truth diagnosis

The RFC-0015 motivation inherited a claim from the planning recon that the v0.6
pack is "CodeNet stdout-text", which is why `output_magnitude_bucket` /
`output_length_bucket` had no labels. **Inspecting the actual pack
(`.artifacts/v0_6/execution-pack/pack.jsonl`, 1605 records) overturns that.**

The v0.6 pack is **typed function-call data** (`input_kind=function_call`,
`source_dataset=mbpp`), with a rich typed-output distribution:

```
output_type: int 494, str 385, list 312, bool 189, float 88, tuple 87,
             dict 33, exception 11, none 6
output_kind: value 1427, stdout 178
```

So the data is already typed. The probe gate is closed for two concrete,
fixable reasons — not a data-source problem:

### Bug 1 — `will_raise` read the wrong field (FIXED, validated)

Packed records mark a raising run with `execution_status="raised"` and
`output_type="exception"` (the `output_kind` stays `"value"`). The extractor
`_label_will_raise` checked `output_kind == "exception"`, which is **never
set**, so the target always collapsed to a single class and the **11 exception
records already in the pack were invisible**. Fixed to key on
`execution_status`/`output_type`.

Validated on the existing pack (no rebuild):

| | available targets | count |
|---|---|---|
| before fix | output_type, arithmetic | **2/5** → not_evaluable |
| after fix | output_type, arithmetic, **will_raise** | **3/5** → not_evaluable |

(`will_raise` train/val/test all populated; True/False both present in
train 10/1359 and test 1/156.)

### Bug 2 — the pack omits raw `output_repr` (needs a schema change + rebuild)

`output_magnitude_bucket` and `output_length_bucket` extractors
`ast.literal_eval(record["output_repr"])`, but the published record stores only
`output_tokens` (hashed) and `output_repr_checksum` — **`output_repr` is not a
record field** (a privacy/size design choice; `builder.py:411` computes it then
drops it). So both buckets get 0 labels regardless of how typed the outputs are.

**Fix (next increment, separate PR):** make the build **precompute the bucket
labels** (`output_magnitude_bucket`, `output_length_bucket`) from the in-hand
`output_repr` and persist them on the record (privacy-safe — no raw value
stored), bump the record schema version, and rebuild. The probe extractors then
read the precomputed buckets. That takes available targets from 3 → 5 and flips
`semantic_structure_status` to `evaluable`.

## Revised WS-B/WS-C plan implications

- The probe-evaluability fix is **not** "fetch new typed sources" (the recon's
  premise) — the data is already typed. It is two pack/extractor fixes:
  1. `will_raise` extractor (shipped),
  2. precompute magnitude/length buckets in the pack record (schema bump +
     rebuild).
- For a **robust** `will_raise` (and an `exception_type` target), the pack needs
  more than 11 exception records: broaden the source verdicts
  (`flatten-* --keep-verdicts accepted,wrong_answer,runtime_error`) so failing
  submissions contribute raising executions. This is the genuine data lever and
  remains a build-time change.
- WS-B1 ("add typed function-call sources") is **already satisfied** for mbpp;
  APPS can be added for diversity but is not required to make the probe
  evaluable.

## Related

- RFC-0015 (`docs/rfcs/RFC-0015-v0-7-execution-substrate-improvements.md`)
- Tool: `codelewm/eval/execution_probe_targets.py`; CLI `codelewm eval execution-probe`
- Phase-0: `docs/benchmark/V0_7_PHASE0_RERANK_CALIBRATION_2026-06-02.md`
