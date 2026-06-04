"""A1.5 full-latent correctness probe (de-risks A3).

A1 showed the SCALAR model energies decode correctness at chance. A3's head
reads the FULL latent vectors, which the scalar energy may collapse away. This
probe encodes every WS-D candidate through the v0.7 checkpoint, extracts the
full z_code / z_pred_after vectors, and fits a grouped-k-fold logistic probe
to `passed`. Gate: ROC-AUC > ~0.65 -> latents encode correctness -> build A3;
~0.5 -> representation lacks it -> pivot to WS-C.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from codelewm.harness.scorer import ExecutionTorchTransitionScoringBackend
from codelewm.eval.rerank_calibrator import (
    _fit_logreg,
    _grouped_folds,
    _predict_logreg,
    _standardize,
    roc_auc,
)


def encode_rows(backend, rows):
    runtime = backend.runtime
    feats = {"z_code": [], "z_pred": [], "z_diff": [], "concat": []}
    y, groups = [], []
    for r in rows:
        code = r["code"]
        si = r.get("scoring_inputs") or []
        if not si:
            continue
        instr = si[0]["input_repr"]
        cs = backend._state_batch_from_text(code, field_name="candidate")
        ab = backend._action_batch_from_text(instr)
        with runtime.no_grad():
            z_code = backend.model.encode_state(cs).float().cpu().numpy().reshape(-1)
            action_emb = backend.model.encode_action(ab)
            z_pred = backend.model.predict_after(
                backend.model.encode_state(cs), action_emb
            ).float().cpu().numpy().reshape(-1)
        feats["z_code"].append(z_code)
        feats["z_pred"].append(z_pred)
        feats["z_diff"].append(z_pred - z_code)
        feats["concat"].append(np.concatenate([z_code, z_pred]))
        y.append(1 if r.get("passed") else 0)
        groups.append(str(r.get("problem_id")))
    return {k: np.array(v, dtype=float) for k, v in feats.items()}, np.array(y, dtype=float), groups


def probe(x, y, groups, k=5, seed=0, l2=1.0):
    n = len(y)
    fold_of = _grouped_folds(groups, k, seed)
    oof = np.full(n, np.nan)
    for fold in range(k):
        test = np.array([fold_of[g] == fold for g in groups])
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue
        if y[train].sum() in (0, train.sum()):
            oof[test] = float(y[train].mean()); continue
        xt = _standardize(x[train], x)
        w = _fit_logreg(xt[train], y[train], l2=l2)
        oof[test] = _predict_logreg(xt[test], w)
    oof = np.where(np.isnan(oof), float(y.mean()), oof)
    return roc_auc(y, oof)


def main():
    ckpt = sys.argv[1]
    label_paths = sys.argv[2:]
    rows = []
    for p in label_paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} completions from {len(label_paths)} file(s)")
    backend = ExecutionTorchTransitionScoringBackend.load(ckpt, device="cpu")
    feats, y, groups = encode_rows(backend, rows)
    print(f"encoded {len(y)} candidates | pass rate {y.mean():.3f} | dim z={feats['z_code'].shape[1]}")
    print("=== full-latent correctness probe (grouped 5-fold ROC-AUC) ===")
    for name in ("z_code", "z_pred", "z_diff", "concat"):
        auc = probe(feats[name], y, groups, seed=0)
        print(f"  {name:<8} AUC = {auc:.3f}")


if __name__ == "__main__":
    main()
