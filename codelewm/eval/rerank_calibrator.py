"""Phase-0 rerank calibration probe (RFC-0015 WS-A1).

Answers one question without any model inference or GPU: *is execution
correctness decodable from the per-completion features the rerank eval already
dumps?* It loads ``completion_scores.jsonl`` artifacts (schema
``codelewm.eval.completion_score.v1``), fits a small numpy logistic-regression
calibrator on the available features against the ``passed`` label using
problem-grouped cross-validation, and reports:

- completion-level decodability: cross-validated ROC-AUC of the calibrator and
  the univariate ROC-AUC of each individual feature;
- rerank pass@1 for each ranking (llm_order, no_action, codelewm, lexical, and
  the out-of-fold calibrator), over all problems AND over the *unsaturated*
  subset (problems with a genuine pass/fail mix), which is where reranking can
  change anything;
- the structural ceiling: how many problems are even rerankable.

No third-party ML dependency is used (numpy only), so this runs in the base
environment.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


RERANK_CALIBRATION_REPORT_SCHEMA_VERSION = "codelewm.eval.rerank_calibration.v1"

# Baselines whose raw score is also a candidate ranking (higher = better).
_BASELINE_SCORE_KEYS = ("codelewm", "no_action", "lexical", "shuffled_action", "random")
# Feature columns fed to the calibrator (derived from the score row).
_FEATURE_KEYS = (
    "codelewm",
    "no_action",
    "lexical",
    "shuffled_action",
    "random",
    "codelewm_minus_no_action",
    "neg_llm_order_rank",
)


class RerankCalibratorError(ValueError):
    """Raised when calibration inputs are malformed."""


def load_completion_scores(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load and concatenate ``completion_scores.jsonl`` rows from ``paths``."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        if not source.is_file():
            raise RerankCalibratorError(f"completion scores file not found: {source}")
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise RerankCalibratorError(f"completion score row is not an object in {source}")
            rows.append(dict(record))
    if not rows:
        raise RerankCalibratorError("no completion score rows loaded")
    return rows


def _feature_vector(row: Mapping[str, Any]) -> list[float]:
    scores = row.get("scores") if isinstance(row.get("scores"), Mapping) else {}
    codelewm = float(scores.get("codelewm", 0.0))
    no_action = float(scores.get("no_action", 0.0))
    return [
        codelewm,
        no_action,
        float(scores.get("lexical", 0.0)),
        float(scores.get("shuffled_action", 0.0)),
        float(scores.get("random", 0.0)),
        codelewm - no_action,
        -float(row.get("llm_order_rank", 0)),
    ]


def _ranking_score(row: Mapping[str, Any], key: str) -> float:
    if key == "llm_order":
        return -float(row.get("llm_order_rank", math.inf))
    scores = row.get("scores") if isinstance(row.get("scores"), Mapping) else {}
    return float(scores.get(key, float("-inf")))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _fit_logreg(
    x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iters: int = 800, lr: float = 0.5
) -> np.ndarray:
    """L2-regularized logistic regression via full-batch gradient descent."""

    n, d = x.shape
    xb = np.ascontiguousarray(np.concatenate([x, np.ones((n, 1))], axis=1), dtype=np.float64)
    w = np.zeros(d + 1)
    # The Accelerate/vecLib BLAS backend on Apple Silicon emits spurious
    # "divide by zero / overflow / invalid encountered in matmul" warnings even
    # for finite, well-conditioned inputs; inputs are standardized and weights
    # stay bounded, so these are safe to silence.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(iters):
            pred = _sigmoid(xb @ w)
            grad = xb.T @ (pred - y) / n
            grad[:-1] += (l2 / n) * w[:-1]  # do not regularize bias
            w -= lr * grad
    return w


def _predict_logreg(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.ascontiguousarray(np.concatenate([x, np.ones((x.shape[0], 1))], axis=1), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _sigmoid(xb @ w)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based ROC-AUC (Mann-Whitney U). None if only one class present."""

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _assign_tie_ranks(scores, order, ranks)
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _assign_tie_ranks(scores: np.ndarray, order: np.ndarray, ranks: np.ndarray) -> None:
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1


def _grouped_folds(groups: Sequence[str], k: int, seed: int) -> dict[str, int]:
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return {g: idx % k for idx, g in enumerate(unique)}


def _standardize(train: np.ndarray, full: np.ndarray) -> np.ndarray:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (full - mean) / std


def _pass_at_1(
    grouped: Mapping[str, list[int]],
    rows: Sequence[Mapping[str, Any]],
    score_fn,
    only: set[str] | None = None,
) -> float | None:
    groups = [g for g in grouped if only is None or g in only]
    if not groups:
        return None
    hits = 0
    for g in groups:
        idxs = grouped[g]
        best = max(idxs, key=lambda i: (score_fn(i), -i))
        hits += 1 if bool(rows[best].get("passed")) else 0
    return hits / len(groups)


def evaluate_rerank_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    k_folds: int = 5,
    seed: int = 0,
    l2: float = 1.0,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Fit the OOF calibrator and report decodability + rerank pass@1."""

    rows = list(rows)
    n = len(rows)
    y = np.array([1 if bool(r.get("passed")) else 0 for r in rows], dtype=float)
    x = np.array([_feature_vector(r) for r in rows], dtype=float)
    groups = [str(r.get("problem_id", f"_row{i}")) for i, r in enumerate(rows)]

    # Problem-grouped k-fold; out-of-fold calibrator predictions (no leakage).
    fold_of = _grouped_folds(groups, k_folds, seed)
    oof = np.full(n, np.nan)
    for fold in range(k_folds):
        test_mask = np.array([fold_of[g] == fold for g in groups])
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if y[train_mask].sum() in (0, train_mask.sum()):
            # degenerate train fold: fall back to base rate
            oof[test_mask] = float(y[train_mask].mean())
            continue
        xt = _standardize(x[train_mask], x)
        w = _fit_logreg(xt[train_mask], y[train_mask], l2=l2)
        oof[test_mask] = _predict_logreg(xt[test_mask], w)
    oof = np.where(np.isnan(oof), float(y.mean()), oof)

    # Decodability.
    calibrator_auc = roc_auc(y, oof)
    univariate_auc = {
        name: roc_auc(y, x[:, j]) for j, name in enumerate(_FEATURE_KEYS)
    }

    # Rerank pass@1.
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        grouped[g].append(i)
    rates = {g: np.mean([y[i] for i in idxs]) for g, idxs in grouped.items()}
    unsaturated = {g for g, r in rates.items() if 0.0 < r < 1.0}

    rankings = {
        "llm_order": lambda i: _ranking_score(rows[i], "llm_order"),
        "no_action": lambda i: _ranking_score(rows[i], "no_action"),
        "codelewm": lambda i: _ranking_score(rows[i], "codelewm"),
        "lexical": lambda i: _ranking_score(rows[i], "lexical"),
        "calibrator": lambda i: float(oof[i]),
    }
    pass_at_1_all = {
        name: _pass_at_1(grouped, rows, fn) for name, fn in rankings.items()
    }
    pass_at_1_unsat = {
        name: _pass_at_1(grouped, rows, fn, only=unsaturated)
        for name, fn in rankings.items()
    }

    # Bootstrap CI for calibrator lift over no_action and codelewm (by problem).
    lift_ci = _bootstrap_lift_ci(
        grouped, rows, rankings, oof, seed=seed, samples=bootstrap_samples
    )

    return {
        "schema_version": RERANK_CALIBRATION_REPORT_SCHEMA_VERSION,
        "completion_count": n,
        "problem_count": len(grouped),
        "candidates_per_problem": sorted({len(v) for v in grouped.values()}),
        "overall_pass_rate": float(y.mean()),
        "rerankable_problem_count": len(unsaturated),
        "all_pass_problem_count": int(sum(1 for r in rates.values() if r == 1.0)),
        "all_fail_problem_count": int(sum(1 for r in rates.values() if r == 0.0)),
        "max_possible_rerank_headroom_pts": round(
            100.0 * len(unsaturated) / max(1, len(grouped)), 4
        ),
        "decodability": {
            "calibrator_cv_auc": calibrator_auc,
            "univariate_auc": univariate_auc,
            "k_folds": k_folds,
        },
        "rerank_pass_at_1": {
            "all_problems": pass_at_1_all,
            "unsaturated_only": pass_at_1_unsat,
        },
        "calibrator_lift_pts": lift_ci,
        "features": list(_FEATURE_KEYS),
    }


def _bootstrap_lift_ci(
    grouped: Mapping[str, list[int]],
    rows: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Any],
    oof: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    group_ids = list(grouped.keys())
    rng = np.random.default_rng(seed + 1)

    def top1_passed(g: str, fn) -> int:
        best = max(grouped[g], key=lambda i: (fn(i), -i))
        return 1 if bool(rows[best].get("passed")) else 0

    cal = {g: top1_passed(g, rankings["calibrator"]) for g in group_ids}
    base = {
        b: {g: top1_passed(g, rankings[b]) for g in group_ids}
        for b in ("no_action", "codelewm", "llm_order")
    }
    out: dict[str, Any] = {}
    for b in ("no_action", "codelewm", "llm_order"):
        diffs = []
        m = len(group_ids)
        for _ in range(samples):
            pick = rng.integers(0, m, size=m)
            c = np.mean([cal[group_ids[i]] for i in pick])
            d = np.mean([base[b][group_ids[i]] for i in pick])
            diffs.append(100.0 * (c - d))
        point = 100.0 * (
            np.mean([cal[g] for g in group_ids])
            - np.mean([base[b][g] for g in group_ids])
        )
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[f"calibrator_minus_{b}"] = {
            "point_pts": round(float(point), 4),
            "ci95_pts": [round(float(lo), 4), round(float(hi), 4)],
        }
    return out


def _expand_paths(patterns: Sequence[str]) -> list[str]:
    found: list[str] = []
    for pat in patterns:
        hits = sorted(_glob.glob(pat, recursive=True))
        found.extend(hits if hits else [pat])
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase-0 rerank calibration probe over completion_scores.jsonl"
    )
    parser.add_argument(
        "scores", nargs="+", help="completion_scores.jsonl paths or globs (recursive)"
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--out", type=Path, help="write the JSON report to this path")
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args(argv)

    paths = _expand_paths(args.scores)
    rows = load_completion_scores(paths)
    report = evaluate_rerank_calibration(
        rows,
        k_folds=args.k_folds,
        seed=args.seed,
        l2=args.l2,
        bootstrap_samples=args.bootstrap_samples,
    )
    report["sources"] = paths
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_summary(report)
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_summary(report: Mapping[str, Any]) -> None:
    print("rerank calibration probe (RFC-0015 WS-A1)")
    print(
        f"  completions={report['completion_count']} problems={report['problem_count']} "
        f"cand/prob={report['candidates_per_problem']} pass_rate={_fmt(report['overall_pass_rate'])}"
    )
    print(
        f"  rerankable(unsaturated)={report['rerankable_problem_count']} "
        f"all_pass={report['all_pass_problem_count']} all_fail={report['all_fail_problem_count']} "
        f"max_headroom={_fmt(report['max_possible_rerank_headroom_pts'])} pts"
    )
    dec = report["decodability"]
    print(f"  decodability calibrator CV AUC = {_fmt(dec['calibrator_cv_auc'])}")
    for k, v in dec["univariate_auc"].items():
        print(f"    univariate AUC {k:26} = {_fmt(v)}")
    print("  rerank pass@1 (all problems):")
    for k, v in report["rerank_pass_at_1"]["all_problems"].items():
        print(f"    {k:12} = {_fmt(v)}")
    print("  rerank pass@1 (unsaturated only):")
    for k, v in report["rerank_pass_at_1"]["unsaturated_only"].items():
        print(f"    {k:12} = {_fmt(v)}")
    print("  calibrator lift (pts, 95% CI):")
    for k, v in report["calibrator_lift_pts"].items():
        print(f"    {k:24} {_fmt(v['point_pts'])}  CI {v['ci95_pts']}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
