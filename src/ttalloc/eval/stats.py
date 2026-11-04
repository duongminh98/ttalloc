"""Thống kê (PLAN A12, P2.T17)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def bonferroni_level(k: int, alpha: float = 0.05) -> float:
    return 1 - alpha / k


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, level: float = 0.95
) -> dict[str, float]:
    """a[d, q], b[d, q]: đúng sai của hai phương pháp ở lần rút d, câu q (cùng các lần rút). Bootstrap theo câu hỏi."""
    if a.shape != b.shape:
        raise ValueError(f"kích thước khác nhau: {a.shape} và {b.shape}")
    d = (np.asarray(a, float) - np.asarray(b, float)).mean(axis=0)
    q = len(d)
    idx = rng.integers(0, q, size=(n_boot, q))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "level": level, "n_questions": q}


def bootstrap_over_draws(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, level: float = 0.95
) -> dict[str, float]:
    """Khi lô đổi theo lần rút (không ghép được theo câu): bootstrap trên chênh lệch độ chính xác của từng lần rút."""
    d = np.asarray(a, float) - np.asarray(b, float)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "level": level, "n_draws": len(d)}


def compare_methods(
    pq: pd.DataFrame, m_a: str, m_b: str, alpha: float, rng: np.random.Generator, n_boot: int, level: float
) -> dict[str, Any]:
    """So sánh a − b trên các lần rút chung, ở một mức alpha (một split, một lô)."""
    sub = pq[pq.alpha == alpha]
    A = sub[sub.method == m_a]
    B = sub[sub.method == m_b]
    if A.empty or B.empty:
        raise ValueError(f"thiếu dữ liệu cho {m_a} hoặc {m_b} ở alpha={alpha}")
    same_q = A.groupby("draw").qid.apply(frozenset).nunique() == 1
    if same_q:
        pa = A.pivot_table(index="draw", columns="qid", values="correct", aggfunc="first").astype(float)
        pb = B.pivot_table(index="draw", columns="qid", values="correct", aggfunc="first").astype(float)
        draws = pa.index.intersection(pb.index)
        pa, pb = pa.loc[draws], pb.loc[draws, pa.columns]
        return paired_bootstrap(pa.to_numpy(), pb.to_numpy(), rng, n_boot, level)
    acc_a = A.groupby("draw").correct.mean()
    acc_b = B.groupby("draw").correct.mean()
    draws = acc_a.index.intersection(acc_b.index)
    return bootstrap_over_draws(acc_a.loc[draws].to_numpy(), acc_b.loc[draws].to_numpy(), rng, n_boot, level)


def mcnemar_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Kiểm định McNemar theo từng câu (một seed chạy thật); dạng exact khi số cặp lệch nhỏ (< 25)."""
    from statsmodels.stats.contingency_tables import mcnemar

    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    table = [[int(np.sum(a & b)), int(np.sum(a & ~b))], [int(np.sum(~a & b)), int(np.sum(~a & ~b))]]
    disc = table[0][1] + table[1][0]
    exact = disc < 25
    res = mcnemar(table, exact=exact, correction=not exact)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "exact": exact,
        "n01": table[0][1],
        "n10": table[1][0],
    }


def acceptance_band(sim_acc: np.ndarray, k: int, rng: np.random.Generator, n_sets: int = 2000) -> tuple[float, float]:
    """Dải chấp nhận của G3: phân vị 2,5–97,5% của độ chính xác trung bình trên các bộ k lần rút khác nhau."""
    sim_acc = np.asarray(sim_acc, float)
    if len(sim_acc) < k:
        raise ValueError(f"cần ít nhất {k} lần rút mô phỏng, có {len(sim_acc)}")
    means = np.array([sim_acc[rng.choice(len(sim_acc), size=k, replace=False)].mean() for _ in range(n_sets)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
