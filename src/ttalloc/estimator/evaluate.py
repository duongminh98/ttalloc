"""Đánh giá bộ ước lượng (E1; PLAN P2.T6)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ttalloc.features.curve import EmpiricalCurves

EPS = 1e-12


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def log_loss(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(ll, weights=w))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Các nhóm cùng số mẫu theo p: trung bình p, tỉ lệ đúng, số mẫu."""
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = [
        {"bin": i, "n": len(b), "p_mean": float(np.mean(p[b])), "y_mean": float(np.mean(y[b]))}
        for i, b in enumerate(bins)
        if len(b)
    ]
    return pd.DataFrame(rows)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error với 15 nhóm cùng số mẫu."""
    r = reliability(np.asarray(y, float), np.asarray(p, float), n_bins)
    if r.empty:
        return float("nan")
    return float((r.n * (r.p_mean - r.y_mean).abs()).sum() / r.n.sum())


def gain_spearman(
    pred: pd.DataFrame, curves: EmpiricalCurves, gen_end: dict[tuple[str, int, int], int], hs: list[int]
) -> dict[str, float]:
    """Spearman giữa qhat(h) - qhat(0) và lợi ích thực nghiệm tính từ các seed khác của cùng câu."""
    from scipy.stats import spearmanr

    out: dict[str, float] = {}
    b0 = pred[pred.h == 0]
    base: dict[tuple[str, int, int], float] = {
        (str(q), int(s), int(i)): float(p) for q, s, i, p in zip(b0.qid, b0.seed, b0.point_idx, b0.p, strict=True)
    }
    px: list[float] = []
    py: list[float] = []
    for hv in hs:
        g = pred[pred.h == hv]
        xs: list[float] = []
        ys: list[float] = []
        for q_, s_, i_, p_ in zip(g.qid, g.seed, g.point_idx, g.p, strict=True):
            key = (str(q_), int(s_), int(i_))
            b = gen_end[key]
            q = curves.q_many(key[0], np.array([b, b + hv], float), exclude_seed=key[1])
            if q is None:
                continue
            xs.append(float(p_) - base[key])
            ys.append(float(q[1] - q[0]))
        out[str(hv)] = float(spearmanr(xs, ys).statistic) if len(xs) > 2 else float("nan")
        px += xs
        py += ys
    per_h = [v for v in out.values() if v == v]
    out["mean"] = float(np.mean(per_h)) if per_h else float("nan")
    out["pooled"] = float(spearmanr(px, py).statistic) if len(px) > 2 else float("nan")  # dùng cho G1(a)
    return out


def evaluate(
    pred: pd.DataFrame, curves: EmpiricalCurves | None, gen_end: dict[tuple[str, int, int], int] | None, hs: list[int]
) -> dict[str, Any]:
    """`pred` có cột qid, seed, point_idx, h, y, p (dự đoán đã hiệu chuẩn)."""
    rep: dict[str, Any] = {"per_h": {}}
    for hv, g in pred.groupby("h"):
        y = g.y.to_numpy(int)
        p = g.p.to_numpy(float)
        rep["per_h"][str(hv)] = {
            "n": len(g),
            "auroc": auroc(y, p),
            "ece": ece(y, p),
            "brier": brier(y, p),
            "log_loss": log_loss(y, p),
            "base_rate": float(y.mean()),
        }
    rep["auroc_q0"] = rep["per_h"].get("0", {}).get("auroc")
    rep["max_ece"] = max(v["ece"] for v in rep["per_h"].values())
    rep["mean_log_loss"] = float(np.mean([v["log_loss"] for v in rep["per_h"].values()]))
    if curves is not None and gen_end is not None:
        rep["gain_spearman"] = gain_spearman(pred, curves, gen_end, [h for h in hs if h > 0])
    return rep
