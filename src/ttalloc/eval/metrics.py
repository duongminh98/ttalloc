"""Độ đo (PLAN mục 7.3, P2.T16)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ttalloc.logging.schema import COST_FIELDS

NAT_BINS = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0000001)]


def with_total(pq: pd.DataFrame) -> pd.DataFrame:
    return pq.assign(total=pq[list(COST_FIELDS)].sum(axis=1))


def cell_keys(pq: pd.DataFrame) -> list[str]:
    """Khóa của một ô; có cột `run` (gộp nhiều thư mục chạy) thì tách theo lần chạy, vì số lần rút trùng nhau."""
    return (["run"] if "run" in pq.columns else []) + ["split", "batch", "method", "alpha"]


def cell_table(pq: pd.DataFrame) -> pd.DataFrame:
    """Độ chính xác (trung bình qua lần rút) và token theo loại cho mỗi (split, batch, method, alpha)."""
    pq = with_total(pq)
    keys = cell_keys(pq)
    per_draw = pq.groupby([*keys, "draw"]).agg(
        acc=("correct", "mean"),
        tokens=("total", "sum"),
        probe=("probe_output", "sum"),
        probe_p=("probe_prompt", "sum"),
        budget=("budget", "first"),
    )
    out = (
        per_draw.groupby(keys)
        .agg(
            accuracy=("acc", "mean"),
            accuracy_std=("acc", "std"),
            tokens=("tokens", "mean"),
            budget=("budget", "mean"),
            probe=("probe", "mean"),
            probe_p=("probe_p", "mean"),
            n_draws=("acc", "size"),
        )
        .reset_index()
    )
    out["probe_share"] = (out.probe + out.probe_p) / out.tokens.where(out.tokens > 0, 1)
    out["spend_ratio"] = out.tokens / out.budget.where(out.budget > 0, 1)
    return out.drop(columns=["probe", "probe_p"])


def auc_over_alpha(alphas: np.ndarray, accs: np.ndarray) -> float:
    """Diện tích dưới đường độ chính xác theo alpha, chuẩn hóa về [0, 1] bằng độ rộng của lưới alpha."""
    o = np.argsort(alphas)
    a, y = np.asarray(alphas, float)[o], np.asarray(accs, float)[o]
    if len(a) < 2 or a[-1] == a[0]:
        return float(y.mean())
    return float(np.sum((a[1:] - a[:-1]) * (y[1:] + y[:-1]) / 2) / (a[-1] - a[0]))


def tokens_saved_vs(ref_tokens: np.ndarray, ref_acc: np.ndarray, tokens: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Tỉ lệ token tiết kiệm so với phương pháp tham chiếu ở cùng độ chính xác (nội suy ngược trên đường tham chiếu).

    NaN khi độ chính xác nằm ngoài khoảng tham chiếu đạt được.
    """
    o = np.argsort(ref_acc)
    ra, rt = np.asarray(ref_acc, float)[o], np.asarray(ref_tokens, float)[o]
    need = np.interp(acc, ra, rt, left=np.nan, right=np.nan)
    return np.asarray(1.0 - np.asarray(tokens, float) / need)


def miscut_rate(pq: pd.DataFrame, nat_correct: dict[tuple[str, int], bool]) -> pd.DataFrame:
    """Tỉ lệ câu trả lời sai trong khi câu trả lời tự nhiên của cùng chuỗi là đúng."""
    nat = np.array([nat_correct.get((q, int(s)), False) for q, s in zip(pq.qid, pq.seed, strict=True)])
    df = pq.assign(miscut=(~pq.correct.astype(bool)) & nat, nat_ok=nat)
    return (
        df.groupby(["split", "batch", "method", "alpha"])
        .agg(miscut_rate=("miscut", "mean"), nat_ok_rate=("nat_ok", "mean"))
        .reset_index()
    )


def natural_accuracy(nat_correct: dict[tuple[str, int], bool]) -> dict[str, float]:
    by: dict[str, list[bool]] = {}
    for (q, _), v in nat_correct.items():
        by.setdefault(q, []).append(bool(v))
    return {q: float(np.mean(v)) for q, v in by.items()}


def difficulty_bins(pq: pd.DataFrame, nat_acc: dict[str, float] | None) -> pd.Series:
    """Nhóm độ khó: mức MATH nếu có, ngược lại theo tỉ lệ đúng của câu trả lời tự nhiên qua các seed."""
    if pq.difficulty.notna().all():
        return pq.difficulty.astype(int).map(lambda x: f"L{x}")
    if nat_acc is None:
        return pd.Series(["all"] * len(pq), index=pq.index)

    def lab(q: str) -> str:
        v = nat_acc.get(q, np.nan)
        for lo, hi in NAT_BINS:
            if lo <= v < hi:
                return f"nat[{lo:.2f},{min(hi, 1):.2f}]"
        return "nat_na"

    return pq.qid.map(lab)


def by_difficulty(pq: pd.DataFrame, nat_acc: dict[str, float] | None) -> pd.DataFrame:
    pq = with_total(pq).assign(diff_bin=lambda d: difficulty_bins(d, nat_acc))
    return (
        pq.groupby([*cell_keys(pq), "diff_bin"])
        .agg(accuracy=("correct", "mean"), tokens=("total", "mean"), n=("correct", "size"))
        .reset_index()
    )
