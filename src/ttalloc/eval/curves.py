"""Phân tích thăm dò đường cong đúng sai theo số token (PLAN P2.T20)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ttalloc.features.curve import EmpiricalCurves
from ttalloc.types import Question


def question_curves(curves: EmpiricalCurves, qids: list[str], bs: list[int]) -> pd.DataFrame:
    """q_i(b) trung bình trên mọi seed, với b thuộc {0} ∪ h_train ∪ {L_log}."""
    rows = []
    arr = np.asarray(bs, float)
    for q in qids:
        v = curves.q_many(q, arr)
        if v is None:
            continue
        rows.append({"qid": q, **{f"b_{b}": float(x) for b, x in zip(bs, v, strict=True)}})
    return pd.DataFrame(rows)


def curve_stats(df: pd.DataFrame, bs: list[int], l_log: int, delta: float, groups: pd.Series) -> dict[str, Any]:
    cols = [f"b_{b}" for b in bs]
    q = df[cols].to_numpy()
    q0 = df[f"b_{bs[0]}"].to_numpy()
    ql = df[f"b_{l_log}"].to_numpy()
    qmax = q.max(axis=1)
    gain = ql - q0 >= delta
    drop = qmax - ql >= delta
    b90 = np.array([bs[int(np.argmax(row >= 0.9 * m))] if m > 0 else np.nan for row, m in zip(q, qmax, strict=True)])
    df = df.assign(gain=gain, drop=drop, b90=b90, group=groups.to_numpy())
    out: dict[str, Any] = {"delta": delta, "overall": _agg(df), "by_group": {}}
    for g, sub in df.groupby("group"):
        out["by_group"][str(g)] = _agg(sub)
    out["mean_curve_by_group"] = {str(g): sub[cols].mean().round(4).tolist() for g, sub in df.groupby("group")}
    out["bs"] = bs
    return out


def _agg(df: pd.DataFrame) -> dict[str, Any]:
    b90 = df.b90.dropna()
    return {
        "n": len(df),
        "gain_rate": float(df.gain.mean()),
        "overthinking_rate": float(df["drop"].mean()),
        "b90_quantiles": {str(p): float(np.percentile(b90, p)) for p in (10, 25, 50, 75, 90)} if len(b90) else {},
    }


def difficulty_groups(df: pd.DataFrame, questions: dict[str, Question], nat_acc: dict[str, float]) -> pd.Series:
    def lab(qid: str) -> str:
        d = questions[qid].difficulty if qid in questions else None
        if d is not None:
            return f"L{d}"
        v = nat_acc.get(qid, np.nan)
        return "nat<0.25" if v < 0.25 else ("nat<0.75" if v < 0.75 else "nat>=0.75")

    return df.qid.map(lab)
