"""Bộ dữ liệu huấn luyện bộ ước lượng (PLAN A4, P2.T3). Đặc trưng chỉ tính trên dãy chuẩn (A3)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from ttalloc.config import DecisionPointCfg, FeatureCfg
from ttalloc.features.events import LoggedChain, canonical_points
from ttalloc.features.extract import FeatureState, feature_names


def features_along(c: LoggedChain, path: list[int], k: int) -> Iterator[dict[str, float]]:
    """Véc-tơ đặc trưng sau mỗi điểm của `path`, cập nhật tuần tự như khi chạy (A5)."""
    fs = FeatureState(k=k)
    for i in path:
        lc = c.local_conf_min[i]
        fs.update(int(c.gen_end[i]), c.probe(i), int(c.n_markers[i]), None if lc != lc else float(lc))
        yield fs.vector()


def label_at(c: LoggedChain, sub: list[int], j: int, h: int) -> bool:
    """y(h) cho mẫu tại điểm thứ j của dãy chuẩn (A4)."""
    if h == 0:
        return bool(c.ans_correct[sub[j]])
    lim = int(c.gen_end[sub[j]]) + h
    if c.natural_end and c.chain_len <= lim:
        return bool(c.nat_correct)
    last = sub[j]
    for q in sub[j:]:
        if c.gen_end[q] <= lim:
            last = q
        else:
            break
    return bool(c.ans_correct[last])


def samples_for_chain(
    c: LoggedChain, dp: DecisionPointCfg, feat: FeatureCfg, h_train: list[int], sub: list[int] | None = None
) -> list[dict[str, Any]]:
    sub = canonical_points(c, dp) if sub is None else sub
    if not sub:
        return []
    names = feature_names(feat.drop_groups)
    w = 1.0 / (len(sub) * (1 + len(h_train)))
    rows = []
    for j, x in enumerate(features_along(c, sub, feat.k)):
        base = {n: x[n] for n in names}
        for h in (0, *h_train):
            rows.append(
                {
                    **base,
                    "h": h,
                    "y": label_at(c, sub, j, h),
                    "w": w,
                    "qid": c.key.qid,
                    "seed": c.key.seed,
                    "point_idx": sub[j],
                }
            )
    return rows


def build_dataset(
    chains: dict[Any, LoggedChain], dp: DecisionPointCfg, feat: FeatureCfg, h_train: list[int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in chains.values():
        rows += samples_for_chain(c, dp, feat, h_train)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("bộ dữ liệu huấn luyện rỗng: không chuỗi nào có điểm trên dãy chuẩn")
    return df
