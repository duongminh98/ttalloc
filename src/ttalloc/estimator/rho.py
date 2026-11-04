"""Lợi ích trên mỗi token (PLAN A6, P2.T5). Đơn vị của rho trong mã: xác suất trên mỗi token."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from ttalloc.config import DecisionPointCfg
from ttalloc.features.events import LoggedChain, canonical_points

RHO_REPORT_SCALE = 1e5  # nhân để ra "điểm phần trăm trên 1.000 token" khi báo cáo


def rho_hstar(
    qhat: Mapping[int, float], b: int, horizons: Iterable[int], b_max: float, omega: float
) -> tuple[float, int | None]:
    ok = [h for h in horizons if h <= b_max - b]
    if not ok:
        return -math.inf, None
    gains = [(qhat[h] - qhat[0]) / (h * (1 + omega)) for h in ok]
    k = max(range(len(ok)), key=lambda i: (gains[i], -i))  # hòa thì chọn tầm ngắn hơn
    return gains[k], ok[k]


def estimate_omega(chains: Iterable[LoggedChain], dp: DecisionPointCfg) -> float:
    """Tổng chi phí gợi tại các điểm của dãy chuẩn chia cho tổng token suy luận tới điểm cuối của dãy (A6)."""
    probe = think = 0
    for c in chains:
        sub = canonical_points(c, dp)
        if not sub:
            continue
        probe += sum(c.probe_cost(i) for i in sub)
        think += int(c.gen_end[sub[-1]])
    return probe / think if think else 0.0
