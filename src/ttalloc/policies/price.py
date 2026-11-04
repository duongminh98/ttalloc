"""Chính sách theo giá token (M2; M3 là M2 với CurveRhoProvider). PLAN A7, mục 7.2."""

from __future__ import annotations

from dataclasses import dataclass

from ttalloc.config import DecisionPointCfg
from ttalloc.policies.base import tie_key
from ttalloc.types import Decision, PoolView, QuestionState, TraceKey


def adaptive_price(active: list[QuestionState], avail: int, run_seed: int) -> float:
    """Giá mu: rho của câu đầu tiên làm tổng h* cộng dồn (theo rho giảm dần) vượt quỹ; không vượt thì 0."""
    order = sorted(active, key=lambda s: (-(s.rho if s.rho is not None else float("-inf")), tie_key(s.key, run_seed)))
    cum = 0
    for s in order:
        if s.hstar is None:
            continue
        cum += s.hstar
        if cum > avail:
            return float(s.rho)  # type: ignore[arg-type]
    return 0.0


@dataclass
class PricePolicy:
    dp: DecisionPointCfg
    mu_min: float
    b_max: int
    run_seed: int
    uses_probes: bool = True
    uses_rho: bool = True
    stop_at_markers: bool = True
    last_mu: float | None = None  # để ghi lại và kiểm tra

    def initial_request(self, s: QuestionState, pool: PoolView) -> Decision:
        return Decision("continue", self.dp.b_min, self.dp.b_min + self.dp.l_max)

    def decide(self, active: list[QuestionState], pool: PoolView) -> dict[TraceKey, Decision]:
        for s in active:
            if s.rho is None:
                raise ValueError(f"{s.key}: chưa có rho (RhoProvider chưa điền)")
        mu = adaptive_price(active, pool.available(), self.run_seed)
        self.last_mu = mu
        out = {}
        for s in active:
            rho = float(s.rho)  # type: ignore[arg-type]
            go = rho >= mu and rho > self.mu_min and s.gen_end < self.b_max
            out[s.key] = Decision("continue", self.dp.m, self.dp.l_max) if go else Decision("stop")
        return out
