"""DEER với hạn mức theo câu (B2; B1 dùng lại với `cap: per_question_uniform` ở P5.T1). Ngữ nghĩa ở PLAN mục 7.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ttalloc.config import DecisionPointCfg
from ttalloc.types import Decision, PoolView, QuestionState, TraceKey


@dataclass
class DeerPolicy:
    lam: float
    conf_mode: Literal["geo", "arith"]
    require_think_end: bool
    dp: DecisionPointCfg
    c_ans_max: int
    c_probe_max: int
    min_segment: int
    allowance: int | None = None  # hạn mức tổng chi phí T_i mỗi câu (A11); None: không giới hạn
    uses_probes: bool = True
    uses_rho: bool = False
    stop_at_markers: bool = True

    def _cap(self, s: QuestionState, base: int) -> int:
        """A11: max_tokens = min(base, a - spent_i - c_ans_max - c_probe_max)."""
        if self.allowance is None:
            return base
        return min(base, self.allowance - s.costs.total - self.c_ans_max - self.c_probe_max)

    def _go(self, s: QuestionState, min_tokens: int, base: int) -> Decision:
        cap = self._cap(s, base)
        if cap < self.min_segment:
            return Decision("stop")
        return Decision("continue", min_tokens, cap)

    def initial_request(self, s: QuestionState, pool: PoolView) -> Decision:
        return self._go(s, self.dp.b_min, self.dp.b_min + self.dp.l_max)

    def confident(self, s: QuestionState) -> bool:
        last = s.features.last
        if last is None:
            return False
        conf = last.conf if self.conf_mode == "geo" else last.conf_arith
        return conf >= self.lam and (not self.require_think_end or last.has_think_end)

    def decide(self, active: list[QuestionState], pool: PoolView) -> dict[TraceKey, Decision]:
        return {s.key: Decision("stop") if self.confident(s) else self._go(s, self.dp.m, self.dp.l_max) for s in active}
