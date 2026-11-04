"""Quỹ và bộ bảo vệ ngân sách (PLAN A8, P2.T7)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ttalloc.policies.base import tie_key
from ttalloc.types import Costs, Decision, QuestionState, TraceKey


class InfeasibleBudget(RuntimeError):
    pass


class BudgetExceeded(AssertionError):
    """Vi phạm I1. Không bao giờ được xảy ra; nếu xảy ra là lỗi blocker."""


COST_CATEGORIES = ("think", "probe_prompt", "probe_output", "answer_prompt", "answer_output")


@dataclass
class Pool:
    budget: int
    c_ans_max: int
    c_probe_max: int
    b_min: int
    spent: Costs = field(default_factory=Costs)
    n_active: int = 0
    n_waiting: int = 0

    def remaining(self) -> int:
        return self.budget - self.spent.total

    def reserve(self) -> int:
        # Mỗi câu đang chạy hoặc đang chờ giữ một câu trả lời; câu đang chờ giữ thêm một đoạn b_min và một lần gợi.
        return (self.n_active + self.n_waiting) * self.c_ans_max + self.n_waiting * (self.b_min + self.c_probe_max)

    def available(self) -> int:
        return self.remaining() - self.reserve()

    def charge(self, category: str, n: int) -> None:
        if category not in COST_CATEGORIES:
            raise ValueError(f"loại chi phí lạ: {category}")
        if n < 0:
            raise ValueError(f"chi phí âm: {category}={n}")
        setattr(self.spent, category, getattr(self.spent, category) + int(n))
        if self.spent.total > self.budget:
            raise BudgetExceeded(f"I1: chi tiêu {self.spent.total} > B = {self.budget} (sau {category} +{n})")

    def check_feasible(self, n_total: int) -> None:
        saved = (self.n_active, self.n_waiting)
        self.n_active, self.n_waiting = 0, n_total
        try:
            if self.available() < 0:
                raise InfeasibleBudget(
                    f"B = {self.budget} không đủ phần dự trữ {self.reserve()} cho {n_total} câu "
                    f"(c_ans_max={self.c_ans_max}, b_min={self.b_min}, c_probe_max={self.c_probe_max})"
                )
        finally:
            self.n_active, self.n_waiting = saved

    def check_round_invariants(self) -> None:
        if self.spent.total > self.budget:
            raise BudgetExceeded(f"I1: chi tiêu {self.spent.total} > B = {self.budget}")
        if self.available() < 0:
            raise BudgetExceeded(f"quỹ dùng được âm ({self.available()}) ở cuối vòng")


def guard(
    decisions: dict[TraceKey, Decision],
    active: list[QuestionState],
    pool: Pool,
    uses_probes: bool,
    b_max: int,
    min_segment: int,
    c_probe_max: int,
    run_seed: int,
) -> tuple[dict[TraceKey, Decision], dict[TraceKey, tuple[int, int]]]:
    """Cấp hạn mức cho các câu tiếp tục sao cho chi tiêu xấu nhất của vòng cộng phần dự trữ không vượt quỹ (A8)."""
    decisions = dict(decisions)
    avail = pool.available()
    cp = c_probe_max if uses_probes else 0
    cont = [s for s in active if decisions[s.key].action == "continue"]
    want = {s.key: min(decisions[s.key].max_tokens, b_max - s.gen_end) for s in cont}
    allot: dict[TraceKey, tuple[int, int]] = {}

    def give(s: QuestionState, cap: int) -> None:
        nonlocal avail
        if cap < min_segment:
            decisions[s.key] = Decision("stop")
            return
        allot[s.key] = (min(decisions[s.key].min_tokens, cap), cap)
        avail -= cap + cp

    # 1) Câu chưa có rho (vòng đầu, câu mới nạp, hoặc chính sách không dùng rho): chia đều khi không đủ.
    fresh = sorted((s for s in cont if s.rho is None), key=lambda s: tie_key(s.key, run_seed))
    for s in fresh:
        if want[s.key] < min_segment:
            give(s, want[s.key])  # chạm b_max: dừng
    fresh = [s for s in fresh if want[s.key] >= min_segment]
    if sum(want[s.key] + cp for s in fresh) <= avail:
        for s in fresh:
            give(s, want[s.key])
    else:
        share = avail // max(len(fresh), 1)
        if share - cp < min_segment:
            # Phần chia đều không đủ min_segment: chỉ cấp cho số câu nhiều nhất còn đủ (theo tie_key), các câu còn lại
            # dừng. Nếu dừng tất cả, quỹ còn lại bị bỏ phí và B2 bị thiệt so với M2 (ADR-012).
            n_fund = avail // (min_segment + cp)
            for s in fresh[n_fund:]:
                decisions[s.key] = Decision("stop")
            fresh = fresh[:n_fund]
            share = avail // max(n_fund, 1)
        for s in fresh:
            give(s, min(want[s.key], share - cp))
    # 2) Câu đã có rho: cấp theo rho giảm dần từ phần còn lại.
    ranked = sorted((s for s in cont if s.rho is not None), key=lambda s: (-s.rho, tie_key(s.key, run_seed)))  # type: ignore[operator]
    for s in ranked:
        give(s, min(want[s.key], avail - cp))
    return decisions, allot
