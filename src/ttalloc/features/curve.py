"""Đường cong đúng sai thực nghiệm q_i(b) từ dữ liệu ghi (dùng cho M3, E1 và P2.T20).

Với một chuỗi, "đúng khi dừng tại b" là kết quả câu trả lời tại điểm cuối cùng của dãy chuẩn có `gen_end <= b`; nếu
chuỗi tự kết thúc trước b thì dùng câu trả lời tự nhiên (cùng quy ước với nhãn A4). Trước điểm đầu tiên của dãy chuẩn
dùng câu trả lời tại điểm start.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ttalloc.config import DecisionPointCfg
from ttalloc.features.events import LoggedChain, canonical_points
from ttalloc.types import TraceKey


@dataclass(frozen=True)
class StepCurve:
    ge: np.ndarray  # gen_end của các điểm, bắt đầu bằng 0 (điểm start)
    ok: np.ndarray  # đúng sai tại các điểm đó
    nat_len: int | None
    nat_ok: bool

    def at(self, b: float) -> float:
        if self.nat_len is not None and self.nat_len <= b:
            return float(self.nat_ok)
        i = int(np.searchsorted(self.ge, b, side="right")) - 1
        return float(self.ok[max(i, 0)])

    def at_many(self, bs: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(self.ge, bs, side="right") - 1, 0, None)
        out = self.ok[i].astype(float)
        if self.nat_len is not None:
            out = np.where(bs >= self.nat_len, float(self.nat_ok), out)
        return out


def step_curve(c: LoggedChain, dp: DecisionPointCfg) -> StepCurve:
    sub = canonical_points(c, dp)
    idx = [0, *sub]
    return StepCurve(
        ge=c.gen_end[idx].astype(float),
        ok=c.ans_correct[idx].astype(bool),
        nat_len=c.chain_len if c.natural_end else None,
        nat_ok=bool(c.nat_correct),
    )


class EmpiricalCurves:
    def __init__(self, chains: dict[TraceKey, LoggedChain], dp: DecisionPointCfg) -> None:
        self.by_q: dict[str, dict[int, StepCurve]] = defaultdict(dict)
        for k, c in chains.items():
            self.by_q[k.qid][k.seed] = step_curve(c, dp)

    def q(self, qid: str, b: float, exclude_seed: int | None = None) -> float | None:
        """Trung bình trên các seed (trừ `exclude_seed`); None nếu không còn seed nào."""
        curves = [cv for s, cv in self.by_q[qid].items() if s != exclude_seed]
        if not curves:
            return None
        return float(np.mean([cv.at(b) for cv in curves]))

    def q_many(self, qid: str, bs: np.ndarray, exclude_seed: int | None = None) -> np.ndarray | None:
        curves = [cv for s, cv in self.by_q[qid].items() if s != exclude_seed]
        if not curves:
            return None
        return np.mean([cv.at_many(bs) for cv in curves], axis=0)
