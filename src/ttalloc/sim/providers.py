"""Các RhoProvider (PLAN mục 3.4, P2.T10): bộ ước lượng, bảng tính sẵn, đường cong thực nghiệm (M3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ttalloc.estimator.rho import rho_hstar
from ttalloc.estimator.train import Estimator
from ttalloc.features.curve import EmpiricalCurves
from ttalloc.types import QuestionState, TraceKey


def _set(s: QuestionState, hs: list[int], q: np.ndarray, horizons: list[int], b_max: int, omega: float) -> None:
    qd = dict(zip(hs, (float(x) for x in q), strict=True))
    s.qhat = np.asarray(q, float)
    s.rho, s.hstar = rho_hstar(qd, s.gen_end, horizons, b_max, omega)


class EstimatorRhoProvider:
    def __init__(self, estimator: Estimator, horizons: list[int], b_max: int, omega: float) -> None:
        self.est = estimator
        self.horizons = list(horizons)
        self.hs = [0, *self.horizons]
        self.b_max = b_max
        self.omega = omega
        self.n_calls = 0

    def predict_states(self, states: list[QuestionState]) -> np.ndarray:
        X = pd.DataFrame([s.features.vector() for s in states])
        self.n_calls += 1
        return self.est.predict(X, self.hs)

    def fill(self, states: list[QuestionState]) -> None:
        if not states:
            return
        q = self.predict_states(states)
        for s, row in zip(states, q, strict=True):
            _set(s, self.hs, row, self.horizons, self.b_max, self.omega)


class TableRhoProvider:
    """Dùng bảng tính sẵn khi lịch sử điểm đã ghé trùng tiền tố của dãy chuẩn; nếu không, gọi bộ ước lượng."""

    def __init__(
        self,
        table: pd.DataFrame,
        canonical: dict[TraceKey, list[int]],
        fallback: EstimatorRhoProvider,
    ) -> None:
        self.fallback = fallback
        self.horizons = fallback.horizons
        self.hs = fallback.hs
        cols = [f"q_{h}" for h in self.hs]
        missing = [c for c in cols if c not in table.columns]
        if missing:
            raise ValueError(f"bảng tính sẵn thiếu cột {missing}")
        vals = table[cols].to_numpy(float)
        self.lookup: dict[tuple[str, int, int], np.ndarray] = {
            (q, int(s), int(p)): vals[i]
            for i, (q, s, p) in enumerate(zip(table.qid, table.seed, table.point_idx, strict=True))
        }
        self.canonical = canonical
        self.n_table = 0
        self.n_fallback = 0

    def fill(self, states: list[QuestionState]) -> None:
        miss = []
        for s in states:
            canon = self.canonical.get(s.key, [])
            n = len(s.visited)
            row = None
            if n and s.visited == canon[:n]:
                row = self.lookup.get((s.key.qid, s.key.seed, s.visited[-1]))
            if row is None:
                miss.append(s)
                continue
            self.n_table += 1
            _set(s, self.hs, row, self.horizons, self.fallback.b_max, self.fallback.omega)
        if miss:
            self.n_fallback += len(miss)
            self.fallback.fill(miss)


class CurveRhoProvider:
    """M3: q_i(b) là tỉ lệ đúng thực nghiệm, trung bình trên các seed KHÁC seed đang mô phỏng (cố ý nhìn trước)."""

    def __init__(self, curves: EmpiricalCurves, horizons: list[int], b_max: int, omega: float) -> None:
        self.curves = curves
        self.horizons = list(horizons)
        self.hs = [0, *self.horizons]
        self.b_max = b_max
        self.omega = omega

    def fill(self, states: list[QuestionState]) -> None:
        for s in states:
            bs = np.array([s.gen_end + h for h in self.hs], float)
            q = self.curves.q_many(s.key.qid, bs, exclude_seed=s.key.seed)
            if q is None:
                # Câu chỉ có một seed: không có thông tin từ seed khác, coi như không có lợi ích.
                s.qhat, s.rho, s.hstar = None, float("-inf"), None
                continue
            _set(s, self.hs, q, self.horizons, self.b_max, self.omega)


def describe(provider: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type(provider).__name__}
    for attr in ("n_table", "n_fallback", "n_calls"):
        if hasattr(provider, attr):
            out[attr] = getattr(provider, attr)
    return out
