"""Môi trường mô phỏng: dữ liệu ghi của một tập, bộ ước lượng, bảng tính sẵn và cách dựng chính sách."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ttalloc.config import BatchCfg, ExperimentCfg, MethodCfg, check_b_max, load_model_cfg
from ttalloc.controller.runner import BatchResult, RunnerCfg, run_batch
from ttalloc.estimator.train import Estimator
from ttalloc.features.curve import EmpiricalCurves
from ttalloc.features.events import LoggedChain
from ttalloc.policies.deer import DeerPolicy
from ttalloc.policies.price import PricePolicy
from ttalloc.sim.logged_source import LoggedSource
from ttalloc.sim.providers import CurveRhoProvider, EstimatorRhoProvider, TableRhoProvider
from ttalloc.types import Policy, Question, RhoProvider, TraceKey


@dataclass(frozen=True)
class Draw:
    draw: int
    keys: list[TraceKey]
    budget: int
    c_full: float


def batch_name(b: BatchCfg) -> str:
    if b.composition == "mix":
        return "mix_" + "_".join(f"{k}{v}" for k, v in sorted(b.sources.items()))
    return b.composition if b.n is None else f"{b.composition}{b.n}"


def resolve_params(m: MethodCfg, cfg: ExperimentCfg) -> dict[str, Any]:
    """Thay giá trị `tuned` bằng giá trị đã chỉnh (tệp best.json trong thư mục `tuned_from`)."""
    out = dict(m.params)
    tuned = [k for k, v in out.items() if v == "tuned"]
    if tuned:
        if not cfg.tuned_from:
            raise ValueError(f"{m.id}: tham số {tuned} là 'tuned' nhưng config không có tuned_from")
        best = json.loads((Path(cfg.tuned_from) / "best.json").read_text(encoding="utf-8"))
        for k in tuned:
            out[k] = best[m.id][k]
    return out


@dataclass
class SimEnv:
    cfg: ExperimentCfg
    split: str
    chains: dict[TraceKey, LoggedChain]
    questions: dict[str, Question]
    c_ans_max: int
    c_probe_max: int
    ref_median: float
    estimator: Estimator | None = None
    omega: float = 0.0
    table: Any = None  # pandas.DataFrame của bảng tính sẵn
    paths: dict[TraceKey, list[int]] | None = None
    conf_mode: str = "geo"
    require_think_end: bool = False
    seeds_by_q: dict[str, list[int]] = field(default_factory=dict)
    _curves: EmpiricalCurves | None = None
    _cfull: dict[tuple[str, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        by: dict[str, list[int]] = {}
        for k in self.chains:
            by.setdefault(k.qid, []).append(k.seed)
        self.seeds_by_q = {q: sorted(v) for q, v in by.items()}
        missing = [q for q in self.questions if q not in self.seeds_by_q]
        if missing:
            raise ValueError(f"{self.split}: {len(missing)} câu chưa có dữ liệu ghi, ví dụ {missing[:3]}")
        if self.cfg.model:
            try:
                mc = load_model_cfg(self.cfg.model)
                self.conf_mode, self.require_think_end = mc.deer_conf_mode, mc.deer_require_think_end_after_probe
            except FileNotFoundError:
                pass

    # ---------------------------------------------------------------- tham số dẫn xuất

    @property
    def l_log(self) -> int:
        return self.cfg.log.l_log

    def b_max_for(self, policy: str, params: dict[str, Any]) -> int:
        cap = self.l_log - self.c_ans_max
        if policy == "price_adaptive":
            mult = params.get("b_max_mult", self.cfg.policy.b_max_mult)
            b = cap if mult is None else min(int(float(mult) * self.ref_median), cap)
        else:
            b = cap
        check_b_max(self.cfg, b, self.c_ans_max)
        return b

    def runner_cfg(self, b_max: int) -> RunnerCfg:
        mx = self.cfg.sim.max_active
        return RunnerCfg(
            b_max=b_max,
            min_segment=self.cfg.guard.min_segment,
            c_ans_max=self.c_ans_max,
            c_probe_max=self.c_probe_max,
            b_min=self.cfg.dp.b_min,
            run_seed=self.cfg.run_seed,
            k=self.cfg.features.k,
            max_active=mx,
        )

    def c_hi(self) -> int:
        """Cận trên khi chia đôi tìm hạn mức chung của B2 (A11)."""
        return self.l_log + self.c_ans_max + math.ceil(self.l_log / self.cfg.dp.m) * self.c_probe_max

    @property
    def curves(self) -> EmpiricalCurves:
        if self._curves is None:
            self._curves = EmpiricalCurves(self.chains, self.cfg.dp)
        return self._curves

    # ---------------------------------------------------------------- lô và lần rút

    def c_full(self, qids: list[str]) -> float:
        """C_full: tổng theo câu của trung bình qua seed của chain_len + chi phí câu trả lời tự nhiên (mục 7.3)."""
        tot = 0.0
        for q in qids:
            vals = []
            for s in self.seeds_by_q[q]:
                c = self.chains[TraceKey(q, s)]
                ans = c.nat_cost() if c.natural_end else c.answer_cost(c.n_points - 1)
                vals.append(c.chain_len + ans)
            tot += float(np.mean(vals))
        return tot

    def draw_rng(self, draw: int) -> np.random.Generator:
        return np.random.default_rng([self.cfg.sim.seed, draw])

    def draw_keys(self, qids: list[str], rng: np.random.Generator) -> list[TraceKey]:
        return [TraceKey(q, int(rng.choice(self.seeds_by_q[q]))) for q in qids]

    def draws(self, batch: BatchCfg, alpha: float, n: int, start: int = 0) -> list[Draw]:
        """Các lần rút dùng chung cho mọi phương pháp: lô dựng lại bằng seed của lần rút, rồi chọn một seed mỗi câu."""
        from ttalloc.data.batches import build_batch

        qs = list(self.questions.values())
        out = []
        for d in range(start, start + n):
            rng = self.draw_rng(d)
            qids = build_batch(qs, batch, rng)
            keys = self.draw_keys(qids, rng)
            key = (batch_name(batch), d)
            if key not in self._cfull:
                self._cfull[key] = self.c_full(qids)
            out.append(Draw(d, keys, int(alpha * self._cfull[key]), self._cfull[key]))
        return out

    # ---------------------------------------------------------------- chính sách và RhoProvider

    def make(self, m: MethodCfg, allowance: int | None = None) -> tuple[Policy, RhoProvider | None, int]:
        params = resolve_params(m, self.cfg)
        b_max = self.b_max_for(m.policy, params)
        if m.policy == "deer":
            lam = float(params.get("lambda", self.cfg.deer.lambda_))
            pol: Policy = DeerPolicy(
                lam=lam,
                conf_mode=self.conf_mode,  # type: ignore[arg-type]
                require_think_end=self.require_think_end,
                dp=self.cfg.dp,
                c_ans_max=self.c_ans_max,
                c_probe_max=self.c_probe_max,
                min_segment=self.cfg.guard.min_segment,
                allowance=allowance,
            )
            return pol, None, b_max
        if m.policy == "price_adaptive":
            mu_min = params.get("mu_min", self.cfg.policy.mu_min)
            if mu_min is None:
                raise ValueError(f"{m.id}: chưa có mu_min (đặt policy.mu_min hoặc params.mu_min, hoặc chạy tune)")
            pol = PricePolicy(dp=self.cfg.dp, mu_min=float(mu_min), b_max=b_max, run_seed=self.cfg.run_seed)
            return pol, self.provider(m, b_max), b_max
        raise ValueError(f"chính sách chưa cài: {m.policy} (các phương pháp để sau thuộc P5.T1)")

    def provider(self, m: MethodCfg, b_max: int) -> RhoProvider:
        horizons = self.cfg.policy.horizons
        if m.rho_provider == "curve":
            return CurveRhoProvider(self.curves, horizons, b_max, self.omega)
        if self.estimator is None:
            raise ValueError(f"{m.id}: cần bộ ước lượng (ttalloc estimator train) nhưng chưa nạp")
        est = EstimatorRhoProvider(self.estimator, horizons, b_max, self.omega)
        if self.table is not None and self.paths is not None and m.rho_provider in (None, "table"):
            return TableRhoProvider(self.table, self.paths, est)
        return est

    def run(self, m: MethodCfg, keys: list[TraceKey], budget: int, allowance: int | None = None) -> BatchResult:
        pol, prov, b_max = self.make(m, allowance)
        src = LoggedSource(
            {k: self.chains[k] for k in keys},
            on_stop=self.cfg.answer.on_stop,
            answer_types={q: self.questions[q].answer_type for q in {k.qid for k in keys}},
        )
        return run_batch(keys, pol, src, prov, self.runner_cfg(b_max), budget)
