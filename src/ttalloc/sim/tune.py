"""Hiệu chỉnh theo ngân sách (A11, P2.T13) và tìm kiếm ngẫu nhiên tham số (P3.T4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ttalloc.config import BatchCfg, MethodCfg
from ttalloc.sim.env import Draw, SimEnv

TARGET_LO = 0.98
MAX_ITERS = 30
TOL = 0.01
GRID = 6  # số điểm lưới bên trong (B/N, c_hi) khi c_hi chưa đạt mục tiêu


@dataclass(frozen=True)
class Calibration:
    allowance: int
    spend_ratio: float
    budget_underused: bool
    iters: int


def spend_ratio(env: SimEnv, m: MethodCfg, draws: list[Draw], allowance: int) -> float:
    return float(np.mean([env.run(m, d.keys, d.budget, allowance).spent.total / d.budget for d in draws]))


def calibrate_b2(env: SimEnv, m: MethodCfg, draws: list[Draw]) -> Calibration:
    """Chia đôi tìm hạn mức chung c trên [B/N, c_hi] sao cho chi tiêu trung bình nằm trong [0,98B; B] (A11).

    Bộ bảo vệ bảo đảm chi tiêu không vượt B, nên chỉ cần tìm c nhỏ nhất (trong sai số 1%) có tỉ lệ chi tiêu >= 0,98.
    Tỉ lệ chi tiêu không đơn điệu theo c: khi c lớn, quỹ chung cạn trước và bộ bảo vệ dừng các câu còn lại cùng lúc
    (phần chia đều < min_segment), nên chi tiêu tại c_hi có thể thấp hơn tại c vừa phải. Vì vậy nếu c_hi chưa đạt,
    thử thêm một lưới đều trên [B/N, c_hi] rồi chia đôi trên đoạn đầu tiên vượt mục tiêu (ADR-011). Không điểm nào
    đạt thì dùng c có tỉ lệ chi tiêu cao nhất và đánh dấu budget_underused.
    """
    lo = float(np.mean([d.budget / len(d.keys) for d in draws]))
    hi = float(env.c_hi())
    r_hi = spend_ratio(env, m, draws, int(hi))
    it = 1
    if r_hi < TARGET_LO:
        tried = [(hi, r_hi)]
        prev = lo
        for c in np.linspace(lo, hi, GRID + 2)[:-1]:
            r = spend_ratio(env, m, draws, int(c))
            it += 1
            tried.append((float(c), r))
            if r >= TARGET_LO:
                if c == lo:
                    return Calibration(int(lo), r, False, it)
                lo, hi, r_hi = prev, float(c), r
                break
            prev = float(c)
        else:
            c_best, r_best = max(tried, key=lambda t: (t[1], -t[0]))
            return Calibration(int(c_best), r_best, True, it)
    else:
        r_lo = spend_ratio(env, m, draws, int(lo))
        it += 1
        if r_lo >= TARGET_LO:
            return Calibration(int(lo), r_lo, False, it)
    width0 = hi - lo
    best = (int(hi), r_hi)
    while it < MAX_ITERS and hi - lo > TOL * width0:
        mid = (lo + hi) / 2
        r = spend_ratio(env, m, draws, int(mid))
        it += 1
        if r >= TARGET_LO:
            best = (int(mid), r)
            hi = mid
            break
        lo = mid
    return Calibration(best[0], best[1], False, it)


# ---------------------------------------------------------------- tìm kiếm ngẫu nhiên (P3.T4)


def sample_configs(m: MethodCfg, env: SimEnv, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Cấu hình ngẫu nhiên cho một phương pháp; chỉ các tham số không đòi học lại bộ ước lượng."""
    t = env.cfg.tune
    out = []
    for _ in range(n):
        if m.policy == "deer":
            out.append({"lambda": float(rng.uniform(t.lambda_range[0], t.lambda_range[1]))})
        elif m.policy == "price_adaptive":
            lo, hi = (math.log(x) for x in t.mu_min_range)
            out.append(
                {"mu_min": float(math.exp(rng.uniform(lo, hi))), "b_max_mult": float(rng.choice(t.b_max_mult_choices))}
            )
        else:
            raise ValueError(f"không có không gian tìm kiếm cho chính sách {m.policy}")
    return out


def evaluate_config(
    env: SimEnv, m: MethodCfg, params: dict[str, Any], batch: BatchCfg, alphas: list[float], n_draws: int
) -> dict[str, Any]:
    mm = m.model_copy(update={"params": {**m.params, **params}})
    accs = {}
    extra: dict[str, Any] = {}
    for a in alphas:
        draws = env.draws(batch, a, n_draws)
        allowance = None
        if mm.policy == "deer" and mm.params.get("cap", "calibrated") == "calibrated":
            cal = calibrate_b2(env, mm, draws)
            allowance = cal.allowance
            extra[str(a)] = {"allowance": cal.allowance, "spend_ratio": cal.spend_ratio}
        accs[str(a)] = float(np.mean([env.run(mm, d.keys, d.budget, allowance).accuracy for d in draws]))
    return {"params": params, "acc": accs, "objective": float(np.mean(list(accs.values()))), "calibration": extra}


def random_search(
    env: SimEnv, m: MethodCfg, batch: BatchCfg, alphas: list[float], seed: int | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mỗi phương pháp cùng số cấu hình (`tune.n_configs`); mục tiêu là độ chính xác trung bình trên lưới alpha."""
    t = env.cfg.tune
    rng = np.random.default_rng([t.seed if seed is None else seed, sum(map(ord, m.id))])
    trials = [evaluate_config(env, m, p, batch, alphas, t.draws) for p in sample_configs(m, env, t.n_configs, rng)]
    best = max(trials, key=lambda r: (r["objective"], -trials.index(r)))
    return best, trials
