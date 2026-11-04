"""P2.T7, P2.T8, P2.T11: quỹ, bộ bảo vệ, Runner, chính sách B2/M2/M3."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sim_helpers import C_ANS_MAX, C_PROBE_MAX, synth_cfg, synth_env
from ttalloc.config import DecisionPointCfg, MethodCfg
from ttalloc.controller.budget import BudgetExceeded, InfeasibleBudget, Pool, guard
from ttalloc.controller.runner import BatchResult, RunnerCfg, run_batch
from ttalloc.estimator.rho import rho_hstar
from ttalloc.features.extract import FeatureState
from ttalloc.policies.base import tie_key
from ttalloc.policies.deer import DeerPolicy
from ttalloc.policies.price import PricePolicy, adaptive_price
from ttalloc.sim.logged_source import LoggedSource
from ttalloc.sim.providers import CurveRhoProvider
from ttalloc.types import Decision, ProbeResult, QuestionState, TraceKey

ENV = synth_env(n=16, seeds=[0, 1, 2], seed=3)
DP = DecisionPointCfg()
B2 = MethodCfg(id="B2", policy="deer", params={"lambda": 0.9})
M3 = MethodCfg(id="M3", policy="price_adaptive", rho_provider="curve", params={"mu_min": 0.0})


class _View:
    def __init__(self, avail: int) -> None:
        self.avail = avail

    def available(self) -> int:
        return self.avail


@dataclass
class RandomPolicy:
    """Chính sách ngẫu nhiên để kiểm bất biến của Runner không phụ thuộc chính sách."""

    seed: int
    uses_probes: bool = True
    uses_rho: bool = False
    stop_at_markers: bool = True
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def _d(self) -> Decision:
        if self.rng.random() < 0.15:
            return Decision("stop")
        return Decision("continue", self.rng.randint(0, 600), self.rng.randint(1, 3000))

    def initial_request(self, s: QuestionState, pool: object) -> Decision:
        return self._d()

    def decide(self, active: list[QuestionState], pool: object) -> dict[TraceKey, Decision]:
        return {s.key: self._d() for s in active}


def _keys(env=ENV, draw: int = 0) -> list[TraceKey]:  # type: ignore[no-untyped-def]
    return env.draw_keys(sorted(env.questions), env.draw_rng(draw))


def _min_budget(n: int) -> int:
    return n * (C_ANS_MAX + DP.b_min + C_PROBE_MAX)


def _run(policy, keys, budget, provider=None, max_active=None, run_seed=0) -> BatchResult:  # type: ignore[no-untyped-def]
    rc = RunnerCfg(
        b_max=ENV.l_log - C_ANS_MAX,
        min_segment=256,
        c_ans_max=C_ANS_MAX,
        c_probe_max=C_PROBE_MAX,
        b_min=DP.b_min,
        run_seed=run_seed,
        max_active=max_active,
    )
    src = LoggedSource({k: ENV.chains[k] for k in keys})
    return run_batch(keys, policy, src, provider, rc, budget)


# ---------------------------------------------------------------- quỹ và bộ bảo vệ (P2.T7)


def test_pool_reserve_and_infeasible() -> None:
    p = Pool(budget=1000, c_ans_max=40, c_probe_max=20, b_min=500, n_active=2, n_waiting=1)
    assert p.reserve() == 3 * 40 + 520 and p.available() == 1000 - 640
    with pytest.raises(InfeasibleBudget):
        Pool(1000, 40, 20, 500).check_feasible(n_total=2)  # 2 * (40 + 500 + 20) > 1000
    Pool(1120, 40, 20, 500).check_feasible(n_total=2)
    with pytest.raises(BudgetExceeded):
        p.charge("think", 1001)
    with pytest.raises(ValueError):
        p.charge("nope", 1)


def _state(i: int, gen_end: int, rho: float | None) -> QuestionState:
    s = QuestionState(TraceKey(f"q{i}", 0), gen_end=gen_end)
    s.rho = rho
    return s


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(1, 12),
    budget=st.integers(0, 40000),
    seed=st.integers(0, 10**6),
    uses_probes=st.booleans(),
)
def test_guard_worst_case_within_pool(n: int, budget: int, seed: int, uses_probes: bool) -> None:
    rng = random.Random(seed)
    states = [
        _state(i, rng.randint(0, 3000), None if rng.random() < 0.4 else rng.uniform(-1e-4, 1e-3)) for i in range(n)
    ]
    pool = Pool(budget=budget + n * 50, c_ans_max=50, c_probe_max=20, b_min=512, n_active=n)
    decs = {
        s.key: Decision("continue", rng.randint(0, 500), rng.randint(1, 4000))
        if rng.random() < 0.8
        else Decision("stop")
        for s in states
    }
    min_seg, b_max = 256, 4000
    out, allot = guard(decs, states, pool, uses_probes, b_max, min_seg, 20, run_seed=seed % 7)
    cp = 20 if uses_probes else 0
    worst = sum(cap + cp for _, cap in allot.values())
    assert worst + pool.reserve() <= pool.remaining()
    for k, (mn, cap) in allot.items():
        assert out[k].action == "continue" and min_seg <= cap and 0 <= mn <= cap
        s = next(x for x in states if x.key == k)
        assert s.gen_end + cap <= b_max
    for s in states:
        assert (out[s.key].action == "continue") == (s.key in allot)
    # Câu chưa có rho được cấp trước: nếu một câu đã có rho được cấp thì mọi câu chưa có rho muốn tiếp tục đều được cấp.
    want = {s.key: min(decs[s.key].max_tokens, b_max - s.gen_end) for s in states if decs[s.key].action == "continue"}
    ranked = [s for s in states if s.rho is not None and s.key in want]
    fresh = [s for s in states if s.rho is None and s.key in want and want[s.key] >= min_seg]
    if any(s.key in allot for s in ranked):
        assert all(s.key in allot for s in fresh)
    # Không đủ cho mọi câu chưa có rho: cấp cho số câu nhiều nhất còn đủ min_segment (ADR-012), không bỏ phí quỹ.
    n_fresh = sum(s.key in allot for s in fresh)
    assert n_fresh == min(len(fresh), max(pool.available(), 0) // (min_seg + cp)) or n_fresh == len(fresh)
    # Trong nhóm đã có rho: câu rho thấp được cấp thì câu rho cao hơn (đủ chỗ cho min_segment) cũng được cấp.
    for lo in ranked:
        if lo.key in allot:
            for hi in ranked:
                if hi.rho > lo.rho and want[hi.key] >= min_seg:  # type: ignore[operator]
                    assert hi.key in allot


def test_guard_stops_everyone_when_budget_tiny() -> None:
    states = [_state(i, 100, None) for i in range(5)]
    pool = Pool(budget=5 * 50 + 100, c_ans_max=50, c_probe_max=20, b_min=512, n_active=5)
    decs = {s.key: Decision("continue", 0, 1000) for s in states}
    out, allot = guard(decs, states, pool, True, 10**6, 256, 20, 0)
    assert not allot and all(d.action == "stop" for d in out.values())


def test_ties_broken_by_seeded_hash() -> None:
    a, b = _state(0, 0, 1e-4), _state(1, 0, 1e-4)
    winners = set()
    for run_seed in range(30):
        pool = Pool(budget=2 * 50 + 1100, c_ans_max=50, c_probe_max=20, b_min=512, n_active=2)
        decs = {s.key: Decision("continue", 0, 1000) for s in (a, b)}
        _, allot = guard(decs, [a, b], pool, True, 10**6, 256, 20, run_seed)
        assert len(allot) == 1
        (k,) = allot
        assert k == min((a.key, b.key), key=lambda x: tie_key(x, run_seed))
        winners.add(k)
    assert winners == {a.key, b.key}  # thứ tự đổi theo run_seed, không cố định theo qid


# ---------------------------------------------------------------- Runner (P2.T8)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), frac=st.floats(0.0, 1.5), max_active=st.sampled_from([None, 3, 7]))
def test_budget_never_exceeded(seed: int, frac: float, max_active: int | None) -> None:
    keys = _keys(draw=seed % 5)
    budget = _min_budget(len(keys)) + int(frac * ENV.c_full([k.qid for k in keys]))
    res = _run(RandomPolicy(seed), keys, budget, max_active=max_active, run_seed=seed % 3)
    assert res.spent.total <= budget  # I1
    got = sorted((o.key.qid, o.key.seed) for o in res.outcomes)
    assert got == sorted((k.qid, k.seed) for k in keys)  # I3: mỗi câu đúng một câu trả lời
    for cat in ("think", "probe_prompt", "probe_output", "answer_prompt", "answer_output"):  # I2
        assert sum(getattr(o.costs, cat) for o in res.outcomes) == getattr(res.spent, cat)


def test_infeasible_budget_raises() -> None:
    keys = _keys()
    with pytest.raises(InfeasibleBudget):
        _run(RandomPolicy(0), keys, _min_budget(len(keys)) - 1)


def test_sliding_window_reserve() -> None:
    keys = _keys()
    res = _run(
        DeerPolicy(0.9, "geo", False, DP, C_ANS_MAX, C_PROBE_MAX, 256),
        keys,
        _min_budget(len(keys)) + 20000,
        max_active=4,
    )
    assert max(r["n_active"] for r in res.rounds) <= 4
    assert res.spent.total <= res.budget and len(res.outcomes) == len(keys)


def test_sliding_window_newcomers_not_starved() -> None:
    keys = _keys()
    prov = CurveRhoProvider(ENV.curves, ENV.cfg.policy.horizons, ENV.l_log - C_ANS_MAX, 0.0)
    pol = PricePolicy(DP, mu_min=0.0, b_max=ENV.l_log - C_ANS_MAX, run_seed=0)
    for extra in (0, 3000, 30000):
        res = _run(pol, keys, _min_budget(len(keys)) + extra, provider=prov, max_active=3)
        assert all(o.costs.think >= 256 or o.stop_kind in ("natural_end", "eos") for o in res.outcomes)
        assert all(o.n_rounds >= 1 for o in res.outcomes)


def test_deterministic() -> None:
    keys = _keys()
    budget = _min_budget(len(keys)) + 15000
    r1, r2 = (_run(RandomPolicy(5), keys, budget, run_seed=1) for _ in range(2))
    assert [(o.key, o.correct, o.costs.as_dict(), o.stop_point_idx) for o in r1.outcomes] == [
        (o.key, o.correct, o.costs.as_dict(), o.stop_point_idx) for o in r2.outcomes
    ]


def test_runner_rejects_duplicate_keys() -> None:
    k = _keys()[:1]
    with pytest.raises(ValueError):
        _run(RandomPolicy(0), k * 2, 10**6)


# ---------------------------------------------------------------- chính sách (P2.T11)

WORKED = {
    "Q1": [0.92, 0.93, 0.93, 0.93],
    "Q2": [0.40, 0.55, 0.85, 0.88],
    "Q3": [0.10, 0.12, 0.15, 0.50],
    "Q4": [0.05, 0.05, 0.06, 0.06],
}
HZ = [1000, 2000, 4000]


def _worked_states(names: dict[str, list[float]]) -> list[QuestionState]:
    out = []
    for name, q in names.items():
        s = QuestionState(TraceKey(name, 0), gen_end=1000)
        s.rho, s.hstar = rho_hstar(dict(zip([0, *HZ], q, strict=True)), 1000, HZ, b_max=10**9, omega=0.0)
        out.append(s)
    return out


def _price() -> PricePolicy:
    return PricePolicy(DecisionPointCfg(), mu_min=0.0, b_max=10**9, run_seed=0)


def test_m2_worked_example_round1() -> None:
    pol = _price()
    dec = pol.decide(_worked_states(WORKED), _View(5080))
    assert pol.last_mu == pytest.approx(1e-4)  # 10 điểm phần trăm trên 1.000 token
    assert {k.qid: d.action for k, d in dec.items()} == {"Q1": "stop", "Q2": "continue", "Q3": "continue", "Q4": "stop"}


def test_m2_worked_example_round2() -> None:
    pol = _price()
    states = _worked_states({"Q2": [0.88, 0.88, 0.87, 0.86], "Q3": [0.15, 0.35, 0.50, 0.70]})
    assert states[0].rho == 0.0
    assert states[1].rho == pytest.approx(2e-4) and states[1].hstar == 1000
    dec = pol.decide(states, _View(10000))
    assert pol.last_mu == 0.0
    assert dec[TraceKey("Q2", 0)].action == "stop" and dec[TraceKey("Q3", 0)].action == "continue"


def test_adaptive_price_zero_when_everything_fits() -> None:
    assert adaptive_price(_worked_states(WORKED), 10**6, 0) == 0.0


def test_price_policy_requires_rho_and_respects_b_max() -> None:
    s = QuestionState(TraceKey("q", 0), gen_end=5000)
    with pytest.raises(ValueError):
        _price().decide([s], _View(10**6))
    s.rho, s.hstar = 1e-3, 512
    pol = PricePolicy(DecisionPointCfg(), mu_min=0.0, b_max=5000, run_seed=0)
    assert pol.decide([s], _View(10**6))[s.key].action == "stop"


def _with_probe(
    conf: float, think_end: bool, costs: tuple[int, int] = (0, 0), arith: float | None = None
) -> QuestionState:
    s = QuestionState(TraceKey("q", 0), gen_end=1000, features=FeatureState(k=4))
    ca = conf if arith is None else arith
    s.features.update(1000, ProbeResult(s.key, "1", "int:1", conf, ca, (0.0,), 6, 4, think_end), 0, None)
    s.costs.think, s.costs.probe_prompt = costs
    return s


def _deer(require_think_end: bool = False, allowance: int | None = None, mode: str = "geo") -> DeerPolicy:
    return DeerPolicy(
        0.95,
        mode,
        require_think_end,
        DecisionPointCfg(),
        C_ANS_MAX,
        C_PROBE_MAX,
        256,  # type: ignore[arg-type]
        allowance=allowance,
    )


def test_deer_stops_at_threshold() -> None:
    assert _deer().decide([_with_probe(0.95, False)], _View(0))[TraceKey("q", 0)].action == "stop"
    d = _deer().decide([_with_probe(0.94, False)], _View(0))[TraceKey("q", 0)]
    assert d.action == "continue" and (d.min_tokens, d.max_tokens) == (DP.m, DP.l_max)
    s = _with_probe(0.93, False, arith=0.97)  # trung bình cộng >= trung bình nhân
    assert _deer(mode="arith").decide([s], _View(0))[s.key].action == "stop"
    assert _deer(mode="geo").decide([s], _View(0))[s.key].action == "continue"


def test_deer_qwen3_requires_think_end() -> None:
    assert _deer(True).decide([_with_probe(0.99, False)], _View(0))[TraceKey("q", 0)].action == "continue"
    assert _deer(True).decide([_with_probe(0.99, True)], _View(0))[TraceKey("q", 0)].action == "stop"


def test_allowance_is_on_total_cost() -> None:
    s = _with_probe(0.1, False, costs=(1000, 200))
    d = _deer(allowance=2000).decide([s], _View(0))[s.key]
    assert d.max_tokens == 2000 - 1200 - C_ANS_MAX - C_PROBE_MAX  # A11: tính trên T_i, gồm cả chi phí gợi
    s2 = _with_probe(0.1, False, costs=(1500, 200))
    assert _deer(allowance=2000).decide([s2], _View(0))[s2.key].action == "stop"  # còn < min_segment
    assert _deer(allowance=300).initial_request(QuestionState(TraceKey("x", 0)), _View(0)).action == "stop"


@dataclass
class Recording:
    inner: DeerPolicy | PricePolicy
    log: list[tuple[int, int, str, int, int]] = field(default_factory=list)

    @property
    def uses_probes(self) -> bool:
        return self.inner.uses_probes

    @property
    def uses_rho(self) -> bool:
        return self.inner.uses_rho

    @property
    def stop_at_markers(self) -> bool:
        return self.inner.stop_at_markers

    def initial_request(self, s: QuestionState, pool: object) -> Decision:
        return self.inner.initial_request(s, pool)  # type: ignore[arg-type]

    def decide(self, active: list[QuestionState], pool: object) -> dict[TraceKey, Decision]:
        out = self.inner.decide(active, pool)  # type: ignore[arg-type]
        for s in active:
            d = out[s.key]
            self.log.append((s.visited[-1] if s.visited else 0, s.gen_end, d.action, d.min_tokens, d.max_tokens))
        return out


@pytest.mark.parametrize("which", ["deer", "price"])
def test_non_anticipation(which: str) -> None:
    """I4: cắt chuỗi đang mô phỏng sau điểm j không đổi các quyết định tại các điểm <= j."""
    b_max = ENV.l_log - C_ANS_MAX
    checked = 0
    for key, chain in list(ENV.chains.items())[:12]:
        if chain.n_points < 6:
            continue
        j = chain.n_points * 2 // 3
        logs = []
        for c in (chain, chain.truncate_after(j)):
            inner: DeerPolicy | PricePolicy = (
                DeerPolicy(0.999, "geo", False, DP, C_ANS_MAX, C_PROBE_MAX, 256)
                if which == "deer"
                else PricePolicy(DP, mu_min=0.0, b_max=b_max, run_seed=0)
            )
            pol = Recording(inner)
            prov = CurveRhoProvider(ENV.curves, ENV.cfg.policy.horizons, b_max, 0.0) if which == "price" else None
            rc = RunnerCfg(b_max, 256, C_ANS_MAX, C_PROBE_MAX, DP.b_min, 0)
            run_batch([key], pol, LoggedSource({key: c}), prov, rc, 10**7)  # type: ignore[arg-type]
            logs.append([r for r in pol.log if r[0] <= j])
        # Cùng trạng thái thì cùng quyết định; quỹ đạo chỉ được rẽ khi nguồn trả về điểm khác (sự kiện sau j).
        n_same = 0
        for a, b in zip(logs[0], logs[1], strict=False):
            if a[:2] != b[:2]:
                break
            assert a == b
            n_same += 1
        checked += n_same
    assert checked >= 3


def test_m3_sim_run_smoke() -> None:
    keys = _keys()
    res = ENV.run(M3, keys, _min_budget(len(keys)) + 20000)
    assert res.spent.total <= res.budget and 0.0 <= res.accuracy <= 1.0
    res2 = ENV.run(B2, keys, _min_budget(len(keys)) + 20000, allowance=3000)
    assert all(o.costs.total <= 3000 for o in res2.outcomes)  # hạn mức theo câu tính trên T_i


def test_cfg_helpers_consistent() -> None:
    cfg = synth_cfg()
    assert ENV.c_hi() == cfg.log.l_log + C_ANS_MAX + -(-cfg.log.l_log // cfg.dp.m) * C_PROBE_MAX
    assert np.isfinite(ENV.ref_median)
