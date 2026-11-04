"""Vòng lặp Runner dùng chung cho mô phỏng và chạy thật (PLAN A9, P2.T8)."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ttalloc.controller.budget import Pool, guard
from ttalloc.features.extract import FeatureState
from ttalloc.policies.base import tie_key
from ttalloc.types import (
    AdvanceRequest,
    AnswerResult,
    Costs,
    Decision,
    Policy,
    QuestionState,
    RhoProvider,
    SegmentSource,
    TraceKey,
)


class GradeFn(Protocol):
    def __call__(self, key: TraceKey, extracted: str | None) -> bool: ...


@dataclass(frozen=True)
class RunnerCfg:
    b_max: int
    min_segment: int
    c_ans_max: int
    c_probe_max: int
    b_min: int
    run_seed: int
    k: int = 4
    max_active: int | None = None


@dataclass
class Outcome:
    key: TraceKey
    correct: bool
    costs: Costs
    stop_point_idx: int
    stop_kind: str
    n_probes: int
    n_rounds: int
    extracted: str | None


@dataclass
class BatchResult:
    outcomes: list[Outcome]
    spent: Costs
    budget: int
    rounds: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(o.correct for o in self.outcomes) / len(self.outcomes) if self.outcomes else math.nan


def run_batch(
    keys: list[TraceKey],
    policy: Policy,
    source: SegmentSource,
    rho_provider: RhoProvider | None,
    rc: RunnerCfg,
    budget: int,
    grade_fn: GradeFn | None = None,
    on_round: Callable[[dict[str, Any]], None] | None = None,
) -> BatchResult:
    if len(set(keys)) != len(keys):
        raise ValueError("khóa trùng trong lô")
    pool = Pool(budget, rc.c_ans_max, rc.c_probe_max, rc.b_min, n_waiting=len(keys))
    pool.check_feasible(n_total=len(keys))  # InfeasibleBudget
    waiting = deque(sorted(keys, key=lambda k: tie_key(k, rc.run_seed)))  # cùng thứ tự nạp ở hai môi trường
    active: dict[TraceKey, QuestionState] = {}
    fresh: set[TraceKey] = set()
    ended_kind: dict[TraceKey, str] = {}
    outcomes: list[Outcome] = []
    rounds: list[dict[str, Any]] = []
    max_active = rc.max_active or len(keys)

    def admit() -> None:
        new: list[TraceKey] = []
        while waiting and len(active) + len(new) < max_active:
            new.append(waiting.popleft())
        if not new:
            return
        source.start(new)
        for k in new:
            active[k] = QuestionState(key=k, features=FeatureState(k=rc.k))
            fresh.add(k)
        pool.n_active += len(new)
        pool.n_waiting -= len(new)

    def finalize(answers: list[AnswerResult]) -> None:
        for a in answers:
            s = active.pop(a.key)
            pool.charge("answer_prompt", a.prompt_tokens)
            pool.charge("answer_output", a.output_tokens)
            s.costs.answer_prompt += a.prompt_tokens
            s.costs.answer_output += a.output_tokens
            pool.n_active -= 1
            correct = a.correct
            if correct is None:
                if grade_fn is None:
                    raise ValueError(f"{a.key}: nguồn không chấm điểm và không có grade_fn")
                correct = grade_fn(a.key, a.extracted)
            outcomes.append(
                Outcome(
                    key=a.key,
                    correct=bool(correct),
                    costs=s.costs,
                    stop_point_idx=s.visited[-1] if s.visited else 0,
                    stop_kind=ended_kind.get(a.key, s.point_kind),
                    n_probes=len(s.features.probes),
                    n_rounds=s.n_rounds,
                    extracted=a.extracted,
                )
            )

    admit()
    while active:
        t0 = time.perf_counter()
        states = list(active.values())
        dec: dict[TraceKey, Decision] = {k: policy.initial_request(active[k], pool) for k in fresh}
        rest = [s for s in states if s.key not in fresh]
        if rest:
            if policy.uses_rho:
                if rho_provider is None:
                    raise ValueError("chính sách dùng rho nhưng không có RhoProvider")
                rho_provider.fill(rest)
            dec |= policy.decide(rest, pool)
        fresh.clear()
        dec, allot = guard(dec, states, pool, policy.uses_probes, rc.b_max, rc.min_segment, rc.c_probe_max, rc.run_seed)
        t1 = time.perf_counter()
        stops = [s.key for s in states if dec[s.key].action == "stop"]
        if stops:
            finalize(source.answer(stops))
        t2 = time.perf_counter()
        segs = source.advance([AdvanceRequest(k, mn, mx, policy.stop_at_markers) for k, (mn, mx) in allot.items()])
        for r in segs:
            if r.n_new_tokens > allot[r.key][1]:
                raise AssertionError(f"{r.key}: đoạn sinh {r.n_new_tokens} token > hạn mức {allot[r.key][1]}")
            pool.charge("think", r.n_new_tokens)
            active[r.key].apply(r)
        t3 = time.perf_counter()
        ended = [r.key for r in segs if r.kind in ("natural_end", "eos", "truncated")]
        for r in segs:
            if r.kind in ("natural_end", "eos", "truncated"):
                ended_kind[r.key] = r.kind
        if ended:
            finalize(source.answer(ended))
        t4 = time.perf_counter()
        at_point = [r for r in segs if r.kind in ("marker", "length")]
        if policy.uses_probes and at_point:
            for r, p in zip(at_point, source.probe([r.key for r in at_point]), strict=True):
                pool.charge("probe_prompt", p.prompt_tokens)
                pool.charge("probe_output", p.output_tokens)
                s = active[p.key]
                s.costs.probe_prompt += p.prompt_tokens
                s.costs.probe_output += p.output_tokens
                s.features.update(r.gen_end, p, r.n_markers, r.local_conf_min)
        t5 = time.perf_counter()
        n_active = len(active) + len(stops) + len(ended)
        admit()
        pool.check_round_invariants()
        info = {
            "round": len(rounds),
            "t_controller": t1 - t0,
            "t_answer": (t2 - t1) + (t4 - t3),
            "t_advance": t3 - t2,
            "t_probe": t5 - t4,
            "n_active": n_active,
            "spent": pool.spent.total,
        }
        rounds.append(info)
        if on_round:
            on_round(info)
    if len(outcomes) != len(keys):
        raise AssertionError(f"I3: {len(outcomes)} câu trả lời cho {len(keys)} câu")
    return BatchResult(outcomes=outcomes, spent=pool.spent, budget=budget, rounds=rounds)
