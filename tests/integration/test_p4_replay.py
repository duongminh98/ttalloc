"""P4.T2: kiểm thử phát lại. Cùng một thế giới theo kịch bản được ghi bằng logger rồi chạy bằng LoggedSource, và chạy
trực tiếp bằng LiveSource; với log.m = dp.m, log.l_max = dp.l_max và quỹ rộng, hai môi trường phải cho cùng kết quả.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripted_world import make_world
from ttalloc.config import load_model_cfg
from ttalloc.controller.runner import BatchResult, RunnerCfg, run_batch
from ttalloc.features.events import chains_from_tables
from ttalloc.logging.logger import run_logger
from ttalloc.logging.store import LogStore
from ttalloc.policies.deer import DeerPolicy
from ttalloc.policies.price import PricePolicy
from ttalloc.sim.logged_source import LoggedSource
from ttalloc.types import QuestionState, TraceKey


class FeatureRho:
    """RhoProvider tất định chỉ dựa vào đặc trưng: lệch đặc trưng giữa hai môi trường sẽ làm lệch quyết định."""

    def fill(self, states: list[QuestionState]) -> None:
        for s in states:
            v = s.features.vector()
            conf = v["conf_cur"]
            s.rho = 1e-3 if conf != conf else (1.0 - conf) * 1e-3 - 2e-4 * v["consistency"]
            s.hstar = 64


@pytest.fixture(scope="module")
def world(qwen3_tokenizer: Any) -> Any:
    return make_world(qwen3_tokenizer, n_questions=6, seed=4, max_len_q=None)


@pytest.fixture(scope="module")
def logged(world: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[TraceKey, Any]:
    store = LogStore(tmp_path_factory.mktemp("logs") / "qwen3_1_7b" / "val")
    store.init("LH", {})
    keys = [TraceKey(q, 0) for q in world.questions]
    gold = {q: v.gold for q, v in world.questions.items()}
    run_logger(world.source(), keys, store, world.cfg, "qwen3_1_7b", "val", "LH", gold)
    return chains_from_tables(store.read_traces(), store.read_points())


def _policies(world: Any, c_ans: int, c_probe: int, b_max: int) -> dict[str, Any]:
    mc = load_model_cfg("qwen3_1_7b")
    cfg = world.cfg
    return {
        "B2": (
            DeerPolicy(
                0.9,
                mc.deer_conf_mode,
                mc.deer_require_think_end_after_probe,
                cfg.dp,
                c_ans,
                c_probe,
                cfg.guard.min_segment,
            ),
            None,
        ),
        "M2": (PricePolicy(cfg.dp, mu_min=0.0, b_max=b_max, run_seed=0), FeatureRho()),
    }


def _run(world: Any, src: Any, name: str, budget: int, c_ans: int, c_probe: int) -> BatchResult:
    cfg = world.cfg
    b_max = cfg.log.l_log - c_ans
    pol, prov = _policies(world, c_ans, c_probe, b_max)[name]
    rc = RunnerCfg(b_max, cfg.guard.min_segment, c_ans, c_probe, cfg.dp.b_min, cfg.run_seed, cfg.features.k)
    keys = [TraceKey(q, 0) for q in world.questions]
    return run_batch(keys, pol, src, prov, rc, budget)


def _sig(r: BatchResult) -> list[tuple[Any, ...]]:
    return sorted(
        (o.key.qid, o.correct, tuple(o.costs.as_dict().values()), o.stop_kind, o.n_probes, o.n_rounds)
        for o in r.outcomes
    )


@pytest.mark.parametrize("name", ["B2", "M2"])
def test_replay_sim_equals_live(world: Any, logged: dict[TraceKey, Any], name: str) -> None:
    live_src = world.source()
    c_ans, c_probe = live_src.c_ans_max(), live_src.c_probe_max()
    live = _run(world, live_src, name, 10**6, c_ans, c_probe)
    sim = _run(world, LoggedSource(logged), name, 10**6, c_ans, c_probe)
    assert _sig(live) == _sig(sim)
    assert live.spent.as_dict() == sim.spent.as_dict()


@pytest.mark.parametrize("name", ["B2", "M2"])
def test_replay_small_budget_never_exceeds(world: Any, logged: dict[TraceKey, Any], name: str) -> None:
    live_src = world.source()
    c_ans, c_probe = live_src.c_ans_max(), live_src.c_probe_max()
    n = len(world.questions)
    budget = n * (c_ans + world.cfg.dp.b_min + c_probe) + 300
    for src in (live_src, LoggedSource(logged)):
        r = _run(world, src, name, budget, c_ans, c_probe)
        assert r.spent.total <= budget and len(r.outcomes) == n  # I1, I3 ở cả hai môi trường


def test_logged_world_shape(logged: dict[TraceKey, Any], world: Any, tmp_path: Path) -> None:
    assert len(logged) == len(world.questions)
    assert all(c.natural_end for c in logged.values())
