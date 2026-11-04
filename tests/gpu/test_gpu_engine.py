"""Test GPU (P1.T4, P4.T3): VllmGenerator, LiveSource và Runner trên Qwen3-1.7B thật.

Chạy trên máy có GPU và vLLM: `pytest -m gpu tests/gpu` (Kaggle: ops/kaggle/p1_gpu_tests.py; Vast: công việc gpu_check).
Biến TTALLOC_DTYPE chọn kiểu số (mặc định bfloat16 nếu GPU hỗ trợ, ngược lại float16, ví dụ T4).
Không dùng tập dữ liệu nào: câu hỏi viết sẵn trong tệp này.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = pytest.mark.gpu

QUESTIONS = [
    ("g/0", "What is 7 * 8 + 3?", "59"),
    ("g/1", "Compute 2^10 - 24.", "1000"),
    ("g/2", "What is the sum of the first 10 positive integers?", "55"),
    ("g/3", "Solve for x: 3x + 5 = 20.", "5"),
    ("g/4", "What is the remainder when 100 is divided by 7?", "2"),
    ("g/5", "How many positive divisors does 36 have?", "9"),
    ("g/6", "What is 15% of 240?", "36"),
    ("g/7", "What is the least common multiple of 12 and 18?", "36"),
]


def _dtype() -> str:
    import torch

    # T4 (7.5) báo is_bf16_supported() = True vì torch giả lập, nhưng vLLM chỉ nhận bf16 từ compute capability 8.0.
    native_bf16 = torch.cuda.get_device_capability(0) >= (8, 0)
    return os.environ.get("TTALLOC_DTYPE") or ("bfloat16" if native_bf16 else "float16")


@pytest.fixture(scope="module")
def world() -> dict[str, Any]:
    pytest.importorskip("vllm")
    from ttalloc.config import ExperimentCfg
    from ttalloc.context import model_context, prompt_ids_for
    from ttalloc.engine.vllm_generator import VllmGenerator
    from ttalloc.types import Question

    cfg = ExperimentCfg.model_validate(
        {
            "model": "qwen3_1_7b",
            "engine": {"dtype": _dtype(), "gpu_memory_utilization": 0.85},
            "dp": {"m": 128, "l_max": 256, "b_min": 128},
            "log": {"m": 128, "l_max": 256, "l_log": 2048},
            "guard": {"min_segment": 256},
            "answer": {"mode": "full", "r_ans": 256},
            "policy": {"horizons": [256, 512], "mu_min": 0.0, "b_max_mult": None},
        }
    )
    ctx = model_context(cfg)
    qs = [Question(q, "g", p, None, g, "math", None) for q, p, g in QUESTIONS]
    pids = prompt_ids_for(qs, ctx)
    gen = VllmGenerator(ctx.model_cfg, cfg.engine, 4096)
    return {"cfg": cfg, "ctx": ctx, "qs": qs, "pids": pids, "gen": gen}


def _source(w: dict[str, Any]) -> Any:
    from ttalloc.engine.live_source import LiveSource

    ctx = w["ctx"]
    return LiveSource(
        w["gen"],
        ctx.codec,
        {q.qid: q for q in w["qs"]},
        w["pids"],
        w["cfg"],
        ctx.model_cfg,
        ctx.prompts,
        l_log=w["cfg"].log.l_log,
    )


def test_segment_stop_strings_and_logprobs(world: dict[str, Any]) -> None:
    from ttalloc.engine.cut import stop_strings
    from ttalloc.types import GenRequest

    stops = tuple(stop_strings(world["cfg"].dp.markers))
    codec = world["ctx"].codec
    (out,) = world["gen"].generate(
        [GenRequest(tuple(world["pids"]["g/0"]), 512, stops, (codec.think_end_id,), 0.6, 0.95, 20, 0, seed=1)]
    )
    assert out.logprobs is not None and len(out.logprobs) == len(out.token_ids)
    assert all(lp <= 0.0 for lp in out.logprobs)
    if out.finish_reason == "stop" and isinstance(out.stop_reason, str):
        assert out.stop_reason in stops and codec.decode(list(out.token_ids)).endswith(out.stop_reason)


def test_greedy_is_deterministic(world: dict[str, Any]) -> None:
    from ttalloc.types import GenRequest

    req = GenRequest(tuple(world["pids"]["g/1"]), 64, (), (), 0.0, logprobs=1)
    a, b = world["gen"].generate([req, req])
    assert a.token_ids == b.token_ids


def test_live_source_walk(world: dict[str, Any]) -> None:
    from ttalloc.types import AdvanceRequest

    src = _source(world)
    keys = _keys(world, 0)[:4]
    src.start(keys)
    cfg = world["cfg"]
    seen = {k: 0 for k in keys}
    live = list(keys)
    reqs = [AdvanceRequest(k, cfg.dp.b_min, cfg.dp.b_min + cfg.dp.l_max, True) for k in live]
    for _ in range(12):
        if not live:
            break
        segs = src.advance(reqs)
        at_point = [r for r in segs if r.kind in ("marker", "length")]
        for r in segs:
            assert r.n_new_tokens >= 0 and r.gen_end == seen[r.key] + r.n_new_tokens and r.pos <= r.gen_end
            seen[r.key] = r.gen_end
        for p in src.probe([r.key for r in at_point]):
            assert p.output_tokens <= cfg.probe.r_probe + world["ctx"].model_cfg.probe_think_end_lookahead
            assert 0.0 <= p.conf_geo <= p.conf_arith <= 1.0 + 1e-9
        live = [r.key for r in at_point]
        reqs = [AdvanceRequest(k, cfg.dp.m, cfg.dp.l_max, True) for k in live]
    for a in src.answer(keys):
        assert a.output_tokens <= 256 and isinstance(a.correct, bool)
    # Đoạn sau, gợi và trả lời đều nối tiếp tiền tố đã sinh, nên prefix cache phải trúng; số này vào bảng thời gian E5.
    rate = world["gen"].prefix_cache_hit_rate()
    names = sorted({getattr(m, "name", "") for m in world["gen"].llm.get_metrics()} - {""})
    assert rate is not None and rate > 0.0, f"hit rate {rate}; số đo có: {[n for n in names if 'prefix' in n]}"
    kv = world["gen"].kv_cache_tokens()  # logger dùng số này để không chạy quá sức chứa KV
    assert kv is not None and kv > 10_000, kv


def _keys(world: dict[str, Any], seed: int) -> list[Any]:
    from ttalloc.types import TraceKey

    return [TraceKey(q.qid, seed) for q in world["qs"]]


class _FeatureRho:
    def fill(self, states: list[Any]) -> None:
        for s in states:
            conf = s.features.vector()["conf_cur"]
            s.rho = 1e-3 if conf != conf else (1.0 - conf) * 1e-3
            s.hstar = 256


@pytest.mark.parametrize("name", ["B2", "M2"])
@pytest.mark.parametrize("size", ["tiny", "medium"])
def test_runner_smoke(world: dict[str, Any], name: str, size: str) -> None:
    """P4.T3: N = 8; không vượt B; mọi câu có câu trả lời; số token sinh theo loại khớp với engine."""
    from ttalloc.config import load_model_cfg
    from ttalloc.controller.runner import RunnerCfg, run_batch
    from ttalloc.policies.deer import DeerPolicy
    from ttalloc.policies.price import PricePolicy

    cfg = world["cfg"]
    src = _source(world)
    c_ans, c_probe = src.c_ans_max(), src.c_probe_max()
    keys = _keys(world, 3 if name == "B2" else 4)
    n = len(keys)
    floor = n * (c_ans + cfg.dp.b_min + c_probe)
    budget = floor + 300 if size == "tiny" else floor + n * 1200
    b_max = cfg.log.l_log - c_ans
    mc = load_model_cfg("qwen3_1_7b")
    if name == "B2":
        pol: Any = DeerPolicy(
            0.95,
            mc.deer_conf_mode,
            mc.deer_require_think_end_after_probe,
            cfg.dp,
            c_ans,
            c_probe,
            cfg.guard.min_segment,
        )
        prov = None
    else:
        pol, prov = PricePolicy(cfg.dp, 0.0, b_max, 0), _FeatureRho()
    rc = RunnerCfg(b_max, cfg.guard.min_segment, c_ans, c_probe, cfg.dp.b_min, 0, cfg.features.k, max_active=4)
    before = world["gen"].n_generated
    res = run_batch(keys, pol, src, prov, rc, budget)
    assert res.spent.total <= budget  # I1
    assert sorted(o.key.qid for o in res.outcomes) == sorted(k.qid for k in keys)  # I3
    produced = res.spent.think + res.spent.probe_output + res.spent.answer_output
    assert produced == world["gen"].n_generated - before  # mỗi token sinh ra được tính đúng một lần
