"""Chạy thật: Runner + LiveSource trên vLLM + EstimatorRhoProvider (PLAN P4.T1).

Với mỗi cấu hình (alpha, phương pháp, seed), lệnh cũng chạy mô phỏng tương ứng với cùng lô, cùng B, cùng `max_active`
và cùng thứ tự nạp, để cổng G3 dựng dải chấp nhận (A12).
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttalloc.config import ExperimentCfg, MethodCfg
from ttalloc.context import ModelContext
from ttalloc.controller.runner import run_batch
from ttalloc.data.batches import build_batch
from ttalloc.engine.live_source import LiveSource
from ttalloc.logging.schema import PER_QUESTION_SCHEMA, TIMING_SCHEMA, to_table
from ttalloc.logging.store import atomic_write_table
from ttalloc.online.timing import RoundTimer
from ttalloc.sim.env import Draw, SimEnv, batch_name
from ttalloc.sim.run import needs_calibration, outcome_rows, run_draws, summarize_cell
from ttalloc.sim.tune import calibrate_b2
from ttalloc.types import Generator, TraceKey


def sim_draws_for_batch(env: SimEnv, qids: list[str], budget: int, n: int) -> list[Draw]:
    """Lần rút mô phỏng trên đúng lô của chạy thật: chỉ seed của mỗi câu thay đổi; B giữ nguyên."""
    cf = env.c_full(qids)
    return [Draw(d, env.draw_keys(qids, env.draw_rng(10_000 + d)), budget, cf) for d in range(n)]


def run_online(
    cfg: ExperimentCfg,
    ctx: ModelContext,
    generator: Generator,
    env: SimEnv,
    prompt_ids: dict[str, list[int]],
    run_dir: Path,
    max_active: int,
    hit_rate: Callable[[], float | None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    qs = list(env.questions.values())
    qids = build_batch(qs, cfg.batch, np.random.default_rng(cfg.batch.seed))
    bname = batch_name(cfg.batch)
    c_full = env.c_full(qids)
    rows: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    sim_ref: list[dict[str, Any]] = []
    for alpha in cfg.alphas:
        budget = int(alpha * c_full)
        ref_draws = sim_draws_for_batch(env, qids, budget, cfg.sim.draws)
        for m in cfg.methods:
            m_live: MethodCfg = m.model_copy(
                update={"rho_provider": "estimator" if m.rho_provider is None else m.rho_provider}
            )
            allowance = None
            cell_base: dict[str, Any] = {
                "split": env.split,
                "batch": bname,
                "alpha": alpha,
                "method": m.id,
                "budget": budget,
                "max_active": max_active,
            }
            if needs_calibration(m):
                cal = calibrate_b2(env, m, ref_draws[: cfg.online.calib_draws])
                allowance = cal.allowance
                cell_base |= {
                    "allowance": allowance,
                    "calib_spend_ratio": cal.spend_ratio,
                    "budget_underused": cal.budget_underused,
                }
            # Một tiến trình: run_draws với workers > 1 fork, mà tiến trình này đang giữ engine vLLM (luồng, CUDA);
            # fork lúc đó dễ treo. Kết quả mỗi lần rút không phụ thuộc số tiến trình.
            sim_res = run_draws(env, m, ref_draws, allowance, 1)
            sim_ref += [
                {
                    "alpha": alpha,
                    "method": m.id,
                    "draw": d.draw,
                    "accuracy": r.accuracy,
                    "spent": r.spent.total,
                    "budget": budget,
                }
                for d, r in zip(ref_draws, sim_res, strict=True)
            ]
            for seed in cfg.online.seeds:
                keys = [TraceKey(q, seed) for q in qids]
                src = LiveSource(
                    generator,
                    ctx.codec,
                    env.questions,
                    prompt_ids,
                    cfg,
                    ctx.model_cfg,
                    ctx.prompts,
                    l_log=cfg.log.l_log,
                )
                if (src.c_ans_max(), src.c_probe_max()) != (env.c_ans_max, env.c_probe_max):
                    raise ValueError(
                        f"c_ans_max/c_probe_max của chạy thật {(src.c_ans_max(), src.c_probe_max())} "
                        f"khác dữ liệu ghi {(env.c_ans_max, env.c_probe_max)} (I9)"
                    )
                pol, prov, b_max = env.make(m_live, allowance)
                rc = dataclasses.replace(env.runner_cfg(b_max), max_active=max_active)
                timer = RoundTimer(hit_rate=hit_rate)
                t0 = time.time()
                res = run_batch(keys, pol, src, prov, rc, budget, on_round=timer)
                wall = time.time() - t0
                d = Draw(seed, keys, budget, c_full)
                rows += outcome_rows(env, m, alpha, d, res, bname)
                timing += [{**r, "alpha": alpha, "method": m.id, "seed": seed} for r in timer.rows]
                cell = (
                    cell_base
                    | summarize_cell([res], [d])
                    | {
                        "seed": seed,
                        "wall_seconds": round(wall, 1),
                        "timing": timer.totals(),
                        "exceeded": res.spent.total > budget,
                    }
                )
                cell["sim_mean_accuracy"] = float(np.mean([r.accuracy for r in sim_res]))
                cells.append(cell)
                log(
                    f"online alpha={alpha} {m.id} seed={seed}: acc={cell['accuracy']:.4f} "
                    f"spend={cell['spend_ratio']:.3f} wall={wall:.0f}s; sim={cell['sim_mean_accuracy']:.4f}"
                )
    atomic_write_table(to_table(rows, PER_QUESTION_SCHEMA), run_dir / "per_question.parquet")
    tt = to_table([{k: r[k] for k in TIMING_SCHEMA.names} for r in timing], TIMING_SCHEMA)
    atomic_write_table(tt, run_dir / "timing.parquet")
    pd.DataFrame(timing).to_parquet(run_dir / "timing_full.parquet", index=False)
    pd.DataFrame(sim_ref).to_parquet(run_dir / "sim_reference.parquet", index=False)
    summary = {
        "name": cfg.name,
        "kind": "online",
        "cells": cells,
        "batch_qids": qids,
        "gpu": None,
        "known_sim_live_gaps": "PLAN mục 7.6",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
