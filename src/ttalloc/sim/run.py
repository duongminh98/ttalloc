"""Chạy mô phỏng theo lưới tập × lô × alpha × phương pháp × lần rút (PLAN P2.T15)."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ttalloc.config import ExperimentCfg, MethodCfg
from ttalloc.controller.runner import BatchResult
from ttalloc.logging.schema import PER_QUESTION_SCHEMA, to_table
from ttalloc.logging.store import atomic_write_table
from ttalloc.sim.env import Draw, SimEnv, batch_name
from ttalloc.sim.tune import calibrate_b2

_ENV: SimEnv | None = None


def _worker(m: MethodCfg, d: Draw, allowance: int | None) -> BatchResult:
    assert _ENV is not None
    return _ENV.run(m, d.keys, d.budget, allowance)


def run_draws(env: SimEnv, m: MethodCfg, draws: list[Draw], allowance: int | None, workers: int) -> list[BatchResult]:
    if workers <= 1 or "fork" not in mp.get_all_start_methods():
        return [env.run(m, d.keys, d.budget, allowance) for d in draws]
    global _ENV
    _ENV = env
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.starmap(_worker, [(m, d, allowance) for d in draws])
    finally:
        _ENV = None


def needs_calibration(m: MethodCfg) -> bool:
    return m.policy == "deer" and m.params.get("cap", "calibrated") == "calibrated"


def outcome_rows(
    env: SimEnv, m: MethodCfg, alpha: float, d: Draw, res: BatchResult, batch: str
) -> list[dict[str, Any]]:
    rows = []
    for o in res.outcomes:
        q = env.questions[o.key.qid]
        rows.append(
            {
                "method": m.id,
                "alpha": float(alpha),
                "draw": d.draw,
                "qid": o.key.qid,
                "seed": o.key.seed,
                "correct": o.correct,
                **o.costs.as_dict(),
                "stop_point_idx": o.stop_point_idx,
                "stop_kind": o.stop_kind,
                "n_probes": o.n_probes,
                "n_rounds": o.n_rounds,
                "split": env.split,
                "batch": batch,
                "budget": d.budget,
                "dataset": q.dataset,
                "difficulty": q.difficulty,
            }
        )
    return rows


def summarize_cell(results: list[BatchResult], draws: list[Draw]) -> dict[str, Any]:
    accs = np.array([r.accuracy for r in results])
    ratio = np.array([r.spent.total / d.budget for r, d in zip(results, draws, strict=True)])
    cats = {
        k: float(np.mean([getattr(r.spent, k) for r in results]))
        for k in ("think", "probe_prompt", "probe_output", "answer_prompt", "answer_output")
    }
    n = np.array([len(d.keys) for d in draws])
    return {
        "accuracy": float(accs.mean()),
        "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "spend_ratio": float(ratio.mean()),
        "tokens": cats,
        "tokens_total": float(sum(cats.values())),
        "budget_mean": float(np.mean([d.budget for d in draws])),
        "budget_per_q": float(np.mean([d.budget for d in draws] / n)),
        "c_full_mean": float(np.mean([d.c_full for d in draws])),
        "n_draws": len(draws),
        "n_questions": int(n.mean()),
        "probe_share": (cats["probe_prompt"] + cats["probe_output"]) / max(sum(cats.values()), 1.0),
    }


def run_experiment(
    envs: dict[str, SimEnv],
    cfg: ExperimentCfg,
    run_dir: Path,
    hashes: dict[str, Any],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    t_start = time.time()
    for split in cfg.splits:
        env = envs[split]
        for batch in cfg.batches or [cfg.batch]:
            bname = batch_name(batch)
            for alpha in cfg.alphas:
                draws = env.draws(batch, alpha, cfg.sim.draws)
                for m in cfg.methods:
                    t0 = time.time()
                    cell: dict[str, Any] = {"split": split, "batch": bname, "alpha": alpha, "method": m.id}
                    allowance = None
                    if needs_calibration(m):
                        cal = calibrate_b2(env, m, draws[: cfg.sim.calib_draws or len(draws)])
                        allowance = cal.allowance
                        cell |= {
                            "allowance": cal.allowance,
                            "calib_spend_ratio": cal.spend_ratio,
                            "budget_underused": cal.budget_underused,
                            "calib_iters": cal.iters,
                        }
                    results = run_draws(env, m, draws, allowance, cfg.sim.workers)
                    for d, r in zip(draws, results, strict=True):
                        rows += outcome_rows(env, m, alpha, d, r, bname)
                    cell |= summarize_cell(results, draws)
                    cell["seconds"] = round(time.time() - t0, 2)
                    cells.append(cell)
                    log(
                        f"{split}/{bname} alpha={alpha} {m.id}: acc={cell['accuracy']:.4f} "
                        f"spend={cell['spend_ratio']:.3f} ({cell['seconds']}s)"
                    )
    atomic_write_table(to_table(rows, PER_QUESTION_SCHEMA), run_dir / "per_question.parquet")
    summary = {
        "name": cfg.name,
        "cells": cells,
        "hashes": hashes,
        "seconds": round(time.time() - t_start, 1),
        "known_sim_live_gaps": "PLAN mục 7.6: length làm tròn xuống điểm đã ghi; log.m khác dp.m; "
        "chạy thật sinh chuỗi mới; thời gian và KV cache không mô phỏng được",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
