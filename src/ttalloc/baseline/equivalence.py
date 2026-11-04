"""Kiểm định tương đương giữa sinh liền mạch và sinh theo đoạn có nhánh phụ (PLAN P1.T6).

Đạt khi khoảng tin cậy bootstrap 90% (theo câu hỏi) của chênh lệch nằm trọn trong biên: độ chính xác ±3 điểm phần trăm,
độ dài suy luận trung bình ±5%.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ttalloc.baseline.full import Sampling, generate_full
from ttalloc.config import ExperimentCfg
from ttalloc.engine.live_source import LiveSource
from ttalloc.logging.studies import walk_chains
from ttalloc.types import Question, SegmentResult, TraceKey


def segmented_rows(
    source: LiveSource, questions: list[Question], seeds: list[int], cfg: ExperimentCfg
) -> list[dict[str, Any]]:
    keys = [TraceKey(q.qid, s) for s in seeds for q in questions]
    rows: list[dict[str, Any]] = []
    n_probes: dict[TraceKey, int] = {}

    def at_point(segs: list[SegmentResult]) -> None:
        source.probe([r.key for r in segs])
        for r in segs:
            n_probes[r.key] = n_probes.get(r.key, 0) + 1

    def at_end(segs: list[SegmentResult]) -> None:
        for r, a in zip(segs, source.answer([r.key for r in segs], finalize=True), strict=True):
            rows.append(
                {
                    "qid": r.key.qid,
                    "seed": r.key.seed,
                    "think_tokens": r.gen_end,
                    "correct": bool(a.correct),
                    "natural_end": True,
                    "n_probes": n_probes.get(r.key, 0),
                }
            )

    walk_chains(source, keys, cfg, at_point, at_end)
    done = {(r["qid"], r["seed"]) for r in rows}
    for k in keys:  # chuỗi chạm L_log: không có câu trả lời tự nhiên, tính là sai và dài L_log
        if (k.qid, k.seed) not in done:
            rows.append(
                {
                    "qid": k.qid,
                    "seed": k.seed,
                    "think_tokens": cfg.log.l_log,
                    "correct": False,
                    "natural_end": False,
                    "n_probes": n_probes.get(k, 0),
                }
            )
    return rows


def paired_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int, level: float
) -> tuple[float, float, float]:
    d = a - b
    q = len(d)
    boot = np.array([d[rng.integers(0, q, q)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(d.mean()), float(lo), float(hi)


def equivalence_report(
    full: pd.DataFrame,
    seg: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int = 2000,
    acc_margin: float = 0.03,
    len_margin: float = 0.05,
    level: float = 0.90,
) -> dict[str, Any]:
    """`full` và `seg` có cột qid, seed, correct, think_tokens. Ghép theo câu hỏi (trung bình qua seed)."""
    f = full.groupby("qid").agg(acc=("correct", "mean"), length=("think_tokens", "mean"))
    s = seg.groupby("qid").agg(acc=("correct", "mean"), length=("think_tokens", "mean"))
    j = f.add_suffix("_full").join(s.add_suffix("_seg"), how="inner")
    acc = paired_ci(j.acc_seg.to_numpy(float), j.acc_full.to_numpy(float), rng, n_boot, level)
    base_len = float(j.length_full.mean())
    rel = paired_ci(
        j.length_seg.to_numpy(float) / base_len, j.length_full.to_numpy(float) / base_len, rng, n_boot, level
    )
    acc_ok = -acc_margin <= acc[1] and acc[2] <= acc_margin
    len_ok = -len_margin <= rel[1] and rel[2] <= len_margin
    return {
        "n_questions": len(j),
        "acc_full": float(j.acc_full.mean()),
        "acc_seg": float(j.acc_seg.mean()),
        "acc_diff": acc[0],
        "acc_ci90": [acc[1], acc[2]],
        "acc_ok": acc_ok,
        "len_full": base_len,
        "len_seg": float(j.length_seg.mean()),
        "len_rel_diff": rel[0],
        "len_rel_ci90": [rel[1], rel[2]],
        "len_ok": len_ok,
        "pass": bool(acc_ok and len_ok),
        "inconclusive_hint": "nếu khoảng quá rộng để kết luận, tăng lên 4 seed (PLAN P1.T6)",
    }


def run_equivalence(
    source: LiveSource,
    questions: list[Question],
    prompt_ids: dict[str, list[int]],
    seeds: list[int],
    sampling: Sampling,
    cfg: ExperimentCfg,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    full = pd.DataFrame(
        generate_full(source.gen, source.codec, questions, prompt_ids, seeds, sampling, chunk=cfg.baseline.chunk)
    )
    seg = pd.DataFrame(segmented_rows(source, questions, seeds, cfg))
    return full, seg, equivalence_report(full, seg, rng, cfg.sim.n_boot)
