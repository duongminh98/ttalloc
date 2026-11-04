"""Hai nghiên cứu nhỏ trước khi ghi dữ liệu chính: dạng câu trả lời (P1.T9) và nhiễu do lời nhắc gợi (P1.T10)."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ttalloc.config import ExperimentCfg
from ttalloc.data.prompts import PromptSet
from ttalloc.engine.live_source import LiveSource
from ttalloc.types import AdvanceRequest, SegmentResult, TraceKey

PointFn = Callable[[list[SegmentResult]], None]
EndFn = Callable[[list[SegmentResult]], None]


def walk_chains(source: LiveSource, keys: list[TraceKey], cfg: ExperimentCfg, at_point: PointFn, at_end: EndFn) -> None:
    """Sinh các chuỗi theo đoạn như logger; gọi `at_point` ở mọi điểm mới và `at_end` khi chuỗi tự kết thúc.

    `at_end` phải tự kết thúc chuỗi (answer với finalize hoặc drop). Chuỗi chạm L_log bị bỏ.
    """
    queue = deque(keys)
    active: set[TraceKey] = set()
    while queue or active:
        new: list[TraceKey] = []
        while queue and len(active) + len(new) < cfg.log.window:
            new.append(queue.popleft())
        source.start(new)
        active |= set(new)
        segs = source.advance([AdvanceRequest(k, cfg.log.m, cfg.log.l_max, True) for k in sorted(active, key=str)])
        pts = [r for r in segs if r.kind in ("marker", "length")]
        if pts:
            at_point(pts)
        ended = [r for r in segs if r.kind in ("natural_end", "eos")]
        if ended:
            at_end(ended)
        full = [r.key for r in segs if r.kind == "truncated" or (r.kind == "length" and r.gen_end >= cfg.log.l_log)]
        source.drop(full)
        active -= {r.key for r in ended} | set(full)


def study_answer_mode(
    source: LiveSource, keys: list[TraceKey], cfg: ExperimentCfg
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record(loc: str, segs: list[SegmentResult], finalize: bool) -> None:
        res = source.answer_modes([r.key for r in segs], ("full", "short"), finalize=finalize)
        for r, f, s in zip(segs, res["full"], res["short"], strict=True):
            rows.append(
                {
                    "qid": r.key.qid,
                    "seed": r.key.seed,
                    "location": loc,
                    "gen_end": r.gen_end,
                    "full_correct": bool(f.correct),
                    "short_correct": bool(s.correct),
                    "full_extracted": f.extracted,
                    "short_extracted": s.extracted,
                    "full_cost": f.prompt_tokens + f.output_tokens,
                    "short_cost": s.prompt_tokens + s.output_tokens,
                }
            )

    walk_chains(
        source, keys, cfg, lambda segs: record("point", segs, False), lambda segs: record("natural", segs, True)
    )
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for loc, g in df.groupby("location") if len(df) else []:
        summary[str(loc)] = {
            "n": len(g),
            "same_grade_rate": float((g.full_correct == g.short_correct).mean()),
            "full_accuracy": float(g.full_correct.mean()),
            "short_accuracy": float(g.short_correct.mean()),
            "full_unextracted_rate": float(g.full_extracted.isna().mean()),
            "short_unextracted_rate": float(g.short_extracted.isna().mean()),
            "full_mean_cost": float(g.full_cost.mean()),
            "short_mean_cost": float(g.short_cost.mean()),
        }
    rates = [v["same_grade_rate"] for v in summary.values()]
    summary["proposal"] = "short" if rates and min(rates) >= 0.98 else "full"
    summary["rule"] = "chọn short nếu tỉ lệ cùng kết quả chấm >= 98% ở cả hai loại vị trí (PLAN P1.T9)"
    return df, summary


def study_probe_noise(
    source: LiveSource, keys: list[TraceKey], cfg: ExperimentCfg, alt: PromptSet, n_bins: int = 5
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def at_point(segs: list[SegmentResult]) -> None:
        ks = [r.key for r in segs]
        a = source.probe(ks)
        b = source.probe_with(ks, alt)
        for r, p, q in zip(segs, a, b, strict=True):
            rows.append(
                {
                    "qid": r.key.qid,
                    "seed": r.key.seed,
                    "pos": r.pos,
                    "key_a": p.answer_key,
                    "key_b": q.answer_key,
                    "conf_a": p.conf_geo,
                    "conf_b": q.conf_geo,
                }
            )

    lengths: dict[tuple[str, int], int] = {}

    def at_end(segs: list[SegmentResult]) -> None:
        for r in segs:
            lengths[(r.key.qid, r.key.seed)] = r.gen_end
        source.drop([r.key for r in segs])

    walk_chains(source, keys, cfg, at_point, at_end)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"n": 0}
    df["chain_len"] = [lengths.get((q, s), cfg.log.l_log) for q, s in zip(df.qid, df.seed, strict=True)]
    df["rel_pos"] = (df.pos / df.chain_len).clip(0, 1)
    df["differs"] = df.key_a != df.key_b
    df["bin"] = np.minimum((df.rel_pos * n_bins).astype(int), n_bins - 1)
    by_bin = df.groupby("bin").agg(n=("differs", "size"), differ_rate=("differs", "mean")).reset_index()
    summary = {
        "n": len(df),
        "differ_rate": float(df.differs.mean()),
        "conf_corr": float(np.corrcoef(df.conf_a, df.conf_b)[0, 1]) if len(df) > 1 else float("nan"),
        "by_rel_pos": [
            {"rel_pos": f"[{b / n_bins:.1f}, {(b + 1) / n_bins:.1f})", "n": int(n), "differ_rate": float(d)}
            for b, n, d in zip(by_bin["bin"], by_bin["n"], by_bin["differ_rate"], strict=True)
        ],
    }
    return df, summary
