"""Ghi dữ liệu cho bộ ước lượng và bộ mô phỏng (PLAN P1.T8).

Logger dùng trực tiếp `LiveSource` (không qua Runner): tối đa `log.window` chuỗi chạy đồng thời, và tổng ngữ cảnh
không quá `log.kv_fraction` KV cache; mỗi vòng sinh các chuỗi tới điểm kế tiếp với `min_tokens = log.m`,
`max_tokens = log.l_max`; tại mọi điểm mới gợi đáp án thử và sinh câu trả lời ép buộc; chuỗi tự kết thúc thì sinh
câu trả lời tự nhiên; chuỗi chạm L_log được đánh dấu `max_len`.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ttalloc.config import ExperimentCfg
from ttalloc.engine.live_source import LiveSource
from ttalloc.logging.store import LogStore
from ttalloc.types import AdvanceRequest, AnswerResult, ProbeResult, SegmentResult, TraceKey

log = logging.getLogger("ttalloc.logger")


@dataclass
class _Rec:
    points: list[dict[str, Any]] = field(default_factory=list)


def start_row(key: TraceKey, a: AnswerResult) -> dict[str, Any]:
    return {
        "qid": key.qid,
        "seed": key.seed,
        "point_idx": 0,
        "kind": "start",
        "marker": None,
        "pos": 0,
        "gen_end": 0,
        "n_markers": 0,
        "local_conf_min": None,
        "probe_prompt_tokens": None,
        "probe_output_tokens": None,
        "probe_output_ids": None,
        "probe_logprobs": None,
        "probe_answer_raw": None,
        "probe_answer_key": None,
        "probe_conf_geo": None,
        "probe_conf_arith": None,
        "probe_has_think_end": None,
        "answer_prompt_tokens": a.prompt_tokens,
        "answer_output_tokens": a.output_tokens,
        "answer_text": a.text,
        "answer_extracted": a.extracted,
        "answer_correct": bool(a.correct),
    }


def point_row(idx: int, r: SegmentResult, p: ProbeResult, a: AnswerResult) -> dict[str, Any]:
    return {
        "qid": r.key.qid,
        "seed": r.key.seed,
        "point_idx": idx,
        "kind": r.kind,
        "marker": r.marker,
        "pos": r.pos,
        "gen_end": r.gen_end,
        "n_markers": r.n_markers,
        "local_conf_min": r.local_conf_min,
        "probe_prompt_tokens": p.prompt_tokens,
        "probe_output_tokens": p.output_tokens,
        "probe_output_ids": list(p.output_ids),
        "probe_logprobs": list(p.logprobs),
        "probe_answer_raw": p.answer_raw,
        "probe_answer_key": p.answer_key,
        "probe_conf_geo": p.conf_geo,
        "probe_conf_arith": p.conf_arith,
        "probe_has_think_end": p.has_think_end,
        "answer_prompt_tokens": a.prompt_tokens,
        "answer_output_tokens": a.output_tokens,
        "answer_text": a.text,
        "answer_extracted": a.extracted,
        "answer_correct": bool(a.correct),
    }


def run_logger(
    source: LiveSource,
    keys: list[TraceKey],
    store: LogStore,
    cfg: ExperimentCfg,
    model: str,
    split: str,
    log_hash: str,
    gold: dict[str, str],
    on_round: Callable[[dict[str, Any]], None] | None = None,
    kv_tokens: int | None = None,
) -> dict[str, Any]:
    if source.l_log != cfg.log.l_log:
        raise ValueError(f"LiveSource.l_log ({source.l_log}) phải bằng log.l_log ({cfg.log.l_log})")
    done = store.completed_keys()
    queue = deque(k for k in keys if k not in done)
    stats: dict[str, Any] = {"skipped": len(keys) - len(queue), "chains": 0, "points": 0, "max_len": 0, "rounds": 0}
    active: dict[TraceKey, _Rec] = {}
    buf_traces: list[dict[str, Any]] = []
    buf_points: list[dict[str, Any]] = []

    # Giới hạn theo KV cache: tổng ngữ cảnh của các chuỗi chạy trong một vòng phải vừa KV, để prefix của chúng còn
    # trong cache giữa lượt sinh, lượt gợi và lượt trả lời. Bản tập dượt Kaggle lần 1–2 chạy 256 chuỗi trên T4 (KV
    # 74k token): KV đầy 99%, tỉ lệ trúng prefix cache 13%, mỗi vòng tính lại cả ngữ cảnh, vòng 3 mất 550 s.
    # Seed của từng yêu cầu chỉ phụ thuộc khóa và vị trí, nên thứ tự và cách gom lô không đổi dữ liệu.
    budget = int(kv_tokens * cfg.log.kv_fraction) if kv_tokens else None

    def demand(k: TraceKey) -> int:
        return source.kv_demand(k, cfg.log.l_max)

    def admit() -> None:
        new: list[TraceKey] = []
        used = sum(demand(k) for k in active)
        while queue and len(active) + len(new) < cfg.log.window:
            need = demand(queue[0])
            if budget is not None and active and used + need > budget:
                break
            used += need
            new.append(queue.popleft())
        if not new:
            return
        source.start(new)
        for k, a in zip(new, source.answer(new, finalize=False), strict=True):
            active[k] = _Rec(points=[start_row(k, a)])

    def finish(key: TraceKey, end_reason: str, nat: AnswerResult | None, chain: list[int], lps: list[float]) -> None:
        rec = active.pop(key)
        buf_points.extend(rec.points)
        buf_traces.append(
            {
                "qid": key.qid,
                "seed": key.seed,
                "model": model,
                "split": split,
                "log_hash": log_hash,
                "chain_ids": chain,
                "chain_logprobs": np.asarray(lps, dtype=np.float16).tolist(),
                "chain_len": len(chain),
                "natural_end": nat is not None,
                "end_reason": end_reason,
                "nat_answer_text": nat.text if nat else None,
                "nat_answer_extracted": nat.extracted if nat else None,
                "nat_answer_prompt_tokens": nat.prompt_tokens if nat else None,
                "nat_answer_output_tokens": nat.output_tokens if nat else None,
                "nat_correct": bool(nat.correct) if nat else None,
                "gold": gold[key.qid],
            }
        )
        stats["chains"] += 1
        stats["points"] += len(rec.points)

    def flush(force: bool = False) -> None:
        if buf_traces and (force or len(buf_traces) >= cfg.log.flush_every):
            store.write_part(list(buf_traces), list(buf_points))
            buf_traces.clear()
            buf_points.clear()

    admit()
    t_start = time.time()
    while active:
        t0 = time.time()
        # Chuỗi dài ra qua các vòng: nếu tổng vượt KV thì vòng này chỉ chạy các chuỗi vào trước (để chúng sớm xong).
        run_keys: list[TraceKey] = []
        used = 0
        for k in active:
            need = demand(k)
            if budget is not None and run_keys and used + need > budget:
                continue
            used += need
            run_keys.append(k)
        segs = source.advance([AdvanceRequest(k, cfg.log.m, cfg.log.l_max, True) for k in run_keys])
        at_point = [r for r in segs if r.kind in ("marker", "length")]
        ended = [r for r in segs if r.kind in ("natural_end", "eos")]
        truncated = [r for r in segs if r.kind == "truncated"]
        t1 = time.time()
        if at_point:
            probes, answers = source.probe_and_answer([r.key for r in at_point])
            for r, p, a in zip(at_point, probes, answers[source.answer_mode], strict=True):
                rec = active[r.key]
                rec.points.append(point_row(len(rec.points), r, p, a))
        t2 = time.time()
        if ended:
            snaps = {r.key: (list(source.chains[r.key].chain), list(source.chains[r.key].logprobs)) for r in ended}
            nats = source.answer([r.key for r in ended], finalize=True)
            for r, a in zip(ended, nats, strict=True):
                finish(r.key, "think_end" if r.kind == "natural_end" else "eos", a, *snaps[r.key])
        # Chuỗi chạm L_log: điểm cuối (nếu có) đã được ghi ở trên; kết thúc với max_len.
        full = [r for r in at_point if r.gen_end >= cfg.log.l_log] + truncated
        for r in full:
            ch = source.chains[r.key]
            finish(r.key, "max_len", None, list(ch.chain), list(ch.logprobs))
            stats["max_len"] += 1
        source.drop([r.key for r in full])
        stats["rounds"] += 1
        flush()
        admit()
        info = {
            "round": stats["rounds"],
            "active": len(active),
            "ran": len(run_keys),
            "kv_demand": used,
            "queued": len(queue),
            "done": stats["chains"],
            "t_advance": round(t1 - t0, 2),
            "t_probe_answer": round(t2 - t1, 2),
            "elapsed": round(time.time() - t_start, 1),
        }
        if on_round:
            on_round(info)
        if stats["rounds"] % 20 == 0:
            log.info("%s", info)
    flush(force=True)
    stats["seconds"] = round(time.time() - t_start, 1)
    return stats
