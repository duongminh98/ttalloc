"""Sinh dữ liệu ghi tổng hợp đúng schema của `traces` và `points` (PLAN P0.T5, mục 2.5).

Đường cong đúng sai theo số token của mỗi câu có dạng chữ S với tham số ngẫu nhiên; độ tự tin của đáp án thử
tương quan với tính đúng sai. Dữ liệu này chỉ dùng để kiểm thử mã, không bao giờ được báo cáo như kết quả.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from ttalloc.logging.schema import POINTS_SCHEMA, TRACES_SCHEMA, to_table


@dataclass(frozen=True)
class SynthKnobs:
    l_log: int = 4096
    log_m: int = 128
    log_l_max: int = 256
    p_natural: float = 0.7
    p_long_gap: float = 0.15  # xác suất bắt đầu một đoạn dài không có từ chuyển ý
    model: str = "synth"
    split: str = "val"
    log_hash: str = "synth"
    force_cases: bool = True  # chuỗi 0 tự kết thúc, chuỗi 1 chạm L_log, chuỗi 2 có đoạn dài không có từ chuyển ý


MARKERS = ("Wait", "Alternatively", "Hmm", "But", "Let me check")


def _q_curve(b: float, p_max: float, center: float, slope: float) -> float:
    return float(p_max / (1.0 + np.exp(-slope * (b - center))))


def make_logs(
    n_questions: int, seeds: list[int] | range, rng: np.random.Generator, **knobs: Any
) -> tuple[pa.Table, pa.Table]:
    k = SynthKnobs(**knobs)
    traces: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    chain_no = 0
    for i in range(n_questions):
        qid = f"synth/{i}"
        p_max = rng.uniform(0.3, 1.0)
        center = rng.uniform(0.05, 0.6) * k.l_log
        slope = 1.0 / rng.uniform(0.02, 0.15) / k.l_log
        for seed in seeds:
            forced = chain_no if k.force_cases else -1
            natural = forced == 0 or (forced != 1 and rng.random() < k.p_natural)
            length = int(rng.uniform(0.3, 0.95) * k.l_log) if natural else k.l_log
            u = rng.random()  # cùng một ngưỡng cho cả chuỗi: đã đúng thì tiếp tục đúng khi suy luận thêm

            def correct_at(
                b: float, u: float = u, p_max: float = p_max, center: float = center, slope: float = slope
            ) -> bool:
                return u < _q_curve(b, p_max, center, slope)

            rows = _points_for_chain(qid, seed, length, k, rng, correct_at, force_long_gap=forced == 2)
            points.extend(rows)
            end_reason = ("think_end" if rng.random() < 0.9 else "eos") if natural else "max_len"
            nat_ok = correct_at(length) if natural else None
            traces.append(
                {
                    "qid": qid,
                    "seed": int(seed),
                    "model": k.model,
                    "split": k.split,
                    "log_hash": k.log_hash,
                    "chain_ids": rng.integers(0, 150_000, size=length).astype(np.int32).tolist(),
                    "chain_logprobs": (-rng.exponential(0.3, size=length)).astype(np.float16).tolist(),
                    "chain_len": length,
                    "natural_end": natural,
                    "end_reason": end_reason,
                    "nat_answer_text": ("\\boxed{gold}" if nat_ok else "\\boxed{wrong}") if natural else None,
                    "nat_answer_extracted": ("gold" if nat_ok else "wrong") if natural else None,
                    "nat_answer_prompt_tokens": 3 if natural else None,
                    "nat_answer_output_tokens": int(rng.integers(8, 40)) if natural else None,
                    "nat_correct": nat_ok,
                    "gold": "gold",
                }
            )
            chain_no += 1
    return to_table(traces, TRACES_SCHEMA), to_table(points, POINTS_SCHEMA)


def _points_for_chain(
    qid: str,
    seed: int,
    length: int,
    k: SynthKnobs,
    rng: np.random.Generator,
    correct_at: Any,
    force_long_gap: bool,
) -> list[dict[str, Any]]:
    rows = [_start_row(qid, seed, rng, bool(correct_at(0)))]
    cursor = 0
    n_markers = 0
    long_left = 0
    forced_done = not force_long_gap
    while True:
        if long_left == 0 and (not forced_done or rng.random() < k.p_long_gap):
            long_left = int(rng.integers(2, 5))
            forced_done = True
        if long_left > 0:
            gen_end, kind, pos = cursor + k.log_l_max, "length", cursor + k.log_l_max
            long_left -= 1
        else:
            gen_end = cursor + int(rng.integers(k.log_m + 1, k.log_l_max + 1))
            kind, pos = "marker", gen_end - int(rng.integers(1, 4))
        if gen_end >= length:
            break
        ok = bool(correct_at(pos))
        conf = float(rng.beta(8, 2) if ok else rng.beta(2, 4))
        n_probe_out = int(rng.integers(3, 12))
        probe_lp = np.log(np.clip(rng.normal(conf, 0.05, size=n_probe_out), 1e-3, 1.0)).astype(np.float32)
        rows.append(
            {
                "qid": qid,
                "seed": int(seed),
                "point_idx": len(rows),
                "kind": kind,
                "marker": str(rng.choice(MARKERS)) if kind == "marker" else None,
                "pos": pos,
                "gen_end": gen_end,
                "n_markers": n_markers,
                "local_conf_min": float(rng.uniform(0.05, 0.9)),
                "probe_prompt_tokens": 6,
                "probe_output_tokens": n_probe_out,
                "probe_output_ids": rng.integers(0, 150_000, size=n_probe_out).astype(np.int32).tolist(),
                "probe_logprobs": probe_lp.tolist(),
                "probe_answer_raw": "gold" if ok else f"w{int(rng.integers(0, 3))}",
                "probe_answer_key": "str:gold" if ok else f"str:w{int(rng.integers(0, 3))}",
                "probe_conf_geo": float(np.exp(probe_lp.mean())),
                "probe_conf_arith": float(np.exp(probe_lp).mean()),
                "probe_has_think_end": bool(rng.random() < 0.8),
                "answer_prompt_tokens": 3,
                "answer_output_tokens": int(rng.integers(8, 40)),
                "answer_text": "\\boxed{gold}" if ok else "\\boxed{wrong}",
                "answer_extracted": "gold" if ok else "wrong",
                "answer_correct": ok,
            }
        )
        if kind == "marker":
            n_markers += 1
        cursor = gen_end
    return rows


def _start_row(qid: str, seed: int, rng: np.random.Generator, ok: bool) -> dict[str, Any]:
    return {
        "qid": qid,
        "seed": int(seed),
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
        "answer_prompt_tokens": 3,
        "answer_output_tokens": int(rng.integers(8, 40)),
        "answer_text": "\\boxed{gold}" if ok else "\\boxed{wrong}",
        "answer_extracted": "gold" if ok else "wrong",
        "answer_correct": ok,
    }
