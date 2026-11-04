"""Kiểm tra dữ liệu ghi (PLAN P1.T11). Kết quả dùng cho tiêu chí (c) của cổng G1: tỉ lệ chuỗi đạt."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from ttalloc.config import ExperimentCfg
from ttalloc.logging.schema import POINTS_SCHEMA, TRACES_SCHEMA, SchemaError, validate_table


def validate_logs(
    traces: pa.Table,
    points: pa.Table,
    cfg: ExperimentCfg,
    log_hash: str | None = None,
    think_end_id: int | None = None,
    c_ans_max: int | None = None,
    c_probe_max: int | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Trả về báo cáo: số chuỗi, tỉ lệ chuỗi đạt, số lỗi theo loại và ví dụ lỗi (khóa, điểm, giá trị)."""
    for t, s, name in ((traces, TRACES_SCHEMA, "traces"), (points, POINTS_SCHEMA, "points")):
        try:
            validate_table(t, s)
        except SchemaError as exc:
            return {"ok": False, "schema_error": f"{name}: {exc}", "n_chains": traces.num_rows, "pass_rate": 0.0}
    tr = traces.select(
        [
            "qid",
            "seed",
            "log_hash",
            "chain_ids",
            "chain_len",
            "natural_end",
            "end_reason",
            "nat_answer_extracted",
            "nat_answer_prompt_tokens",
            "nat_answer_output_tokens",
        ]
    ).to_pandas()
    pt = points.drop_columns(["probe_logprobs", "answer_text"]).to_pandas()
    errors: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    bad: set[tuple[str, int]] = set()

    def err(kind: str, key: tuple[str, int], **info: Any) -> None:
        errors[kind] += 1
        bad.add(key)
        if len(examples) < max_examples:
            examples.append({"error": kind, "qid": key[0], "seed": key[1], **info})

    dup = tr.duplicated(["qid", "seed"])
    for _, r in tr[dup].iterrows():
        err("duplicate_trace", (r.qid, int(r.seed)))
    groups = {k: g for k, g in pt.groupby(["qid", "seed"], sort=False)}
    for r in tr.itertuples(index=False):
        key = (r.qid, int(r.seed))
        if log_hash is not None and r.log_hash != log_hash:
            err("log_hash", key, value=r.log_hash, expected=log_hash)
        if r.chain_len != len(r.chain_ids):
            err("chain_len", key, chain_len=int(r.chain_len), n_ids=len(r.chain_ids))
        if r.natural_end != (r.end_reason in ("think_end", "eos")):
            err("end_reason", key, end_reason=r.end_reason, natural_end=bool(r.natural_end))
        if r.end_reason == "max_len" and r.chain_len < cfg.log.l_log:
            err("max_len_short", key, chain_len=int(r.chain_len))
        if r.chain_len > cfg.log.l_log:
            err("chain_too_long", key, chain_len=int(r.chain_len))
        if (
            think_end_id is not None
            and r.end_reason == "think_end"
            and (len(r.chain_ids) == 0 or int(r.chain_ids[-1]) != think_end_id)
        ):
            err("think_end_missing", key)
        if (
            r.natural_end
            and c_ans_max is not None
            and ((r.nat_answer_prompt_tokens or 0) + (r.nat_answer_output_tokens or 0) > c_ans_max)
        ):
            err("nat_answer_cost", key)
        g = groups.get(key)
        if g is None or g.empty:
            err("no_points", key)
            continue
        g = g.sort_values("point_idx")
        idx = g.point_idx.to_numpy()
        ge = g.gen_end.to_numpy()
        if idx[0] != 0 or g.kind.iloc[0] != "start" or ge[0] != 0 or g.pos.iloc[0] != 0:
            err("start_point", key)
        if not np.array_equal(idx, np.arange(len(idx))):
            err("point_idx_gaps", key)
        if np.any(np.diff(ge) <= 0):
            err("not_sorted_by_gen_end", key)
        if np.any(g.pos.to_numpy() > ge):
            err("pos_after_gen_end", key)
        gaps = np.diff(ge)
        if np.any(gaps > cfg.log.l_max):
            err("gap_over_l_max", key, max_gap=int(gaps.max()))
        mk = g.kind.to_numpy()[1:] == "marker"
        if np.any(gaps[mk] <= cfg.log.m):
            err("marker_too_close", key, min_gap=int(gaps[mk].min()))
        if ge[-1] > r.chain_len:
            err("point_after_chain_end", key)
        rest = g.iloc[1:]
        if len(rest):
            if (rest.probe_output_tokens.to_numpy() != rest.probe_output_ids.map(len).to_numpy()).any():
                err("probe_token_count", key)
            if (
                c_probe_max is not None
                and ((rest.probe_prompt_tokens + rest.probe_output_tokens).to_numpy() > c_probe_max).any()
            ):
                err("probe_cost", key)
        if c_ans_max is not None and ((g.answer_prompt_tokens + g.answer_output_tokens).to_numpy() > c_ans_max).any():
            err("answer_cost", key)
    n = len(tr)
    nat = tr[tr.natural_end]
    rep = {
        "ok": not errors,
        "n_chains": n,
        "n_points": len(pt),
        "pass_rate": (n - len(bad)) / n if n else 0.0,
        "errors": dict(errors),
        "examples": examples,
        "max_len_rate": float((tr.end_reason == "max_len").mean()) if n else 0.0,
        "eos_rate": float((tr.end_reason == "eos").mean()) if n else 0.0,
        "unextracted_point_rate": float(pt.answer_extracted.isna().mean()) if len(pt) else 0.0,
        "unextracted_natural_rate": float(nat.nat_answer_extracted.isna().mean()) if len(nat) else 0.0,
        "mean_chain_len": float(tr.chain_len.mean()) if n else 0.0,
        "median_chain_len": float(tr.chain_len.median()) if n else 0.0,
        "points_per_chain": len(pt) / n if n else 0.0,
    }
    rep["token_ratio"] = token_ratio(tr, pt)
    return rep


def token_ratio(tr: pd.DataFrame, pt: pd.DataFrame) -> float:
    """Tổng token sinh khi ghi (chuỗi chính, gợi, trả lời) chia cho số token chuỗi chính (P1, bước chạy 5)."""
    chain = float(tr.chain_len.sum())
    if chain == 0:
        return 0.0
    extra = float(pt.probe_output_tokens.fillna(0).sum() + pt.answer_output_tokens.sum())
    extra += float(tr.nat_answer_output_tokens.fillna(0).sum())
    return (chain + extra) / chain
