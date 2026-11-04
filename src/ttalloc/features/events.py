"""Sự kiện kế tiếp trên chuỗi đã ghi (A2) và dãy điểm chuẩn (A3).

`next_event` là nguồn chung cho `LoggedSource.advance` (P2.T9) và cho việc chọn dãy chuẩn; không cài đặt hai lần.
`LoggedChain` giữ một chuỗi đã ghi dưới dạng mảng numpy để tra cứu nhanh bằng `searchsorted`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from ttalloc.config import DecisionPointCfg
from ttalloc.types import ProbeResult, TraceKey

KIND_CODES = {"start": 0, "marker": 1, "length": 2}
EventKind = Literal["natural_end", "marker", "length", "truncated"]


class SimValidityError(RuntimeError):
    """Vi phạm I8: đoạn được cấp không chứa điểm đã ghi nào và chuỗi chưa kết thúc trong hạn mức."""

    def __init__(self, key: TraceKey, s: int, lim: int, detail: str = "") -> None:
        super().__init__(f"{key}: không có sự kiện nào trong ({s}, {lim}] {detail}".strip())
        self.key, self.s, self.lim = key, s, lim


@dataclass(frozen=True)
class Event:
    kind: EventKind
    gen_end: int
    idx: int | None = None  # chỉ số điểm (natural_end: None)


@dataclass
class LoggedChain:
    key: TraceKey
    chain_len: int
    natural_end: bool
    end_reason: str
    nat_correct: bool | None
    nat_prompt_tokens: int
    nat_output_tokens: int
    nat_extracted: str | None
    gold: str
    gen_end: np.ndarray
    pos: np.ndarray
    kind: np.ndarray
    marker_idx: np.ndarray
    marker_ge: np.ndarray
    ans_correct: np.ndarray
    ans_prompt: np.ndarray
    ans_output: np.ndarray
    ans_extracted: list[str | None]
    probe_prompt: np.ndarray
    probe_output: np.ndarray
    conf_geo: np.ndarray
    conf_arith: np.ndarray
    probe_key: list[str | None]
    probe_raw: list[str | None]
    has_think_end: np.ndarray
    ans_tokens: np.ndarray
    n_markers: np.ndarray
    local_conf_min: np.ndarray

    @property
    def n_points(self) -> int:
        return len(self.gen_end)

    def probe(self, i: int) -> ProbeResult:
        if i == 0:
            raise ValueError(f"{self.key}: điểm start không có đáp án thử")
        return ProbeResult(
            key=self.key,
            answer_raw=self.probe_raw[i] or "",
            answer_key=self.probe_key[i] or "",
            conf_geo=float(self.conf_geo[i]),
            conf_arith=float(self.conf_arith[i]),
            logprobs=(0.0,) * int(self.ans_tokens[i]),
            prompt_tokens=int(self.probe_prompt[i]),
            output_tokens=int(self.probe_output[i]),
            has_think_end=bool(self.has_think_end[i]),
            gen_end=int(self.gen_end[i]),
        )

    def answer_cost(self, i: int) -> int:
        return int(self.ans_prompt[i] + self.ans_output[i])

    def probe_cost(self, i: int) -> int:
        return int(self.probe_prompt[i] + self.probe_output[i])

    def nat_cost(self) -> int:
        return self.nat_prompt_tokens + self.nat_output_tokens

    def truncate_after(self, j: int) -> LoggedChain:
        """Bản sao chỉ giữ các điểm 0..j và coi chuỗi như chưa kết thúc (dùng cho test không nhìn trước, I4)."""
        sl = slice(0, j + 1)
        mk = self.marker_idx <= j
        return LoggedChain(
            key=self.key,
            chain_len=int(self.gen_end[j]) + 1,
            natural_end=False,
            end_reason="max_len",
            nat_correct=None,
            nat_prompt_tokens=0,
            nat_output_tokens=0,
            nat_extracted=None,
            gold=self.gold,
            gen_end=self.gen_end[sl],
            pos=self.pos[sl],
            kind=self.kind[sl],
            marker_idx=self.marker_idx[mk],
            marker_ge=self.marker_ge[mk],
            ans_correct=self.ans_correct[sl],
            ans_prompt=self.ans_prompt[sl],
            ans_output=self.ans_output[sl],
            ans_extracted=self.ans_extracted[sl],
            probe_prompt=self.probe_prompt[sl],
            probe_output=self.probe_output[sl],
            conf_geo=self.conf_geo[sl],
            conf_arith=self.conf_arith[sl],
            probe_key=self.probe_key[sl],
            probe_raw=self.probe_raw[sl],
            has_think_end=self.has_think_end[sl],
            ans_tokens=self.ans_tokens[sl],
            n_markers=self.n_markers[sl],
            local_conf_min=self.local_conf_min[sl],
        )


def _i(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def _int0(x: Any) -> int:
    """Giá trị nguyên có thể thiếu (None hoặc NaN sau khi qua pandas) -> 0."""
    return 0 if x is None or x != x else int(x)


def chains_from_tables(traces: pa.Table, points: pa.Table) -> dict[TraceKey, LoggedChain]:
    tr = traces.drop_columns(["chain_ids", "chain_logprobs"]).to_pandas()
    pt = points.drop_columns(["probe_output_ids", "answer_text"]).to_pandas()
    pt = pt.sort_values(["qid", "seed", "point_idx"], kind="stable")
    pt["ans_tokens"] = pt.probe_logprobs.map(lambda v: 0 if v is None else len(v))
    groups = {k: g for k, g in pt.groupby(["qid", "seed"], sort=False)}
    out: dict[TraceKey, LoggedChain] = {}
    for r in tr.itertuples(index=False):
        key = TraceKey(str(r.qid), int(r.seed))
        g = groups.get((r.qid, r.seed))
        if g is None:
            raise ValueError(f"{key}: không có điểm nào trong bảng points")
        kind = g.kind.map(KIND_CODES).to_numpy(np.int8)
        ge = _i(g.gen_end)
        if kind[0] != 0 or ge[0] != 0:
            raise ValueError(f"{key}: thiếu điểm start")
        if np.any(np.diff(ge) <= 0):
            raise ValueError(f"{key}: điểm không tăng ngặt theo gen_end")
        midx = np.flatnonzero(kind == 1)
        out[key] = LoggedChain(
            key=key,
            chain_len=int(r.chain_len),
            natural_end=bool(r.natural_end),
            end_reason=str(r.end_reason),
            nat_correct=None if r.nat_correct is None or r.nat_correct != r.nat_correct else bool(r.nat_correct),
            nat_prompt_tokens=_int0(r.nat_answer_prompt_tokens),
            nat_output_tokens=_int0(r.nat_answer_output_tokens),
            nat_extracted=r.nat_answer_extracted,
            gold=str(r.gold),
            gen_end=ge,
            pos=_i(g.pos),
            kind=kind,
            marker_idx=midx,
            marker_ge=ge[midx],
            ans_correct=g.answer_correct.to_numpy(bool),
            ans_prompt=_i(g.answer_prompt_tokens),
            ans_output=_i(g.answer_output_tokens),
            ans_extracted=list(g.answer_extracted),
            probe_prompt=_i(g.probe_prompt_tokens.fillna(0)),
            probe_output=_i(g.probe_output_tokens.fillna(0)),
            conf_geo=g.probe_conf_geo.to_numpy(float),
            conf_arith=g.probe_conf_arith.to_numpy(float),
            probe_key=list(g.probe_answer_key),
            probe_raw=list(g.probe_answer_raw),
            has_think_end=g.probe_has_think_end.fillna(False).to_numpy(bool),
            ans_tokens=_i(g.ans_tokens),
            n_markers=_i(g.n_markers),
            local_conf_min=g.local_conf_min.to_numpy(float),
        )
    return out


def next_event(c: LoggedChain, cur: int, s: int, min_tokens: int, max_tokens: int, stop_at_markers: bool) -> Event:
    """A2. `s` là gen_end hiện tại; `cur` là chỉ số điểm đang đứng (gen_end[cur] <= s)."""
    lim = s + max_tokens
    marker: int | None = None
    if stop_at_markers and len(c.marker_ge):
        j = int(np.searchsorted(c.marker_ge, s + min_tokens, side="right"))
        if j < len(c.marker_ge) and c.marker_ge[j] <= lim and c.marker_idx[j] > cur:
            marker = int(c.marker_idx[j])
    if c.natural_end and c.chain_len <= lim and (marker is None or c.chain_len <= c.gen_end[marker]):
        return Event("natural_end", gen_end=c.chain_len)
    if marker is not None:
        return Event("marker", gen_end=int(c.gen_end[marker]), idx=marker)
    i = int(np.searchsorted(c.gen_end, lim, side="right")) - 1
    if i > cur and c.gen_end[i] > s:
        return Event("length", gen_end=int(c.gen_end[i]), idx=i)  # làm tròn xuống điểm đã ghi gần nhất
    if not c.natural_end and c.chain_len <= lim:
        return Event("truncated", gen_end=c.chain_len, idx=cur)  # chuỗi ghi đã chạm L_log
    raise SimValidityError(c.key, s, lim, f"(chain_len={c.chain_len}, natural_end={c.natural_end})")


def canonical_points(c: LoggedChain, dp: DecisionPointCfg) -> list[int]:
    """A3: dãy điểm mà một chính sách luôn tiếp tục sẽ ghé (không gồm điểm start)."""
    out: list[int] = []
    cur, s = 0, 0
    min_tok, max_tok = dp.b_min, dp.b_min + dp.l_max
    while True:
        ev = next_event(c, cur, s, min_tok, max_tok, stop_at_markers=True)
        if ev.kind in ("natural_end", "truncated"):
            return out
        assert ev.idx is not None
        out.append(ev.idx)
        cur, s = ev.idx, ev.gen_end
        min_tok, max_tok = dp.m, dp.l_max
