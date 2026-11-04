"""SegmentSource đọc dữ liệu ghi (PLAN P2.T9). `advance` dùng `next_event` (A2); chi phí lấy đúng số token đã ghi."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from ttalloc.data.grading import grade
from ttalloc.features.events import LoggedChain, chains_from_tables, next_event
from ttalloc.logging.store import LogHashMismatch, LogStore
from ttalloc.types import AdvanceRequest, AnswerResult, ProbeResult, SegmentResult, TraceKey

OnStop = Literal["forced", "best_probe", "vote"]


@lru_cache(maxsize=1 << 18)
def _grade_cached(pred: str, gold: str, answer_type: str) -> bool:
    return grade(pred, gold, answer_type)  # type: ignore[arg-type]


def load_logged(
    model: str, split: str, expected_log_hash: str | None, logs_dir: Path | None = None
) -> tuple[dict[TraceKey, LoggedChain], dict[str, Any]]:
    """Nạp dữ liệu ghi của một tập; từ chối nếu `log_hash` khác config hiện hành (I9)."""
    store = LogStore.for_split(model, split) if logs_dir is None else LogStore(logs_dir / model / split)
    meta = store.meta()
    if meta is None:
        raise FileNotFoundError(f"chưa có dữ liệu ghi ở {store.root}")
    if expected_log_hash is not None and meta["log_hash"] != expected_log_hash:
        raise LogHashMismatch(
            f"{store.root}: log_hash {meta['log_hash']} khác config hiện hành {expected_log_hash} (I9)"
        )
    return chains_from_tables(store.read_traces(), store.read_points()), meta


@dataclass
class _St:
    cur: int = 0
    s: int = 0
    ended: str | None = None
    probes: list[int] = field(default_factory=list)


class LoggedSource:
    def __init__(
        self,
        chains: dict[TraceKey, LoggedChain],
        on_stop: OnStop = "forced",
        answer_types: dict[str, str] | None = None,
    ) -> None:
        self.chains = chains
        self.on_stop = on_stop
        self.answer_types = answer_types or {}
        self.st: dict[TraceKey, _St] = {}

    def start(self, keys: list[TraceKey]) -> None:
        for k in keys:
            if k not in self.chains:
                raise KeyError(f"{k}: không có trong dữ liệu ghi")
            if k in self.st:
                raise ValueError(f"{k}: đã start")
            self.st[k] = _St()

    def advance(self, reqs: list[AdvanceRequest]) -> list[SegmentResult]:
        out = []
        for r in reqs:
            st, c = self.st[r.key], self.chains[r.key]
            if st.ended is not None:
                raise ValueError(f"{r.key}: chuỗi đã kết thúc ({st.ended})")
            ev = next_event(c, st.cur, st.s, r.min_tokens, r.max_tokens, r.stop_at_markers)
            n_new = ev.gen_end - st.s
            if ev.kind == "natural_end":
                kind = "eos" if c.end_reason == "eos" else "natural_end"
                st.ended, st.s = kind, ev.gen_end
                out.append(SegmentResult(r.key, kind, pos=c.chain_len, gen_end=c.chain_len, n_new_tokens=n_new))  # type: ignore[arg-type]
            elif ev.kind == "truncated":
                st.ended, st.s = "truncated", ev.gen_end
                out.append(
                    SegmentResult(r.key, "truncated", pos=int(c.pos[st.cur]), gen_end=ev.gen_end, n_new_tokens=n_new)
                )
            else:
                i = int(ev.idx)  # type: ignore[arg-type]
                st.cur, st.s = i, ev.gen_end
                lc = float(c.local_conf_min[i])
                out.append(
                    SegmentResult(
                        r.key,
                        ev.kind,
                        pos=int(c.pos[i]),
                        gen_end=ev.gen_end,
                        n_new_tokens=n_new,
                        point_idx=i,
                        n_markers=int(c.n_markers[i]),
                        local_conf_min=None if lc != lc else lc,
                    )
                )
        return out

    def probe(self, keys: list[TraceKey]) -> list[ProbeResult]:
        out = []
        for k in keys:
            st = self.st[k]
            st.probes.append(st.cur)
            out.append(self.chains[k].probe(st.cur))
        return out

    def answer(self, keys: list[TraceKey]) -> list[AnswerResult]:
        out = [self._answer(k) for k in keys]
        for k in keys:
            del self.st[k]
        return out

    def _answer(self, k: TraceKey) -> AnswerResult:
        st, c = self.st[k], self.chains[k]
        if st.ended in ("natural_end", "eos"):
            return AnswerResult(
                k,
                text="",
                extracted=c.nat_extracted,
                correct=bool(c.nat_correct),
                prompt_tokens=c.nat_prompt_tokens,
                output_tokens=c.nat_output_tokens,
            )
        if self.on_stop == "forced" or not st.probes:
            i = st.cur
            return AnswerResult(
                k,
                text="",
                extracted=c.ans_extracted[i],
                correct=bool(c.ans_correct[i]),
                prompt_tokens=int(c.ans_prompt[i]),
                output_tokens=int(c.ans_output[i]),
            )
        # Biến thể ablation (P3.T3): không sinh câu trả lời, chi phí trả lời bằng 0, đáp án chấm bằng grade.
        if self.on_stop == "best_probe":
            i = max(st.probes, key=lambda j: (c.conf_geo[j], j))
            raw = c.probe_raw[i] or ""
        else:  # vote
            weight: dict[str, float] = {}
            best_raw: dict[str, tuple[float, str]] = {}
            for j in st.probes:
                key = c.probe_key[j] or ""
                weight[key] = weight.get(key, 0.0) + float(c.conf_geo[j])
                if key not in best_raw or c.conf_geo[j] > best_raw[key][0]:
                    best_raw[key] = (float(c.conf_geo[j]), c.probe_raw[j] or "")
            win = max(weight, key=lambda kk: (weight[kk], kk))
            raw = best_raw[win][1]
        atype = self.answer_types.get(k.qid, "math")
        return AnswerResult(
            k, text="", extracted=raw, correct=_grade_cached(raw, c.gold, atype), prompt_tokens=0, output_tokens=0
        )
