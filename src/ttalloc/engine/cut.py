"""Xác định điểm sau một đoạn sinh (PLAN A1a, P1.T2). Mọi vị trí tính trên token id của chuỗi chính."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ttalloc.engine.generator import TokenCodec
from ttalloc.types import GenOutput

CutKind = Literal["natural_end", "eos", "marker", "length", "resume", "peek"]

# Số token tối đa tìm ngược từ cuối đoạn để tìm ranh giới "\n\n" + từ chuyển ý (A1a).
LOOKBACK = 8
HEAD_TOKENS = 4


def marker_regex(markers: list[str] | tuple[str, ...]) -> re.Pattern[str]:
    alts = "|".join(re.escape(m) for m in markers)
    return re.compile(rf"\s*({alts})\b")


def marker_count_regex(markers: list[str] | tuple[str, ...]) -> re.Pattern[str]:
    # Từ chuyển ý ở đầu đoạn. Dùng lookahead thay cho \b để một từ ở cuối văn bản (chưa biết ký tự sau) chưa được đếm;
    # nhờ vậy đếm tăng dần theo cửa sổ cho cùng kết quả với đếm trên toàn văn bản.
    alts = "|".join(re.escape(m) for m in markers)
    return re.compile(rf"\n\n\s*(?:{alts})(?=\W)")


def stop_strings(markers: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple("\n\n" + w for w in markers)


def count_markers(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


@dataclass(frozen=True)
class CutEvent:
    kind: CutKind
    pos: int = 0
    gen_end: int = 0
    marker: str | None = None


def newline_cutback(full: list[int], start: int, gen_end: int, codec: TokenCodec) -> int:
    """Lùi vị trí cắt về sau ký tự xuống dòng gần nhất, chỉ trong 25% cuối đoạn; không có thì cắt ở gen_end."""
    lo = gen_end - (gen_end - start) // 4
    for t in range(gen_end, lo, -1):
        if codec.decode(full[t - 1 : t]).endswith("\n"):
            return t
    return gen_end


def classify(
    chain_ids: list[int],
    start: int,
    seg_start: int,
    out: GenOutput,
    codec: TokenCodec,
    min_tokens: int,
    marker_re: re.Pattern[str],
    final: bool = True,
) -> CutEvent:
    """Phân loại kết quả một lần gửi (A1a).

    `chain_ids` là chuỗi chính trước lần gửi này (độ dài `seg_start`, không gồm prompt); `start` là gen_end ở đầu đoạn,
    `seg_start >= start` khi đã gửi lại. Trả về `resume` nếu phải sinh tiếp.

    Khi `final=False` và từ chuyển ý nằm sát cuối văn bản (chưa biết ký tự sau, ví dụ "\\n\\nBut" có thể là
    "\\n\\nButter"), trả về `peek`: LiveSource sinh thêm một token rồi phân loại lại với `final=True` (ADR-010).
    """
    seg = list(out.token_ids)
    gen_end = seg_start + len(seg)
    full = chain_ids[:seg_start] + seg
    if codec.think_end_id in seg:  # tự kết thúc
        return CutEvent("natural_end", pos=seg_start + seg.index(codec.think_end_id) + 1, gen_end=gen_end)
    if out.finish_reason == "stop" and out.stop_reason is None:  # EOS mà không có </think>
        return CutEvent("eos", pos=gen_end - 1, gen_end=gen_end)
    if out.finish_reason == "stop" and isinstance(out.stop_reason, str):
        for t in range(gen_end - 1, max(seg_start - 1, gen_end - LOOKBACK - 1), -1):
            head = codec.decode(full[max(0, t - HEAD_TOKENS) : t])
            rest = codec.decode(full[t:gen_end])
            m = marker_re.match(rest)
            if head.endswith("\n\n") and m:
                if not final and m.end(1) == len(rest):
                    return CutEvent("peek", gen_end=gen_end)
                if gen_end - start <= min_tokens:
                    return CutEvent("resume", gen_end=gen_end)  # từ chuyển ý đến quá sớm
                return CutEvent("marker", pos=t, gen_end=gen_end, marker=m.group(1))
        return CutEvent("resume", gen_end=gen_end)  # khớp chuỗi dừng nhưng không phải từ chuyển ý, ví dụ "Butter"
    return CutEvent("length", pos=newline_cutback(full, start, gen_end, codec), gen_end=gen_end)
