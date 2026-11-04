"""Giao diện Generator và các hàm thuần dùng chung cho mọi generator (PLAN mục 3.3)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from ttalloc.types import GenRequest


def to_sampling_kwargs(req: GenRequest) -> dict[str, Any]:
    """Tham số `SamplingParams` của vLLM cho một request.

    Không bao giờ đặt `min_tokens`: tham số này chặn cả EOS và mọi `stop_token_ids` (gồm `</think>`) cho tới khi đủ số
    token, làm thay đổi phân phối của chuỗi. Khoảng cách tối thiểu giữa hai điểm do LiveSource tự áp dụng (A1a).
    """
    if req.max_tokens <= 0:
        raise ValueError(f"max_tokens phải > 0, nhận {req.max_tokens}")
    kw: dict[str, Any] = {
        "max_tokens": int(req.max_tokens),
        "temperature": float(req.temperature),
        "top_p": float(req.top_p),
        "top_k": int(req.top_k),
        "stop": list(req.stop),
        "stop_token_ids": list(req.stop_token_ids),
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    if req.logprobs is not None:
        kw["logprobs"] = int(req.logprobs)
    if req.seed is not None:
        kw["seed"] = int(req.seed)
    return kw


def stable_seed(*parts: object) -> int:
    """Seed 31 bit ổn định giữa các tiến trình (không dùng `hash()` của Python)."""
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little") & 0x7FFFFFFF


class TokenizerLike(Protocol):
    def decode(self, ids: Any, **kwargs: Any) -> str: ...
    def __call__(self, text: str, **kwargs: Any) -> Any: ...
    def convert_tokens_to_ids(self, tokens: Any) -> Any: ...


@dataclass
class TokenCodec:
    """Bọc tokenizer với các cờ giải mã cố định (A1a: `skip_special_tokens=False`, không dọn khoảng trắng)."""

    tok: Any
    think_end_id: int
    eos_ids: tuple[int, ...]

    @classmethod
    def from_tokenizer(cls, tok: Any, think_end: str = "</think>") -> TokenCodec:
        tid = tok.convert_tokens_to_ids(think_end)
        if tid is None or tid == getattr(tok, "unk_token_id", None):
            raise ValueError(f"tokenizer không có token {think_end!r}")
        eos = getattr(tok, "eos_token_id", None)
        eos_ids = tuple(e for e in ([eos] if isinstance(eos, int) else list(eos or [])) if e is not None)
        return cls(tok=tok, think_end_id=int(tid), eos_ids=eos_ids)

    def decode(self, ids: Any) -> str:
        return str(self.tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False))

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self.tok(text, add_special_tokens=False)["input_ids"]]
