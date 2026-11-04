"""Đo thời gian mỗi vòng khi chạy thật (PLAN mục 4.5, `timing.parquet`)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundTimer:
    hit_rate: Callable[[], float | None] | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)

    def __call__(self, info: dict[str, Any]) -> None:
        self.rows.append(
            {
                "round": int(info["round"]),
                "t_advance": float(info["t_advance"]),
                "t_probe": float(info["t_probe"]),
                "t_answer": float(info["t_answer"]),
                "t_controller": float(info["t_controller"]),
                "n_active": int(info["n_active"]),
                "prefix_cache_hit_rate": self.hit_rate() if self.hit_rate else None,
            }
        )

    def totals(self) -> dict[str, float]:
        keys = ("t_advance", "t_probe", "t_answer", "t_controller")
        return {k: float(sum(r[k] for r in self.rows)) for k in keys} | {"rounds": float(len(self.rows))}


def kv_cache_tokens(llm: Any) -> int | None:
    """Số token KV cache của engine vLLM, nếu đọc được (khác nhau giữa các phiên bản)."""
    for path in ("llm_engine.vllm_config.cache_config", "llm_engine.cache_config"):
        obj: Any = llm
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            blocks, size = obj.num_gpu_blocks, obj.block_size
            if blocks:
                return int(blocks) * int(size)
        except AttributeError:
            continue
    return None


def estimate_max_active(kv_tokens: int | None, n: int, budget: int, prompt_len: int, seg: int) -> int:
    """Số câu chạy đồng thời sao cho KV cache đủ cho độ dài trung bình mỗi câu (B/N), tối thiểu một đoạn."""
    if not kv_tokens:
        return n
    per_q = max(budget / max(n, 1), seg) + prompt_len
    return max(1, min(n, int(0.9 * kv_tokens / per_q)))
