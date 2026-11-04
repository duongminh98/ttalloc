"""Thông lượng sinh liền mạch theo kích thước lô (PLAN P1.T5, `baseline throughput`)."""

from __future__ import annotations

import time
from typing import Any

from ttalloc.baseline.full import Sampling
from ttalloc.engine.generator import stable_seed
from ttalloc.types import Generator, GenRequest, Question


def measure_throughput(
    generator: Generator,
    questions: list[Question],
    prompt_ids: dict[str, list[int]],
    batch_sizes: list[int],
    sampling: Sampling,
) -> list[dict[str, Any]]:
    rows = []
    # Chạy một lô nhỏ trước để khởi động engine (CUDA graph, cấp phát), không tính vào kết quả.
    warm = [GenRequest(tuple(prompt_ids[questions[0].qid]), max_tokens=32, temperature=0.0)]
    generator.generate(warm)
    for b in batch_sizes:
        qs = [questions[i % len(questions)] for i in range(b)]
        reqs = [
            GenRequest(
                tuple(prompt_ids[q.qid]),
                max_tokens=sampling.max_tokens,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                seed=stable_seed(q.qid, i, "throughput"),
            )
            for i, q in enumerate(qs)
        ]
        t0 = time.time()
        outs = generator.generate(reqs)
        sec = time.time() - t0
        n = sum(len(o.token_ids) for o in outs)
        rows.append(
            {
                "batch_size": b,
                "tokens": n,
                "seconds": round(sec, 2),
                "tok_per_s": n / sec if sec > 0 else None,
                "mean_len": n / b,
                "max_tokens": sampling.max_tokens,
            }
        )
    return rows
