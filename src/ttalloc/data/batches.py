"""Thành phần lô (PLAN P2.T14).

Mô phỏng dựng lại lô ở mỗi lần rút bằng seed của lần rút; chạy thật dựng lô một lần bằng `batch.seed`.
"""

from __future__ import annotations

import numpy as np

from ttalloc.config import BatchCfg
from ttalloc.types import Question

LEVEL_SHARES: dict[str, dict[int, float]] = {
    "balanced": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
    "easy": {1: 0.35, 2: 0.35, 3: 0.1, 4: 0.1, 5: 0.1},
    "hard": {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.35, 5: 0.35},
    "u": {1: 0.35, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.35},
}
DEFAULT_N = 100


def level_counts(composition: str, n: int) -> dict[int, int]:
    """Số câu mỗi mức; làm tròn theo phần dư lớn nhất để tổng đúng bằng n."""
    shares = LEVEL_SHARES[composition]
    raw = {lv: n * p for lv, p in shares.items()}
    out = {lv: int(np.floor(v)) for lv, v in raw.items()}
    rest = n - sum(out.values())
    for lv in sorted(raw, key=lambda lv: (-(raw[lv] - out[lv]), lv))[:rest]:
        out[lv] += 1
    return out


def build_batch(questions: list[Question], spec: BatchCfg, rng: np.random.Generator) -> list[str]:
    if spec.composition == "all":
        n = len(questions) if spec.n is None else spec.n
        if n > len(questions):
            raise ValueError(f"lô all: cần {n} câu, tập chỉ có {len(questions)}")
        if n == len(questions):
            return [q.qid for q in questions]
        idx = rng.choice(len(questions), size=n, replace=False)
        return [questions[i].qid for i in idx]
    if spec.composition == "mix":
        if not spec.sources:
            raise ValueError("lô mix cần `sources`, ví dụ {aime_2024: 30, math500: 100}")
        out: list[str] = []
        for ds, k in spec.sources.items():
            pool = [q for q in questions if q.dataset == ds]
            if len(pool) < k:
                raise ValueError(f"lô mix: {ds} có {len(pool)} câu, cần {k}")
            out += [pool[i].qid for i in rng.choice(len(pool), size=k, replace=False)]
        return out
    counts = level_counts(spec.composition, spec.n or DEFAULT_N)
    out = []
    for lv, k in counts.items():
        pool = [q for q in questions if q.difficulty == lv]
        if len(pool) < k:
            raise ValueError(f"lô {spec.composition}: mức {lv} có {len(pool)} câu, cần {k}")
        out += [pool[i].qid for i in rng.choice(len(pool), size=k, replace=False)]
    return out
