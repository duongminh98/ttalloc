"""Đặc trưng tại điểm quyết định (PLAN A5; báo cáo phương pháp mục 4.3).

Một lớp `FeatureState` duy nhất dùng khi huấn luyện, mô phỏng và chạy thật. Module này không import `ttalloc.types`
(để `types.QuestionState` dùng được lớp này); lần gợi được truyền vào dưới dạng bất kỳ đối tượng nào có các trường
của `ProbeLike`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Protocol

ANSWER_TYPES = ("integer", "fraction", "expression", "choice", "other")

# Nhóm đặc trưng cho ablation (P3.T3, khóa features.drop_groups).
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "consistency": ("consistency", "n_recent", "n_changes"),
    "trend": ("conf_slope_per_1k", "has_slope"),
    "local_conf": ("local_conf_min",),
    "ans_type": ("ans_len", *(f"ans_type_{t}" for t in ANSWER_TYPES)),
}

ALL_FEATURES: tuple[str, ...] = (
    "conf_cur",
    "conf_max",
    "conf_slope_per_1k",
    "has_slope",
    "consistency",
    "n_recent",
    "n_changes",
    "n_probes",
    "b",
    "marker_rate_per_1k",
    "local_conf_min",
    "ans_len",
    *(f"ans_type_{t}" for t in ANSWER_TYPES),
)


def feature_names(drop_groups: list[str] | tuple[str, ...] = ()) -> list[str]:
    unknown = set(drop_groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"features.drop_groups có nhóm lạ: {sorted(unknown)}; hợp lệ: {sorted(FEATURE_GROUPS)}")
    dropped = {f for g in drop_groups for f in FEATURE_GROUPS[g]}
    return [f for f in ALL_FEATURES if f not in dropped]


class ProbeLike(Protocol):
    @property
    def answer_raw(self) -> str: ...
    @property
    def answer_key(self) -> str: ...
    @property
    def conf_geo(self) -> float: ...
    @property
    def answer_tokens(self) -> int: ...


@dataclass(frozen=True)
class ProbeObs:
    gen_end: int
    conf: float  # conf_geo
    conf_arith: float
    key: str
    raw: str
    answer_tokens: int
    answer_type: str
    has_think_end: bool


def same_answer(a: ProbeObs, b: ProbeObs) -> bool:
    """Hai đáp án thử trùng nhau khi khóa bằng nhau, hoặc cả hai là biểu thức và `equivalent` trả về True (A5).

    Chỉ gọi `equivalent` khi cả hai khóa là biểu thức đã phân tích được: khóa dạng chuỗi khác nhau gần như không bao giờ
    tương đương, và bỏ qua chúng tránh gọi math_verify trên đáp án rác.
    """
    if a.key == b.key:
        return True
    if a.key.startswith("expr:") and b.key.startswith("expr:"):
        from ttalloc.data.grading import equivalent

        return equivalent(a.raw, b.raw)
    return False


def ols_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


@dataclass
class FeatureState:
    k: int = 4
    probes: list[ProbeObs] = field(default_factory=list)
    gen_end: int = 0
    n_markers: int = 0
    local_conf_min: float = math.nan
    n_changes: int = 0

    def update(
        self,
        gen_end: int,
        probe: ProbeLike,
        n_markers: int,
        local_conf_min: float | None,
        has_think_end: bool | None = None,
    ) -> None:
        from ttalloc.data.grading import classify_answer

        obs = ProbeObs(
            gen_end=gen_end,
            conf=float(probe.conf_geo),
            conf_arith=float(getattr(probe, "conf_arith", probe.conf_geo)),
            key=probe.answer_key,
            raw=probe.answer_raw,
            answer_tokens=int(probe.answer_tokens),
            answer_type=classify_answer(probe.answer_raw),
            has_think_end=bool(getattr(probe, "has_think_end", False) if has_think_end is None else has_think_end),
        )
        if self.probes and not same_answer(self.probes[-1], obs):
            self.n_changes += 1
        self.probes.append(obs)
        self.gen_end = gen_end
        self.n_markers = n_markers
        self.local_conf_min = math.nan if local_conf_min is None else float(local_conf_min)

    @property
    def last(self) -> ProbeObs | None:
        return self.probes[-1] if self.probes else None

    def vector(self) -> dict[str, float]:
        if not self.probes:
            return {name: math.nan for name in ALL_FEATURES}
        c = [p.conf for p in self.probes]
        recent = self.probes[-self.k :]
        cur = recent[-1]
        out = {
            "conf_cur": c[-1],
            "conf_max": max(c),
            "conf_slope_per_1k": (
                ols_slope([p.gen_end / 1000 for p in recent], [p.conf for p in recent]) if len(recent) >= 2 else 0.0
            ),
            "has_slope": float(len(recent) >= 2),
            "consistency": sum(same_answer(p, cur) for p in recent) / len(recent),
            "n_recent": float(len(recent)),
            "n_changes": float(self.n_changes),
            "n_probes": float(len(c)),
            "b": float(cur.gen_end),
            "marker_rate_per_1k": 1000.0 * self.n_markers / max(self.gen_end, 1),
            "local_conf_min": self.local_conf_min,
            "ans_len": float(cur.answer_tokens),
        }
        for t in ANSWER_TYPES:
            out[f"ans_type_{t}"] = float(cur.answer_type == t)
        return out


def count_changes(probes: list[ProbeObs]) -> int:
    """Số lần đổi đáp án qua dãy lần gợi (dùng để kiểm tra `FeatureState.n_changes`)."""
    return sum(not same_answer(a, b) for a, b in pairwise(probes))
