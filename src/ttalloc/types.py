"""Kiểu dữ liệu dùng chung (PLAN mục 3.2–3.4, 4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from ttalloc.features.extract import FeatureState

# ---------------------------------------------------------------- câu hỏi (mục 4.1)


@dataclass(frozen=True)
class Question:
    qid: str  # "<dataset>/<id gốc hoặc chỉ số>"
    dataset: str
    problem: str
    options: tuple[str, ...] | None
    gold: str
    answer_type: Literal["math", "choice"]
    difficulty: int | None  # mức 1–5 của MATH, nếu có


# ---------------------------------------------------------------- nguồn đoạn suy luận (mục 3.2)


@dataclass(frozen=True)
class TraceKey:
    qid: str
    seed: int


SegmentKind = Literal["marker", "length", "natural_end", "eos", "truncated"]


@dataclass(frozen=True)
class AdvanceRequest:
    key: TraceKey
    min_tokens: int  # chỉ chấp nhận từ chuyển ý khi số token đã sinh trong đoạn > min_tokens (A1a)
    max_tokens: int  # hạn mức token chuỗi chính của đoạn
    stop_at_markers: bool  # False: chỉ dừng khi tự kết thúc hoặc hết hạn mức


@dataclass(frozen=True)
class SegmentResult:
    key: TraceKey
    kind: SegmentKind
    pos: int
    gen_end: int
    n_new_tokens: int
    marker: str | None = None
    # Thông tin của điểm mới (chỉ có với marker và length), dùng cho đặc trưng (A5).
    point_idx: int | None = None  # chỉ số điểm đã ghi (LoggedSource); LiveSource đánh số theo thứ tự ghé
    n_markers: int = 0
    local_conf_min: float | None = None


@dataclass(frozen=True)
class ProbeResult:
    key: TraceKey
    answer_raw: str
    answer_key: str
    conf_geo: float
    conf_arith: float
    logprobs: tuple[float, ...]  # logprob của các token tính độ tự tin (phần trong khung, A1b)
    prompt_tokens: int
    output_tokens: int
    has_think_end: bool = False
    output_ids: tuple[int, ...] = ()  # mọi token đã sinh ở nhánh gợi
    gen_end: int = 0  # gen_end của điểm được gợi

    @property
    def answer_tokens(self) -> int:
        return len(self.logprobs)


@dataclass(frozen=True)
class AnswerResult:
    key: TraceKey
    text: str
    extracted: str | None
    correct: bool | None  # None: Runner chấm
    prompt_tokens: int
    output_tokens: int


class SegmentSource(Protocol):
    def start(self, keys: list[TraceKey]) -> None: ...
    def advance(self, reqs: list[AdvanceRequest]) -> list[SegmentResult]: ...
    def probe(self, keys: list[TraceKey]) -> list[ProbeResult]: ...
    def answer(self, keys: list[TraceKey]) -> list[AnswerResult]: ...


# ---------------------------------------------------------------- generator (mục 3.3)


@dataclass(frozen=True)
class GenRequest:
    prompt_ids: tuple[int, ...]
    max_tokens: int
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    logprobs: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class GenOutput:
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None  # logprob của token đã sinh
    finish_reason: Literal["stop", "length"]
    stop_reason: str | int | None  # chuỗi dừng, token dừng, hoặc None nếu gặp EOS


class Generator(Protocol):
    def generate(self, reqs: list[GenRequest]) -> list[GenOutput]: ...


# ---------------------------------------------------------------- chi phí (mục 2.6)


@dataclass
class Costs:
    think: int = 0
    probe_prompt: int = 0
    probe_output: int = 0
    answer_prompt: int = 0
    answer_output: int = 0

    @property
    def total(self) -> int:
        return self.think + self.probe_prompt + self.probe_output + self.answer_prompt + self.answer_output

    def add(self, other: Costs) -> None:
        self.think += other.think
        self.probe_prompt += other.probe_prompt
        self.probe_output += other.probe_output
        self.answer_prompt += other.answer_prompt
        self.answer_output += other.answer_output

    def as_dict(self) -> dict[str, int]:
        return {
            "think": self.think,
            "probe_prompt": self.probe_prompt,
            "probe_output": self.probe_output,
            "answer_prompt": self.answer_prompt,
            "answer_output": self.answer_output,
        }


# ---------------------------------------------------------------- trạng thái, chính sách (mục 3.4)


@dataclass
class QuestionState:
    key: TraceKey
    gen_end: int = 0  # b_i
    pos: int = 0
    point_kind: Literal["start", "marker", "length"] = "start"
    visited: list[int] = field(default_factory=list)  # thứ tự điểm đã ghé
    costs: Costs = field(default_factory=Costs)  # theo năm loại ở mục 2.6
    features: FeatureState = field(default_factory=FeatureState)
    qhat: np.ndarray | None = None
    rho: float | None = None
    hstar: int | None = None
    n_rounds: int = 0

    def apply(self, r: SegmentResult) -> None:
        """Cập nhật vị trí sau một đoạn; chi phí think do Runner trừ vào quỹ và vào `costs`."""
        self.gen_end = r.gen_end
        self.pos = r.pos
        self.costs.think += r.n_new_tokens
        self.n_rounds += 1
        if r.kind in ("marker", "length"):
            self.point_kind = r.kind
            self.visited.append(r.point_idx if r.point_idx is not None else len(self.visited) + 1)
        self.qhat, self.rho, self.hstar = None, None, None


@dataclass(frozen=True)
class Decision:
    action: Literal["continue", "stop"]
    min_tokens: int = 0
    max_tokens: int = 0  # hạn mức mong muốn; guard có thể giảm


class PoolView(Protocol):
    def available(self) -> int: ...


class Policy(Protocol):
    uses_probes: bool  # B0: False
    uses_rho: bool
    stop_at_markers: bool  # B0: False

    def initial_request(self, s: QuestionState, pool: PoolView) -> Decision: ...
    def decide(self, active: list[QuestionState], pool: PoolView) -> dict[TraceKey, Decision]: ...


class RhoProvider(Protocol):
    def fill(self, states: list[QuestionState]) -> None: ...  # ghi qhat, rho, hstar vào state
