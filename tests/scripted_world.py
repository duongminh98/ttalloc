"""Một "thế giới" theo kịch bản cho test CPU: chuỗi suy luận, nhánh gợi và câu trả lời đều do một hàm tất định sinh ra
từ token id của prompt, dùng tokenizer thật của Qwen3 (tests/fixtures). Dùng cho test LiveSource, logger và kiểm thử
phát lại (P4.T2): cùng một thế giới được ghi bằng logger rồi chạy lại bằng LoggedSource, và chạy thật bằng LiveSource.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ttalloc.config import ExperimentCfg, load_model_cfg
from ttalloc.data.prompts import build_prompt_ids, load_prompts
from ttalloc.engine.generator import TokenCodec
from ttalloc.engine.live_source import LiveSource
from ttalloc.engine.scripted import ScriptedGenerator
from ttalloc.types import Question

GOLD, WRONG = "42", "17"


def small_cfg(**over: Any) -> ExperimentCfg:
    """Config nhỏ: điểm cách nhau 22–42 token, log và chính sách cùng khoảng cách (điều kiện của P4.T2)."""
    base: dict[str, Any] = {
        "model": "qwen3_1_7b",
        "dp": {"m": 16, "l_max": 48, "b_min": 16},
        "log": {"m": 16, "l_max": 48, "l_log": 600, "window": 3, "flush_every": 2},
        "answer": {"mode": "full", "r_ans": 24},
        "guard": {"min_segment": 48},
        "features": {"local_window": 32},
        "estimator": {"h_train": [64, 128, 256], "n_folds": 2, "max_rounds": 50, "early_stopping": 10},
        "policy": {"horizons": [64, 128], "mu_min": 0.0, "b_max_mult": None},
    }
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return ExperimentCfg.model_validate(base)


@dataclass
class World:
    tok: Any
    codec: TokenCodec
    cfg: ExperimentCfg
    questions: dict[str, Question]
    prompt_ids: dict[str, list[int]]
    chains: dict[str, list[int]]  # chuỗi chính đầy đủ của mỗi câu (gồm </think> nếu tự kết thúc)
    thresholds: dict[str, int]  # vị trí từ đó câu trả lời đúng
    probe_ids: list[int] = field(default_factory=list)
    answer_suffix: list[int] = field(default_factory=list)

    def tokens(self, text: str) -> list[int]:
        return self.codec.encode(text)

    def _which(self, prompt: tuple[int, ...]) -> str:
        for qid, p in self.prompt_ids.items():
            if prompt[: len(p)] == tuple(p):
                return qid
        raise KeyError("prompt không thuộc câu nào")

    def continuation(self, prompt: tuple[int, ...]) -> list[int]:
        qid = self._which(prompt)
        rest = list(prompt[len(self.prompt_ids[qid]) :])
        k = _find(rest, self.probe_ids)
        if k is not None:
            gen = rest[k + len(self.probe_ids) :]
            ans = GOLD if k >= self.thresholds[qid] - 40 else WRONG
            script = self.tokens("{" + ans + "}") + self.tokens("\n") + [self.codec.think_end_id]
            return script[len(gen) :]
        if self.codec.think_end_id in rest:
            i = rest.index(self.codec.think_end_id)
            gen = rest[i + 1 + len(self.answer_suffix) :]
            ans = GOLD if i >= self.thresholds[qid] else WRONG
            script = [*self.tokens("The answer is \\boxed{" + ans + "}."), self.codec.eos_ids[0]]
            return script[len(gen) :]
        return self.chains[qid][len(rest) :]

    def generator(self) -> ScriptedGenerator:
        return ScriptedGenerator(self.continuation, self.codec, _logprob)

    def source(self, gen: ScriptedGenerator | None = None, cfg: ExperimentCfg | None = None) -> LiveSource:
        c = cfg or self.cfg
        return LiveSource(
            gen or self.generator(),
            self.codec,
            self.questions,
            self.prompt_ids,
            c,
            load_model_cfg("qwen3_1_7b"),
            load_prompts(),
            l_log=c.log.l_log,
        )


def _logprob(ctx: tuple[int, ...], token: int) -> float:
    return -0.002 * ((token * 31 + len(ctx) * 7) % 40)


def _find(seq: list[int], sub: list[int]) -> int | None:
    n = len(sub)
    for i in range(len(seq) - n, -1, -1):
        if seq[i : i + n] == sub:
            return i
    return None


def make_world(
    tok: Any, n_questions: int = 4, seed: int = 0, cfg: ExperimentCfg | None = None, max_len_q: int | None = 1
) -> World:
    cfg = cfg or small_cfg()
    rng = np.random.default_rng(seed)
    codec = TokenCodec.from_tokenizer(tok)
    prompts = load_prompts()
    mcfg = load_model_cfg("qwen3_1_7b")
    filler = codec.encode(" step")
    nn = codec.encode("\n\n")
    markers = [codec.encode(m) for m in ("Wait", "Hmm", "Alternatively")]
    qs, pids, chains, th = {}, {}, {}, {}
    for i in range(n_questions):
        q = Question(f"w/{i}", "w", f"Problem number {i}: compute something.", None, GOLD, "math", 1 + i % 5)
        qs[q.qid] = q
        pids[q.qid] = build_prompt_ids(q, mcfg, tok, prompts)
        chain: list[int] = []
        long_q = max_len_q is not None and i == max_len_q
        n_seg = 40 if long_q else int(rng.integers(3, 9))
        for _ in range(n_seg):
            chain += filler * int(rng.integers(20, 41)) + nn + markers[int(rng.integers(0, len(markers)))]
        if not long_q:
            chain += filler * int(rng.integers(5, 30)) + [codec.think_end_id]
        chains[q.qid] = chain
        th[q.qid] = int(rng.integers(0, max(len(chain), 1)))
    return World(
        tok, codec, cfg, qs, pids, chains, th, codec.encode(prompts.probe_math), codec.encode(prompts.answer("full"))
    )
