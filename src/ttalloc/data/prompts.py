"""Lời nhắc và dựng prompt ban đầu cho model (PLAN P0.T8, mục 9.2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from ttalloc.config import CONFIGS_DIR, ModelCfg
from ttalloc.types import Question

CHOICE_LETTERS = "ABCDEFGHIJ"


class TokenizerLike(Protocol):
    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...
    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class PromptSet:
    instruction_role: str
    instruction_math: str
    instruction_choice: str
    probe_math: str
    probe_choice: str
    answer_full: str
    answer_short: str
    source_commit: str

    def instruction(self, q: Question) -> str:
        return self.instruction_choice if q.answer_type == "choice" else self.instruction_math

    def probe(self, q: Question) -> str:
        return self.probe_choice if q.answer_type == "choice" else self.probe_math

    def answer(self, mode: str) -> str:
        if mode not in ("full", "short"):
            raise ValueError(f"answer.mode không hợp lệ: {mode}")
        return self.answer_full if mode == "full" else self.answer_short

    def digest(self) -> str:
        """Mã băm của mọi lời nhắc, đưa vào log_hash (I9)."""
        blob = "\x1f".join(
            [
                self.instruction_role,
                self.instruction_math,
                self.instruction_choice,
                self.probe_math,
                self.probe_choice,
                self.answer_full,
                self.answer_short,
            ]
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def load_prompts(name: str = "deer", configs_dir: Path = CONFIGS_DIR) -> PromptSet:
    p = yaml.safe_load((configs_dir / "prompts" / f"{name}.yaml").read_text(encoding="utf-8"))
    a = yaml.safe_load((configs_dir / "prompts" / "answer.yaml").read_text(encoding="utf-8"))
    return PromptSet(
        instruction_role=p["instruction_role"],
        instruction_math=p["instruction_math"],
        instruction_choice=p["instruction_choice"],
        probe_math=p["probe_math"],
        probe_choice=p["probe_choice"],
        answer_full=a["full"],
        answer_short=a["short"],
        source_commit=str(p["source_commit"]),
    )


def shuffle_choices(qid: str, correct: str, incorrect: list[str]) -> tuple[tuple[str, ...], str]:
    """Xáo các lựa chọn bằng seed cố định theo `qid` (giống nhau ở mọi seed sinh); trả về lựa chọn và chữ cái đúng."""
    options = [correct, *incorrect]
    seed = int.from_bytes(hashlib.sha256(qid.encode("utf-8")).digest()[:8], "little")
    order = np.random.default_rng(seed).permutation(len(options))
    shuffled = tuple(options[i] for i in order)
    gold = CHOICE_LETTERS[int(np.flatnonzero(order == 0)[0])]
    return shuffled, gold


def format_problem(q: Question) -> str:
    if q.answer_type != "choice":
        return q.problem
    if not q.options:
        raise ValueError(f"{q.qid}: câu trắc nghiệm không có lựa chọn")
    lines = [q.problem.strip(), ""]
    lines += [f"{CHOICE_LETTERS[i]}) {opt.strip()}" for i, opt in enumerate(q.options)]
    return "\n".join(lines)


def build_messages(q: Question, prompts: PromptSet) -> list[dict[str, str]]:
    body = format_problem(q)
    instr = prompts.instruction(q)
    if prompts.instruction_role == "system":
        return [{"role": "system", "content": instr}, {"role": "user", "content": body}]
    return [{"role": "user", "content": f"{body}\n\n{instr}"}]


def build_prompt_text(q: Question, model_cfg: ModelCfg, tokenizer: TokenizerLike, prompts: PromptSet) -> str:
    messages = build_messages(q, prompts)
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model_cfg.family == "qwen3":
        kwargs["enable_thinking"] = model_cfg.enable_thinking
    text = str(tokenizer.apply_chat_template(messages, **kwargs))
    if model_cfg.family == "r1_distill" and not text.endswith("<think>\n"):
        # Template cũ của R1-Distill không tự thêm "<think>\n"; thêm để model luôn bắt đầu bằng phần suy luận.
        text += "<think>\n"
    return text


def build_prompt_ids(
    q: Question, model_cfg: ModelCfg, tokenizer: TokenizerLike, prompts: PromptSet | None = None
) -> list[int]:
    """Token id của prompt ban đầu. Văn bản template đã chứa token đặc biệt, nên không thêm BOS lần nữa."""
    text = build_prompt_text(q, model_cfg, tokenizer, prompts or load_prompts())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(i) for i in ids]
