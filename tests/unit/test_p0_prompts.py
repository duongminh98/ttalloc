"""P0.T8: lời nhắc."""

from __future__ import annotations

from typing import Any

from ttalloc.config import load_model_cfg
from ttalloc.data.prompts import build_prompt_ids, build_prompt_text, load_prompts, shuffle_choices
from ttalloc.types import Question

MATH_Q = Question("aime_2025/1", "aime_2025", "Find 1+1.", None, "2", "math", None)


def test_r1_prompt_ends_with_think(r1_tokenizer: Any) -> None:
    cfg = load_model_cfg("r1_distill_qwen_1_5b")
    text = build_prompt_text(MATH_Q, cfg, r1_tokenizer, load_prompts())
    assert text.endswith("<think>\n")
    ids = build_prompt_ids(MATH_Q, cfg, r1_tokenizer)
    assert r1_tokenizer.decode(ids).endswith("<think>\n")
    assert ids.count(r1_tokenizer.bos_token_id) == 1  # không lặp BOS


def test_qwen3_prompt_has_thinking(qwen3_tokenizer: Any) -> None:
    cfg = load_model_cfg("qwen3_1_7b")
    prompts = load_prompts()
    text = build_prompt_text(MATH_Q, cfg, qwen3_tokenizer, prompts)
    assert text.endswith("<|im_start|>assistant\n")
    assert "</think>" not in text  # enable_thinking=False sẽ chèn sẵn "<think>\n\n</think>"
    assert f"<|im_start|>system\n{prompts.instruction_math}<|im_end|>" in text
    assert "<|im_start|>user\nFind 1+1.<|im_end|>" in text
    ids = build_prompt_ids(MATH_Q, cfg, qwen3_tokenizer)
    assert qwen3_tokenizer.decode(ids) == text


def test_choice_prompt_lists_options(qwen3_tokenizer: Any) -> None:
    options, gold = shuffle_choices("gpqa_diamond/rec1", "right", ["w1", "w2", "w3"])
    q = Question("gpqa_diamond/rec1", "gpqa_diamond", "Which one?", options, gold, "choice", None)
    prompts = load_prompts()
    text = build_prompt_text(q, load_model_cfg("qwen3_1_7b"), qwen3_tokenizer, prompts)
    for letter, opt in zip("ABCD", options, strict=True):
        assert f"{letter}) {opt}" in text
    assert prompts.instruction_choice in text and "letter" in prompts.instruction_choice


def test_choice_shuffle_deterministic_and_gold_remapped() -> None:
    a = shuffle_choices("gpqa_diamond/rec1", "right", ["w1", "w2", "w3"])
    b = shuffle_choices("gpqa_diamond/rec1", "right", ["w1", "w2", "w3"])
    assert a == b
    options, gold = a
    assert sorted(options) == ["right", "w1", "w2", "w3"]
    assert options["ABCD".index(gold)] == "right"
    golds = {shuffle_choices(f"gpqa_diamond/rec{i}", "right", ["w1", "w2", "w3"])[1] for i in range(40)}
    assert len(golds) > 1  # đáp án đúng không luôn ở cùng vị trí


def test_prompt_digest_changes_with_content() -> None:
    p = load_prompts()
    assert len(p.digest()) == 12
    assert p.source_commit.startswith("c9dd19f")
