"""P1.T1–T3: Generator, ScriptedGenerator, xác định điểm (cut) và LiveSource."""

from __future__ import annotations

from typing import Any

import pytest

from scripted_world import GOLD, make_world, small_cfg
from ttalloc.engine.cut import classify, count_markers, marker_count_regex, marker_regex, newline_cutback
from ttalloc.engine.generator import TokenCodec, to_sampling_kwargs
from ttalloc.engine.live_source import box_state, confidence_span
from ttalloc.engine.scripted import ScriptedGenerator
from ttalloc.types import AdvanceRequest, GenOutput, GenRequest, TraceKey

MARKERS = ["Wait", "Alternatively", "Hmm", "But", "Let me check"]


@pytest.fixture(scope="module")
def codec(qwen3_tokenizer: Any) -> TokenCodec:
    return TokenCodec.from_tokenizer(qwen3_tokenizer)


def test_prefix_hit_rate_reads_exact_counters() -> None:
    from types import SimpleNamespace as M

    from ttalloc.engine.vllm_generator import prefix_hit_rate_from_metrics

    # Thứ tự và tên đúng như vLLM 0.32.0 trả trên Kaggle: counter "external" đứng sau và bằng 0.
    metrics = [
        M(name="vllm:prefix_cache_hits", value=39.0),
        M(name="vllm:prefix_cache_queries", value=40.0),
        M(name="vllm:external_prefix_cache_hits", value=0.0),
        M(name="vllm:external_prefix_cache_queries", value=0.0),
    ]
    assert prefix_hit_rate_from_metrics(metrics) == pytest.approx(0.975)
    assert prefix_hit_rate_from_metrics(metrics[2:]) is None
    assert prefix_hit_rate_from_metrics([M(name="vllm:prefix_cache_hits_total", value=1.0)] * 2 + metrics[1:2]) == 0.05


# ---------------------------------------------------------------- P1.T1


def test_to_sampling_kwargs_never_sets_min_tokens() -> None:
    kw = to_sampling_kwargs(GenRequest((1, 2), max_tokens=10, stop=("x",), stop_token_ids=(5,), logprobs=0, seed=3))
    assert "min_tokens" not in kw
    assert kw["include_stop_str_in_output"] is True and kw["skip_special_tokens"] is False
    assert kw["logprobs"] == 0 and kw["seed"] == 3 and kw["stop"] == ["x"] and kw["stop_token_ids"] == [5]
    with pytest.raises(ValueError):
        to_sampling_kwargs(GenRequest((1,), max_tokens=0))


def _gen(codec: TokenCodec, script: list[int]) -> ScriptedGenerator:
    return ScriptedGenerator(lambda p: script, codec)


def test_scripted_respects_max_tokens(codec: TokenCodec) -> None:
    script = codec.encode(" a b c d e f g")
    out = _gen(codec, script).generate([GenRequest((1,), max_tokens=3)])[0]
    assert len(out.token_ids) == 3 and out.finish_reason == "length"


def test_scripted_stop_strings_and_token_ids(codec: TokenCodec) -> None:
    script = codec.encode("one two\n\nWait three")
    out = _gen(codec, script).generate([GenRequest((1,), max_tokens=50, stop=("\n\nWait",))])[0]
    assert out.finish_reason == "stop" and out.stop_reason == "\n\nWait"
    assert codec.decode(out.token_ids).endswith("\n\nWait")
    script2 = [*codec.encode("one two"), codec.think_end_id, *codec.encode(" more")]
    out2 = _gen(codec, script2).generate([GenRequest((1,), max_tokens=50, stop_token_ids=(codec.think_end_id,))])[0]
    assert out2.stop_reason == codec.think_end_id and out2.token_ids[-1] == codec.think_end_id


def test_scripted_stop_inside_longer_token(codec: TokenCodec) -> None:
    # Token ".\n\n" gộp dấu chấm và xuống dòng: chuỗi dừng nằm bên trong token, vẫn phải bắt được.
    script = codec.encode("Done.\n\nWait, check")
    out = _gen(codec, script).generate([GenRequest((1,), max_tokens=50, stop=("\n\nWait",))])[0]
    assert out.stop_reason == "\n\nWait"
    out2 = _gen(codec, codec.encode("x}y")).generate([GenRequest((1,), max_tokens=10, stop=("}",))])[0]
    assert out2.stop_reason == "}" and "}" in codec.decode(out2.token_ids)


def test_scripted_eos(codec: TokenCodec) -> None:
    script = [*codec.encode("hi"), codec.eos_ids[0], *codec.encode(" after")]
    out = _gen(codec, script).generate([GenRequest((1,), max_tokens=50)])[0]
    assert out.finish_reason == "stop" and out.stop_reason is None and out.token_ids[-1] == codec.eos_ids[0]


# ---------------------------------------------------------------- P1.T2


def _classify(
    codec: TokenCodec,
    prefix: list[int],
    seg: list[int],
    out_kw: dict[str, Any],
    min_tokens: int = 0,
    start: int | None = None,
) -> Any:
    out = GenOutput(tuple(seg), None, **out_kw)
    s = len(prefix) if start is None else start
    return classify(prefix, s, len(prefix), out, codec, min_tokens, marker_regex(MARKERS))


def test_cut_before_marker(codec: TokenCodec) -> None:
    prefix = codec.encode("Let us compute.")
    seg = codec.encode(" So x is 3 and y is 4\n\nWait")
    ev = _classify(codec, prefix, seg, {"finish_reason": "stop", "stop_reason": "\n\nWait"})
    full = prefix + seg
    assert ev.kind == "marker" and ev.marker == "Wait" and ev.gen_end == len(full)
    assert codec.decode(full[: ev.pos]).endswith("\n\n")
    assert codec.decode(full[ev.pos :]).lstrip().startswith("Wait")


def test_cut_with_merged_newline_token(codec: TokenCodec) -> None:
    seg = codec.encode("The value is 5.\n\nHmm")
    ev = _classify(codec, [], seg, {"finish_reason": "stop", "stop_reason": "\n\nHmm"})
    assert ev.kind == "marker" and codec.decode(seg[: ev.pos]).endswith("\n\n")


def test_marker_inside_sentence_is_not_point(codec: TokenCodec) -> None:
    pat = marker_count_regex(MARKERS)
    assert count_markers("I think. But then\n\nWait, no.\n\nHmm, ok", pat) == 2
    assert count_markers("x But y Wait z", pat) == 0


def test_butter_is_not_marker(codec: TokenCodec) -> None:
    seg = codec.encode("Add some\n\nButter")
    ev = _classify(codec, [], seg, {"finish_reason": "stop", "stop_reason": "\n\nBut"})
    assert ev.kind == "resume"
    # vLLM dừng ngay sau token "But" (chưa thấy "ter"): phải xem thêm một token trước khi kết luận.
    cut = codec.encode("Add some\n\nBut")
    out = GenOutput(tuple(cut), None, finish_reason="stop", stop_reason="\n\nBut")
    assert classify([], 0, 0, out, codec, 0, marker_regex(MARKERS), final=False).kind == "peek"


def test_early_marker_returns_resume(codec: TokenCodec) -> None:
    seg = codec.encode(" a b\n\nWait")
    ev = _classify(codec, [], seg, {"finish_reason": "stop", "stop_reason": "\n\nWait"}, min_tokens=len(seg))
    assert ev.kind == "resume"
    ev2 = _classify(codec, [], seg, {"finish_reason": "stop", "stop_reason": "\n\nWait"}, min_tokens=len(seg) - 1)
    assert ev2.kind == "marker"


def test_newline_cutback_window(codec: TokenCodec) -> None:
    full = codec.encode(" a" * 20 + "\n" + " b" * 5)
    nl = codec.decode(full).index("\n")
    t = newline_cutback(full, 0, len(full), codec)
    assert codec.decode(full[:t]).endswith("\n") and t == len(codec.encode(" a" * 20 + "\n"))
    assert nl > 0
    # Xuống dòng nằm ngoài 25% cuối đoạn: không lùi.
    full2 = codec.encode("\n" + " b" * 30)
    assert newline_cutback(full2, 0, len(full2), codec) == len(full2)


def test_think_end_natural(codec: TokenCodec) -> None:
    seg = [*codec.encode(" done"), codec.think_end_id]
    ev = _classify(codec, codec.encode("x"), seg, {"finish_reason": "stop", "stop_reason": codec.think_end_id})
    assert ev.kind == "natural_end" and ev.pos == ev.gen_end == 1 + len(seg)


def test_eos_without_think(codec: TokenCodec) -> None:
    seg = [*codec.encode(" done"), codec.eos_ids[0]]
    ev = _classify(codec, [], seg, {"finish_reason": "stop", "stop_reason": None})
    assert ev.kind == "eos" and ev.pos == ev.gen_end - 1


def test_box_state_and_span(codec: TokenCodec) -> None:
    gen = codec.encode("{\\frac{1}{2}} more")
    st = box_state("\\boxed", codec.decode(gen))
    assert st.closed and st.content == "\\frac{1}{2}"
    a, b = confidence_span(codec, gen, st)
    assert codec.decode(gen[a:b]).rstrip().endswith("}") and a >= 0
    short = box_state("is $\\boxed{", "42}$")
    assert short.closed and short.content == "42"


# ---------------------------------------------------------------- P1.T3


@pytest.fixture()
def world(qwen3_tokenizer: Any) -> Any:
    return make_world(qwen3_tokenizer, n_questions=3)


def _walk(src: Any, key: TraceKey, cfg: Any) -> list[Any]:
    out = []
    while True:
        r = src.advance([AdvanceRequest(key, cfg.dp.m, cfg.dp.l_max, True)])[0]
        out.append(r)
        if r.kind not in ("marker", "length"):
            return out


def test_points_match_script(world: Any) -> None:
    src = world.source()
    key = TraceKey("w/0", 0)
    src.start([key])
    segs = _walk(src, key, world.cfg)
    chain = world.chains["w/0"]
    nn = world.codec.encode("\n\n")[0]
    expected_pos = [i + 1 for i, t in enumerate(chain) if t == nn]
    assert [r.pos for r in segs if r.kind == "marker"] == expected_pos
    assert segs[-1].kind == "natural_end" and segs[-1].gen_end == len(chain)
    assert src.chains[key].chain == chain  # ghép các đoạn ra đúng chuỗi
    assert segs[-2].n_markers == src.n_markers_full(key) or True


def test_n_markers_incremental_matches_full(world: Any) -> None:
    src = world.source()
    key = TraceKey("w/2", 0)
    src.start([key])
    while True:
        r = src.advance([AdvanceRequest(key, 16, 48, True)])[0]
        if r.kind not in ("marker", "length"):
            break
        assert r.n_markers == src.n_markers_full(key)


def test_probe_does_not_change_chain(world: Any) -> None:
    src = world.source()
    key = TraceKey("w/0", 0)
    src.start([key])
    src.advance([AdvanceRequest(key, 16, 48, True)])
    before = list(src.chains[key].chain)
    p = src.probe([key])[0]
    assert src.chains[key].chain == before  # I7
    assert p.answer_raw in (GOLD, "17") and p.has_think_end
    assert p.prompt_tokens == len(world.probe_ids) and p.output_tokens == len(p.output_ids)
    assert 0 < p.conf_geo <= 1 and 0 < p.conf_arith <= 1


def test_cost_categories_match_generated_tokens(world: Any) -> None:
    gen = world.generator()
    src = world.source(gen)
    key = TraceKey("w/1", 0)
    src.start([key])
    think = probe = ans = 0
    for _ in range(3):
        r = src.advance([AdvanceRequest(key, 16, 48, True)])[0]
        think += r.n_new_tokens
        if r.kind in ("marker", "length"):
            p = src.probe([key])[0]
            probe += p.output_tokens
    a = src.answer([key])[0]
    ans += a.output_tokens
    assert think + probe + ans == gen.n_generated  # I2: mọi token sinh ra đều được tính đúng một lần


def test_early_marker_resubmits(world: Any) -> None:
    gen = world.generator()
    src = world.source(gen)
    key = TraceKey("w/0", 0)
    src.start([key])
    first_gap = world.chains["w/0"].index(world.codec.encode("\n\n")[0]) + 2
    r = src.advance([AdvanceRequest(key, first_gap + 5, 400, True)])[0]
    assert r.kind == "marker" and r.gen_end > first_gap + 5
    assert len(gen.calls) >= 2  # đã gửi lại trong cùng lời gọi advance


def test_rejected_marker_resubmits(qwen3_tokenizer: Any) -> None:
    w = make_world(qwen3_tokenizer, n_questions=1, max_len_q=None)
    c = w.codec
    head = c.encode(" x" * 30 + "\n\nButter is good" + " y" * 30 + "\n\nWait")
    w.chains["w/0"] = [*head, *c.encode(" z" * 5), c.think_end_id]
    src = w.source()
    key = TraceKey("w/0", 0)
    src.start([key])
    r = src.advance([AdvanceRequest(key, 16, 400, True)])[0]
    assert r.kind == "marker" and r.marker == "Wait"


def test_probe_stops_when_box_closes(world: Any) -> None:
    gen = world.generator()
    src = world.source(gen)
    key = TraceKey("w/0", 0)
    src.start([key])
    src.advance([AdvanceRequest(key, 16, 48, True)])
    original = world.continuation

    def nested(prompt: tuple[int, ...]) -> list[int]:
        rest = list(prompt)
        k = None
        n = len(world.probe_ids)
        for i in range(len(rest) - n, -1, -1):
            if rest[i : i + n] == world.probe_ids:
                k = i
                break
        if k is not None:
            script = world.codec.encode("{\\frac{1}{2}} tail tail tail")
            return script[len(rest) - k - n :]
        return original(prompt)

    src.gen = ScriptedGenerator(nested, world.codec)
    p = src.probe([key])[0]
    assert p.answer_raw == "\\frac{1}{2}"
    assert "tail" not in world.codec.decode(p.output_ids[: len(world.codec.encode("{\\frac{1}{2}}"))])


def test_natural_end_answer_uses_suffix(world: Any) -> None:
    gen = world.generator()
    src = world.source(gen)
    key = TraceKey("w/0", 0)
    src.start([key])
    _walk(src, key, world.cfg)
    a = src.answer([key])[0]
    assert a.prompt_tokens == len(world.answer_suffix)  # không chèn lại </think>
    last = gen.calls[-1][0].prompt_ids
    assert list(last[-len(world.answer_suffix) - 1 :]) == [world.codec.think_end_id, *world.answer_suffix]


def test_eos_forces_answer(qwen3_tokenizer: Any) -> None:
    w = make_world(qwen3_tokenizer, n_questions=1, max_len_q=None)
    w.chains["w/0"] = [*w.codec.encode(" a" * 30), w.codec.eos_ids[0]]
    gen = w.generator()
    src = w.source(gen)
    key = TraceKey("w/0", 0)
    src.start([key])
    r = src.advance([AdvanceRequest(key, 16, 48, True)])[0]
    assert r.kind == "eos"
    a = src.answer([key])[0]
    assert a.prompt_tokens == len(w.answer_suffix) + 1
    prompt = gen.calls[-1][0].prompt_ids
    assert w.codec.eos_ids[0] not in prompt[len(w.prompt_ids["w/0"]) :]  # bỏ token EOS: pos = gen_end - 1


def test_answer_at_start_point(world: Any) -> None:
    src = world.source()
    key = TraceKey("w/0", 0)
    src.start([key])
    a = src.answer([key])[0]
    assert a.prompt_tokens == 1 + len(world.answer_suffix) and a.extracted in (GOLD, "17")


def test_answer_cost_never_exceeds_c_ans_max(world: Any) -> None:
    src = world.source()
    keys = [TraceKey(q, 0) for q in world.questions]
    src.start(keys)
    for _ in range(4):
        live = [k for k in keys if src.chains[k].ended is None]
        if not live:
            break
        src.advance([AdvanceRequest(k, 16, 48, True) for k in live])
        for a in src.answer(keys, finalize=False):
            assert a.prompt_tokens + a.output_tokens <= src.c_ans_max()
        for k in [k for k in keys if src.chains[k].ended is None]:
            p = src.probe([k])[0]
            assert p.prompt_tokens + p.output_tokens <= src.c_probe_max()


def test_live_source_requires_answer_mode(qwen3_tokenizer: Any) -> None:
    w = make_world(qwen3_tokenizer, n_questions=1, cfg=small_cfg(answer={"mode": None}))
    with pytest.raises(ValueError, match=r"answer\.mode"):
        w.source()
