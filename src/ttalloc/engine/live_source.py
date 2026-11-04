"""SegmentSource chạy thật trên một Generator (PLAN A1, P1.T3).

Mỗi thao tác (sinh đoạn, gợi đáp án thử, trả lời) là một coroutine: nó `yield` một `GenRequest`, nhận lại `GenOutput`
và quyết định gửi tiếp hay kết thúc. `_drive` gom mọi request đang chờ của mọi câu thành một lời gọi `generate` mỗi
lượt, nên các lần gửi lại (A1a, A1b) cũng được gom lô và tận dụng prefix caching.
"""

from __future__ import annotations

import math
from collections.abc import Generator as Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from ttalloc.config import ExperimentCfg, ModelCfg
from ttalloc.data.grading import canonical_key, extract_boxed, extract_choice, grade
from ttalloc.data.prompts import PromptSet
from ttalloc.engine.cut import (
    classify,
    count_markers,
    marker_count_regex,
    marker_regex,
    newline_cutback,
    stop_strings,
)
from ttalloc.engine.generator import TokenCodec, stable_seed
from ttalloc.types import (
    AdvanceRequest,
    AnswerResult,
    Generator,
    GenOutput,
    GenRequest,
    ProbeResult,
    Question,
    SegmentResult,
    TraceKey,
)

R = TypeVar("R")
Job = Coroutine[GenRequest, GenOutput, R]
AnswerMode = Literal["full", "short"]

MARKER_CTX_TOKENS = 16  # số token ngữ cảnh khi đếm từ chuyển ý tăng dần


def extract_answer(text: str, answer_type: str) -> str | None:
    return extract_choice(text) if answer_type == "choice" else extract_boxed(text)


@dataclass
class BoxState:
    closed: bool
    content: str
    open_at: int | None  # vị trí ký tự '{' mở khung, tính trong phần văn bản đã sinh
    close_at: int | None  # vị trí ký tự '}' đóng khung, tính trong phần văn bản đã sinh


def box_state(prefix: str, generated: str) -> BoxState:
    """Trạng thái của khung mở bởi `\\boxed` cuối cùng trong `prefix` (lời nhắc), đếm ngoặc trên phần đã sinh (A1b).

    Lời nhắc gợi kết thúc bằng `\\boxed` (model tự mở ngoặc); lời nhắc trả lời `short` kết thúc bằng `\\boxed{`
    (khung đã mở, `open_at = -1`).
    """
    n = len(generated)
    if prefix.endswith("{"):
        i, depth = -1, 1
    else:
        i = 0
        while i < n and generated[i] == " ":
            i += 1
        if i >= n or generated[i] != "{":
            return BoxState(False, generated.strip(), None, None)
        depth = 0
    for j in range(max(i, 0), n):
        c = generated[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return BoxState(True, generated[i + 1 : j].strip(), i, j)
    return BoxState(False, generated[i + 1 :].strip(), i, None)


def confidence_span(codec: TokenCodec, gen: list[int], box: BoxState) -> tuple[int, int]:
    """Khoảng token [a, b) dùng tính độ tự tin: từ token đầu tiên sau dấu `{` mở khung tới token chứa dấu `}` đóng
    khung (gồm cả token đó). Token chứa `{` được tính nếu nó chứa luôn nội dung đáp án (ví dụ token `{4`)."""
    if box.open_at is None or not gen:
        return 0, 0
    ends = [len(codec.decode(gen[: i + 1])) for i in range(len(gen))]

    def tok_at(c: int) -> int:
        for i, e in enumerate(ends):
            if e > c:
                return i
        return len(gen) - 1

    i_open = tok_at(box.open_at)
    a = i_open if ends[i_open] > box.open_at + 1 else i_open + 1
    b = tok_at(box.close_at) + 1 if box.close_at is not None else len(gen)
    if a >= b:
        a = max(0, b - 1)
    return a, b


def conf_from_logprobs(lps: list[float]) -> tuple[float, float]:
    if not lps:
        return 0.0, 0.0
    geo = math.exp(sum(lps) / len(lps))
    arith = sum(math.exp(x) for x in lps) / len(lps)
    return geo, arith


@dataclass
class LocalConf:
    """Giá trị nhỏ nhất của trung bình logprob trên mọi cửa sổ `w` token tới `pos` (A5), tính tăng dần."""

    w: int
    prefix: list[float] = field(default_factory=lambda: [0.0])
    last_t: int = 0
    cur_min: float = math.inf

    def extend(self, lps: list[float]) -> None:
        s = self.prefix[-1]
        for x in lps:
            s += x
            self.prefix.append(s)

    def min_upto(self, pos: int) -> float | None:
        if pos <= 0:
            return None
        if pos < self.w:
            return self.prefix[pos] / pos
        for t in range(max(self.w, self.last_t + 1), pos + 1):
            m = (self.prefix[t] - self.prefix[t - self.w]) / self.w
            if m < self.cur_min:
                self.cur_min = m
        self.last_t = max(self.last_t, pos)
        return self.cur_min


@dataclass
class ChainState:
    key: TraceKey
    question: Question
    prompt_ids: list[int]
    chain: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    pos: int = 0
    point_kind: str = "start"
    ended: str | None = None  # natural_end | eos | truncated
    n_points: int = 0
    marker_pos: int = 0
    marker_count: int = 0
    local: LocalConf | None = None

    @property
    def gen_end(self) -> int:
        return len(self.chain)


class LiveSource:
    def __init__(
        self,
        generator: Generator,
        codec: TokenCodec,
        questions: dict[str, Question],
        prompt_ids: dict[str, list[int]],
        cfg: ExperimentCfg,
        model_cfg: ModelCfg,
        prompts: PromptSet,
        l_log: int | None = None,
        answer_mode: AnswerMode | None = None,
    ) -> None:
        self.gen = generator
        self.codec = codec
        self.questions = questions
        self.prompt_ids = prompt_ids
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.prompts = prompts
        self.l_log = l_log
        mode = answer_mode or cfg.answer.mode
        if mode is None:
            raise ValueError("answer.mode chưa được chọn (P1.T9); truyền answer_mode khi chạy study answer-mode")
        self.answer_mode: AnswerMode = mode
        self.stops = stop_strings(cfg.dp.markers)
        self.marker_re = marker_regex(cfg.dp.markers)
        self.count_re = marker_count_regex(cfg.dp.markers)
        self.probe_ids = {
            t: codec.encode(prompts.probe_choice if t == "choice" else prompts.probe_math) for t in ("math", "choice")
        }
        self.chains: dict[TraceKey, ChainState] = {}
        self.n_generate_calls = 0

    # ------------------------------------------------------------ chi phí tối đa (mục 2.6)

    def answer_prompt_ids(self, mode: AnswerMode | None = None) -> list[int]:
        return [self.codec.think_end_id, *self.codec.encode(self.prompts.answer(mode or self.answer_mode))]

    def r_ans(self, mode: AnswerMode | None = None) -> int:
        m = mode or self.answer_mode
        if self.cfg.answer.r_ans is not None and m == self.cfg.answer.mode:
            return self.cfg.answer.r_ans
        return 256 if m == "full" else 32

    def c_probe_max(self) -> int:
        return (
            max(len(v) for v in self.probe_ids.values())
            + self.cfg.probe.r_probe
            + self.model_cfg.probe_think_end_lookahead
        )

    def c_ans_max(self, mode: AnswerMode | None = None) -> int:
        return len(self.answer_prompt_ids(mode)) + self.r_ans(mode)

    # ------------------------------------------------------------ SegmentSource

    def start(self, keys: list[TraceKey]) -> None:
        for k in keys:
            if k in self.chains:
                raise ValueError(f"{k}: đã start")
            q = self.questions[k.qid]
            local = LocalConf(self.cfg.features.local_window) if self.cfg.features.local_conf else None
            self.chains[k] = ChainState(key=k, question=q, prompt_ids=list(self.prompt_ids[k.qid]), local=local)

    def advance(self, reqs: list[AdvanceRequest]) -> list[SegmentResult]:
        return self._drive([self._advance_job(self.chains[r.key], r) for r in reqs])

    def probe(self, keys: list[TraceKey]) -> list[ProbeResult]:
        return self._drive([self._probe_job(self.chains[k]) for k in keys])

    def answer(self, keys: list[TraceKey], finalize: bool = True, mode: AnswerMode | None = None) -> list[AnswerResult]:
        res = self._drive([self._answer_job(self.chains[k], mode or self.answer_mode) for k in keys])
        if finalize:
            for k in keys:
                del self.chains[k]
        return res

    def probe_and_answer(
        self, keys: list[TraceKey], modes: tuple[AnswerMode, ...] | None = None
    ) -> tuple[list[ProbeResult], dict[AnswerMode, list[AnswerResult]]]:
        """Gợi đáp án thử và sinh câu trả lời ép buộc tại điểm hiện tại, gom chung lô (dùng cho logger và study)."""
        modes = modes or (self.answer_mode,)
        jobs: list[Job[Any]] = [self._probe_job(self.chains[k]) for k in keys]
        for m in modes:
            jobs += [self._answer_job(self.chains[k], m) for k in keys]
        out = self._drive(jobs)
        n = len(keys)
        return out[:n], {m: out[n * (i + 1) : n * (i + 2)] for i, m in enumerate(modes)}

    def answer_modes(
        self, keys: list[TraceKey], modes: tuple[AnswerMode, ...], finalize: bool = False
    ) -> dict[AnswerMode, list[AnswerResult]]:
        """Câu trả lời theo nhiều dạng tại cùng vị trí, gom chung lô (study answer-mode, P1.T9)."""
        jobs = [self._answer_job(self.chains[k], m) for m in modes for k in keys]
        out = self._drive(jobs)
        if finalize:
            self.drop(keys)
        n = len(keys)
        return {m: out[n * i : n * (i + 1)] for i, m in enumerate(modes)}

    def probe_with(self, keys: list[TraceKey], prompts: PromptSet) -> list[ProbeResult]:
        """Gợi bằng một bộ lời nhắc khác (study probe-noise, P1.T10); không đổi chuỗi chính."""
        return self._drive([self._probe_job(self.chains[k], prompts) for k in keys])

    def drop(self, keys: list[TraceKey]) -> None:
        for k in keys:
            self.chains.pop(k, None)

    # ------------------------------------------------------------ bộ điều phối

    def _drive(self, jobs: list[Job[R]]) -> list[R]:
        results: list[Any] = [None] * len(jobs)
        pending: dict[int, GenRequest] = {}
        for i, j in enumerate(jobs):
            try:
                pending[i] = next(j)
            except StopIteration as stop:
                results[i] = stop.value
        while pending:
            idx = list(pending)
            outs = self.gen.generate([pending[i] for i in idx])
            self.n_generate_calls += 1
            nxt: dict[int, GenRequest] = {}
            for i, o in zip(idx, outs, strict=True):
                try:
                    nxt[i] = jobs[i].send(o)
                except StopIteration as stop:
                    results[i] = stop.value
            pending = nxt
        return results

    def _sampling(self) -> dict[str, Any]:
        s = self.model_cfg.sampling
        return {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k}

    def _advance_job(self, ch: ChainState, req: AdvanceRequest) -> Job[SegmentResult]:
        if ch.ended is not None:
            raise ValueError(f"{ch.key}: chuỗi đã kết thúc ({ch.ended}), không sinh tiếp được")
        start = ch.gen_end
        stops = self.stops if req.stop_at_markers else ()
        lp = 0 if ch.local is not None else None
        while True:
            seg_start = ch.gen_end
            remaining = req.max_tokens - (seg_start - start)
            if self.l_log is not None:
                remaining = min(remaining, self.l_log - seg_start)
            if remaining <= 0:
                if self.l_log is not None and seg_start >= self.l_log and seg_start == start:
                    ch.ended = "truncated"
                    ch.pos = seg_start
                    return SegmentResult(ch.key, "truncated", pos=seg_start, gen_end=seg_start, n_new_tokens=0)
                pos = newline_cutback(ch.chain, start, seg_start, self.codec)
                return self._point(ch, "length", pos, start, None)
            out = yield GenRequest(
                prompt_ids=tuple(ch.prompt_ids + ch.chain),
                max_tokens=remaining,
                stop=stops,
                stop_token_ids=(self.codec.think_end_id,),
                logprobs=lp,
                seed=stable_seed(ch.key.qid, ch.key.seed, seg_start, self.cfg.engine.seed),
                **self._sampling(),
            )
            ev = classify(ch.chain, start, seg_start, out, self.codec, req.min_tokens, self.marker_re, final=False)
            if ev.kind == "peek":
                if remaining - len(out.token_ids) > 0:
                    # Từ chuyển ý nằm sát cuối văn bản: sinh thêm một token để biết ranh giới từ ("But" hay "Butter").
                    extra = yield GenRequest(
                        prompt_ids=tuple(ch.prompt_ids + ch.chain + list(out.token_ids)),
                        max_tokens=1,
                        stop_token_ids=(self.codec.think_end_id,),
                        logprobs=lp,
                        seed=stable_seed(ch.key.qid, ch.key.seed, seg_start + len(out.token_ids), self.cfg.engine.seed),
                        **self._sampling(),
                    )
                    eos = extra.finish_reason == "stop" and extra.stop_reason is None
                    out = GenOutput(
                        token_ids=out.token_ids + extra.token_ids,
                        logprobs=None if lp is None else (out.logprobs or ()) + (extra.logprobs or ()),
                        finish_reason="stop",
                        stop_reason=None if eos else out.stop_reason,
                    )
                ev = classify(ch.chain, start, seg_start, out, self.codec, req.min_tokens, self.marker_re)
            ch.chain.extend(out.token_ids)
            if ch.local is not None:
                lps = list(out.logprobs or [])
                if len(lps) != len(out.token_ids):
                    raise RuntimeError(f"{ch.key}: số logprob ({len(lps)}) khác số token ({len(out.token_ids)})")
                ch.logprobs.extend(lps)
                ch.local.extend(lps)
            if ev.kind == "resume":
                continue
            if ev.kind in ("natural_end", "eos"):
                ch.ended = ev.kind
                ch.pos = ev.pos
                return SegmentResult(ch.key, ev.kind, pos=ev.pos, gen_end=ch.gen_end, n_new_tokens=ch.gen_end - start)
            return self._point(ch, ev.kind, ev.pos, start, ev.marker)

    def _point(self, ch: ChainState, kind: str, pos: int, start: int, marker: str | None) -> SegmentResult:
        ch.pos = pos
        ch.point_kind = kind
        ch.n_points += 1
        return SegmentResult(
            ch.key,
            kind,  # type: ignore[arg-type]
            pos=pos,
            gen_end=ch.gen_end,
            n_new_tokens=ch.gen_end - start,
            marker=marker,
            point_idx=ch.n_points,
            n_markers=self._n_markers(ch, pos),
            local_conf_min=ch.local.min_upto(pos) if ch.local is not None else None,
        )

    def _n_markers(self, ch: ChainState, pos: int) -> int:
        """Số từ chuyển ý ở đầu đoạn trong chain[:pos], đếm tăng dần từ điểm trước (không giải mã lại cả chuỗi)."""
        a = max(0, ch.marker_pos - MARKER_CTX_TOKENS)
        head_len = len(self.codec.decode(ch.chain[a : ch.marker_pos]))
        text = self.codec.decode(ch.chain[a:pos])
        ch.marker_count += sum(1 for m in self.count_re.finditer(text) if m.end() >= head_len)
        ch.marker_pos = pos
        return ch.marker_count

    def n_markers_full(self, key: TraceKey) -> int:
        """Đếm trên toàn bộ chain[:pos] (chỉ dùng trong test để đối chiếu)."""
        ch = self.chains[key]
        return count_markers(self.codec.decode(ch.chain[: ch.pos]), self.count_re)

    def _gen_until_box_closed(
        self, base: list[int], prefix_text: str, limit: int
    ) -> Coroutine[GenRequest, GenOutput, tuple[list[int], list[float], BoxState, bool]]:
        gen: list[int] = []
        lps: list[float] = []
        eos = False
        box = BoxState(False, "", None, None)
        while len(gen) < limit:
            out = yield GenRequest(
                prompt_ids=tuple(base + gen), max_tokens=limit - len(gen), stop=("}",), temperature=0.0, logprobs=1
            )
            gen.extend(out.token_ids)
            lps.extend(out.logprobs or [0.0] * len(out.token_ids))
            box = box_state(prefix_text, self.codec.decode(gen))
            eos = out.finish_reason == "stop" and out.stop_reason is None
            if box.closed or eos or out.finish_reason == "length" or self.codec.think_end_id in out.token_ids:
                break
        return gen, lps, box, eos

    def _probe_job(self, ch: ChainState, prompts: PromptSet | None = None) -> Job[ProbeResult]:
        if ch.ended is not None:
            raise ValueError(f"{ch.key}: không gợi được sau khi chuỗi kết thúc ({ch.ended})")
        if prompts is None:
            probe_ids = self.probe_ids[ch.question.answer_type]
            probe_text = self.prompts.probe(ch.question)
        else:
            probe_text = prompts.probe(ch.question)
            probe_ids = self.codec.encode(probe_text)
        base = ch.prompt_ids + ch.chain[: ch.pos] + probe_ids
        gen, lps, box, eos = yield from self._gen_until_box_closed(base, probe_text, self.cfg.probe.r_probe)
        a, b = confidence_span(self.codec, gen, box)
        span = lps[a:b]
        geo, arith = conf_from_logprobs(span)
        has_end = self.codec.think_end_id in gen
        look = self.model_cfg.probe_think_end_lookahead
        if look > 0 and box.closed and not eos and not has_end:
            out = yield GenRequest(
                prompt_ids=tuple(base + gen),
                max_tokens=look,
                stop_token_ids=(self.codec.think_end_id,),
                temperature=0.0,
                logprobs=1,
            )
            gen = gen + list(out.token_ids)
            has_end = self.codec.think_end_id in out.token_ids
        raw = box.content if box.open_at is not None else (extract_boxed("\\boxed" + self.codec.decode(gen)) or "")
        return ProbeResult(
            key=ch.key,
            answer_raw=raw,
            answer_key=canonical_key(raw),
            conf_geo=geo,
            conf_arith=arith,
            logprobs=tuple(span),
            prompt_tokens=len(probe_ids),
            output_tokens=len(gen),
            has_think_end=has_end,
            output_ids=tuple(gen),
            gen_end=ch.gen_end,
        )

    def _answer_job(self, ch: ChainState, mode: AnswerMode) -> Job[AnswerResult]:
        if ch.ended == "natural_end":
            pos = ch.pos  # chain[:pos] đã gồm </think> do model tự sinh
            ids = self.answer_prompt_ids(mode)[1:]
        elif ch.ended == "eos":
            pos = ch.gen_end - 1
            ids = self.answer_prompt_ids(mode)
        else:
            pos = ch.pos
            ids = self.answer_prompt_ids(mode)
        base = ch.prompt_ids + ch.chain[:pos] + ids
        r_ans = self.r_ans(mode)
        prefix_text = self.prompts.answer(mode)
        if mode == "short":
            gen, _, _, _ = yield from self._gen_until_box_closed(base, prefix_text, r_ans)
        else:
            out = yield GenRequest(prompt_ids=tuple(base), max_tokens=r_ans, temperature=0.0)
            gen = list(out.token_ids)
        text = self.codec.decode(gen)
        extracted = extract_answer(prefix_text + text, ch.question.answer_type)
        q = ch.question
        return AnswerResult(
            key=ch.key,
            text=text,
            extracted=extracted,
            correct=grade(extracted, q.gold, q.answer_type),
            prompt_tokens=len(ids),
            output_tokens=len(gen),
        )
