"""Generator theo kịch bản cho test CPU (PLAN mục 3.3).

`continuation(prompt_ids)` trả về "chuỗi tiếp nối thật" của prompt. Generator mô phỏng `max_tokens`, chuỗi dừng (so
trên văn bản giải mã, kể cả khi chuỗi dừng nằm bên trong một token dài hơn), `stop_token_ids` và EOS như vLLM:
token hoàn thành chuỗi dừng, token dừng và token EOS đều nằm trong `token_ids` đầu ra.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ttalloc.engine.generator import TokenCodec
from ttalloc.types import GenOutput, GenRequest

Continuation = Callable[[tuple[int, ...]], Sequence[int]]
LogprobFn = Callable[[tuple[int, ...], int], float]


def _default_logprob(context: tuple[int, ...], token: int) -> float:
    return -0.01 * ((token * 7 + len(context)) % 13)


class ScriptedGenerator:
    def __init__(self, continuation: Continuation, codec: TokenCodec, logprob_fn: LogprobFn | None = None) -> None:
        self.continuation = continuation
        self.codec = codec
        self.logprob_fn = logprob_fn or _default_logprob
        self.calls: list[list[GenRequest]] = []  # để test kiểm tra số lần gọi và gom lô
        self.n_generated = 0

    def generate(self, reqs: list[GenRequest]) -> list[GenOutput]:
        self.calls.append(list(reqs))
        return [self._one(r) for r in reqs]

    def _one(self, req: GenRequest) -> GenOutput:
        script = list(self.continuation(req.prompt_ids))
        out: list[int] = []
        lps: list[float] = []
        for tok in script:
            if len(out) >= req.max_tokens:
                break
            ctx = req.prompt_ids + tuple(out)
            out.append(int(tok))
            lps.append(self.logprob_fn(ctx, int(tok)))
            if tok in self.codec.eos_ids:
                return self._done(out, lps, req, "stop", None)
            if tok in req.stop_token_ids:
                return self._done(out, lps, req, "stop", int(tok))
            if req.stop:
                text = self.codec.decode(out)
                hit = [s for s in req.stop if s in text]
                if hit:
                    # Chuỗi dừng xuất hiện sớm nhất trong văn bản là chuỗi vLLM báo.
                    first = min(hit, key=lambda s: text.index(s))
                    return self._done(out, lps, req, "stop", first)
        if len(out) < req.max_tokens:
            # Kịch bản hết token mà chưa gặp EOS: coi như model sinh EOS.
            if not self.codec.eos_ids:
                raise ValueError("kịch bản hết token và tokenizer không có EOS")
            out.append(self.codec.eos_ids[0])
            lps.append(self.logprob_fn(req.prompt_ids + tuple(out[:-1]), out[-1]))
            return self._done(out, lps, req, "stop", None)
        return self._done(out, lps, req, "length", None)

    def _done(
        self, out: list[int], lps: list[float], req: GenRequest, finish: str, stop_reason: str | int | None
    ) -> GenOutput:
        self.n_generated += len(out)
        return GenOutput(
            token_ids=tuple(out),
            logprobs=tuple(lps) if req.logprobs is not None else None,
            finish_reason=finish,  # type: ignore[arg-type]
            stop_reason=stop_reason,
        )
