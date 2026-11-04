"""Generator trên vLLM (PLAN mục 3.3, P1.T4). Module duy nhất import `vllm`, và chỉ import bên trong hàm."""

from __future__ import annotations

from typing import Any, Literal

from ttalloc.config import EngineCfg, ModelCfg
from ttalloc.engine.generator import to_sampling_kwargs
from ttalloc.types import GenOutput, GenRequest


class VllmGenerator:
    def __init__(
        self,
        model_cfg: ModelCfg,
        engine_cfg: EngineCfg,
        max_model_len: int,
        max_num_seqs: int | None = None,
        model_path: str | None = None,
    ) -> None:
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": model_path or model_cfg.hf_id,
            "dtype": engine_cfg.dtype,
            "max_model_len": int(max_model_len),
            "enable_prefix_caching": engine_cfg.enable_prefix_caching,
            "gpu_memory_utilization": engine_cfg.gpu_memory_utilization,
            "seed": engine_cfg.seed,
        }
        from ttalloc.context import tokenizer_roundtrip_ok, vllm_tokenizer_dir

        tok_dir = vllm_tokenizer_dir(model_cfg, model_path)
        if model_path is None and model_cfg.revision:
            kwargs["revision"] = model_cfg.revision
            if tok_dir is None:
                kwargs["tokenizer_revision"] = model_cfg.revision
        if tok_dir is not None:
            kwargs["tokenizer"] = tok_dir
        if max_num_seqs is not None:
            kwargs["max_num_seqs"] = int(max_num_seqs)
        self.llm = LLM(**kwargs)
        if not tokenizer_roundtrip_ok(self.llm.get_tokenizer()):
            raise RuntimeError("tokenizer của vLLM không giải mã đúng câu mẫu; stop string sẽ không khớp (ADR-013)")
        self.n_generated = 0
        self.n_prompt = 0

    def generate(self, reqs: list[GenRequest]) -> list[GenOutput]:
        if not reqs:
            return []
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=list(r.prompt_ids)) for r in reqs]
        params = [SamplingParams(**to_sampling_kwargs(r)) for r in reqs]
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        res = []
        for r, o in zip(reqs, outs, strict=True):
            c = o.outputs[0]
            ids = tuple(int(t) for t in c.token_ids)
            lps: tuple[float, ...] | None = None
            if r.logprobs is not None:
                if c.logprobs is None or len(c.logprobs) != len(ids):
                    raise RuntimeError(
                        f"vLLM trả {0 if c.logprobs is None else len(c.logprobs)} logprob cho {len(ids)} token"
                    )
                lps = tuple(float(d[t].logprob) for d, t in zip(c.logprobs, ids, strict=True))
            finish: Literal["stop", "length"] = "length" if c.finish_reason == "length" else "stop"
            res.append(GenOutput(token_ids=ids, logprobs=lps, finish_reason=finish, stop_reason=c.stop_reason))
            self.n_generated += len(ids)
            self.n_prompt += len(r.prompt_ids)
        return res

    def prefix_cache_hit_rate(self) -> float | None:
        """Tỉ lệ trúng prefix cache cộng dồn, nếu phiên bản vLLM cung cấp số đo (không có thì None)."""
        try:
            metrics = self.llm.get_metrics()
        except Exception:  # API số đo khác nhau giữa các phiên bản; số đo này chỉ để báo cáo
            return None
        hits = queries = None
        for m in metrics:
            name = getattr(m, "name", "")
            if name.endswith("prefix_cache_hits") or name.endswith("prefix_cache_hits_total"):
                hits = float(getattr(m, "value", 0.0))
            if name.endswith("prefix_cache_queries") or name.endswith("prefix_cache_queries_total"):
                queries = float(getattr(m, "value", 0.0))
        if hits is None or not queries:
            return None
        return hits / queries
