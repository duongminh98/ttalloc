"""Dựng các đối tượng dùng chung cho mọi lệnh chạy trên model: tokenizer, lời nhắc, câu hỏi, prompt, `log_hash`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ttalloc.config import REPO_ROOT, ExperimentCfg, ModelCfg, load_model_cfg
from ttalloc.data.prompts import PromptSet, build_prompt_ids, load_prompts
from ttalloc.engine.generator import TokenCodec
from ttalloc.types import Question

DATA_DIR = REPO_ROOT / "data"
QUESTIONS_DIR = DATA_DIR / "questions"


@dataclass
class ModelContext:
    cfg: ExperimentCfg
    model_cfg: ModelCfg
    tokenizer: Any
    codec: TokenCodec
    prompts: PromptSet


_ROUNDTRIP_PROBE = "Please reason step by step.\n<think>\n"


def tokenizer_roundtrip_ok(tok: Any) -> bool:
    """Mã hóa rồi giải mã một câu mẫu có dấu cách, xuống dòng và `<think>`; tokenizer dựng sai thì không khớp."""
    ids = tok.encode(_ROUNDTRIP_PROBE, add_special_tokens=False)
    return bool(tok.decode(ids) == _ROUNDTRIP_PROBE)


def load_tokenizer(model_cfg: ModelCfg, path: str | Path | None = None) -> Any:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    src = str(path) if path is not None else model_cfg.hf_id
    kw: dict[str, Any] = {} if path is not None else {"revision": model_cfg.revision}
    tok = AutoTokenizer.from_pretrained(src, **kw)
    if tokenizer_roundtrip_ok(tok):
        return tok
    # transformers 5.16–5.19 (khoảng vLLM 0.32.0 yêu cầu) dựng tokenizer ByteLevel khai `LlamaTokenizerFast`
    # (R1-Distill) thành LlamaTokenizer hỏng, mất dấu cách. Đọc thẳng tokenizer.json thì đúng (ADR-013).
    tok = PreTrainedTokenizerFast.from_pretrained(src, **kw)
    if not tokenizer_roundtrip_ok(tok):
        raise RuntimeError(f"tokenizer của {src} không mã hóa rồi giải mã lại đúng câu mẫu")
    return tok


def vllm_tokenizer_dir(model_cfg: ModelCfg, path: str | Path | None = None) -> str | None:
    """Thư mục tokenizer đã sửa để đưa cho vLLM, hoặc None nếu AutoTokenizer (cách vLLM tự nạp) đã đúng.

    vLLM giải mã đầu ra và so stop string bằng tokenizer riêng của nó; tokenizer hỏng làm marker không bao giờ khớp.
    """
    import tempfile

    from transformers import AutoTokenizer

    src = str(path) if path is not None else model_cfg.hf_id
    kw: dict[str, Any] = {} if path is not None else {"revision": model_cfg.revision}
    if tokenizer_roundtrip_ok(AutoTokenizer.from_pretrained(src, **kw)):
        return None
    out = tempfile.mkdtemp(prefix="ttalloc-tok-")
    load_tokenizer(model_cfg, path).save_pretrained(out)
    return out


def model_context(cfg: ExperimentCfg, model: str | None = None, tokenizer: Any = None) -> ModelContext:
    name = model or cfg.model
    if not name:
        raise ValueError("chưa chọn model: đặt khóa `model` trong config hoặc truyền --model")
    mcfg = load_model_cfg(name)
    tok = tokenizer if tokenizer is not None else load_tokenizer(mcfg)
    return ModelContext(cfg, mcfg, tok, TokenCodec.from_tokenizer(tok, mcfg.think_end), load_prompts(cfg.probe.prompt))


def log_hash(cfg: ExperimentCfg, model_cfg: ModelCfg, prompts: PromptSet) -> str:
    """Mã băm của config ghi dữ liệu và lời nhắc (I9). Mô phỏng và chạy thật từ chối chạy nếu khác."""
    if cfg.answer.mode is None:
        raise ValueError("answer.mode chưa được chọn (P1.T9); với lần ghi thử dùng --set answer.mode=full")
    blob = {
        "model": [
            model_cfg.hf_id,
            model_cfg.revision,
            model_cfg.sampling.model_dump(),
            model_cfg.probe_think_end_lookahead,
        ],
        "engine": [cfg.engine.dtype, cfg.engine.seed],
        "markers": cfg.dp.markers,
        "log": [cfg.log.m, cfg.log.l_max, cfg.log.l_log],
        "probe": [cfg.probe.r_probe, cfg.probe.prompt],
        "answer": [cfg.answer.mode, cfg.answer.resolved_r_ans()],
        "local_conf": [cfg.features.local_conf, cfg.features.local_window],
        "prompts": prompts.digest(),
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _q_to_json(q: Question) -> dict[str, Any]:
    return {
        "qid": q.qid,
        "dataset": q.dataset,
        "problem": q.problem,
        "options": list(q.options) if q.options else None,
        "gold": q.gold,
        "answer_type": q.answer_type,
        "difficulty": q.difficulty,
    }


def _q_from_json(d: dict[str, Any]) -> Question:
    return Question(
        d["qid"],
        d["dataset"],
        d["problem"],
        tuple(d["options"]) if d["options"] else None,
        d["gold"],
        d["answer_type"],
        d["difficulty"],
    )


def load_split_questions(split: str, questions_dir: Path = QUESTIONS_DIR) -> list[Question]:
    """Câu hỏi của một tập đã khóa, theo thứ tự trong splits/<tên>.json.

    Lần đầu tải các tập nguồn từ Hugging Face (cần mạng) và lưu vào data/questions/<tên>.jsonl; các lần sau đọc tệp đó.
    """
    from ttalloc.data.splits import read_split

    qids = read_split(split)
    cache = questions_dir / f"{split}.jsonl"
    if cache.exists():
        rows = [_q_from_json(json.loads(line)) for line in cache.read_text(encoding="utf-8").splitlines() if line]
        by_id = {q.qid: q for q in rows}
    else:
        from ttalloc.data.loaders import load_questions

        datasets = sorted({qid.split("/", 1)[0] for qid in qids})
        by_id = {q.qid: q for name in datasets for q in load_questions(name)}
        missing = [qid for qid in qids if qid not in by_id]
        if missing:
            raise KeyError(f"{split}: {len(missing)} qid không có trong tập nguồn, ví dụ {missing[:3]}")
        questions_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(
            "".join(json.dumps(_q_to_json(by_id[q]), ensure_ascii=False) + "\n" for q in qids), encoding="utf-8"
        )
        tmp.replace(cache)
    return [by_id[q] for q in qids]


def filter_datasets(questions: list[Question], datasets: list[str] | None) -> list[Question]:
    if not datasets:
        return questions
    out = [q for q in questions if q.dataset in set(datasets)]
    if not out:
        raise ValueError(f"không có câu nào thuộc {datasets}")
    return out


def prompt_ids_for(questions: list[Question], ctx: ModelContext) -> dict[str, list[int]]:
    return {q.qid: build_prompt_ids(q, ctx.model_cfg, ctx.tokenizer, ctx.prompts) for q in questions}


def max_model_len(prompt_ids: dict[str, list[int]], chain_max: int, extra: int) -> int:
    """Độ dài ngữ cảnh cần cho vLLM: prompt dài nhất + chuỗi chính tối đa + phần gợi hoặc trả lời, làm tròn lên 256."""
    need = max(len(v) for v in prompt_ids.values()) + chain_max + extra
    return -(-need // 256) * 256
