"""Sinh liền mạch: không gợi, không dừng sớm (PLAN P1.T5). Dùng lại ở P1.T6 và P4.T5."""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ttalloc.config import REPO_ROOT, ExperimentCfg, ModelCfg
from ttalloc.data.grading import grade
from ttalloc.engine.generator import TokenCodec, stable_seed
from ttalloc.engine.live_source import extract_answer
from ttalloc.logging.schema import ANSWERS_SCHEMA, to_table
from ttalloc.logging.store import atomic_write_table
from ttalloc.types import Generator, GenRequest, Question

Profile = Literal["log", "deer_paper"]
TEST_LOG = REPO_ROOT / "docs" / "test-log.md"


class ApprovalRequired(PermissionError):
    pass


def require_approval(
    split: str, approved_by: str | None, command: str, detail: str = "", log_path: Path = TEST_LOG
) -> None:
    """Mọi lần chạy trên `test_*` ngoài eval-test cần người duyệt và được ghi vào docs/test-log.md (mục 7.5)."""
    if not split.startswith("test"):
        return
    if not approved_by:
        raise ApprovalRequired(f"{command} trên {split} cần --approved-by <tên> (PLAN mục 7.5)")
    append_test_log(command, split, "—", approved_by, detail, log_path)


def repo_rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def append_test_log(
    command: str, split: str, config_hash: str, approved_by: str, note: str, log_path: Path = TEST_LOG
) -> None:
    """Một dòng của docs/test-log.md, đúng 6 cột của bảng: Ngày | Lệnh | Tập | Mã băm config | Người duyệt | Ghi chú.

    Trước đây lock ghi 5 cột và eval-test ghi 7 cột, lệch với tiêu đề bảng (bản tập dượt lần 1)."""
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    cells = [stamp, command, split, config_hash, approved_by, note]
    with log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("| " + " | ".join(c.replace("|", "/") for c in cells) + " |\n")


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int


def profile_sampling(profile: Profile, model_cfg: ModelCfg, cfg: ExperimentCfg) -> Sampling:
    if profile == "log":
        s = model_cfg.sampling
        return Sampling(s.temperature, s.top_p, s.top_k, cfg.log.l_log)
    if profile == "deer_paper":
        # Cấu hình của bài DEER để đối chiếu số công bố: giải mã greedy, 16384 token.
        return Sampling(0.0, 1.0, -1, cfg.baseline.paper_max_tokens)
    raise ValueError(f"profile không hợp lệ: {profile}")


def split_think(ids: list[int], codec: TokenCodec) -> tuple[int, str]:
    """Số token phần suy luận (gồm `</think>`, hoặc cả chuỗi nếu không tự kết thúc) và văn bản phần trả lời."""
    if codec.think_end_id in ids:
        k = ids.index(codec.think_end_id) + 1
        return k, codec.decode(ids[k:])
    return len(ids), codec.decode(ids)


def generate_full(
    generator: Generator,
    codec: TokenCodec,
    questions: list[Question],
    prompt_ids: dict[str, list[int]],
    seeds: list[int],
    sampling: Sampling,
    chunk: int = 512,
    on_chunk: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    jobs = [(q, s) for s in seeds for q in questions]
    rows: list[dict[str, Any]] = []
    for i in range(0, len(jobs), chunk):
        part = jobs[i : i + chunk]
        reqs = [
            GenRequest(
                prompt_ids=tuple(prompt_ids[q.qid]),
                max_tokens=sampling.max_tokens,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                seed=stable_seed(q.qid, s, "full"),
            )
            for q, s in part
        ]
        t0 = time.time()
        outs = generator.generate(reqs)
        dt_s = time.time() - t0
        new = []
        for (q, s), o in zip(part, outs, strict=True):
            ids = list(o.token_ids)
            think, ans_text = split_think(ids, codec)
            extracted = extract_answer(ans_text, q.answer_type)
            new.append(
                {
                    "qid": q.qid,
                    "seed": s,
                    "dataset": q.dataset,
                    "problem": q.problem,
                    "gold": q.gold,
                    "n_tokens": len(ids),
                    "think_tokens": think,
                    "natural_end": codec.think_end_id in ids,
                    "finish_reason": o.finish_reason,
                    "answer_text": ans_text,
                    "extracted": extracted,
                    "correct": grade(extracted, q.gold, q.answer_type),
                    "chunk_seconds": dt_s,
                }
            )
        rows += new
        if on_chunk:
            on_chunk(new)
    return rows


def summarize(rows: list[dict[str, Any]], seconds: float | None = None) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "n": len(df),
        "accuracy": float(df.correct.mean()),
        "mean_tokens": float(df.n_tokens.mean()),
        "median_tokens": float(df.n_tokens.median()),
        "mean_think_tokens": float(df.think_tokens.mean()),
        "natural_end_rate": float(df.natural_end.mean()) if "natural_end" in df else None,
        "unextracted_rate": float(df.extracted.isna().mean()),
        "by_dataset": {
            str(k): {
                "n": len(g),
                "accuracy": float(g.correct.mean()),
                "mean_tokens": float(g.n_tokens.mean()),
                "acc_by_seed": [float(x) for x in g.groupby("seed").correct.mean()],
            }
            for k, g in df.groupby("dataset")
        },
    }
    if seconds is not None:
        out["seconds"] = round(seconds, 1)
        out["tok_per_s"] = float(df.n_tokens.sum() / seconds) if seconds > 0 else None
    return out


def write_results(run_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    ans = [
        {k: r[k] for k in ANSWERS_SCHEMA.names} | {"seed": int(r["seed"]), "correct": bool(r["correct"])} for r in rows
    ]
    atomic_write_table(to_table(ans, ANSWERS_SCHEMA), run_dir / "answers.parquet")
    df = pd.DataFrame(rows).drop(columns=["problem", "answer_text"])
    df.to_parquet(run_dir / "per_question.parquet", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json), encoding="utf-8"
    )


def _json(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))
