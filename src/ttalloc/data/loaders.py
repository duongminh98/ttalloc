"""Nạp các tập dữ liệu thành danh sách `Question` (PLAN P0.T6, mục 4.1)."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ttalloc.config import DatasetCfg, load_dataset_cfg
from ttalloc.data.grading import extract_boxed
from ttalloc.data.prompts import shuffle_choices
from ttalloc.types import Question

log = logging.getLogger(__name__)

_LEVEL_RE = re.compile(r"(\d+)")


def parse_difficulty(raw: Any) -> int | None:
    """`"Level 3"` hoặc `3` thành 3; `"Level ?"`, rỗng hoặc ngoài khoảng 1–5 thành None."""
    if raw is None:
        return None
    if isinstance(raw, int):
        level = raw
    else:
        m = _LEVEL_RE.search(str(raw))
        if not m:
            return None
        level = int(m.group(1))
    return level if 1 <= level <= 5 else None


def make_qid(cfg: DatasetCfg, subset: str | None, row: Mapping[str, Any], index: int) -> str:
    """`<dataset>/[<subset>/]<id gốc hoặc chỉ số>`; ổn định qua các lần nạp với cùng revision."""
    local = str(row[cfg.fields.id]) if cfg.fields.id else str(index)
    parts = [cfg.name] + ([subset] if subset and len(cfg.hf_configs) > 1 else []) + [local]
    return "/".join(parts)


def row_to_question(cfg: DatasetCfg, subset: str | None, row: Mapping[str, Any], index: int) -> Question | None:
    f = cfg.fields
    qid = make_qid(cfg, subset, row, index)
    problem = str(row[f.problem]).strip()
    options: tuple[str, ...] | None = None
    if cfg.answer_type == "choice":
        if f.correct is None or not f.incorrect:
            raise ValueError(f"{cfg.name}: tập trắc nghiệm cần fields.correct và fields.incorrect")
        options, gold = shuffle_choices(qid, str(row[f.correct]).strip(), [str(row[k]).strip() for k in f.incorrect])
    elif f.gold is not None:
        gold = str(row[f.gold]).strip()
    elif f.gold_from_boxed is not None:
        boxed = extract_boxed(str(row[f.gold_from_boxed]))
        if boxed is None:
            return None
        gold = boxed
    else:
        raise ValueError(f"{cfg.name}: cần fields.gold hoặc fields.gold_from_boxed")
    if not problem or not gold:
        return None
    return Question(
        qid=qid,
        dataset=cfg.name,
        problem=problem,
        options=options,
        gold=gold,
        answer_type=cfg.answer_type,
        difficulty=parse_difficulty(row[f.difficulty]) if f.difficulty else None,
    )


def rows_to_questions(cfg: DatasetCfg, subset: str | None, rows: Iterable[Mapping[str, Any]]) -> list[Question]:
    """Hàm thuần: chuyển các dòng thô của một subset thành Question; chỉ số tính theo thứ tự dòng của subset."""
    out: list[Question] = []
    skipped = 0
    for i, row in enumerate(rows):
        q = row_to_question(cfg, subset, row, i)
        if q is None:
            skipped += 1
        else:
            out.append(q)
    if skipped:
        log.warning("%s/%s: bỏ %d dòng không có đề hoặc đáp án chuẩn", cfg.name, subset, skipped)
    qids = [q.qid for q in out]
    if len(set(qids)) != len(qids):
        raise ValueError(f"{cfg.name}/{subset}: qid bị trùng")
    return out


def load_questions(cfg: DatasetCfg | str) -> list[Question]:
    """Tải từ Hugging Face (cần mạng; tập có kiểm soát truy cập cần HF_TOKEN) ở revision đã ghim."""
    import datasets

    if isinstance(cfg, str):
        cfg = load_dataset_cfg(cfg)
    out: list[Question] = []
    for subset in cfg.hf_configs:
        ds = datasets.load_dataset(cfg.hf_id, name=subset, split=cfg.split, revision=cfg.revision)
        out.extend(rows_to_questions(cfg, subset, ds))
    return out
