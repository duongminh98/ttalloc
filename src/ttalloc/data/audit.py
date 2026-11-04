"""Lấy mẫu câu trả lời để chấm tay, kiểm tra bộ chấm tự động (PLAN P0.T10)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

AUDIT_COLUMNS = [
    "qid",
    "seed",
    "source",
    "problem",
    "gold",
    "answer_text",
    "extracted",
    "auto_correct",
    "manual_correct",
]


def sample_from_run(run_dir: Path, n: int, rng: np.random.Generator) -> list[dict[str, object]]:
    """Đọc `answers.parquet` của một thư mục chạy (P1.T5) và lấy ngẫu nhiên n dòng (ít hơn nếu không đủ)."""
    rows = pq.read_table(run_dir / "answers.parquet").to_pylist()
    take = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False)) if rows else []
    out = []
    for i in take:
        r = rows[int(i)]
        out.append(
            {
                "qid": r["qid"],
                "seed": r["seed"],
                "source": "run",
                "problem": r["problem"],
                "gold": r["gold"],
                "answer_text": r["answer_text"],
                "extracted": r["extracted"],
                "auto_correct": int(bool(r["correct"])),
                "manual_correct": "",
            }
        )
    return out


def sample_from_logs(
    points: list[dict[str, object]], problems: dict[str, tuple[str, str]], n: int, rng: np.random.Generator
) -> list[dict[str, object]]:
    """Lấy mẫu từ câu trả lời đã ghi (bảng points và câu trả lời tự nhiên). `problems`: qid -> (đề, đáp án chuẩn)."""
    take = sorted(rng.choice(len(points), size=min(n, len(points)), replace=False)) if points else []
    out = []
    for i in take:
        r = points[int(i)]
        prob, gold = problems[str(r["qid"])]
        out.append(
            {
                "qid": r["qid"],
                "seed": r["seed"],
                "source": r.get("source", "logs"),
                "problem": prob,
                "gold": gold,
                "answer_text": r["answer_text"],
                "extracted": r["answer_extracted"],
                "auto_correct": int(bool(r["answer_correct"])),
                "manual_correct": "",
            }
        )
    return out


def write_audit_csv(rows: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path
