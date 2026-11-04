"""Quy tắc phát hiện sự cố trong log khi đang chạy (dùng chung cho ops/kaggle/push.py và ops/vast/vast.py watch).

Mỗi quy tắc là (tên, regex). Khớp là dấu hiệu cần dừng theo dõi để xem và sửa ngay, không đợi công việc kết thúc.
"""

from __future__ import annotations

import re

RULES: list[tuple[str, re.Pattern[str]]] = [
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("ttalloc_error", re.compile(r"\bLỖI\b|BudgetExceeded|SimValidityError|LogHashMismatch|InfeasibleBudget")),
    ("oom", re.compile(r"CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError|Killed\b")),
    ("cuda", re.compile(r"CUDA error|no kernel image is available|NCCL error")),
    ("pip", re.compile(r"ERROR: (Could not|Package|No matching|pip's dependency resolver)")),
    ("disk", re.compile(r"No space left on device")),
]


def scan(lines: list[str]) -> list[tuple[str, str]]:
    """Trả về các (quy tắc, dòng) khớp, theo thứ tự xuất hiện."""
    out = []
    for line in lines:
        for name, pat in RULES:
            if pat.search(line):
                out.append((name, line.strip()[:300]))
                break
    return out


def new_lines(prev: list[str], cur: list[str]) -> list[str]:
    """Các dòng mới của `cur` so với `prev`, khi cả hai là đuôi (tail) của cùng một log đang dài thêm."""
    if not prev:
        return cur
    if cur[: len(prev)] == prev:
        return cur[len(prev) :]
    # Cửa sổ đuôi đã trượt: tìm dòng cuối của lần trước trong lần này.
    for k in range(min(len(prev), 50), 0, -1):
        tail = prev[-k:]
        for i in range(len(cur) - k, -1, -1):
            if cur[i : i + k] == tail:
                return cur[i + k :]
    return cur
