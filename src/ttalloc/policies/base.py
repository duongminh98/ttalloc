"""Phá thế hòa và registry chính sách (PLAN mục 3.4, P2.T11)."""

from __future__ import annotations

import hashlib

from ttalloc.types import TraceKey


def tie_key(key: TraceKey, run_seed: int) -> int:
    """Mã băm ổn định của khóa câu hỏi và seed của lần chạy.

    Không phá thế hòa bằng qid, vì như vậy tên tập dữ liệu sẽ quyết định câu nào được ưu tiên.
    """
    blob = f"{key.qid}\x1f{key.seed}\x1f{run_seed}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


# Tên chính sách trong config thí nghiệm (mục 7.2) -> mô-đun cài đặt. Dựng đối tượng ở ttalloc.policies.factory.
REGISTRY = {
    "deer": "ttalloc.policies.deer",
    "price_adaptive": "ttalloc.policies.price",
}
