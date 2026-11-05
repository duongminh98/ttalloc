"""Learning curve của bộ ước lượng theo số câu train, để quyết định có ghi thêm train ở P3 không (ADR-015).

Học trên các tập con lồng nhau của train (cùng thứ tự `hash_order` mà `ttalloc log --limit` dùng), đánh giá trên val.
Ngưỡng được chốt trước khi có dữ liệu thật và chỉ dùng val; không chỉnh theo kết quả.
"""

from __future__ import annotations

from typing import Any

# Ghi thêm train khi gấp đôi số câu còn cải thiện ít nhất một trong hai mức sau (ADR-015).
AUROC_MIN_GAIN = 0.01
LOG_LOSS_MIN_REL_GAIN = 0.01
TRAIN_MAX = 1500  # số câu của tập train (PLAN mục 4.1)


def nested_subsets(order: list[str], sizes: list[int]) -> dict[int, set[str]]:
    """Tập con lồng nhau: `sizes[i]` câu đầu của `order`."""
    if any(n <= 0 or n > len(order) for n in sizes):
        raise ValueError(f"cỡ {sizes} phải trong [1, {len(order)}] (số câu train đã ghi)")
    return {n: set(order[:n]) for n in sorted(set(sizes))}


def decide(rows: list[dict[str, Any]], logged: int) -> dict[str, Any]:
    """So hai cỡ lớn nhất; `logged` là số câu train đã ghi. Trả về cỡ train nên ghi tới."""
    if len(rows) < 2:
        raise ValueError("cần ít nhất hai cỡ để so")
    a, b = sorted(rows, key=lambda r: r["n_questions"])[-2:]
    gain_auroc = float(b["auroc_q0"] - a["auroc_q0"])
    gain_ll = float((a["mean_log_loss"] - b["mean_log_loss"]) / a["mean_log_loss"])
    extend = gain_auroc >= AUROC_MIN_GAIN or gain_ll >= LOG_LOSS_MIN_REL_GAIN
    target = min(2 * logged, TRAIN_MAX) if extend else logged
    return {
        "from": a["n_questions"],
        "to": b["n_questions"],
        "gain_auroc_q0": gain_auroc,
        "gain_log_loss_rel": gain_ll,
        "thresholds": {"auroc_q0": AUROC_MIN_GAIN, "log_loss_rel": LOG_LOSS_MIN_REL_GAIN},
        "extend": extend,
        "train_limit": target,
    }
