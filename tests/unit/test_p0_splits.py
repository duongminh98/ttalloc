"""P0.T7: chia tập, khử trùng lặp, khóa và kiểm tra."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ttalloc.data.splits import (
    SplitsTamperedError,
    build_splits,
    normalize_text,
    verify_all,
    write_splits,
)
from ttalloc.types import Question


def _q(dataset: str, i: int, text: str | None = None, level: int | None = None) -> Question:
    return Question(
        qid=f"{dataset}/{i}",
        dataset=dataset,
        # Mỗi từ là duy nhất theo (tập, chỉ số), để các câu khác nhau không vô tình trùng 13-gram.
        problem=text or " ".join(f"{dataset}_{i}_{k}" for k in range(20)),
        options=None,
        gold=str(i),
        answer_type="math",
        difficulty=level,
    )


def _pool() -> dict[str, list[Question]]:
    return {
        "testset": [_q("testset", i) for i in range(20)],
        "math_pool": [_q("math_pool", i, level=1 + i % 5) for i in range(200)],
        "extra": [_q("extra", i) for i in range(50)],
    }


CFG: dict[str, Any] = {
    "seed": 123,
    "dedup": {"ngram": 13},
    "order": ["test_a", "val", "train"],
    "splits": {
        "test_a": [{"dataset": "testset"}],
        "val": [{"dataset": "math_pool", "n": 25, "stratify": "difficulty"}],
        "train": [{"dataset": "math_pool", "n": 50, "stratify": "difficulty"}, {"dataset": "extra", "n": 30}],
    },
}


def test_splits_disjoint() -> None:
    res = build_splits(_pool(), CFG)
    sets = {k: set(v.qids) for k, v in res.items()}
    assert sets["test_a"].isdisjoint(sets["val"]) and sets["val"].isdisjoint(sets["train"])
    assert sets["test_a"].isdisjoint(sets["train"])
    assert len(sets["test_a"]) == 20 and len(sets["val"]) == 25 and len(sets["train"]) == 80
    assert res["train"].counts == {"math_pool": 50, "extra": 30}
    pool = {q.qid: q for q in _pool()["math_pool"]}
    levels = [pool[qid].difficulty for qid in res["val"].qids]
    assert all(levels.count(lv) == 5 for lv in range(1, 6))  # phân tầng đều theo mức


def test_splits_deterministic() -> None:
    a = build_splits(_pool(), CFG)
    b = build_splits(_pool(), CFG)
    assert {k: v.to_json() for k, v in a.items()} == {k: v.to_json() for k, v in b.items()}
    other = dict(CFG, seed=124)
    c = build_splits(_pool(), other)
    assert c["train"].qids != a["train"].qids


def test_dedup_detects_planted_duplicate() -> None:
    pool = _pool()
    target = pool["testset"][3]
    # Cùng đề bài nhưng khác cách trình bày: chữ hoa, ký hiệu LaTeX trình bày, khoảng trắng.
    words = target.problem.split()
    planted = _q("extra", 999, text="  " + " ".join(w.upper() for w in words[:3]) + " $" + " ".join(words[3:6])
                 + "$ \\left " + "   ".join(words[6:]))
    pool["extra"].append(planted)
    # Trùng 13-gram: chép phần lớn một đề kiểm tra, thêm vài chữ ở đầu và cuối.
    long_test = _q("testset", 100, text=" ".join(f"w{i}" for i in range(40)))
    pool["testset"].append(long_test)
    ngram_dup = _q("extra", 998, text="Intro. " + " ".join(f"w{i}" for i in range(5, 36)) + " tail.")
    pool["extra"].append(ngram_dup)
    # Chỉ chung một câu mẫu (vài 13-gram) với đề kiểm tra: không phải trùng lặp.
    boiler = "can be written as m over n where m and n are relatively prime positive integers find m plus n"
    pool["testset"].append(_q("testset", 101, text="A long unique story about a frog and a pond. " + boiler))
    not_dup = _q("extra", 997, text=" ".join(f"unique{i}" for i in range(30)) + " " + boiler)
    pool["extra"].append(not_dup)
    cfg = dict(CFG)
    cfg["splits"] = dict(CFG["splits"], train=[{"dataset": "extra"}])
    res = build_splits(pool, cfg)
    assert normalize_text(planted.problem).replace(" ", "") == normalize_text(target.problem).replace(" ", "")
    assert planted.qid not in res["train"].qids
    assert ngram_dup.qid not in res["train"].qids
    assert not_dup.qid in res["train"].qids
    assert res["train"].dedup_removed["extra"] == 2
    assert (planted.qid, target.qid) in res["train"].dedup_pairs


def test_verify_detects_tampering(tmp_path: Path) -> None:
    res = build_splits(_pool(), CFG)
    write_splits(res, tmp_path)
    assert verify_all(tmp_path) == {"test_a": 20, "train": 80, "val": 25}
    path = tmp_path / "val.json"
    path.write_text(path.read_text(encoding="utf-8").replace("math_pool/", "math_pooI/", 1), encoding="utf-8")
    with pytest.raises(SplitsTamperedError):
        verify_all(tmp_path)


def test_stratify_raises_when_level_short() -> None:
    cfg = dict(CFG)
    cfg["splits"] = dict(CFG["splits"], val=[{"dataset": "math_pool", "n": 500, "stratify": "difficulty"}])
    with pytest.raises(ValueError, match="mức"):
        build_splits(_pool(), cfg)
