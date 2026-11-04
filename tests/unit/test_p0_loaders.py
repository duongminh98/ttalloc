"""P0.T6: bộ nạp dữ liệu, dùng tệp mẫu offline trong tests/fixtures/datasets (dòng thô lấy từ Hugging Face)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from ttalloc.config import load_dataset_cfg
from ttalloc.data.loaders import parse_difficulty, rows_to_questions
from ttalloc.types import Question

DATASETS = [
    "math_train",
    "deepscaler",
    "amc23",
    "math500",
    "aime_2024",
    "aime_2025",
    "aime_2026",
    "hmmt_feb_2026",
    "gpqa_diamond",
]


def _load_fixture(fixtures_dir: Path, name: str) -> list[Question]:
    cfg = load_dataset_cfg(name)
    groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for line in (fixtures_dir / "datasets" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        groups[row.pop("__config", cfg.hf_configs[0])].append(row)
    out: list[Question] = []
    for subset, rows in groups.items():
        out.extend(rows_to_questions(cfg, subset, rows))
    return out


@pytest.mark.parametrize("name", DATASETS)
def test_loader_fields(fixtures_dir: Path, name: str) -> None:
    qs = _load_fixture(fixtures_dir, name)
    assert qs
    cfg = load_dataset_cfg(name)
    for q in qs:
        assert q.qid.startswith(f"{name}/")
        assert q.dataset == name and q.problem and q.gold
        assert q.answer_type == cfg.answer_type
        if cfg.answer_type == "choice":
            assert q.options is not None and len(q.options) == 4 and q.gold in "ABCD"
        else:
            assert q.options is None
        if name in ("math_train", "math500"):
            assert q.difficulty in (1, 2, 3, 4, 5)
        else:
            assert q.difficulty is None


def test_loader_specific_values(fixtures_dir: Path) -> None:
    by_name = {n: _load_fixture(fixtures_dir, n) for n in ["math_train", "aime_2025", "math500", "gpqa_diamond"]}
    assert by_name["math_train"][0].qid == "math_train/algebra/0"
    assert by_name["math_train"][0].gold == "0"  # đáp án lấy từ \boxed{} cuối của lời giải
    assert by_name["aime_2025"][0].qid == "aime_2025/1" and by_name["aime_2025"][0].gold == "70"
    assert by_name["math500"][0].qid.startswith("math500/test/")
    g = by_name["gpqa_diamond"][0]
    assert g.options is not None and g.options["ABCD".index(g.gold)] == "Photon"


def test_qid_stable(fixtures_dir: Path) -> None:
    for name in DATASETS:
        first = [q.qid for q in _load_fixture(fixtures_dir, name)]
        second = [q.qid for q in _load_fixture(fixtures_dir, name)]
        assert first == second and len(set(first)) == len(first)


@pytest.mark.parametrize(("raw", "level"), [("Level 3", 3), (5, 5), ("Level ?", None), (None, None), ("Level 9", None)])
def test_parse_difficulty(raw: object, level: int | None) -> None:
    assert parse_difficulty(raw) == level


@pytest.mark.net
@pytest.mark.parametrize("name", ["aime_2025", "amc23"])
def test_load_questions_from_hub(name: str) -> None:
    from ttalloc.data.loaders import load_questions

    qs = load_questions(name)
    assert len(qs) in (30, 40)
