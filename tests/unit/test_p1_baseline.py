"""P1.T5, P1.T6: baseline (phần chạy được trên CPU) và báo cáo kiểm định tương đương."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripted_world import GOLD, make_world
from ttalloc.baseline.deer_official import parse_deer_output, write_dataset
from ttalloc.baseline.equivalence import equivalence_report, segmented_rows
from ttalloc.baseline.full import ApprovalRequired, Sampling, generate_full, require_approval, summarize
from ttalloc.engine.generator import TokenCodec
from ttalloc.types import Question


def _qs() -> list[Question]:
    return [
        Question("t/0", "t", "What is 1+1?", None, "2", "math", None),
        Question("t/1", "t", "Compute 3*3.", None, "9", "math", None),
        Question("t/2", "t", "Long one?", None, "5", "math", None),
    ]


def test_deer_output_parser(fixtures_dir: Path, qwen3_tokenizer: Any) -> None:
    lines = (fixtures_dir / "deer_output.jsonl").read_text(encoding="utf-8").splitlines()
    rows = parse_deer_output(lines, _qs(), TokenCodec.from_tokenizer(qwen3_tokenizer), seed=0)
    by = {r["qid"]: r for r in rows}
    assert by["t/0"]["extracted"] == "2" and by["t/0"]["correct"]  # lấy khung sau </think> cuối cùng
    assert by["t/1"]["extracted"] == "6" and not by["t/1"]["correct"]
    assert by["t/2"]["extracted"] is None and by["t/2"]["too_long"] == 1
    assert all(r["n_tokens"] > 0 for r in rows)


def test_deer_dataset_has_problem_and_answer(tmp_path: Path) -> None:
    p = write_dataset(_qs(), tmp_path, "x")
    first = p.read_text(encoding="utf-8").splitlines()[0]
    assert '"problem": "What is 1+1?"' in first and '"answer": "2"' in first


def test_baseline_refuses_test_split_without_approval(tmp_path: Path) -> None:
    log = tmp_path / "test-log.md"
    with pytest.raises(ApprovalRequired):
        require_approval("test_main", None, "baseline full", log_path=log)
    require_approval("val", None, "baseline full", log_path=log)  # tập không phải kiểm tra: không cần duyệt
    assert not log.exists()
    require_approval("test_main", "Minh", "baseline full", "profile=deer_paper", log_path=log)
    assert "baseline full | test_main | Minh" in log.read_text(encoding="utf-8")


def test_generate_full_scripted(qwen3_tokenizer: Any) -> None:
    w = make_world(qwen3_tokenizer, n_questions=3, max_len_q=None)

    def cont(prompt: tuple[int, ...]) -> list[int]:
        qid = w._which(prompt)
        tail = [*w.codec.encode("\n\nSo the answer is \\boxed{" + GOLD + "}."), w.codec.eos_ids[0]]
        return (w.chains[qid] + tail)[len(prompt) - len(w.prompt_ids[qid]) :]

    from ttalloc.engine.scripted import ScriptedGenerator

    gen = ScriptedGenerator(cont, w.codec)
    rows = generate_full(gen, w.codec, list(w.questions.values()), w.prompt_ids, [0, 1], Sampling(0.6, 0.95, 20, 5000))
    assert len(rows) == 6 and all(r["correct"] and r["natural_end"] for r in rows)
    assert all(r["think_tokens"] == len(w.chains[r["qid"]]) for r in rows)
    s = summarize(rows, 1.0)
    assert s["accuracy"] == 1.0 and s["by_dataset"]["w"]["n"] == 6


def test_segmented_rows_scripted(qwen3_tokenizer: Any) -> None:
    w = make_world(qwen3_tokenizer, n_questions=3)
    rows = segmented_rows(w.source(), list(w.questions.values()), [0], w.cfg)
    by = {r["qid"]: r for r in rows}
    assert by["w/1"]["natural_end"] is False and by["w/1"]["think_tokens"] == w.cfg.log.l_log
    assert by["w/0"]["think_tokens"] == len(w.chains["w/0"]) and by["w/0"]["n_probes"] > 0


def test_equivalence_report() -> None:
    rng = np.random.default_rng(0)
    n = 400
    acc = rng.random(n) < 0.6
    ln = rng.integers(1000, 5000, n)
    full = pd.DataFrame({"qid": [f"q{i}" for i in range(n)], "seed": 0, "correct": acc, "think_tokens": ln})
    same = equivalence_report(full, full.copy(), rng, n_boot=300)
    assert same["pass"] and same["acc_diff"] == 0
    worse = full.assign(correct=False)
    assert not equivalence_report(full, worse, rng, n_boot=300)["acc_ok"]
    longer = full.assign(think_tokens=full.think_tokens * 1.2)
    assert not equivalence_report(full, longer, rng, n_boot=300)["len_ok"]
