"""P1.T7, P1.T8, P1.T11: lưu trữ, logger, kiểm tra dữ liệu ghi."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from scripted_world import make_world
from synth import make_logs
from ttalloc.config import ExperimentCfg
from ttalloc.logging.logger import run_logger
from ttalloc.logging.schema import POINTS_SCHEMA, TRACES_SCHEMA
from ttalloc.logging.store import LogHashMismatch, LogStore
from ttalloc.logging.validate import validate_logs
from ttalloc.types import TraceKey


def _rows(table: pa.Table) -> list[dict[str, Any]]:
    return table.to_pylist()


def test_atomic_write(tmp_path: Path) -> None:
    tr, pt = make_logs(3, [0], np.random.default_rng(0))
    store = LogStore(tmp_path / "s")
    store.init("h1", {})
    store.write_part(_rows(tr), _rows(pt))
    assert not list((tmp_path / "s").rglob("*.tmp"))
    assert store.read_traces().num_rows == 3
    # Phần ghi dở (không có trong tệp kê khai) bị bỏ qua khi đọc.
    (tmp_path / "s" / "traces" / "part-00001.parquet").write_bytes(b"rac")
    assert store.read_traces().num_rows == 3
    with pytest.raises(LogHashMismatch):
        store.init("h2", {})


def test_completed_keys_after_partial_run(tmp_path: Path) -> None:
    tr, pt = make_logs(4, [0, 1], np.random.default_rng(1))
    rows_t, rows_p = _rows(tr), _rows(pt)
    store = LogStore(tmp_path / "s")
    first = [r for r in rows_t if r["qid"] in ("synth/0", "synth/1")]
    store.write_part(first, [p for p in rows_p if p["qid"] in ("synth/0", "synth/1")])
    with store.manifest_path.open("a", encoding="utf-8") as f:
        f.write('{"part": 9, "file": "x", "ke')  # dòng ghi dở khi bị ngắt
    assert store.completed_keys() == {TraceKey(f"synth/{i}", s) for i in (0, 1) for s in (0, 1)}
    with pytest.raises(ValueError, match="đã ghi"):
        store.write_part(first[:1], [])


@pytest.fixture()
def world(qwen3_tokenizer: Any) -> Any:
    return make_world(qwen3_tokenizer, n_questions=4)


def _log(world: Any, store: LogStore, keys: list[TraceKey], cfg: ExperimentCfg | None = None) -> dict[str, Any]:
    cfg = cfg or world.cfg
    src = world.source(cfg=cfg)
    gold = {q: v.gold for q, v in world.questions.items()}
    return run_logger(src, keys, store, cfg, "qwen3_1_7b", "val", "LH", gold)


def test_logger_end_to_end(world: Any, tmp_path: Path) -> None:
    store = LogStore(tmp_path / "logs")
    store.init("LH", {})
    keys = [TraceKey(q, 0) for q in world.questions]
    stats = _log(world, store, keys)
    tr, pt = store.read_traces(), store.read_points()
    assert tr.num_rows == 4 and stats["chains"] == 4 and stats["max_len"] == 1
    rep = validate_logs(tr, pt, world.cfg, "LH", world.codec.think_end_id)
    assert rep["ok"], rep
    t = {r["qid"]: r for r in tr.to_pylist()}
    for qid, chain in world.chains.items():
        if world.codec.think_end_id in chain:
            assert t[qid]["chain_ids"] == chain and t[qid]["end_reason"] == "think_end" and t[qid]["natural_end"]
        else:
            assert t[qid]["chain_len"] == world.cfg.log.l_log and t[qid]["end_reason"] == "max_len"


def test_start_point_present(world: Any, tmp_path: Path) -> None:
    store = LogStore(tmp_path / "logs")
    _log(world, store, [TraceKey("w/0", 0)])
    pts = [p for p in store.read_points().to_pylist() if p["point_idx"] == 0]
    assert len(pts) == 1 and pts[0]["kind"] == "start" and pts[0]["pos"] == 0 and pts[0]["probe_answer_raw"] is None


def test_logger_resume_no_duplicates(world: Any, tmp_path: Path) -> None:
    store = LogStore(tmp_path / "logs")
    keys = [TraceKey(q, s) for q in world.questions for s in (0, 1)]
    _log(world, store, keys[:3])  # giả lập dừng giữa chừng: mới ghi được 3 chuỗi
    done = store.completed_keys()
    stats = _log(world, store, keys)
    assert stats["skipped"] == len(done)
    tr = store.read_traces().to_pylist()
    assert len(tr) == len(keys) and len({(r["qid"], r["seed"]) for r in tr}) == len(keys)


def _bad(world: Any, tmp_path: Path) -> tuple[pa.Table, pa.Table]:
    store = LogStore(tmp_path / "logs")
    _log(world, store, [TraceKey(q, 0) for q in world.questions])
    return store.read_traces(), store.read_points()


def _mutate(table: pa.Table, schema: pa.Schema, fn: Any) -> pa.Table:
    rows = table.to_pylist()
    fn(rows)
    return pa.Table.from_pylist(rows, schema=schema)


@pytest.mark.parametrize(
    ("which", "error", "fn"),
    [
        ("traces", "chain_len", lambda rows: rows[0].update(chain_len=rows[0]["chain_len"] + 1)),
        ("traces", "log_hash", lambda rows: rows[0].update(log_hash="khac")),
        ("traces", "end_reason", lambda rows: rows[0].update(end_reason="max_len")),
        ("points", "start_point", lambda rows: rows.pop(0)),
        ("points", "pos_after_gen_end", lambda rows: rows[1].update(pos=rows[1]["gen_end"] + 1)),
        ("points", "not_sorted_by_gen_end", lambda rows: rows[2].update(gen_end=rows[1]["gen_end"])),
        ("points", "gap_over_l_max", lambda rows: rows[1].update(gen_end=rows[1]["gen_end"] + 100)),
        ("points", "probe_token_count", lambda rows: rows[1].update(probe_output_tokens=99)),
        ("points", "marker_too_close", lambda rows: rows[1].update(gen_end=5, pos=4)),
    ],
)
def test_validate_detects_injected_errors(world: Any, tmp_path: Path, which: str, error: str, fn: Any) -> None:
    tr, pt = _bad(world, tmp_path)
    if which == "traces":
        tr = _mutate(tr, TRACES_SCHEMA, fn)
    else:
        pt = _mutate(pt, POINTS_SCHEMA, fn)
    rep = validate_logs(tr, pt, world.cfg, "LH", world.codec.think_end_id)
    assert not rep["ok"] and error in rep["errors"], rep["errors"]
    assert rep["pass_rate"] < 1.0


def test_validate_synth_passes() -> None:
    tr, pt = make_logs(20, [0, 1], np.random.default_rng(3))
    cfg = ExperimentCfg.model_validate({"log": {"l_log": 4096}})
    rep = validate_logs(tr, pt, cfg, "synth")
    assert rep["ok"], rep["errors"]
    assert 0 < rep["max_len_rate"] < 1
