"""P0.T10: công cụ kiểm tra chấm điểm."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq
from typer.testing import CliRunner

from ttalloc.cli import app
from ttalloc.data.audit import AUDIT_COLUMNS
from ttalloc.logging.schema import ANSWERS_SCHEMA, to_table


def _synthetic_run(tmp_path: Path, n: int) -> Path:
    run = tmp_path / "20261103-000000_baseline_abc"
    run.mkdir()
    rows = [
        {"qid": f"synth/{i}", "seed": i % 2, "problem": f"Đề {i}", "gold": str(i), "answer_text": f"\\boxed{{{i}}}",
         "extracted": str(i), "correct": i % 3 != 0}
        for i in range(n)
    ]
    pq.write_table(to_table(rows, ANSWERS_SCHEMA), run / "answers.parquet")
    return run


def test_audit_csv_columns_and_rows(tmp_path: Path) -> None:
    run = _synthetic_run(tmp_path, 30)
    result = CliRunner().invoke(app, ["audit-grading", "--run", str(run), "--n", "10"])
    assert result.exit_code == 0, result.output
    with (run / "audit_grading.csv").open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 10
    assert list(rows[0].keys()) == AUDIT_COLUMNS
    assert len({r["qid"] for r in rows}) == 10


def test_audit_fewer_rows_than_requested(tmp_path: Path) -> None:
    run = _synthetic_run(tmp_path, 4)
    out = tmp_path / "audit.csv"
    result = CliRunner().invoke(app, ["audit-grading", "--run", str(run), "--n", "100", "--out", str(out)])
    assert result.exit_code == 0, result.output
    with out.open(encoding="utf-8-sig") as fh:
        assert len(list(csv.DictReader(fh))) == 4
