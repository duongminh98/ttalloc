"""P0.T4: kiểu dữ liệu và schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from synth import make_logs
from ttalloc.logging.schema import (
    ANSWERS_SCHEMA,
    PER_QUESTION_SCHEMA,
    POINTS_SCHEMA,
    TIMING_SCHEMA,
    TRACES_SCHEMA,
    SchemaError,
    to_table,
    validate_table,
)
from ttalloc.types import Costs


def test_costs_total_is_sum_of_five() -> None:
    c = Costs(think=100, probe_prompt=7, probe_output=11, answer_prompt=3, answer_output=40)
    assert c.total == 161
    c.add(Costs(think=1, probe_prompt=1, probe_output=1, answer_prompt=1, answer_output=1))
    assert c.total == 166
    assert sum(c.as_dict().values()) == c.total


def test_schema_roundtrip(tmp_path: Path) -> None:
    traces, points = make_logs(3, [0, 1], np.random.default_rng(0))
    per_q = to_table(
        [
            {"method": "M2", "alpha": 0.5, "draw": 0, "qid": "synth/0", "seed": 0, "correct": True, "think": 10,
             "probe_prompt": 1, "probe_output": 2, "answer_prompt": 3, "answer_output": 4, "stop_point_idx": 2,
             "stop_kind": "marker", "n_probes": 2, "n_rounds": 3}
        ],
        PER_QUESTION_SCHEMA,
    )
    timing = to_table(
        [{"round": 0, "t_advance": 1.0, "t_probe": 0.1, "t_answer": 0.0, "t_controller": 0.01, "n_active": 8,
          "prefix_cache_hit_rate": None}],
        TIMING_SCHEMA,
    )
    answers = to_table(
        [{"qid": "q", "seed": 0, "problem": "1+1?", "gold": "2", "answer_text": "\\boxed{2}", "extracted": "2",
          "correct": True}],
        ANSWERS_SCHEMA,
    )
    for name, table, schema in [
        ("traces", traces, TRACES_SCHEMA),
        ("points", points, POINTS_SCHEMA),
        ("per_question", per_q, PER_QUESTION_SCHEMA),
        ("timing", timing, TIMING_SCHEMA),
        ("answers", answers, ANSWERS_SCHEMA),
    ]:
        path = tmp_path / f"{name}.parquet"
        pq.write_table(table, path)
        back = pq.read_table(path)
        validate_table(back, schema)
        assert back.equals(table)


def test_to_table_rejects_missing_and_extra_fields() -> None:
    with pytest.raises(SchemaError):
        to_table([{"round": 0}], TIMING_SCHEMA)
    with pytest.raises(SchemaError):
        to_table(
            [{"round": 0, "t_advance": 1.0, "t_probe": 0.1, "t_answer": 0.0, "t_controller": 0.0, "n_active": 1,
              "bogus": 1}],
            TIMING_SCHEMA,
        )
