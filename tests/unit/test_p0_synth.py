"""P0.T5: dữ liệu tổng hợp."""

from __future__ import annotations

import itertools

import numpy as np

from synth import make_logs
from ttalloc.logging.schema import POINTS_SCHEMA, TRACES_SCHEMA, validate_table


def test_synth_conforms_to_schema() -> None:
    traces, points = make_logs(6, range(4), np.random.default_rng(1), l_log=4096, log_m=128, log_l_max=256)
    validate_table(traces, TRACES_SCHEMA)
    validate_table(points, POINTS_SCHEMA)
    t = traces.to_pylist()
    p = points.to_pylist()
    # Đủ các trường hợp: tự kết thúc, chạm L_log, đoạn dài không có từ chuyển ý.
    assert any(r["natural_end"] for r in t)
    assert any(r["end_reason"] == "max_len" and r["chain_len"] == 4096 for r in t)
    assert any(r["kind"] == "length" for r in p)
    by_chain: dict[tuple[str, int], list[dict[str, object]]] = {}
    for r in p:
        by_chain.setdefault((str(r["qid"]), int(r["seed"])), []).append(r)  # type: ignore[call-overload]
    lengths = {(r["qid"], r["seed"]): r["chain_len"] for r in t}
    assert set(by_chain) == set(lengths)
    for key, rows in by_chain.items():
        assert rows[0]["kind"] == "start" and rows[0]["pos"] == 0 and rows[0]["gen_end"] == 0
        ends = [int(r["gen_end"]) for r in rows]  # type: ignore[call-overload]
        assert ends == sorted(ends)
        assert [r["point_idx"] for r in rows] == list(range(len(rows)))
        for a, b in itertools.pairwise(ends):
            assert 128 < b - a <= 256
        assert all(int(r["pos"]) <= int(r["gen_end"]) for r in rows)  # type: ignore[call-overload]
        assert int(lengths[key]) - ends[-1] <= 256  # type: ignore[call-overload]


def test_synth_deterministic() -> None:
    a = make_logs(4, [0, 1], np.random.default_rng(7))
    b = make_logs(4, [0, 1], np.random.default_rng(7))
    c = make_logs(4, [0, 1], np.random.default_rng(8))
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    assert not a[1].equals(c[1])
