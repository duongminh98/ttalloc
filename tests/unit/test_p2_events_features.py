"""P2.T1, P2.T2: sự kiện kế tiếp, dãy chuẩn, đặc trưng."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from sim_helpers import synth_chains
from ttalloc.config import DecisionPointCfg
from ttalloc.features.events import LoggedChain, SimValidityError, canonical_points, next_event
from ttalloc.features.extract import FEATURE_GROUPS, FeatureState, feature_names
from ttalloc.types import ProbeResult, TraceKey


def _chain(gen_end: list[int], kinds: list[str], chain_len: int, natural: bool) -> LoggedChain:
    n = len(gen_end)
    code = {"start": 0, "marker": 1, "length": 2}
    kind = np.array([code[k] for k in kinds], np.int8)
    ge = np.array(gen_end, np.int64)
    midx = np.flatnonzero(kind == 1)
    z = np.zeros(n, np.int64)
    return LoggedChain(
        key=TraceKey("q", 0),
        chain_len=chain_len,
        natural_end=natural,
        end_reason="think_end" if natural else "max_len",
        nat_correct=True if natural else None,
        nat_prompt_tokens=2,
        nat_output_tokens=5,
        nat_extracted="1",
        gold="1",
        gen_end=ge,
        pos=ge - 1,
        kind=kind,
        marker_idx=midx,
        marker_ge=ge[midx],
        ans_correct=np.zeros(n, bool),
        ans_prompt=z + 3,
        ans_output=z + 10,
        ans_extracted=["x"] * n,
        probe_prompt=z + 6,
        probe_output=z + 5,
        conf_geo=np.linspace(0.1, 0.9, n),
        conf_arith=np.linspace(0.1, 0.9, n),
        probe_key=["k"] * n,
        probe_raw=["1"] * n,
        has_think_end=np.ones(n, bool),
        ans_tokens=z + 2,
        n_markers=z,
        local_conf_min=z * 1.0,
    )


C = _chain([0, 150, 300, 556, 700], ["start", "marker", "marker", "length", "marker"], chain_len=800, natural=True)


def test_natural_end_has_priority_over_later_marker() -> None:
    ev = next_event(C, 4, 700, 0, 200, True)
    assert ev.kind == "natural_end" and ev.gen_end == 800
    c2 = _chain([0, 150, 790], ["start", "marker", "marker"], chain_len=780, natural=True)
    assert next_event(c2, 1, 150, 0, 1000, True).kind == "natural_end"  # tự kết thúc trước từ chuyển ý


def test_min_tokens_strict() -> None:
    assert next_event(C, 0, 0, 149, 400, True).idx == 1  # 150 - 0 > 149
    assert next_event(C, 0, 0, 150, 400, True).idx == 2  # 150 - 0 không > 150


def test_length_rounds_down_to_recorded_point() -> None:
    ev = next_event(C, 2, 300, 300, 299, True)  # không có từ chuyển ý hợp lệ trong (300, 599]
    assert ev.kind == "length" and ev.idx == 3 and ev.gen_end == 556
    ev2 = next_event(C, 0, 0, 0, 400, False)  # không dừng ở từ chuyển ý: điểm đã ghi cuối cùng <= lim
    assert ev2.kind == "length" and ev2.idx == 2


def test_truncated_event() -> None:
    c = _chain([0, 200, 456], ["start", "marker", "length"], chain_len=500, natural=False)
    ev = next_event(c, 2, 456, 0, 256, True)
    assert ev.kind == "truncated" and ev.idx == 2 and ev.gen_end == 500


def test_sim_validity_error_when_points_missing() -> None:
    c = _chain([0, 200], ["start", "marker"], chain_len=5000, natural=True)
    with pytest.raises(SimValidityError):
        next_event(c, 1, 200, 0, 256, True)


def test_canonical_points_synth() -> None:
    dp = DecisionPointCfg()
    for c in synth_chains(10, [0, 1], seed=4).values():
        sub = canonical_points(c, dp)
        ge = [int(c.gen_end[i]) for i in sub]
        assert ge == sorted(ge) and (not ge or ge[0] <= dp.b_min + dp.l_max)
        assert all(b - a <= dp.l_max for a, b in itertools.pairwise(ge))


def _p(conf: float, key: str, raw: str, n_tok: int = 2) -> ProbeResult:
    return ProbeResult(TraceKey("q", 0), raw, key, conf, conf, (0.0,) * n_tok, 6, 4)


def test_features_hand_computed() -> None:
    fs = FeatureState(k=3)
    assert math.isnan(fs.vector()["conf_cur"])  # trước lần gợi đầu tiên
    fs.update(1000, _p(0.2, "str:a", "a"), 1, None)
    v = fs.vector()
    assert v["conf_slope_per_1k"] == 0.0 and v["has_slope"] == 0.0 and v["n_recent"] == 1 and v["consistency"] == 1
    fs.update(2000, _p(0.4, "str:b", "b"), 3, -0.5)
    fs.update(3000, _p(0.6, "str:b", "b"), 4, -0.7)
    fs.update(4000, _p(0.5, "str:c", "12", 3), 6, -0.9)
    v = fs.vector()
    # 3 lần gợi gần nhất: (2,0.4), (3,0.6), (4,0.5) theo đơn vị 1k token -> hệ số góc 0.05
    assert v["conf_slope_per_1k"] == pytest.approx(0.05)
    assert v["conf_cur"] == 0.5 and v["conf_max"] == 0.6 and v["n_probes"] == 4 and v["b"] == 4000
    assert v["consistency"] == pytest.approx(1 / 3) and v["n_recent"] == 3  # b, b, c so với c
    assert v["n_changes"] == 2  # a -> b, b -> c
    assert v["marker_rate_per_1k"] == pytest.approx(1.5)
    assert v["local_conf_min"] == -0.9 and v["ans_len"] == 3 and v["ans_type_integer"] == 1.0


def test_n_changes_and_equivalent_keys() -> None:
    fs = FeatureState(k=4)
    for raw in ["1/2", "0.5", "3"]:
        from ttalloc.data.grading import canonical_key

        fs.update(100, _p(0.5, canonical_key(raw), raw), 0, None)
    v = fs.vector()
    assert v["n_changes"] == 1  # 1/2 và 0.5 là cùng một đáp án
    assert v["consistency"] == pytest.approx(1 / 3)


def test_drop_groups_removes_columns() -> None:
    names = feature_names(["trend", "ans_type"])
    assert not set(FEATURE_GROUPS["trend"]) & set(names) and "ans_len" not in names and "conf_cur" in names
    with pytest.raises(ValueError):
        feature_names(["nope"])


def test_truncate_after_keeps_prefix() -> None:
    t = C.truncate_after(2)
    assert t.n_points == 3 and not t.natural_end and replace(t).chain_len == 301
