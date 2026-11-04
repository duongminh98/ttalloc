"""P2.T3–T6 và P3.T3 (phần bộ ước lượng): bộ dữ liệu, huấn luyện, hiệu chuẩn, rho, đánh giá."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim_helpers import synth_cfg, synth_chains
from ttalloc.config import DecisionPointCfg, EstimatorCfg
from ttalloc.estimator.dataset import build_dataset, label_at, samples_for_chain
from ttalloc.estimator.evaluate import auroc, ece, evaluate
from ttalloc.estimator.rho import RHO_REPORT_SCALE, estimate_omega, rho_hstar
from ttalloc.estimator.train import Estimator, group_folds, train_estimator
from ttalloc.features.events import canonical_points
from ttalloc.features.extract import feature_names

CFG = synth_cfg()


@pytest.fixture(scope="module")
def chains() -> dict:  # type: ignore[type-arg]
    return synth_chains(60, [0, 1, 2], seed=11)


@pytest.fixture(scope="module")
def data(chains: dict) -> pd.DataFrame:  # type: ignore[type-arg]
    return build_dataset(chains, CFG.dp, CFG.features, CFG.estimator.h_train)


def test_labels(chains: dict) -> None:  # type: ignore[type-arg]
    dp = DecisionPointCfg()
    nat = next(c for c in chains.values() if c.natural_end and len(canonical_points(c, dp)) > 2)
    sub = canonical_points(nat, dp)
    assert label_at(nat, sub, 0, 0) == bool(nat.ans_correct[sub[0]])
    big = nat.chain_len  # tự kết thúc trước lim: nhãn là câu trả lời tự nhiên
    assert label_at(nat, sub, 0, big) == bool(nat.nat_correct)
    h = int(nat.gen_end[sub[1]] - nat.gen_end[sub[0]])
    assert label_at(nat, sub, 0, h) == bool(nat.ans_correct[sub[1]])
    trunc = next(c for c in chains.values() if not c.natural_end and canonical_points(c, dp))
    st = canonical_points(trunc, dp)
    assert label_at(trunc, st, 0, 10**6) == bool(trunc.ans_correct[st[-1]])  # chạm L_log: điểm cuối của dãy chuẩn


def test_weights_sum_to_one_per_chain(data: pd.DataFrame) -> None:
    w = data.groupby(["qid", "seed"]).w.sum()
    assert np.allclose(w.to_numpy(), 1.0)
    assert set(data.h.unique()) == {0, *CFG.estimator.h_train}


def test_group_folds_disjoint(data: pd.DataFrame) -> None:
    for tr, te in group_folds(data.qid.to_numpy(), 5):
        assert not set(data.qid.iloc[tr]) & set(data.qid.iloc[te])  # I5


@pytest.fixture(scope="module")
def trained(data: pd.DataFrame) -> tuple[Estimator, pd.DataFrame]:
    return train_estimator(data, feature_names(), CFG.estimator)


def test_predictions_in_unit_interval(trained: tuple[Estimator, pd.DataFrame], data: pd.DataFrame) -> None:
    est, oof = trained
    p = est.predict(data[est.features].head(200), est.hs)
    assert p.shape == (200, len(est.hs)) and (p >= 0).all() and (p <= 1).all()
    assert oof.p.between(0, 1).all()


def test_isotonic_monotone(trained: tuple[Estimator, pd.DataFrame]) -> None:
    est, _ = trained
    assert est.isotonic is not None
    x = np.linspace(0, 1, 50)
    for iso in est.isotonic.values():
        assert np.all(np.diff(iso.predict(x)) >= -1e-12)


def test_reproducible_with_seed(data: pd.DataFrame, trained: tuple[Estimator, pd.DataFrame]) -> None:
    est2, oof2 = train_estimator(data, feature_names(), CFG.estimator)
    np.testing.assert_allclose(trained[1].p.to_numpy(), oof2.p.to_numpy())
    X = data[est2.features].head(50)
    np.testing.assert_allclose(trained[0].predict(X, [0, 512]), est2.predict(X, [0, 512]))


def test_artifact_roundtrip(trained: tuple[Estimator, pd.DataFrame], data: pd.DataFrame, tmp_path: Path) -> None:
    est, _ = trained
    est.save(tmp_path / "e")
    back = Estimator.load(tmp_path / "e")
    X = data[est.features].head(30)
    np.testing.assert_allclose(est.predict(X, est.hs), back.predict(X, est.hs))
    with pytest.raises(ValueError):
        back.predict(X, [123])


def test_uncalibrated_estimator_skips_isotonic(data: pd.DataFrame) -> None:
    est, _ = train_estimator(data, feature_names(), CFG.estimator.model_copy(update={"calibrate": False}))
    assert est.isotonic is None
    assert est.predict(data[est.features].head(5), [0]).shape == (5, 1)


def test_logreg_estimator_same_api(data: pd.DataFrame, tmp_path: Path) -> None:
    cfg = EstimatorCfg.model_validate({**CFG.estimator.model_dump(), "kind": "logreg"})
    est, _oof = train_estimator(data, feature_names(["local_conf"]), cfg)
    est.save(tmp_path / "lr")
    back = Estimator.load(tmp_path / "lr")
    X = data[est.features].head(20)
    p = back.predict(X, est.hs)
    assert p.shape == (20, len(est.hs)) and (p >= 0).all() and (p <= 1).all()


def test_rho_worked_example() -> None:
    # Báo cáo phương pháp mục 9: H = {1k, 2k, 4k}, omega = 0. Đơn vị báo cáo: điểm phần trăm trên 1.000 token.
    rows = {
        1: ({0: 0.92, 1000: 0.93, 2000: 0.93, 4000: 0.93}, 1.0, 1000),
        2: ({0: 0.40, 1000: 0.55, 2000: 0.85, 4000: 0.88}, 22.5, 2000),
        3: ({0: 0.10, 1000: 0.12, 2000: 0.15, 4000: 0.50}, 10.0, 4000),
        4: ({0: 0.05, 1000: 0.05, 2000: 0.06, 4000: 0.06}, 0.5, 2000),
    }
    for q, exp_rho, exp_h in rows.values():
        rho, h = rho_hstar(q, 1000, [1000, 2000, 4000], b_max=10**9, omega=0.0)
        assert rho * RHO_REPORT_SCALE == pytest.approx(exp_rho) and h == exp_h


def test_rho_edge_cases() -> None:
    q = {0: 0.9, 512: 0.8, 1024: 0.7}
    rho, _ = rho_hstar(q, 0, [512, 1024], 10**6, 0.0)
    assert rho < 0
    rho, h = rho_hstar({0: 0.1, 512: 0.2, 1024: 0.9}, 1000, [512, 1024], b_max=1600, omega=0.0)
    assert h == 512  # tầm 1024 bị lọc vì vượt b_max - b
    assert rho_hstar(q, 2000, [512, 1024], b_max=2100, omega=0.0) == (-math.inf, None)
    r0, _ = rho_hstar({0: 0.1, 512: 0.3}, 0, [512], 10**6, 0.0)
    r1, _ = rho_hstar({0: 0.1, 512: 0.3}, 0, [512], 10**6, 0.1)
    assert r1 == pytest.approx(r0 / 1.1)


def test_estimate_omega(chains: dict) -> None:  # type: ignore[type-arg]
    om = estimate_omega(chains.values(), DecisionPointCfg())
    assert 0 < om < 0.5


def test_metrics_basic() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert ece(y, p) < 0.02  # dữ liệu đã hiệu chuẩn hoàn hảo
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_evaluate_report(trained: tuple[Estimator, pd.DataFrame], chains: dict) -> None:  # type: ignore[type-arg]
    from ttalloc.features.curve import EmpiricalCurves

    _, oof = trained
    gen_end = {(k.qid, k.seed, i): int(c.gen_end[i]) for k, c in chains.items() for i in range(c.n_points)}
    rep = evaluate(oof, EmpiricalCurves(chains, DecisionPointCfg()), gen_end, [0, *CFG.estimator.h_train])
    assert set(rep["per_h"]) == {str(h) for h in [0, *CFG.estimator.h_train]}
    assert 0.5 < rep["auroc_q0"] <= 1 and "pooled" in rep["gain_spearman"]


def test_samples_skip_chain_without_points() -> None:
    c = next(iter(synth_chains(1, [0], seed=2).values()))
    assert samples_for_chain(c, DecisionPointCfg(b_min=10**6), CFG.features, [256], sub=[]) == []
