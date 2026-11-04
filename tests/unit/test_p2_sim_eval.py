"""P2.T9–T10, P2.T13–T19, P3.T3–T5: nguồn đã ghi, RhoProvider, lô, chạy mô phỏng, thống kê, cổng, khóa config."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim_helpers import C_ANS_MAX, C_PROBE_MAX, synth_cfg, synth_chains, synth_env, synth_questions
from synth import make_logs
from ttalloc.config import BatchCfg, DecisionPointCfg, MethodCfg, load_config
from ttalloc.data.batches import build_batch, level_counts
from ttalloc.estimator.dataset import build_dataset
from ttalloc.estimator.train import train_estimator
from ttalloc.eval.gates import gate_g1, gate_g2, gate_g3, gate_markdown
from ttalloc.eval.metrics import auc_over_alpha, cell_table, tokens_saved_vs
from ttalloc.eval.report import build_report
from ttalloc.eval.stats import acceptance_band, mcnemar_test, paired_bootstrap
from ttalloc.features.curve import EmpiricalCurves
from ttalloc.features.extract import FeatureState, feature_names
from ttalloc.locking import NotLocked, check_locked, lock, record_eval
from ttalloc.logging.store import LogHashMismatch, LogStore
from ttalloc.sim.env import SimEnv
from ttalloc.sim.logged_source import LoggedSource, load_logged
from ttalloc.sim.providers import CurveRhoProvider, EstimatorRhoProvider, TableRhoProvider
from ttalloc.sim.run import run_experiment
from ttalloc.sim.table import build_table, canonical_paths
from ttalloc.sim.tune import calibrate_b2, random_search
from ttalloc.types import AdvanceRequest, Question, QuestionState, TraceKey

DP = DecisionPointCfg()
B2 = MethodCfg(id="B2", policy="deer", params={"lambda": 0.9})
M2 = MethodCfg(id="M2", policy="price_adaptive", params={"mu_min": 0.0})
M2_EST = MethodCfg(id="M2", policy="price_adaptive", rho_provider="estimator", params={"mu_min": 0.0})
M3 = MethodCfg(id="M3", policy="price_adaptive", rho_provider="curve", params={"mu_min": 0.0})
ALL = BatchCfg(composition="all")


# ---------------------------------------------------------------- LoggedSource (P2.T9)


def _chain_with_points(min_points: int = 6) -> tuple[TraceKey, object]:
    chains = synth_chains(8, [0, 1], seed=21)
    return next((k, c) for k, c in chains.items() if c.n_points >= min_points)


def test_answer_at_start_point() -> None:
    k, c = _chain_with_points()
    src = LoggedSource({k: c})  # type: ignore[dict-item]
    src.start([k])
    (a,) = src.answer([k])
    assert a.correct == bool(c.ans_correct[0]) and a.prompt_tokens == c.ans_prompt[0]  # type: ignore[attr-defined]
    assert k not in src.st


def _walk(src: LoggedSource, k: TraceKey, n: int) -> list[int]:
    seen = []
    for _ in range(n):
        (r,) = src.advance([AdvanceRequest(k, 0, 256, True)])
        if r.kind not in ("marker", "length"):
            break
        src.probe([k])
        seen.append(int(r.point_idx))  # type: ignore[arg-type]
    return seen


def test_on_stop_best_probe_and_vote() -> None:
    k, c = _chain_with_points()
    src = LoggedSource({k: c}, on_stop="best_probe")  # type: ignore[dict-item]
    src.start([k])
    seen = _walk(src, k, 4)
    assert seen
    (a,) = src.answer([k])
    best = max(seen, key=lambda j: (c.conf_geo[j], j))  # type: ignore[attr-defined]
    assert a.extracted == c.probe_raw[best] and a.prompt_tokens == a.output_tokens == 0  # type: ignore[attr-defined]
    src = LoggedSource({k: c}, on_stop="vote")  # type: ignore[dict-item]
    src.start([k])
    seen = _walk(src, k, 4)
    (a,) = src.answer([k])
    w: dict[str, float] = {}
    for j in seen:
        w[c.probe_key[j]] = w.get(c.probe_key[j], 0.0) + c.conf_geo[j]  # type: ignore[attr-defined]
    win = max(w, key=lambda kk: (w[kk], kk))
    assert a.extracted in {c.probe_raw[j] for j in seen if c.probe_key[j] == win}  # type: ignore[attr-defined]
    src = LoggedSource({k: c}, on_stop="vote")  # type: ignore[dict-item]
    src.start([k])
    (a,) = src.answer([k])  # chưa gợi lần nào: quay về câu trả lời ép buộc
    assert a.prompt_tokens == c.ans_prompt[0]  # type: ignore[attr-defined]


def test_logged_source_refuses_hash_mismatch(tmp_path: Path) -> None:
    tr, pt = make_logs(3, [0], np.random.default_rng(0))
    store = LogStore(tmp_path / "synth" / "val")
    store.init("abc", {"c_ans_max": C_ANS_MAX})
    store.write_part(tr.to_pylist(), pt.to_pylist())
    chains, meta = load_logged("synth", "val", "abc", logs_dir=tmp_path)
    assert len(chains) == 3 and meta["c_ans_max"] == C_ANS_MAX
    with pytest.raises(LogHashMismatch):
        load_logged("synth", "val", "xyz", logs_dir=tmp_path)


def test_logged_source_rejects_advance_after_end() -> None:
    k, c = _chain_with_points()
    src = LoggedSource({k: c})  # type: ignore[dict-item]
    src.start([k])
    (r,) = src.advance([AdvanceRequest(k, 0, 10**6, False)])
    assert r.kind in ("natural_end", "eos", "truncated")
    with pytest.raises(ValueError):
        src.advance([AdvanceRequest(k, 0, 256, True)])


# ---------------------------------------------------------------- RhoProvider (P2.T10)


@pytest.fixture(scope="module")
def est_env() -> SimEnv:
    cfg = synth_cfg()
    train = synth_chains(40, [0, 1, 2], seed=5)
    est, _ = train_estimator(
        build_dataset(train, cfg.dp, cfg.features, cfg.estimator.h_train), feature_names(), cfg.estimator
    )
    chains = synth_chains(12, [0, 1, 2], seed=6)
    paths = canonical_paths(chains, cfg.dp)
    table = build_table(chains, cfg.dp, cfg.features, est, paths)
    return SimEnv(
        cfg=cfg,
        split="val",
        chains=chains,
        questions=synth_questions(12),
        c_ans_max=C_ANS_MAX,
        c_probe_max=C_PROBE_MAX,
        ref_median=float(np.median([c.chain_len for c in chains.values()])),
        estimator=est,
        table=table,
        paths=paths,
    )


def test_table_equals_estimator_on_canonical_path(est_env: SimEnv) -> None:
    keys = est_env.draw_keys(sorted(est_env.questions), est_env.draw_rng(0))
    budget = 10**7
    r_tab = est_env.run(M2, keys, budget)
    r_est = est_env.run(M2_EST, keys, budget)
    sig = [(o.key, o.correct, o.costs.as_dict(), o.stop_point_idx) for o in r_tab.outcomes]
    assert sig == [(o.key, o.correct, o.costs.as_dict(), o.stop_point_idx) for o in r_est.outcomes]
    prov = est_env.provider(M2, est_env.b_max_for("price_adaptive", {}))
    assert isinstance(prov, TableRhoProvider)


def test_table_falls_back_when_path_differs(est_env: SimEnv) -> None:
    b_max = est_env.b_max_for("price_adaptive", {})
    est = EstimatorRhoProvider(est_env.estimator, est_env.cfg.policy.horizons, b_max, 0.0)  # type: ignore[arg-type]
    tab = TableRhoProvider(est_env.table, est_env.paths, est)  # type: ignore[arg-type]
    key, path = next((k, p) for k, p in est_env.paths.items() if len(p) >= 2)  # type: ignore[union-attr]
    c = est_env.chains[key]
    on = QuestionState(key, gen_end=int(c.gen_end[path[0]]), visited=[path[0]], features=FeatureState(k=4))
    on.features.update(on.gen_end, c.probe(path[0]), int(c.n_markers[path[0]]), None)
    off = QuestionState(key, gen_end=int(c.gen_end[path[1]]), visited=[path[1]], features=FeatureState(k=4))
    tab.fill([on, off])
    assert tab.n_table == 1 and tab.n_fallback == 1
    ref = QuestionState(key, gen_end=off.gen_end, visited=list(off.visited), features=off.features)
    est.fill([ref])
    assert off.rho == ref.rho and off.hstar == ref.hstar


def test_curve_excludes_current_seed() -> None:
    env = synth_env(n=6, seeds=[0, 1, 2], seed=8)
    key = TraceKey("synth/0", 0)
    c = env.chains[key]
    flipped = dict(env.chains)
    flipped[key] = replace(
        c, ans_correct=~c.ans_correct, nat_correct=None if c.nat_correct is None else not c.nat_correct
    )
    out = []
    for chains in (env.chains, flipped):
        prov = CurveRhoProvider(EmpiricalCurves(chains, DP), [512, 1024, 2048], 4000, 0.0)
        s = QuestionState(key, gen_end=600)
        prov.fill([s])
        out.append((s.rho, s.hstar, None if s.qhat is None else tuple(s.qhat)))
    assert out[0] == out[1]  # seed đang mô phỏng không góp vào đường cong của chính nó
    prov = CurveRhoProvider(EmpiricalCurves({key: c}, DP), [512], 4000, 0.0)
    s = QuestionState(key, gen_end=0)
    prov.fill([s])
    assert s.rho == float("-inf") and s.hstar is None  # chỉ một seed: không có thông tin


# ---------------------------------------------------------------- thành phần lô (P2.T14)


def _qs(per_level: int = 40) -> list[Question]:
    out = []
    for lv in range(1, 6):
        out += [Question(f"math500/{lv}_{i}", "math500", "p", None, "1", "math", lv) for i in range(per_level)]
    out += [Question(f"aime_2024/{i}", "aime_2024", "p", None, "1", "math", None) for i in range(30)]
    return out


def test_batch_level_proportions() -> None:
    qs = _qs()
    by = {q.qid: q.difficulty for q in qs}
    for comp, shares in [("balanced", [20] * 5), ("easy", [35, 35, 10, 10, 10]), ("u", [35, 10, 10, 10, 35])]:
        ids = build_batch(qs, BatchCfg(composition=comp, n=100), np.random.default_rng(0))  # type: ignore[arg-type]
        cnt = [sum(by[i] == lv for i in ids) for lv in range(1, 6)]
        assert cnt == shares
    assert sum(level_counts("hard", 37).values()) == 37


def test_batch_deterministic_and_no_duplicates() -> None:
    qs = _qs()
    for spec in [BatchCfg(composition="hard", n=100), BatchCfg(composition="all", n=50)]:
        a = build_batch(qs, spec, np.random.default_rng(3))
        assert a == build_batch(qs, spec, np.random.default_rng(3))
        assert len(set(a)) == len(a)
    assert build_batch(qs, BatchCfg(composition="all"), np.random.default_rng(0)) == [q.qid for q in qs]


def test_batch_raises_when_level_short() -> None:
    with pytest.raises(ValueError):
        build_batch(_qs(per_level=20), BatchCfg(composition="easy", n=100), np.random.default_rng(0))


def test_mix_batch_counts_per_source() -> None:
    ids = build_batch(
        _qs(), BatchCfg(composition="mix", sources={"aime_2024": 30, "math500": 100}), np.random.default_rng(0)
    )
    assert sum(i.startswith("aime_2024/") for i in ids) == 30 and sum(i.startswith("math500/") for i in ids) == 100
    with pytest.raises(ValueError):
        build_batch(_qs(), BatchCfg(composition="mix", sources={"aime_2024": 31}), np.random.default_rng(0))


# ---------------------------------------------------------------- hiệu chỉnh B2 và tìm kiếm (P2.T13, P3.T4)


def test_calibrate_b2_hits_target() -> None:
    env = synth_env(n=40)
    b2 = B2
    draws = env.draws(ALL, 0.3, 2)
    cal = calibrate_b2(env, b2, draws)
    assert not cal.budget_underused and 0.98 <= cal.spend_ratio <= 1.0
    for d in draws:
        assert all(o.costs.total <= cal.allowance for o in env.run(b2, d.keys, d.budget, cal.allowance).outcomes)


def test_calibrate_b2_underused_uses_upper_bound() -> None:
    env = synth_env(n=10)
    cal = calibrate_b2(env, B2, env.draws(ALL, 5.0, 2))
    assert cal.budget_underused and cal.allowance <= env.c_hi() and cal.spend_ratio < 0.98
    assert cal.spend_ratio == pytest.approx(
        np.mean([env.run(B2, d.keys, d.budget, env.c_hi()).spent.total / d.budget for d in env.draws(ALL, 5.0, 2)])
    )  # quỹ quá rộng: không hạn mức nào dùng hết, chọn hạn mức nhỏ nhất đạt chi tiêu cao nhất


def test_random_search_reproducible() -> None:
    env = synth_env(n=8, cfg=synth_cfg(tune={"n_configs": 3, "draws": 2}))
    a = random_search(env, M3, ALL, [0.3, 0.6])
    b = random_search(env, M3, ALL, [0.3, 0.6])
    assert a == b and len(a[1]) == 3
    assert a[0]["objective"] == max(t["objective"] for t in a[1])


# ---------------------------------------------------------------- chạy mô phỏng (P2.T15)


def _exp_cfg():  # type: ignore[no-untyped-def]
    return synth_cfg(
        splits=["val"],
        alphas=[0.3, 0.6],
        methods=[B2.model_dump(), M3.model_dump()],
        sim={"draws": 3, "calib_draws": 2},
    )


def _digest(p: Path) -> str:
    df = pd.read_parquet(p)
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


@pytest.fixture(scope="module")
def sim_runs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    cfg = _exp_cfg()
    dirs = []
    for i in range(2):
        d = tmp_path_factory.mktemp(f"run{i}")
        run_experiment({"val": synth_env(n=12, cfg=cfg)}, cfg, d, {"log_hash": "synth"}, log=lambda _: None)
        dirs.append(d)
    return dirs


def test_same_seed_same_output_hash(sim_runs: list[Path]) -> None:
    assert _digest(sim_runs[0] / "per_question.parquet") == _digest(sim_runs[1] / "per_question.parquet")


def test_draws_paired_across_methods(sim_runs: list[Path]) -> None:
    pq = pd.read_parquet(sim_runs[0] / "per_question.parquet")
    for (_, _), g in pq.groupby(["alpha", "draw"]):
        sets = {m: set(zip(x.qid, x.seed, strict=True)) for m, x in g.groupby("method")}
        assert sets["B2"] == sets["M3"]
        assert g.groupby("method").budget.nunique().eq(1).all() and g.budget.nunique() == 1
    s = json.loads((sim_runs[0] / "summary.json").read_text(encoding="utf-8"))
    assert all(c["spend_ratio"] <= 1.0 for c in s["cells"]) and any("allowance" in c for c in s["cells"])


def test_report_builds(sim_runs: list[Path], tmp_path: Path) -> None:
    man = build_report(sim_runs[:1], tmp_path / "rep", np.random.default_rng(0), n_boot=200)
    assert (tmp_path / "rep" / "accuracy.csv").exists() and man["figures"]
    cells = cell_table(pd.read_parquet(sim_runs[0] / "per_question.parquet"))
    assert set(cells.method) == {"B2", "M3"} and (cells.spend_ratio <= 1.0).all()


# ---------------------------------------------------------------- thống kê, chỉ số, cổng (P2.T17–T19)


def test_mcnemar_exact_hand_computed() -> None:
    a = np.array([1] * 10 + [0] * 2 + [1] * 30, bool)
    b = np.array([0] * 10 + [1] * 2 + [1] * 30, bool)
    r = mcnemar_test(a, b)
    assert r["exact"] and (r["n01"], r["n10"]) == (10, 2)
    assert r["pvalue"] == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_paired_bootstrap_basic() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((5, 50)) < 0.6
    r = paired_bootstrap(a, a, rng, 500)
    assert r["diff"] == r["lo"] == r["hi"] == 0.0
    b = np.zeros_like(a)
    r = paired_bootstrap(a, b, rng, 500)
    assert r["lo"] > 0 and r["lo"] <= r["diff"] <= r["hi"]
    assert acceptance_band(np.full(10, 0.5), 3, rng, 100) == (0.5, 0.5)


@pytest.mark.slow
def test_bootstrap_coverage() -> None:
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(300):
        p = rng.uniform(0.2, 0.8, 200)
        a = rng.random((4, 200)) < p
        b = rng.random((4, 200)) < p
        r = paired_bootstrap(a, b, rng, 400, 0.9)
        hits += r["lo"] <= 0 <= r["hi"]
    assert 0.85 <= hits / 300 <= 0.95


def test_auc_and_tokens_saved() -> None:
    assert auc_over_alpha(np.array([0.1, 0.5, 0.3]), np.array([0.4, 0.4, 0.4])) == pytest.approx(0.4)
    assert auc_over_alpha(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == pytest.approx(0.5)
    s = tokens_saved_vs(np.array([100, 200, 400]), np.array([0.2, 0.4, 0.6]), np.array([150, 50]), np.array([0.5, 0.9]))
    assert s[0] == pytest.approx(0.5) and np.isnan(s[1])


def _pq(win: str, lose: str, n_draws: int = 4, nq: int = 60) -> pd.DataFrame:
    rows = []
    for d in range(n_draws):
        for q in range(nq):
            for m, ok in ((win, True), (lose, q % 3 == 0)):
                rows.append(
                    {
                        "split": "val",
                        "batch": "all",
                        "method": m,
                        "alpha": 0.3,
                        "draw": d,
                        "qid": f"q{q}",
                        "correct": ok,
                    }
                )
    return pd.DataFrame(rows)


def test_gate_g2_pass_and_fail() -> None:
    rng = np.random.default_rng(0)
    assert gate_g2(_pq("M2", "B2"), rng, 300)["pass"]
    res = gate_g2(_pq("B2", "M2"), rng, 300)
    assert not res["pass"] and "KHÔNG ĐẠT" in gate_markdown(res)


def test_gate_g1_pass_and_fail() -> None:
    rng = np.random.default_rng(0)
    good = {"auroc_q0": 0.8, "max_ece": 0.03, "per_h": {"0": {"ece": 0.03}}, "gain_spearman": {"pooled": 0.5}}
    assert gate_g1(good, _pq("M3", "B2"), {"pass_rate": 0.99}, rng, 300)["pass"]
    assert not gate_g1({**good, "auroc_q0": 0.6}, _pq("M3", "B2"), {"pass_rate": 0.99}, rng, 300)["pass"]
    assert not gate_g1(good, _pq("B2", "M3"), {"pass_rate": 0.99}, rng, 300)["pass"]


def test_gate_g3(tmp_path: Path) -> None:
    d = tmp_path / "online"
    d.mkdir()
    cells = [{"alpha": 0.5, "method": m, "accuracy": 0.6, "exceeded": False} for m in ("B2", "M2") for _ in range(3)]
    (d / "summary.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    ref = pd.DataFrame([{"alpha": 0.5, "method": m, "accuracy": a} for m in ("B2", "M2") for a in (0.58, 0.6, 0.62)])
    ref.to_parquet(d / "sim_reference.parquet")
    assert gate_g3([d], np.random.default_rng(0), 200)["pass"]
    cells[0]["exceeded"] = True
    (d / "summary.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    assert not gate_g3([d], np.random.default_rng(0), 200)["pass"]


# ---------------------------------------------------------------- khóa config (P3.T5)


def test_eval_test_refuses_unlocked_or_mismatch(tmp_path: Path) -> None:
    cfg = synth_cfg(name="lock_test", splits=["test_main"])
    log = tmp_path / "test-log.md"
    with pytest.raises(NotLocked):
        check_locked(cfg, "ai", tmp_path / "locked")
    path = lock(cfg, tmp_path / "locked", log)
    locked = load_config(path)
    assert check_locked(locked, "minh", tmp_path / "locked")
    with pytest.raises(NotLocked):
        check_locked(locked, None, tmp_path / "locked")  # thiếu người duyệt
    with pytest.raises(NotLocked):
        check_locked(locked.model_copy(update={"run_seed": 7}), "minh", tmp_path / "locked")
    with pytest.raises(FileExistsError):
        lock(cfg, tmp_path / "locked", log)
    text = log.read_text(encoding="utf-8")
    assert "| lock lock_test | test_main |" in text
    record_eval(locked, check_locked(locked, "minh", tmp_path / "locked"), "minh", tmp_path / "run", log)
    rows = log.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and all(r.count(" | ") == 5 for r in rows)  # đúng 6 cột như tiêu đề bảng
