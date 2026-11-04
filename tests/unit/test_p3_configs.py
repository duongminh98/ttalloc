"""P3.T6, P4.T4: mọi config thí nghiệm nạp được, nhất quán với nhau và với ràng buộc ở PLAN mục 9.1."""

from __future__ import annotations

from pathlib import Path

import pytest

from ttalloc.config import CONFIGS_DIR, load_config
from ttalloc.sim.env import resolve_params

EXP_DIR = CONFIGS_DIR / "experiments"
FILES = sorted(EXP_DIR.rglob("*.yaml"))
POLICIES = {"deer", "price_adaptive"}  # các chính sách đã cài (B2, M2, M3); còn lại ở P5.T1


def test_experiment_dir_not_empty() -> None:
    names = {p.relative_to(EXP_DIR).as_posix() for p in FILES}
    assert {"qwen3.yaml", "g1.yaml", "tune_val.yaml", "e2_main.yaml", "e5_online.yaml"} <= names
    assert sum(n.startswith("e4/") for n in names) >= 15


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(EXP_DIR).as_posix())
def test_experiment_configs_resolve(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.model == "qwen3_1_7b"
    assert cfg.log.m <= cfg.dp.m and cfg.log.l_max <= cfg.dp.l_max  # mục 9.1
    assert cfg.guard.min_segment >= cfg.log.l_max  # A8: mọi đoạn được cấp chứa ít nhất một điểm đã ghi
    assert set(cfg.policy.horizons) <= set(cfg.estimator.h_train)
    for m in cfg.methods:
        assert m.policy in POLICIES, f"{path}: {m.id} dùng chính sách chưa cài ({m.policy})"
        if "tuned" in m.params.values() and not cfg.tuned_from:
            with pytest.raises(ValueError, match="tuned_from"):
                resolve_params(m, cfg)
    if cfg.kind == "online":
        assert len(cfg.splits) == 1 and cfg.online.seeds
    for s in cfg.splits:
        assert s in {"train", "val", "test_main", "test_clean", "test_ood"}


def test_experiment_names_unique() -> None:
    names = [load_config(p).name for p in FILES]
    assert len(names) == len(set(names))


def test_e5_is_minimal_grid() -> None:
    cfg = load_config(EXP_DIR / "e5_online.yaml")
    assert cfg.kind == "online" and cfg.alphas == [0.5] and [m.id for m in cfg.methods] == ["B2", "M2"]
    assert cfg.online.seeds == [0, 1, 2] and cfg.batch.sources == {"math500": 500}  # 6 lần chạy (mục 7.3)


def test_e4_shares_draws_with_e2() -> None:
    e2 = load_config(EXP_DIR / "e2_main.yaml")
    for p in sorted((EXP_DIR / "e4").glob("*.yaml")):
        c = load_config(p)
        assert (c.sim.seed, c.sim.draws, c.alphas, c.splits, c.batch) == (
            e2.sim.seed,
            e2.sim.draws,
            e2.alphas,
            e2.splits,
            e2.batch,
        ), p.name
