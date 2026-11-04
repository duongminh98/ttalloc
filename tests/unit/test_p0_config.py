"""P0.T2: config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ttalloc.config import (
    CONFIGS_DIR,
    ExperimentCfg,
    check_b_max,
    config_hash,
    load_config,
    load_dataset_cfg,
    load_model_cfg,
)


def _write(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_config_extends_and_override(tmp_path: Path) -> None:
    _write(tmp_path / "parent.yaml", {"name": "parent", "dp": {"m": 300, "l_max": 1200}, "sim": {"draws": 50}})
    child = _write(tmp_path / "child.yaml", {"extends": "parent.yaml", "name": "child", "dp": {"m": 320}})
    cfg = load_config(child, ["sim.draws=7", "policy.mu_min=1e-5", "dp.markers=[Wait, Hmm]"])
    assert cfg.name == "child"
    assert cfg.dp.m == 320  # tệp con ghi đè
    assert cfg.dp.l_max == 1200  # kế thừa từ tệp cha
    assert cfg.sim.draws == 7  # ghi đè từ dòng lệnh
    assert cfg.policy.mu_min == pytest.approx(1e-5)
    assert cfg.dp.markers == ["Wait", "Hmm"]
    assert cfg.log.l_log == 32768  # mặc định


def test_config_extends_repo_relative() -> None:
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    assert cfg.deer.lambda_ == pytest.approx(0.95)


def test_config_hash_stable_under_key_order(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", {"name": "x", "dp": {"m": 300, "b_min": 600}, "sim": {"draws": 10, "seed": 3}})
    b = _write(tmp_path / "b.yaml", {"sim": {"seed": 3, "draws": 10}, "dp": {"b_min": 600, "m": 300}, "name": "x"})
    assert config_hash(load_config(a)) == config_hash(load_config(b))
    assert len(config_hash(load_config(a))) == 12
    c = _write(tmp_path / "c.yaml", {"name": "x", "dp": {"m": 300, "b_min": 601}, "sim": {"draws": 10, "seed": 3}})
    assert config_hash(load_config(c)) != config_hash(load_config(a))


@pytest.mark.parametrize(
    "override",
    [
        "log.m=512",  # log.m > dp.m
        "log.l_max=2048",  # log.l_max > dp.l_max
        "guard.min_segment=100",  # min_segment < log.l_max
        "policy.horizons=[512, 3000]",  # 3000 không thuộc h_train
        "dp.unknown_key=1",  # khóa lạ
        "answer.mode=medium",  # giá trị ngoài danh sách
    ],
)
def test_config_validators_reject_bad_values(override: str) -> None:
    with pytest.raises(ValidationError):
        load_config(CONFIGS_DIR / "base.yaml", [override])


def test_check_b_max() -> None:
    cfg = ExperimentCfg()
    check_b_max(cfg, b_max=cfg.log.l_log - 300, c_ans_max=300)
    with pytest.raises(ValueError):
        check_b_max(cfg, b_max=cfg.log.l_log - 299, c_ans_max=300)


def test_base_yaml_matches_defaults() -> None:
    assert load_config(CONFIGS_DIR / "base.yaml") == ExperimentCfg()


def test_model_and_dataset_configs_load() -> None:
    m = load_model_cfg("qwen3_1_7b")
    assert m.family == "qwen3" and m.sampling.top_k == 20 and m.deer_require_think_end_after_probe
    for p in sorted((CONFIGS_DIR / "datasets").glob("*.yaml")):
        assert load_dataset_cfg(p.stem).name == p.stem
