"""Nạp môi trường mô phỏng từ config: dữ liệu ghi, câu hỏi, bộ ước lượng và bảng tính sẵn theo mã băm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ttalloc.config import ExperimentCfg, load_model_cfg
from ttalloc.context import DATA_DIR, load_split_questions, log_hash
from ttalloc.data.prompts import load_prompts
from ttalloc.estimator.train import Estimator
from ttalloc.logging.store import LogStore
from ttalloc.sim.env import SimEnv
from ttalloc.sim.logged_source import load_logged
from ttalloc.sim.table import build_table, canonical_paths, table_hash, table_path

ESTIMATORS_DIR = DATA_DIR / "estimators"


def expected_log_hash(cfg: ExperimentCfg) -> str:
    if not cfg.model:
        raise ValueError("config thiếu `model`")
    return log_hash(cfg, load_model_cfg(cfg.model), load_prompts(cfg.probe.prompt))


def estimator_hash(cfg: ExperimentCfg, train_log_hash: str) -> str:
    blob = json.dumps(
        {
            "estimator": cfg.estimator.model_dump(),
            "features": cfg.features.model_dump(),
            "dp": cfg.dp.model_dump(),
            "train": cfg.train_split,
            "log_hash": train_log_hash,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def estimator_dir(cfg: ExperimentCfg, train_log_hash: str) -> Path:
    if cfg.estimator_path:
        return Path(cfg.estimator_path)
    assert cfg.model
    return ESTIMATORS_DIR / cfg.model / estimator_hash(cfg, train_log_hash)


def ref_median_chain_len(cfg: ExperimentCfg) -> float:
    assert cfg.model
    t = LogStore.for_split(cfg.model, cfg.ref_split).read_traces(columns=["chain_len"])
    if t.num_rows == 0:
        raise FileNotFoundError(f"chưa có dữ liệu ghi của {cfg.ref_split} để tính trung vị chain_len (b_max)")
    return float(pd.Series(t.column("chain_len").to_numpy()).median())


def needs_estimator(cfg: ExperimentCfg) -> bool:
    return any(m.policy == "price_adaptive" and m.rho_provider != "curve" for m in cfg.methods)


def load_env(cfg: ExperimentCfg, split: str, build_missing_table: bool = True) -> SimEnv:
    lh = expected_log_hash(cfg)
    assert cfg.model
    chains, meta = load_logged(cfg.model, split, lh)
    questions = {q.qid: q for q in load_split_questions(split)}
    est = None
    omega = 0.0
    table = None
    paths = None
    if needs_estimator(cfg):
        d = estimator_dir(cfg, lh)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"chưa có bộ ước lượng ở {d}; chạy `ttalloc estimator train --exp ...` trước")
        est = Estimator.load(d)
        omega = float(est.meta.get("omega", 0.0))
        paths = canonical_paths(chains, cfg.dp)
        th = table_hash(cfg.dp, cfg.features, d.name, lh)
        tp = table_path(cfg.model, split, th)
        if tp.exists():
            table = pd.read_parquet(tp)
        elif build_missing_table:
            table = build_table(chains, cfg.dp, cfg.features, est, paths)
            tp.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(tp, index=False)
    return SimEnv(
        cfg=cfg,
        split=split,
        chains=chains,
        questions=questions,
        c_ans_max=int(meta["c_ans_max"]),
        c_probe_max=int(meta["c_probe_max"]),
        ref_median=ref_median_chain_len(cfg),
        estimator=est,
        omega=omega,
        table=table,
        paths=paths,
    )
