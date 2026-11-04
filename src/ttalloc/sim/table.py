"""Bảng điểm tính sẵn cho mô phỏng (PLAN mục 4.4, P2.T10).

Mỗi dòng ứng với một điểm của dãy chuẩn (A3): khóa, chỉ số điểm trước và sau trong dãy, véc-tơ đặc trưng và `qhat`
cho mọi h trong {0} ∪ `estimator.h_train`. Mã băm gồm config điểm quyết định, config đặc trưng và mã băm bộ ước lượng.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ttalloc.config import DecisionPointCfg, FeatureCfg
from ttalloc.context import DATA_DIR
from ttalloc.estimator.dataset import features_along
from ttalloc.estimator.train import Estimator
from ttalloc.features.events import LoggedChain, canonical_points
from ttalloc.features.extract import feature_names
from ttalloc.types import TraceKey

TABLES_DIR = DATA_DIR / "tables"


def table_hash(dp: DecisionPointCfg, feat: FeatureCfg, estimator_id: str, log_hash: str) -> str:
    blob = json.dumps(
        {"dp": dp.model_dump(), "features": feat.model_dump(), "estimator": estimator_id, "log_hash": log_hash},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def canonical_paths(chains: dict[TraceKey, LoggedChain], dp: DecisionPointCfg) -> dict[TraceKey, list[int]]:
    return {k: canonical_points(c, dp) for k, c in chains.items()}


def build_table(
    chains: dict[TraceKey, LoggedChain],
    dp: DecisionPointCfg,
    feat: FeatureCfg,
    est: Estimator,
    paths: dict[TraceKey, list[int]] | None = None,
) -> pd.DataFrame:
    paths = paths or canonical_paths(chains, dp)
    names = feature_names(feat.drop_groups)
    if names != est.features:
        raise ValueError(f"đặc trưng của config {names} khác của bộ ước lượng {est.features}")
    rows = []
    for k, c in chains.items():
        sub = paths[k]
        for j, x in enumerate(features_along(c, sub, feat.k)):
            rows.append(
                {
                    "qid": k.qid,
                    "seed": k.seed,
                    "point_idx": sub[j],
                    "gen_end": int(c.gen_end[sub[j]]),
                    "prev_idx": sub[j - 1] if j else 0,
                    "next_idx": sub[j + 1] if j + 1 < len(sub) else -1,
                    **{n: x[n] for n in names},
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    q = est.predict(df[names], est.hs)
    for i, h in enumerate(est.hs):
        df[f"q_{h}"] = q[:, i]
    return df


def table_path(model: str, split: str, h: str, tables_dir: Path = TABLES_DIR) -> Path:
    return tables_dir / model / split / f"{h}.parquet"
