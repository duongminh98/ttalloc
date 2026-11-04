"""Huấn luyện và hiệu chuẩn bộ ước lượng (PLAN P2.T4, P3.T3).

LightGBM nhị phân với đầu vào là đặc trưng và h; `GroupKFold` theo `qid` cho dự đoán ngoài fold; trong mỗi fold,
early stopping trên một phần tách theo nhóm của phần huấn luyện. Model cuối học trên toàn bộ dữ liệu với số cây bằng
trung vị số cây tốt nhất của các fold. Isotonic regression riêng cho từng h, học trên dự đoán ngoài fold.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttalloc.config import EstimatorCfg

ID_COLS = ("qid", "seed", "point_idx")


def import_lightgbm() -> Any:
    """ADR-007: trên Windows, `lib_lightgbm.dll` cần `vcomp140.dll`; dùng bản có sẵn trong `sklearn/.libs`."""
    if os.name == "nt":
        import sklearn

        libs = Path(sklearn.__file__).parent / ".libs"
        if libs.exists():
            os.add_dll_directory(str(libs))
    import lightgbm

    return lightgbm


def _lgb_params(cfg: EstimatorCfg) -> dict[str, Any]:
    return {**cfg.lgb_params, "seed": cfg.seed, "deterministic": True, "force_row_wise": True, "num_threads": 1}


class _LogReg:
    """Logistic regression riêng cho từng h trên đặc trưng chuẩn hóa; thiếu thì thay bằng trung vị kèm cột chỉ báo."""

    def __init__(self, features: list[str], c: float, seed: int) -> None:
        self.features = features
        self.c = c
        self.seed = seed
        self.models: dict[int, Any] = {}
        self.median: dict[str, float] = {}
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for f in self.features:
            v = X[f].to_numpy(float)
            miss = np.isnan(v)
            cols.append(np.where(miss, self.median[f], v))
            cols.append(miss.astype(float))
        m = np.column_stack(cols)
        assert self.mean is not None and self.std is not None
        return np.asarray((m - self.mean) / self.std)

    def fit(self, X: pd.DataFrame, h: np.ndarray, y: np.ndarray, w: np.ndarray) -> _LogReg:
        from sklearn.linear_model import LogisticRegression

        self.median = {
            f: float(np.nanmedian(X[f].to_numpy(float))) if X[f].notna().any() else 0.0 for f in self.features
        }
        cols = []
        for f in self.features:
            v = X[f].to_numpy(float)
            miss = np.isnan(v)
            cols += [np.where(miss, self.median[f], v), miss.astype(float)]
        raw = np.column_stack(cols)
        self.mean = raw.mean(axis=0)
        self.std = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)
        m = (raw - self.mean) / self.std
        for hv in np.unique(h):
            sel = h == hv
            yy = y[sel]
            if yy.min() == yy.max():
                self.models[int(hv)] = float(yy[0])  # chỉ một lớp: hằng số
                continue
            self.models[int(hv)] = LogisticRegression(C=self.c, max_iter=2000, random_state=self.seed).fit(
                m[sel], yy, sample_weight=w[sel]
            )
        return self

    def predict(self, X: pd.DataFrame, hv: int) -> np.ndarray:
        model = self.models[int(hv)]
        if isinstance(model, float):
            return np.full(len(X), model)
        return np.asarray(model.predict_proba(self._matrix(X))[:, 1])


@dataclass
class Estimator:
    kind: str
    features: list[str]
    hs: list[int]  # {0} ∪ h_train
    model: Any
    isotonic: dict[int, Any] | None
    meta: dict[str, Any] = field(default_factory=dict)

    def raw_predict(self, X: pd.DataFrame, hv: int) -> np.ndarray:
        if self.kind == "lightgbm":
            Xh = X[self.features].assign(h=float(hv))
            return np.asarray(self.model.predict(Xh, num_iteration=self.meta.get("n_trees")))
        return np.asarray(self.model.predict(X, hv))

    def predict(self, X: pd.DataFrame, hs: list[int]) -> np.ndarray:
        bad = [h for h in hs if h not in self.hs]
        if bad:
            raise ValueError(f"h {bad} không có trong bộ ước lượng (đã học {self.hs})")
        out = np.empty((len(X), len(hs)))
        for j, hv in enumerate(hs):
            p = self.raw_predict(X, hv)
            if self.isotonic is not None:
                p = self.isotonic[hv].predict(p)
            out[:, j] = np.clip(p, 0.0, 1.0)
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self.kind == "lightgbm":
            self.model.save_model(str(path / "model.txt"))
        else:
            with (path / "model.pkl").open("wb") as f:
                pickle.dump(self.model, f)
        with (path / "isotonic.pkl").open("wb") as f:
            pickle.dump(self.isotonic, f)
        meta = {**self.meta, "kind": self.kind, "features": self.features, "hs": self.hs}
        (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Estimator:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["kind"] == "lightgbm":
            lgb = import_lightgbm()
            model = lgb.Booster(model_file=str(path / "model.txt"))
        else:
            with (path / "model.pkl").open("rb") as f:
                model = pickle.load(f)  # tệp do chính repo ghi ra
        with (path / "isotonic.pkl").open("rb") as f:
            iso = pickle.load(f)
        return cls(meta["kind"], list(meta["features"]), [int(h) for h in meta["hs"]], model, iso, meta)


def group_folds(groups: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    if n_groups < n_folds:
        raise ValueError(f"chỉ có {n_groups} câu hỏi, không đủ cho {n_folds} fold")
    return list(GroupKFold(n_splits=n_folds).split(np.zeros(len(groups)), groups=groups))


def _es_split(groups: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed).split(np.zeros(len(groups)), groups=groups)
    )
    return a, b


def _fit_lgb(
    cfg: EstimatorCfg,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    rounds: int,
    valid: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
) -> Any:
    lgb = import_lightgbm()
    dtrain = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
    kwargs: dict[str, Any] = {}
    if valid is not None:
        dvalid = lgb.Dataset(valid[0], label=valid[1], weight=valid[2], reference=dtrain)
        kwargs = {"valid_sets": [dvalid], "callbacks": [lgb.early_stopping(cfg.early_stopping, verbose=False)]}
    return lgb.train(_lgb_params(cfg), dtrain, num_boost_round=rounds, **kwargs)


def train_estimator(df: pd.DataFrame, features: list[str], cfg: EstimatorCfg) -> tuple[Estimator, pd.DataFrame]:
    """Học bộ ước lượng trên `df` (cột đặc trưng, h, y, w, qid). Trả về bộ ước lượng và bảng dự đoán ngoài fold."""
    hs = [0, *cfg.h_train]
    groups = df["qid"].to_numpy()
    y = df["y"].to_numpy(int)
    w = df["w"].to_numpy(float)
    h = df["h"].to_numpy(int)
    X = df[features].assign(h=df["h"].astype(float))
    oof = np.zeros(len(df))
    best: list[int] = []
    for tr, te in group_folds(groups, cfg.n_folds):
        if cfg.kind == "lightgbm":
            a, b = _es_split(groups[tr], cfg.es_frac, cfg.seed)
            ia, ib = tr[a], tr[b]
            booster = _fit_lgb(cfg, X.iloc[ia], y[ia], w[ia], cfg.max_rounds, (X.iloc[ib], y[ib], w[ib]))
            it = int(booster.best_iteration or cfg.max_rounds)
            best.append(it)
            oof[te] = booster.predict(X.iloc[te], num_iteration=it)
        else:
            lr = _LogReg(features, cfg.logreg_c, cfg.seed).fit(df.iloc[tr], h[tr], y[tr], w[tr])
            for hv in np.unique(h[te]):
                sel = te[h[te] == hv]
                oof[sel] = lr.predict(df.iloc[sel], int(hv))
    meta: dict[str, Any] = {"cfg": cfg.model_dump(), "n_samples": len(df), "n_questions": len(np.unique(groups))}
    if cfg.kind == "lightgbm":
        n_trees = max(1, int(np.median(best)))
        model: Any = _fit_lgb(cfg, X, y, w, n_trees)
        meta["n_trees"] = n_trees
        meta["fold_best_iterations"] = best
    else:
        model = _LogReg(features, cfg.logreg_c, cfg.seed).fit(df, h, y, w)
    iso = None
    if cfg.calibrate:
        from sklearn.isotonic import IsotonicRegression

        iso = {}
        for hv in hs:
            sel = h == hv
            iso[hv] = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                oof[sel], y[sel], sample_weight=w[sel]
            )
    est = Estimator(cfg.kind, features, hs, model, iso, meta)
    oof_cal = oof.copy()
    if iso is not None:
        for hv in hs:
            sel = h == hv
            oof_cal[sel] = iso[hv].predict(oof[sel])
    oof_df = df[["qid", "seed", "point_idx", "h", "y", "w"]].assign(p_raw=oof, p=np.clip(oof_cal, 0, 1))
    est.meta["oof"] = oof_metrics(oof_df)
    return est, oof_df


def oof_metrics(oof: pd.DataFrame) -> dict[str, Any]:
    from ttalloc.estimator.evaluate import auroc, ece, log_loss

    out: dict[str, Any] = {}
    for hv, g in oof.groupby("h"):
        y = g.y.to_numpy(int)
        p = g.p.to_numpy(float)
        out[str(hv)] = {"log_loss": log_loss(y, p), "ece": ece(y, p), "auroc": auroc(y, p)}
    return out
