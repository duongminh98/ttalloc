"""Config: mô hình pydantic, nạp YAML có `extends`, ghi đè từ dòng lệnh và mã băm (PLAN mục 2.4, 9.1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class SamplingCfg(_Cfg):
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = -1


class ModelCfg(_Cfg):
    name: str
    hf_id: str
    revision: str | None = None
    family: Literal["qwen3", "r1_distill"]
    enable_thinking: bool = True
    think_end: str = "</think>"
    sampling: SamplingCfg = SamplingCfg()
    # Số token sinh thêm sau khi khung đóng để ghi probe_has_think_end (PLAN A1b, mục 9.1).
    probe_think_end_lookahead: int = 0
    # DEER: trung bình nhân (geo) với Qwen3, trung bình cộng (arith) với R1-Distill (PLAN mục 7.2).
    deer_conf_mode: Literal["geo", "arith"] = "geo"
    deer_require_think_end_after_probe: bool = False


class DatasetFields(_Cfg):
    problem: str = "problem"
    gold: str | None = "answer"
    # Lấy đáp án chuẩn từ khung \boxed{} cuối của trường này khi bộ dữ liệu không có trường đáp án riêng.
    gold_from_boxed: str | None = None
    id: str | None = None
    difficulty: str | None = None
    # Trắc nghiệm (GPQA): trường đáp án đúng và các đáp án sai.
    correct: str | None = None
    incorrect: list[str] = Field(default_factory=list)


class DatasetCfg(_Cfg):
    name: str
    hf_id: str
    hf_configs: list[str | None] = Field(default_factory=lambda: list[str | None]([None]))
    split: str = "train"
    revision: str | None = None
    answer_type: Literal["math", "choice"] = "math"
    fields: DatasetFields = DatasetFields()
    gated: bool = False


class EngineCfg(_Cfg):
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = True
    seed: int = 0


class DecisionPointCfg(_Cfg):
    markers: list[str] = Field(default_factory=lambda: ["Wait", "Alternatively", "Hmm", "But", "Let me check"])
    m: int = 256
    l_max: int = 1024
    b_min: int = 512


class LogCfg(_Cfg):
    m: int = 128
    l_max: int = 256
    l_log: int = 32768
    window: int = 256
    flush_every: int = 64  # số chuỗi đã xong mỗi lần ghi một phần (P1.T7)


class ProbeCfg(_Cfg):
    r_probe: int = 20
    prompt: str = "deer"


class AnswerCfg(_Cfg):
    # Chọn ở P1.T9; None nghĩa là chưa chọn, các bước cần dạng câu trả lời sẽ báo lỗi.
    mode: Literal["full", "short"] | None = None
    r_ans: int | None = None
    on_stop: Literal["forced", "best_probe", "vote"] = "forced"

    def resolved_r_ans(self) -> int:
        if self.r_ans is not None:
            return self.r_ans
        if self.mode is None:
            raise ValueError("answer.mode chưa được chọn (P1.T9)")
        return 256 if self.mode == "full" else 32


class FeatureCfg(_Cfg):
    k: int = 4
    local_conf: bool = True
    local_window: int = 1024
    drop_groups: list[str] = Field(default_factory=list)


def _lgb_defaults() -> dict[str, Any]:
    return {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbosity": -1,
    }


class EstimatorCfg(_Cfg):
    h_train: list[int] = Field(default_factory=lambda: [256, 512, 1024, 2048, 4096, 8192, 16384])
    kind: Literal["lightgbm", "logreg"] = "lightgbm"
    calibrate: bool = True
    # P2.T4: GroupKFold theo qid, early stopping trên phần tách theo nhóm của phần huấn luyện.
    n_folds: int = 5
    es_frac: float = 0.2
    max_rounds: int = 2000
    early_stopping: int = 100
    seed: int = 0
    lgb_params: dict[str, Any] = Field(default_factory=_lgb_defaults)
    logreg_c: float = 1.0


class PolicyCfg(_Cfg):
    horizons: list[int] = Field(default_factory=lambda: [512, 1024, 2048, 4096, 8192])
    # Chỉnh trên val; cùng đơn vị với rho trong mã (xác suất trên mỗi token).
    mu_min: float | None = None
    b_max_mult: float | None = 3.0


class DeerCfg(_Cfg):
    lambda_: float = Field(0.95, alias="lambda")


class GuardCfg(_Cfg):
    min_segment: int = 256


class SimCfg(_Cfg):
    draws: int = 200
    seed: int = 0
    max_active: int | None = None  # None: mọi câu của lô chạy cùng lúc
    workers: int = 1  # số tiến trình chạy song song theo lần rút
    n_boot: int = 2000
    calib_draws: int | None = 50  # số lần rút dùng khi hiệu chỉnh hạn mức B2 (A11); None: mọi lần rút (ADR-011)


class TuneCfg(_Cfg):
    """Tìm kiếm ngẫu nhiên (P3.T4). Chỉ các tham số không đòi học lại bộ ước lượng."""

    n_configs: int = 50
    draws: int = 20
    seed: int = 0
    lambda_range: list[float] = Field(default_factory=lambda: [0.80, 0.995])
    mu_min_range: list[float] = Field(default_factory=lambda: [1e-7, 3e-4])  # lấy mẫu theo log
    b_max_mult_choices: list[float] = Field(default_factory=lambda: [2.0, 3.0, 4.0])


class BatchCfg(_Cfg):
    """Thành phần lô (P2.T14)."""

    composition: Literal["all", "balanced", "easy", "hard", "u", "mix"] = "all"
    n: int | None = None
    sources: dict[str, int] = Field(default_factory=dict)
    seed: int = 0  # chạy thật dựng lô một lần bằng seed này; mô phỏng dùng seed của lần rút


class OnlineCfg(_Cfg):
    max_active: int | None = None
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    max_num_seqs: int | None = None
    calib_draws: int = 200  # số lần rút mô phỏng dùng để hiệu chỉnh hạn mức của B2 cho chạy thật


class BaselineCfg(_Cfg):
    deer_repo: str = "https://github.com/iie-ycx/DEER.git"
    deer_commit: str = "c9dd19fbffa27f841cfe47502d015b63811b4d1b"
    paper_max_tokens: int = 16384
    deer_think_ratio: float = 0.8  # theo bashes/bash-vllm-deer-qwen3.sh của repo DEER
    deer_policy: str = "avg2"
    throughput_batch_sizes: list[int] = Field(default_factory=lambda: [8, 32, 128, 256])
    throughput_max_tokens: int = 4096
    chunk: int = 512  # số request mỗi lần gọi generate, để ghi dần kết quả


class StudyCfg(_Cfg):
    curves_delta: float = 0.2
    probe_alt: str = "probe_alt"  # tệp lời nhắc gợi thứ hai cho study probe-noise (P1.T10)


class MethodCfg(_Cfg):
    id: str
    policy: str
    rho_provider: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ExperimentCfg(_Cfg):
    name: str = "base"
    model: str | None = None
    splits: list[str] = Field(default_factory=list)
    alphas: list[float] = Field(default_factory=list)
    methods: list[MethodCfg] = Field(default_factory=list)
    tuned_from: str | None = None
    locked: str | None = None
    run_seed: int = 0
    kind: Literal["sim", "online"] = "sim"
    batches: list[BatchCfg] = Field(default_factory=list)  # rỗng: dùng `batch`
    estimator_path: str | None = None  # data/estimators/<model>/<hash>; None: bản mới nhất của model
    train_split: str = "train"  # tập dữ liệu ghi dùng để học bộ ước lượng và ước lượng omega
    ref_split: str = "val"  # tập dùng tính trung vị chain_len cho b_max (mục 9.1)

    engine: EngineCfg = EngineCfg()
    dp: DecisionPointCfg = DecisionPointCfg()
    log: LogCfg = LogCfg()
    probe: ProbeCfg = ProbeCfg()
    answer: AnswerCfg = AnswerCfg()
    features: FeatureCfg = FeatureCfg()
    estimator: EstimatorCfg = EstimatorCfg()
    policy: PolicyCfg = PolicyCfg()
    deer: DeerCfg = DeerCfg()
    guard: GuardCfg = GuardCfg()
    sim: SimCfg = SimCfg()
    tune: TuneCfg = TuneCfg()
    batch: BatchCfg = BatchCfg()
    online: OnlineCfg = OnlineCfg()
    baseline: BaselineCfg = BaselineCfg()
    study: StudyCfg = StudyCfg()

    @model_validator(mode="after")
    def _check(self) -> ExperimentCfg:
        errors = []
        if self.log.m > self.dp.m:
            errors.append(f"log.m ({self.log.m}) phải <= dp.m ({self.dp.m})")
        if self.log.l_max > self.dp.l_max:
            errors.append(f"log.l_max ({self.log.l_max}) phải <= dp.l_max ({self.dp.l_max})")
        if self.guard.min_segment < self.log.l_max:
            errors.append(f"guard.min_segment ({self.guard.min_segment}) phải >= log.l_max ({self.log.l_max})")
        extra = sorted(set(self.policy.horizons) - set(self.estimator.h_train))
        if extra:
            errors.append(f"policy.horizons {extra} không thuộc estimator.h_train")
        if errors:
            raise ValueError("; ".join(errors))
        return self


def check_b_max(cfg: ExperimentCfg, b_max: int, c_ans_max: int) -> None:
    """Bất biến I8: b_max <= L_log - c_ans_max. Kiểm khi đã có tokenizer để tính c_ans_max."""
    limit = cfg.log.l_log - c_ans_max
    if b_max > limit:
        raise ValueError(f"b_max ({b_max}) vượt L_log - c_ans_max ({limit})")


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _resolve_path(ref: str, origin: Path) -> Path:
    for cand in (origin.parent / ref, REPO_ROOT / ref):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"không tìm thấy tệp extends: {ref} (từ {origin})")


def load_yaml_resolved(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Đọc YAML và phân giải khóa `extends` (có thể là một chuỗi hoặc danh sách), tệp sau ghi đè tệp trước."""
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"extends vòng lặp tại {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: nội dung phải là một mapping")
    parents = raw.pop("extends", None)
    merged: dict[str, Any] = {}
    for ref in [parents] if isinstance(parents, str) else parents or []:
        merged = _deep_merge(merged, load_yaml_resolved(_resolve_path(ref, path), _seen | {path}))
    return _deep_merge(merged, raw)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Ghi đè dạng `a.b=giá trị`; giá trị được đọc bằng YAML (số, bool, danh sách...)."""
    out: dict[str, Any] = json.loads(json.dumps(data))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"ghi đè sai dạng (cần khóa=giá trị): {item}")
        dotted, raw_val = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = out
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"không ghi đè được {dotted}: {k} không phải mapping")
        node[keys[-1]] = yaml.safe_load(raw_val)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentCfg:
    data = apply_overrides(load_yaml_resolved(Path(path)), overrides or [])
    return ExperimentCfg.model_validate(data)


def config_to_dict(cfg: BaseModel) -> dict[str, Any]:
    out: dict[str, Any] = cfg.model_dump(mode="json", by_alias=True)
    return out


def config_hash(cfg: BaseModel, n: int = 12) -> str:
    """sha256 của JSON chuẩn hóa (khóa sắp xếp) của config đã phân giải; lấy n ký tự đầu."""
    blob = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def load_model_cfg(name: str, configs_dir: Path = CONFIGS_DIR) -> ModelCfg:
    data = yaml.safe_load((configs_dir / "models" / f"{name}.yaml").read_text(encoding="utf-8"))
    return ModelCfg.model_validate(data)


def load_dataset_cfg(name: str, configs_dir: Path = CONFIGS_DIR) -> DatasetCfg:
    data = yaml.safe_load((configs_dir / "datasets" / f"{name}.yaml").read_text(encoding="utf-8"))
    return DatasetCfg.model_validate(data)
