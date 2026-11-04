"""Khóa cấu hình và kỷ luật với tập kiểm tra (PLAN P3.T5, mục 7.5)."""

from __future__ import annotations

from pathlib import Path

import yaml

from ttalloc.baseline.full import TEST_LOG, append_test_log, repo_rel
from ttalloc.config import CONFIGS_DIR, ExperimentCfg, config_hash, config_to_dict, load_config

LOCKED_DIR = CONFIGS_DIR / "locked"


class NotLocked(PermissionError):
    pass


def lock(cfg: ExperimentCfg, locked_dir: Path = LOCKED_DIR, log_path: Path = TEST_LOG) -> Path:
    """Ghi config đã phân giải vào configs/locked/<tên>.yaml (kèm .sha) và thêm một dòng vào docs/test-log.md."""
    locked_dir.mkdir(parents=True, exist_ok=True)
    path = locked_dir / f"{cfg.name}.yaml"
    if path.exists():
        raise FileExistsError(f"{path} đã có; khóa lại phải dùng tên mới và ghi lý do vào docs/test-log.md (mục 7.5)")
    try:
        rel = path.resolve().relative_to(CONFIGS_DIR.parent.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    cfg = cfg.model_copy(update={"locked": rel})  # eval-test chạy thẳng từ tệp đã khóa
    h = config_hash(cfg)
    path.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8")
    path.with_suffix(".sha").write_text(h + "\n", encoding="utf-8")
    append_test_log(f"lock {cfg.name}", ",".join(cfg.splits), h, "—", repo_rel(path), log_path)
    return path


def check_locked(cfg: ExperimentCfg, approved_by: str | None, locked_dir: Path = LOCKED_DIR) -> str:
    """Từ chối nếu config chưa khóa, mã băm không khớp hoặc thiếu người duyệt. Trả về mã băm."""
    path = Path(cfg.locked) if cfg.locked else locked_dir / f"{cfg.name}.yaml"
    if not path.is_absolute() and not path.exists():
        path = CONFIGS_DIR.parent / path
    if not path.exists():
        raise NotLocked(f"config {cfg.name} chưa được khóa ({path} không có); chạy `ttalloc lock --exp ...`")
    locked = load_config(path)
    sha = path.with_suffix(".sha")
    stored = sha.read_text(encoding="utf-8").strip() if sha.exists() else None
    h = config_hash(cfg)
    if stored != config_hash(locked) or h != stored:
        raise NotLocked(f"mã băm không khớp: hiện hành {h}, đã khóa {stored} ({path})")
    if not approved_by:
        raise NotLocked("eval-test cần --approved-by <tên> (mục 7.5)")
    return h


def record_eval(cfg: ExperimentCfg, h: str, approved_by: str, run_dir: Path, log_path: Path = TEST_LOG) -> None:
    append_test_log(f"eval-test {cfg.name}", ",".join(cfg.splits), h, approved_by, repo_rel(run_dir), log_path)
