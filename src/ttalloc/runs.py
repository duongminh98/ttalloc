"""Thư mục chạy: `runs/<YYYYmmdd-HHMMSS>_<tên>_<hash>/`, không bao giờ ghi đè (PLAN mục 2.4)."""

from __future__ import annotations

import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ttalloc.config import REPO_ROOT, config_hash, config_to_dict

TRACKED_PACKAGES = (
    "ttalloc", "numpy", "pandas", "pyarrow", "pydantic", "transformers", "tokenizers", "datasets",
    "math-verify", "lightgbm", "scikit-learn", "scipy", "statsmodels", "torch", "vllm",
)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(cwd: Path = REPO_ROOT) -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain"], cwd)
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> list[str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def build_meta(cwd: Path = REPO_ROOT) -> dict[str, Any]:
    return {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_info(cwd),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_info(),
        "argv": sys.argv,
    }


def create_run_dir(name: str, cfg: BaseModel, root: Path | None = None, now: dt.datetime | None = None) -> Path:
    """Tạo thư mục chạy mới, ghi config.yaml, meta.json và log.txt rỗng. Không bao giờ dùng lại thư mục đã có."""
    root = root if root is not None else REPO_ROOT / "runs"
    root.mkdir(parents=True, exist_ok=True)
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{name}_{config_hash(cfg)}"
    path = root / base
    k = 1
    while True:
        try:
            path.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            path = root / f"{base}-{k}"
            k += 1
    (path / "config.yaml").write_text(
        yaml.safe_dump(config_to_dict(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    meta = build_meta()
    meta["config_hash"] = config_hash(cfg)
    (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (path / "log.txt").touch()
    return path
