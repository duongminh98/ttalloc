"""P0.T3: thư mục chạy."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ttalloc.config import ExperimentCfg, config_hash
from ttalloc.runs import create_run_dir


def test_rundir_unique_and_never_overwritten(tmp_path: Path) -> None:
    cfg = ExperimentCfg()
    now = dt.datetime(2026, 11, 3, 10, 0, 0)
    first = create_run_dir("demo", cfg, root=tmp_path, now=now)
    (first / "result.txt").write_text("giữ nguyên", encoding="utf-8")
    second = create_run_dir("demo", cfg, root=tmp_path, now=now)  # cùng giây, cùng config
    assert first != second
    assert first.name == f"20261103-100000_demo_{config_hash(cfg)}"
    assert (first / "result.txt").read_text(encoding="utf-8") == "giữ nguyên"
    for d in (first, second):
        assert (d / "config.yaml").exists() and (d / "meta.json").exists() and (d / "log.txt").exists()


def test_meta_has_git_and_versions(tmp_path: Path) -> None:
    d = create_run_dir("demo", ExperimentCfg(), root=tmp_path)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["git"]) == {"commit", "dirty"}
    assert meta["packages"]["numpy"] is not None
    assert "vllm" in meta["packages"]  # None trên máy không có vLLM
    assert meta["python"]
    assert meta["config_hash"] == config_hash(ExperimentCfg())
