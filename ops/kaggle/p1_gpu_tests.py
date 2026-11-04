"""P1.T4 và P4.T3 trên Kaggle (T4, float16): cài vLLM và ttalloc, chạy `pytest -m gpu tests/gpu` và bộ test CPU.

Đẩy lên bằng: uv run --group ops python ops/kaggle/push.py p1_gpu_tests
Kết quả: /kaggle/working/p1_gpu_tests.json (tóm tắt) và các tệp pytest-*.txt (đầu ra đầy đủ). Không dùng tập dữ liệu.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

VLLM_VERSION = os.environ.get("TTALLOC_VLLM", "0.32.0")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
RUN_ID = globals().get("TTALLOC_RUN_ID")  # push.py chèn vào; trình theo dõi dùng để bỏ qua output của bản trước
REPORT: dict[str, Any] = {"run_id": RUN_ID, "vllm_version_requested": VLLM_VERSION, "steps": {}}


def save() -> None:
    (WORK / "p1_gpu_tests.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding="utf-8")


def run(
    name: str, cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 5400
) -> bool:
    print(f"=== {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        out = subprocess.run(
            cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=timeout
        )
        text, rc = out.stdout + out.stderr, out.returncode
    except subprocess.TimeoutExpired as exc:
        text, rc = f"timeout sau {exc.timeout}s", -1
    (WORK / f"{name}.txt").write_text(text, encoding="utf-8")
    summary = re.findall(r"=+ (.*(?:passed|failed|error).*) =+", text)
    REPORT["steps"][name] = {
        "ok": rc == 0,
        "rc": rc,
        "seconds": round(time.time() - t0, 1),
        "summary": summary[-1] if summary else None,
        "tail": text[-3000:],
    }
    print(text[-2000:], flush=True)
    save()
    return rc == 0


def find_source() -> Path:
    target = WORK / "ttalloc"
    for tgz in Path("/kaggle/input").rglob("ttalloc-src.tar.gz"):
        if target.exists():
            shutil.rmtree(target)
        with tarfile.open(tgz) as tf:
            tf.extractall(target, filter="data")
        return target
    raise FileNotFoundError("không thấy ttalloc-src.tar.gz trong /kaggle/input (đính kèm dataset ttalloc-src)")


def main() -> None:
    print(f"RUN_ID {RUN_ID}", flush=True)
    run("gpu_info", ["nvidia-smi"])
    src = find_source()
    REPORT["source"] = str(src)
    py = sys.executable
    if run("pip_vllm", [py, "-m", "pip", "install", "-q", f"vllm=={VLLM_VERSION}"]):
        run(
            "pip_ttalloc",
            [py, "-m", "pip", "install", "-q", "--ignore-requires-python", "-e", str(src), "pytest", "hypothesis"],
        )
        env = {"TTALLOC_DTYPE": "float16", "PYTHONPATH": str(src / "src")}
        run(
            "pytest-gpu",
            [py, "-m", "pytest", "-m", "gpu", "tests/gpu", "-q", "-rA", "-p", "no:cacheprovider"],
            cwd=src,
            env=env,
        )
        run("pytest-cpu", [py, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=src, env=env, timeout=1800)
    REPORT["ok"] = all(s["ok"] for s in REPORT["steps"].values())
    save()
    print(json.dumps({k: (v["ok"], v["summary"]) for k, v in REPORT["steps"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
