"""Tập dượt các công việc Vast trên Kaggle T4 (float16), cỡ rất nhỏ, chỉ val và train: bắt lỗi trước khi thuê máy.

Đẩy lên bằng: uv run --group ops python ops/kaggle/push.py rehearsal
Chạy đúng các lệnh `ttalloc` mà ops/vast/jobs/*.sh gọi (baseline, equivalence, log, validate-logs, study), rồi đường
CPU của P2–P3 (estimator, table, sim, tune, lock, eval-test, report, gate) và chạy thật P4 qua eval-test, với config ở
configs/experiments/rehearsal/. Không chạm test_*. Số liệu chỉ để bắt lỗi, không đưa vào khóa luận.

Kết quả: /kaggle/working/rehearsal.json (từng bước: ok, mã thoát, thời gian, đuôi log), steps/<bước>.txt (đầu ra đầy
đủ), out/ (runs, data/logs, data/estimators, reports, configs/locked) để chạy lại phần CPU ở máy cá nhân nếu cần.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

VLLM_VERSION = os.environ.get("TTALLOC_VLLM", "0.32.0")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
SRC = Path(tempfile.gettempdir()) / "ttalloc"  # ngoài /kaggle/working để cây mã không thành output
RUN_ID = globals().get("TTALLOC_RUN_ID")  # push.py chèn vào; trình theo dõi dùng để bỏ qua output của bản trước
REPORT: dict[str, Any] = {"run_id": RUN_ID, "vllm_version_requested": VLLM_VERSION, "steps": {}}
APPROVER = "rehearsal (Duong Tuan Minh)"
EXP = "configs/experiments/rehearsal"
B = f"{EXP}/base.yaml"
NB = ["--n-boot", "200"]

Arg = str | Callable[[], str]


def save() -> None:
    (WORK / "rehearsal.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding="utf-8")


def latest(pattern: str, root: str = "runs") -> str:
    """Thư mục hoặc tệp mới nhất khớp mẫu, đường dẫn tương đối từ gốc mã."""
    found = sorted((SRC / root).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"không có {root}/{pattern}")
    return found[-1].relative_to(SRC).as_posix()


def run_dir(name: str) -> Callable[[], str]:
    return lambda: latest(f"*_{name}_*")


def tuned_from() -> str:
    return "tuned_from=" + run_dir("tune-rehearsal_tune")()


# (tên, đối số của `ttalloc`, các bước phải đạt trước). Thứ tự giống ops/vast/jobs: p1_baseline, p1_pilot,
# p2_log/p3_log, phần CPU của P2–P3, rồi p4_online.
STEPS: list[tuple[str, list[Arg], list[str]]] = [
    ("splits_verify", ["splits", "verify"], []),
    (
        "baseline_full",
        ["baseline", "full", "--exp", B, "--split", "val", "--limit", "8", "--profile", "deer_paper"],
        [],
    ),
    ("baseline_deer", ["baseline", "deer", "--exp", B, "--split", "val", "--limit", "8"], []),
    ("throughput", ["baseline", "throughput", "--exp", B, "--n", "64"], []),
    ("equivalence", ["equivalence", "--exp", B, "--n", "8", "--seeds", "0"], []),
    ("log_val", ["log", "--exp", B, "--split", "val", "--seeds", "0-1"], []),
    ("validate_val", ["validate-logs", "--exp", B, "--split", "val"], ["log_val"]),
    ("study_answer_mode", ["study", "answer-mode", "--exp", B, "--n", "8"], []),
    ("study_probe_noise", ["study", "probe-noise", "--exp", B, "--n", "8"], []),
    ("log_train", ["log", "--exp", B, "--split", "train", "--limit", "100", "--seeds", "0-1"], []),
    ("validate_train", ["validate-logs", "--exp", B, "--split", "train"], ["log_train"]),
    ("estimator_train", ["estimator", "train", "--exp", B], ["log_train"]),
    ("estimator_eval", ["estimator", "eval", "--exp", B, "--split", "val"], ["estimator_train", "log_val"]),
    ("study_curves", ["study", "curves", "--exp", B, "--split", "val"], ["log_val"]),
    ("table_build", ["table", "build", "--exp", f"{EXP}/sim.yaml", "--split", "val"], ["estimator_train", "log_val"]),
    ("sim_run", ["sim", "run", "--exp", f"{EXP}/sim.yaml"], ["table_build"]),
    (
        "gate_g1",
        [
            "gate", "g1", "--runs", run_dir("sim-rehearsal_sim"),
            "--est-eval", lambda: run_dir("estimator-eval-val")() + "/eval.json",
            "--validate", "data/logs/qwen3_1_7b/val/validate.json", *NB,
        ],
        ["sim_run", "estimator_eval", "validate_val"],
    ),
    ("tune", ["tune", "--exp", f"{EXP}/tune.yaml"], ["table_build"]),
    (
        "lock_e2",
        ["lock", "--exp", f"{EXP}/e2.yaml", "--set", tuned_from],
        ["tune"],
    ),
    (
        "eval_test_e2",
        ["eval-test", "--exp", "configs/locked/rehearsal_e2.yaml", "--approved-by", APPROVER],
        ["lock_e2"],
    ),
    ("gate_g2", ["gate", "g2", "--runs", run_dir("eval-test-rehearsal_e2"), *NB], ["eval_test_e2"]),
    (
        "report",
        [
            "report", "--runs", run_dir("sim-rehearsal_sim"), "--runs", run_dir("eval-test-rehearsal_e2"),
            "--out", "reports/rehearsal", "--est-eval", lambda: run_dir("estimator-eval-val")() + "/eval.json", *NB,
        ],
        ["sim_run", "eval_test_e2", "estimator_eval"],
    ),
    (
        "online_run",
        ["online", "run", "--exp", f"{EXP}/online.yaml", "--set", tuned_from],
        ["tune"],
    ),
    (
        "lock_online",
        ["lock", "--exp", f"{EXP}/online.yaml", "--set", tuned_from],
        ["tune"],
    ),
    (
        "eval_test_online",
        ["eval-test", "--exp", "configs/locked/rehearsal_online.yaml", "--approved-by", APPROVER],
        ["lock_online"],
    ),
    ("gate_g3", ["gate", "g3", "--runs", run_dir("eval-test-rehearsal_online")], ["eval_test_online"]),
]  # fmt: skip


NOISE = re.compile(r"\d+%\|.*(it/s|s/it)\]")  # thanh tiến trình của vLLM/tqdm: chỉ lưu vào tệp, không in ra log
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
# API Kaggle không cho hủy phiên đang chạy khi không biết kernel_session_id (CLI không có lệnh hủy, và đẩy bản mới thì
# bản cũ vẫn chạy song song). Nên kernel tự đọc tệp này trên repo công khai; `push.py <tên> --stop` ghi RUN_ID vào đó.
STOP_URL = "https://raw.githubusercontent.com/duongminh98/ttalloc/main/ops/kaggle/stop.txt"
STOP = threading.Event()


def stop_requested(text: str) -> bool:
    return RUN_ID is not None and any(line.strip() == RUN_ID for line in text.splitlines())


def watch_stop(every: int = 60) -> None:
    while not STOP.is_set():
        try:
            with urllib.request.urlopen(f"{STOP_URL}?t={int(time.time())}", timeout=20) as r:
                if stop_requested(r.read().decode("utf-8", "replace")):
                    print(f"YÊU CẦU DỪNG cho {RUN_ID}: dừng bước đang chạy, gom kết quả rồi thoát", flush=True)
                    STOP.set()
        except Exception:  # mạng chập chờn: thử lại lần sau
            pass
        STOP.wait(every)


def run(name: str, cmd: list[str], timeout: int, cwd: Path | None = None) -> bool:
    """Chạy một bước, in đầu ra ngay khi có (log trực tiếp của Kaggle: `kaggle kernels logs -f`) và lưu đầy đủ.

    Lần 1 của bản tập dượt gom đầu ra tới cuối bước, nên suốt 3 giờ của bước log không thấy được gì.
    """
    print(f"=== {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=ENV
    )

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if not NOISE.search(line):
                print(line, end="", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    deadline = t0 + timeout
    while True:
        try:
            rc = proc.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            if STOP.is_set() or time.time() > deadline:
                proc.kill()
                proc.wait()
                rc = -2 if STOP.is_set() else -1
                break
    reader.join(timeout=60)
    text = "".join(lines) + {-1: f"\ntimeout sau {timeout}s\n", -2: "\ndừng theo yêu cầu\n"}.get(rc, "")
    steps = WORK / "steps"
    steps.mkdir(exist_ok=True)
    (steps / f"{name}.txt").write_text(text, encoding="utf-8")
    REPORT["steps"][name] = {"ok": rc == 0, "rc": rc, "seconds": round(time.time() - t0, 1), "tail": text[-3000:]}
    if rc != 0:
        print(f"BƯỚC LỖI {name} rc={rc}", flush=True)
    save()
    return rc == 0


def find_source() -> Path:
    for tgz in Path("/kaggle/input").rglob("ttalloc-src.tar.gz"):
        if SRC.exists():
            shutil.rmtree(SRC)
        with tarfile.open(tgz) as tf:
            tf.extractall(SRC, filter="data")
        return SRC
    for pp in Path("/kaggle/input").rglob("pyproject.toml"):
        if 'name = "ttalloc"' in pp.read_text(encoding="utf-8"):
            if SRC.exists():
                shutil.rmtree(SRC)
            shutil.copytree(pp.parent, SRC)
            return SRC
    listing = [str(p) for p in Path("/kaggle/input").rglob("*")][:50]
    raise FileNotFoundError(f"không thấy mã nguồn ttalloc trong /kaggle/input; có: {listing}")


def collect() -> None:
    """Chép kết quả nhỏ ra /kaggle/working/out để tải về (bỏ tệp > 50 MB và data/questions)."""
    out = WORK / "out"
    for rel in ("runs", "data/logs", "data/estimators", "data/tables", "reports/rehearsal", "configs/locked"):
        src = SRC / rel
        if not src.exists():
            continue
        for p in src.rglob("*"):
            # deer_out/ (đầu ra thô của mã DEER gốc) có đường dẫn > 260 ký tự: tải về Windows bị lỗi, mà phần CPU
            # chạy lại không cần nó.
            if "deer_out" in p.parts:
                continue
            if p.is_file() and p.stat().st_size <= 50_000_000:
                dst = out / p.relative_to(SRC)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
    for rel in ("docs/test-log.md",):
        if (SRC / rel).exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / rel, out / rel)


def main() -> None:
    print(f"RUN_ID {RUN_ID}", flush=True)
    threading.Thread(target=watch_stop, daemon=True).start()
    py = sys.executable
    run("gpu_info", ["nvidia-smi"], 60)
    src = find_source()
    REPORT["source"] = str(src)
    ok = run("pip_vllm", [py, "-m", "pip", "install", "-q", f"vllm=={VLLM_VERSION}"], 5400) and run(
        "pip_ttalloc", [py, "-m", "pip", "install", "-q", "--ignore-requires-python", "-e", str(src)], 1800
    )
    if ok:
        passed: set[str] = set()
        for name, args, needs in STEPS:
            if STOP.is_set():
                REPORT["steps"][name] = {"ok": False, "rc": None, "skipped": "dừng theo yêu cầu"}
                continue
            missing = [n for n in needs if n not in passed]
            if missing:
                REPORT["steps"][name] = {"ok": False, "rc": None, "skipped": f"cần {missing}"}
                print(f"=== {name}: bỏ qua, cần {missing}", flush=True)
                save()
                continue
            try:
                argv = [a() if callable(a) else a for a in args]
            except FileNotFoundError as exc:
                REPORT["steps"][name] = {"ok": False, "rc": None, "error": str(exc)}
                save()
                continue
            if run(name, [py, "-m", "ttalloc.cli", *argv], 3 * 3600, cwd=src):
                passed.add(name)
        collect()
    REPORT["ok"] = all(s["ok"] for s in REPORT["steps"].values())
    REPORT["summary"] = {
        k: ("ok" if v["ok"] else v.get("skipped") or v.get("error") or f"rc={v['rc']}")
        for k, v in REPORT["steps"].items()
    }
    save()
    print(json.dumps(REPORT["summary"], indent=2, ensure_ascii=False))
    stats = {k: re.findall(r"tok/s[^\n]*", v.get("tail", "")) for k, v in REPORT["steps"].items() if k == "throughput"}
    if stats:
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
