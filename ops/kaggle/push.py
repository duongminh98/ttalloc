"""Đẩy mã nguồn và một script lên Kaggle, chạy trên GPU, chờ xong rồi tải kết quả về.

Cách dùng (từ gốc repo, cần Kaggle API token):
    uv run --group ops python ops/kaggle/push.py p0_smoke
    uv run --group ops python ops/kaggle/push.py p0_smoke --no-upload   # dùng lại bản mã nguồn đã đẩy
    uv run --group ops python ops/kaggle/push.py p0_smoke --no-wait     # đẩy xong là thoát

Mã nguồn được đóng gói thành dataset riêng tư `<user>/ttalloc-src` (tệp ttalloc-src.tar.gz), nên không cần
GitHub trên Kaggle. Script chạy là `ops/kaggle/<tên>.py`; kernel tên `<user>/ttalloc-<tên>`, bật GPU (T4) và internet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "ops" / "kaggle" / "_build"
DATASET_SLUG = "ttalloc-src"


def kaggle(*args: str, check: bool = True) -> str:
    exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
    out = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = (out.stdout or "") + (out.stderr or "")
    if check and out.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} lỗi (mã {out.returncode}):\n{text}")
    return text


def kaggle_user() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if cfg.exists():
        return str(json.loads(cfg.read_text(encoding="utf-8"))["username"])
    if os.environ.get("KAGGLE_API_TOKEN"):  # token kiểu mới (KGAT_...): hỏi CLI tên người dùng
        m = re.search(r"username:\s*(\S+)", kaggle("config", "view", check=False))
        if m and m.group(1) != "None":
            return m.group(1)
    raise SystemExit(
        "Chưa có Kaggle API token: đặt KAGGLE_API_TOKEN, hoặc KAGGLE_USERNAME và KAGGLE_KEY, hoặc ~/.kaggle/kaggle.json"
    )


def git_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [f for f in out.stdout.splitlines() if f and (ROOT / f).is_file() and not f.startswith("ops/kaggle/_build")]


def upload_source(user: str) -> None:
    ds_dir = BUILD / "dataset"
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    ds_dir.mkdir(parents=True)
    with tarfile.open(ds_dir / f"{DATASET_SLUG}.tar.gz", "w:gz") as tf:
        for f in git_files():
            tf.add(ROOT / f, arcname=f)
    meta = {"title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}
    (ds_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    status = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).lower()
    if "ready" in status or "pending" in status:
        stamp = time.strftime("%Y-%m-%d %H:%M")
        print(kaggle("datasets", "version", "-p", str(ds_dir), "-m", f"ttalloc {stamp}"))
    else:
        print(kaggle("datasets", "create", "-p", str(ds_dir)))
    for _ in range(60):
        status = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).lower()
        if "ready" in status:
            return
        time.sleep(10)
    raise SystemExit(f"dataset {user}/{DATASET_SLUG} chưa sẵn sàng sau 10 phút: {status}")


def RUN_ID_FILE(name: str) -> Path:
    return BUILD / f"run_id-{name}.txt"


def inject_run_id(src: str, run_id: str) -> str:
    """Chèn `TTALLOC_RUN_ID` ngay sau `from __future__`, để script ghi nó vào log và báo cáo.

    Trong lúc phiên bản mới đang chạy, `kaggle kernels output` vẫn trả kết quả của phiên bản trước; trình theo dõi
    chỉ tin log và báo cáo mang đúng mã lần chạy này.
    """
    line = f'TTALLOC_RUN_ID = "{run_id}"\n'
    marker = "from __future__ import annotations\n"
    if marker in src:
        return src.replace(marker, marker + line, 1)
    return line + src


def push_kernel(user: str, name: str) -> str:
    script = ROOT / "ops" / "kaggle" / f"{name}.py"
    if not script.exists():
        raise SystemExit(f"không có {script}")
    slug = f"ttalloc-{name.replace('_', '-')}"
    k_dir = BUILD / f"kernel-{name}"
    if k_dir.exists():
        shutil.rmtree(k_dir)
    k_dir.mkdir(parents=True)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    (k_dir / script.name).write_text(inject_run_id(script.read_text(encoding="utf-8"), run_id), encoding="utf-8")
    RUN_ID_FILE(name).write_text(run_id, encoding="utf-8")
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": script.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (k_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(kaggle("kernels", "push", "-p", str(k_dir)))
    return f"{user}/{slug}"


def log_lines(path: Path) -> list[str]:
    """Log của kernel là một mảng JSON {stream_name, time, data}; lúc đang chạy có thể chưa đóng ngoặc."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        items = json.loads(raw if raw.endswith("]") else raw.rstrip(",") + "]")
    except json.JSONDecodeError:
        return raw.splitlines()
    return "".join(str(it.get("data", "")) for it in items).splitlines()


def fetch(kernel: str, name: str) -> Path:
    out_dir = BUILD / f"out-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel, "-p", str(out_dir), "-o", check=False)
    return out_dir


def failed_steps(out_dir: Path, name: str, run_id: str | None = None) -> dict[str, str]:
    report = out_dir / f"{name}.json"
    if not report.exists():
        return {}
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if run_id is not None and data.get("run_id") != run_id:
        return {}  # báo cáo của phiên bản trước
    steps = data.get("steps", {})
    return {k: str(v.get("error") or v.get("summary") or v.get("rc")) for k, v in steps.items() if not v.get("ok")}


def wait_and_fetch(kernel: str, name: str, poll: int = 60, fail_fast: bool = False) -> int:
    """Tải log và báo cáo mỗi `poll` giây ngay khi kernel đang chạy; in dòng log mới và bước lỗi.

    Mã thoát: 0 kernel xong và mọi bước đạt; 2 có bước lỗi; 4 log có dấu hiệu sự cố (với --fail-fast, kernel vẫn chạy
    tiếp trên Kaggle); 1 kernel lỗi hoặc bị hủy.
    """
    sys.path.insert(0, str(ROOT / "ops"))
    import watch_rules

    rid_file = RUN_ID_FILE(name)
    run_id = rid_file.read_text(encoding="utf-8").strip() if rid_file.exists() else None
    prev: list[str] = []
    seen_fail: set[str] = set()
    while True:
        status = kaggle("kernels", "status", kernel, check=False)
        state = re.search(r"(complete|error|cancel\w*|running|queued)", status.lower())
        out_dir = fetch(kernel, name)
        cur = log_lines(out_dir / f"{kernel.split('/')[1]}.log")
        current = run_id is None or any(f"RUN_ID {run_id}" in x for x in cur)
        if not current:
            cur = []  # output của phiên bản trước; phiên bản này chưa ghi gì
        fresh = watch_rules.new_lines(prev, cur)
        prev = cur
        if fresh:
            print("\n".join(fresh[-60:]))
        fails = failed_steps(out_dir, name, run_id) if current else {}
        for k in sorted(set(fails) - seen_fail):
            print(f"!!! bước {k} lỗi: {fails[k][:400]}")
        seen_fail |= set(fails)
        print(f"--- {time.strftime('%H:%M:%S')} {status.strip()}", flush=True)
        if state and state.group(1) not in ("running", "queued"):
            done_ok = state.group(1) == "complete" and not fails
            print(f"kết quả ở {out_dir}")
            return 0 if done_ok else (2 if fails else 1)
        if fail_fast and (fails or watch_rules.scan(fresh)):
            for n, line in watch_rules.scan(fresh):
                print(f"!!! {n}: {line}")
            return 2 if fails else 4
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="tên script trong ops/kaggle, ví dụ p0_smoke")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--watch-only", action="store_true", help="không đẩy gì, chỉ theo dõi kernel đã đẩy")
    ap.add_argument("--fail-fast", action="store_true", help="thoát ngay khi có bước lỗi hoặc log có dấu hiệu sự cố")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()
    user = kaggle_user()
    kernel = f"{user}/ttalloc-{args.name.replace('_', '-')}"
    if not args.watch_only:
        if not args.no_upload:
            upload_source(user)
        kernel = push_kernel(user, args.name)
        if args.no_wait:
            print(f"Đã đẩy {kernel}. Theo dõi: uv run --group ops python ops/kaggle/push.py {args.name} --watch-only")
            return
    rc = wait_and_fetch(kernel, args.name, args.poll, args.fail_fast)
    report = BUILD / f"out-{args.name}" / f"{args.name}.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        print(json.dumps({k: v.get("ok") for k, v in data.get("steps", {}).items()}, indent=2, ensure_ascii=False))
        print(f"Báo cáo đầy đủ: {report}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
