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
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "ops" / "kaggle" / "_build"
DATASET_SLUG = "ttalloc-src"
NOISE = re.compile(r"\d+%\|.*(it/s|s/it)\]")  # thanh tiến trình tqdm


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


class LiveLog:
    """Log của phiên đang chạy qua `kaggle kernels logs -f` (`kernels output` chỉ có sau khi kernel xong).

    Mỗi lần kết nối, CLI phát lại log từ đầu nên số thứ tự dòng không đổi; `fresh()` trả các dòng chưa trả, và nối lại
    khi luồng bị ngắt.
    """

    def __init__(self, cmd: list[str]) -> None:
        self.cmd = cmd
        self.lines: list[str] = []  # lần kết nối hiện tại
        self.emitted = 0
        self.reader: threading.Thread | None = None
        self.proc: subprocess.Popen[str] | None = None

    def _connect(self) -> None:
        lines: list[str] = []
        self.lines = lines
        proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
        self.proc = proc

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line.rstrip("\n"))

        self.reader = threading.Thread(target=pump, daemon=True)
        self.reader.start()

    def fresh(self) -> list[str]:
        out: list[str] = []
        if self.reader is not None:
            cur = list(self.lines)
            out = cur[self.emitted :]
            self.emitted = max(self.emitted, len(cur))
            if not self.reader.is_alive():
                self.reader = None
        if self.reader is None:
            self._connect()
        return out

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()


def fetch(kernel: str, name: str) -> Path:
    out_dir = BUILD / f"out-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel, "-p", str(out_dir), "-o", check=False)
    return out_dir


def report_run_id(out_dir: Path, name: str) -> str | None:
    try:
        return json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8")).get("run_id")
    except (OSError, json.JSONDecodeError):
        return None


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
    return {
        k: str(v.get("error") or v.get("skipped") or v.get("summary") or f"rc={v.get('rc')}")
        for k, v in steps.items()
        if not v.get("ok")
    }


def wait_and_fetch(kernel: str, name: str, poll: int = 60, fail_fast: bool = False) -> int:
    """Theo log trực tiếp mỗi `poll` giây khi kernel đang chạy và in dòng mới; xong thì tải báo cáo, in bước lỗi.

    Mã thoát: 0 kernel xong và mọi bước đạt; 2 có bước lỗi; 4 log có dấu hiệu sự cố (với --fail-fast, kernel vẫn chạy
    tiếp trên Kaggle); 1 kernel lỗi hoặc bị hủy.
    """
    sys.path.insert(0, str(ROOT / "ops"))
    import watch_rules

    rid_file = RUN_ID_FILE(name)
    run_id = rid_file.read_text(encoding="utf-8").strip() if rid_file.exists() else None
    exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
    live = LiveLog([exe, "kernels", "logs", "-f", kernel])
    current = run_id is None
    try:
        while True:
            status = kaggle("kernels", "status", kernel, check=False)
            state = re.search(r"(complete|error|cancel\w*|running|queued)", status.lower())
            fresh = live.fresh()
            if not current:  # bỏ log của phiên bản trước cho tới khi thấy mã lần chạy này
                hit = [i for i, x in enumerate(fresh) if f"RUN_ID {run_id}" in x]
                current = bool(hit)
                fresh = fresh[hit[0] :] if hit else []
            shown = [x for x in fresh if not NOISE.search(x)]
            if shown:
                print("\n".join(shown[-80:]))
            print(f"--- {time.strftime('%H:%M:%S')} {status.strip()}", flush=True)
            if state and state.group(1) not in ("running", "queued"):
                # Output (báo cáo, steps/, out/) chỉ có khi kernel đã xong, và có khi vài chục giây đầu vẫn là của bản
                # trước (bản tập dượt lần 4: watcher tải báo cáo lần 3 rồi báo đạt). Chỉ tin báo cáo đúng RUN_ID.
                for _ in range(10):
                    out_dir = fetch(kernel, name)
                    if report_run_id(out_dir, name) == run_id or run_id is None:
                        break
                    time.sleep(20)
                else:
                    print(f"!!! chưa tải được báo cáo của lần chạy {run_id}; xem {out_dir}")
                    return 1
                fails = failed_steps(out_dir, name, run_id)
                for k in sorted(fails):
                    print(f"!!! bước {k} lỗi: {fails[k][:400]}")
                done_ok = state.group(1) == "complete" and not fails
                print(f"kết quả ở {out_dir}")
                return 0 if done_ok else (2 if fails else 1)
            hits = watch_rules.scan(fresh)
            if fail_fast and hits:
                for n, line in hits:
                    print(f"!!! {n}: {line}")
                return 2 if any(n == "step_failed" for n, _ in hits) else 4
            time.sleep(poll)
    finally:
        live.close()


def request_stop(name: str) -> None:
    """Ghi RUN_ID của lần chạy gần nhất vào ops/kaggle/stop.txt rồi commit và push; kernel đọc tệp này mỗi phút.

    API Kaggle chỉ hủy được phiên khi biết kernel_session_id, mà API không trả id của phiên đang chạy.
    """
    rid_file = RUN_ID_FILE(name)
    if not rid_file.exists():
        raise SystemExit(f"không có {rid_file}: chưa đẩy {name} từ máy này")
    run_id = rid_file.read_text(encoding="utf-8").strip()
    stop = ROOT / "ops" / "kaggle" / "stop.txt"
    lines = stop.read_text(encoding="utf-8").splitlines() if stop.exists() else []
    if run_id not in lines:
        stop.write_text("\n".join([*lines, run_id]) + "\n", encoding="utf-8", newline="\n")
    rel = stop.relative_to(ROOT).as_posix()
    subprocess.run(["git", "add", rel], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"Stop Kaggle run {run_id}", "--", rel], cwd=ROOT, check=False)
    subprocess.run(["git", "push", "-q"], cwd=ROOT, check=False)
    print(f"đã yêu cầu dừng {name} {run_id}; kernel sẽ dừng trong vài phút (raw.githubusercontent.com có thể trễ)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="tên script trong ops/kaggle, ví dụ p0_smoke")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--watch-only", action="store_true", help="không đẩy gì, chỉ theo dõi kernel đã đẩy")
    ap.add_argument("--fail-fast", action="store_true", help="thoát ngay khi có bước lỗi hoặc log có dấu hiệu sự cố")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--stop", action="store_true", help="yêu cầu lần chạy gần nhất tự dừng (ops/kaggle/stop.txt)")
    args = ap.parse_args()
    if args.stop:
        request_stop(args.name)
        return
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
