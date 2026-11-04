"""Đồng bộ mã nguồn, dữ liệu và kết quả qua một dataset riêng tư trên Hugging Face (chỉ cần HTTPS, không cần SSH).

Dùng ở cả máy cá nhân lẫn máy GPU (Vast, Kaggle). Cần hai biến môi trường:
    HF_TOKEN          token có quyền ghi (không đưa vào git, không dán vào chat)
    TTALLOC_HF_REPO   tên dataset, ví dụ <user>/ttalloc-data (tạo tự động ở chế độ riêng tư nếu chưa có)

Cách dùng (từ gốc repo):
    python ops/hf_sync.py push-code                       # đóng gói các tệp git theo dõi -> code/ttalloc-src.tar.gz
    python ops/hf_sync.py push data/logs/qwen3_1_7b runs   # tải thư mục lên, giữ nguyên đường dẫn
    python ops/hf_sync.py pull data/logs runs              # tải về đúng chỗ
    python ops/hf_sync.py status p2_log                    # trạng thái công việc trên máy GPU
    python ops/hf_sync.py set-status p2_log running "bước 1/3"
    python ops/hf_sync.py tail p2_log -n 80                # đuôi log máy GPU đẩy lên mỗi vài phút
    python ops/hf_sync.py code-rev                         # bản mã nguồn hiện có (máy GPU dùng để biết có bản sửa)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE_PATH = "code/ttalloc-src.tar.gz"
# Không bao giờ đẩy các tệp này lên, kể cả khi lỡ có trong cây thư mục.
NEVER = (".env", "kaggle.json", ".git/", "ops/kaggle/_build/", "ops/vast/_state/")


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"thiếu biến môi trường {name} (xem ops/RUNBOOK.md mục 0)")
    return v


def _api() -> Any:
    from huggingface_hub import HfApi

    api = HfApi(token=_env("HF_TOKEN"))
    api.create_repo(_env("TTALLOC_HF_REPO"), repo_type="dataset", private=True, exist_ok=True)
    return api


def _blocked(rel: str) -> bool:
    return any(rel == n or rel.startswith(n) or rel.endswith("/" + n) for n in NEVER)


def git_files(root: Path = ROOT) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"], cwd=root, capture_output=True, text=True, check=True
    )
    return [f for f in out.stdout.splitlines() if f and (root / f).is_file() and not _blocked(f)]


def code_tarball(root: Path = ROOT) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in git_files(root):
            tf.add(root / f, arcname=f)
    return buf.getvalue()


def push_code() -> None:
    blob = code_tarball()
    _api().upload_file(
        path_or_fileobj=blob,
        path_in_repo=CODE_PATH,
        repo_id=_env("TTALLOC_HF_REPO"),
        repo_type="dataset",
        commit_message=f"code {time.strftime('%Y-%m-%d %H:%M')}",
    )
    print(f"đã đẩy {CODE_PATH} ({len(blob) / 1e6:.1f} MB)")


def push(paths: list[str], message: str | None = None) -> None:
    api = _api()
    for p in paths:
        local = ROOT / p
        if not local.exists():
            print(f"bỏ qua {p}: không có")
            continue
        rel = local.resolve().relative_to(ROOT).as_posix()
        if _blocked(rel):
            raise SystemExit(f"từ chối đẩy {rel}")
        if local.is_file():
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=rel,
                repo_id=_env("TTALLOC_HF_REPO"),
                repo_type="dataset",
                commit_message=message or f"sync {rel}",
            )
        else:
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=rel,
                repo_id=_env("TTALLOC_HF_REPO"),
                repo_type="dataset",
                commit_message=message or f"sync {rel}",
                ignore_patterns=["*.tmp", "**/__pycache__/**"],
            )
        print(f"đã đẩy {rel}")


def pull(paths: list[str]) -> None:
    from huggingface_hub import snapshot_download

    pats = [f"{p.rstrip('/')}/**" for p in paths] + list(paths)
    snapshot_download(
        _env("TTALLOC_HF_REPO"), repo_type="dataset", allow_patterns=pats, local_dir=str(ROOT), token=_env("HF_TOKEN")
    )
    print(f"đã tải {paths} về {ROOT}")


def set_status(job: str, state: str, msg: str = "") -> None:
    body = {
        "job": job,
        "state": state,
        "msg": msg,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.environ.get("CONTAINER_ID") or platform.node(),
    }
    _api().upload_file(
        path_or_fileobj=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        path_in_repo=f"status/{job}.json",
        repo_id=_env("TTALLOC_HF_REPO"),
        repo_type="dataset",
        commit_message=f"status {job} {state}",
    )


def get_status(job: str) -> dict[str, Any] | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        p = hf_hub_download(
            _env("TTALLOC_HF_REPO"),
            f"status/{job}.json",
            repo_type="dataset",
            token=_env("HF_TOKEN"),
            force_download=True,
        )
    except EntryNotFoundError:
        return None
    out: dict[str, Any] = json.loads(Path(p).read_text(encoding="utf-8"))
    return out


def code_rev() -> str:
    """Mã nhận diện bản mã nguồn hiện có trên Hugging Face (đổi mỗi lần push-code); rỗng nếu chưa có."""
    infos = _api().get_paths_info(_env("TTALLOC_HF_REPO"), [CODE_PATH], repo_type="dataset")
    if not infos:
        return ""
    f = infos[0]
    lfs = getattr(f, "lfs", None)
    return str(getattr(lfs, "sha256", None) or getattr(f, "blob_id", ""))


def pull_code(dest: Path) -> None:
    """Giải nén bản mã nguồn mới nhất đè lên `dest` (giữ data/ và runs/, vốn không có trong gói)."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        _env("TTALLOC_HF_REPO"), CODE_PATH, repo_type="dataset", token=_env("HF_TOKEN"), force_download=True
    )
    with tarfile.open(p) as tf:
        tf.extractall(dest, filter="data")
    print(f"đã giải nén mã nguồn vào {dest}")


def put_log(job: str, path: Path, lines: int = 400) -> None:
    """Đẩy `lines` dòng cuối của log công việc lên status/<job>.log để máy cá nhân theo dõi khi đang chạy."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:] if path.exists() else []
    _api().upload_file(
        path_or_fileobj="\n".join(text).encode("utf-8"),
        path_in_repo=f"status/{job}.log",
        repo_id=_env("TTALLOC_HF_REPO"),
        repo_type="dataset",
        commit_message=f"log {job}",
    )


def get_log(job: str) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        p = hf_hub_download(
            _env("TTALLOC_HF_REPO"),
            f"status/{job}.log",
            repo_type="dataset",
            token=_env("HF_TOKEN"),
            force_download=True,
        )
    except EntryNotFoundError:
        return ""
    return Path(p).read_text(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push-code")
    sub.add_parser("code-rev")
    p = sub.add_parser("pull-code")
    p.add_argument("dest", type=Path)
    p = sub.add_parser("put-log")
    p.add_argument("job")
    p.add_argument("file", type=Path)
    p = sub.add_parser("tail")
    p.add_argument("job")
    p.add_argument("-n", type=int, default=80)
    p = sub.add_parser("push")
    p.add_argument("paths", nargs="+")
    p.add_argument("-m", "--message")
    p = sub.add_parser("pull")
    p.add_argument("paths", nargs="+")
    p = sub.add_parser("status")
    p.add_argument("job")
    p = sub.add_parser("set-status")
    p.add_argument("job")
    p.add_argument("state", choices=["starting", "running", "waiting_fix", "done", "failed"])
    p.add_argument("msg", nargs="?", default="")
    a = ap.parse_args()
    if a.cmd == "push-code":
        push_code()
    elif a.cmd == "code-rev":
        print(code_rev())
    elif a.cmd == "pull-code":
        pull_code(a.dest)
    elif a.cmd == "put-log":
        put_log(a.job, a.file)
    elif a.cmd == "tail":
        print("\n".join(get_log(a.job).splitlines()[-a.n :]))
    elif a.cmd == "push":
        push(a.paths, a.message)
    elif a.cmd == "pull":
        pull(a.paths)
    elif a.cmd == "status":
        print(json.dumps(get_status(a.job), indent=2, ensure_ascii=False))
    else:
        set_status(a.job, a.state, a.msg)


if __name__ == "__main__":
    sys.exit(main())
