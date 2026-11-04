"""Thuê và điều khiển máy Vast.ai chỉ qua HTTPS (REST API của Vast và Hugging Face), không cần SSH.

Máy GPU tự chạy: onstart tải mã nguồn từ dataset Hugging Face, chạy ops/vast/bootstrap.sh, rồi
ops/vast/jobs/<job>.sh; dữ liệu và trạng thái được đẩy lên Hugging Face định kỳ (ops/hf_sync.py).

Biến môi trường trên máy cá nhân: VAST_API_KEY, HF_TOKEN, TTALLOC_HF_REPO. Không đưa vào git, không dán vào chat.

Cách dùng (từ gốc repo):
    uv run --group ops python ops/vast/vast.py search                       # máy khớp cấu hình, rẻ nhất trước
    uv run --group ops python ops/hf_sync.py push-code                      # đẩy mã nguồn trước khi tạo máy
    uv run --group ops python ops/vast/vast.py create --offer 123 --job p2_log      # hỏi xác nhận, tốn tiền
    uv run --group ops python ops/vast/vast.py status
    uv run --group ops python ops/vast/vast.py logs 456 --tail 200
    uv run --group ops python ops/vast/vast.py watch 456 --job p2_log --destroy-when-done
    uv run --group ops python ops/vast/vast.py destroy 456                  # xóa hẳn, không chỉ dừng
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "ops" / "vast" / "_state" / "instances.json"
API = "https://console.vast.ai/api/v0"
IMAGE = "vllm/vllm-openai:v0.32.0"  # phiên bản chốt ở ADR-002
JOBS_DIR = ROOT / "ops" / "vast" / "jobs"

# Cấu hình thuê (Ke-hoach-thuc-hien-de-tai.md mục 3.1).
DEFAULT_QUERY: dict[str, Any] = {
    "gpu_name": {"eq": "RTX 3090"},
    "num_gpus": {"eq": 1},
    "rentable": {"eq": True},
    "rented": {"eq": False},
    "verified": {"eq": True},
    "reliability2": {"gte": 0.98},
    "gpu_mem_bw": {"gte": 780},
    "cuda_max_good": {"gte": 13.0},
    "cpu_cores_effective": {"gte": 8},
    "cpu_ram": {"gte": 32 * 1024},
    "inet_down": {"gte": 300},
    "disk_space": {"gte": 60},
}

ONSTART = r"""#!/bin/bash
cd /root
python3 -m pip install -q "huggingface_hub>=0.25" > /root/onstart.log 2>&1
python3 - >> /root/onstart.log 2>&1 <<'EOF'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["TTALLOC_HF_REPO"], "code/ttalloc-src.tar.gz", repo_type="dataset",
                    token=os.environ["HF_TOKEN"], force_download=True)
tarfile.open(p).extractall("/root/ttalloc", filter="data")
EOF
cd /root/ttalloc && sed 's/\r$//' ops/vast/bootstrap.sh > /root/bootstrap.sh
nohup bash /root/bootstrap.sh > /root/job.log 2>&1 &
"""

# Mã thoát của `watch`, để agent đang theo dõi biết phải làm gì.
EXIT_DONE, EXIT_NEEDS_FIX, EXIT_GONE, EXIT_ANOMALY, EXIT_STALLED = 0, 2, 3, 4, 5


def _key() -> str:
    k = os.environ.get("VAST_API_KEY")
    if not k:
        raise SystemExit("thiếu VAST_API_KEY (Vast console → Account → API Keys); đặt làm biến môi trường của user")
    return k


def call(method: str, path: str, body: dict[str, Any] | None = None, query: str = "") -> Any:
    url = f"{API}{path}{query}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {_key()}")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Vast {method} {path}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:500]}") from e
    return json.loads(text) if text else {}


def build_query(max_dph: float, interruptible: bool, limit: int) -> dict[str, Any]:
    q = dict(DEFAULT_QUERY)
    q["dph_total"] = {"lte": max_dph}
    q["order"] = [["dph_total", "asc"]]
    q["type"] = "bid" if interruptible else "on-demand"
    q["limit"] = limit
    return q


def search(max_dph: float, interruptible: bool, limit: int) -> list[dict[str, Any]]:
    res = call("POST", "/bundles/", build_query(max_dph, interruptible, limit))
    offers: list[dict[str, Any]] = res.get("offers", [])
    return offers


def job_names() -> list[str]:
    return sorted(p.stem for p in JOBS_DIR.glob("*.sh") if p.stem != "common")


def job_env(job: str, approved_by: str | None, self_destroy: bool) -> dict[str, str]:
    if job not in job_names():
        raise SystemExit(f"không có công việc {job}: các công việc có sẵn {job_names()}")
    env = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "TTALLOC_HF_REPO": os.environ.get("TTALLOC_HF_REPO", ""),
        "TTALLOC_JOB": job,
    }
    missing = [k for k, v in env.items() if not v]
    if missing:
        raise SystemExit(f"thiếu biến môi trường {missing}")
    if approved_by:
        env["TTALLOC_APPROVED_BY"] = approved_by
    if self_destroy:
        env["TTALLOC_SELF_DESTROY"] = "1"
        env["VAST_API_KEY"] = _key()
    return env


def create_body(job: str, env: dict[str, str], disk: int, price: float | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "client_id": "me",
        "image": IMAGE,
        "env": env,
        "disk": disk,
        "label": f"ttalloc-{job}",
        "onstart": ONSTART,
        "runtype": "ssh",
    }
    if price is not None:
        body["price"] = price  # giá đặt cho máy có thể bị ngắt (bid)
    return body


def _save_state(entry: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []
    rows.append(entry)
    STATE.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def instances() -> list[dict[str, Any]]:
    res = call("GET", "/instances/", query="?owner=me")
    rows: list[dict[str, Any]] = res.get("instances", [])
    return rows


def logs(iid: int, tail: int) -> str:
    res = call("PUT", f"/instances/request_logs/{iid}/", {"tail": str(tail)})
    url = res.get("result_url")
    if not url:
        return str(res)
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return str(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError:
            continue
    return f"chưa lấy được log: {res.get('msg', '')}"


def destroy(iid: int) -> None:
    call("DELETE", f"/instances/{iid}/", {})
    print(f"đã xóa máy {iid}")


def _ops() -> Any:
    sys.path.insert(0, str(ROOT / "ops"))
    import hf_sync
    import watch_rules

    return hf_sync, watch_rules


def watch(iid: int, job: str, destroy_when_done: bool, poll: int, fail_fast: bool, stall_min: int) -> int:
    """Theo dõi trạng thái và đuôi log (máy đẩy lên mỗi vài phút); in dòng log mới; thoát sớm khi cần người sửa.

    Mã thoát: 0 xong; 2 công việc lỗi (máy đang chờ bản sửa, hoặc đã thoát); 3 máy không còn; 4 log có dấu hiệu
    sự cố (với --fail-fast); 5 log đứng yên quá --stall-min phút.
    """
    hf, rules = _ops()
    prev: list[str] = []
    last_change = time.time()
    while True:
        st = hf.get_status(job) or {}
        inst = {int(r["id"]): r for r in instances()}.get(iid)
        cur = hf.get_log(job).splitlines()
        fresh = rules.new_lines(prev, cur)
        if fresh:
            last_change = time.time()
            print("\n".join(fresh))
        prev = cur
        state = st.get("state", "?")
        print(
            f"--- {time.strftime('%H:%M:%S')} máy {iid}: {inst and inst.get('actual_status')} | {job}: {state} "
            f"{st.get('msg', '')}",
            flush=True,
        )
        if state == "done":
            if destroy_when_done and inst is not None:
                destroy(iid)
            print("tải kết quả: uv run --group ops python ops/hf_sync.py pull data runs docs/test-log.md")
            return EXIT_DONE
        if state in ("failed", "waiting_fix"):
            if state == "failed" and destroy_when_done and inst is not None:
                destroy(iid)
            print(
                f"công việc lỗi ({state}). Sửa mã, chạy test CPU, rồi `python ops/hf_sync.py push-code`: "
                "máy đang chờ sẽ tự cài lại và chạy tiếp."
            )
            return EXIT_NEEDS_FIX
        if inst is None:
            print("máy không còn (bị thu hồi hoặc đã xóa); tạo máy mới với cùng --job để chạy tiếp")
            return EXIT_GONE
        hits = rules.scan(fresh)
        if fail_fast and hits:
            for name, line in hits:
                print(f"!!! {name}: {line}")
            return EXIT_ANOMALY
        if stall_min and time.time() - last_change > stall_min * 60:
            print(f"!!! log không đổi trong {stall_min} phút")
            return EXIT_STALLED
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("search")
    p.add_argument("--max-dph", type=float, default=0.40, help="giá tối đa USD/giờ, đã gồm ổ đĩa")
    p.add_argument("--interruptible", action="store_true", help="máy có thể bị ngắt (rẻ hơn; logger chạy tiếp được)")
    p.add_argument("--limit", type=int, default=10)
    p = sub.add_parser("create")
    p.add_argument("--offer", type=int, required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--disk", type=int, default=60)
    p.add_argument("--price", type=float, help="giá đặt cho máy có thể bị ngắt")
    p.add_argument("--approved-by", help="người duyệt cho công việc chạm test_* (mục 7.5)")
    p.add_argument("--self-destroy", action="store_true", help="máy tự xóa khi xong (gửi VAST_API_KEY lên máy)")
    p.add_argument("--yes", action="store_true", help="bỏ hỏi xác nhận (chỉ dùng khi người dùng đã đồng ý thuê)")
    sub.add_parser("status")
    p = sub.add_parser("logs")
    p.add_argument("id", type=int)
    p.add_argument("--tail", type=int, default=200)
    p = sub.add_parser("watch")
    p.add_argument("id", type=int)
    p.add_argument("--job", required=True)
    p.add_argument("--destroy-when-done", action="store_true")
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--fail-fast", action="store_true", help="thoát ngay khi log có dấu hiệu sự cố (ops/watch_rules.py)")
    p.add_argument("--stall-min", type=int, default=40, help="thoát nếu log đứng yên quá bấy nhiêu phút; 0 để tắt")
    p = sub.add_parser("destroy")
    p.add_argument("id", type=int)
    a = ap.parse_args()

    if a.cmd == "search":
        rows = search(a.max_dph, a.interruptible, a.limit)
        print(f"{'offer':>10} {'$/h':>6} {'rel':>5} {'bw':>5} {'cuda':>5} {'cpu':>4} {'ram':>4} {'net':>5}  nơi")
        for o in rows:
            print(
                f"{o['id']:>10} {o.get('dph_total', 0):6.3f} {o.get('reliability2', 0):5.3f} "
                f"{o.get('gpu_mem_bw', 0):5.0f} {o.get('cuda_max_good', 0):5.1f} "
                f"{o.get('cpu_cores_effective', 0):4.0f} "
                f"{o.get('cpu_ram', 0) / 1024:4.0f} {o.get('inet_down', 0):5.0f}  {o.get('geolocation', '')}"
            )
        if not rows:
            print("không có máy khớp; nới --max-dph hoặc thử --interruptible")
    elif a.cmd == "create":
        env = job_env(a.job, a.approved_by, a.self_destroy)
        body = create_body(a.job, env, a.disk, a.price)
        shown = {k: ("***" if k in ("HF_TOKEN", "VAST_API_KEY") else v) for k, v in env.items()}
        print(
            json.dumps({"offer": a.offer, "image": IMAGE, "disk": a.disk, "env": shown}, indent=2, ensure_ascii=False)
        )
        if not a.yes and input("Thuê máy này (tốn tiền)? gõ yes để tiếp tục: ").strip().lower() != "yes":
            raise SystemExit("đã hủy")
        res = call("PUT", f"/asks/{a.offer}/", body)
        iid = res.get("new_contract")
        _save_state({"id": iid, "offer": a.offer, "job": a.job, "time": time.strftime("%Y-%m-%d %H:%M")})
        print(json.dumps(res, indent=2))
        print(f"máy {iid}: theo dõi bằng `vast.py watch {iid} --job {a.job} --destroy-when-done`")
    elif a.cmd == "status":
        for r in instances():
            print(
                f"{r['id']} {r.get('label')} {r.get('actual_status')} {r.get('gpu_name')} "
                f"${r.get('dph_total', 0):.3f}/h start={r.get('start_date')}"
            )
    elif a.cmd == "logs":
        print(logs(a.id, a.tail))
    elif a.cmd == "watch":
        sys.exit(watch(a.id, a.job, a.destroy_when_done, a.poll, a.fail_fast, a.stall_min))
    elif a.cmd == "destroy":
        destroy(a.id)


if __name__ == "__main__":
    main()
