"""Server-side supervisor: bounded work, bounded sync, then destroy only this instance."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any


def stop_reason(
    now: float, deadline: float, spent: float, limit: float, margin: float,
    last_progress: float, stall_seconds: float,
) -> str | None:
    if now >= deadline:
        return "deadline"
    if spent >= limit - margin:
        return "budget_margin"
    if now - last_progress >= stall_seconds:
        return "no_generation_or_checkpoint_progress"
    return None


def api(method: str, path: str) -> Any:
    req = urllib.request.Request("https://console.vast.ai/api/v0" + path, method=method)
    req.add_header("Authorization", "Bearer " + os.environ["VAST_API_KEY"])
    with urllib.request.urlopen(req, timeout=15) as response:
        blob = response.read()
        return json.loads(blob) if blob else {}


def bounded_sync(timeout: float) -> bool:
    try:
        p = subprocess.run([sys.executable, "ops/hf_sync.py", "push", "data/logs", "runs", "docs/test-log.md"],
                           timeout=timeout, check=False)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main() -> int:
    iid = os.environ["CONTAINER_ID"]
    if not iid.isdigit():
        raise ValueError("invalid instance ID")
    deadline = float(os.environ["TTALLOC_DEADLINE_EPOCH"])
    start = float(os.environ["TTALLOC_RENT_START_EPOCH"])
    credit_start = float(os.environ["TTALLOC_PHASE_CREDIT_START"])
    limit = float(os.environ["TTALLOC_PHASE_LIMIT_USD"])
    margin = float(os.environ["TTALLOC_BUDGET_MARGIN_USD"])
    dph = float(os.environ["TTALLOC_DPH"])
    stall = float(os.environ["TTALLOC_STALL_SECONDS"])
    poll = float(os.environ["TTALLOC_GUARD_POLL"])
    sync_seconds = float(os.environ["TTALLOC_FINAL_SYNC_SECONDS"])
    job_log = Path(os.environ.get("TTALLOC_JOB_LOG", "/root/job.log"))
    log = job_log.open("a", encoding="utf-8")
    child: subprocess.Popen[str] | None = None
    reason = "startup_failure"
    last_progress = time.time()
    observed = 0.0
    seen: dict[str, float] = {}
    try:
        # Use a separate session so the entire bootstrap/job/engine tree can be terminated.
        child = subprocess.Popen(["bash", "/root/bootstrap.sh"], stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, text=True)
        while True:
            now = time.time()
            # Durable progress, not merely a heartbeat, refreshes the stall timer.
            files = list(Path("runs").glob("**/*checkpoint.json")) + list(Path("data/logs").glob("**/manifest.jsonl"))
            for f in files:
                stamp = f.stat().st_mtime
                if seen.get(str(f)) != stamp:
                    seen[str(f)] = stamp
                    last_progress = now
            if job_log.exists():
                tail = job_log.read_text(encoding="utf-8", errors="replace")[-16000:]
                import re
                # Repeated unchanged tail does not refresh progress.
                matches = re.findall(r".*Avg generation throughput: ([0-9.]+) tokens/s.*", tail)
                key = "generation_tail"
                if matches and float(matches[-1]) > 0 and seen.get(key) != job_log.stat().st_mtime:
                    seen[key] = job_log.stat().st_mtime
                    last_progress = now
            try:
                observed = max(observed, credit_start - float(api("GET", "/users/current/")["credit"]))
            except Exception as exc:
                print(f"credit API unavailable: {type(exc).__name__}; time estimate still enforced", flush=True)
            estimated = max(observed, max(0, now - start) / 3600 * dph)
            reason_or_none = stop_reason(now, deadline - sync_seconds - 30, estimated, limit, margin,
                                         last_progress, stall)
            if reason_or_none:
                reason = reason_or_none
                break
            if child.poll() is not None:
                reason = "done" if child.returncode == 0 else f"job_failed_{child.returncode}"
                break
            time.sleep(poll)
    except BaseException as exc:
        reason = f"supervisor_error_{type(exc).__name__}"
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
        log.close()
        try:
            Path("runs").mkdir(exist_ok=True)
            receipt = {"instance": iid, "reason": reason, "time": time.time(), "observed_spent": observed,
                       "deadline": deadline, "limit_usd": limit}
            Path(f"runs/guard-{iid}.json").write_text(json.dumps(receipt), encoding="utf-8")
            if job_log.exists():
                import shutil
                shutil.copyfile(job_log, f"runs/vast-{os.environ['TTALLOC_JOB']}.log")
            bounded_sync(max(1, min(sync_seconds, deadline - time.time() - 20)))
        except Exception as exc:
            print(f"final save/sync failed: {type(exc).__name__}; destroying anyway", flush=True)
        # API failure cannot leave the rental silently alive: retry destruction indefinitely.
        while True:
            try:
                api("DELETE", f"/instances/{iid}/")
                print(f"destroyed {iid}: {reason}", flush=True)
                break
            except Exception as exc:
                print(f"destroy retry: {type(exc).__name__}", flush=True)
                time.sleep(5)
    return 0 if reason == "done" else 2


if __name__ == "__main__":
    raise SystemExit(main())
