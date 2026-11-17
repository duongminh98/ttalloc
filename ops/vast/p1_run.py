"""Checkpointed P1 stage driver; no automatic rental, retuning or test retries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def stages(approved_by: str) -> list[tuple[str, list[str]]]:
    cfg = ["--exp", "configs/experiments/p1_safe.yaml"]
    approval = ["--approved-by", approved_by]
    test = ["--split", "test_main", "--profile", "deer_paper"]
    out = []
    for ds, seeds in (("math500", "0"), ("aime_2024", "0-3")):
        for method in ("full", "deer"):
            out.append((f"{method}_{ds}", ["baseline", method, *cfg, *test, "--datasets", ds,
                                         "--seeds", seeds, *approval]))
    out += [
        ("throughput", ["baseline", "throughput", *cfg, "--n", "256"]),
        ("equivalence", ["equivalence", *cfg, "--n", "400", "--seeds", "0-1"]),
        ("pilot", ["log", *cfg, "--split", "val", "--limit", "100", "--seeds", "0-1"]),
        ("validate", ["validate-logs", *cfg, "--split", "val"]),
        ("answer_mode", ["study", "answer-mode", *cfg, "--n", "100", "--seeds", "0"]),
        ("probe_noise", ["study", "probe-noise", *cfg, "--n", "100", "--seeds", "0"]),
    ]
    return out


def stage_gate(name: str, result: Path | None) -> str | None:
    if name in ("full_math500", "deer_math500"):
        if result is None:
            return "missing baseline run directory"
        obj = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        if obj["n"] != 500 or abs(float(obj["accuracy"]) - .856) > .020000001:
            return f"MATH baseline outside PLAN tolerance: {obj['accuracy']}"
    if name == "equivalence":
        if result is None:
            return "missing equivalence directory"
        obj = json.loads((result / "equivalence.json").read_text(encoding="utf-8"))
        if obj["n_questions"] != 400 or not obj["pass"]:
            return "equivalence not established; no automatic extra seeds"
    return None


def save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    session = os.environ["TTALLOC_P1_SESSION"]
    if not session.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid session")
    by = os.environ["TTALLOC_APPROVED_BY"]
    path = ROOT / "runs" / session / "stages.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    for name, argv in stages(by):
        old = state["stages"].get(name, {})
        if old.get("done"):
            continue
        # Only reuse verified row checkpoints. New attempt always gets a new run directory.
        prior = Path(old["run"]) if old.get("run") else None
        if prior and (prior / ("full_checkpoint.json" if name == "equivalence" else "checkpoint.json")).exists():
            argv += ["--resume-from", str(prior)]
        state["stages"][name] = {"started": time.time(), "argv": argv, "done": False}
        save(path, state)
        print(f"P1_STAGE {name}", flush=True)
        subprocess.run([sys.executable, "ops/hf_sync.py", "set-status", "p1_all", "running", name], check=False)
        result: Path | None = None
        child = subprocess.Popen(["ttalloc", *argv], cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1)
        assert child.stdout is not None
        with (path.parent / f"{name}.log").open("a", encoding="utf-8") as log:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith("thư mục chạy: "):
                    result = Path(line.partition(": ")[2].strip())
                    state["stages"][name]["run"] = str(result)
                    save(path, state)
        rc = child.wait()
        reason = stage_gate(name, result) if rc == 0 else f"command failed rc={rc}"
        state["stages"][name].update({"finished": time.time(), "rc": rc, "done": reason is None, "stop": reason})
        save(path, state)
        # Keep completed stages in durable remote storage before proceeding.
        synced = subprocess.run([sys.executable, "ops/hf_sync.py", "push", "runs", "docs/test-log.md"])
        if synced.returncode:
            print("P1_STOP synchronization failed", flush=True)
            return 2
        if reason:
            print(f"P1_STOP {reason}", flush=True)
            return 2
    print("P1_GPU_STEPS_DONE; audit and phase decision reported separately", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
