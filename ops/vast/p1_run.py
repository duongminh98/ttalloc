"""Checkpointed P1 stage driver; no automatic rental, retuning or test retries."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FULL_RUN = Path("runs/20261117-181157_baseline-full-test_main-deer_paper_3c9fbf102c27")
FULL_SHA256 = "345203efd070bb0399cd2d6037cd03df9b378e5968e0de080cd3e083177163fa"
FULL_CONFIG_HASH = "3c9fbf102c27"
FULL_SOURCE_SESSION = "runs/p1-20261118-010031"
APPROVED_BY = "Dương Tuấn Minh"
PILOT_LOGS = Path("data/logs/qwen3_1_7b/val")
FINISH_STAGES = ("pilot", "validate", "answer_mode", "probe_noise")
EQUIVALENCE_RUNS = (
    "20261119-035954_equivalence_3c9fbf102c27",
    "20261119-153426_equivalence_3c9fbf102c27",
)
EQUIVALENCE_SHA256 = "55db1ba494e4c203b270cef6c1b3be2a6d5096b780f1551c13e1972672e13e00"
REVIEWED_ACC_MARGIN = 0.031


def verify_reviewed_equivalence() -> dict[str, Any]:
    from merge_equivalence import combined_report

    first, second = (ROOT / "runs" / name for name in EQUIVALENCE_RUNS)
    report_path = second / "equivalence_4seed.json"
    raw = report_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EQUIVALENCE_SHA256:
        raise ValueError("ADR-023 requires the original four-seed report")
    original = json.loads(raw)
    if original != combined_report(first, second):
        raise ValueError("ADR-023 report does not match the verified equivalence checkpoints")
    lo, hi = original["acc_ci90"]
    if original["pass"] or not original["len_ok"] or not -REVIEWED_ACC_MARGIN <= lo <= hi <= REVIEWED_ACC_MARGIN:
        raise ValueError("ADR-023 reviewed continuation conditions are not satisfied")
    return {
        "approval": "ADR-023",
        "original_report": report_path.relative_to(ROOT).as_posix(),
        "original_report_sha256": EQUIVALENCE_SHA256,
        "original_pass": original["pass"],
        "acc_ci90": original["acc_ci90"],
        "len_rel_ci90": original["len_rel_ci90"],
        "reviewed_acc_margin": REVIEWED_ACC_MARGIN,
        "threshold_chosen_after_observing_results": True,
        "continuation_approved": True,
    }


def verify_full_handoff(run: Path, approved_by: str) -> dict[str, Any]:
    from ttalloc.baseline.checkpoint import read_payload

    if approved_by != APPROVED_BY or run.resolve() != (ROOT / FULL_RUN).resolve():
        raise ValueError("ADR-022 requires the approved Full MATH-500 run and approver")
    payload = read_payload(run / "checkpoint.json", expected_sha256=FULL_SHA256)
    identity = payload["identity"]
    split = json.loads((ROOT / "splits/test_main.json").read_text(encoding="utf-8"))
    expected = {(qid, 0) for qid in split["qids"] if qid.startswith("math500/")}
    rows = payload["rows"]
    if (
        identity["config_hash"] != FULL_CONFIG_HASH
        or identity["role"] != "full:test_main:deer_paper"
        or len(expected) != 500
        or len(identity["keys"]) != 500
        or {tuple(key) for key in identity["keys"]} != expected
        or len(rows) != 500
        or {(row["qid"], row["seed"]) for row in rows} != expected
        or sum(row["correct"] for row in rows) != 441
    ):
        raise ValueError("ADR-022 requires the original config, all 500 Full MATH-500 keys and 441 correct")
    return {
        "status": "approved_exception",
        "numeric_gate_pass": False,
        "approval": "ADR-022",
        "approved_by": approved_by,
        "approved_on": "2026-11-18",
        "run": FULL_RUN.as_posix(),
        "source_session": FULL_SOURCE_SESSION,
        "checkpoint_sha256": FULL_SHA256,
        "config_hash": FULL_CONFIG_HASH,
        "n": 500,
        "grade_correct": 441,
        "grade_accuracy": 0.882,
        "deer_saved_answer_correct": 435,
        "deer_saved_answer_accuracy": 0.87,
        "missing_reasoning_cases": 6,
        "deer_comparison": "runs/p1-deer-comparison-20261118",
    }


def stages(approved_by: str) -> list[tuple[str, list[str]]]:
    cfg = ["--exp", "configs/experiments/p1_safe.yaml"]
    approval = ["--approved-by", approved_by]
    test = ["--split", "test_main", "--profile", "deer_paper"]
    out = [
        (f"{method}_{ds}", [
            "baseline", method, *cfg, *test, "--datasets", ds, "--seeds", seeds, *approval, "--save-raw",
        ])
        for method, ds, seeds in (("deer", "math500", "0"), ("full", "aime_2024", "0-3"), ("deer", "aime_2024", "0-3"))
    ]
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
    if name == "validate":
        report = ROOT / PILOT_LOGS / "validate.json"
        if not report.is_file():
            return "missing pilot validation report"
        obj = json.loads(report.read_text(encoding="utf-8"))
        if not obj["ok"] or obj["n_chains"] < 200:
            return f"pilot validation failed: n_chains={obj['n_chains']}, pass_rate={obj['pass_rate']}"
    return None


def save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = sys.argv[1:]
    if args == ["--verify-reviewed-equivalence"]:
        print(json.dumps(verify_reviewed_equivalence(), ensure_ascii=False), flush=True)
        return 0
    finish = args == ["--finish-reviewed-equivalence"]
    if args not in ([], ["--verify-handoff"], ["--finish-reviewed-equivalence"]):
        raise ValueError("unrecognized P1 arguments")
    by = os.environ["TTALLOC_APPROVED_BY"]
    full = verify_full_handoff(ROOT / FULL_RUN, by)
    if args == ["--verify-handoff"]:
        print(json.dumps(full, ensure_ascii=False), flush=True)
        return 0
    session = os.environ["TTALLOC_P1_SESSION"]
    if not session.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid session")
    if session == Path(FULL_SOURCE_SESSION).name:
        raise ValueError("ADR-022 continuation requires a new P1 session")
    path = ROOT / "runs" / session / "stages.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    state["full_math500"] = full
    if finish:
        state["equivalence_continuation"] = verify_reviewed_equivalence()
    save(path, state)
    print(f"P1_FULL_HANDOFF {json.dumps(full, ensure_ascii=False)}", flush=True)
    job = "p1_finish" if finish else "p1_all"
    for name, argv in stages(by):
        if finish and name not in FINISH_STAGES:
            continue
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
        subprocess.run([sys.executable, "ops/hf_sync.py", "set-status", job, "running", name], check=False)
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
        if (
            finish and rc == 0 and name in ("answer_mode", "probe_noise")
            and (result is None or not all((result / f"{name}.{ext}").is_file() for ext in ("json", "csv")))
        ):
            reason = f"missing {name} study artifacts"
        if rc == 0 and argv[0] == "baseline" and "--save-raw" in argv:
            if result is None:
                reason = "missing baseline run directory"
            else:
                scored = subprocess.run([
                    sys.executable, "-I", "ops/vast/deer_score.py",
                    str(result / "checkpoint.json"), str(result / "deer_comparison.json"),
                ])
                if scored.returncode:
                    reason = reason or f"DEER comparison failed rc={scored.returncode}"
                else:
                    state["stages"][name]["deer_comparison"] = str(result / "deer_comparison.json")
        state["stages"][name].update({"finished": time.time(), "rc": rc, "done": reason is None, "stop": reason})
        save(path, state)
        # Keep completed stages in durable remote storage before proceeding.
        sync_paths = ["runs", "docs/test-log.md"]
        if finish:
            sync_paths = [path.parent.relative_to(ROOT).as_posix()]
            if result is not None:
                sync_paths.append(str(result))
        if name in ("pilot", "validate"):
            sync_paths.append(PILOT_LOGS.as_posix())
        synced = subprocess.run([sys.executable, "ops/hf_sync.py", "push", *sync_paths])
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
