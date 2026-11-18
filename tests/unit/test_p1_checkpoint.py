"""Interrupt/resume without regenerated rows, corruption rejection and stop thresholds."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from scripted_world import make_world
from ttalloc.baseline.checkpoint import RowCheckpoint
from ttalloc.baseline.full import Sampling, generate_full


def test_full_checkpoint_interrupt_resume(qwen3_tokenizer: Any, tmp_path: Path) -> None:
    w = make_world(qwen3_tokenizer, n_questions=3)
    qs = list(w.questions.values())
    gen = w.source().gen
    keys = [(q.qid, 0) for q in qs]
    cp = RowCheckpoint(tmp_path / "first.json", {"config_hash": "A"}, keys)

    def interrupted(rows: list[dict[str, Any]]) -> None:
        cp.add(rows, 2)
        raise RuntimeError("power loss after durable checkpoint")

    with pytest.raises(RuntimeError, match="power loss"):
        generate_full(gen, w.codec, qs, w.prompt_ids, [0], Sampling(0, 1, -1, 100),
                      chunk=1, on_chunk=interrupted, save_raw=True)
    assert len(cp.rows) == 1
    resumed = RowCheckpoint(tmp_path / "second.json", {"config_hash": "A"}, keys, tmp_path / "first.json")
    called: list[str] = []
    def record(rows: list[dict[str, Any]]) -> None:
        called.extend(r["qid"] for r in rows)
        resumed.add(rows, 1)
    rows = generate_full(gen, w.codec, qs, w.prompt_ids, [0], Sampling(0, 1, -1, 100),
                         chunk=1, on_chunk=record, initial_rows=resumed.rows, save_raw=True)
    resumed.complete()
    assert len(rows) == 3 and len(set(r["qid"] for r in rows)) == 3
    assert all(r["raw_token_source"] == "engine" and len(r["raw_token_ids"]) == r["n_tokens"] for r in rows)
    assert all(r["raw_text"] == w.codec.decode(r["raw_token_ids"]) for r in rows)
    assert cp.rows[0]["qid"] not in called and resumed.seconds == 4
    assert len(json.loads((tmp_path / "first.json").read_text())["payload"]["rows"]) == 1


def test_checkpoint_rejects_corrupt_foreign_and_duplicate(tmp_path: Path) -> None:
    p = tmp_path / "cp.json"
    cp = RowCheckpoint(p, {"hash": "A"}, [("q", 0)])
    with pytest.raises(ValueError, match="incomplete"):
        cp.complete()
    cp.add([{"qid": "q", "seed": 0}], 1)
    cp.complete()
    with pytest.raises(ValueError, match="duplicate"):
        cp.add([{"qid": "q", "seed": 0}], 1)
    with pytest.raises(ValueError, match="mismatch"):
        RowCheckpoint(tmp_path / "new.json", {"hash": "B"}, [("q", 0)], p)
    obj = json.loads(p.read_text())
    obj["payload"]["rows"][0]["seed"] = 99
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="checksum"):
        RowCheckpoint(tmp_path / "new.json", {"hash": "A"}, [("q", 0)], p)


def _module(name: str) -> Any:
    path = Path(__file__).resolve().parents[2] / "ops/vast" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_deadline_budget_and_stall() -> None:
    guard = _module("guard")
    assert guard.stop_reason(50, 100, .2, 3.25, .25, 45, 30) is None
    assert guard.stop_reason(100, 100, .2, 3.25, .25, 99, 30) == "deadline"
    assert guard.stop_reason(50, 100, 3.0, 3.25, .25, 45, 30) == "budget_margin"
    assert guard.stop_reason(50, 100, .2, 3.25, .25, 10, 30) == "no_generation_or_checkpoint_progress"


def test_pipeline_stops_on_bad_baseline_equivalence_or_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _module("p1_run")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({"n": 500, "accuracy": .85}))
    assert runner.stage_gate("full_math500", tmp_path) is None
    (tmp_path / "summary.json").write_text(json.dumps({"n": 500, "accuracy": .80}))
    assert runner.stage_gate("full_math500", tmp_path)
    (tmp_path / "equivalence.json").write_text(json.dumps({"n_questions": 400, "pass": False}))
    assert runner.stage_gate("equivalence", tmp_path)
    commands = runner.stages("Minh")
    assert len(commands) == 9
    assert "full_math500" not in dict(commands)
    assert "--save-raw" in dict(commands)["deer_math500"]
    assert runner.stage_gate("deer_math500", tmp_path)
    assert all("--approved-by" in argv for name, argv in commands if name.startswith(("full_", "deer_")))
    pilot = tmp_path / runner.PILOT_LOGS / "validate.json"
    pilot.parent.mkdir(parents=True)
    pilot.write_text(json.dumps({"n_chains": 200, "ok": False, "pass_rate": .99}))
    assert runner.stage_gate("validate", None)
    pilot.write_text(json.dumps({"n_chains": 200, "ok": True, "pass_rate": 1.0}))
    assert runner.stage_gate("validate", None) is None


def test_full_handoff_pins_run_checksum_config_and_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _module("p1_run")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    keys = [(f"math500/{i}", 0) for i in range(500)]
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits/test_main.json").write_text(json.dumps({"qids": [q for q, _ in keys]}))
    run = tmp_path / runner.FULL_RUN
    identity = {"config_hash": runner.FULL_CONFIG_HASH, "role": "full:test_main:deer_paper",
                "keys": [list(key) for key in keys]}
    cp = RowCheckpoint(run / "checkpoint.json", identity, keys)
    rows = [{"qid": qid, "seed": seed, "correct": i < 441} for i, (qid, seed) in enumerate(keys)]
    cp.add(rows, 1.0)
    sha = json.loads(cp.path.read_text())["sha256"]
    monkeypatch.setattr(runner, "FULL_SHA256", sha)
    report = runner.verify_full_handoff(run, runner.APPROVED_BY)
    assert report["status"] == "approved_exception" and report["numeric_gate_pass"] is False
    assert (report["grade_correct"], report["deer_saved_answer_correct"], report["missing_reasoning_cases"]) == (
        441, 435, 6,
    )
    with pytest.raises(ValueError, match="approved Full"):
        runner.verify_full_handoff(run, "other")
    with pytest.raises(ValueError, match="approved Full"):
        runner.verify_full_handoff(run.parent / "another-run", runner.APPROVED_BY)
    foreign = RowCheckpoint(cp.path, identity | {"config_hash": "other"}, keys)
    foreign.add(rows, 1.0)
    with pytest.raises(ValueError, match="checksum"):
        runner.verify_full_handoff(run, runner.APPROVED_BY)
    monkeypatch.setattr(runner, "FULL_SHA256", json.loads(cp.path.read_text())["sha256"])
    with pytest.raises(ValueError, match="original config"):
        runner.verify_full_handoff(run, runner.APPROVED_BY)


@pytest.mark.parametrize("trigger", ["budget", "missing_credit", "deadline", "done", "sync_failure"])
def test_guard_supervisor_terminates_syncs_and_destroys_exact_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trigger: str,
) -> None:
    guard = _module("guard")
    monkeypatch.chdir(tmp_path)
    env = {
        "CONTAINER_ID": "123", "TTALLOC_DEADLINE_EPOCH": "200" if trigger != "deadline" else "100",
        "TTALLOC_RENT_START_EPOCH": "0", "TTALLOC_PHASE_CREDIT_START": "5",
        "TTALLOC_PHASE_PRIOR_SPEND_USD": ".795" if trigger == "missing_credit" else ".10",
        "TTALLOC_PHASE_LIMIT_USD": "1", "TTALLOC_BUDGET_MARGIN_USD": ".2", "TTALLOC_DPH": ".24",
        "TTALLOC_STALL_SECONDS": "1000", "TTALLOC_GUARD_POLL": "1", "TTALLOC_FINAL_SYNC_SECONDS": "10",
        "TTALLOC_JOB_LOG": str(tmp_path / "job.log"), "TTALLOC_JOB": "p1_all",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    calls: list[str] = []
    class Child:
        pid = 456
        returncode = 0
        def poll(self) -> int | None:
            return 0 if trigger in ("done", "sync_failure") else None
        def wait(self, timeout: float) -> int:
            return 0
    def api(method: str, path: str) -> dict[str, Any]:
        calls.append(method + path)
        if method == "GET" and trigger == "missing_credit":
            raise OSError("credit unavailable")
        return {"credit": 4 if trigger == "budget" else 5}
    def sync(timeout: float) -> bool:
        calls.append("sync")
        if trigger == "sync_failure":
            raise OSError("network down")
        return True
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(guard.os, "killpg", lambda pid, sig: calls.append(f"kill:{pid}"), raising=False)
    monkeypatch.setattr(guard.time, "time", lambda: 100)
    monkeypatch.setattr(guard, "api", api)
    monkeypatch.setattr(guard, "bounded_sync", sync)
    rc = guard.main()
    assert rc == (0 if trigger in ("done", "sync_failure") else 2)
    assert calls[-1] == "DELETE/instances/123/"
    assert calls.index("kill:456") < calls.index("sync") < len(calls) - 1
    receipt = json.loads((tmp_path / "runs/guard-123.json").read_text())
    assert receipt["instance"] == "123"
    if trigger == "missing_credit":
        assert receipt["reason"] == "budget_margin"
        assert receipt["estimated_spent"] == pytest.approx(.795 + .24 * 100 / 3600)


def test_safe_config_keeps_logging_hash_unchanged() -> None:
    from ttalloc.config import load_config, load_model_cfg
    from ttalloc.context import log_hash
    from ttalloc.data.prompts import load_prompts
    cfg = load_config(Path("configs/experiments/qwen3.yaml"))
    safe = load_config(Path("configs/experiments/p1_safe.yaml"))
    model = load_model_cfg(cfg.model)
    prompts = load_prompts()
    assert log_hash(cfg, model, prompts) == log_hash(safe, model, prompts)
    assert safe.baseline.chunk == safe.online.max_num_seqs == safe.log.window == 32


def test_deer_chunk_checkpoint_resume(
    tmp_path: Path, qwen3_tokenizer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ttalloc.baseline.deer_official as deer
    from ttalloc.config import load_model_cfg
    w = make_world(qwen3_tokenizer, n_questions=3)
    cfg = w.cfg.model_copy(update={"baseline": w.cfg.baseline.model_copy(update={"chunk": 1})})
    qs = list(w.questions.values())
    keys = [(q.qid, 0) for q in qs]
    cp = RowCheckpoint(tmp_path / "first.json", {"hash": "A"}, keys)
    calls: list[str] = []
    monkeypatch.setattr(deer, "ensure_repo", lambda cfg: tmp_path)
    def process(cmd: list[str], **kwargs: Any) -> None:
        name = cmd[cmd.index("--dataset") + 1]
        data = Path(cmd[cmd.index("--dataset_dir") + 1]) / name / "test.jsonl"
        output = Path(cmd[cmd.index("--output_path") + 1]) / name / "result.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in data.read_text().splitlines()]
        output.write_text("\n".join(json.dumps({"question": r["problem"],
            "generated_responses": ["</think>\\boxed{42}"]}) for r in rows), encoding="utf-8")
        calls.extend(r["qid"] for r in rows)
    monkeypatch.setattr(deer.subprocess, "run", process)
    def interrupted(new: list[dict[str, Any]]) -> None:
        cp.add(new, 1)
        raise RuntimeError("interrupt")
    first = tmp_path / "run1"
    first.mkdir()
    with pytest.raises(RuntimeError, match="interrupt"):
        deer.run_deer(cfg, load_model_cfg(cfg.model), qs, w.codec, first, [0], 0, 1, 100, "model",
                      on_chunk=interrupted)
    resumed = RowCheckpoint(tmp_path / "second.json", {"hash": "A"}, keys, tmp_path / "first.json")
    second = tmp_path / "run2"
    second.mkdir()
    rows, _ = deer.run_deer(cfg, load_model_cfg(cfg.model), qs, w.codec, second, [0], 0, 1, 100, "model",
                           on_chunk=lambda new: resumed.add(new, 1), initial_rows=resumed.rows)
    resumed.complete()
    assert len(rows) == len(calls) == len(set(calls)) == 3


def test_equivalence_resumes_full_and_segmented_independently(
    tmp_path: Path, qwen3_tokenizer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    import ttalloc.baseline.equivalence as eq
    w = make_world(qwen3_tokenizer, n_questions=3)
    cfg = w.cfg.model_copy(update={"baseline": w.cfg.baseline.model_copy(update={"chunk": 1})})
    qs = list(w.questions.values())
    keys = [(q.qid, 0) for q in qs]
    f = RowCheckpoint(tmp_path / "f.json", {"role": "f"}, keys)
    s = RowCheckpoint(tmp_path / "s.json", {"role": "s"}, keys)
    calls: list[str] = []
    def segmented(source: Any, questions: Any, seeds: Any, cfg: Any) -> list[dict[str, Any]]:
        calls.extend(q.qid for q in questions)
        return [{"qid": q.qid, "seed": seed, "correct": False, "think_tokens": 100}
                for seed in seeds for q in questions]
    monkeypatch.setattr(eq, "segmented_rows", segmented)
    original = s.add
    def interrupted(rows: list[dict[str, Any]], seconds: float) -> None:
        original(rows, seconds)
        raise RuntimeError("interrupt")
    monkeypatch.setattr(s, "add", interrupted)
    with pytest.raises(RuntimeError, match="interrupt"):
        eq.run_equivalence(w.source(), qs, w.prompt_ids, [0], Sampling(0, 1, -1, 100), cfg,
                           np.random.default_rng(0), f, s)
    f2 = RowCheckpoint(tmp_path / "f2.json", {"role": "f"}, keys, tmp_path / "f.json")
    s2 = RowCheckpoint(tmp_path / "s2.json", {"role": "s"}, keys, tmp_path / "s.json")
    source = w.source()
    eq.run_equivalence(source, qs, w.prompt_ids, [0], Sampling(0, 1, -1, 100), cfg,
                       np.random.default_rng(0), f2, s2)
    assert source.gen.n_generated == 0  # full side was complete; no full request regenerated
    f2.complete()
    s2.complete()
    assert len(calls) == len(set(calls)) == 3
