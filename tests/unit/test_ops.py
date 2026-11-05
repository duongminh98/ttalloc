"""Script vận hành (ops/): công việc Vast chỉ gọi lệnh có thật, không lộ token, onstart đủ ngắn."""

from __future__ import annotations

import importlib.util
import re
import shlex
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest
import typer

from ttalloc.cli import app

ROOT = Path(__file__).resolve().parents[2]
JOBS = sorted((ROOT / "ops" / "vast" / "jobs").glob("*.sh"))


def _load(rel: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _commands() -> dict[str, set[str]]:
    group = typer.main.get_command(app)
    out: dict[str, set[str]] = {}
    for name, cmd in group.commands.items():  # type: ignore[attr-defined]
        out[name] = set(getattr(cmd, "commands", {}) or {})
    return out


def _calls(text: str) -> list[list[str]]:
    joined = re.sub(r"\\\n\s*", " ", text)
    return [
        shlex.split(line.split("ttalloc", 1)[1]) for line in joined.splitlines() if re.match(r"^\s*ttalloc\s", line)
    ]


@pytest.mark.parametrize("job", JOBS, ids=lambda p: p.stem)
def test_job_uses_existing_commands_and_configs(job: Path) -> None:
    cmds = _commands()
    text = job.read_text(encoding="utf-8")
    assert "\r\n" not in text
    for argv in _calls(text):
        top = argv[0]
        assert top in cmds, f"{job.name}: lệnh lạ ttalloc {top}"
        if cmds[top]:
            assert argv[1] in cmds[top], f"{job.name}: lệnh lạ ttalloc {top} {argv[1]}"
        for i, a in enumerate(argv):
            if a == "--exp" and "$" not in argv[i + 1] and "configs/locked/" not in argv[i + 1]:
                assert (ROOT / argv[i + 1]).exists(), f"{job.name}: không có {argv[i + 1]}"
    if job.stem != "common":
        assert "source ops/vast/jobs/common.sh" in text
    if "test_main" in text or "eval-test" in text:
        assert "need_approval" in text or job.stem == "p3_log"  # P3 chỉ ghi dữ liệu, có [dừng] duyệt GPU


def test_rehearsal_steps_use_existing_commands_and_cover_jobs() -> None:
    reh = _load("ops/kaggle/rehearsal.py")
    cmds = _commands()
    seen: set[str] = set()
    used: set[tuple[str, ...]] = set()
    for name, args, needs in reh.STEPS:
        assert set(needs) <= seen, f"{name}: bước cần {needs} phải đứng trước"
        seen.add(name)
        top = args[0]
        assert top in cmds, f"{name}: lệnh lạ ttalloc {top}"
        if cmds[top]:
            assert args[1] in cmds[top], f"{name}: lệnh lạ ttalloc {top} {args[1]}"
        used.add(tuple(args[:2]) if cmds[top] else (top,))
        for i, a in enumerate(args):
            if a == "--exp" and "configs/locked/" not in args[i + 1]:
                assert (ROOT / args[i + 1]).exists(), f"{name}: không có {args[i + 1]}"
            assert not (isinstance(a, str) and a.startswith("test")), f"{name}: tập dượt không chạm test_*"
    # Mọi lệnh ttalloc mà các công việc Vast gọi đều được tập dượt.
    for job in JOBS:
        for argv in _calls(job.read_text(encoding="utf-8")):
            key = tuple(argv[:2]) if cmds[argv[0]] else (argv[0],)
            assert key in used, f"{job.name}: ttalloc {' '.join(key)} chưa có trong ops/kaggle/rehearsal.py"


def test_vast_create_body_hides_key(monkeypatch: pytest.MonkeyPatch) -> None:
    vast = _load("ops/vast/vast.py")
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("TTALLOC_HF_REPO", "u/ttalloc-data")
    monkeypatch.setenv("VAST_API_KEY", "vk")
    env = vast.job_env("p2_log", None, self_destroy=False)
    assert "VAST_API_KEY" not in env and env["TTALLOC_JOB"] == "p2_log"
    assert vast.job_env("p2_log", "minh", self_destroy=True)["VAST_API_KEY"] == "vk"
    body = vast.create_body("p2_log", env, 60, None)
    assert body["image"].startswith("vllm/vllm-openai:") and body["disk"] == 60 and "price" not in body
    assert len(body["onstart"]) < 4000 and "hf_x" not in body["onstart"]
    assert "common" not in vast.job_names() and {"p1_baseline", "p2_log", "p3_log", "p4_online"} <= set(
        vast.job_names()
    )
    with pytest.raises(SystemExit):
        vast.job_env("nope", None, False)
    q = vast.build_query(0.3, interruptible=True, limit=5)
    assert q["type"] == "bid" and q["dph_total"] == {"lte": 0.3} and q["gpu_name"] == {"eq": "RTX 3090"}


def test_vast_instances_pages_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    vast = _load("ops/vast/vast.py")
    pages = {"?limit=25": ([{"id": 1}], "t+1"), "?limit=25&after_token=t%2B1": ([{"id": 2}], None)}
    seen = []

    def call(method: str, path: str, body: object = None, query: str = "", base: str = "") -> dict:
        seen.append((base, path, query))
        rows, token = pages[query]
        return {"instances": rows, "next_token": token}

    monkeypatch.setattr(vast, "call", call)
    assert [r["id"] for r in vast.instances()] == [1, 2]
    assert all(b == vast.API_V1 and p == "/instances/" for b, p, _ in seen)


def test_watch_rules_detect_and_diff() -> None:
    rules = _load("ops/watch_rules.py")
    hits = rules.scan(
        ["ok", "Traceback (most recent call last):", "torch.OutOfMemoryError: CUDA out of memory", "LỖI: x"]
    )
    assert [h[0] for h in hits] == ["traceback", "oom", "ttalloc_error"]
    assert rules.scan(["=== step", "xong 64 chuỗi"]) == []
    resolver = (
        "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed."
    )
    assert rules.scan([resolver]) == []  # chỉ là cảnh báo xung đột với gói có sẵn (Kaggle, p1_gpu_tests lần 6)
    assert [h[0] for h in rules.scan(["ERROR: No matching distribution found for vllm==9"])] == ["pip"]
    assert [h[0] for h in rules.scan(["BƯỚC LỖI log_val rc=-1"])] == ["step_failed"]
    assert rules.new_lines(["a", "b"], ["a", "b", "c"]) == ["c"]
    assert rules.new_lines(["a", "b", "c"], ["c", "d", "e"]) == ["d", "e"]  # cửa sổ đuôi đã trượt
    assert rules.new_lines([], ["x"]) == ["x"]


def test_kaggle_partial_log_parses(tmp_path: Path) -> None:
    push = _load("ops/kaggle/push.py")
    p = tmp_path / "k.log"
    p.write_text(
        '[{"stream_name":"stdout","time":0.1,"data":"=== a\\n"}\n,{"stream_name":"stdout","time":1,"data":"b\\n"}\n,',
        encoding="utf-8",
    )  # kernel đang chạy: mảng chưa đóng
    assert push.log_lines(p) == ["=== a", "b"]


def test_kaggle_live_log_reconnects_without_repeating() -> None:
    push = _load("ops/kaggle/push.py")
    # Mỗi lần kết nối, `kaggle kernels logs -f` phát lại từ đầu; ở đây luồng in 3 dòng rồi đóng (như bị ngắt).
    live = push.LiveLog([sys.executable, "-c", "print('RUN_ID x'); print('a'); print('b')"])
    got: list[str] = []
    for _ in range(40):
        got += live.fresh()
        if len(got) >= 3:
            break
        time.sleep(0.1)
    for _ in range(5):
        got += live.fresh()  # các lần nối lại chỉ phát lại dòng cũ
        time.sleep(0.2)
    live.close()
    assert got == ["RUN_ID x", "a", "b"]


def test_rehearsal_streams_step_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reh = _load("ops/kaggle/rehearsal.py")
    monkeypatch.setattr(reh, "WORK", tmp_path)
    code = "import sys; print('x'); print(' 50%|##   | 1/2 [00:01<00:01, 1.00it/s]'); sys.exit(3)"
    assert not reh.run("s", [sys.executable, "-c", code], 60)
    out = capsys.readouterr().out.split("\n", 1)[1]  # bỏ dòng "=== s: <lệnh>"
    assert out == "x\nBƯỚC LỖI s rc=3\n"
    assert "it/s" in (tmp_path / "steps" / "s.txt").read_text(encoding="utf-8")
    assert reh.REPORT["steps"]["s"]["rc"] == 3
    monkeypatch.setattr(reh, "RUN_ID", "R7")
    assert reh.stop_requested("# chú thích\nR6\nR7\n") and not reh.stop_requested("R70\n")
    reh.STOP.set()  # yêu cầu dừng: bước đang chạy bị ngắt ngay
    try:
        t0 = time.time()
        assert not reh.run("long", [sys.executable, "-c", "import time; time.sleep(60)"], 600)
        assert reh.REPORT["steps"]["long"]["rc"] == -2 and time.time() - t0 < 30
    finally:
        reh.STOP.clear()


def test_kaggle_run_id_injection(tmp_path: Path) -> None:
    push = _load("ops/kaggle/push.py")
    for name in ("p0_smoke", "p1_gpu_tests"):
        src = (ROOT / "ops" / "kaggle" / f"{name}.py").read_text(encoding="utf-8")
        out = push.inject_run_id(src, "R1")
        compile(out, name, "exec")  # `from __future__` vẫn đứng đầu
        assert 'TTALLOC_RUN_ID = "R1"' in out and 'print(f"RUN_ID {RUN_ID}"' in out
    rep = tmp_path / "x.json"
    rep.write_text('{"run_id": "OLD", "steps": {"a": {"ok": false, "error": "e"}}}', encoding="utf-8")
    assert push.failed_steps(tmp_path, "x", "NEW") == {}  # báo cáo của bản trước bị bỏ qua
    assert push.failed_steps(tmp_path, "x", "OLD") == {"a": "e"}


def test_bootstrap_runs_from_copy_and_waits_for_fix() -> None:
    vast = _load("ops/vast/vast.py")
    assert "/root/bootstrap.sh" in vast.ONSTART
    text = (ROOT / "ops" / "vast" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "waiting_fix" in text and "code-rev" in text and "pull-code" in text and "put-log" in text


def test_put_log_skips_unchanged_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hf = _load("ops/hf_sync.py")
    uploads: list[bytes] = []

    class _Api:
        def upload_file(self, path_or_fileobj: bytes, **_: object) -> None:
            uploads.append(path_or_fileobj)

    monkeypatch.setattr(hf, "_api", _Api)
    monkeypatch.setattr(hf.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("TTALLOC_HF_REPO", "u/r")
    log = tmp_path / "job.log"
    log.write_text("a\nb\n", encoding="utf-8")
    hf.put_log("j", log)
    hf.put_log("j", log)  # log không đổi: không commit thêm
    log.write_text("a\nb\nc\n", encoding="utf-8")
    hf.put_log("j", log)
    assert uploads == [b"a\nb", b"a\nb\nc"]


def test_hf_sync_never_ships_secrets() -> None:
    hf = _load("ops/hf_sync.py")
    for bad in (".env", "kaggle.json", "sub/.env", "ops/kaggle/_build/x", "ops/vast/_state/instances.json"):
        assert hf._blocked(bad)
    assert not hf._blocked("src/ttalloc/cli.py")
    if not (ROOT / ".git").exists():
        pytest.skip("cần repo git; mã đang chạy từ gói tar (Kaggle, Vast)")
    files = hf.git_files(ROOT)
    assert "pyproject.toml" in files and "ops/vast/bootstrap.sh" in files
    # Mọi tệp .py của gói đều phải được đóng gói (lỗi cũ: `data/` trong .gitignore bỏ mất src/ttalloc/data).
    pkg = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src" / "ttalloc").rglob("*.py")}
    assert pkg <= set(files), sorted(pkg - set(files))[:5]
    assert not any(hf._blocked(f) for f in files)
