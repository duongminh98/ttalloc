"""Dòng lệnh `ttalloc`. Mỗi lệnh ứng với một task hoặc bước chạy trong docs/PLAN.md mục 6; xem ops/RUNBOOK.md."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
splits_app = typer.Typer(no_args_is_help=True, help="Chia tập và khóa danh sách ID (P0.T7).")
baseline_app = typer.Typer(no_args_is_help=True, help="Tái lập baseline (P1.T5).")
study_app = typer.Typer(
    no_args_is_help=True, help="Nghiên cứu nhỏ: answer-mode (P1.T9), probe-noise (P1.T10), curves (P2.T20)."
)
estimator_app = typer.Typer(no_args_is_help=True, help="Bộ ước lượng (P2.T4, P2.T6).")
table_app = typer.Typer(no_args_is_help=True, help="Bảng tính sẵn cho mô phỏng (P2.T10).")
sim_app = typer.Typer(no_args_is_help=True, help="Mô phỏng (P2.T15).")
online_app = typer.Typer(no_args_is_help=True, help="Chạy thật (P4.T1).")
gate_app = typer.Typer(no_args_is_help=True, help="Cổng quyết định (P2.T19, P4.T4).")
for sub, name in [
    (splits_app, "splits"),
    (baseline_app, "baseline"),
    (study_app, "study"),
    (estimator_app, "estimator"),
    (table_app, "table"),
    (sim_app, "sim"),
    (online_app, "online"),
    (gate_app, "gate"),
]:
    app.add_typer(sub, name=name)

Exp = Annotated[Path | None, typer.Option("--exp", help="Config thí nghiệm YAML; mặc định configs/base.yaml")]
Sets = Annotated[list[str] | None, typer.Option("--set", help="Ghi đè dạng khóa=giá trị, ví dụ --set dp.m=256")]
Model = Annotated[str | None, typer.Option("--model", help="Tên tệp trong configs/models, ví dụ qwen3_1_7b")]
Split = Annotated[str, typer.Option("--split")]
Seeds = Annotated[str, typer.Option("--seeds", help="Ví dụ 0-3 hoặc 0,2,5")]
Approved = Annotated[str | None, typer.Option("--approved-by", help="Người duyệt khi chạy trên test_* (mục 7.5)")]

log = logging.getLogger("ttalloc")


# ======================================================================== tiện ích


def _cfg(exp: Path | None, sets: list[str] | None, model: str | None = None) -> Any:
    from ttalloc.config import CONFIGS_DIR, load_config

    over = list(sets or [])
    if model:
        over.append(f"model={model}")
    return load_config(exp or CONFIGS_DIR / "base.yaml", over)


def _seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def hash_order(qids: list[str], salt: str) -> list[str]:
    """Thứ tự giả ngẫu nhiên cố định; tiền tố dài n là một mẫu ngẫu nhiên và mở rộng được (500 câu rồi 1.500 câu)."""
    return sorted(qids, key=lambda q: hashlib.sha256(f"{salt}|{q}".encode()).hexdigest())


def _run_dir(name: str, cfg: Any) -> Path:
    from ttalloc.runs import create_run_dir

    d = create_run_dir(name, cfg)
    h = logging.FileHandler(d / "log.txt", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger("ttalloc").addHandler(h)
    typer.echo(f"thư mục chạy: {d}")
    return d


def _echo(msg: str) -> None:
    typer.echo(msg)
    log.info(msg)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _questions(split: str, datasets: list[str] | None = None, limit: int | None = None) -> list[Any]:
    from ttalloc.context import filter_datasets, load_split_questions

    qs = filter_datasets(load_split_questions(split), datasets)
    if limit is not None:
        order = hash_order([q.qid for q in qs], split)[:limit]
        keep = set(order)
        qs = [q for q in qs if q.qid in keep]
    return qs


def _vllm(ctx: Any, max_len: int) -> Any:
    from ttalloc.engine.vllm_generator import VllmGenerator

    _echo(
        f"khởi tạo vLLM: {ctx.model_cfg.hf_id}@{ctx.model_cfg.revision} dtype={ctx.cfg.engine.dtype} "
        f"max_model_len={max_len}"
    )
    return VllmGenerator(ctx.model_cfg, ctx.cfg.engine, max_len, ctx.cfg.online.max_num_seqs)


def _live_source(ctx: Any, gen: Any, qs: list[Any], pids: dict[str, list[int]], mode: Any = None) -> Any:
    from ttalloc.engine.live_source import LiveSource

    return LiveSource(
        gen,
        ctx.codec,
        {q.qid: q for q in qs},
        pids,
        ctx.cfg,
        ctx.model_cfg,
        ctx.prompts,
        l_log=ctx.cfg.log.l_log,
        answer_mode=mode,
    )


def _bounds_extra(ctx: Any) -> int:
    """Số token tối đa của nhánh gợi hoặc trả lời, cộng dư một ít, để tính max_model_len."""
    probe = max(len(ctx.codec.encode(t)) for t in (ctx.prompts.probe_math, ctx.prompts.probe_choice))
    probe += ctx.cfg.probe.r_probe + ctx.model_cfg.probe_think_end_lookahead
    ans = 1 + max(len(ctx.codec.encode(ctx.prompts.answer(m))) for m in ("full", "short")) + 256
    return int(max(probe, ans)) + 16


def _refuse_test(split: str, what: str) -> None:
    if split.startswith("test"):
        typer.echo(f"LỖI: {what} trên {split} phải đi qua `ttalloc eval-test` với config đã khóa (mục 7.5)", err=True)
        raise typer.Exit(code=2)


def _verify_splits() -> None:
    from ttalloc.data.splits import SPLITS_DIR, SplitsTamperedError, verify_all

    if not SPLITS_DIR.exists():
        typer.echo("LỖI: chưa có splits/ (chạy `ttalloc splits build` và khóa ID trước, P0.T7)", err=True)
        raise typer.Exit(code=2)
    try:
        verify_all(SPLITS_DIR)
    except SplitsTamperedError as exc:
        typer.echo(f"LỖI: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s"
    )
    logging.getLogger("ttalloc").setLevel(logging.INFO)


# ======================================================================== P0


@splits_app.command("build")
def splits_build(
    skip: Annotated[
        list[str] | None, typer.Option(help="Bỏ qua tập này, ví dụ --skip test_ood khi chưa có HF_TOKEN cho GPQA")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Thư mục ghi; mặc định splits/")] = None,
) -> None:
    """Tải các tập nguồn (cần mạng) rồi dựng splits/<tên>.json và .sha256. Sau đó **[dừng]** chờ khóa ID."""
    from ttalloc.data.loaders import load_questions
    from ttalloc.data.splits import SPLITS_DIR, build_splits, load_splits_cfg, write_splits

    cfg = load_splits_cfg()
    skipped = set(skip or [])
    datasets = sorted({s["dataset"] for name, specs in cfg["splits"].items() if name not in skipped for s in specs})
    pool = {}
    for name in datasets:
        typer.echo(f"nạp {name} ...")
        pool[name] = load_questions(name)
    results = build_splits(pool, cfg, skip=skipped)
    for p in write_splits(results, out or SPLITS_DIR):
        res = results[p.stem]
        typer.echo(f"{p}: {len(res.qids)} câu {res.counts}; loại trùng {res.dedup_removed}")
    typer.echo("Tiếp theo: con người xác nhận khóa danh sách ID (PLAN P0.T7), rồi commit splits/.")


@splits_app.command("verify")
def splits_verify(splits_dir: Annotated[Path | None, typer.Option()] = None) -> None:
    """Kiểm tra mã băm và tính rời nhau của mọi tập; thoát với mã lỗi nếu không khớp."""
    from ttalloc.data.splits import SPLITS_DIR, SplitsTamperedError, verify_all

    try:
        sizes = verify_all(splits_dir or SPLITS_DIR)
    except SplitsTamperedError as exc:
        typer.echo(f"LỖI: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"OK: {sizes}")


@app.command("audit-grading")
def audit_grading(
    run: Annotated[Path | None, typer.Option(help="Thư mục chạy có answers.parquet")] = None,
    model: Model = None,
    split: Annotated[str | None, typer.Option("--split")] = None,
    n: Annotated[int, typer.Option()] = 100,
    seed: Annotated[int, typer.Option()] = 0,
    out: Annotated[Path | None, typer.Option(help="Tệp CSV")] = None,
) -> None:
    """Lấy mẫu câu trả lời để chấm tay (P0.T10): từ một thư mục chạy, hoặc từ dữ liệu ghi (--model --split)."""
    from ttalloc.data.audit import sample_from_logs, sample_from_run, write_audit_csv

    rng = np.random.default_rng(seed)
    if run is not None:
        rows = sample_from_run(run, n, rng)
        path = out or run / "audit_grading.csv"
    elif model and split:
        from ttalloc.context import load_split_questions
        from ttalloc.logging.store import LogStore

        store = LogStore.for_split(model, split)
        pts = store.read_points(
            columns=["qid", "seed", "answer_text", "answer_extracted", "answer_correct"]
        ).to_pylist()
        tr = store.read_traces(columns=["qid", "seed", "nat_answer_text", "nat_answer_extracted", "nat_correct"])
        nat = [
            {
                "qid": r["qid"],
                "seed": r["seed"],
                "answer_text": r["nat_answer_text"],
                "source": "natural",
                "answer_extracted": r["nat_answer_extracted"],
                "answer_correct": r["nat_correct"],
            }
            for r in tr.to_pylist()
            if r["nat_answer_text"] is not None
        ]
        probs = {q.qid: (q.problem, q.gold) for q in load_split_questions(split)}
        rows = sample_from_logs(pts + nat, probs, n, rng)
        path = out or store.root / "audit_grading.csv"
    else:
        typer.echo("LỖI: cần --run hoặc --model và --split", err=True)
        raise typer.Exit(code=2)
    write_audit_csv(rows, path)
    typer.echo(f"{path}: {len(rows)} dòng. Điền cột manual_correct (1/0) rồi so với auto_correct; đạt khi khớp >= 99%.")


@app.command("audit-score")
def audit_score(csv_path: Annotated[Path, typer.Argument()]) -> None:
    """Tỉ lệ khớp giữa chấm tay (manual_correct) và chấm tự động (P1, bước chạy 3)."""
    import pandas as pd

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df.manual_correct.astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    agree = (df.manual_correct.astype(float).astype(int) == df.auto_correct.astype(int)).mean()
    typer.echo(json.dumps({"n_scored": len(df), "agreement": float(agree), "pass": bool(agree >= 0.99)}, indent=2))


# ======================================================================== P1


@baseline_app.command("full")
def baseline_full(
    model: Model = None,
    split: Split = "val",
    seeds: Seeds = "0",
    profile: Annotated[str, typer.Option()] = "log",
    datasets: Annotated[list[str] | None, typer.Option("--datasets")] = None,
    limit: Annotated[int | None, typer.Option()] = None,
    approved_by: Approved = None,
    exp: Exp = None,
    sets: Sets = None,
) -> None:
    """Sinh liền mạch, không gợi, không dừng sớm (P1.T5)."""
    from ttalloc.baseline.full import generate_full, profile_sampling, require_approval, summarize, write_results
    from ttalloc.context import model_context, prompt_ids_for

    _verify_splits()
    cfg = _cfg(exp, sets, model)
    qs = _questions(split, datasets, limit)
    require_approval(split, approved_by, "baseline full", f"profile={profile} datasets={datasets} n={len(qs)}")
    ctx = model_context(cfg)
    sampling = profile_sampling(profile, ctx.model_cfg, cfg)  # type: ignore[arg-type]
    pids = prompt_ids_for(qs, ctx)
    d = _run_dir(f"baseline-full-{split}-{profile}", cfg)
    gen = _vllm(ctx, max(len(v) for v in pids.values()) + sampling.max_tokens + 16)
    import time

    t0 = time.time()
    rows = generate_full(
        gen,
        ctx.codec,
        qs,
        pids,
        _seeds(seeds),
        sampling,
        cfg.baseline.chunk,
        on_chunk=lambda new: _echo(f"xong {len(new)} chuỗi"),
    )
    s = summarize(rows, time.time() - t0) | {"profile": profile, "split": split, "datasets": datasets}
    write_results(d, rows, s)
    _echo(json.dumps({k: s[k] for k in ("n", "accuracy", "mean_tokens", "by_dataset")}, indent=2, ensure_ascii=False))


@baseline_app.command("deer")
def baseline_deer(
    model: Model = None,
    split: Split = "val",
    seeds: Seeds = "0",
    profile: Annotated[str, typer.Option()] = "deer_paper",
    datasets: Annotated[list[str] | None, typer.Option("--datasets")] = None,
    limit: Annotated[int | None, typer.Option()] = None,
    approved_by: Approved = None,
    exp: Exp = None,
    sets: Sets = None,
) -> None:
    """Chạy repo DEER chính thức (third_party/DEER ở commit cố định) và chấm lại (P1.T5, P4.T5)."""
    import time

    from huggingface_hub import snapshot_download

    from ttalloc.baseline.deer_official import run_deer
    from ttalloc.baseline.full import profile_sampling, require_approval, summarize, write_results
    from ttalloc.context import model_context

    _verify_splits()
    cfg = _cfg(exp, sets, model)
    qs = _questions(split, datasets, limit)
    require_approval(split, approved_by, "baseline deer", f"profile={profile} datasets={datasets} n={len(qs)}")
    ctx = model_context(cfg)
    sampling = profile_sampling(profile, ctx.model_cfg, cfg)  # type: ignore[arg-type]
    d = _run_dir(f"baseline-deer-{split}-{profile}", cfg)
    path = snapshot_download(ctx.model_cfg.hf_id, revision=ctx.model_cfg.revision)
    t0 = time.time()
    rows, timing = run_deer(
        cfg,
        ctx.model_cfg,
        qs,
        ctx.codec,
        d,
        _seeds(seeds),
        sampling.temperature,
        sampling.top_p,
        sampling.max_tokens,
        path,
    )
    s = summarize(rows, time.time() - t0) | {"profile": profile, "timing": timing, "split": split}
    write_results(d, rows, s)
    _echo(json.dumps({k: s[k] for k in ("n", "accuracy", "mean_tokens", "by_dataset")}, indent=2, ensure_ascii=False))


@baseline_app.command("throughput")
def baseline_throughput(
    model: Model = None, n: Annotated[int, typer.Option()] = 256, exp: Exp = None, sets: Sets = None
) -> None:
    """Số token mỗi giây theo kích thước lô, trên một tập con của val (P1, bước chạy 2)."""
    from ttalloc.baseline.full import Sampling
    from ttalloc.baseline.throughput import measure_throughput
    from ttalloc.context import model_context, prompt_ids_for

    _verify_splits()
    cfg = _cfg(exp, sets, model)
    qs = _questions("val", None, n)
    ctx = model_context(cfg)
    pids = prompt_ids_for(qs, ctx)
    s = ctx.model_cfg.sampling
    sampling = Sampling(s.temperature, s.top_p, s.top_k, cfg.baseline.throughput_max_tokens)
    d = _run_dir("baseline-throughput", cfg)
    gen = _vllm(ctx, max(len(v) for v in pids.values()) + sampling.max_tokens + 16)
    rows = measure_throughput(gen, qs, pids, cfg.baseline.throughput_batch_sizes, sampling)
    _write_json(d / "throughput.json", rows)
    for r in rows:
        _echo(f"batch={r['batch_size']}: {r['tok_per_s']:.0f} tok/s ({r['tokens']} token, {r['seconds']} s)")


@app.command("equivalence")
def equivalence(
    model: Model = None,
    n: Annotated[int, typer.Option()] = 400,
    seeds: Seeds = "0-1",
    exp: Exp = None,
    sets: Sets = None,
) -> None:
    """Kiểm định tương đương giữa sinh liền mạch và sinh theo đoạn có nhánh phụ, trên val (P1.T6)."""
    from ttalloc.baseline.equivalence import run_equivalence
    from ttalloc.baseline.full import profile_sampling
    from ttalloc.context import model_context, prompt_ids_for

    _verify_splits()
    cfg = _cfg(exp, sets, model)
    qs = _questions("val", None, n)
    ctx = model_context(cfg)
    pids = prompt_ids_for(qs, ctx)
    d = _run_dir("equivalence", cfg)
    gen = _vllm(ctx, max(len(v) for v in pids.values()) + cfg.log.l_log + _bounds_extra(ctx))
    src = _live_source(ctx, gen, qs, pids, cfg.answer.mode or "full")
    full, seg, rep = run_equivalence(
        src,
        qs,
        pids,
        _seeds(seeds),
        profile_sampling("log", ctx.model_cfg, cfg),
        cfg,
        np.random.default_rng(cfg.sim.seed),
    )
    full.drop(columns=["problem", "answer_text"]).to_parquet(d / "full.parquet", index=False)
    seg.to_parquet(d / "segmented.parquet", index=False)
    _write_json(d / "equivalence.json", rep)
    _echo(json.dumps(rep, indent=2, ensure_ascii=False))


@app.command("log")
def log_cmd(
    model: Model = None,
    split: Split = "val",
    seeds: Seeds = "0-3",
    limit: Annotated[int | None, typer.Option()] = None,
    exp: Exp = None,
    sets: Sets = None,
) -> None:
    """Ghi dữ liệu cho bộ ước lượng và mô phỏng; chạy tiếp được sau khi bị ngắt (P1.T8)."""
    from ttalloc.context import log_hash, model_context, prompt_ids_for
    from ttalloc.logging.logger import run_logger
    from ttalloc.logging.store import LogStore
    from ttalloc.types import TraceKey

    _verify_splits()
    cfg = _cfg(exp, sets, model)
    ctx = model_context(cfg)
    lh = log_hash(cfg, ctx.model_cfg, ctx.prompts)
    qs = _questions(split, None, limit)
    keys = [TraceKey(q.qid, s) for s in _seeds(seeds) for q in qs]
    store = LogStore.for_split(ctx.model_cfg.name, split)
    pids = prompt_ids_for(qs, ctx)
    gen = _vllm(ctx, max(len(v) for v in pids.values()) + cfg.log.l_log + _bounds_extra(ctx))
    src = _live_source(ctx, gen, qs, pids)
    store.init(
        lh,
        {
            "model": ctx.model_cfg.name,
            "hf_id": ctx.model_cfg.hf_id,
            "revision": ctx.model_cfg.revision,
            "split": split,
            "c_ans_max": src.c_ans_max(),
            "c_probe_max": src.c_probe_max(),
            "think_end_id": ctx.codec.think_end_id,
            "answer_mode": src.answer_mode,
            "config": json.loads(json.dumps(cfg.model_dump(mode="json", by_alias=True))),
        },
    )
    kv = gen.kv_cache_tokens()
    _echo(f"ghi {len(keys)} chuỗi vào {store.root} (log_hash {lh}); đã có {len(store.completed_keys())}; KV {kv} token")
    if kv is None:
        log.warning(
            "không đọc được sức chứa KV cache: chạy tới log.window chuỗi, có thể phải tính lại ngữ cảnh mỗi vòng"
        )
    stats = run_logger(
        src,
        keys,
        store,
        cfg,
        ctx.model_cfg.name,
        split,
        lh,
        {q.qid: q.gold for q in qs},
        on_round=lambda i: log.info("%s", i),
        kv_tokens=kv,
    )
    _echo(json.dumps(stats, indent=2, ensure_ascii=False))


@app.command("validate-logs")
def validate_logs_cmd(model: Model = None, split: Split = "val", exp: Exp = None, sets: Sets = None) -> None:
    """Kiểm tra dữ liệu ghi (P1.T11); ghi validate.json cạnh dữ liệu."""
    from ttalloc.logging.store import LogStore
    from ttalloc.logging.validate import validate_logs

    cfg = _cfg(exp, sets, model)
    assert cfg.model
    store = LogStore.for_split(cfg.model, split)
    meta = store.meta() or {}
    rep = validate_logs(
        store.read_traces(),
        store.read_points(),
        cfg,
        meta.get("log_hash"),
        meta.get("think_end_id"),
        meta.get("c_ans_max"),
        meta.get("c_probe_max"),
    )
    _write_json(store.root / "validate.json", rep)
    _echo(json.dumps({k: v for k, v in rep.items() if k != "examples"}, indent=2, ensure_ascii=False))


@study_app.command("answer-mode")
def study_answer_mode_cmd(
    model: Model = None, n: Annotated[int, typer.Option()] = 100, seeds: Seeds = "0", exp: Exp = None, sets: Sets = None
) -> None:
    """Tại mỗi điểm và điểm tự kết thúc, sinh cả dạng full và short; đề xuất dạng câu trả lời (P1.T9). **[dừng]**"""
    from ttalloc.context import model_context, prompt_ids_for
    from ttalloc.logging.studies import study_answer_mode
    from ttalloc.types import TraceKey

    _verify_splits()
    cfg = _cfg(exp, sets, model)
    ctx = model_context(cfg)
    qs = _questions("val", None, n)
    pids = prompt_ids_for(qs, ctx)
    d = _run_dir("study-answer-mode", cfg)
    gen = _vllm(ctx, max(len(v) for v in pids.values()) + cfg.log.l_log + _bounds_extra(ctx))
    src = _live_source(ctx, gen, qs, pids, "full")
    df, summary = study_answer_mode(src, [TraceKey(q.qid, s) for s in _seeds(seeds) for q in qs], cfg)
    df.to_csv(d / "answer_mode.csv", index=False)
    _write_json(d / "answer_mode.json", summary)
    _echo(json.dumps(summary, indent=2, ensure_ascii=False))
    _echo("Tiếp theo: con người chọn answer.mode, ghi ADR, đặt answer.mode trong configs/base.yaml (P1.T9).")


@study_app.command("probe-noise")
def study_probe_noise_cmd(
    model: Model = None, n: Annotated[int, typer.Option()] = 100, seeds: Seeds = "0", exp: Exp = None, sets: Sets = None
) -> None:
    """Tại cùng điểm, gợi bằng hai mẫu lời nhắc; tỉ lệ đáp án khác nhau theo vị trí tương đối (P1.T10)."""
    from ttalloc.context import model_context, prompt_ids_for
    from ttalloc.data.prompts import load_prompts
    from ttalloc.logging.studies import study_probe_noise
    from ttalloc.types import TraceKey

    _verify_splits()
    cfg = _cfg(exp, sets, model)
    ctx = model_context(cfg)
    qs = _questions("val", None, n)
    pids = prompt_ids_for(qs, ctx)
    d = _run_dir("study-probe-noise", cfg)
    gen = _vllm(ctx, max(len(v) for v in pids.values()) + cfg.log.l_log + _bounds_extra(ctx) + 32)
    src = _live_source(ctx, gen, qs, pids, cfg.answer.mode or "full")
    df, summary = study_probe_noise(
        src, [TraceKey(q.qid, s) for s in _seeds(seeds) for q in qs], cfg, load_prompts(cfg.study.probe_alt)
    )
    df.to_csv(d / "probe_noise.csv", index=False)
    _write_json(d / "probe_noise.json", summary)
    _echo(json.dumps(summary, indent=2, ensure_ascii=False))


# ======================================================================== P2–P3: bộ ước lượng, bảng, mô phỏng


@estimator_app.command("train")
def estimator_train(exp: Exp = None, sets: Sets = None) -> None:
    """Học bộ ước lượng trên `train_split` và hiệu chuẩn bằng dự đoán ngoài fold (P2.T4)."""
    from ttalloc.estimator.dataset import build_dataset
    from ttalloc.estimator.rho import estimate_omega
    from ttalloc.estimator.train import train_estimator
    from ttalloc.features.extract import feature_names
    from ttalloc.sim.load import estimator_dir, expected_log_hash
    from ttalloc.sim.logged_source import load_logged

    cfg = _cfg(exp, sets)
    assert cfg.model
    lh = expected_log_hash(cfg)
    chains, _ = load_logged(cfg.model, cfg.train_split, lh)
    df = build_dataset(chains, cfg.dp, cfg.features, cfg.estimator.h_train)
    _echo(f"{len(chains)} chuỗi, {df.qid.nunique()} câu, {len(df)} mẫu")
    est, oof = train_estimator(df, feature_names(cfg.features.drop_groups), cfg.estimator)
    est.meta |= {
        "omega": estimate_omega(chains.values(), cfg.dp),
        "log_hash": lh,
        "train_split": cfg.train_split,
        "dp": cfg.dp.model_dump(),
        "features_cfg": cfg.features.model_dump(),
        "n_chains": len(chains),
    }
    d = estimator_dir(cfg, lh)
    est.save(d)
    oof.to_parquet(d / "oof.parquet", index=False)
    _echo(f"đã lưu {d}; omega={est.meta['omega']:.4f}")
    _echo(json.dumps(est.meta["oof"], indent=2))


@estimator_app.command("eval")
def estimator_eval(split: Split = "val", exp: Exp = None, sets: Sets = None) -> None:
    """E1 trên một tập: AUROC, ECE, Brier, biểu đồ độ tin cậy, Spearman của lợi ích, mức quan trọng (P2.T6)."""
    import pandas as pd

    from ttalloc.estimator.dataset import build_dataset
    from ttalloc.estimator.evaluate import evaluate, reliability
    from ttalloc.estimator.train import Estimator
    from ttalloc.features.curve import EmpiricalCurves
    from ttalloc.sim.load import estimator_dir, expected_log_hash
    from ttalloc.sim.logged_source import load_logged

    cfg = _cfg(exp, sets)
    if not cfg.locked:  # E1 trên test_main chỉ chạy sau khi khóa (mục 7.3)
        _refuse_test(split, "estimator eval")
    assert cfg.model
    lh = expected_log_hash(cfg)
    est = Estimator.load(estimator_dir(cfg, lh))
    chains, _ = load_logged(cfg.model, split, lh)
    df = build_dataset(chains, cfg.dp, cfg.features, cfg.estimator.h_train)
    parts = []
    for hv in sorted(df.h.unique()):
        g = df[df.h == hv]
        parts.append(g[["qid", "seed", "point_idx", "h", "y"]].assign(p=est.predict(g[est.features], [int(hv)])[:, 0]))
    pred = pd.concat(parts)
    gen_end = {(k.qid, k.seed, int(i)): int(c.gen_end[i]) for k, c in chains.items() for i in range(c.n_points)}
    rep = evaluate(pred, EmpiricalCurves(chains, cfg.dp), gen_end, est.hs)
    rep["reliability"] = {
        str(h): reliability(g.y.to_numpy(float), g.p.to_numpy(float)).to_dict("records") for h, g in pred.groupby("h")
    }
    if est.kind == "lightgbm":
        names = [*est.features, "h"]
        rep["importance_gain"] = dict(zip(names, map(float, est.model.feature_importance("gain")), strict=True))
    d = _run_dir(f"estimator-eval-{split}", cfg)
    pred.to_parquet(d / "pred.parquet", index=False)
    _write_json(d / "eval.json", rep)
    _echo(
        json.dumps(
            {k: rep[k] for k in ("auroc_q0", "max_ece", "mean_log_loss")} | {"gain_spearman": rep.get("gain_spearman")},
            indent=2,
        )
    )


@table_app.command("build")
def table_build(split: Split = "val", exp: Exp = None, sets: Sets = None) -> None:
    """Dựng bảng tính sẵn của một tập theo mã băm (P2.T10); dựng lại nếu config hoặc bộ ước lượng đổi."""
    from ttalloc.sim.load import load_env

    cfg = _cfg(exp, sets)
    env = load_env(cfg, split, build_missing_table=True)
    _echo(f"bảng {split}: {0 if env.table is None else len(env.table)} dòng")


@sim_app.command("run")
def sim_run(exp: Exp = None, sets: Sets = None) -> None:
    """Chạy lưới mô phỏng (P2.T15). Không chạy trên test_* (dùng eval-test)."""
    for s in _cfg(exp, sets).splits:
        _refuse_test(s, "sim run")
    _sim(_cfg(exp, sets), "sim")


def _sim(cfg: Any, prefix: str) -> Path:
    from ttalloc.config import config_hash
    from ttalloc.sim.load import expected_log_hash, load_env
    from ttalloc.sim.run import run_experiment

    if not cfg.splits or not cfg.alphas or not cfg.methods:
        raise typer.BadParameter("config thí nghiệm cần splits, alphas và methods")
    envs = {s: load_env(cfg, s) for s in cfg.splits}
    d = _run_dir(f"{prefix}-{cfg.name}", cfg)
    any_env = next(iter(envs.values()))
    hashes = {
        "config": config_hash(cfg),
        "log_hash": expected_log_hash(cfg),
        "estimator": None if any_env.estimator is None else any_env.estimator.meta.get("log_hash"),
        "estimator_dir": None if any_env.estimator is None else str(any_env.estimator.meta.get("train_split")),
    }
    run_experiment(envs, cfg, d, hashes, log=_echo)
    return d


@app.command("tune")
def tune_cmd(exp: Exp = None, sets: Sets = None) -> None:
    """Tìm kiếm ngẫu nhiên trên val cho mọi phương pháp trong config (P3.T4); ghi best.json."""
    from ttalloc.sim.load import load_env
    from ttalloc.sim.tune import random_search

    cfg = _cfg(exp, sets)
    for s in cfg.splits:
        _refuse_test(s, "tune")
    env = load_env(cfg, cfg.splits[0])
    d = _run_dir(f"tune-{cfg.name}", cfg)
    best_all: dict[str, Any] = {}
    trials_all: dict[str, Any] = {}
    for m in cfg.methods:
        best, trials = random_search(env, m, cfg.batch, cfg.alphas)
        best_all[m.id] = best["params"]
        trials_all[m.id] = trials
        _echo(f"{m.id}: tốt nhất {best['params']} objective={best['objective']:.4f}")
    _write_json(d / "best.json", best_all)
    _write_json(d / "trials.json", trials_all)
    _echo(f"đặt tuned_from: {d} trong config thí nghiệm rồi chạy `ttalloc lock`")


@app.command("lock")
def lock_cmd(exp: Exp = None, sets: Sets = None) -> None:
    """Thay tham số `tuned` bằng giá trị đã chỉnh, ghi configs/locked/<tên>.yaml và docs/test-log.md (P3.T5). [dừng]"""
    from ttalloc.locking import lock
    from ttalloc.sim.env import resolve_params

    cfg = _cfg(exp, sets)
    methods = [m.model_copy(update={"params": resolve_params(m, cfg)}) for m in cfg.methods]
    path = lock(cfg.model_copy(update={"methods": methods}))
    _echo(f"đã khóa: {path}. Chạy trên tập kiểm tra: ttalloc eval-test --exp {path.as_posix()} --approved-by <tên>")


@app.command("eval-test")
def eval_test(exp: Annotated[Path, typer.Option("--exp")], approved_by: Approved = None) -> None:
    """Chạy trên test_* với config đã khóa và có người duyệt (P3.T5, mục 7.5): mô phỏng hoặc chạy thật."""
    from ttalloc.config import load_config
    from ttalloc.locking import NotLocked, check_locked, record_eval

    _verify_splits()
    cfg = load_config(exp)
    try:
        h = check_locked(cfg, approved_by)
    except NotLocked as exc:
        typer.echo(f"LỖI: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    assert approved_by
    d = _sim(cfg, "eval-test") if cfg.kind == "sim" else _online(cfg, "eval-test")
    record_eval(cfg, h, approved_by, d)


# ======================================================================== P4


@online_app.command("run")
def online_run(exp: Exp = None, sets: Sets = None) -> None:
    """Chạy thật trên val (thử nghiệm); lưới E5 trên test_main chạy qua eval-test (P4, bước chạy 1)."""
    cfg = _cfg(exp, sets)
    for s in cfg.splits:
        _refuse_test(s, "online run")
    _online(cfg, "online")


def _online(cfg: Any, prefix: str) -> Path:
    from ttalloc.context import load_split_questions, model_context, prompt_ids_for
    from ttalloc.online.run import run_online
    from ttalloc.online.timing import estimate_max_active, kv_cache_tokens
    from ttalloc.runs import gpu_info
    from ttalloc.sim.load import load_env

    if len(cfg.splits) != 1:
        raise typer.BadParameter("chạy thật nhận đúng một split")
    split = cfg.splits[0]
    ctx = model_context(cfg)
    qs = load_split_questions(split)  # đúng thứ tự và nội dung của env.questions
    pids = prompt_ids_for(qs, ctx)
    # Khởi tạo vLLM trước khi nạp bộ ước lượng: load_env chạy LightGBM (OpenMP), và vLLM fork tiến trình EngineCore;
    # fork sau khi OpenMP đã có luồng thì EngineCore treo ở bước nạp model (bản tập dượt lần 5: online run và
    # eval-test online treo tới hết 3 giờ ở "Using V2 Model Runner").
    gen = _vllm(ctx, max(len(v) for v in pids.values()) + cfg.log.l_log + _bounds_extra(ctx))
    env = load_env(cfg, split)
    if [q.qid for q in qs] != list(env.questions):
        raise AssertionError("câu hỏi của chạy thật khác env.questions")
    if cfg.online.max_active is not None:
        m_active = cfg.online.max_active
    else:
        from ttalloc.data.batches import build_batch

        n = len(build_batch(qs, cfg.batch, np.random.default_rng(cfg.batch.seed)))
        b_min = min(cfg.alphas) * env.c_full([q.qid for q in qs]) * n / max(len(qs), 1)
        m_active = estimate_max_active(
            kv_cache_tokens(gen.llm), n, int(b_min), max(len(v) for v in pids.values()), cfg.dp.b_min + cfg.dp.l_max
        )
    _echo(f"max_active = {m_active}")
    env.cfg = cfg.model_copy(update={"sim": cfg.sim.model_copy(update={"max_active": m_active})})
    d = _run_dir(f"{prefix}-{cfg.name}", cfg)
    s = run_online(env.cfg, ctx, gen, env, pids, d, m_active, hit_rate=gen.prefix_cache_hit_rate, log=_echo)
    s["gpu"] = gpu_info()
    _write_json(d / "summary.json", s)
    return d


# ======================================================================== báo cáo và cổng


@study_app.command("curves")
def study_curves(split: Split = "val", exp: Exp = None, sets: Sets = None) -> None:
    """Đường cong đúng sai thực nghiệm theo nhóm độ khó (P2.T20)."""
    from ttalloc.context import load_split_questions
    from ttalloc.eval.curves import curve_stats, difficulty_groups, question_curves
    from ttalloc.eval.metrics import natural_accuracy
    from ttalloc.features.curve import EmpiricalCurves
    from ttalloc.sim.load import expected_log_hash
    from ttalloc.sim.logged_source import load_logged

    cfg = _cfg(exp, sets)
    _refuse_test(split, "study curves")
    assert cfg.model
    chains, _ = load_logged(cfg.model, split, expected_log_hash(cfg))
    curves = EmpiricalCurves(chains, cfg.dp)
    bs = sorted({0, *cfg.estimator.h_train, cfg.log.l_log})
    qids = sorted(curves.by_q)
    df = question_curves(curves, qids, bs)
    nat = natural_accuracy({(k.qid, k.seed): bool(c.nat_correct) for k, c in chains.items()})
    qs = {q.qid: q for q in load_split_questions(split)}
    stats = curve_stats(df, bs, cfg.log.l_log, cfg.study.curves_delta, difficulty_groups(df, qs, nat))
    d = _run_dir(f"study-curves-{split}", cfg)
    df.to_csv(d / "curves.csv", index=False)
    _write_json(d / "curves.json", stats)
    _echo(json.dumps({"overall": stats["overall"], "by_group": stats["by_group"]}, indent=2, ensure_ascii=False))


@app.command("report")
def report_cmd(
    runs: Annotated[list[Path], typer.Option("--runs")],
    out: Annotated[Path, typer.Option("--out")] = Path("reports"),
    ref: Annotated[str, typer.Option()] = "B2",
    est_eval: Annotated[Path | None, typer.Option("--est-eval", help="eval.json của estimator eval")] = None,
    n_boot: Annotated[int, typer.Option()] = 2000,
) -> None:
    """Bảng (Markdown, CSV, LaTeX) và hình chỉ từ dữ liệu trong runs/ (P2.T18)."""
    from ttalloc.eval.report import build_report

    ev = json.loads(est_eval.read_text(encoding="utf-8")) if est_eval else None
    man = build_report(runs, out, np.random.default_rng(0), n_boot=n_boot, ref=ref, est_eval=ev)
    _echo(f"{len(man['tables'])} tệp bảng, {len(man['figures'])} hình trong {out}")


def _gate_out(res: dict[str, Any], out: Path) -> None:
    from ttalloc.eval.gates import gate_markdown

    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / f"gate_{res['gate'].lower()}.json", res)
    (out / f"gate_{res['gate'].lower()}.md").write_text(gate_markdown(res), encoding="utf-8")
    typer.echo(gate_markdown(res))


@gate_app.command("g1")
def gate_g1_cmd(
    runs: Annotated[Path, typer.Option("--runs", help="Thư mục chạy sim run --exp g1")],
    est_eval: Annotated[Path, typer.Option("--est-eval", help="eval.json của estimator eval --split val")],
    validate: Annotated[Path, typer.Option("--validate", help="validate.json của val")],
    n_boot: Annotated[int, typer.Option()] = 2000,
) -> None:
    import pandas as pd

    from ttalloc.eval.gates import gate_g1

    res = gate_g1(
        json.loads(est_eval.read_text(encoding="utf-8")),
        pd.read_parquet(runs / "per_question.parquet"),
        json.loads(validate.read_text(encoding="utf-8")),
        np.random.default_rng(0),
        n_boot,
    )
    _gate_out(res, runs)


@gate_app.command("g2")
def gate_g2_cmd(runs: Annotated[Path, typer.Option("--runs")], n_boot: Annotated[int, typer.Option()] = 2000) -> None:
    import pandas as pd

    from ttalloc.eval.gates import gate_g2

    _gate_out(gate_g2(pd.read_parquet(runs / "per_question.parquet"), np.random.default_rng(0), n_boot), runs)


@gate_app.command("g3")
def gate_g3_cmd(
    runs: Annotated[list[Path], typer.Option("--runs")], out: Annotated[Path | None, typer.Option("--out")] = None
) -> None:
    from ttalloc.eval.gates import gate_g3

    _gate_out(gate_g3(runs, np.random.default_rng(0)), out or runs[0])


if __name__ == "__main__":
    app()
