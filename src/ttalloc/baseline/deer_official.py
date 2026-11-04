"""Chạy repo DEER chính thức bằng subprocess và chấm lại kết quả (PLAN P1.T5, `baseline deer`).

Repo được clone vào `third_party/DEER` ở commit `baseline.deer_commit`. Script của DEER đọc
`<dataset_dir>/<dataset>/test.jsonl` (trường `problem`, `answer`) và ghi một tệp jsonl với `question`,
`generated_responses`, `gold_answer`, `too_long`, `thinking_steps`. Đầu ra không có số token, nên số token được đếm lại
bằng tokenizer trên văn bản sinh ra (ADR-010).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ttalloc.config import REPO_ROOT, ExperimentCfg, ModelCfg
from ttalloc.data.grading import grade
from ttalloc.data.prompts import format_problem
from ttalloc.engine.generator import TokenCodec
from ttalloc.engine.live_source import extract_answer
from ttalloc.types import Question

THIRD_PARTY = REPO_ROOT / "third_party"
SCRIPTS = {"qwen3": "vllm-deer-qwen3.py", "r1_distill": "vllm-deer.py"}


def ensure_repo(cfg: ExperimentCfg, root: Path = THIRD_PARTY) -> Path:
    path = root / "DEER"
    if not (path / ".git").exists():
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", cfg.baseline.deer_repo, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--all", "--quiet"], check=False)
    subprocess.run(["git", "-C", str(path), "checkout", "--quiet", cfg.baseline.deer_commit], check=True)
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if head.stdout.strip() != cfg.baseline.deer_commit:
        raise RuntimeError(f"third_party/DEER ở {head.stdout.strip()}, cần {cfg.baseline.deer_commit}")
    return path


def write_dataset(questions: list[Question], data_dir: Path, name: str) -> Path:
    path = data_dir / name / "test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(
                json.dumps({"problem": format_problem(q), "answer": q.gold, "qid": q.qid}, ensure_ascii=False) + "\n"
            )
    return path


def deer_command(
    repo: Path,
    model_cfg: ModelCfg,
    model_path: str,
    data_dir: Path,
    out_dir: Path,
    name: str,
    cfg: ExperimentCfg,
    temperature: float,
    top_p: float,
    max_tokens: int,
    run_time: int,
) -> list[str]:
    return [
        sys.executable,
        str(repo / SCRIPTS[model_cfg.family]),
        "--model_name_or_path",
        model_path,
        "--dataset_dir",
        str(data_dir),
        "--output_path",
        str(out_dir),
        "--dataset",
        name,
        "--threshold",
        str(cfg.deer.lambda_),
        "--max_generated_tokens",
        str(max_tokens),
        "--think_ratio",
        str(cfg.baseline.deer_think_ratio),
        "--policy",
        cfg.baseline.deer_policy,
        "--batch_size",
        "2000",
        "--dtype",
        cfg.engine.dtype,
        "--gpu-memory-utilization",
        str(cfg.engine.gpu_memory_utilization),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--run_time",
        str(run_time),
    ]


def parse_deer_output(
    lines: list[str], questions: list[Question], codec: TokenCodec | None, seed: int
) -> list[dict[str, Any]]:
    """Ghép đầu ra DEER với câu hỏi theo nội dung đề bài, trích đáp án sau `</think>` cuối cùng và chấm lại."""
    by_problem = {format_problem(q): q for q in questions}
    rows = []
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        q = by_problem.get(d["question"])
        if q is None:
            raise KeyError(f"không ghép được câu trong đầu ra DEER: {d['question'][:80]!r}")
        text = (d.get("generated_responses") or [""])[0]
        tail = text.rsplit("</think>", 1)[-1]
        extracted = extract_answer(tail, q.answer_type)
        n_tokens = len(codec.encode(text)) if codec is not None else None
        rows.append(
            {
                "qid": q.qid,
                "seed": seed,
                "dataset": q.dataset,
                "problem": q.problem,
                "gold": q.gold,
                "n_tokens": n_tokens,
                "think_tokens": len(codec.encode(text.split("</think>", 1)[0])) if codec else None,
                "natural_end": "</think>" in text,
                "finish_reason": "too_long" if d.get("too_long") else "stop",
                "answer_text": tail,
                "extracted": extracted,
                "correct": grade(extracted, q.gold, q.answer_type),
                "thinking_steps": d.get("thinking_steps"),
                "too_long": d.get("too_long"),
            }
        )
    return rows


def run_deer(
    cfg: ExperimentCfg,
    model_cfg: ModelCfg,
    questions: list[Question],
    codec: TokenCodec,
    work_dir: Path,
    seeds: list[int],
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo = ensure_repo(cfg)
    data_dir = work_dir / "deer_data"
    out_dir = work_dir / "deer_out"
    rows: list[dict[str, Any]] = []
    timing: dict[str, Any] = {}
    by_ds: dict[str, list[Question]] = {}
    for q in questions:
        by_ds.setdefault(q.dataset, []).append(q)
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    for seed in seeds:
        for ds, qs in by_ds.items():
            name = f"{ds}_s{seed}"
            write_dataset(qs, data_dir, name)
            cmd = deer_command(
                repo, model_cfg, model_path, data_dir, out_dir, name, cfg, temperature, top_p, max_tokens, run_time=seed
            )
            t0 = time.time()
            with (work_dir / f"deer_{name}.log").open("w", encoding="utf-8") as logf:
                subprocess.run(cmd, cwd=repo, env=env, stdout=logf, stderr=subprocess.STDOUT, check=True)
            timing[name] = round(time.time() - t0, 1)
            files = sorted((out_dir).rglob(f"{name}/*.jsonl"))
            if not files:
                raise FileNotFoundError(f"DEER không ghi đầu ra cho {name} trong {out_dir}")
            rows += parse_deer_output(files[-1].read_text(encoding="utf-8").splitlines(), qs, codec, seed)
    return rows, timing
